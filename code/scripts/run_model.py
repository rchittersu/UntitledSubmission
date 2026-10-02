#!/usr/bin/env python
"""Run a registered model over a folder of images, one process per GPU.

  run_model.py --model restormer_dpdd --inputs IN --out OUT --gpus all \
      [--tile 512 --overlap 64 --tile-batch 8 --blend linear] [--precision fp32]

Tiling defaults to each method's own inference setup (registry `paper_input` / `paper_tile`, see
uhdd.tiling.default_tiling), decided per image; overlap defaults to tile * --overlap-ratio (1/8).
Whole-image mode (--tile 0) pads to the model's required multiple and crops back; tiled mode
uses tiles that are multiples of it. Outputs are 16-bit PNGs (default) + OUT/meta.json with
per-image timing, peak memory and tile grid (used by the seam metric).

Pipelines are composed by chaining runs, e.g. low-res deblur + x4 bicubic:
  run_model.py --model restormer_dpdd --inputs data/x4/inputs --out res/x4/restormer
  run_model.py --model bicubic_x4     --inputs res/x4/restormer --out res/x1/restormer+bicubic
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from uhdd import models  # noqa: E402
from uhdd.io import ImageDataset, list_images, to_numpy, to_tensor, write_image  # noqa: E402
from uhdd.parallel import AsyncWriter, launch, loader, parse_gpus, shard  # noqa: E402
from uhdd.tiling import TileSpec, default_tiling, run_tiled  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"
PRECISION = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def worker(rank: int, world: int, device: torch.device, a: argparse.Namespace, todo: list) -> None:
    items = shard(todo, rank, world, key=lambda r: r["input"].stat().st_size)
    model = models.build(a.model, a.config, device, channels_last=a.channels_last, compile=a.compile)
    amp = PRECISION[a.precision]
    in_bits = a.input_bits if a.input_bits is not None else model.spec.get("input_bits")
    out_dtype = np.uint16 if a.save_bits == 16 else np.uint8
    writer = AsyncWriter(workers=a.write_threads)
    meta, cuda, mps = {}, device.type == "cuda", device.type == "mps"

    for rec in loader(ImageDataset(items), a.workers):
        x = to_tensor(rec["input"], device)
        if in_bits:  # reproduce models trained/tested on 8-bit data (e.g. Restormer DPDD)
            q = 2 ** in_bits - 1
            x = x.mul(q).round_().div_(q)
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        tile, overlap = default_tiling(model.spec, x.shape[-2:], a.overlap_ratio)
        if a.tile is not None:
            tile, overlap = a.tile, round(a.tile * a.overlap_ratio / 2) * 2
        if a.overlap is not None:
            overlap = a.overlap
        batch = a.tile_batch or int(model.spec.get("tile_batch", 2 if tile >= 1024 or tile == 0 else 4))
        spec = TileSpec(tile, overlap, batch, a.blend, a.grid_offset)
        t0 = time.perf_counter()
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            y, info = run_tiled(model, x, spec, model.multiple, model.scale)
        if cuda:
            torch.cuda.synchronize(device)
        if mps:
            torch.mps.synchronize()
        dt = time.perf_counter() - t0
        writer.submit(write_image, Path(a.out) / f"{rec['name']}.png", to_numpy(y, out_dtype))
        meta[rec["name"]] = {
            "time_s": round(dt, 4),
            "peak_mem_gb": (round(torch.cuda.max_memory_allocated(device) / 2**30, 3) if cuda else
                            round(torch.mps.driver_allocated_memory() / 2**30, 3) if mps else None),
            "in_hw": list(x.shape[-2:]), "out_hw": list(y.shape[-2:]), "tiling": info,
        }
        print(f"[rank {rank}] {rec['name']} {dt:.2f}s", flush=True)
    writer.close()
    with open(Path(a.out) / f".meta_rank{rank}.json", "w") as f:
        json.dump(meta, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="all", help="'all', 'cpu', or e.g. '0,1,3'")
    ap.add_argument("--tile", type=int, default=None,
                    help="tile size (input px); 0 = whole image; default: the method's paper setup (registry)")
    ap.add_argument("--overlap", type=int, default=None, help="default: tile * --overlap-ratio")
    ap.add_argument("--overlap-ratio", type=float, default=0.125, help="standard overlap as a fraction of the tile")
    ap.add_argument("--tile-batch", type=int, default=None, help="default: registry tile_batch, else 2 (tile >= 1024) / 4")
    ap.add_argument("--blend", default="linear", choices=["linear", "gaussian", "hard", "mean"])
    ap.add_argument("--grid-offset", type=int, default=0, help="shift the tile grid (grid-shift consistency runs)")
    ap.add_argument("--precision", default="fp32", choices=list(PRECISION))
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--compile", action="store_true", help="torch.compile (pays off with fixed tile size)")
    ap.add_argument("--input-bits", type=int, choices=[8, 16],
                    help="quantize inputs to this bit depth (default: model's input_bits in the config, else none)")
    ap.add_argument("--save-bits", type=int, default=16, choices=[8, 16])
    ap.add_argument("--workers", type=int, default=2, help="reader processes per GPU")
    ap.add_argument("--write-threads", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (debugging)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    imgs = list_images(a.inputs)
    todo = [{"name": k, "input": p} for k, p in imgs.items()
            if not (a.skip_existing and (out / f"{k}.png").exists())]
    if a.limit:
        todo = todo[: a.limit]
    gpus = parse_gpus(a.gpus)
    print(f"{a.model}: {len(todo)} images on {gpus or 'cpu'}", flush=True)
    if todo:
        launch(worker, gpus, a, todo)

    # merge per-rank metadata (keeps entries from earlier --skip-existing runs)
    meta_path = out / "meta.json"
    merged = json.loads(meta_path.read_text())["images"] if meta_path.exists() else {}
    for f in sorted(out.glob(".meta_rank*.json")):
        merged.update(json.loads(f.read_text()))
        f.unlink()
    gpu_name = (torch.cuda.get_device_name(gpus[0]) if gpus and gpus[0] >= 0 else
                "apple-mps" if gpus and gpus[0] == -2 else "cpu")
    run = {k: v for k, v in vars(a).items() if k not in ("inputs", "out", "config")}
    times = [m["time_s"] for m in merged.values()]
    summary = {"model": a.model, "n": len(merged), "device": gpu_name,
               "mean_time_s": round(float(np.mean(times)), 4) if times else None,
               "torch": torch.__version__, "run": run}
    meta_path.write_text(json.dumps({"summary": summary, "images": merged}, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
