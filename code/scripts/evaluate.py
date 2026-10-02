#!/usr/bin/env python
"""Evaluate predictions against targets, one process per GPU.

  evaluate.py --pred RES/x1/restormer --targets DATA/x1/targets --masks DATA/x1/masks \
      --metrics psnr,ssim,pm,hb,lpips,drift,seam,stats --scale 1 --crop 64 --gpus all \
      --label restormer_tiled512

Writes <pred>/metrics[_<tag>].csv (per image) and <pred>/metrics[_<tag>].json (mean/std/n,
settings, and time/memory from the run's meta.json). `--targets` omitted -> NR metrics only.
--scale = native / evaluated resolution (1 = native, 4 = 1680x1120 on DPDD); sets the PSF
pixel pitch. Seam metrics use the tile grid stored in <pred>/meta.json; the 'gridshift' metric
needs --pred-shift (the same model run with run_model.py --grid-offset); 'noharm' needs --inputs.
Metric definitions: docs/evaluation.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from uhdd import metrics  # noqa: E402
from uhdd.io import ImageDataset, list_images, to_tensor  # noqa: E402
from uhdd.parallel import launch, loader, parse_gpus, shard  # noqa: E402


def _crop(t: torch.Tensor | None, c: int) -> torch.Tensor | None:
    return t if t is None or c <= 0 else t[..., c:-c, c:-c]


def _blur_map(dp_dir: str, name: str, hw, device) -> torch.Tensor | None:
    """Smoothed confidence-weighted |DP disparity| (DP px), resized to the evaluated image."""
    from uhdd.dualpixel import load_blur_map
    k = load_blur_map(dp_dir, name, hw)
    return None if k is None else torch.from_numpy(k)[None, None].to(device)


def worker(rank: int, world: int, device: torch.device, a, records: list, infos: dict, out: Path,
           infos_shift: dict | None = None):
    items = shard(records, rank, world)
    names = a.metrics.split(",")
    ctx_base = {"bin_tile": a.bin_tile, "align_radius": a.align_radius,
                "eval_scales": tuple(int(f) for f in a.eval_scales.split(",")),
                "scale": a.scale, "fnum": a.fnum, "pm_sigma": a.pm_sigma, "hb_s": a.hb_s, "crop": a.crop,
                "fr_tile": a.fr_tile, "cell": a.cell}
    rows = []
    for rec in loader(ImageDataset(items), a.workers):
        pred = to_tensor(rec["pred"], device)
        gt = to_tensor(rec["target"], device) if rec.get("target") is not None else None
        if gt is not None and a.target_bits:  # e.g. 8 to match protocols evaluated on 8-bit targets
            q = 2 ** a.target_bits - 1
            gt = gt.mul(q).round_().div_(q)
        mask = rec["mask"].to(device)[None, None] if rec.get("mask") is not None else None
        if gt is not None and pred.shape != gt.shape:
            raise ValueError(f"{rec['name']}: pred {tuple(pred.shape)} vs target {tuple(gt.shape)}")
        blur_map = _blur_map(a.dp_maps, rec["name"], pred.shape[-2:], device) if a.dp_maps else None
        pred, gt, mask = _crop(pred, a.crop), _crop(gt, a.crop), _crop(mask, a.crop)
        ctx = {**ctx_base, "info": infos.get(rec["name"], {}).get("tiling"), "blur_map": _crop(blur_map, a.crop)}
        if rec.get("input") is not None:
            ctx["input"] = _crop(to_tensor(rec["input"], device), a.crop)
        if rec.get("pred_shift") is not None:
            ctx["pred_shift"] = _crop(to_tensor(rec["pred_shift"], device), a.crop)
            ctx["info_shift"] = (infos_shift or {}).get(rec["name"], {}).get("tiling")
        row = {"name": rec["name"], **metrics.compute(names, pred, gt, mask, ctx)}
        rows.append(row)
        print(f"[rank {rank}] {rec['name']} " + " ".join(
            f"{k}={v:.4g}" for k, v in row.items() if isinstance(v, float)), flush=True)
    with open(out.with_name(f".{out.stem}_rank{rank}.json"), "w") as f:
        json.dump(rows, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--targets")
    ap.add_argument("--masks")
    ap.add_argument("--dp-maps", help="dir of dp_maps.py outputs (<stem>_disp/_conf.png) for 'blurbins'")
    ap.add_argument("--inputs", help="blurry inputs (same names) for 'noharm'")
    ap.add_argument("--bin-tile", type=int, default=512, help="tile size of 'percbins'")
    ap.add_argument("--align-radius", type=int, default=2, help="shift search radius (px) of 'apsnr'")
    ap.add_argument("--eval-scales", default="2,4", help="downscale factors of 'msres'")
    ap.add_argument("--pred-shift", help="same model run with a shifted tile grid (metric 'gridshift')")
    ap.add_argument("--metrics", default="psnr,ssim,mae,pm,hb,lpips,dists,drift,seam,stats")
    ap.add_argument("--label", help="method label for tables (default: pred folder name)")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. the test set")
    ap.add_argument("--scale", type=int, default=1, help="native / evaluated resolution")
    ap.add_argument("--fnum", type=float, default=22.0, help="target capture f-number (PSF-matched metrics)")
    ap.add_argument("--pm-sigma", type=float, default=0.8, help="extra Gaussian sigma (native px) for 'pmg'")
    ap.add_argument("--hb-s", type=int, default=4, help="high-band cutoff factor")
    ap.add_argument("--target-bits", type=int, choices=[8, 16], help="quantize targets (8 = Restormer DPDD protocol)")
    ap.add_argument("--crop", type=int, default=0, help="border pixels removed before evaluation")
    ap.add_argument("--fr-tile", type=int, default=1024)
    ap.add_argument("--cell", type=int, default=256)
    ap.add_argument("--gpus", default="all")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    pred_dir = Path(a.pred)
    preds = list_images(pred_dir)
    tgts = list_images(a.targets) if a.targets else {}
    msks = list_images(a.masks) if a.masks else {}
    if tgts and (missing := sorted(set(preds) - set(tgts))):
        sys.exit(f"{len(missing)} predictions without target: {missing[:10]}")
    shifted = list_images(a.pred_shift) if a.pred_shift else {}
    inps = list_images(a.inputs) if a.inputs else {}
    records = [{"name": k, "pred": p, "target": tgts.get(k), "mask": msks.get(k), "pred_shift": shifted.get(k),
                "input": inps.get(k)} for k, p in preds.items()]

    meta_path = pred_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"images": {}, "summary": {}}
    stem = "metrics" + (f"_{a.tag}" if a.tag else "")
    out = pred_dir / f"{stem}.csv"

    meta_shift = {}
    if a.pred_shift and (Path(a.pred_shift) / "meta.json").exists():
        meta_shift = json.loads((Path(a.pred_shift) / "meta.json").read_text())["images"]
    launch(worker, parse_gpus(a.gpus), a, records, meta["images"], out, meta_shift)

    rows = []
    for f in sorted(pred_dir.glob(f".{stem}_rank*.json")):
        rows += json.loads(f.read_text())
        f.unlink()
    rows.sort(key=lambda r: r["name"])
    for r in rows:  # attach runtime/memory from the inference run
        m = meta["images"].get(r["name"], {})
        r["time_s"], r["peak_mem_gb"] = m.get("time_s"), m.get("peak_mem_gb")
    cols = ["name"] + list(dict.fromkeys(k for r in rows for k in r if k != "name"))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    stats = {}
    for k in cols[1:]:
        v = np.array([r.get(k) for r in rows if isinstance(r.get(k), (int, float))], dtype=float)
        v = v[np.isfinite(v)]
        if len(v):
            stats[k] = {"mean": float(v.mean()), "std": float(v.std()), "n": int(len(v))}
    summary = {"label": a.label or pred_dir.name, "tag": a.tag, "n_images": len(rows),
               "settings": {k: v for k, v in vars(a).items() if k not in ("pred", "targets", "masks")},
               "run": meta.get("summary", {}), "metrics": stats}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: round(v["mean"], 4) for k, v in stats.items()}))


if __name__ == "__main__":
    main()
