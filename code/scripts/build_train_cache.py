#!/usr/bin/env python
"""Build the training cache of one split (plan/method_plan.md §4, docs/training.md).

Per scene, under $UHDD_RESULTS/dpdd/cache/<split>/<name>/:
  x.npy, y.npy        native input / target, HxWx3 in native bit depth (memory-mapped random 512 crops)
  mask.npy            HxW uint8 validity mask
  blur.npy            H4xW4 float16 |DP disparity| (confidence-weighted, box `--blur-win`), DP px at the anchor res
  anchor_<model>.npy  H4xW4x3 x4 anchors of every deblurrer given (from deblur/<split>_x4/<model>@whole)
  mem.npz             exemplar memory on the primary anchor: key boxes / scales, query grid, top-K retrieval
                      (idx, score), oracle exemplars (idx, score)
  meta.json           counts and the per-scene measurements below
Measurements recorded on the way (manifest.json, all scenes):
  M1  pixel share per blur bin (b0 < 0.4 <= b1 < 1.5 <= b2 < 3.5 <= b3, DP px at 1680)
  M2  retrieval: keys per scene, coverage (defocused query tokens with a key), recall@K of the oracle top-1 among the
      retrieved top-K (and the chance level K / #keys); with --compare-pixels also for pixel features.

  python code/scripts/build_train_cache.py --split train --gpus all
  python code/scripts/build_train_cache.py --split val --gpus all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from uhdd import layout  # noqa: E402
from uhdd.dualpixel import load_blur_map  # noqa: E402
from uhdd.io import list_images, read_image, read_mask, to_tensor  # noqa: E402
from uhdd.parallel import launch, parse_gpus, shard  # noqa: E402
from uhdd.train import memory as mem_mod  # noqa: E402

BINS = (0.4, 1.5, 3.5)


def save(path: Path, arr: np.ndarray) -> None:
    m = np.lib.format.open_memmap(path, mode="w+", dtype=arr.dtype, shape=arr.shape)
    m[...] = arr
    m.flush()
    del m


def retrieval_stats(mem, idx, oidx, blur_t, k: int, thr: float = 0.4) -> dict:
    gh, gw = mem.grid
    qb = mem_mod._pool(blur_t.float(), gh, gw, mem.tok_native / 4).cpu().numpy().ravel()
    dq = qb >= thr
    nk = len(mem.key_box)
    if nk == 0 or dq.sum() == 0:
        return {"keys": nk, "defocused_tokens": int(dq.sum()), "coverage": 0.0, "recall": None, "chance": None}
    has = (idx[dq, 0] >= 0)
    hit = (oidx[dq, :1] == idx[dq, :k]).any(1) & (oidx[dq, 0] >= 0)
    return {"keys": nk, "defocused_tokens": int(dq.sum()), "coverage": float(has.mean()),
            f"recall@{k}": float(hit.mean()), "chance": float(min(1.0, k / nk))}


def worker(rank, world, device, a, names, dirs):
    names = shard(names, rank, world)
    feats = mem_mod.Features(a.features, a.dino_name, a.dino_hub, device)
    pix = mem_mod.Features("pixels", device=device) if a.compare_pixels else None
    out_root = Path(a.out)
    for n in names:
        od = out_root / n
        if (od / "meta.json").exists() and not a.overwrite:
            continue
        od.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        x, y = read_image(dirs["inputs"] / f"{n}.png"), read_image(dirs["targets"] / f"{n}.png")
        H, W = x.shape[:2]
        H4, W4 = H // 4, W // 4
        mpath = dirs["masks"] / f"{n}.png" if dirs.get("masks") else None
        mask = read_mask(mpath) if mpath and mpath.exists() else np.ones((H, W), bool)
        blur = load_blur_map(dirs["dp"], n, hw=(H4, W4), win=a.blur_win)
        if blur is None:
            raise FileNotFoundError(f"{n}: no DP map in {dirs['dp']}")
        save(od / "x.npy", x)
        save(od / "y.npy", y)
        save(od / "mask.npy", mask.astype(np.uint8))
        save(od / "blur.npy", blur.astype(np.float16))
        for m, d in dirs["anchors"].items():
            an = read_image(d / f"{n}.png")
            if an.shape[:2] != (H4, W4):
                raise ValueError(f"{n}: anchor {m} is {an.shape[:2]}, expected {(H4, W4)}")
            save(od / f"anchor_{m}.npy", an)
        anchor = to_tensor(read_image(dirs["anchors"][a.primary] / f"{n}.png"), device)
        blur_t = torch.from_numpy(blur)[None, None].to(device)
        mem = mem_mod.build_memory(anchor, blur_t, feats, a.scales, a.focus_thr, a.tex_pct, a.value_px)
        idx, sc = mem_mod.retrieve(mem, a.topk)
        xt, yt = to_tensor(x, device), to_tensor(y, device)
        oidx, osc = mem_mod.oracle(mem, yt, xt, a.oracle_k)
        np.savez(od / "mem.npz", key_box=mem.key_box, key_scale=mem.key_scale, grid=np.asarray(mem.grid),
                 tok_native=mem.tok_native, idx=idx, score=sc, oracle_idx=oidx, oracle_score=osc)
        hist = np.histogram(blur, bins=(-1,) + BINS + (1e9,))[0] / blur.size
        meta = {"name": n, "hw": [H, W], "dtype": str(x.dtype), "anchors": sorted(dirs["anchors"]),
                "m1_blur_bins": [round(float(v), 4) for v in hist],
                "m2": retrieval_stats(mem, idx, oidx, blur_t, a.k_eval), "time_s": round(time.time() - t0, 1)}
        if pix is not None:
            pm = mem_mod.build_memory(anchor, blur_t, pix, a.scales, a.focus_thr, a.tex_pct, a.value_px)
            pidx, _ = mem_mod.retrieve(pm, a.topk)
            poidx, _ = mem_mod.oracle(pm, yt, xt, a.oracle_k)
            meta["m2_pixels"] = retrieval_stats(pm, pidx, poidx, blur_t, a.k_eval)
        (od / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"[{rank}] {n} keys={len(mem.key_box)} {meta['time_s']}s", flush=True)


def manifest(out: Path, names: list[str], a) -> dict:
    metas = [json.loads((out / n / "meta.json").read_text()) for n in names if (out / n / "meta.json").exists()]
    m1 = np.mean([m["m1_blur_bins"] for m in metas], 0).round(4).tolist() if metas else None

    def agg(key):
        rows = [m[key] for m in metas if key in m and m[key].get("defocused_tokens")]
        if not rows:
            return None
        ks = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
        return {k: round(float(np.mean([r[k] for r in rows if r.get(k) is not None])), 4) for k in ks}

    man = {"split": a.split, "n": len(metas), "missing": sorted(set(names) - {m["name"] for m in metas}),
           "primary_anchor": a.primary, "features": a.features, "scales": a.scales, "topk": a.topk,
           "m1_blur_bins_b0_b3": m1, "m2": agg("m2"), "m2_pixels": agg("m2_pixels"),
           "args": {k: v for k, v in vars(a).items() if k not in ("out",)}, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--anchors", default="bokehlicious_deblur,drbnet_single,restormer_dpdd",
                    help="registry names of the x4 deblurrers whose anchors are cached")
    ap.add_argument("--primary", default="bokehlicious_deblur", help="anchor the memory / retrieval is built on")
    ap.add_argument("--anchor-dir", action="append", default=[], help="model=dir override (tests, non-layout runs)")
    ap.add_argument("--inputs"), ap.add_argument("--targets"), ap.add_argument("--masks"), ap.add_argument("--dp")
    ap.add_argument("--out", help="default: $UHDD_RESULTS/dpdd/cache/<split>")
    ap.add_argument("--features", default="dinov2", choices=["dinov2", "pixels"])
    ap.add_argument("--dino-name", default="dinov2_vitl14")
    ap.add_argument("--dino-hub", help="torch hub dir with facebookresearch_dinov2_main + checkpoints/ "
                                       "(default: $UHDD_WEIGHTS/vosr/torch_cache)")
    ap.add_argument("--scales", default="1,0.75,0.5")
    ap.add_argument("--focus-thr", type=float, default=0.4)
    ap.add_argument("--tex-pct", type=float, default=30.0)
    ap.add_argument("--value-px", type=int, default=64)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--k-eval", type=int, default=16)
    ap.add_argument("--oracle-k", type=int, default=4)
    ap.add_argument("--blur-win", type=int, default=15)
    ap.add_argument("--compare-pixels", action="store_true", help="M2 also for pixel features (ablation)")
    ap.add_argument("--gpus", default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    a.scales = tuple(float(s) for s in a.scales.split(","))
    import os
    if a.features == "dinov2" and not a.dino_hub and os.environ.get("UHDD_WEIGHTS"):
        a.dino_hub = str(Path(os.environ["UHDD_WEIGHTS"]) / "vosr" / "torch_cache")

    nat = layout.SPLITS[a.split]["native"]
    dirs = {"inputs": Path(a.inputs) if a.inputs else layout.input_dir(nat, "inputs"),
            "targets": Path(a.targets) if a.targets else layout.input_dir(nat, "targets"),
            "masks": Path(a.masks) if a.masks else layout.input_dir(nat, "masks"),
            "dp": Path(a.dp) if a.dp else layout.dp_maps_dir(a.split)}
    over = dict(s.split("=", 1) for s in a.anchor_dir)
    x4 = f"{a.split}_x4" if a.split != "test" else "ours_x4"
    dirs["anchors"] = {m: Path(over[m]) if m in over else layout.deblur_dir(x4, layout.tag(m))
                       for m in a.anchors.split(",")}
    if a.primary not in dirs["anchors"]:
        sys.exit(f"--primary {a.primary} must be one of --anchors")
    for k, v in [(k, v) for k, v in dirs.items() if k != "anchors"] + list(dirs["anchors"].items()):
        if not Path(v).is_dir():
            sys.exit(f"missing {k}: {v}")
    names = sorted(list_images(dirs["inputs"]))
    if a.limit:
        names = names[: a.limit]
    a.out = str(Path(a.out) if a.out else layout.cache_dir(a.split))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    launch(worker, parse_gpus(a.gpus), a, names, dirs)
    man = manifest(Path(a.out), names, a)
    print(json.dumps({k: man[k] for k in ("n", "missing", "m1_blur_bins_b0_b3", "m2", "m2_pixels")}, indent=1))
    return man


if __name__ == "__main__":
    main()
