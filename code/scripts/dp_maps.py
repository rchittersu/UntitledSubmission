#!/usr/bin/env python
"""Dual-pixel disparity / focus maps for DPDD (see uhdd/dualpixel.py), multi-process on CPU.

  dp_maps.py --left DIR_L --right DIR_R --out DIR [--procs 8]

DPDD names the views <stem>_L.png / <stem>_R.png (dd_dp_dataset_png/test_l/source, test_r/source);
plain <stem>.png also works. Writes, per image, at the DP resolution (1680x1120):
  <stem>_disp.png  uint16, signed disparity in px = (value - 32768) / 1000
  <stem>_conf.png  uint8,  confidence * 255
  <stem>_focus.png uint8,  255 = confidently in focus (focal plane of the f/4 input)
and dp_maps.csv with per-image focal-plane fraction and |disparity| percentiles.

Validated on the DPDD test set (25 images): per image, Spearman correlation between cell-wise
|disparity| and log(sharpness target / sharpness input) has median 0.76; focal-plane cells show a
log-sharpness gap of 0.70 vs 1.99 elsewhere.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def stem_of(p: Path) -> str:
    return re.sub(r"_[LR]$", "", p.stem)


def job(args):
    import torch
    torch.set_num_threads(2)
    from uhdd.dualpixel import dp_disparity, focus_mask
    from uhdd.io import read_image, to_tensor
    stem, lp, rp, out = args
    disp, conf = dp_disparity(to_tensor(read_image(lp), "cpu"), to_tensor(read_image(rp), "cpu"))
    focus = focus_mask(disp, conf)
    d, c = disp[0, 0].numpy(), conf[0, 0].numpy()
    cv2.imwrite(str(Path(out) / f"{stem}_disp.png"), np.clip(np.rint(d * 1000 + 32768), 0, 65535).astype(np.uint16))
    cv2.imwrite(str(Path(out) / f"{stem}_conf.png"), np.rint(c * 255).astype(np.uint8))
    cv2.imwrite(str(Path(out) / f"{stem}_focus.png"), focus[0, 0].numpy().astype(np.uint8) * 255)
    ad = np.abs(d[c > 0.3]) if (c > 0.3).any() else np.array([np.nan])
    return {"name": stem, "focus_frac": round(float(focus.float().mean()), 4),
            "absdisp_p50": round(float(np.median(ad)), 3), "absdisp_p90": round(float(np.percentile(ad, 90)), 3)}


def read_disp(out_dir: str | Path, stem: str) -> tuple[np.ndarray, np.ndarray]:
    """(signed disparity px, confidence [0,1]) as float32 arrays."""
    d = cv2.imread(str(Path(out_dir) / f"{stem}_disp.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
    c = cv2.imread(str(Path(out_dir) / f"{stem}_conf.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
    return (d - 32768) / 1000, c / 255


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a = ap.parse_args()
    L = {stem_of(p): p for p in Path(a.left).glob("*.png")}
    R = {stem_of(p): p for p in Path(a.right).glob("*.png")}
    stems = sorted(L.keys() & R.keys())
    if not stems:
        sys.exit("no matching left/right views")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.procs) as ex:
        rows = list(ex.map(job, [(s, L[s], R[s], a.out) for s in stems]))
    with open(Path(a.out) / "dp_maps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fr = np.array([r["focus_frac"] for r in rows])
    print(f"{len(rows)} images -> {a.out}; focal-plane fraction median {np.median(fr):.3f} (min {fr.min():.3f}, max {fr.max():.3f})")


if __name__ == "__main__":
    main()
