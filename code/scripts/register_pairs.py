#!/usr/bin/env python
"""Align targets to inputs (multi-process, CPU). See uhdd/registration.py.

  register_pairs.py --inputs IN --targets TG --out OUT [--motion homography] [--procs N]
  -> OUT/targets/<name>.png (aligned), OUT/masks/<name>.png (255 = valid), OUT/registration.csv

The CSV (corner shift in px, ECC correlation, warped yes/no) is what goes into the handoff
report as registration residual statistics.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from uhdd.io import pair_folders, read_image, write_image  # noqa: E402
from uhdd.registration import RegConfig, register_pair  # noqa: E402


def job(args):
    name, inp, tgt, out, cfg, threads = args
    cv2.setNumThreads(threads)
    r = register_pair(read_image(inp), read_image(tgt), cfg)
    write_image(Path(out) / "targets" / f"{name}.png", r["aligned"], png_compression=3)
    (Path(out) / "masks").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(Path(out) / "masks" / f"{name}.png"), r["mask"])
    return {"name": name, "corner_shift_px": round(r["corner_shift_px"], 3),
            "ecc": round(r["ecc"], 5), "warped": int(r["warped"]),
            "warp": " ".join(f"{v:.6g}" for v in np.asarray(r["warp"]).ravel())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--motion", default="homography", choices=["translation", "euclidean", "affine", "homography"])
    ap.add_argument("--levels", type=int, nargs="+", default=[16, 8, 4, 2])
    ap.add_argument("--min-shift", type=float, default=0.3)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 4) // 4))
    ap.add_argument("--cv-threads", type=int, default=4, help="OpenCV threads per process")
    a = ap.parse_args()

    cfg = RegConfig(motion=a.motion, levels=tuple(a.levels), min_shift=a.min_shift)
    pairs = pair_folders(a.inputs, a.targets)
    jobs = [(p.name, p.input, p.target, a.out, cfg, a.cv_threads) for p in pairs]
    rows = []
    with ProcessPoolExecutor(a.procs) as ex:
        for i, row in enumerate(ex.map(job, jobs), 1):
            rows.append(row)
            print(f"[{i}/{len(jobs)}] {row['name']} shift={row['corner_shift_px']}px ecc={row['ecc']}", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with open(Path(a.out) / "registration.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["name"]))
    s = np.array([r["corner_shift_px"] for r in rows])
    print(f"corner shift px: median {np.median(s):.2f}, p90 {np.percentile(s, 90):.2f}, max {s.max():.2f}; "
          f"warped {sum(r['warped'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
