#!/usr/bin/env python
"""Save the blur-bin maps (b0 focal plane ... b3 most defocused) of the DP blur level, per scene.

  save_blur_bins.py --dp-maps DIR --out DIR [--inputs DIR] [--size WxH] [--names a,b]

Per scene <name>: label/<name>.png (uint8, 0..3 = bin), b0..b3/<name>.png (uint8, 255 = pixel in that bin),
and, with --inputs, overlay/<name>.jpg (bins coloured over the input: b0 blue, b1 green, b2 yellow, b3 red).
Blur level = confidence-weighted |disparity| (`uhdd.dualpixel.load_blur_map`), bins as in the `blurbins` metric
(`uhdd.metrics.binned.bin_edges`). Default size = the DP resolution (1680x1120); other sizes resize the blur level
bilinearly before binning, exactly like the metric.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uhdd.dualpixel import load_blur_map  # noqa: E402
from uhdd.metrics.binned import bin_edges  # noqa: E402

COLORS = np.array([[255, 80, 0], [0, 200, 0], [0, 220, 255], [0, 0, 255]], np.uint8)   # BGR: b0 blue .. b3 red


def bins_of(blur: np.ndarray) -> np.ndarray:
    """uint8 label map 0..3 from a blur-level map."""
    e = bin_edges({})
    lab = np.zeros(blur.shape, np.uint8)
    for k in range(1, len(e) - 1):
        lab[blur >= e[k]] = k
    return lab


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dp-maps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", help="folder of blurry inputs (same names) for the overlay")
    ap.add_argument("--size", help="WxH of the saved maps (default: DP resolution)")
    ap.add_argument("--names", help="comma-separated scene names (default: all with a DP map)")
    a = ap.parse_args()
    out = Path(a.out)
    names = a.names.split(",") if a.names else sorted(p.name[: -len("_disp.png")] for p in Path(a.dp_maps).glob("*_disp.png"))
    for sub in ("label", "b0", "b1", "b2", "b3") + (("overlay",) if a.inputs else ()):
        (out / sub).mkdir(parents=True, exist_ok=True)
    hw = None
    if a.size:
        w, h = map(int, a.size.lower().split("x"))
        hw = (h, w)
    for n in names:
        blur = load_blur_map(a.dp_maps, n, hw)
        if blur is None:
            print(f"skip {n}: no DP map")
            continue
        lab = bins_of(blur)
        cv2.imwrite(str(out / "label" / f"{n}.png"), lab)
        for k in range(4):
            cv2.imwrite(str(out / f"b{k}" / f"{n}.png"), (lab == k).astype(np.uint8) * 255)
        if a.inputs:
            img = cv2.imread(str(Path(a.inputs) / f"{n}.png"), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (lab.shape[1], lab.shape[0]), interpolation=cv2.INTER_AREA)
            ov = (0.55 * img + 0.45 * COLORS[lab]).astype(np.uint8)
            cv2.imwrite(str(out / "overlay" / f"{n}.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(n, [round(float((lab == k).mean()), 3) for k in range(4)], flush=True)


if __name__ == "__main__":
    main()
