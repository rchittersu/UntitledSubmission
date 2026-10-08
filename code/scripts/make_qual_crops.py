#!/usr/bin/env python
"""Crops for the qualitative figures: one JPEG per (row, column) = a native-resolution window of a method's output.

  make_qual_crops.py [--config code/configs/qual_figure.yaml] [--size N] [--rows a,b]

Writes <out>/<row>_<column>.jpg (what Template/fig/qual*.tex includes) and <out>/<row>_context.jpg (the input at
1/8 scale with the window drawn). 16-bit sources are scaled to 8 bit. Images never leave the secure environment
with the patch (text only): copy the folder to Template/fig/qual/ where the paper is compiled.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import yaml


def expand(s: str) -> str:
    return re.sub(r"\$\{(\w+)\}", lambda m: os.environ[m.group(1)], s)


def to8(img: np.ndarray) -> np.ndarray:
    return np.clip(img.astype(np.float32) / (257.0 if img.dtype == np.uint16 else 1.0), 0, 255).astype(np.uint8)


def crop(path: str, y: int, x: int, size: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    H, W = img.shape[:2]
    y, x = int(np.clip(y, 0, H - size)), int(np.clip(x, 0, W - size))
    return to8(img[y:y + size, x:x + size])


def context(path: str, y: int, x: int, size: int, scale: int = 8) -> np.ndarray:
    img = to8(cv2.imread(path, cv2.IMREAD_UNCHANGED))
    small = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale), interpolation=cv2.INTER_AREA)
    cv2.rectangle(small, (x // scale, y // scale), ((x + size) // scale, (y + size) // scale), (0, 255, 255), 2)
    return small


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "qual_figure.yaml"))
    ap.add_argument("--size", type=int)
    ap.add_argument("--rows")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    size = a.size or cfg.get("size", 256)
    out = Path(expand(cfg["out"]))
    out.mkdir(parents=True, exist_ok=True)
    for name, r in cfg["rows"].items():
        if a.rows and name not in a.rows.split(","):
            continue
        rs = int(r.get("size", size))                      # per-row window (native px); saved at `out_px` (default 256)
        px = int(r.get("out_px", size if a.size else cfg.get("size", 256)))
        for col, tpl in cfg["columns"].items():
            p = expand(tpl).format(scene=r["scene"])
            t = crop(p, r["y"], r["x"], rs)
            if t.shape[0] != px:
                t = cv2.resize(t, (px, px), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / f"{name}_{col}.jpg"), t, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out / f"{name}_context.jpg"), context(expand(cfg["columns"]["input"]).format(scene=r["scene"]),
                                                              r["y"], r["x"], rs), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{name}: scene {r['scene']} window y={r['y']} x={r['x']} size={rs} -> {px} px, {out}")


if __name__ == "__main__":
    main()
