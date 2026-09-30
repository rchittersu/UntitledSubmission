#!/usr/bin/env python
"""Build downscaled copies of a paired set (multi-process).

  prepare_scales.py --inputs IN --targets TG --out OUT --factors 2 4 [--masks M] [--filter area]
  -> OUT/x2/{inputs,targets,masks}/..., OUT/x4/...

Integer factors with `area` = exact s x s box average (cv2.INTER_AREA), `bicubic_aa` = torch
antialiased bicubic. Masks are downscaled conservatively (a low-res pixel is valid only if all
its native pixels are). Native bit depth is preserved.

--reference DIR compares our 1/4 inputs against an official low-res release (e.g. DPDD
1680x1120) for every filter and prints PSNR, to pick the filter that reproduces it.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("UHDD_CV2_THREADS", "1")  # parallel over processes instead

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from uhdd.io import list_images, pair_folders, read_image, write_image  # noqa: E402


def downscale(img: np.ndarray, f: int, filt: str) -> np.ndarray:
    H, W = img.shape[:2]
    if H % f or W % f:
        img = img[: H // f * f, : W // f * f]  # crop remainder so the grid stays aligned
    if filt == "area":
        return cv2.resize(img, (img.shape[1] // f, img.shape[0] // f), interpolation=cv2.INTER_AREA)
    if filt == "bicubic_aa":
        import torch
        import torch.nn.functional as F
        maxv = 65535.0 if img.dtype == np.uint16 else 255.0
        t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)[None] / maxv
        t = F.interpolate(t, scale_factor=1 / f, mode="bicubic", antialias=True, align_corners=False)
        return np.clip(np.rint(t[0].permute(1, 2, 0).numpy() * maxv), 0, maxv).astype(img.dtype)
    raise ValueError(filt)


def downscale_mask(m: np.ndarray, f: int) -> np.ndarray:
    H, W = m.shape
    m = m[: H // f * f, : W // f * f]
    return (m.reshape(H // f, f, W // f, f).min(axis=(1, 3)) > 127).astype(np.uint8) * 255


def job(args):
    name, paths, out, factors, filt, overwrite = args
    for key, p in paths.items():
        if p is None:
            continue
        if key == "masks":
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        else:
            img = read_image(p)
        for f in factors:
            dst = Path(out) / f"x{f}" / key / f"{name}.png"
            if dst.exists() and not overwrite:
                continue
            if key == "masks":
                dst.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dst), downscale_mask(img, f))
            else:
                write_image(dst, downscale(img, f, filt), png_compression=3)
    return name


def compare_reference(pairs, ref_dir: str, f: int, n: int) -> None:
    ref = list_images(ref_dir)
    names = [p.name for p in pairs if p.name in ref][:n]
    if not names:
        sys.exit(f"no names in common with {ref_dir}")
    byname = {p.name: p for p in pairs}
    norm = lambda x: x.astype(np.float64) / (65535.0 if x.dtype == np.uint16 else 255.0)
    imgs = {k: (read_image(byname[k].input), norm(read_image(ref[k]))) for k in names}
    for filt in ("area", "bicubic_aa"):
        ps = []
        for k, (native, b) in imgs.items():
            a = norm(downscale(native, f, filt))
            if a.shape != b.shape:
                sys.exit(f"shape mismatch {k}: ours {a.shape} vs reference {b.shape}")
            ps.append(10 * np.log10(1 / max(np.mean((a - b) ** 2), 1e-20)))
        print(f"filter={filt:11s} PSNR vs reference over {len(ps)} images: "
              f"mean {np.mean(ps):.2f} dB, min {np.min(ps):.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--targets")
    ap.add_argument("--masks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--factors", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--filter", default="area", choices=["area", "bicubic_aa"])
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--overwrite", action="store_true", help="rewrite existing outputs")
    ap.add_argument("--reference", help="official low-res inputs to compare filters against")
    ap.add_argument("--reference-n", type=int, default=10)
    a = ap.parse_args()

    pairs = pair_folders(a.inputs, a.targets)
    if a.reference:
        compare_reference(pairs, a.reference, max(a.factors), a.reference_n)
        return
    masks = list_images(a.masks) if a.masks else {}
    jobs = [(p.name, {"inputs": p.input, "targets": p.target, "masks": masks.get(p.name)},
             a.out, a.factors, a.filter, a.overwrite) for p in pairs]
    with ProcessPoolExecutor(a.procs) as ex:
        for i, name in enumerate(ex.map(job, jobs), 1):
            print(f"[{i}/{len(jobs)}] {name}", flush=True)


if __name__ == "__main__":
    main()
