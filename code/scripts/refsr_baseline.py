#!/usr/bin/env python
"""Reference-based SR baseline (docs/baselines.md, D1): DATSR x4 on the 1/4-res deblurred anchor, with the
blurry native input as the reference. The closest prior-work alternative to the in-focus exemplar memory.

  refsr_baseline.py --anchors4 RES/drb_x4 --inputs NAT/inputs --dp-maps DP --out RES/drb_x4+datsr_mosaic \
      [--ref mosaic|colocated] [--weights restoration_mse|restoration_gan] [--tile 128 --overlap 16] [--device cuda]

DATSR needs a reference of the same size as each HR output tile (its deformable aggregation samples the ref
features on the output grid), so per LR tile of `tile` px (HR 4*tile):
  colocated  ref = the blurry native input at the tile's own location (in-focus tiles: the exact sharp content;
             defocused tiles: blurry, so DATSR falls back to SR).
  mosaic     ref = 2x2 mosaic of 2*tile-px native crops: the central co-located crop + the 3 in-focus crops of the
             whole image most similar to the anchor tile. Candidates: crops on a stride-tile grid whose DP copy
             weight (ramp 0.4-1.2 DP px, as the composite) averages > 0.8. Similarity: Euclidean distance of
             standardized 15-d statistics at 1/4 res (mean / std RGB, 8-bin magnitude-weighted gradient-orientation
             histogram, log mean gradient magnitude) of the anchor tile vs. the 4x-downscaled candidate.
             DATSR's own patch matching then picks what it uses from the mosaic.
Output: 16-bit native PNGs + meta.json (time_s per image). Weights: github.com/caojiezhang/DATSR/releases.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from uhdd.adapters import datsr  # noqa: E402
from uhdd.dualpixel import load_blur_map  # noqa: E402
from uhdd.io import read_image, write_image  # noqa: E402
from uhdd.tiling import TileSpec, run_tiled  # noqa: E402


def stats_desc(img: np.ndarray) -> np.ndarray:
    """15-d texture/colour statistics of an HxWx3 [0,1] crop."""
    y = img @ np.array([0.299, 0.587, 0.114], np.float32)
    gx, gy = cv2.Sobel(y, cv2.CV_32F, 1, 0), cv2.Sobel(y, cv2.CV_32F, 0, 1)
    mag, ang = np.hypot(gx, gy), np.mod(np.arctan2(gy, gx), np.pi)
    hist = np.bincount(np.minimum((ang / np.pi * 8).astype(int), 7).ravel(), mag.ravel(), 8)
    hist = hist / (hist.sum() + 1e-6)
    return np.concatenate([img.reshape(-1, 3).mean(0), img.reshape(-1, 3).std(0), hist, [np.log(mag.mean() + 1e-3)]])


class RefBuilder:
    def __init__(self, x: np.ndarray, anchor4: np.ndarray, blur4: np.ndarray, tile: int, mode: str):
        self.x, self.a4, self.mode, self.cr = x, anchor4, mode, 2 * tile            # native crop size
        if mode != "mosaic":
            return
        H, W = x.shape[:2]
        x4 = cv2.resize(x, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        w4 = np.clip((1.2 - blur4) / 0.8, 0, 1)
        c4 = self.cr // 4
        self.cands, descs = [], []
        for y in range(0, x4.shape[0] - c4 + 1, tile // 4):
            for xx in range(0, x4.shape[1] - c4 + 1, tile // 4):
                if w4[y:y + c4, xx:xx + c4].mean() > 0.8:
                    self.cands.append((4 * y, 4 * xx))
                    descs.append(stats_desc(x4[y:y + c4, xx:xx + c4]))
        if descs:
            d = np.array(descs)
            self.mu, self.sd = d.mean(0), d.std(0) + 1e-6
            self.descs = (d - self.mu) / self.sd

    def crop(self, y: int, x: int, s: int, sw: int | None = None) -> np.ndarray:
        """s x sw native crop at (y, x), shifted inside the image."""
        H, W = self.x.shape[:2]
        sw = sw or s
        y, x = min(max(y, 0), H - s), min(max(x, 0), W - sw)
        return self.x[y:y + s, x:x + sw]

    def __call__(self, box) -> np.ndarray:
        y, x, h, w = box                                                           # LR coords (may exceed image)
        if self.mode == "colocated":
            return self.crop(4 * y, 4 * x, 4 * h, 4 * w)
        cr = self.cr
        cy, cx = 4 * y + 2 * h - cr // 2, 4 * x + 2 * w - cr // 2
        parts = [self.crop(cy, cx, cr)]
        if getattr(self, "cands", None):
            Ha, Wa = self.a4.shape[:2]
            q = self.a4[max(y, 0):min(y + h, Ha), max(x, 0):min(x + w, Wa)]
            dq = (stats_desc(q) - self.mu) / self.sd
            for j in np.argsort(((self.descs - dq) ** 2).sum(1))[:3]:
                parts.append(self.crop(*self.cands[j], cr))
        parts += [parts[0]] * (4 - len(parts))
        return np.concatenate([np.concatenate(parts[:2], 1), np.concatenate(parts[2:], 1)], 0)[:4 * h, :4 * w]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchors4", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--dp-maps")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default="mosaic", choices=["mosaic", "colocated"])
    ap.add_argument("--weights", default="restoration_mse", choices=["restoration_mse", "restoration_gan"])
    ap.add_argument("--repo", default=os.path.join(os.environ.get("UHDD_REPOS", "ext/repos"), "DATSR"))
    ap.add_argument("--weights-dir", default=os.path.join(os.environ.get("UHDD_WEIGHTS", "ext/weights"), "datsr"))
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--overlap", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--only", help="comma-separated names")
    a = ap.parse_args()
    if a.ref == "mosaic" and not a.dp_maps:
        sys.exit("--ref mosaic needs --dp-maps")
    device = torch.device(a.device)
    nets = datsr.build(a.repo, f"{a.weights_dir}/{a.weights}.pth", f"{a.weights_dir}/feature_extraction.pth", device)
    names = sorted(p.stem for p in Path(a.anchors4).glob("*.png"))
    if a.only:
        names = [n for n in names if n in set(a.only.split(","))]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"images": {}, "summary": {"method": f"datsr_{a.weights}_{a.ref}", "tile": a.tile, "overlap": a.overlap}}
    for n in names:
        a4 = read_image(Path(a.anchors4) / f"{n}.png")
        a4 = a4.astype(np.float32) / (65535.0 if a4.dtype == np.uint16 else 255.0)
        x = read_image(Path(a.inputs) / f"{n}.png").astype(np.float32) / 65535.0
        blur4 = load_blur_map(a.dp_maps, n, a4.shape[:2]) if a.dp_maps else None
        refs = RefBuilder(x, a4, blur4, a.tile, a.ref)

        def fn(t, boxes, full_hw):
            ref = torch.stack([torch.from_numpy(np.ascontiguousarray(refs(b))).permute(2, 0, 1) for b in boxes]).to(t)
            return datsr.run(nets, t, ref)
        fn.positional = True

        t0 = time.time()
        y, grid = run_tiled(fn, torch.from_numpy(a4).permute(2, 0, 1)[None].to(device),
                            TileSpec(tile=a.tile, overlap=a.overlap, batch=a.batch), multiple=16, scale=4)
        dt = time.time() - t0
        write_image(out / f"{n}.png", np.rint(y[0].permute(1, 2, 0).cpu().numpy() * 65535).astype(np.uint16))
        meta["images"][n] = {"time_s": dt, "tiling": grid, "n_candidates": len(getattr(refs, "cands", []))}
        print(f"{n}: {dt:.0f} s, {len(getattr(refs, 'cands', []))} in-focus candidates", flush=True)
        (out / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
