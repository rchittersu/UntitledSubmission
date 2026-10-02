#!/usr/bin/env python
"""VAE reconstruction ceiling at native resolution (chooses the latent backbone of variant B).

  vae_ceiling.py --targets DATA/x1/targets --dp-maps DATA/dp_maps --out vae_ceiling.csv \
      --vae sd35=stabilityai/stable-diffusion-3.5-medium:vae --vae sdxl=madebyollin/sdxl-vae-fp16-fix \
      [--crop 1024 --per-region 2 --device cuda:0 --dtype bf16]

Any latent prior can at best reproduce what its VAE reconstructs. Per image, crops of --crop native px are
taken on a grid; the --per-region crops with the lowest mean DP blur level ("focus": real native texture
that must survive) and the highest ("defocus") are encoded + decoded by each VAE (spec name=repo[:subfolder];
'dc-ae' in the repo name -> AutoencoderDC, else AutoencoderKL). Reports PSNR, LPIPS and high-band NMSE of the
reconstruction vs. the target crop, per VAE x region (mean over crops), and per crop in --out (CSV).
Without --dp-maps, crops are taken at random positions (region "all").
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from uhdd.dualpixel import load_blur_map  # noqa: E402
from uhdd.io import list_images, read_image, to_tensor  # noqa: E402
from uhdd.metrics import fidelity, iqa, optics  # noqa: E402


def pick_crops(blur: np.ndarray | None, hw, crop: int, k: int, rng) -> list[tuple[str, int, int]]:
    """Top-left corners of the k lowest- and k highest-blur crops on a stride-`crop` grid."""
    H, W = hw
    ys, xs = range(0, H - crop + 1, crop), range(0, W - crop + 1, crop)
    cells = [(y, x) for y in ys for x in xs]
    if blur is None:
        idx = rng.choice(len(cells), size=min(2 * k, len(cells)), replace=False)
        return [("all", *cells[i]) for i in idx]
    order = np.argsort([blur[y:y + crop, x:x + crop].mean() for y, x in cells])
    return [("focus", *cells[i]) for i in order[:k]] + [("defocus", *cells[i]) for i in order[::-1][:k]]


def load_vae(spec: str, device, dtype):
    name, ref = spec.split("=", 1)
    repo, _, sub = ref.partition(":")
    import diffusers
    cls = diffusers.AutoencoderDC if "dc-ae" in repo.lower() else diffusers.AutoencoderKL
    vae = cls.from_pretrained(repo, subfolder=sub or None, torch_dtype=dtype).to(device).eval()
    return name, vae


@torch.no_grad()
def reconstruct(vae, x: torch.Tensor) -> torch.Tensor:
    """x in [0,1], 1x3xHxW -> reconstruction in [0,1] (float32)."""
    z = vae.encode((x * 2 - 1).to(vae.dtype))
    z = z.latent_dist.mode() if hasattr(z, "latent_dist") else z.latent
    return ((vae.decode(z).sample.float() + 1) / 2).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--dp-maps")
    ap.add_argument("--vae", action="append", required=True, help="name=repo[:subfolder], repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--per-region", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="first N images only")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    a = ap.parse_args()
    device = torch.device(a.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    rng = np.random.default_rng(0)

    imgs = sorted(list_images(a.targets).items())[: a.limit or None]
    crops = []
    for name, path in imgs:
        img = read_image(path)
        blur = load_blur_map(a.dp_maps, name, img.shape[:2]) if a.dp_maps else None
        for region, y, x in pick_crops(blur, img.shape[:2], a.crop, a.per_region, rng):
            crops.append((name, region, y, x, img[y:y + a.crop, x:x + a.crop].copy()))
    print(f"{len(crops)} crops from {len(imgs)} images", flush=True)

    rows, agg = [], defaultdict(list)
    for spec in a.vae:
        vname, vae = load_vae(spec, device, dtype)
        for name, region, y, x, c in crops:
            gt = to_tensor(c, device)
            rec = reconstruct(vae, gt)
            r = {"vae": vname, "name": name, "region": region, "y": y, "x": x,
                 "psnr": fidelity.psnr(rec, gt), "lpips": iqa.full_reference("lpips", rec, gt),
                 "hb_nmse_db": optics.highband_nmse_db(rec, gt)}
            rows.append(r)
            agg[(vname, region)].append(r)
        del vae
        torch.cuda.empty_cache() if device.type == "cuda" else None
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("| VAE | region | n | PSNR ↑ | LPIPS ↓ | HB-NMSE dB ↓ |\n|---|---|---|---|---|---|")
    for (v, reg), rs in agg.items():
        m = lambda k: np.mean([r[k] for r in rs])
        print(f"| {v} | {reg} | {len(rs)} | {m('psnr'):.2f} | {m('lpips'):.3f} | {m('hb_nmse_db'):+.2f} |")


if __name__ == "__main__":
    main()
