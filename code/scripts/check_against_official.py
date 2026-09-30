#!/usr/bin/env python
"""Compare our pipeline (models.build + run_tiled, whole image) with each repo's own test code path.

  check_against_official.py restormer? lakdnet_dpdd_l drbnet_single ifan swinir_x4 swinir_x4_real

Uses the demo images shipped with Restormer ($UHDD_REPOS/Restormer/demo/degraded) unless
--images is given. Prints max |ours - official| (expect ~1e-6), tiled-vs-whole PSNR and runtime.
Bokehlicious and Restormer were checked separately (see README).
"""
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_ps = F.pixel_shuffle  # some old x86 CPUs: channels_last pixel_shuffle needs FBGEMM; harmless elsewhere
F.pixel_shuffle = lambda x, r: _ps(x.contiguous(), r)

import argparse
import glob
import os
from pathlib import Path

CODE = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, CODE)
from uhdd import models  # noqa: E402
from uhdd.io import read_image, to_tensor  # noqa: E402
from uhdd.tiling import TileSpec, run_tiled  # noqa: E402

IMGS: list[str] = []
CFG = f"{CODE}/configs/models.yaml"


def img01(path):  # 8-bit RGB float [0,1], NCHW
    return torch.from_numpy(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) / 255.).float().permute(2, 0, 1)[None]


def reflect_pad(C, f):  # LaKDNet/Restormer crop_image
    h, w = C.shape[-2:]
    H, W = ((h + f) // f) * f, ((w + f) // f) * f
    return F.pad(C, (0, W - w if w % f else 0, 0, H - h if h % f else 0), "reflect"), h, w


def ref_lakdnet(net, path):
    C, h, w = reflect_pad(img01(path), 8)
    return net(C)[..., :h, :w].clamp(0, 1), (h, w)


def ref_drbnet(net, path):
    C = img01(path)
    h, w = C.shape[-2:]
    C = C[..., : h - h % 16, : w - w % 16]
    return ((net(C * 2 - 1) + 1) / 2).clamp(0, 1), C.shape[-2:]


def ref_ifan(net, path):
    C = img01(path)
    h, w = C.shape[-2:]
    C = C[..., : h - h % 8, : w - w % 8]
    return net(C)["result"], C.shape[-2:]


def ref_swinir(net, path):  # crop to multiple of 8 so the official flip-padding is a no-op
    C = img01(path)
    h, w = C.shape[-2:]
    C = C[..., : h - h % 8, : w - w % 8]
    return net(C).clamp(0, 1), C.shape[-2:]


REFS = {"lakdnet_dpdd_l": ref_lakdnet, "lakdnet_dpdd_s": ref_lakdnet, "drbnet_single": ref_drbnet,
        "ifan": ref_ifan, "swinir_x4": ref_swinir, "swinir_x4_real": ref_swinir}


def raw_net(m):
    """The nn.Module inside our wrapper (closure of _wrap / adapter)."""
    for c in m.fn.__closure__ or ():
        if isinstance(c.cell_contents, torch.nn.Module):
            return c.cell_contents
    raise RuntimeError("no module in closure")


@torch.no_grad()
def main(names):
    for name in names:
        m = models.build(name, CFG, "cpu")
        net = raw_net(m)
        for path in IMGS:
            ref, (h, w) = REFS[name](net, path)
            x = to_tensor(read_image(path), "cpu")[..., :h, :w]
            if m.spec.get("input_bits"):
                x = x.mul(255).round().div(255)
            t = time.time()
            ours, _ = run_tiled(m, x, TileSpec(0), m.multiple, m.scale)
            dt = time.time() - t
            til, _ = run_tiled(m, x, TileSpec(128, 32, 1, "linear"), m.multiple, m.scale)
            psnr = lambda a, b: float(10 * torch.log10(1 / ((a - b) ** 2).mean()))
            base = F.interpolate(x, scale_factor=m.scale, mode="bicubic") if m.scale > 1 else x
            print(f"{name:16s} {path.split('/')[-1]:14s} max|ours-ref|={float((ours - ref).abs().max()):.2e}  "
                  f"tiled128 vs whole {psnr(til, ours):5.2f} dB  change vs input {psnr(ours.clamp(0, 1), base.clamp(0, 1)):5.2f} dB  {dt:.1f}s",
                  flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", choices=list(REFS))
    ap.add_argument("--images", default=os.path.join(os.environ.get("UHDD_REPOS", "."), "Restormer/demo/degraded"))
    a = ap.parse_args()
    IMGS[:] = sorted(glob.glob(os.path.join(a.images, "*")))[:2]
    main(a.models)
