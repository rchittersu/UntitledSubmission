"""Pad-to-multiple handling for architectures with size constraints.

Many restoration networks need H and W divisible by some factor m:
  2^(#downsamplings) for U-Nets (Restormer: 8, NAFNet: 16), and
  window_size * 2^(#downsamplings) for windowed transformers (e.g. Uformer: 8*16 = 128).

We never resize to satisfy this (resizing changes the circle-of-confusion size and the
pixel grid, which breaks both the task and the metrics). Instead we pad bottom/right with
reflection, run the model, and crop back. Reflection keeps image statistics at the border
(zero padding creates an artificial edge the network tries to deblur).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def round_up(x: int, m: int) -> int:
    return -(-x // m) * m


def pad_to(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Pad NCHW `x` at bottom/right to at least h x w.

    Uses reflection where possible; reflect requires pad < size, so very small inputs
    fall back to repeated reflection and finally replicate.
    """
    H, W = x.shape[-2:]
    ph, pw = max(h - H, 0), max(w - W, 0)
    while ph > 0 or pw > 0:
        H, W = x.shape[-2:]
        sh, sw = min(ph, H - 1), min(pw, W - 1)
        if sh == 0 and sw == 0:  # 1-pixel image dimension: cannot reflect
            return F.pad(x, (0, pw, 0, ph), mode="replicate")
        x = F.pad(x, (0, sw, 0, sh), mode="reflect")
        ph, pw = ph - sh, pw - sw
    return x


def pad_to_multiple(x: torch.Tensor, m: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad so H and W are multiples of m. Returns (padded, original (H, W))."""
    H, W = x.shape[-2:]
    if m <= 1:
        return x, (H, W)
    return pad_to(x, round_up(H, m), round_up(W, m)), (H, W)


def crop(x: torch.Tensor, hw: tuple[int, int], scale: int = 1) -> torch.Tensor:
    return x[..., : hw[0] * scale, : hw[1] * scale]
