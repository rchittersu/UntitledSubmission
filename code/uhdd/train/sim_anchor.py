"""Simulated anchors for training (plan/method_plan.md §4): the target downscaled x4 and degraded like a real
low-resolution deblurring result -- residual blur that grows with the local defocus, ringing (over-sharpening),
noise. Parameters are drawn per sample from ranges in the training config; calibrate the ranges so that the
per-blur-bin error of simulated anchors matches real test-set anchors (measurement M3).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT = {
    "blur_per_disp": [0.15, 0.6],   # residual Gaussian sigma (anchor px) per unit |DP disparity|
    "blur_floor": [0.0, 0.4],       # sigma everywhere (anchor px)
    "ringing": [0.0, 0.6],          # unsharp-mask amount (sigma 1 anchor px)
    "noise": [0.0, 0.01],           # Gaussian noise std ([0, 1] scale)
    "levels": [0.0, 0.5, 1.0, 2.0, 3.0],   # sigmas blended per pixel (spatially varying blur)
}


def _gauss(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.05:
        return x
    r = max(1, int(3 * sigma + 0.5))
    k = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    c = x.shape[1]
    x = F.pad(x, (r, r, r, r), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    return F.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def simulate(target4: torch.Tensor, blur4: torch.Tensor, rng: torch.Generator, cfg: dict | None = None) -> torch.Tensor:
    """target4: 1x3xhxw target area-downscaled x4 (with margin); blur4: 1x1xhxw |DP disparity| -> 1x3xhxw anchor."""
    c = {**DEFAULT, **(cfg or {})}

    def u(lo_hi):
        lo, hi = lo_hi
        return lo + (hi - lo) * torch.rand((), generator=rng).item()

    sigma = (u(c["blur_floor"]) + u(c["blur_per_disp"]) * blur4).clamp(0, c["levels"][-1])
    levels = c["levels"]
    stack = torch.stack([_gauss(target4, s) for s in levels], 0)             # L x 1 x 3 x h x w
    pos = torch.zeros_like(sigma)
    for i in range(len(levels) - 1):                                         # fractional level index per pixel
        lo, hi = levels[i], levels[i + 1]
        pos = torch.where((sigma >= lo) & (sigma <= hi), i + (sigma - lo) / (hi - lo), pos)
    i0 = pos.floor().long().clamp(max=len(levels) - 2)
    w = (pos - i0).unsqueeze(0)
    a0 = torch.gather(stack, 0, i0.unsqueeze(0).expand(1, *stack.shape[1:]))
    a1 = torch.gather(stack, 0, (i0 + 1).unsqueeze(0).expand(1, *stack.shape[1:]))
    a = ((1 - w) * a0 + w * a1)[0]
    ring = u(c["ringing"])
    if ring > 0:
        a = a + ring * (a - _gauss(a, 1.0))
    n = u(c["noise"])
    if n > 0:
        a = a + n * torch.randn(a.shape, generator=rng, dtype=a.dtype)
    return a.clamp(0, 1)
