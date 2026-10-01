"""Defocus from dual-pixel (DP) views.

In a DP capture the left/right sub-aperture views of an out-of-focus point are displaced
horizontally by an amount proportional to its signed defocus (blur radius), and coincide at the
focal plane. DPDD provides both views of the blurry f/4 capture (`*_l`, `*_r`, 1680x1120).

`dp_disparity` estimates a dense horizontal disparity with windowed block matching on
band-passed luminance (robust to the brightness difference of the two views), sub-pixel by a
parabola fit, plus a confidence from the cost curvature and local texture. Uses:
  - focal-plane mask (|disparity| small, confident): where the f/4 input is in focus;
  - |disparity| as a per-pixel blur level for stratified metrics (plan/evaluation.md 3.3).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _luma(x: torch.Tensor) -> torch.Tensor:
    w = torch.tensor([0.299, 0.587, 0.114], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * w).sum(1, keepdim=True) if x.shape[1] == 3 else x


def _box(x: torch.Tensor, r: int) -> torch.Tensor:
    return F.avg_pool2d(F.pad(x, (r, r, r, r), mode="reflect"), 2 * r + 1, stride=1)


def _bandpass(x: torch.Tensor, r: int = 6) -> torch.Tensor:
    y = x - _box(x, r)                                   # remove illumination / view gain
    return y / (_box(y.abs(), r) + 1e-4)                 # local contrast normalization


@torch.inference_mode()
def dp_disparity(left: torch.Tensor, right: torch.Tensor, max_disp: int = 12, win: int = 7
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Horizontal disparity (pixels, signed; right shifted to match left) and confidence in [0,1].

    left/right: 1xCxHxW in [0,1]. Returns two 1x1xHxW tensors.
    """
    L, R = _bandpass(_luma(left)), _bandpass(_luma(right))
    costs = []
    for d in range(-max_disp, max_disp + 1):
        Rs = torch.roll(R, shifts=d, dims=-1)
        costs.append(_box((L - Rs) ** 2, win))
    C = torch.cat(costs, 1)                              # 1 x D x H x W
    i = C.argmin(1, keepdim=True)
    ic = i.clamp(1, C.shape[1] - 2)
    c0, cm, cp = C.gather(1, ic), C.gather(1, ic - 1), C.gather(1, ic + 1)
    denom = (cm - 2 * c0 + cp)
    sub = torch.where(denom > 1e-6, 0.5 * (cm - cp) / denom, torch.zeros_like(c0)).clamp(-0.5, 0.5)
    disp = (ic - max_disp).float() + sub
    # confidence: cost curvature relative to the mean cost, and not at the search boundary
    curv = (denom / (C.mean(1, keepdim=True) + 1e-6)).clamp(0, 1)
    edge = ((i > 0) & (i < C.shape[1] - 1)).float()
    hp = _luma(left) - _box(_luma(left), 3)
    tex = (_box(hp ** 2, win).sqrt() / 0.01).clamp(0, 1)   # flat regions give unreliable matches
    return disp, curv * edge * tex


def focus_mask(disp: torch.Tensor, conf: torch.Tensor, max_abs: float = 0.35, min_conf: float = 0.3
               ) -> torch.Tensor:
    """Boolean 1x1xHxW mask of confidently in-focus pixels (focal plane)."""
    return (disp.abs() <= max_abs) & (conf >= min_conf)
