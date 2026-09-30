"""Full-reference fidelity metrics on the GPU.

SSIM reproduces `skimage.metrics.structural_similarity(a, b, data_range=1, channel_axis=-1)`
(7x7 uniform window, sample covariance, mean over the border-cropped map and channels), which
is what the common DPDD evaluation scripts use [V: Restormer/IFAN eval scripts]. Computed in
float64 per channel so that flat regions (uxx - ux^2 cancellation) are exact.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean()
    m = mask.to(x.dtype).expand_as(x)
    return (x * m).sum() / m.sum().clamp(min=1)


def mse(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    d = (pred.double() - gt.double()).pow_(2)
    return _masked_mean(d, mask).item()


def psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    e = mse(pred, gt, mask)
    return float("inf") if e == 0 else 10 * math.log10(1.0 / e)


def mae(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    return _masked_mean((pred.double() - gt.double()).abs_(), mask).item()


def ssim(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None,
         win: int = 7, k1: float = 0.01, k2: float = 0.03) -> float:
    """skimage-compatible SSIM for 1xCxHxW in [0,1]; optional 1x1xHxW validity mask."""
    c1, c2 = (k1 * 1.0) ** 2, (k2 * 1.0) ** 2
    cov_norm = win * win / (win * win - 1)
    pool = lambda t: F.avg_pool2d(t, win, stride=1)  # 'valid' == skimage's border crop
    if mask is not None:  # a window counts only if all its pixels are valid
        mvalid = -F.max_pool2d(-mask.double(), win, stride=1)
    vals = []
    for c in range(pred.shape[1]):
        x, y = pred[:, c:c + 1].double(), gt[:, c:c + 1].double()
        ux, uy = pool(x), pool(y)
        vx = cov_norm * (pool(x * x) - ux * ux)
        vy = cov_norm * (pool(y * y) - uy * uy)
        vxy = cov_norm * (pool(x * y) - ux * uy)
        s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
        vals.append(_masked_mean(s, mvalid if mask is not None else None))
    return torch.stack(vals).mean().item()
