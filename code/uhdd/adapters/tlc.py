"""TLC — test-time local converter (Chu et al., ECCV 2022, "Improving Image Restoration by Revisiting Global
Information Aggregation") for Restormer's channel attention (MDTA). Baseline A4 in docs/baselines.md.

MDTA computes a C x C attention matrix from *all* pixels of its input. Trained on 128-384 px crops, at test time on
512-1024 px tiles (or whole images) these statistics come from a much larger area than in training. TLC computes them
over local windows instead: each attention layer runs on overlapping windows of size `k` (k = `tlc_base` input px
scaled to the layer's resolution; the official recommendation is 1.5x the largest training crop, Restormer
DPDD: 1.5 * 384 = 576), outputs averaged over the overlaps (stride k/2, last window shifted inwards — the "grids"
implementation of the official TLC code). Layers whose feature map fits in one window are unchanged.
"""
from __future__ import annotations

import types

import torch


def _starts(n: int, k: int) -> list[int]:
    if n <= k:
        return [0]
    s = list(range(0, n - k, max(k // 2, 1)))
    return s + [n - k]


def _local_forward(self, x):
    h, w = x.shape[-2:]
    k = max(int(round(self._tlc_base * h / self._tlc_ref[0])), 8)
    if h <= k and w <= k:
        return self._global_forward(x)
    out = torch.zeros_like(x)
    cnt = torch.zeros((1, 1, h, w), device=x.device, dtype=x.dtype)
    for y0 in _starts(h, k):
        for x0 in _starts(w, k):
            out[..., y0:y0 + k, x0:x0 + k] += self._global_forward(x[..., y0:y0 + k, x0:x0 + k])
            cnt[..., y0:y0 + k, x0:x0 + k] += 1
    return out / cnt


def wrap(net: torch.nn.Module, spec: dict):
    """Patch every MDTA `Attention` module; the window is relative to the network input size of each call."""
    base = float(spec.get("tlc_base", 576))
    ref = [1, 1]
    mods = [m for m in net.modules() if type(m).__name__ == "Attention" and hasattr(m, "temperature")]
    for m in mods:
        m._global_forward = m.forward
        m._tlc_base, m._tlc_ref = base, ref
        m.forward = types.MethodType(_local_forward, m)
    from uhdd.models import _wrap
    inner = _wrap(net, spec)

    def fn(x: torch.Tensor) -> torch.Tensor:
        ref[0], ref[1] = x.shape[-2:]
        return inner(x)

    fn.n_patched = len(mods)
    return fn
