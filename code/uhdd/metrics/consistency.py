"""Cross-patch consistency metrics (taxonomy in Template/notes.txt).

1. Low-frequency drift: tone/color error that varies across the image.
2. Seams: step discontinuities of the error along a method's own tile boundaries.
3. Semantic/texture consistency: TODO (DINOv2-matched patch pairs), see plan/evaluation.md 4.3.
4. Statistical: per-cell sharpness / noise / spectral slope compared with the target; the
   *spread* of the log-ratio across cells measures inconsistency (a uniform bias is not).

All functions take 1x3xHxW tensors in [0,1] and return plain floats (8-bit units where noted).
With gt=None, seam metrics run on the prediction alone (no-reference variant).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_LUMA = (0.299, 0.587, 0.114)


def luma(x: torch.Tensor) -> torch.Tensor:
    w = torch.tensor(_LUMA, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * w).sum(1, keepdim=True)


# ---------------------------------------------------------------- 1. low-frequency drift
def lf_drift(pred: torch.Tensor, gt: torch.Tensor, cell: int = 256) -> float:
    """Std over cells of the per-cell mean error, averaged over channels, in 8-bit units."""
    m = F.avg_pool2d(pred - gt, cell)                      # 1x3xnhxnw (remainder dropped)
    return (m.flatten(2).std(dim=2).mean() * 255).item()


# ---------------------------------------------------------------- 2. seams
def seam_lines(info: dict, crop: int = 0) -> tuple[list[int], list[int]]:
    """Overlap-center lines (output pixel coords after cropping `crop`) from a tiling info dict."""
    if info.get("mode") != "tiled":
        return [], []
    s, T = info["scale"], info["tile"]

    def centers(pos: list[int]) -> list[int]:
        return [int(round((b + a + T) / 2 * s)) - crop for a, b in zip(pos[:-1], pos[1:])]

    return centers(info["ys"]), centers(info["xs"])


def _step_at(e: torch.Tensor, cols: list[int], w: int, block: int) -> torch.Tensor | None:
    """|mean over w px left - mean over w px right| of e (1x1xHxW) at columns, averaged
    over `block`-row segments (block averaging suppresses texture, keeps low-freq steps)."""
    W = e.shape[-1]
    cols = [c for c in cols if w <= c <= W - w]
    if not cols:
        return None
    left = torch.stack([e[..., c - w:c].mean(-1) for c in cols], -1)   # 1x1xHxN
    right = torch.stack([e[..., c:c + w].mean(-1) for c in cols], -1)
    d = F.avg_pool2d(left - right, (min(block, e.shape[-2]), 1))       # per row block
    return d.abs().mean()


def seam_metrics(pred: torch.Tensor, gt: torch.Tensor | None, info: dict, crop: int = 0,
                 w: int = 16, block: int = 64) -> dict[str, float]:
    """seam_step: mean error step across seams (8-bit units);
    seam_ratio: seam_step / the same statistic at tile centers (1 = no seam)."""
    ys, xs = seam_lines(info, crop)
    if not ys and not xs:
        return {"seam_step": float("nan"), "seam_ratio": float("nan")}
    e = luma(pred - gt) if gt is not None else luma(pred)
    s, T = info["scale"], info["tile"]
    ctrl_y = [int(round((a + T / 2) * s)) - crop for a in info["ys"]]
    ctrl_x = [int(round((a + T / 2) * s)) - crop for a in info["xs"]]

    def both(rows, cols):
        vals = [v for v in (_step_at(e, cols, w, block),
                            _step_at(e.transpose(-1, -2), rows, w, block)) if v is not None]
        return torch.stack(vals).mean() if vals else None

    seam, ctrl = both(ys, xs), both(ctrl_y, ctrl_x)
    if seam is None:
        return {"seam_step": float("nan"), "seam_ratio": float("nan")}
    ratio = (seam / ctrl).item() if ctrl is not None and ctrl > 0 else float("nan")
    return {"seam_step": seam.item() * 255, "seam_ratio": ratio}


# ---------------------------------------------------------------- 4. statistical maps
def _cells(y: torch.Tensor, cell: int) -> torch.Tensor:
    """1x1xHxW -> Nxcellxcell non-overlapping cells (remainder dropped)."""
    H, W = y.shape[-2:]
    y = y[..., : H // cell * cell, : W // cell * cell]
    return y.unfold(2, cell, cell).unfold(3, cell, cell).reshape(-1, cell, cell)


_LAP = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
_NOISE = torch.tensor([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=torch.float32)


def cell_stats(x: torch.Tensor, cell: int = 256, band: tuple[float, float] = (0.05, 0.45)) -> dict:
    """Per-cell sharpness (Laplacian variance), noise sigma (Immerkaer 1996), spectral slope."""
    c = _cells(luma(x), cell)[:, None]                                  # Nx1xcxc
    dev = c.device
    lap = F.conv2d(c, _LAP.to(dev).view(1, 1, 3, 3))
    sharp = lap.flatten(1).var(dim=1)
    nz = F.conv2d(c, _NOISE.to(dev).view(1, 1, 3, 3)).abs().flatten(1).mean(1)
    noise = nz * math.sqrt(math.pi / 2) / 6

    # radially averaged power spectrum (Hann window), log-log slope in `band` (cycles/px)
    hann = torch.hann_window(cell, periodic=False, device=dev)
    win = hann[:, None] * hann[None, :]
    z = (c[:, 0] - c[:, 0].mean((1, 2), keepdim=True)) * win
    P = torch.fft.rfft2(z).abs().pow(2)                                 # N x c x (c/2+1)
    fy = torch.fft.fftfreq(cell, device=dev)[:, None]
    fx = torch.fft.rfftfreq(cell, device=dev)[None, :]
    fr = torch.sqrt(fy ** 2 + fx ** 2)
    nb = 24
    edges = torch.logspace(math.log10(band[0]), math.log10(band[1]), nb + 1, device=dev)
    idx = torch.bucketize(fr.flatten(), edges) - 1                      # -1 / nb = outside
    ok = (idx >= 0) & (idx < nb)
    idx_ok = idx[ok]
    Pf = P.flatten(1)[:, ok]
    cnt = torch.zeros(nb, device=dev).index_add_(0, idx_ok, torch.ones_like(idx_ok, dtype=torch.float32))
    sums = torch.zeros(P.shape[0], nb, device=dev).index_add_(1, idx_ok, Pf)
    logP = torch.log(sums / cnt.clamp(min=1) + 1e-20)
    logf = torch.log(torch.sqrt(edges[:-1] * edges[1:]))
    lf = logf - logf.mean()
    slope = ((logP - logP.mean(1, keepdim=True)) * lf).sum(1) / (lf * lf).sum()
    return {"sharp": sharp, "noise": noise, "slope": slope}


def stat_consistency(pred: torch.Tensor, gt: torch.Tensor, cell: int = 256) -> dict[str, float]:
    """Mean |log ratio| (bias+inconsistency) and std of log ratio (inconsistency only)."""
    sp, sg = cell_stats(pred, cell), cell_stats(gt, cell)
    out = {}
    for k in ("sharp", "noise"):
        eps = 1e-3 * sg[k].median().clamp(min=1e-12)
        lr = torch.log((sp[k] + eps) / (sg[k] + eps))
        out[f"{k}_lr_abs"], out[f"{k}_lr_std"] = lr.abs().mean().item(), lr.std().item()
    d = sp["slope"] - sg["slope"]
    out["slope_err_abs"], out["slope_err_std"] = d.abs().mean().item(), d.std().item()
    return out
