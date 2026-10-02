"""Blur-stratified and alignment-tolerant metrics (docs/evaluation.md, sections 3.4-3.7).

percbins  LPIPS / DISTS per DP blur bin. Perceptual networks need whole tiles, so the image is cut into
          non-overlapping `tile`-px tiles; a tile counts if >= 90 % of it is valid (mask) and is assigned
          to the bin of its *median* blur level. Columns lpips_b{k}, dists_b{k}, ntile_b{k}.
noharm    How much a method changes the in-focus region, where the input already is the answer:
          keep_psnr_b0 = PSNR(pred, input) on focal-plane pixels (higher = less change, GT-free) and
          dpsnr_b0 = PSNR_b0(pred, gt) - PSNR_b0(input, gt) (< 0 = the method damaged in-focus detail).
apsnr     Tile-aligned PSNR: per valid tile, the integer shift within +-radius px that minimises the error
          is applied before scoring (residual misalignment of the translation-registered target is ~1-2
          native px). Columns apsnr, apsnr_shift (mean |shift| in px over tiles).
msres     Resolution sweep: pred, gt (area) and mask (min) downscaled by each factor f in ctx["eval_scales"];
          psnr_s{f}, lpips_s{f}, dists_s{f}. Shows at which resolution a method's gains live.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from . import fidelity, iqa

EDGES = (0.4, 1.5, 3.5)


def bin_edges(ctx: dict) -> tuple[float, ...]:
    return (0.0, *ctx.get("blur_edges", EDGES), float("inf"))


def valid_tiles(H: int, W: int, tile: int, mask: torch.Tensor | None, min_valid: float = 0.9):
    out = []
    for y in range(0, H - tile + 1, tile):
        for x in range(0, W - tile + 1, tile):
            if mask is None or mask[..., y:y + tile, x:x + tile].float().mean().item() >= min_valid:
                out.append((y, x))
    return out


@torch.inference_mode()
def perceptual_bins(pred, gt, mask, blur_map, ctx: dict, names=("lpips", "dists")) -> dict[str, float]:
    tile = ctx.get("bin_tile", 512)
    edges = bin_edges(ctx)
    groups: dict[int, list] = {}
    for y, x in valid_tiles(*pred.shape[-2:], tile, mask):
        med = blur_map[..., y:y + tile, x:x + tile].median().item()
        k = max(i for i in range(len(edges) - 1) if med >= edges[i])
        groups.setdefault(k, []).append((y, x))
    out = {}
    for k in range(len(edges) - 1):
        pos = groups.get(k, [])
        out[f"ntile_b{k}"] = float(len(pos))
        for n in names:
            if not pos:
                out[f"{n}_b{k}"] = float("nan")
                continue
            m = iqa._metric(n, pred.device)
            vals = []
            for i in range(0, len(pos), 8):
                ch = pos[i:i + 8]
                p = torch.cat([pred[..., y:y + tile, x:x + tile] for y, x in ch])
                g = torch.cat([gt[..., y:y + tile, x:x + tile] for y, x in ch])
                vals.append(m(p, g).flatten().float().cpu())
            out[f"{n}_b{k}"] = torch.cat(vals).mean().item()
    return out


def no_harm(pred, gt, inp, mask, blur_map, ctx: dict) -> dict[str, float]:
    m0 = blur_map < bin_edges(ctx)[1]
    if mask is not None:
        m0 = m0 & mask
    if not m0.any():
        return {"keep_psnr_b0": float("nan"), "dpsnr_b0": float("nan")}
    out = {"keep_psnr_b0": fidelity.psnr(pred, inp, m0)}
    if gt is not None:
        out["dpsnr_b0"] = fidelity.psnr(pred, gt, m0) - fidelity.psnr(inp, gt, m0)
    return out


@torch.inference_mode()
def aligned_psnr(pred, gt, mask, ctx: dict) -> dict[str, float]:
    tile, r = ctx.get("align_tile", 512), ctx.get("align_radius", 2)
    H, W = pred.shape[-2:]
    se, n, shifts = 0.0, 0, []
    for y, x in valid_tiles(H, W, tile, mask):
        y0, x0, y1, x1 = max(y, r), max(x, r), min(y + tile, H - r), min(x + tile, W - r)
        g = gt[..., y0:y1, x0:x1].float()
        mm = None if mask is None else mask[..., y0:y1, x0:x1]
        best = None
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                d = (pred[..., y0 + dy:y1 + dy, x0 + dx:x1 + dx].float() - g).pow(2)
                e = d.sum(1, keepdim=True)[mm].sum().item() if mm is not None else d.sum().item()
                if best is None or e < best[0]:
                    best = (e, abs(dy) + abs(dx))
        cnt = (int(mm.sum().item()) if mm is not None else (y1 - y0) * (x1 - x0)) * pred.shape[1]
        se, n = se + best[0], n + cnt
        shifts.append(best[1])
    if not n:
        return {"apsnr": float("nan"), "apsnr_shift": float("nan")}
    return {"apsnr": 10 * math.log10(n / max(se, 1e-30)), "apsnr_shift": sum(shifts) / len(shifts)}


def _down(x: torch.Tensor, f: int) -> torch.Tensor:
    H, W = x.shape[-2:]
    return F.avg_pool2d(x[..., : H // f * f, : W // f * f].float(), f)


@torch.inference_mode()
def resolution_sweep(pred, gt, mask, ctx: dict) -> dict[str, float]:
    out = {}
    for f in ctx.get("eval_scales", (2, 4)):
        p, g = _down(pred, f), _down(gt, f)
        m = None if mask is None else _down(mask, f) > 0.999          # all source pixels valid
        out[f"psnr_s{f}"] = fidelity.psnr(p, g, m)
        for n in ("lpips", "dists"):
            out[f"{n}_s{f}"] = iqa.full_reference(n, p, g, tile=ctx.get("fr_tile", 1024))
    return out
