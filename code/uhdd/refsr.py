"""Reference-based x4 upsampling on tiles: same-image references and the reference-model interface.

A reference model maps one anchor tile (x4, 1x3xhxw in [0, 1]) and one reference image (1x3xRxR in [0, 1], native
pixels, or None) to the native tile (1x3x4hx4w in [0, 1]). Registry: code/configs/refsr.yaml (factory per model).

Reference modes (same image only; built by `make_refs`):
  none       no reference (the model's own SR prior)
  self       the native blurry input at the tile's own location: the copy path (exact answer in focus, blurred
             detail when defocused)
  retrieved  a mosaic of the in-focus native crops retrieved for the tile's anchor tokens (uhdd/train/memory.py,
             DINOv2 on the anchor, keys in focus only, crops overlapping the tile removed), best score top-left
"""
from __future__ import annotations

import importlib
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REF_MODES = ("none", "self", "retrieved")
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "refsr.yaml"


# ---------------------------------------------------------------- references
def self_ref(native: torch.Tensor, box: tuple[int, int, int]) -> torch.Tensor:
    """Native input crop at the tile (y, x, size) in native px."""
    y, x, s = box
    return native[..., y:y + s, x:x + s]


def retrieved_ref(native: torch.Tensor, mem, idx: np.ndarray, sc: np.ndarray, tile4: tuple[int, int, int],
                  size: int = 512, grid: int = 4, k: int = 16) -> tuple[torch.Tensor, dict]:
    """Mosaic (size x size) of the grid*grid best in-focus exemplars of the tile (anchor box tile4 = (y4, x4, T4)).
    Exemplar crops (native, value_px / scale) are resized to size / grid; empty cells stay at the tile's mean grey.
    Returns the mosaic and {keys, scores, boxes} for the report."""
    from uhdd.train import memory as mm
    y4, x4, T4 = tile4
    keys, _, ts = mm.tile_exemplars(idx, sc, mem.grid, mem.tok_native // 4, mem.key_box, tile4, m=grid * grid, k=k,
                                    drop=(4 * y4, 4 * x4, 4 * T4, 4 * T4))
    valid = keys >= 0
    cell = size // grid
    out = torch.full((1, 3, size, size), 0.5, device=native.device, dtype=native.dtype)
    if valid.any():
        crops = mm.crops(native, mem.key_box[keys[valid]], cell).to(native.dtype)
        for i, c in enumerate(crops):
            r, q = divmod(i, grid)
            out[..., r * cell:(r + 1) * cell, q * cell:(q + 1) * cell] = c
    best = np.where(np.isfinite(ts), ts, -np.inf).max(0) if ts.size else np.zeros(0)
    info = {"keys": [int(k_) for k_ in keys[valid]], "scores": [round(float(s), 4) for s in best[:int(valid.sum())]],
            "boxes": mem.key_box[keys[valid]].tolist() if valid.any() else []}
    return out, info


# ---------------------------------------------------------------- models
def _expand(v):
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), v)
    return v


def specs(path: str | Path = CONFIG) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text())["models"]


def load(name: str, device="cuda", overrides: dict | None = None, path: str | Path = CONFIG):
    """Build reference model `name` from the registry; returns callable(anchor, ref) -> native tile."""
    spec = _expand({**specs(path)[name], **(overrides or {})})
    mod, fn = spec["factory"].split(":")
    return getattr(importlib.import_module(mod), fn)(spec, device)


class Bicubic:
    """Reference-free floor: bicubic x4 of the anchor."""

    def __init__(self, spec=None, device="cpu"):
        pass

    def __call__(self, anchor: torch.Tensor, ref: torch.Tensor | None) -> torch.Tensor:
        return F.interpolate(anchor.float(), scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)


class HighBand(Bicubic):
    """Non-learned transfer: bicubic + the reference's high band (above the anchor band), energy-matched to the
    bicubic tile's own detail level. Only meaningful for aligned references (`self`): with `self` in focus it copies
    the observation's detail; defocused, it adds what little the blurred input still has."""

    def __init__(self, spec=None, device="cpu"):
        self.gain = float((spec or {}).get("gain", 1.0))

    def __call__(self, anchor, ref):
        up = super().__call__(anchor, ref)
        if ref is None or ref.shape[-2:] != up.shape[-2:]:
            return up
        r = ref.float().to(up.device)
        low = F.interpolate(F.interpolate(r, scale_factor=0.25, mode="area"), scale_factor=4, mode="bicubic",
                            align_corners=False)
        return (up + self.gain * (r - low)).clamp(0, 1)


def build_bicubic(spec, device):
    return Bicubic(spec, device)


def build_highband(spec, device):
    return HighBand(spec, device)
