"""Tiled (patch-wise) inference with overlap blending, fully on the GPU.

Design choices:
- All tiles have the same size T (a multiple of the model's size constraint), so tiles can be
  batched and cuDNN autotuning (`cudnn.benchmark`) pays off. The last tile in each row/column
  is shifted inwards instead of being padded; the image is only padded when it is smaller than
  one tile.
- Blending weights are separable windows that ramp only on sides shared with another tile;
  accumulation is in float32 regardless of the model precision.
- Supports models with an integer output scale s (super-resolution): output tiles are T*s.
- The tile grid is returned so seam metrics can be evaluated on each method's own grid.
- Models whose input depends on absolute image position (e.g. Bokehlicious' position maps)
  set `fn.positional = True` and are called as fn(tiles, boxes=[(y, x, h, w), ...],
  full_hw=(H, W)), with boxes in image coordinates (may extend past H/W into padding).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import torch

from .padding import crop, pad_to, pad_to_multiple, round_up


@dataclass
class TileSpec:
    tile: int = 0            # tile size in input pixels; 0 = whole image in one pass
    overlap: int = 64        # overlap between neighboring tiles (input pixels)
    batch: int = 4           # tiles per forward pass
    blend: str = "linear"    # linear | gaussian | hard | mean
    offset: int = 0          # shift interior tile starts by this many px (grid-shift consistency test)


def grid_positions(length: int, tile: int, overlap: int, offset: int = 0) -> list[int]:
    """Tile start positions covering [0, length) with at least `overlap` overlap.

    offset > 0 shifts every interior tile start by -offset (the first tile stays at 0 and the
    last at length - tile, so their overlaps grow): two runs with offsets 0 and ~stride/2 have
    disjoint seam lines, which the grid-shift consistency metric exploits.
    """
    if length <= tile:
        return [0]
    stride = tile - overlap
    if stride <= 0:
        raise ValueError(f"overlap {overlap} must be < tile {tile}")
    offset %= stride
    pos = [0] + [p for p in range(stride - offset, length - tile, stride) if p > 0]
    pos.append(length - tile)
    return sorted(set(pos))


def blend_window(size: int, overlap: int, mode: str, device,
                 ramp_lo: bool = True, ramp_hi: bool = True) -> torch.Tensor:
    """1-D blending window of length `size`.

    `ramp_lo` / `ramp_hi`: whether the low / high end borders another tile. Sides on the image
    border are not attenuated, so interior edges can go to exactly zero weight (no leakage of
    tile-edge artifacts) while every pixel keeps a positive total weight.
    """
    i = torch.arange(size, device=device, dtype=torch.float32)
    d_lo, d_hi = i + 0.5, size - i - 0.5                  # distance to each tile edge
    one = torch.ones(size, device=device)
    if mode == "mean" or overlap <= 0:
        return one
    if mode == "linear":
        ramp = lambda d: (d / overlap).clamp(max=1.0)
    elif mode == "hard":  # switch at the overlap center -> maximal seam visibility (for analysis)
        ramp = lambda d: (d >= overlap / 2).float()
    elif mode == "gaussian":
        c = (size - 1) / 2
        return torch.exp(-0.5 * ((i - c) / (0.3 * size)) ** 2).clamp(min=1e-3)
    else:
        raise ValueError(f"unknown blend mode {mode}")
    return torch.minimum(ramp(d_lo) if ramp_lo else one, ramp(d_hi) if ramp_hi else one)


@torch.inference_mode()
def run_tiled(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, spec: TileSpec,
              multiple: int = 1, scale: int = 1) -> tuple[torch.Tensor, dict]:
    """Apply `fn` (NCHW [0,1] -> NCHW [0,1], output `scale`x larger) to 1xCxHxW `x`.

    Returns the float32 output (1 x C x H*scale x W*scale) and a dict describing the grid.
    """
    assert x.shape[0] == 1, "one image at a time; tiles are batched internally"
    H, W = x.shape[-2:]

    positional = getattr(fn, "positional", False)
    prepare = getattr(fn, "prepare", None)
    if prepare is not None:          # per-image state of the model (whole image, before tiling)
        prepare(x)

    def call(t: torch.Tensor, boxes: list[tuple[int, int, int, int]]) -> torch.Tensor:
        return (fn(t, boxes=boxes, full_hw=(H, W)) if positional else fn(t)).float()

    if spec.tile <= 0:
        xp, hw = pad_to_multiple(x, multiple)
        y = call(xp, [(0, 0, *xp.shape[-2:])])
        return crop(y, hw, scale), {"mode": "whole", "padded": list(xp.shape[-2:]), "scale": scale}

    T = round_up(spec.tile, max(multiple, 1))
    O = min(spec.overlap, T - 1)
    xp = pad_to(x, max(H, T), max(W, T))
    Hp, Wp = xp.shape[-2:]
    ys, xs = grid_positions(Hp, T, O, spec.offset), grid_positions(Wp, T, O, spec.offset)
    coords = [(iy, ix) for iy in range(len(ys)) for ix in range(len(xs))]

    Ts = T * scale
    wins: dict[tuple, torch.Tensor] = {}

    def window(iy: int, ix: int) -> torch.Tensor:  # 1x1xTsxTs, depends on border position
        key = (iy > 0, iy < len(ys) - 1, ix > 0, ix < len(xs) - 1)
        if key not in wins:
            wy = blend_window(Ts, O * scale, spec.blend, x.device, key[0], key[1])
            wx = blend_window(Ts, O * scale, spec.blend, x.device, key[2], key[3])
            wins[key] = (wy[:, None] * wx[None, :])[None, None]
        return wins[key]

    out = torch.zeros(1, x.shape[1], Hp * scale, Wp * scale, device=x.device, dtype=torch.float32)
    wsum = torch.zeros(1, 1, Hp * scale, Wp * scale, device=x.device, dtype=torch.float32)

    for b in range(0, len(coords), spec.batch):
        chunk = coords[b:b + spec.batch]
        tiles = torch.cat([xp[..., ys[iy]:ys[iy] + T, xs[ix]:xs[ix] + T] for iy, ix in chunk], dim=0)
        pred = call(tiles, [(ys[iy], xs[ix], T, T) for iy, ix in chunk])
        for k, (iy, ix) in enumerate(chunk):
            sy, sx, win = ys[iy] * scale, xs[ix] * scale, window(iy, ix)
            out[..., sy:sy + Ts, sx:sx + Ts].addcmul_(pred[k:k + 1], win)
            wsum[..., sy:sy + Ts, sx:sx + Ts].add_(win)

    out.div_(wsum)
    info = {**asdict(spec), "mode": "tiled", "tile": T, "overlap": O, "ys": ys, "xs": xs,
            "padded": [Hp, Wp], "scale": scale, "n_tiles": len(coords)}
    return crop(out, (H, W), scale), info


OVERLAP_RATIO = 0.125      # standard overlap = tile / 8 for every method (P1: 512 / 64)


def default_tiling(spec: dict, hw: tuple[int, int], overlap_ratio: float | None = None) -> tuple[int, int]:
    """Per-method default (tile, overlap) in input px, from the inference setup of the method's paper/code.

    Registry fields (configs/models.yaml):
      paper_input: [H, W]  the method was evaluated on whole images of this size -> whole image if the input fits
                           (either orientation), else square tiles of the short side (rounded down to `multiple`),
                           i.e. never more pixels per pass than in the paper's own evaluation;
      paper_tile: T        the authors' code tiles with T px (LR px for SR) -> tiles of T, whole if the input fits.
    Neither (builtins such as bicubic) -> whole image. Overlap = round(tile * ratio) (even), ratio from the spec's
    `overlap_ratio`, else `overlap_ratio`, else OVERLAP_RATIO. Returns (0, 0) for whole-image inference.
    """
    m = max(int(spec.get("multiple", 1)), 1)
    h, w = sorted(int(v) for v in hw)
    if spec.get("paper_tile"):
        t = int(spec["paper_tile"]) // m * m
        tile = 0 if w <= t else t
    elif spec.get("paper_input"):
        ph, pw = sorted(int(v) for v in spec["paper_input"])
        tile = 0 if (h <= ph and w <= pw) else ph // m * m
    else:
        tile = 0
    r = spec.get("overlap_ratio", overlap_ratio if overlap_ratio is not None else OVERLAP_RATIO)
    return tile, (0 if tile == 0 else int(round(tile * r / 2)) * 2)
