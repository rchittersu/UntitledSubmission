"""Bokehlicious (Seizinger et al., ICCV 2025) in its RealDefocus defocus-deblurring variant.

Repo: github.com/TimSeizinger/Bokehlicious, checkpoint checkpoints/defocus_deblur.zpaq
(unpack with `zpaq x`), config `bokehlicious_size_builder("defocus_deblur")`.

Inputs besides the image (dataset/util.py, dataset/loader.py with defocus_deblur_mode=True):
- pos_map: 2 x H x W normalized coordinates of the *whole image* (longer side spans [0, 1],
  shorter side centered). For tiles we evaluate the same linear map at the tile's pixels,
  so tiled inference sees exactly the coordinates the whole-image pass would.
- bokeh_strength = 2.0 / f-number of the *blurry input* (controls the attention range);
  DPDD inputs are f/4 -> 0.5. Set `av` in the model spec.
Trained on RealDefocus (Sony, 2000x1500 training resolution), not on DPDD.
"""
from __future__ import annotations

import torch


def build(spec: dict) -> torch.nn.Module:
    from method.config import bokehlicious_size_builder  # repo is on sys.path
    from method.model import Bokehlicious
    return Bokehlicious(**bokehlicious_size_builder(spec.get("size", "defocus_deblur")))


def pos_map(full_h: int, full_w: int, y0: int, x0: int, h: int, w: int, device) -> torch.Tensor:
    """2 x h x w crop of get_pos_map(full_w, full_h) (linspace-based, extrapolated past edges)."""
    if full_w > full_h:
        c = (1 - full_h / full_w) / 2
        x_a, x_b, y_a, y_b = 0.0, 1.0, 1 - c, c
    elif full_w < full_h:
        c = (1 - full_w / full_h) / 2
        x_a, x_b, y_a, y_b = c, 1 - c, 1.0, 0.0
    else:
        x_a, x_b, y_a, y_b = 0.0, 1.0, 1.0, 0.0
    lin = lambda a, b, n_full, start, n: a + (b - a) * (
        torch.arange(start, start + n, device=device, dtype=torch.float32) / max(n_full - 1, 1))
    xs, ys = lin(x_a, x_b, full_w, x0, w), lin(y_a, y_b, full_h, y0, h)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy])


def wrap(net: torch.nn.Module, spec: dict):
    strength = 2.0 / float(spec.get("av", 4.0))

    def fn(x: torch.Tensor, boxes, full_hw) -> torch.Tensor:
        pm = torch.stack([pos_map(*full_hw, y, xx, h, w, x.device) for y, xx, h, w in boxes])
        b = x.shape[0]
        s = torch.full((b,), strength, device=x.device, dtype=torch.float32)
        smap = torch.full((b, 1, *x.shape[-2:]), strength, device=x.device, dtype=x.dtype)
        y = net(source=x, bokeh_strength=s, pos_map=pm.to(x.dtype), bokeh_strength_map=smap)
        return y.clamp(0, 1)

    fn.positional = True
    return fn
