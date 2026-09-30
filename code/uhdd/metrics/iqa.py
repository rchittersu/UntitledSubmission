"""Learned perceptual / no-reference metrics via pyiqa, evaluated tile-wise.

Full 30 MP images do not fit these networks. LPIPS/DISTS are spatial averages of feature
distances, so the area-weighted mean over non-overlapping tiles approximates the full-image
value (up to tile-border effects); tile size is recorded in the results.
No-reference models are evaluated on a fixed, evenly spaced grid of crops.
"""
from __future__ import annotations

import torch

_FR = {"lpips", "dists"}                                   # pyiqa metric names
_NR = {"musiq", "maniqa", "clipiqa"}
_cache: dict[tuple[str, str], torch.nn.Module] = {}


def _metric(name: str, device) -> torch.nn.Module:
    key = (name, str(device))
    if key not in _cache:
        import pyiqa  # imported lazily: heavy, and optional for the rest of the suite
        _cache[key] = pyiqa.create_metric(name, device=device, as_loss=False)
    return _cache[key]


def _tiles(H: int, W: int, t: int) -> list[tuple[int, int, int, int]]:
    return [(y, x, min(t, H - y), min(t, W - x)) for y in range(0, H, t) for x in range(0, W, t)]


@torch.inference_mode()
def full_reference(name: str, pred: torch.Tensor, gt: torch.Tensor, tile: int = 1024,
                   batch: int = 8, min_side: int = 64) -> float:
    assert name in _FR, name
    m = _metric(name, pred.device)
    H, W = pred.shape[-2:]
    boxes = [b for b in _tiles(H, W, tile) if min(b[2], b[3]) >= min_side]
    # full-size tiles are batched; ragged border tiles go one by one
    tot, area = 0.0, 0
    full = [b for b in boxes if b[2] == tile and b[3] == tile]
    rest = [b for b in boxes if b not in full]
    for i in range(0, len(full), batch):
        ch = full[i:i + batch]
        p = torch.cat([pred[..., y:y + h, x:x + w] for y, x, h, w in ch])
        g = torch.cat([gt[..., y:y + h, x:x + w] for y, x, h, w in ch])
        tot += m(p, g).flatten().sum().item() * tile * tile
        area += len(ch) * tile * tile
    for y, x, h, w in rest:
        tot += m(pred[..., y:y + h, x:x + w], gt[..., y:y + h, x:x + w]).item() * h * w
        area += h * w
    return tot / max(area, 1)


@torch.inference_mode()
def no_reference(name: str, img: torch.Tensor, crop: int = 512, n: int = 16, batch: int = 8) -> float:
    assert name in _NR, name
    m = _metric(name, img.device)
    H, W = img.shape[-2:]
    crop = min(crop, H, W)
    k = max(1, int(round(n ** 0.5)))
    ys = torch.linspace(0, H - crop, k).round().int().tolist()
    xs = torch.linspace(0, W - crop, k).round().int().tolist()
    crops = [img[..., y:y + crop, x:x + crop] for y in ys for x in xs]
    vals = [m(torch.cat(crops[i:i + batch])).flatten() for i in range(0, len(crops), batch)]
    return torch.cat(vals).mean().item()
