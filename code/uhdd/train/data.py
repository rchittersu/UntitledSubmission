"""Training / validation tiles from the cache of scripts/build_train_cache.py (plan/method_plan.md §4).

A sample is one 512-px native tile (positions are multiples of 4, so its anchor tile is exact):
  a (3,T,T) anchor tile -- a real anchor (cache anchor_<model>.npy) or a simulated one (sim_anchor), per `anchors`
  x (3,4T,4T) native input, y (3,4T,4T) target, m (1,4T,4T) validity mask,
  d (2,T,T) DP input: lightly smoothed |disparity| + confidence (cache dp.npy)
  ex (M,3,P,P) exemplar crops of the native input, tok_map (T,T), tok_scores (n_tok_max, M) -- see memory.tile_exemplars;
  exemplars overlapping the tile are dropped (training: transfer, not copy in place)
  regime (1,4T,4T) defocus weight for masked losses (1 where blur >= defocus_thr)
Augmentation: the same random flip / transpose on every spatial tensor and on the exemplar crops.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import memory as mem_mod
from .sim_anchor import simulate

def tile_positions(blur: np.ndarray, tile: int, stride: int = 64, base: float = 0.25, thr: float = 0.4):
    """Top-left corners (native px, multiples of 4) of all tiles of size `tile` on a `stride` grid, and sampling
    weights = base + fraction of defocused pixels (blur >= thr) in the tile."""
    H4, W4 = blur.shape
    t4, s4 = tile // 4, max(1, stride // 4)
    ys = np.arange(0, H4 - t4 + 1, s4)
    xs = np.arange(0, W4 - t4 + 1, s4)
    ii = np.pad((blur >= thr).astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    frac = (ii[Y + t4, X + t4] - ii[Y, X + t4] - ii[Y + t4, X] + ii[Y, X]) / (t4 * t4)
    return np.stack([4 * Y.ravel(), 4 * X.ravel()], 1).astype(np.int32), (base + frac.ravel()).astype(np.float32)


TOK_MAX = 24          # max query tokens per tile side (tile 1024 native = 256 anchor px / 14 -> 19; padded)


class Scene:
    def __init__(self, d: Path, anchors: list[str]):
        self.dir = d
        self.x = np.load(d / "x.npy", mmap_mode="r")
        self.y = np.load(d / "y.npy", mmap_mode="r")
        self.mask = np.load(d / "mask.npy", mmap_mode="r")
        self.blur = np.load(d / "blur.npy").astype(np.float32)
        self.dp = np.load(d / "dp.npy").astype(np.float32)
        self.anchors = {m: np.load(d / f"anchor_{m}.npy", mmap_mode="r") for m in anchors if (d / f"anchor_{m}.npy").exists()}
        z = np.load(d / "mem.npz")
        self.key_box, self.grid, self.tok_native = z["key_box"], tuple(z["grid"]), int(z["tok_native"])
        self.idx, self.score = z["idx"], z["score"]
        self.oidx, self.oscore = z["oracle_idx"], z["oracle_score"]
        self.maxv = float(np.iinfo(self.x.dtype).max)


def _t(a: np.ndarray, maxv: float) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a).astype(np.float32) / maxv).permute(2, 0, 1)


class TileDataset(torch.utils.data.IterableDataset):
    """cfg keys: cache (dir), tile (512), anchors {model: prob, "sim": prob}, sim (sim_anchor ranges), m (exemplars per
    tile), k (per token), oracle (bool), drop_colocated (prob), ex_px (64), defocus_thr (0.4), augment (bool),
    names (optional list), seed. Validation: fixed=N gives N deterministic tiles (no augmentation, first anchor)."""

    def __init__(self, cfg: dict, rank: int = 0, world: int = 1, fixed: int = 0):
        super().__init__()
        self.cfg, self.rank, self.world, self.fixed = cfg, rank, world, fixed
        root = Path(cfg["cache"])
        names = cfg.get("names") or json.loads((root / "manifest.json").read_text()).get("names") or \
            sorted(p.parent.name for p in root.glob("*/meta.json"))
        self.names = sorted(names)
        self.anchor_probs = {k: float(v) for k, v in cfg.get("anchors", {"bokehlicious_deblur": 1.0}).items()}
        self.scenes: dict[str, Scene] = {}

    def scene(self, n: str) -> Scene:
        if n not in self.scenes:
            s = Scene(Path(self.cfg["cache"]) / n, [k for k in self.anchor_probs if k != "sim"])
            s.pos, s.w = tile_positions(s.blur, self.cfg.get("tile", 512), self.cfg.get("stride", 64),
                                        self.cfg.get("tile_base_weight", 0.25), self.cfg.get("defocus_thr", 0.4))
            if len(s.pos) == 0:
                raise ValueError(f"{n}: smaller than one {self.cfg.get('tile', 512)}-px tile")
            self.scenes[n] = s
        return self.scenes[n]

    # ------------------------------------------------------------------ one sample
    def sample(self, n: str, pos: tuple[int, int], anchor: str, g: torch.Generator, aug: bool) -> dict:
        c = self.cfg
        s = self.scene(n)
        T = c.get("tile", 512)
        T4 = T // 4
        y0, x0 = pos
        y4, x4 = y0 // 4, x0 // 4
        x = _t(s.x[y0:y0 + T, x0:x0 + T], s.maxv)
        y = _t(s.y[y0:y0 + T, x0:x0 + T], s.maxv)
        m = torch.from_numpy(np.ascontiguousarray(s.mask[y0:y0 + T, x0:x0 + T])).float()[None]
        d = torch.from_numpy(s.dp[:, y4:y4 + T4, x4:x4 + T4].copy())
        bl = torch.from_numpy(s.blur[y4:y4 + T4, x4:x4 + T4].copy())[None]
        if anchor == "sim":
            mg = 4                                                   # anchor-px margin for the blur
            ys, xs = max(0, y4 - mg), max(0, x4 - mg)
            ye, xe = min(s.blur.shape[0], y4 + T4 + mg), min(s.blur.shape[1], x4 + T4 + mg)
            yy = _t(s.y[4 * ys:4 * ye, 4 * xs:4 * xe], s.maxv)[None]
            a = simulate(F.avg_pool2d(yy, 4), torch.from_numpy(s.blur[ys:ye, xs:xe].copy())[None, None], g, c.get("sim"))[0]
            a = a[:, y4 - ys:y4 - ys + T4, x4 - xs:x4 - xs + T4]
        else:
            a = _t(s.anchors[anchor][y4:y4 + T4, x4:x4 + T4], float(np.iinfo(s.anchors[anchor].dtype).max))
        idx, sc = (s.oidx, s.oscore) if c.get("oracle") else (s.idx, s.score)
        drop = (y0, x0, T, T) if torch.rand((), generator=g).item() < c.get("drop_colocated", 1.0) else None
        keys, tmap, ts = mem_mod.tile_exemplars(idx, sc, s.grid, s.tok_native // 4, s.key_box, (y4, x4, T4),
                                                c.get("m", 32), c.get("k", 16), drop)
        P = c.get("ex_px", 64)
        valid = keys >= 0
        ex = torch.zeros(len(keys), 3, P, P)
        if valid.any():
            boxes = s.key_box[keys[valid]]
            crops = [torch.from_numpy(np.ascontiguousarray(s.x[b0:b0 + bs, b1:b1 + bs]).astype(np.float32) / s.maxv)
                     .permute(2, 0, 1)[None] for b0, b1, bs in boxes]
            ex[torch.from_numpy(valid)] = torch.cat([cr if cr.shape[-1] == P else
                                                     F.interpolate(cr, size=(P, P), mode="area" if cr.shape[-1] > P else "bicubic")
                                                     for cr in crops])
        tsp = np.full((TOK_MAX * TOK_MAX, len(keys)), -np.inf, np.float32)
        tsp[:ts.shape[0]] = ts
        r = (F.interpolate(bl[None], scale_factor=4, mode="nearest")[0] >= c.get("defocus_thr", 0.4)).float()
        out = {"a": a, "x": x, "y": y, "m": m, "d": d, "ex": ex, "tok_map": torch.from_numpy(tmap),
               "tok_scores": torch.from_numpy(tsp), "regime": r}
        if aug:
            fl = torch.rand(3, generator=g).tolist()
            for k in ("a", "x", "y", "m", "d", "regime", "ex"):
                t = out[k]
                if fl[0] < 0.5:
                    t = t.flip(-1)
                if fl[1] < 0.5:
                    t = t.flip(-2)
                if fl[2] < 0.5:
                    t = t.transpose(-1, -2)
                out[k] = t.contiguous()
            tm = out["tok_map"]
            if fl[0] < 0.5:
                tm = tm.flip(-1)
            if fl[1] < 0.5:
                tm = tm.flip(-2)
            if fl[2] < 0.5:
                tm = tm.transpose(-1, -2)
            out["tok_map"] = tm.contiguous()
        return out

    # ------------------------------------------------------------------ iteration
    def fixed_tiles(self) -> list[tuple[str, tuple[int, int]]]:
        g = np.random.default_rng(self.cfg.get("seed", 0))
        out = []
        for i in range(self.fixed):
            n = self.names[i % len(self.names)]
            s = self.scene(n)
            j = g.choice(len(s.pos), p=s.w / s.w.sum())
            out.append((n, tuple(int(v) for v in s.pos[j])))
        return out[self.rank::self.world]

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        if self.fixed:
            first = next(k for k in self.anchor_probs if k != "sim") if any(k != "sim" for k in self.anchor_probs) else "sim"
            g = torch.Generator().manual_seed(1234)
            for i, (n, p) in enumerate(self.fixed_tiles()):
                if i % nw == wid:
                    yield {**self.sample(n, p, first, g, False), "name": n}
            return
        g = torch.Generator().manual_seed(self.cfg.get("seed", 0) * 100003 + self.rank * 1009 + wid)
        rng = np.random.default_rng(self.cfg.get("seed", 0) * 100003 + self.rank * 1009 + wid)
        an, ap = list(self.anchor_probs), np.array(list(self.anchor_probs.values()))
        while True:
            n = self.names[rng.integers(len(self.names))]
            s = self.scene(n)
            j = rng.choice(len(s.pos), p=s.w / s.w.sum())
            anchor = an[rng.choice(len(an), p=ap / ap.sum())]
            if anchor != "sim" and anchor not in s.anchors:
                anchor = "sim"
            yield self.sample(n, tuple(int(v) for v in s.pos[j]), anchor, g, self.cfg.get("augment", True))
