"""Same-image in-focus exemplar memory (plan/method_plan.md §3; paper Sec. 3.3).

Matching happens in the ANCHOR domain (x4, deblurred everywhere, so a defocused region and its in-focus counterpart
look alike); the CONTENT is native: the value of a key is the native input crop under it. Used identically by the
training cache builder (scripts/build_train_cache.py) and the inference adapter (uhdd/adapters/ours.py).

Geometry: features are computed on the anchor (H4 x W4) at scale s (anchor area-resized by s); a token covers
`tok` px of the resized anchor = tok / s anchor px = 4 tok / s native px. Queries are the tokens of the s = 1 grid.
Boxes are (y0, x0, size) in native px.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class Features:
    """Token features of an image, L2-normalised. kind = 'dinov2' (DINOv2 hub model, offline cache) or 'pixels'
    (normalised 7x7 colour thumbnails of each 14 px cell: the pixel-feature ablation, and a CPU stand-in)."""

    def __init__(self, kind: str = "dinov2", name: str = "dinov2_vitl14", hub_dir: str | None = None,
                 device="cpu", tok: int = 14):
        self.kind, self.device, self.tok = kind, torch.device(device), tok
        if kind == "dinov2":
            from pathlib import Path
            if hub_dir:
                torch.hub.set_dir(str(hub_dir))
                local = Path(hub_dir) / "facebookresearch_dinov2_main"
                self.net = (torch.hub.load(str(local), name, source="local") if local.is_dir()
                            else torch.hub.load("facebookresearch/dinov2", name))
            else:
                self.net = torch.hub.load("facebookresearch/dinov2", name)
            self.net = self.net.to(self.device).eval()
            self.tok = self.net.patch_size
        elif kind != "pixels":
            raise ValueError(kind)

    @torch.no_grad()
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """img 1x3xHxW in [0, 1] -> features [ceil(H/tok), ceil(W/tok), C] (L2-normalised)."""
        t = self.tok
        H, W = img.shape[-2:]
        gh, gw = math.ceil(H / t), math.ceil(W / t)
        x = F.pad(img.to(self.device).float(), (0, gw * t - W, 0, gh * t - H), mode="replicate")
        if self.kind == "dinov2":
            x = (x - torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)) / torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
            f = self.net.forward_features(x)["x_norm_patchtokens"].reshape(gh, gw, -1)
        else:
            f = F.adaptive_avg_pool2d(x, (gh * 7, gw * 7))
            f = f.reshape(1, 3, gh, 7, gw, 7).permute(0, 2, 4, 1, 3, 5).reshape(gh, gw, -1)
            f = f - f.mean(-1, keepdim=True)
        return F.normalize(f.float(), dim=-1)


@dataclass
class Memory:
    key_feat: torch.Tensor      # [K, C]
    key_box: np.ndarray         # [K, 3] int32 (y0, x0, size) native px, inside the image
    key_scale: np.ndarray       # [K] float32
    query_feat: torch.Tensor    # [gh, gw, C] (s = 1)
    tok_native: int             # native px per query token (4 * tok)
    native_hw: tuple[int, int]

    @property
    def grid(self) -> tuple[int, int]:
        return tuple(self.query_feat.shape[:2])

    def query_boxes(self) -> np.ndarray:
        """[Q, 3] native box of every query token (row-major), clipped to the image."""
        gh, gw = self.grid
        t = self.tok_native
        ii, jj = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        return np.stack([ii.ravel() * t, jj.ravel() * t, np.full(gh * gw, t)], 1).astype(np.int32)


def _pool(m: torch.Tensor, gh: int, gw: int, t: float) -> torch.Tensor:
    """Mean of a 1x1xHxW map over each token cell of size t (anchor px)."""
    H, W = m.shape[-2:]
    m = F.pad(m, (0, max(0, math.ceil(gw * t) - W), 0, max(0, math.ceil(gh * t) - H)), mode="replicate")
    return F.adaptive_avg_pool2d(m, (gh, gw))[0, 0]


def build_memory(anchor: torch.Tensor, blur: torch.Tensor, feats: Features, scales=(1.0, 0.75, 0.5),
                 focus_thr: float = 0.4, tex_pct: float = 30.0, value_px: int = 64) -> Memory:
    """anchor 1x3xH4xW4 in [0, 1]; blur 1x1xH4xW4 (|DP disparity|, DP px at the anchor resolution).
    Keys = tokens whose mean blur < focus_thr and whose anchor texture is above the tex_pct percentile of the
    in-focus tokens; values = native crops of value_px / s px centred on the key token."""
    H4, W4 = anchor.shape[-2:]
    H, W = 4 * H4, 4 * W4
    luma = anchor.mean(1, keepdim=True)
    hp2 = (luma - F.avg_pool2d(luma, 5, 1, 2, count_include_pad=False)) ** 2
    kf, kb, ks, query = [], [], [], None
    for s in scales:
        a = anchor if s == 1 else F.interpolate(anchor, scale_factor=s, mode="area")
        f = feats(a)
        gh, gw = f.shape[:2]
        t = feats.tok / s                                      # anchor px per token
        if s == 1:
            query = f
        b = _pool(blur.float(), gh, gw, t)
        tex = _pool(hp2, gh, gw, t).sqrt()
        ii, jj = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
        cy, cx = (ii + 0.5) * t * 4, (jj + 0.5) * t * 4         # native centre
        size = int(round(value_px / s))
        inside = ((ii + 1) * t <= H4) & ((jj + 1) * t <= W4)
        foc = (b.cpu() < focus_thr) & inside
        if foc.any():
            thr = torch.quantile(tex.cpu()[foc], tex_pct / 100) if foc.sum() > 3 else 0
            foc = foc & (tex.cpu() >= thr)
        sel = foc.nonzero()
        if len(sel) == 0:
            continue
        y0 = (cy[sel[:, 0], sel[:, 1]] - size / 2).round().clamp(0, H - size).long()
        x0 = (cx[sel[:, 0], sel[:, 1]] - size / 2).round().clamp(0, W - size).long()
        kf.append(f[sel[:, 0], sel[:, 1]])
        kb.append(torch.stack([y0, x0, torch.full_like(y0, size)], 1))
        ks.append(torch.full((len(sel),), s))
    C = query.shape[-1]
    key_feat = torch.cat(kf) if kf else torch.zeros(0, C, device=query.device)
    key_box = torch.cat(kb).numpy().astype(np.int32) if kb else np.zeros((0, 3), np.int32)
    key_scale = torch.cat(ks).numpy().astype(np.float32) if ks else np.zeros(0, np.float32)
    return Memory(key_feat, key_box, key_scale, query, 4 * feats.tok, (H, W))


@torch.no_grad()
def retrieve(mem: Memory, k: int = 32, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Top-k keys (cosine) for every query token: idx [Q, k] int32 (-1 = none), score [Q, k] float16."""
    q = mem.query_feat.reshape(-1, mem.query_feat.shape[-1])
    Q, K = q.shape[0], mem.key_feat.shape[0]
    idx = np.full((Q, k), -1, np.int32)
    sc = np.zeros((Q, k), np.float16)
    if K == 0:
        return idx, sc
    kk = min(k, K)
    keys = mem.key_feat.to(q.device)
    for i in range(0, Q, chunk):
        s, j = (q[i:i + chunk] @ keys.T).topk(kk, dim=1)
        idx[i:i + chunk, :kk] = j.cpu().numpy()
        sc[i:i + chunk, :kk] = s.cpu().numpy().astype(np.float16)
    return idx, sc


def spectrum_desc(crops: torch.Tensor, bins_r: int = 6, bins_a: int = 6) -> torch.Tensor:
    """Texture descriptor of N x 3 x P x P native crops: log power of the luma high band in radial x angular
    frequency bins (shift-invariant, so misregistration does not matter), L2-normalised."""
    y = crops.float().mean(1)
    y = y - F.avg_pool2d(y[:, None], 9, 1, 4, count_include_pad=False)[:, 0]
    P = y.shape[-1]
    p = torch.fft.rfft2(y * torch.hann_window(P, device=y.device)[:, None] * torch.hann_window(P, device=y.device)).abs() ** 2
    fy = torch.fft.fftfreq(P, device=y.device)[:, None]
    fx = torch.fft.rfftfreq(P, device=y.device)[None]
    r = (fy ** 2 + fx ** 2).sqrt() / 0.5
    a = torch.atan2(fy, fx).remainder(math.pi) / math.pi
    rb = (r * bins_r).long().clamp(max=bins_r - 1)
    ab = (a * bins_a).long().clamp(max=bins_a - 1)
    lab = (rb * bins_a + ab).flatten()
    valid = (r.flatten() > 0.05) & (r.flatten() <= 1)
    d = torch.zeros(y.shape[0], bins_r * bins_a, device=y.device)
    d.index_add_(1, lab[valid], p.flatten(1)[:, valid])
    d = (d + 1e-8).log()
    return F.normalize(d - d.mean(1, keepdim=True), dim=1)


def crops(img: torch.Tensor, boxes: np.ndarray, out: int = 64) -> torch.Tensor:
    """img 1x3xHxW (native); boxes [N, 3] (y0, x0, size) -> N x 3 x out x out (area-resized)."""
    res = []
    for y0, x0, s in boxes:
        c = img[..., y0:y0 + s, x0:x0 + s]
        res.append(c if s == out else F.interpolate(c, size=(out, out), mode="area" if s > out else "bicubic",
                                                   **({} if s > out else {"align_corners": False})))
    return torch.cat(res) if res else img.new_zeros(0, img.shape[1], out, out)


@torch.no_grad()
def oracle(mem: Memory, target: torch.Tensor, inp: torch.Tensor, k: int = 4, P: int = 64,
           chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Oracle exemplars: for each query token, the keys whose native input crop has the most similar high-band
    texture to the TARGET at the query (spectrum_desc). idx [Q, k] int32, score [Q, k] float16."""
    Q = mem.grid[0] * mem.grid[1]
    idx = np.full((Q, k), -1, np.int32)
    sc = np.zeros((Q, k), np.float16)
    if len(mem.key_box) == 0:
        return idx, sc
    H, W = mem.native_hw
    qb = mem.query_boxes()
    qb[:, 0] = np.clip(qb[:, 0], 0, H - qb[:, 2])
    qb[:, 1] = np.clip(qb[:, 1], 0, W - qb[:, 2])
    kd = torch.cat([spectrum_desc(crops(inp, mem.key_box[i:i + chunk], P)) for i in range(0, len(mem.key_box), chunk)])
    kk = min(k, kd.shape[0])
    for i in range(0, Q, chunk):
        qd = spectrum_desc(crops(target, qb[i:i + chunk], P))
        s, j = (qd @ kd.T).topk(kk, dim=1)
        idx[i:i + chunk, :kk] = j.cpu().numpy()
        sc[i:i + chunk, :kk] = s.cpu().numpy().astype(np.float16)
    return idx, sc


def overlaps(boxes: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    """Which (y0, x0, size) boxes overlap the native region (y, x, h, w)."""
    y, x, h, w = region
    return ((boxes[:, 0] < y + h) & (boxes[:, 0] + boxes[:, 2] > y) & (boxes[:, 1] < x + w) & (boxes[:, 1] + boxes[:, 2] > x))


def tile_exemplars(idx: np.ndarray, score: np.ndarray, grid: tuple[int, int], tok: int, key_box: np.ndarray,
                   tile: tuple[int, int, int], m: int = 32, k: int = 16, drop: tuple | None = None):
    """Exemplars for one tile. tile = (y4, x4, T4) in anchor px; tok = anchor px per query token.
    Returns keys [m] (-1 padded), token map [T4, T4] (local token id of every tile pixel), token scores [n_tok, m]
    (-inf where that exemplar is not among the token's top-k). drop = native region whose overlapping keys are
    removed (training: no copying in place)."""
    gh, gw = grid
    y4, x4, T4 = tile
    rows = np.clip((np.arange(y4, y4 + T4) // tok), 0, gh - 1)
    cols = np.clip((np.arange(x4, x4 + T4) // tok), 0, gw - 1)
    ur, uc = np.unique(rows), np.unique(cols)
    tok_ids = (ur[:, None] * gw + uc[None]).ravel()                       # global ids of the tile's tokens
    local = {g: i for i, g in enumerate(tok_ids)}
    tmap = np.vectorize(local.get)((rows[:, None] * gw + cols[None]))
    I, S = idx[tok_ids].copy(), score[tok_ids].astype(np.float32)
    bad = I < 0
    if drop is not None and len(key_box):
        bad |= overlaps(key_box, drop)[np.clip(I, 0, None)] & (I >= 0)
    S[bad] = -np.inf
    order = np.argsort(-S, axis=1, kind="stable")[:, :k]               # each token keeps its own best k
    I = np.take_along_axis(I, order, 1)
    S = np.take_along_axis(S, order, 1)
    best: dict[int, float] = {}
    for key, s in zip(I.ravel(), S.ravel()):
        if np.isfinite(s) and s > best.get(int(key), -np.inf):
            best[int(key)] = float(s)
    keys = [kk for kk, _ in sorted(best.items(), key=lambda t: -t[1])[:m]]
    pos = {kk: i for i, kk in enumerate(keys)}
    ts = np.full((len(tok_ids), m), -np.inf, np.float32)
    for t in range(len(tok_ids)):
        for key, s in zip(I[t], S[t]):
            if np.isfinite(s) and int(key) in pos:
                ts[t, pos[int(key)]] = s
    out = np.full(m, -1, np.int64)
    out[:len(keys)] = keys
    return out, tmap.astype(np.int64), ts
