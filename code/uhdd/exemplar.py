"""Non-learned in-focus exemplar transfer (training-free baseline + pilot of the exemplar memory).

For every defocused cell (C x C native px, stride C/2) the high band of the best-matching *in-focus* cell
of the same image is added to a base image (normally the DP composite). Matching happens in the anchor
domain (1/4 res, deblurred): query = anchor window, key = 4x-downscaled input window (sharp there, so
comparable with the anchor). See docs/baselines.md (B3) and plan/method_plan.md (2.1).

Arrays are HxWx3 float32 RGB in [0,1] (native) unless noted; `w` is the copy weight (1 = in focus).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

C, S, CTX = 32, 16, 16     # cell (native px), stride, descriptor window (1/4-res px)


def luma(x: np.ndarray) -> np.ndarray:
    return x @ np.array([0.299, 0.587, 0.114], np.float32)


def high_band(y: np.ndarray, s: int = 4) -> np.ndarray:
    """y - up_s(down_s(y)): the band an s-times smaller image cannot carry (area down, bicubic up)."""
    h, w = y.shape[:2]
    lo = cv2.resize(cv2.resize(y, (w // s, h // s), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_CUBIC)
    return y - lo


def cells(h: int, w: int) -> np.ndarray:
    ys, xs = np.arange(0, h - C + 1, S), np.arange(0, w - C + 1, S)
    return np.stack(np.meshgrid(ys, xs, indexing="ij"), -1).reshape(-1, 2)


def cell_mean(img: np.ndarray, pos: np.ndarray) -> np.ndarray:
    box = cv2.boxFilter(img, -1, (C, C), anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    return box[pos[:, 0], pos[:, 1]]


def lowres_desc(y4: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-mean unit-norm CTX x CTX windows of a 1/4-res luma image centred on native cells, and their std."""
    p = CTX // 2
    yp = np.pad(y4, p, mode="reflect")
    c = (pos + C // 2) // 4 + p
    iy = c[:, :1] - p + np.arange(CTX)[None]
    ix = c[:, 1:] - p + np.arange(CTX)[None]
    win = yp[iy[:, :, None], ix[:, None, :]].reshape(len(pos), -1)
    win = win - win.mean(1, keepdims=True)
    sd = win.std(1)
    return win / (np.linalg.norm(win, axis=1, keepdims=True) + 1e-6), sd


def patches(img: np.ndarray, pos: np.ndarray) -> np.ndarray:
    iy = pos[:, :1] + np.arange(C)[None]
    ix = pos[:, 1:] + np.arange(C)[None]
    return img[iy[:, :, None], ix[:, None, :]]


def nn_match(q: np.ndarray, k: np.ndarray, device="cpu", chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Top-1 cosine match of unit-norm rows q against rows k: (index, similarity)."""
    kt = torch.from_numpy(np.ascontiguousarray(k)).to(device)
    best, sim = [], []
    for i in range(0, len(q), chunk):
        s = torch.from_numpy(np.ascontiguousarray(q[i:i + chunk])).to(device) @ kt.T
        v, j = s.max(1)
        best.append(j.cpu().numpy())
        sim.append(v.cpu().numpy())
    return np.concatenate(best), np.concatenate(sim)


def overlap_add(shape, pos: np.ndarray, pats: np.ndarray) -> np.ndarray:
    win = np.outer(np.hanning(C + 2)[1:-1], np.hanning(C + 2)[1:-1]).astype(np.float32)
    acc, wsum = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    for (y, x), p in zip(pos, pats):
        acc[y:y + C, x:x + C] += win * p
        wsum[y:y + C, x:x + C] += win
    return acc / np.maximum(wsum, 1e-6)


def transfer(inp: np.ndarray, anchor4: np.ndarray, w: np.ndarray, mode: str = "exemplar", gt: np.ndarray | None = None,
             device="cpu", rng=None, focus_thr: float = 0.9, gain_clip=(0.5, 2.0)) -> tuple[np.ndarray, dict]:
    """High-band texture (HxW, luma) for defocused cells; add `(1 - w) * tex` to the base image.

    mode: 'exemplar' (anchor-domain match), 'random' (random in-focus cell), 'oracle' (match on gt high band)."""
    H, W = inp.shape[:2]
    yin = luma(inp)
    hf_in = high_band(yin)
    pos = cells(H, W)
    wc = cell_mean(w, pos)
    bank, query = pos[wc > focus_thr], pos[wc < focus_thr]
    stats = {"bank": int(len(bank)), "query": int(len(query))}
    if len(bank) < 16 or not len(query):
        return np.zeros((H, W), np.float32), stats
    y4 = cv2.resize(yin, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    kd, ksd = lowres_desc(y4, bank)
    qd, qsd = lowres_desc(luma(anchor4), query)
    bank_hf = patches(hf_in, bank)
    if mode == "exemplar":
        j, sim = nn_match(qd, kd, device)
        stats.update(sim_median=float(np.median(sim)), sim_frac_08=float((sim > 0.8).mean()))
        gain = qsd / (ksd[j] + 1e-6)
    elif mode == "random":
        j = (rng or np.random.default_rng(0)).integers(0, len(bank), len(query))
        gain = qsd / (ksd[j] + 1e-6)
    elif mode == "oracle":
        hf_gt = high_band(luma(gt))
        gh = patches(hf_gt, query).reshape(len(query), -1)
        bh = bank_hf.reshape(len(bank), -1)
        j, _ = nn_match(gh / (np.linalg.norm(gh, axis=1, keepdims=True) + 1e-6),
                        bh / (np.linalg.norm(bh, axis=1, keepdims=True) + 1e-6), device)
        gain = patches(hf_gt, query).std((1, 2)) / (bank_hf[j].std((1, 2)) + 1e-6)
    else:
        raise ValueError(mode)
    gain = np.clip(gain, *gain_clip)[:, None, None]
    return overlap_add((H, W), query, bank_hf[j] * gain), stats
