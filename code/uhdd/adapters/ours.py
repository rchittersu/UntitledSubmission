"""The method as a registry model (registry `ours_*`), run by run_model.py like every baseline.

run_model passes the ANCHOR (x4) as the input image; the native input and the DP blur map of the same scene come in
as auxiliary inputs (`--aux native=DIR --aux dp=DIR`, passed by launch.py) through `set_image`. Per image (`prepare`):
the exemplar memory is built on the whole anchor and retrieval is run once, with exactly the code of the training
cache (uhdd/train/memory.py). Per tile: native crop, blur crop, the tile's exemplars, network, anchor lock.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from uhdd.train import memory as mem_mod
from uhdd.train.data import TOK_MAX


class OursNet(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net


def build(spec: dict):
    from uhdd.net.ours import build_model
    ck = torch.load(spec["weights"], map_location="cpu", weights_only=False)
    cfg = {**ck["cfg"], "pretrained": False}
    net = build_model(cfg)
    net.load_state_dict(ck["model"])
    return OursNet(net)


def wrap(holder: OursNet, spec: dict):
    from uhdd.dualpixel import load_blur_map, load_dp_input
    from uhdd.io import read_image, to_tensor
    net = holder.net
    dev = next(net.parameters()).device
    feats = mem_mod.Features(spec.get("features", "dinov2"), spec.get("dino_name", "dinov2_vitl14"),
                             spec.get("dino_hub"), dev)
    scales = tuple(spec.get("scales", (1.0, 0.75, 0.5)))
    st: dict = {}

    def set_image(name: str, aux: dict) -> None:
        st.clear()
        st["name"], st["aux"] = name, aux

    @torch.no_grad()
    def prepare(anchor: torch.Tensor) -> None:
        if "aux" not in st:
            raise RuntimeError("ours_* needs --aux native=DIR --aux dp=DIR (launch.py passes them)")
        n, h, w = st["name"], anchor.shape[-2], anchor.shape[-1]
        x = to_tensor(read_image(f"{st['aux']['native']}/{n}.png"), anchor.device)
        if x.shape[-2:] != (4 * h, 4 * w):
            raise ValueError(f"{n}: native {tuple(x.shape[-2:])} is not 4 x anchor {(h, w)}")
        b = load_blur_map(st["aux"]["dp"], n, hw=(h, w), win=spec.get("blur_win", 15))
        if b is None:
            raise FileNotFoundError(f"{n}: no DP map in {st['aux']['dp']}")
        blur = torch.from_numpy(b)[None, None].to(anchor.device)          # memory keys (as in the cache)
        dp = torch.from_numpy(load_dp_input(st["aux"]["dp"], n, hw=(h, w), win=spec.get("dp_win", 3)))[None]
        mem = mem_mod.build_memory(anchor.float(), blur, feats, scales, spec.get("focus_thr", 0.4),
                                   spec.get("tex_pct", 30.0), spec.get("value_px", 64))
        idx, sc = mem_mod.retrieve(mem, spec.get("topk", 32))
        st.update(x=x, dp=dp.to(anchor.device), mem=mem, idx=idx, sc=sc)

    def padded(t: torch.Tensor, H: int, W: int, mode: str) -> torch.Tensor:
        ph, pw = H - t.shape[-2], W - t.shape[-1]
        return t if ph <= 0 and pw <= 0 else F.pad(t, (0, max(pw, 0), 0, max(ph, 0)), mode=mode)

    @torch.no_grad()
    def fn(t: torch.Tensor, boxes=None, full_hw=None) -> torch.Tensor:
        B, _, T, _ = t.shape
        Hp = max(b[0] + b[2] for b in boxes)
        Wp = max(b[1] + b[3] for b in boxes)
        x = padded(st["x"], 4 * Hp, 4 * Wp, "reflect")
        d = padded(st["dp"], Hp, Wp, "replicate")
        mem = st["mem"]
        m, k, P = spec.get("m", 32), spec.get("k", 16), net.cfg.get("ex_px", 64)
        xs, ds, exs, tms, tss = [], [], [], [], []
        for (y4, x4, th, tw) in boxes:
            xs.append(x[..., 4 * y4:4 * (y4 + th), 4 * x4:4 * (x4 + tw)])
            ds.append(d[..., y4:y4 + th, x4:x4 + tw])
            keys, tmap, ts = mem_mod.tile_exemplars(st["idx"], st["sc"], mem.grid, mem.tok_native // 4, mem.key_box,
                                                    (y4, x4, th), m, k, None)
            e = torch.zeros(m, 3, P, P, device=t.device)
            v = keys >= 0
            if v.any():
                e[torch.from_numpy(v).to(t.device)] = mem_mod.crops(st["x"], mem.key_box[keys[v]], P).to(e.dtype)
            tsp = np.full((TOK_MAX * TOK_MAX, m), -np.inf, np.float32)
            tsp[:ts.shape[0]] = ts
            exs.append(e[None])
            tms.append(torch.from_numpy(tmap)[None])
            tss.append(torch.from_numpy(tsp)[None])
        y = net(t, torch.cat(xs).to(t.dtype), torch.cat(ds).to(t.dtype), torch.cat(exs).to(t.dtype),
                torch.cat(tms).to(t.device), torch.cat(tss).to(t.device))
        return y.clamp(0, 1)

    def finalize(y: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """Re-apply the anchor lock to the blended image: tile overlaps are blended per native pixel, which breaks
        down4(y) == anchor slightly at seams; one global back-projection step restores it exactly."""
        from uhdd.net.ours import anchor_lock
        y, a = y.float(), anchor.float()
        for _ in range(spec.get("lock_iters", 16)):  # alternate the lock and [0, 1] (both convex; the anchor's
            y = anchor_lock(y, a).clamp(0, 1)        # upsampling lies in both, so this converges)
        return y

    fn.positional = True
    fn.finalize = finalize
    fn.prepare = prepare
    fn.set_image = set_image
    return fn
