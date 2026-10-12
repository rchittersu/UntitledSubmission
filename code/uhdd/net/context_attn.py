"""Context cross-attention for tiled VOSR (and later other DiT / UNet SR models).

A tile of the image (query tokens of the DiT) attends to *context patches*: native crops of the same image (the tile's
own native crop where in focus, neighbouring crops, retrieved in-focus exemplars), each encoded to tokens in the DiT's
own token space. One `ContextAttention` layer is added after selected DiT blocks:

    x <- x + (gate * strength) . Proj( Attn( Q(norm x), K(ctx + geom), V(ctx) ; bias ) )

- gate: learnable per-channel strength, initialised at 0 -> the wrapped model is exactly the base model at init;
  `strength` is a runtime multiplier (sweeps, ablations).
- Q / K / V / Proj / qk-norm are initialised from the block's self-attention, so with the gate open the layer starts as
  "self-attention extended to the context tokens" (ReFIR-style injection), and fine-tuning starts from there.
- Position ids relative to the query tile: every context token carries its true image position expressed in the query
  tile's token units (query tokens sit at (i, j), i, j = 0..G-1). 2-D axial RoPE with VOSR's own frequencies is applied
  to queries and keys, so attention depends on the relative offset (context - query): a co-located crop lands exactly on
  the query grid, a neighbour 1 tile to the right at +G in x, a retrieved crop at its true (possibly distant) offset.
- Per-patch geometry [dy, dx, log scale, blur, type one-hot (self / neighbour / retrieved), score] -> zero-init MLP added
  to the keys (what RoPE cannot express: scale, blur, provenance, retrieval confidence).
- Learned null tokens (no position) let a query abstain; `bias` (additive logits, -inf = masked) carries retrieval
  scores / top-K masks per query token.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TYPES = ("self", "neighbour", "retrieved")
GEOM_DIM = 5 + len(TYPES) + 1          # dy, dx, log scale, scale ratio, blur | type one-hot | score


# ---------------------------------------------------------------- RoPE with continuous 2-D positions
def _axis_freqs(head_dim: int, theta: float = 10000.0, device=None) -> torch.Tensor:
    """VOSR's VisionRotaryEmbeddingFast('lang'): dim = head_dim / 2 per axis, frequencies 1 / theta^(2k / dim)."""
    dim = head_dim // 2
    return 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: dim // 2].float() / dim))


def rope_cos_sin(coords: torch.Tensor, head_dim: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """coords [..., N, 2] (y, x) in token units (floats) -> cos, sin [..., N, head_dim], VOSR's layout:
    [repeat2(y * f), repeat2(x * f)]."""
    f = _axis_freqs(head_dim, theta, coords.device)
    fy = (coords[..., 0:1].float() * f).repeat_interleave(2, -1)
    fx = (coords[..., 1:2].float() * f).repeat_interleave(2, -1)
    a = torch.cat([fy, fx], -1)
    return a.cos(), a.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    return torch.stack((-x[..., 1], x[..., 0]), -1).flatten(-2)


def apply_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """t [B, H, N, D]; cos / sin [N, D] or [B, N, D]."""
    if cos.dim() == 3:
        cos, sin = cos[:, None], sin[:, None]
    return t * cos.to(t.dtype) + rotate_half(t) * sin.to(t.dtype)


def grid_coords(G: int, device=None) -> torch.Tensor:
    """Query token positions of a G x G tile: [G*G, 2] (row-major)."""
    i, j = torch.meshgrid(torch.arange(G, device=device), torch.arange(G, device=device), indexing="ij")
    return torch.stack([i.flatten(), j.flatten()], -1).float()


def patch_coords(box: tuple[float, float, float], tile: tuple[float, float, float], g: int, G: int) -> torch.Tensor:
    """Positions of the g x g tokens of a context patch, in the query tile's token units.
    box / tile = (y0, x0, size) in native px (same image). A query token (i, j) covers native
    [y0 + i*s/G, y0 + (i+1)*s/G); a context token (u, v) covers its own cell; both are placed at their cell centre minus
    half a token, so the co-located patch (box == tile, g == G) gives exactly grid_coords(G)."""
    by, bx, bs = box
    ty, tx, ts = tile
    step = ts / G
    u = torch.arange(g, dtype=torch.float32)
    cy = (by + (u + 0.5) * bs / g - ty) / step - 0.5
    cx = (bx + (u + 0.5) * bs / g - tx) / step - 0.5
    yy, xx = torch.meshgrid(cy, cx, indexing="ij")
    return torch.stack([yy.flatten(), xx.flatten()], -1)


def patch_geometry(box, tile, blur: float = 0.0, kind: str = "retrieved", score: float = 0.0) -> torch.Tensor:
    """Per-patch geometry vector (GEOM_DIM): offset of the centres / tile size, log(size ratio), size ratio, blur,
    type one-hot, retrieval score."""
    by, bx, bs = box
    ty, tx, ts = tile
    dy = (by + bs / 2 - ty - ts / 2) / ts
    dx = (bx + bs / 2 - tx - ts / 2) / ts
    oh = [1.0 if kind == k else 0.0 for k in TYPES]
    return torch.tensor([dy, dx, math.log(bs / ts), bs / ts, float(blur), *oh, float(score)], dtype=torch.float32)


# ---------------------------------------------------------------- the layer
class ContextAttention(nn.Module):
    def __init__(self, attn: nn.Module, norm: nn.Module, hidden: int, heads: int, n_null: int = 4,
                 gate_init: float = 0.0, geom_dim: int = GEOM_DIM, rope: bool = True):
        """attn: the block's self-attention (LightningDiT `Attention`: qkv, q_norm, k_norm, proj) to initialise from;
        norm: the block's norm1 (query normalisation, without the adaLN modulation)."""
        super().__init__()
        self.heads, self.hd, self.rope = heads, hidden // heads, rope
        w, b = attn.qkv.weight.detach(), attn.qkv.bias.detach() if attn.qkv.bias is not None else None
        self.q, self.k, self.v = (nn.Linear(hidden, hidden, bias=b is not None) for _ in range(3))
        for i, lin in enumerate((self.q, self.k, self.v)):
            lin.weight.data.copy_(w[i * hidden:(i + 1) * hidden])
            if b is not None:
                lin.bias.data.copy_(b[i * hidden:(i + 1) * hidden])
        self.q_norm, self.k_norm = copy.deepcopy(attn.q_norm), copy.deepcopy(attn.k_norm)
        self.proj = copy.deepcopy(attn.proj)
        self.norm = copy.deepcopy(norm)
        self.null = nn.Parameter(torch.zeros(n_null, hidden))
        self.geom = nn.Sequential(nn.Linear(geom_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        nn.init.zeros_(self.geom[-1].weight)
        nn.init.zeros_(self.geom[-1].bias)
        self.gate = nn.Parameter(torch.full((hidden,), float(gate_init)))
        self.strength = 1.0                                       # runtime multiplier (not trained)
        self.last_attn_mass = None                                # diagnostics: attention mass on non-null tokens

    def forward(self, x, q_coords, ctx, ctx_coords, ctx_geom=None, bias=None, keep_attn: bool = False):
        """x [B, N, C]; q_coords [N, 2]; ctx [B, M, C] (context tokens); ctx_coords [B, M, 2]; ctx_geom [B, M, geom_dim]
        (per token, i.e. the patch vector repeated over its tokens) or None; bias [B, N, M] additive logits or None."""
        B, N, C = x.shape
        M = ctx.shape[1]
        h, d = self.heads, self.hd
        kin = ctx if ctx_geom is None else ctx + self.geom(ctx_geom.to(ctx.dtype))
        q = self.q_norm(self.q(self.norm(x)).view(B, N, h, d).transpose(1, 2))
        k = self.k_norm(self.k(kin).view(B, M, h, d).transpose(1, 2))
        v = self.v(ctx).view(B, M, h, d).transpose(1, 2)
        if self.rope:
            qc, qs = rope_cos_sin(q_coords, d)
            kc, ks = rope_cos_sin(ctx_coords, d)
            q, k = apply_rope(q, qc, qs), apply_rope(k, kc, ks)
        nk = self.k_norm(self.k(self.null).view(1, -1, h, d).transpose(1, 2)).expand(B, -1, -1, -1)
        nv = self.v(self.null).view(1, -1, h, d).transpose(1, 2).expand(B, -1, -1, -1)
        k, v = torch.cat([k, nk.to(k.dtype)], 2), torch.cat([v, nv.to(v.dtype)], 2)
        mask = None
        if bias is not None:
            mask = torch.cat([bias, bias.new_zeros(B, N, self.null.shape[0])], -1)[:, None].to(q.dtype)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        if keep_attn:
            with torch.no_grad():
                a = (q @ k.transpose(-2, -1)) / math.sqrt(d)
                a = (a + mask if mask is not None else a).softmax(-1)
                self.last_attn_mass = a[..., :M].sum(-1).mean().item()
        o = self.proj(o.transpose(1, 2).reshape(B, N, C))
        return x + (self.gate * self.strength).to(o.dtype) * o


# ---------------------------------------------------------------- VOSR wrapper
class VOSRContext(nn.Module):
    """LightningDiT (VOSR) + ContextAttention after the blocks in `blocks` (indices; default: the last third).
    forward(x, t, r, z, ctx) with ctx = None is exactly the base model's forward; ctx is a dict:
      tokens [B, M, C] (from `embed_context`), coords [B, M, 2], geom [B, M, GEOM_DIM] or None, bias [B, N, M] or None."""

    def __init__(self, dit: nn.Module, blocks=None, n_null: int = 4, gate_init: float = 0.0, rope: bool = True):
        super().__init__()
        self.dit = dit
        D = len(dit.blocks)
        self.sel = sorted(blocks) if blocks is not None else list(range(D - D // 3, D))
        self.ctx_attn = nn.ModuleDict({str(i): ContextAttention(dit.blocks[i].attn, dit.blocks[i].norm1, dit.hidden_size,
                                                                dit.num_heads, n_null, gate_init, rope=rope)
                                       for i in self.sel})

    def adapter_parameters(self):
        return self.ctx_attn.parameters()

    def embed_context(self, ref_lat: torch.Tensor) -> torch.Tensor:
        """Clean latents of context crops [B, M, c, h, w] -> tokens [B, M*T, C] with the DiT's own patch embedding, fed
        as (lq = crop, state = crop): the token distribution of a clean, already-restored patch."""
        B, Mp = ref_lat.shape[:2]
        t = self.dit.x_embedder(torch.cat([ref_lat, ref_lat], 2).flatten(0, 1))      # [B*M, T, C]
        return t.reshape(B, Mp * t.shape[1], t.shape[2])

    def forward(self, x, t, r=None, z=None, ctx: dict | None = None):
        d = self.dit
        N, _, H, W = x.shape
        G = H // d.patch_size
        h = d.x_embedder(x)
        t_raw = t
        te = d.t_embedder(t)
        re = d.r_embedder(r) * (t_raw - r).unsqueeze(-1) if d.r_embedder is not None else 0
        c = te + re
        c0 = d.t_block(c)
        if d.z_dims is not None and z is not None:
            z = d.mlp_ca(d.layer_norm(z[0]))
        rope = d.feat_rope
        if d.use_rope and G != d.x_embedder.img_size[0] // d.patch_size:
            rope = d._get_dynamic_rope(G, x.device, h.dtype)
        qc = grid_coords(G, x.device)
        for i, blk in enumerate(d.blocks):
            h = blk(h, c0, z, rope)
            if ctx is not None and str(i) in self.ctx_attn:
                h = self.ctx_attn[str(i)](h, qc, ctx["tokens"], ctx["coords"], ctx.get("geom"), ctx.get("bias"),
                                          keep_attn=ctx.get("keep_attn", False))
        h = d.final_layer(h, c)
        return d.unpatchify(h)
