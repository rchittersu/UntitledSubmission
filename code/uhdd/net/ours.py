"""Networks of the method (plan/method_plan.md §2-4; paper Sec. 3).

All variants map one tile to its native estimate under the ANCHOR LOCK:
    inputs  anchor tile a (B,3,T,T), native input tile x (B,3,4T,4T), DP input d (B,2,T,T) at the anchor resolution
            (lightly smoothed |disparity|, confidence),
            exemplar crops e (B,M,3,P,P) with a token map (B,T,T) and per-token exemplar scores (B,n_tok,M)
    output  y = lock(a, Delta): the network's estimate Delta, with its x4 band replaced by the anchor's.

v0 ("hat"): a pretrained x4 SR backbone (HAT-L) on the anchor. A conditioning trunk encodes [native input
(pixel-unshuffled), DP input]; it is added to the features after the first conv and modulates the output of EVERY
backbone stage (SFT: per-pixel scale and shift, zero-initialised, so training starts from the plain backbone) --
the blur map decides the regime (copy / deconvolve / transfer) and must reach every depth. Exemplars enter through
zero-initialised cross-attention after the body; a native-resolution refinement head sees x and d directly.
"plain" is a tiny stand-in backbone with the same interface (CPU tests).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------------------- anchor lock
def down4(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(x, 4)


def up4(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, scale_factor=4, mode="bicubic", align_corners=False)


def up4_exact(x: torch.Tensor) -> torch.Tensor:
    """Smooth x4 upsampling with down4(up4_exact(x)) == x exactly: bicubic, plus the nearest-neighbour upsampled
    residual of bicubic's own round trip (only the anchor's near-Nyquist content is affected)."""
    u = up4(x)
    return u + F.interpolate(x - down4(u), scale_factor=4, mode="nearest")


def anchor_lock(delta: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """Replace the x4 band of `delta` by the anchor (one exact back-projection step, Irani & Peleg):
    y = delta + U(anchor - down4(delta)) with down4(U(e)) = e, hence down4(y) = anchor."""
    return delta + up4_exact(anchor - down4(delta))


# ---------------------------------------------------------------------------------------------------- blocks
class LayerNorm2d(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.ln = nn.GroupNorm(1, c)

    def forward(self, x):
        return self.ln(x)


class NAFBlock(nn.Module):
    """NAFNet block (Chen et al. 2022): depthwise conv, simple gate, simplified channel attention, gated FFN."""

    def __init__(self, c: int, expand: int = 2):
        super().__init__()
        d = c * expand
        self.n1, self.n2 = LayerNorm2d(c), LayerNorm2d(c)
        self.c1, self.c2 = nn.Conv2d(c, d, 1), nn.Conv2d(d, d, 3, 1, 1, groups=d)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(d // 2, d // 2, 1))
        self.c3 = nn.Conv2d(d // 2, c, 1)
        self.f1, self.f2 = nn.Conv2d(c, d, 1), nn.Conv2d(d // 2, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.c2(self.c1(self.n1(x)))
        a, b = y.chunk(2, 1)
        y = a * b
        y = self.c3(y * self.sca(y))
        x = x + self.beta * y
        a, b = self.f1(self.n2(x)).chunk(2, 1)
        return x + self.gamma * self.f2(a * b)


def zero_conv(cin: int, cout: int, k: int = 1) -> nn.Conv2d:
    c = nn.Conv2d(cin, cout, k, 1, k // 2)
    nn.init.zeros_(c.weight)
    nn.init.zeros_(c.bias)
    return c


class CondEncoder(nn.Module):
    """Conditioning trunk: native input tile (pixel-unshuffled to the anchor grid) + DP input -> anchor-grid features.
    `add` (zero-init) is added after the backbone's first conv; one SFT per backbone stage modulates its output."""

    def __init__(self, c: int, width: int = 64, blocks: int = 2, d_ch: int = 2, stages: int = 0):
        super().__init__()
        self.inp = nn.Conv2d(3 * 16 + d_ch, width, 3, 1, 1)
        self.body = nn.Sequential(*[NAFBlock(width) for _ in range(blocks)])
        self.add = zero_conv(width, c)
        self.sft = nn.ModuleList([SFT(width, c) for _ in range(stages)])

    def forward(self, x, d):
        return self.body(self.inp(torch.cat([F.pixel_unshuffle(x, 4), d], 1)))


class SFT(nn.Module):
    """Spatial feature transform: f * (1 + gamma(c)) + beta(c), zero-initialised (identity at start)."""

    def __init__(self, width: int, c: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(width, width, 3, 1, 1), nn.GELU(), zero_conv(width, 2 * c))

    def forward(self, c):
        return self.net(c).chunk(2, 1)


class ExemplarEncoder(nn.Module):
    """Exemplar crops (B, M, 3, P, P) -> tokens (B, M * g * g, C): strided convs to a g x g grid + position embedding."""

    def __init__(self, c: int, p: int = 64, g: int = 4, width: int = 64):
        super().__init__()
        layers, s, w = [], p, 3
        while s > g:
            layers += [nn.Conv2d(w, width, 4, 2, 1), nn.GELU()]
            s, w = s // 2, width
        self.net = nn.Sequential(*layers, nn.Conv2d(width, c, 1))
        self.pos = nn.Parameter(torch.zeros(1, 1, g * g, c))
        self.g = g

    def forward(self, e):
        B, M = e.shape[:2]
        t = self.net(e.flatten(0, 1)).flatten(2).transpose(1, 2)               # (B*M, g*g, C)
        return (t.view(B, M, self.g * self.g, -1) + self.pos).flatten(1, 2)


class ExemplarAttention(nn.Module):
    """Cross-attention from anchor-grid features to exemplar tokens + learned null tokens. Each position may only
    attend to the exemplars retrieved for its query token (bias = alpha * retrieval score, -inf otherwise); null
    tokens are always available (abstain). Output projection zero-initialised (identity at start)."""

    def __init__(self, c: int, heads: int = 6, n_null: int = 4, g: int = 4):
        super().__init__()
        self.h, self.g = heads, g
        self.nq = nn.LayerNorm(c)
        self.nk = nn.LayerNorm(c)
        self.q, self.kv = nn.Linear(c, c), nn.Linear(c, 2 * c)
        self.null = nn.Parameter(torch.randn(1, n_null, c) * 0.02)
        self.alpha = nn.Parameter(torch.tensor(5.0))
        self.out = nn.Linear(c, c)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, f, tokens, tok_map, tok_scores):
        B, C, H, W = f.shape
        q = self.q(self.nq(f.flatten(2).transpose(1, 2)))                       # (B, HW, C)
        kv = torch.cat([self.null.expand(B, -1, -1), tokens], 1)
        k, v = self.kv(self.nk(kv)).chunk(2, -1)
        n_null = self.null.shape[1]
        # bias (B, HW, n_null + M*g*g): per position, its token's score row, repeated over each exemplar's tokens
        rows = torch.gather(tok_scores, 1, tok_map.flatten(1)[..., None].expand(-1, -1, tok_scores.shape[-1]))
        ok = torch.isfinite(rows)                       # mask before scaling: alpha * -inf would give NaN gradients
        rows = (self.alpha * rows.masked_fill(~ok, 0)).masked_fill(~ok, float("-inf"))
        rows = rows.repeat_interleave(self.g * self.g, dim=-1)
        bias = torch.cat([rows.new_zeros(B, H * W, n_null), rows], -1)
        sh = lambda t: t.view(B, -1, self.h, C // self.h).transpose(1, 2)    # noqa: E731
        o = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), attn_mask=bias[:, None].to(q.dtype))
        o = o.transpose(1, 2).reshape(B, H * W, C)
        return f + self.out(o).transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------------------------------------------- backbones
class PlainSR(nn.Module):
    """Tiny x4 SR backbone with HAT's attribute names (CPU tests / debugging)."""

    def __init__(self, c: int = 32, blocks: int = 2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, 3, 1, 1))
        self.img_range = 1.0
        self.conv_first = nn.Conv2d(3, c, 3, 1, 1)
        self.body = nn.Sequential(*[NAFBlock(c) for _ in range(blocks)])
        self.conv_after_body = nn.Conv2d(c, c, 3, 1, 1)
        self.conv_before_upsample = nn.Sequential(nn.Conv2d(c, 64, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsample = nn.Sequential(nn.Conv2d(64, 256, 3, 1, 1), nn.PixelShuffle(2), nn.Conv2d(64, 256, 3, 1, 1), nn.PixelShuffle(2))
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.embed_dim = c

    def forward_features(self, x):
        return self.body(x)


def build_backbone(cfg: dict) -> nn.Module:
    kind = cfg.get("backbone", "plain")
    if kind == "plain":
        return PlainSR(cfg.get("width", 32), cfg.get("blocks", 2))
    if kind == "hat":
        from uhdd import models
        from uhdd.adapters import hat
        from pathlib import Path
        reg = cfg.get("registry_config", str(Path(__file__).resolve().parents[2] / "configs" / "models.yaml"))
        spec = models._expand(models.load_config(reg)[cfg.get("registry", "hat_l_x4")])
        spec["name"] = cfg.get("registry", "hat_l_x4")
        net = hat.build(spec)
        if cfg.get("pretrained", True):
            models._load_state_dict(net, spec)
        net.embed_dim = spec["kwargs"]["embed_dim"]
        models.cache_attn_masks(net)
        return net
    raise ValueError(kind)


# ---------------------------------------------------------------------------------------------------- v0
def stages(sr: nn.Module) -> int:
    return len(sr.layers) if hasattr(sr, "layers") else len(sr.body)


def modulated_features(sr: nn.Module, x: torch.Tensor, cond: torch.Tensor | None, sfts) -> torch.Tensor:
    """The backbone's forward_features with an SFT after every stage (HAT: residual groups in token space;
    plain: blocks)."""
    def mod(f_tokens_or_map, i, to_tokens):
        if cond is None or not sfts:
            return f_tokens_or_map
        g, b = sfts[i](cond)
        if to_tokens:
            g, b = g.flatten(2).transpose(1, 2), b.flatten(2).transpose(1, 2)
        return f_tokens_or_map * (1 + g) + b

    if hasattr(sr, "layers"):                                  # HAT (hat_arch.HAT.forward_features)
        x_size = (x.shape[2], x.shape[3])
        params = {"attn_mask": sr.calculate_mask(x_size).to(x.device), "rpi_sa": sr.relative_position_index_SA,
                  "rpi_oca": sr.relative_position_index_OCA}
        t = sr.patch_embed(x)
        if sr.ape:
            t = t + sr.absolute_pos_embed
        t = sr.pos_drop(t)
        for i, layer in enumerate(sr.layers):
            t = mod(layer(t, x_size, params), i, True)
        return sr.patch_unembed(sr.norm(t), x_size)
    for i, blk in enumerate(sr.body):
        x = mod(blk(x), i, False)
    return x


class OursV0(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.sr = build_backbone(cfg)
        c = self.sr.embed_dim
        self.d_ch = cfg.get("dp_channels", 2)
        self.use_sft = cfg.get("sft", True)
        self.cond = CondEncoder(c, cfg.get("cond_width", 64), cfg.get("cond_blocks", 2), self.d_ch,
                                stages(self.sr) if self.use_sft else 0)
        self.use_ex = cfg.get("exemplars", True)
        if self.use_ex:
            g = cfg.get("ex_grid", 4)
            self.ex_enc = ExemplarEncoder(c, cfg.get("ex_px", 64), g, cfg.get("ex_width", 64))
            self.ex_att = ExemplarAttention(c, cfg.get("heads", 6) if c % cfg.get("heads", 6) == 0 else 4, cfg.get("n_null", 4), g)
        rw = cfg.get("refine_width", 32)
        self.refine_in = nn.Conv2d(3 + 3 + self.d_ch, rw, 3, 1, 1)
        self.refine = nn.Sequential(*[NAFBlock(rw) for _ in range(cfg.get("refine_blocks", 4))])
        self.refine_out = zero_conv(rw, 3, 3)
        self.lock = cfg.get("anchor_lock", True)
        self.use_input = cfg.get("native_input", True)

    def forward(self, a, x, d, ex=None, tok_map=None, tok_scores=None):
        sr = self.sr
        mean = sr.mean.type_as(a)
        f0 = sr.conv_first((a - mean) * sr.img_range)
        cond = self.cond(x if self.use_input else torch.zeros_like(x), d)
        f0 = f0 + self.cond.add(cond)
        body = modulated_features(sr, f0, cond, self.cond.sft)
        if self.use_ex and ex is not None and ex.shape[1] > 0:
            body = self.ex_att(body, self.ex_enc(ex), tok_map, tok_scores)
        h = sr.conv_after_body(body) + f0
        out = sr.conv_last(sr.upsample(sr.conv_before_upsample(h))) / sr.img_range + mean
        xin = x if self.use_input else torch.zeros_like(x)
        r = self.refine_out(self.refine(self.refine_in(torch.cat([out, xin, F.interpolate(d, scale_factor=4, mode="bilinear",
                                                                                         align_corners=False)], 1))))
        delta = out + r
        return anchor_lock(delta, a) if self.lock else delta


def build_model(cfg: dict) -> nn.Module:
    v = cfg.get("version", "v0")
    if v == "v0":
        return OursV0(cfg)
    raise ValueError(f"unknown model version {v}")
