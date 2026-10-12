"""CPU tests of the context cross-attention for tiled VOSR (uhdd/net/context_attn.py) on a tiny random LightningDiT
from the VOSR repo (skipped when the repo is not checked out under $UHDD_REPOS or ext/repos)."""
import os
import sys
from pathlib import Path

import pytest
import torch

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")          # VOSR decorates its blocks with torch.compile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPOS = Path(os.environ.get("UHDD_REPOS", ROOT.parent / "ext" / "repos"))

from uhdd.net.context_attn import (GEOM_DIM, VOSRContext, apply_rope, grid_coords, patch_coords,  # noqa: E402
                                   patch_geometry, rope_cos_sin)


def test_patch_coords_colocated_and_offsets():
    G = 8
    assert torch.allclose(patch_coords((100, 200, 512), (100, 200, 512), G, G), grid_coords(G))
    # a neighbour one tile to the right: same rows, columns shifted by G
    nb = patch_coords((100, 712, 512), (100, 200, 512), G, G)
    assert torch.allclose(nb[:, 0], grid_coords(G)[:, 0]) and torch.allclose(nb[:, 1], grid_coords(G)[:, 1] + G)
    # a half-size crop at the tile's centre (g tokens over 256 px): spans the middle half of the query grid
    c = patch_coords((228, 328, 256), (100, 200, 512), 4, G)
    assert c[:, 0].min() == pytest.approx(2.0) and c[:, 0].max() == pytest.approx(5.0)
    g = patch_geometry((100, 712, 512), (100, 200, 512), blur=0.3, kind="neighbour", score=0.8)
    assert g.shape == (GEOM_DIM,) and g[0] == 0 and g[1] == pytest.approx(1.0)


def test_rope_is_relative():
    torch.manual_seed(0)
    q, k = torch.randn(1, 2, 5, 16), torch.randn(1, 2, 7, 16)
    pq, pk = torch.rand(5, 2) * 10, torch.rand(7, 2) * 10
    def logits(dq, dk):
        cq, sq = rope_cos_sin(pq + dq, 16)
        ck, sk = rope_cos_sin(pk + dk, 16)
        return apply_rope(q, cq, sq) @ apply_rope(k, ck, sk).transpose(-2, -1)
    off = torch.tensor([3.25, -7.5])
    assert torch.allclose(logits(0, 0), logits(off, off), atol=1e-4)     # same shift of both: unchanged
    assert not torch.allclose(logits(0, 0), logits(0, off), atol=1e-2)   # moving the context: changed


@pytest.fixture
def tiny_vosr():
    if not (REPOS / "VOSR" / "models" / "lightningdit.py").is_file():
        pytest.skip("VOSR repo not checked out")
    path, mods = list(sys.path), set(sys.modules)
    sys.path.insert(0, str(REPOS / "VOSR"))
    from models.lightningdit import LightningDiT
    torch.manual_seed(0)
    dit = LightningDiT(input_size=8, patch_size=1, in_channels=8, out_channels=4, hidden_size=64, depth=3, num_heads=4,
                       z_dims=32, encdim_ratio=2, auxiliary_time_cond=True, use_qknorm=True, use_rope=True,
                       use_rmsnorm=True, num_fused_layers=1).eval()
    for p in dit.parameters():                       # VOSR zero-inits its output layers: randomise for a real test
        p.data.add_(0.02 * torch.randn_like(p))
    yield dit
    sys.path[:] = path
    for m in set(sys.modules) - mods:
        f = getattr(sys.modules[m], "__file__", None) or ""
        if "site-packages" not in f:
            del sys.modules[m]


def _ctx(model, B=2, M=3, G=8, g=4):
    lat = torch.randn(B, M, 4, g, g)
    tok = model.embed_context(lat)
    tile = (0, 0, 512)
    boxes = [(0, 0, 512), (0, 512, 512), (1000, 3000, 256)]
    kinds = ["self", "neighbour", "retrieved"]
    coords = torch.cat([patch_coords(b, tile, g, G) for b in boxes]).expand(B, -1, -1)
    geom = torch.cat([patch_geometry(b, tile, 0.1, k, 0.5).expand(g * g, -1) for b, k in zip(boxes, kinds)]).expand(B, -1, -1)
    return {"tokens": tok, "coords": coords, "geom": geom}


def test_gate_zero_is_exactly_vosr(tiny_vosr):
    m = VOSRContext(tiny_vosr, blocks=[1, 2]).eval()
    x, t, r, z = torch.randn(2, 8, 8, 8), torch.full((2,), 1.0), torch.zeros(2), [torch.randn(2, 10, 32)]
    with torch.no_grad():
        base = tiny_vosr.forward_flexible(x, t, r, z)
        assert torch.allclose(m(x, t, r, z, None), base, atol=1e-5)          # no context: base model
        assert torch.allclose(m(x, t, r, z, _ctx(m)), base, atol=1e-5)      # gate 0: base model


def test_context_reaches_output_and_gets_gradients(tiny_vosr):
    m = VOSRContext(tiny_vosr, blocks=[2])
    for p in m.dit.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        for a in m.ctx_attn.values():
            a.gate.fill_(0.5)
    x, t, r, z = torch.randn(2, 8, 8, 8), torch.full((2,), 1.0), torch.zeros(2), [torch.randn(2, 10, 32)]
    ctx = _ctx(m)
    y = m(x, t, r, z, ctx)
    y.square().mean().backward()
    a = m.ctx_attn["2"]
    assert a.gate.grad.abs().sum() > 0 and a.k.weight.grad.abs().sum() > 0 and a.null.grad.abs().sum() > 0
    assert a.geom[-1].weight.grad.abs().sum() > 0                  # zero-init geometry MLP learns from step 1
    with torch.no_grad():
        ctx2 = dict(ctx, tokens=ctx["tokens"] + 1.0)
        assert not torch.allclose(m(x, t, r, z, ctx2), y.detach(), atol=1e-5)     # context content matters
        # masking every context token (only the null tokens remain): the context content no longer matters
        N, M = 64, ctx["tokens"].shape[1]
        mask = torch.full((2, N, M), float("-inf"))
        y1 = m(x, t, r, z, dict(ctx, bias=mask))
        y2 = m(x, t, r, z, dict(ctx2, bias=mask))
        assert torch.allclose(y1, y2, atol=1e-5)
        # moving the retrieved crop changes the result (position ids reach the attention)
        moved = ctx["coords"].clone()
        moved[:, 32:] += 5.0
        assert not torch.allclose(m(x, t, r, z, dict(ctx, coords=moved)), y.detach(), atol=1e-6)
