"""CPU tests of the reference-based tile study (scripts/tile_study.py, uhdd/refsr.py) and of the ReFIR / iRAG
compatibility layer (only when the repos are checked out under $UHDD_REPOS or ext/repos)."""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

REPOS = Path(os.environ.get("UHDD_REPOS", ROOT.parent / "ext" / "repos"))


@pytest.fixture
def isolated_imports():
    """Third-party repos bring top-level names (models, utils, ldm, scripts, train, ...): undo their imports."""
    path, mods = list(sys.path), set(sys.modules)
    yield
    sys.path[:] = path
    for m in set(sys.modules) - mods:
        del sys.modules[m]


def _make_layout(res: Path, names=("v0", "v1")):
    """Synthetic val split in the results layout: in-focus top half (target texture), defocused bottom half."""
    from test_train import H, W, _scene, _w16
    for i, n in enumerate(names):
        tgt, inp, d = _scene(i + 200)
        _w16(res / "inputs/val_x1/inputs" / f"{n}.png", inp)
        _w16(res / "inputs/val_x1/targets" / f"{n}.png", tgt)
        (res / "inputs/val_x1/masks").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(res / "inputs/val_x1/masks" / f"{n}.png"), np.full((H, W), 255, np.uint8))
        _w16(res / "inputs/val_x4/inputs" / f"{n}.png", cv2.resize(inp, (W // 4, H // 4), interpolation=cv2.INTER_AREA))
        _w16(res / "inputs/val_x4/targets" / f"{n}.png", cv2.resize(tgt, (W // 4, H // 4), interpolation=cv2.INTER_AREA))
        dp = res / "inputs/val_dp_maps"
        dp.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dp / f"{n}_disp.png"), (32768 + 1000 * d).astype(np.uint16))
        cv2.imwrite(str(dp / f"{n}_conf.png"), np.full(d.shape, 255, np.uint8))
        _w16(res / "deblur/val_x4/drbnet_single@whole" / f"{n}.png", cv2.resize(tgt, (W // 4, H // 4), interpolation=cv2.INTER_AREA))


def test_tile_study_end_to_end(tmp_path, monkeypatch):
    import tile_study as ts
    res = tmp_path / "dpdd"
    _make_layout(res)
    monkeypatch.setenv("UHDD_RESULTS", str(tmp_path))
    common = ["--split", "val", "--anchor", "drbnet"]
    ts.main(["select", *common, "--tile", "64", "--border", "16", "--per-bin", "3"])
    st = ts.main(["prep", *common, "--modes", "self,retrieved", "--features", "pixels", "--size", "64", "--grid", "2"])
    tiles = st.tiles
    assert {t["bin"] for t in tiles} >= {0, 2} and all(t["size"] == 64 for t in tiles)
    t0 = tiles[0]["id"]
    assert (st.p("refs", "self", f"{t0}.png")).exists() and (st.p("refs", "retrieved@drbnet_single@whole", f"{t0}.png")).exists()
    for model, ref in (("bicubic", "none"), ("highband", "self"), ("highband", "retrieved")):
        ts.main(["run", *common, "--model", model, "--ref", ref])
    ts.main(["report", *common, "--no-perceptual"])
    import csv
    rows = list(csv.DictReader(open(st.p("metrics_drbnet_single@whole.csv"))))
    def mean(run, variant, b):
        v = [float(r["psnr"]) for r in rows if r["run"] == run and r["variant"] == variant and int(r["bin"]) == b]
        return np.mean(v)
    # in focus the observation holds the target's detail: copying its high band beats bicubic by a wide margin
    assert mean("highband__self", "raw", 0) > mean("bicubic", "-", 0) + 3
    # the anchor lock never makes it worse at x4 and keeps the in-focus gain
    assert mean("highband__self", "lock", 0) > mean("bicubic", "-", 0) + 3
    assert (st.p("sheets", "drbnet_single@whole", f"{t0}.jpg")).exists()
    assert "highband__self" in st.p("report_drbnet_single@whole.md").read_text()


def test_lock_holds_on_tiles():
    import tile_study as ts
    from uhdd.net.ours import down4
    import torch.nn.functional as F
    a = 0.1 + 0.8 * torch.rand(1, 3, 16, 16)
    y = F.interpolate(a, scale_factor=4, mode="bicubic", align_corners=False) + 0.05 * torch.randn(1, 3, 64, 64)
    y = ts.lock(y.clamp(0, 1), a)                     # a generated tile with detail that drifted from the anchor
    assert (down4(y) - a).abs().max() < 1e-3 and y.min() >= 0 and y.max() <= 1


@pytest.mark.skipif(not (REPOS / "ReFIR" / "seesr").is_dir(), reason="ReFIR repo not checked out")
def test_refir_sdpa_editor_matches_reference_attention(isolated_imports):
    from uhdd.adapters import compat
    compat.seesr_imports()
    sys.path.insert(0, str(REPOS / "ReFIR" / "seesr"))
    import seesr_register
    from uhdd.adapters.refir import _sdpa_editor
    _sdpa_editor(seesr_register)
    torch.manual_seed(0)
    h, n, dh = 2, 10, 8
    q, k, v = (torch.randn(4 * h, n, dh) for _ in range(3))
    out = seesr_register.AttentionBase().forward(q, k, v, None, None, False, "up", h, scale=dh ** -0.5)
    ref = torch.softmax(q @ k.transpose(1, 2) * dh ** -0.5, -1) @ v
    ref = ref.reshape(4, h, n, dh).permute(0, 2, 1, 3).reshape(4, n, h * dh)
    assert torch.allclose(out, ref, atol=1e-5)


@pytest.mark.skipif(not (REPOS / "iRAG" / "sr").is_dir(), reason="iRAG repo not checked out")
def test_irag_model_code_imports_without_training_deps(isolated_imports):
    from uhdd.adapters import compat
    compat.irag_imports()
    sys.path.insert(0, str(REPOS / "iRAG" / "sr"))
    import importlib
    for m in ("ldm.models.diffusion.ddpm", "ldm.modules.restoration.ttsr", "ldm.models.diffusion.ddim"):
        importlib.import_module(m)
