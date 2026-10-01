"""CPU tests for padding, tiling, I/O, metrics and registration: `pytest code/tests`."""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uhdd import io, metrics, registration  # noqa: E402
from uhdd.metrics import consistency, fidelity, optics  # noqa: E402
from uhdd.padding import crop, pad_to_multiple  # noqa: E402
from uhdd.tiling import TileSpec, grid_positions, run_tiled  # noqa: E402

torch.manual_seed(0)


def rand_img(h, w):
    return torch.rand(1, 3, h, w)


# ------------------------------------------------------------------ padding
@pytest.mark.parametrize("h,w,m", [(37, 53, 8), (64, 64, 16), (5, 3, 128), (1, 7, 8), (129, 257, 128)])
def test_pad_to_multiple_roundtrip(h, w, m):
    x = rand_img(h, w)
    xp, hw = pad_to_multiple(x, m)
    assert xp.shape[-2] % m == 0 and xp.shape[-1] % m == 0
    assert torch.equal(crop(xp, hw), x)


# ------------------------------------------------------------------ tiling
def test_grid_covers():
    for L, T, O in [(100, 32, 8), (32, 32, 8), (33, 32, 8), (1000, 512, 64)]:
        p = grid_positions(L, T, O)
        assert p[0] == 0 and p[-1] + T == max(L, T)
        assert all(b - a <= T - O for a, b in zip(p, p[1:]))


@pytest.mark.parametrize("blend", ["linear", "gaussian", "hard", "mean"])
@pytest.mark.parametrize("tile,multiple", [(0, 8), (32, 8), (48, 16), (100, 8)])
def test_tiled_identity_exact(blend, tile, multiple):
    x = rand_img(70, 91)
    y, _ = run_tiled(lambda t: t, x, TileSpec(tile, 12, 3, blend), multiple)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-6)


def test_tiled_scale():
    x = rand_img(40, 57)
    up = lambda t: F.interpolate(t, scale_factor=2, mode="nearest")
    y, info = run_tiled(up, x, TileSpec(16, 4, 2, "linear"), 8, scale=2)
    assert torch.allclose(y, up(x), atol=1e-6) and info["scale"] == 2


def test_tiled_matches_whole_for_local_model():
    """With hard blending and overlap >= 2*receptive radius, tiles reproduce whole-image output."""
    w = torch.randn(3, 3, 5, 5) * 0.1
    net = lambda t: F.conv2d(F.pad(t, (2, 2, 2, 2), mode="reflect"), w)
    x = rand_img(96, 130)
    whole = net(x)
    y, _ = run_tiled(net, x, TileSpec(32, 8, 4, "hard"), 8)
    assert torch.allclose(y, whole, atol=1e-5)


# ------------------------------------------------------------------ io
def test_uint16_roundtrip(tmp_path):
    a = np.random.randint(0, 65536, (33, 47, 3), dtype=np.uint16)
    io.write_image(tmp_path / "a.png", a)
    b = io.read_image(tmp_path / "a.png")
    assert b.dtype == np.uint16 and np.array_equal(a, b)
    t = io.to_tensor(io.to_transfer(b), "cpu")
    assert t.shape == (1, 3, 33, 47) and float(t.max()) <= 1
    assert np.array_equal(io.to_numpy(t, np.uint16), a)
    assert np.array_equal(io.to_numpy(io.to_tensor(a[..., :3].astype(np.uint8), "cpu"), np.uint8), a.astype(np.uint8))


def test_pairing(tmp_path):
    for d in ("in", "tg"):
        (tmp_path / d).mkdir()
    for n in ("a", "b"):
        io.write_image(tmp_path / "in" / f"{n}.png", np.zeros((4, 4, 3), np.uint8))
        io.write_image(tmp_path / "tg" / f"{n}.png", np.zeros((4, 4, 3), np.uint8))
    assert [p.name for p in io.pair_folders(tmp_path / "in", tmp_path / "tg")] == ["a", "b"]
    io.write_image(tmp_path / "in" / "c.png", np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(ValueError):
        io.pair_folders(tmp_path / "in", tmp_path / "tg")


# ------------------------------------------------------------------ metrics
def test_ssim_matches_skimage():
    skm = pytest.importorskip("skimage.metrics")
    a, b = rand_img(64, 80), rand_img(64, 80)
    b = 0.7 * a + 0.3 * b
    ref = skm.structural_similarity(a[0].permute(1, 2, 0).numpy().astype(np.float64),
                                    b[0].permute(1, 2, 0).numpy().astype(np.float64),
                                    data_range=1.0, channel_axis=-1)
    assert abs(fidelity.ssim(b, a) - ref) < 1e-6


def test_psnr_and_mask():
    a = rand_img(32, 32)
    b = a.clone()
    b[..., :8, :] += 0.1
    mask = torch.ones(1, 1, 32, 32, dtype=torch.bool)
    mask[..., :8, :] = False
    assert fidelity.psnr(b, a, mask) == float("inf")
    assert abs(fidelity.psnr(b.clamp(0, 1), a) - 10 * math.log10(1 / fidelity.mse(b.clamp(0, 1), a))) < 1e-9


def test_airy_kernel():
    k = optics.airy_kernel(0.55, 22.0, optics.DPDD_PITCH_UM)
    assert abs(k.sum() - 1) < 1e-5 and np.allclose(k, k.T, atol=1e-7) and np.allclose(k, k[::-1], atol=1e-7)
    assert k.argmax() == k.size // 2
    # at 1/4 resolution the PSF is nearly a delta
    assert optics.airy_kernel(0.55, 22.0, optics.DPDD_PITCH_UM * 4).max() > 0.5
    x = rand_img(40, 40)
    assert optics.psf_match(x).shape == x.shape


def test_highband():
    x = rand_img(64, 64)
    low = F.interpolate(F.avg_pool2d(x, 4), scale_factor=4, mode="bicubic", align_corners=False)
    assert abs(optics.highband_nmse_db(low, x)) < 0.05    # ~no high band -> ~0 dB
    assert optics.highband_nmse_db(x, x) < -200


def test_consistency_metrics():
    x = rand_img(512, 768)
    assert consistency.lf_drift(x, x) == 0
    s = consistency.stat_consistency(x, x)
    assert all(abs(v) < 1e-6 for v in s.values())
    # simulated tile inconsistency: each tile gets its own brightness offset, hard blend
    offs = iter(torch.linspace(-0.05, 0.05, 1000)[torch.randperm(1000)].tolist())
    net = lambda t: (t + torch.tensor([next(offs) for _ in range(t.shape[0])]).view(-1, 1, 1, 1))
    y, info = run_tiled(net, x, TileSpec(128, 32, 4, "hard"))
    sm = consistency.seam_metrics(y, x, info)
    assert sm["seam_ratio"] > 3
    assert consistency.lf_drift(y, x) > 1
    y2, info2 = run_tiled(lambda t: t + 0.01, x, TileSpec(128, 32, 4, "hard"))
    assert consistency.seam_metrics(y2, x, info2)["seam_step"] < 1e-3


def test_compute_registry():
    a, b = rand_img(300, 300), rand_img(300, 300)
    out = metrics.compute(["psnr", "ssim", "mae", "pm", "hb", "drift", "stats"], a, b, None,
                          {"cell": 128})
    assert {"psnr", "ssim", "pm_psnr", "hb_nmse_db", "lf_drift", "sharp_lr_std"} <= out.keys()
    assert metrics.compute(["psnr"], a, None, None, {}) == {}


# ------------------------------------------------------------------ registration
def test_registration_recovers_shift():
    rng = np.random.default_rng(0)
    base = (rng.random((400, 600)) * 255).astype(np.float32)
    import cv2
    base = cv2.GaussianBlur(base, (0, 0), 3)
    img = np.clip(np.stack([base] * 3, -1) * 3 - 200, 0, 255).astype(np.uint8)
    M = np.float32([[1, 0, 3.4], [0, 1, -2.2]])
    tgt = cv2.warpAffine(img, M, (600, 400), borderMode=cv2.BORDER_REFLECT)
    r = registration.register_pair(img, tgt, registration.RegConfig(motion="translation", levels=(4, 2, 1)))
    assert r["warped"] and abs(r["corner_shift_px"] - 3.4) < 0.3  # max |dx|,|dy|
    err = np.abs(r["aligned"][20:-20, 20:-20].astype(float) - img[20:-20, 20:-20]).mean()
    assert err < 2.0


# ------------------------------------------------------------------ scripts end-to-end (CPU)
def test_scripts_end_to_end(tmp_path):
    for d in ("in", "tg"):
        (tmp_path / d).mkdir()
    for n in ("a", "b", "c"):
        img = np.random.randint(0, 65536, (96, 128, 3), dtype=np.uint16)
        io.write_image(tmp_path / "in" / f"{n}.png", img)
        io.write_image(tmp_path / "tg" / f"{n}.png", img)
    py, s = sys.executable, ROOT / "scripts"
    run = lambda *a: subprocess.run([py, *map(str, a)], check=True, capture_output=True, text=True)
    run(s / "run_model.py", "--model", "identity", "--inputs", tmp_path / "in", "--out", tmp_path / "id",
        "--gpus", "cpu:2", "--tile", "64", "--overlap", "16", "--workers", "1")
    import json
    meta = json.loads((tmp_path / "id" / "meta.json").read_text())
    assert meta["summary"]["n"] == 3 and meta["images"]["a"]["tiling"]["mode"] == "tiled"
    out = run(s / "evaluate.py", "--pred", tmp_path / "id", "--targets", tmp_path / "tg",
              "--metrics", "psnr,ssim,seam", "--gpus", "cpu:2", "--workers", "1", "--tag", "t").stdout
    assert json.loads((tmp_path / "id" / "metrics_t.json").read_text())["n_images"] == 3
    assert '"ssim": 1.0' in out
    run(s / "prepare_scales.py", "--inputs", tmp_path / "in", "--targets", tmp_path / "tg",
        "--out", tmp_path / "sc", "--factors", "2", "4", "--procs", "2")
    assert io.read_image(tmp_path / "sc" / "x4" / "inputs" / "a.png").shape == (24, 32, 3)
    run(s / "run_model.py", "--model", "bicubic_x4", "--inputs", tmp_path / "sc" / "x4" / "inputs",
        "--out", tmp_path / "up", "--gpus", "cpu", "--workers", "0")
    assert io.read_image(tmp_path / "up" / "a.png").shape == (96, 128, 3)
    table = run(s / "summarize.py", tmp_path / "id" / "metrics_t.json").stdout
    assert "| Method |" in table


def test_positional_tiling_matches_whole():
    """Position-dependent models get full-image coordinates per tile (Bokehlicious adapter)."""
    from uhdd.adapters.bokehlicious import pos_map

    def fn(t, boxes, full_hw):
        pm = torch.stack([pos_map(*full_hw, y, x, h, w, t.device) for y, x, h, w in boxes])
        return t * 0.5 + pm[:, :1] * 0.25 + pm[:, 1:] * 0.25   # pointwise in position

    fn.positional = True
    x = rand_img(70, 101)
    whole, _ = run_tiled(fn, x, TileSpec(0), 4)
    tiled, _ = run_tiled(fn, x, TileSpec(32, 8, 3, "linear"), 4)
    assert torch.allclose(whole, tiled, atol=1e-6)
    # whole-image map equals the reference definition on a landscape image: x spans [0, 1]
    pm = pos_map(70, 101, 0, 0, 70, 101, "cpu")
    assert abs(float(pm[0, 0, 0])) < 1e-7 and abs(float(pm[0, 0, -1]) - 1) < 1e-6


def test_run_matrix_end_to_end(tmp_path):
    import json
    import yaml
    for d in ("inputs", "x1/targets", "x4/inputs", "x4/targets"):
        (tmp_path / "data" / d).mkdir(parents=True)
    rng = np.random.default_rng(0)
    for n in ("a", "b"):
        img = rng.integers(0, 65536, (64, 96, 3), dtype=np.uint16)
        io.write_image(tmp_path / "data" / "inputs" / f"{n}.png", img)
        io.write_image(tmp_path / "data" / "x1" / "targets" / f"{n}.png", img)
        small = img[::4, ::4]
        io.write_image(tmp_path / "data" / "x4" / "inputs" / f"{n}.png", small)
        io.write_image(tmp_path / "data" / "x4" / "targets" / f"{n}.png", small)
    D = str(tmp_path / "data")
    exp = {
        "data": {"inputs": {1: f"{D}/inputs", 4: f"{D}/x4/inputs"},
                 "targets": {1: f"{D}/x1/targets", 4: f"{D}/x4/targets"}},
        "results": str(tmp_path / "res"),
        "run": {"tile": 32, "overlap": 8, "tile_batch": 2},
        "eval": {"tag": "t", "metrics": "psnr,ssim,seam,gridshift", "crop_native": 8},
        "pipelines": [
            {"group": "gap", "name": "id @x{s}", "for": {"s": [4, 1]}, "steps": [{"model": "identity", "scale": "{s}"}]},
            {"group": "up", "name": "id @x4 + {up}", "for": {"up": ["bicubic_x4", "identity"]},
             "steps": [{"model": "identity", "scale": 4, "tile": 0}, {"model": "{up}", "tile": 0}]},
        ],
    }
    ef = tmp_path / "exp.yaml"
    ef.write_text(yaml.safe_dump(exp))
    cmd = [sys.executable, str(ROOT / "scripts" / "run_matrix.py"), str(ef), "--gpus", "cpu", "--workers", "0"]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    assert "4 pipelines, 5 unique steps" in out           # x4 identity whole is shared by both "up" rows
    assert "@shift" in out                                # tiled last steps get a shifted-grid twin
    gap = (tmp_path / "res" / "tables" / "gap.md").read_text()
    assert "id @x1" in gap and "id @x4" in gap
    up = (tmp_path / "res" / "tables" / "up.md").read_text()
    assert "id @x4 + bicubic_x4" in up and "pipeline_time_s" in up
    p = json.loads((tmp_path / "res" / "pipelines" / "id_@x4_+_bicubic_x4__t.json").read_text())
    assert len(p["steps"]) == 2 and p["n_images"] == 2
    # identity @x4 + identity would be evaluated at x4 (no upsampling): scale bookkeeping
    assert json.loads((tmp_path / "res" / "pipelines" / "id_@x4_+_identity__t.json").read_text())["settings"]["scale"] == 4
    pj = sorted(str(f) for f in (tmp_path / "res" / "pipelines").glob("id_@x*_t*.json")) or \
        sorted(str(f) for f in (tmp_path / "res" / "pipelines").glob("*.json"))
    tab = subprocess.run([sys.executable, str(ROOT / "scripts" / "summarize.py"), *pj, "--ci",
                          "--ref", json.loads(open(pj[0]).read())["label"], "--cols", "psnr,ssim"],
                         check=True, capture_output=True, text=True).stdout
    assert "Paired differences" in tab and "[" in tab
    out2 = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout   # resumable
    assert "run_model.py" not in out2 and "cached" in out2


def test_grid_offset_positions_cover_and_shift():
    for L, T, O in [(1120, 512, 64), (1680, 512, 64), (700, 256, 32)]:
        stride = T - O
        a, b = grid_positions(L, T, O), grid_positions(L, T, O, stride // 2)
        for p in (a, b):
            assert p[0] == 0 and p[-1] == L - T
            assert all(0 < q - r <= stride for r, q in zip(p, p[1:]))
        assert set(a[1:-1]).isdisjoint(b[1:-1])           # interior seams differ


def test_grid_shift_metric():
    from uhdd.metrics.consistency import grid_shift_metrics
    x = rand_img(300, 420)
    spec, spec_s = TileSpec(128, 32, 4, "hard"), TileSpec(128, 32, 4, "hard", offset=48)
    # tile-local model: identical outputs on both grids -> no difference at all
    y1, i1 = run_tiled(lambda t: t * 0.9, x, spec)
    y2, i2 = run_tiled(lambda t: t * 0.9, x, spec_s)
    m = grid_shift_metrics(y1, y2, i1, i2)
    assert m["gs_mad"] < 1e-4 and m["gs_psnr"] > 80
    # per-tile brightness offsets (context-dependent model) -> steps at seams
    offs = iter(torch.linspace(-0.05, 0.05, 1000)[torch.randperm(1000)].tolist())
    net = lambda t: t + torch.tensor([next(offs) for _ in range(t.shape[0])]).view(-1, 1, 1, 1)
    y1, i1 = run_tiled(net, x, spec)
    y2, i2 = run_tiled(net, x, spec_s)
    m = grid_shift_metrics(y1, y2, i1, i2)
    assert m["gs_mad"] > 1 and m["gs_seam_ratio"] > 3
