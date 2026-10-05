"""CPU tests for P2 helpers (pairing of official splits, VAE-ceiling crop selection, blur map loading)."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pair_official import pair_stems  # noqa: E402
from vae_ceiling import pick_crops  # noqa: E402
from uhdd.dualpixel import load_blur_map  # noqa: E402


def test_pair_stems_sorted_and_checked():
    assert pair_stems(["1P0A0933", "1P0A0917"], ["1P0A0916", "1P0A0932"]) == [("1P0A0917", "1P0A0916"),
                                                                             ("1P0A0933", "1P0A0932")]
    with pytest.raises(ValueError):
        pair_stems(["1P0A0917", "1P0A0933"], ["1P0A0916", "1P0A0990"])
    with pytest.raises(ValueError):
        pair_stems(["1P0A0917"], [])


def test_pick_crops_extremes():
    blur = np.zeros((300, 400), np.float32)
    blur[:100, 300:] = 5.0                      # one strongly defocused cell
    blur[200:, :100] = -1.0                     # one cell sharper than the rest
    c = pick_crops(blur, blur.shape, 100, 1, np.random.default_rng(0))
    assert c == [("focus", 200, 0), ("defocus", 0, 300)]
    assert len(pick_crops(None, blur.shape, 100, 2, np.random.default_rng(0))) == 4


def test_load_blur_map(tmp_path):
    d = np.full((40, 60), 2.5, np.float32)
    d[:, 30:] = -2.5                            # sign is ignored
    cv2.imwrite(str(tmp_path / "a_disp.png"), np.round(d * 1000 + 32768).astype(np.uint16))
    cv2.imwrite(str(tmp_path / "a_conf.png"), np.full((40, 60), 255, np.uint8))
    k = load_blur_map(tmp_path, "a", (80, 120), win=5)
    assert k.shape == (80, 120) and np.allclose(k, 2.5, atol=1e-3)
    assert load_blur_map(tmp_path, "missing") is None


# --- protocol metrics (uhdd.metrics.binned) ---------------------------------------------------------
import torch  # noqa: E402

from uhdd.metrics import binned  # noqa: E402


def _img(h=128, w=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(1, 3, h, w, generator=g)
    return torch.nn.functional.avg_pool2d(x, 3, 1, 1)          # some spatial correlation


def test_aligned_psnr_recovers_shift():
    gt = _img(200, 200)
    pred = torch.roll(gt, shifts=(1, -2), dims=(2, 3))
    r = binned.aligned_psnr(pred, gt, None, {"align_tile": 64, "align_radius": 2})
    assert r["apsnr"] > 80 and r["apsnr_shift"] == 3
    r1 = binned.aligned_psnr(pred, gt, None, {"align_tile": 64, "align_radius": 0})
    assert r1["apsnr"] < 40


def test_no_harm():
    gt, inp = _img(seed=1), _img(seed=2)
    blur = torch.zeros(1, 1, 128, 128)
    blur[..., 64:] = 2.0                                         # left half in focus
    pred = inp.clone()
    pred[..., 64:] = gt[..., 64:]                                # fixes only the defocused half
    r = binned.no_harm(pred, gt, inp, None, blur, {})
    assert r["keep_psnr_b0"] == float("inf") and abs(r["dpsnr_b0"]) < 1e-9
    r2 = binned.no_harm(gt, gt, inp, None, blur, {})
    assert r2["dpsnr_b0"] > 0


def test_perceptual_bins_assignment():
    gt = _img(128, 256, seed=3)
    blur = torch.zeros(1, 1, 128, 256)
    blur[..., 128:] = 5.0                                        # right tiles -> bin 3
    r = binned.perceptual_bins(gt, gt, None, blur, {"bin_tile": 64}, names=())
    assert (r["ntile_b0"], r["ntile_b1"], r["ntile_b3"]) == (4.0, 0.0, 4.0)


def test_resolution_sweep_psnr_only(monkeypatch):
    monkeypatch.setattr(binned.iqa, "full_reference", lambda *a, **k: 0.0)
    gt = _img(64, 64, seed=4)
    r = binned.resolution_sweep(gt + 0.01, gt, torch.ones(1, 1, 64, 64, dtype=torch.bool), {"eval_scales": (2, 4)})
    assert abs(r["psnr_s2"] - 40.0) < 1e-3 and abs(r["psnr_s4"] - 40.0) < 1e-3


# --- training-free fusion baselines (scripts/fuse_baselines.py) --------------------------------------
from fuse_baselines import fuse, guided_filter, ramp  # noqa: E402


def test_multiscale_weights_partition():
    b = np.linspace(0, 5, 101, dtype=np.float32)[None].repeat(4, 0)
    x, u2, u4 = (np.full((4, 101, 3), v, np.float32) for v in (0.0, 0.5, 1.0))
    y, info = fuse("multiscale", {"t0": 0.4, "t1": 1.2, "m0": 1.5, "m1": 2.5}, x, u4, u2, b)
    v = y[0, :, 0]
    assert v[b[0] < 0.4].max() == 0.0                     # input in focus
    assert np.allclose(v[(b[0] > 1.2) & (b[0] < 1.5)], 0.5)  # x2 in the middle
    assert v[b[0] > 2.5].min() == 1.0                      # x4 for strong blur
    assert np.all(np.diff(v) >= -1e-6)


def test_guided_filter_identity_and_flat():
    g = _img(64, 64).numpy()[0].transpose(1, 2, 0)
    assert np.allclose(guided_filter(g, g, 4, 1e-6), g, atol=1e-3)     # src == guide -> unchanged
    flat = np.full_like(g, 0.3)
    assert np.allclose(guided_filter(g, flat, 4, 1e-3), 0.3, atol=1e-5)  # flat src stays flat


def test_ramp():
    assert ramp(np.array([0.0, 0.8, 2.0]), 0.4, 1.2).tolist() == [1.0, 0.5, 0.0]


# --- reference-based SR baseline (scripts/refsr_baseline.py) ------------------------------------------
from refsr_baseline import RefBuilder  # noqa: E402


def test_ref_builder_shapes_and_colocation():
    rng = np.random.default_rng(0)
    x = rng.random((512, 768, 3), dtype=np.float32)                 # native
    a4 = cv2.resize(x, (192, 128), interpolation=cv2.INTER_AREA)    # anchor at 1/4
    blur4 = np.zeros((128, 192), np.float32)
    blur4[:, 96:] = 5.0                                              # right half defocused
    col = RefBuilder(x, a4, blur4, 32, "colocated")((16, 40, 32, 32))
    assert col.shape == (128, 128, 3) and np.array_equal(col, x[64:192, 160:288])
    rb = RefBuilder(x, a4, blur4, 32, "mosaic")
    assert rb.cands and all(c[1] + 64 <= 96 * 4 for c in rb.cands)   # candidates only in the in-focus half
    m = rb((100, 170, 32, 32))                                       # box partly outside the image
    assert m.shape == (128, 128, 3)


# --- TLC adapter (uhdd/adapters/tlc.py) ---------------------------------------------------------------
def test_tlc_local_attention_matches_global_when_small_and_windows_when_large():
    import types as _t
    from uhdd.adapters import tlc

    class Attention(torch.nn.Module):                        # same name/attribute as Restormer's MDTA
        def __init__(self):
            super().__init__()
            self.temperature = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):                                # global statistic: depends on the whole map
            return x - x.mean(dim=(-2, -1), keepdim=True)

    net = torch.nn.Sequential(Attention())
    fn = tlc.wrap(net, {"tlc_base": 32, "clamp": False})
    assert fn.n_patched == 1
    x = torch.rand(1, 1, 32, 32)
    assert torch.allclose(fn(x), x - x.mean())               # fits in one window -> unchanged
    big = torch.zeros(1, 1, 32, 64)
    big[..., 32:] = 1.0
    fn(torch.zeros(1, 1, 8, 8))                              # reference size is per call
    net[0]._tlc_ref[:] = [32, 64]
    y = net[0](big)                                          # windows of 32 px, stride 16 over the width
    assert torch.allclose(y[..., :16], torch.zeros(1, 1, 32, 16))   # left window sees only zeros


# --- per-method default tiling (uhdd.tiling.default_tiling) --------------------------------------------
from uhdd.tiling import default_tiling  # noqa: E402


def test_default_tiling_paper_setups():
    dpdd = {"paper_input": [1120, 1680], "multiple": 16}
    assert default_tiling(dpdd, (1120, 1680)) == (0, 0)            # x4 = paper protocol: whole image
    assert default_tiling(dpdd, (1680, 1120)) == (0, 0)            # either orientation
    assert default_tiling(dpdd, (4480, 6720)) == (1120, 140)       # native: tiles of the paper's short side
    assert default_tiling({"paper_input": [1500, 2000], "multiple": 4}, (2240, 3360)) == (1500, 188)
    sr = {"paper_tile": 400, "multiple": 8}
    assert default_tiling(sr, (1120, 1680)) == (400, 50)
    assert default_tiling(sr, (300, 400)) == (0, 0)                # fits in one tile -> whole
    assert default_tiling({"paper_tile": 130, "multiple": 16}, (1000, 1000))[0] == 128   # rounded to multiple
    assert default_tiling({}, (4480, 6720)) == (0, 0)              # builtins: whole image
    assert default_tiling(dpdd, (4480, 6720), overlap_ratio=0.25) == (1120, 280)


def test_blur_stratified_metrics_fail_loudly_without_dp_maps():
    import importlib.util
    spec = importlib.util.spec_from_file_location("evaluate_script", str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    import pytest
    with pytest.raises(FileNotFoundError):
        ev.require_blur_map(["psnr", "percbins"], None, "/missing", "img")
    ev.require_blur_map(["psnr", "ssim"], None, None, "img")          # metrics that do not need a map: fine
    ev.require_blur_map(["percbins"], object(), "/x", "img")          # map present: fine


def test_subset_results_recomputes_means(tmp_path):
    import csv
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location("subset_results", str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "subset_results.py"))
    sr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sr)
    d = tmp_path / "step"
    d.mkdir()
    with open(d / "metrics_x.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "psnr"])
        for n, v in (("a", 10.0), ("b", 20.0), ("c", 40.0)):
            w.writerow([n, v])
    (d / "metrics_x.json").write_text(json.dumps({"label": "m", "metrics": {"psnr": {"mean": 23.3, "std": 1, "n": 3}, "time_s": 5.0}}))
    out = sr.subset_one(d / "metrics_x.json", {"a", "c"}, tmp_path / "out")
    j = json.loads(out.read_text())
    assert j["metrics"]["psnr"]["mean"] == 25.0 and j["metrics"]["psnr"]["n"] == 2 and j["metrics"]["time_s"] == 5.0
    assert json.loads((tmp_path / "out" / "step__metrics_x.json").read_text())["n_images"] == 2


def test_attn_mask_cache_is_exact_and_computed_once():
    import torch
    from uhdd.models import cache_attn_masks

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))
            self.calls = 0

        def calculate_mask(self, x_size):
            self.calls += 1
            return torch.arange(x_size[0] * x_size[1], dtype=torch.float32).reshape(*x_size)

    net = torch.nn.Sequential(Block(), Block())
    ref = [b.calculate_mask((3, 4)) for b in net]
    assert cache_attn_masks(net) == 2 and cache_attn_masks(net) == 0        # idempotent
    for b in net:
        b.calls = 0
        a1, a2, a3 = b.calculate_mask((3, 4)), b.calculate_mask((3, 4)), b.calculate_mask((2, 2))
        assert b.calls == 2 and a1 is a2 and a3.shape == (2, 2)             # one build per distinct size
    assert all(torch.equal(r, b.calculate_mask((3, 4))) for r, b in zip(ref, net))


def test_missing_outputs_detects_absent_and_empty_files(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_model_script", str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "run_model.py"))
    rm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rm)
    import cv2
    import numpy as np
    cv2.imwrite(str(tmp_path / "a.png"), np.zeros((4, 4, 3), np.uint16))
    (tmp_path / "b.png").write_bytes(b"\x89PNG-truncated")
    assert rm.missing_outputs(tmp_path, ["a", "b", "c"]) == ["b", "c"]


def test_pipeline_cost_is_per_image_and_consistent(tmp_path):
    import csv
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_matrix_script", str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "run_matrix.py"))
    rm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rm)
    steps = [{"a": {"time_s": 1.0, "peak_mem_gb": 60.0}, "b": {"time_s": 2.0, "peak_mem_gb": 2.0}},
             {"a": {"time_s": 3.0, "peak_mem_gb": 8.0}, "b": {"time_s": 4.0, "peak_mem_gb": 9.0}}]
    c = rm.pipeline_cost(steps)
    assert c["pipeline_time_s"]["mean"] == 5.0 and c["peak_mem_gb"]["mean"] == 34.5 and c["peak_mem_gb_max"]["mean"] == 60.0
    src = tmp_path / "m.csv"
    with open(src, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["name", "psnr", "peak_mem_gb"]); w.writerow(["a", 1, 8.0]); w.writerow(["b", 2, 9.0])
    rows = list(csv.DictReader(open(rm.pipeline_csv(src, steps, tmp_path / "p.csv"))))
    assert [float(r["peak_mem_gb"]) for r in rows] == [60.0, 9.0] and [float(r["pipeline_time_s"]) for r in rows] == [4.0, 6.0]


def test_ref_builder_mosaic_focus_drops_blurry_colocated():
    rng = np.random.default_rng(1)
    x = rng.random((512, 768, 3), dtype=np.float32)
    a4 = cv2.resize(x, (192, 128), interpolation=cv2.INTER_AREA)
    blur4 = np.zeros((128, 192), np.float32)
    blur4[:, 96:] = 5.0
    box = (40, 140, 32, 32)                                          # tile in the defocused half
    m = RefBuilder(x, a4, blur4, 32, "mosaic")(box)
    mf = RefBuilder(x, a4, blur4, 32, "mosaic_focus")(box)
    col = RefBuilder(x, a4, blur4, 32, "mosaic").crop(4 * 40 + 64 - 32, 4 * 140 + 64 - 32, 64)
    assert np.array_equal(m[:64, :64], col) and not np.array_equal(mf[:64, :64], col)
    assert np.array_equal(RefBuilder(x, a4, blur4, 32, "mosaic_focus")((40, 10, 32, 32))[:64, :64],
                          RefBuilder(x, a4, blur4, 32, "mosaic").crop(4 * 40 + 64 - 32, 4 * 10 + 64 - 32, 64))


def test_refsr_merge_meta_unions_shards(tmp_path):
    import json
    from refsr_baseline import merge_meta
    (tmp_path / ".meta_1.json").write_text(json.dumps({"images": {"a": {"time_s": 1}}, "summary": {"method": "m"}}))
    (tmp_path / ".meta_2.json").write_text(json.dumps({"images": {"b": {"time_s": 2}}, "summary": {"method": "m"}}))
    merge_meta(tmp_path)
    assert set(json.loads((tmp_path / "meta.json").read_text())["images"]) == {"a", "b"}


def test_ref_builder_counts_tiles_without_candidate():
    x = np.random.default_rng(2).random((256, 384, 3), dtype=np.float32)
    a4 = cv2.resize(x, (96, 64), interpolation=cv2.INTER_AREA)
    rb = RefBuilder(x, a4, np.full((64, 96), 5.0, np.float32), 32, "mosaic_focus")   # nothing in focus
    rb((0, 0, 32, 32))
    assert (rb.tiles, rb.no_candidate) == (1, 1)


def test_launch_sr_models_are_registry_models():
    import launch
    reg = launch.registry()
    assert all(m in reg for m in launch.REGISTRY_SR.values())
    assert launch.step_tag("s3diff_x4", (1120, 1680))[1:] == (192, 24)     # official latent tile 96 = 192 LR px
    assert launch.step_tag("vosr2_x4", (1120, 1680))[1:] == (128, 16)      # official DiT tile 512 out px


def test_layout_paths_and_legacy(monkeypatch, tmp_path):
    from uhdd import layout
    monkeypatch.setenv("UHDD_RESULTS", str(tmp_path))
    a = layout.tag("drbnet_single", 0, 0)
    s = layout.tag("hat_l_x4", 512, 64)
    assert (a, s) == ("drbnet_single@whole", "hat_l_x4@t512o64")
    up = layout.upsample_dir("official_x4", a, s)
    assert up == tmp_path / "dpdd" / "upsample" / "official_x4" / a / s
    assert layout.stage_of(up) == ("upsample", "official_x4")
    assert layout.legacy_dir("upsample", "ours_x4", a, s) == tmp_path / "dpdd_v2" / "steps" / f"x4__{a}__{s}"
    assert layout.legacy_dir("deblur", "official_x4", a) == tmp_path / "dpdd_official_x4" / "steps" / f"x1__{a}"
    assert layout.root().name not in layout.LEGACY
    with pytest.raises(ValueError):
        layout.deblur_dir("ours_x3", a)


def test_launch_step_tag_uses_paper_tiling():
    import launch
    assert launch.step_tag("drbnet_single", (1120, 1680))[0] == "drbnet_single@whole"        # paper input fits
    t, tile, ov = launch.step_tag("drbnet_single", (4480, 6720))                               # native: 1120 tiles
    assert (tile, ov) == (1120, 140) and t == "drbnet_single@t1120o140"
    assert launch.step_tag("hat_l_x4", (1120, 1680))[1:] == (512, 64)
    assert launch.step_tag("drbnet_single", (1120, 1680), tile=256)[1:] == (256, 32)


def test_setup_inputs_name_and_permutation_checks():
    from setup_inputs import names_report, nearest_mismatches
    sets = {"a": {"x", "y", "z"}, "b": {"x", "y"}, "c": {"x", "y", "z", "w"}}
    rep = names_report(sets, "a")
    assert any("b: missing 1" in r for r in rep) and any("c: 1 not in a" in r for r in rep)
    assert names_report({"a": {"x"}, "b": {"x"}}, "a") == []
    rng = np.random.default_rng(0)
    th = {n: rng.random((7, 10, 3)).astype(np.float32) for n in "pqr"}
    same = {n: v + 0.01 for n, v in th.items()}
    assert nearest_mismatches(th, same) == []
    swapped = {"p": same["q"], "q": same["p"], "r": same["r"]}
    assert sorted(nearest_mismatches(th, swapped)) == [("p", "q"), ("q", "p")]


def test_noise_field_shared_by_overlapping_tiles():
    import torch
    from uhdd.adapters.noisefield import NoiseField, gaussian_color_fix
    nf = NoiseField(4, 0.5, seed=3)
    a, b = nf.crop([(0, 0, 32, 32), (0, 16, 32, 32)], (40, 60), "cpu")
    assert a.shape == (4, 16, 16) and torch.equal(a[:, :, 8:], b[:, :, :8])      # overlap -> same noise
    nf.reset()
    assert torch.equal(nf.crop([(0, 0, 32, 32)], (40, 60), "cpu")[0], a)          # seeded per image
    y, src = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
    flat = torch.full_like(src, 0.4)
    out = gaussian_color_fix(flat, src)                                            # low band from src
    assert torch.allclose(out, gaussian_color_fix(torch.full_like(src, 0.7), src), atol=1e-5)


def test_run_tiled_calls_prepare_once_with_whole_image():
    import torch
    from uhdd.tiling import TileSpec, run_tiled
    seen = []

    def fn(t, boxes=None, full_hw=None):
        return t
    fn.positional = True
    fn.prepare = lambda x: seen.append(tuple(x.shape))
    run_tiled(fn, torch.rand(1, 3, 50, 70), TileSpec(32, 4, 2, "linear", 0))
    assert seen == [(1, 3, 50, 70)]


def test_launch_scored_current(tmp_path):
    import json, os, launch
    (tmp_path / "a.png").write_bytes(b"x")
    assert not launch.scored_current(tmp_path)
    m = tmp_path / "metrics_dpdd4.json"
    m.write_text(json.dumps({"n_images": 1}))
    os.utime(m, (2e9, 2e9))
    assert launch.scored_current(tmp_path)
    (tmp_path / "b.png").write_bytes(b"x")          # new image -> stale
    assert not launch.scored_current(tmp_path)
