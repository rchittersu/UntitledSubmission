"""CPU tests of the training setup: exemplar memory / retrieval, anchor lock, simulated anchors, cache builder,
dataset, a few training steps, and the trained model as a registry model through the tiler."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from uhdd.net.ours import anchor_lock, build_model, down4  # noqa: E402
from uhdd.train import memory as mem_mod  # noqa: E402
from uhdd.train.sim_anchor import simulate  # noqa: E402

H, W = 256, 384          # native; anchor 64 x 96; top half in focus (textures A | B), bottom half defocused (A)


def _scene(seed: int):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    A = 0.5 + 0.35 * np.sin(xx * 0.9 + rng.uniform(0, 6))                         # vertical stripes
    B = 0.5 + 0.35 * np.sign(np.sin(xx * 0.35) * np.sin(yy * 0.35 + rng.uniform(0, 6)))   # checker
    tgt = np.where((yy < H // 2) & (xx >= W // 2), B, A)
    tgt = np.stack([tgt, tgt * 0.9, tgt * 0.8], -1)
    tgt = np.clip(tgt + rng.normal(0, 0.01, tgt.shape), 0, 1)
    inp = tgt.copy()
    inp[H // 2:] = cv2.GaussianBlur(tgt, (0, 0), 6)[H // 2:]
    d = np.zeros((H // 4, W // 4), np.float32)
    d[H // 8:] = 3.0
    return tgt.astype(np.float32), inp.astype(np.float32), d


def _w16(p: Path, img):
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16)[..., ::-1])


def _make_split(root: Path, split: str, n: int):
    for i in range(n):
        name = f"{split}{i}"
        tgt, inp, d = _scene(i + (0 if split == "train" else 100))
        _w16(root / split / "inputs" / f"{name}.png", inp)
        _w16(root / split / "targets" / f"{name}.png", tgt)
        (root / split / "masks").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(root / split / "masks" / f"{name}.png"), np.full((H, W), 255, np.uint8))
        (root / split / "dp").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(root / split / "dp" / f"{name}_disp.png"), (32768 + 1000 * d).astype(np.uint16))
        cv2.imwrite(str(root / split / "dp" / f"{name}_conf.png"), np.full(d.shape, 255, np.uint8))
        a = cv2.resize(tgt, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        _w16(root / split / "anc_b" / f"{name}.png", a)
        _w16(root / split / "anc_d" / f"{name}.png", cv2.GaussianBlur(a, (0, 0), 0.7))


@pytest.fixture(scope="module")
def caches(tmp_path_factory):
    import build_train_cache
    root = tmp_path_factory.mktemp("data")
    out = {}
    for split, n in (("train", 3), ("val", 2)):
        _make_split(root, split, n)
        r = root / split
        man = build_train_cache.main(["--split", split, "--inputs", str(r / "inputs"), "--targets", str(r / "targets"),
                                      "--masks", str(r / "masks"), "--dp", str(r / "dp"), "--features", "pixels",
                                      "--anchors", "bokehlicious_deblur,drbnet_single",
                                      "--anchor-dir", f"bokehlicious_deblur={r / 'anc_b'}",
                                      "--anchor-dir", f"drbnet_single={r / 'anc_d'}",
                                      "--out", str(root / "cache" / split), "--gpus", "cpu", "--scales", "1,0.5",
                                      "--topk", "8", "--k-eval", "4"])
        out[split] = (root / "cache" / split, man)
    return out


def test_anchor_lock_exact():
    z, a = torch.rand(2, 3, 64, 64), torch.rand(2, 3, 16, 16)
    assert (down4(anchor_lock(z, a)) - a).abs().max() < 1e-5
    from uhdd.net.ours import up4_exact
    delta = up4_exact(a) + (z - up4_exact(down4(z)))      # anchor's low band + an arbitrary high band (down4 = 0)
    assert (anchor_lock(delta, a) - delta).abs().max() < 1e-5   # passes through unchanged


def test_retrieval_finds_same_texture():
    tgt, _, d = _scene(0)
    a = torch.from_numpy(cv2.resize(tgt, (W // 4, H // 4), interpolation=cv2.INTER_AREA)).permute(2, 0, 1)[None]
    f = mem_mod.Features("pixels")
    mem = mem_mod.build_memory(a, torch.from_numpy(d)[None, None], f, scales=(1.0,), tex_pct=0)
    idx, sc = mem_mod.retrieve(mem, k=4)
    assert len(mem.key_box) > 0 and (mem.key_box[:, 0] + mem.key_box[:, 2] <= H // 2 + 8).all()   # keys in focus
    gh, gw = mem.grid
    q = np.arange(gh * gw).reshape(gh, gw)[gh // 2 + 1:].ravel()                 # defocused tokens (texture A)
    top = mem.key_box[idx[q, 0]]
    in_a = (top[:, 1] + top[:, 2] / 2) < W // 2                                   # key centre in the A half
    assert in_a.mean() > 0.8


def test_tile_exemplars_drop_and_map():
    key_box = np.array([[0, 0, 32], [0, 200, 32], [100, 0, 32]], np.int32)
    idx = np.tile(np.array([[0, 1, 2]], np.int32), (6, 1))
    sc = np.tile(np.array([[0.9, 0.8, 0.7]], np.float16), (6, 1))
    keys, tmap, ts = mem_mod.tile_exemplars(idx, sc, (2, 3), 14, key_box, (0, 0, 28), m=4, k=3, drop=(0, 0, 64, 64))
    assert 0 not in keys and set(keys[keys >= 0]) == {1, 2}
    assert tmap.shape == (28, 28) and tmap.max() < ts.shape[0] and np.isinf(ts[:, -1]).all()


def test_simulated_anchor_blurs_with_disparity():
    g = torch.Generator().manual_seed(0)
    t = torch.rand(1, 3, 32, 64)
    d = torch.zeros(1, 1, 32, 64)
    d[..., 32:] = 4.0
    a = simulate(t, d, g, {"blur_floor": [0, 0], "blur_per_disp": [0.5, 0.5], "ringing": [0, 0], "noise": [0, 0]})
    err_l, err_r = (a - t)[..., 4:28, 4:28].abs().mean(), (a - t)[..., 4:28, 36:60].abs().mean()
    assert err_l < 1e-4 < err_r


def test_cache_manifest(caches):
    d, man = caches["train"]
    assert man["n"] == 3 and not man["missing"]
    assert abs(sum(man["m1_blur_bins_b0_b3"]) - 1) < 1e-3 and sum(man["m1_blur_bins_b0_b3"][2:]) > 0.3
    assert man["m2"]["keys"] > 0 and man["m2"]["coverage"] > 0.5
    s = d / "train0"
    assert np.load(s / "x.npy", mmap_mode="r").shape == (H, W, 3) and np.load(s / "blur.npy").shape == (H // 4, W // 4)


def test_dataset_sample_shapes(caches):
    from uhdd.train.data import TileDataset
    ds = TileDataset({"cache": str(caches["train"][0]), "tile": 128, "m": 8, "k": 4, "seed": 1,
                      "anchors": {"bokehlicious_deblur": 0.5, "sim": 0.5}})
    it = iter(ds)
    for _ in range(4):
        b = next(it)
        assert b["a"].shape == (3, 32, 32) and b["x"].shape == (3, 128, 128) and b["d"].shape == (2, 32, 32)
        assert b["ex"].shape == (8, 3, 64, 64) and b["tok_map"].shape == (32, 32)
        assert b["tok_map"].max() < b["tok_scores"].shape[0]


def test_train_then_infer(caches, tmp_path):
    import train
    from uhdd import models
    from uhdd.tiling import TileSpec, run_tiled
    cfg = str(Path(__file__).resolve().parents[1] / "configs" / "train" / "tiny.yaml")
    out = train.main(["--config", cfg, "--out", str(tmp_path / "run"), "--cpu",
                      "--set", f"data.cache={caches['train'][0]}", f"val.cache={caches['val'][0]}"])
    assert (out / "model_ema.pt").exists() and (out / "log.csv").exists() and (out / "val.csv").exists()
    # resume: a second call with more steps continues from last.pt
    out = train.main(["--config", cfg, "--out", str(tmp_path / "run"), "--cpu",
                      "--set", f"data.cache={caches['train'][0]}", f"val.cache={caches['val'][0]}", "optim.steps=4"])
    assert sum(1 for _ in open(out / "val.csv")) >= 2
    reg = tmp_path / "models.yaml"
    reg.write_text(f"models:\n  ours_tiny:\n    paper_tile: 32\n    factory: uhdd.adapters.ours:build\n"
                   f"    adapter: uhdd.adapters.ours:wrap\n    weights_in_factory: true\n    weights: {out / 'model_ema.pt'}\n"
                   f"    aux: [native, dp]\n    features: pixels\n    scales: [1.0, 0.5]\n    m: 8\n    k: 4\n    scale: 4\n    multiple: 16\n")
    model = models.build("ours_tiny", reg, "cpu")
    r = Path(caches["val"][0]).parents[1] / "val"
    model.fn.set_image("val0", {"native": str(r / "inputs"), "dp": str(r / "dp")})
    a = torch.from_numpy(cv2.imread(str(r / "anc_b" / "val0.png"), cv2.IMREAD_UNCHANGED)[..., ::-1].astype(np.float32) / 65535)
    a = a.permute(2, 0, 1)[None].contiguous()
    with torch.no_grad():
        y, info = run_tiled(model, a, TileSpec(32, 4, 2, "linear", 0), model.multiple, model.scale)
    assert y.shape == (1, 3, H, W) and info["n_tiles"] > 1 and torch.isfinite(y).all()
    assert (down4(y) - a).abs().mean() < 1e-4 and (down4(y) - a).abs().max() < 5e-3   # lock holds on the blended image


def test_model_without_exemplars_and_input():
    m = build_model({"backbone": "plain", "width": 16, "blocks": 1, "exemplars": False, "native_input": False,
                     "anchor_lock": False, "cond_width": 8, "cond_blocks": 1, "refine_width": 8, "refine_blocks": 1})
    y = m(torch.rand(1, 3, 16, 16), torch.rand(1, 3, 64, 64), torch.rand(1, 2, 16, 16))
    assert y.shape == (1, 3, 64, 64)


def test_sft_starts_as_identity_and_reaches_every_stage():
    from uhdd.net.ours import modulated_features
    m = build_model({"backbone": "plain", "width": 16, "blocks": 3, "cond_width": 8, "cond_blocks": 1,
                     "refine_width": 8, "refine_blocks": 1, "exemplars": False})
    assert len(m.cond.sft) == 3
    f = torch.rand(1, 16, 16, 16)
    c = m.cond(torch.rand(1, 3, 64, 64), torch.rand(1, 2, 16, 16))
    assert torch.allclose(modulated_features(m.sr, f, c, m.cond.sft), m.sr.forward_features(f))   # zero-init
    torch.nn.init.normal_(m.cond.sft[1].net[-1].weight, std=0.1)
    assert not torch.allclose(modulated_features(m.sr, f, c, m.cond.sft), m.sr.forward_features(f))


def test_sft_on_hat_matches_forward_features():
    import os
    repo = Path(os.environ.get("UHDD_REPOS", "/nonexistent")) / "HAT"
    if not (repo / "hat" / "archs" / "hat_arch.py").exists():
        pytest.skip("HAT repo not available ($UHDD_REPOS/HAT)")
    from uhdd.adapters import hat
    from uhdd.net.ours import SFT, modulated_features
    sr = hat.build({"repo": str(repo), "kwargs": {"upscale": 4, "in_chans": 3, "img_size": 16, "window_size": 8,
                    "compress_ratio": 3, "squeeze_factor": 4, "conv_scale": 0.01, "overlap_ratio": 0.5, "img_range": 1.0,
                    "depths": [1, 1], "embed_dim": 24, "num_heads": [2, 2], "mlp_ratio": 2, "upsampler": "pixelshuffle",
                    "resi_connection": "1conv"}}).eval()
    f = torch.rand(1, 24, 16, 16)
    sfts = torch.nn.ModuleList([SFT(8, 24) for _ in range(2)])
    with torch.no_grad():
        assert torch.allclose(modulated_features(sr, f, torch.rand(1, 8, 16, 16), sfts), sr.forward_features(f), atol=1e-5)
