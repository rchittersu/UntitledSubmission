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
