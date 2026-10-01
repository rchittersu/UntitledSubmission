import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import erf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import estimate_gt_mtf as m  # noqa: E402


def _edge(sigma, angle_deg=6.0, size=96, seed=0):
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    g = (xx + yy) % 2 == 0                                   # checkerboard = green photosites
    d = (xx - size / 2 - math.tan(math.radians(angle_deg)) * (yy - size / 2)) * math.cos(math.radians(angle_deg))
    v = 0.15 + 0.6 * 0.5 * (1 + erf(d / (sigma * math.sqrt(2))))
    v += np.random.default_rng(seed).normal(0, 0.002, v.shape)
    return xx[g], yy[g], v[g]


def test_gaussian_edge_mtf_matches_analytic():
    for sigma in (0.8, 1.5):
        out = m.edge_mtf(*_edge(sigma))
        assert out is not None
        ref = np.exp(-2 * math.pi ** 2 * sigma ** 2 * m.FREQS ** 2)
        sel = ref > 0.05
        assert np.max(np.abs(np.array(out["mtf"])[sel] - ref[sel])) < 0.05
        assert abs(out["angle_deg"] - 6.0) < 0.5


def test_flat_or_low_contrast_rejected():
    x, y, v = _edge(1.0)
    assert m.edge_mtf(x, y, np.full_like(v, 0.3)) is None
    assert m.edge_mtf(x, y, 0.3 + 0.02 * (v - v.min()) / (v.max() - v.min())) is None


def test_theory_mtf_shape():
    assert abs(m.theory_mtf(0.0) - 1) < 1e-9
    assert m.theory_mtf(0.5) < 1e-9                         # f/22 diffraction cutoff 0.44 c/px < Nyquist
    assert m.theory_mtf(0.5, fnum=4.0) > 0.4                # f/4 has no diffraction limit below Nyquist
