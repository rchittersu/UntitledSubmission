#!/usr/bin/env python
"""Relative blur of the f/22 target vs. the in-focus f/4 input, measured on the SAME edges.

For each pair, slanted edges are detected in the f/4 raw (optionally only on the DP focal plane,
see dp_maps.py); the edge MTF is measured at the same raw location in both captures (they are
aligned to a few pixels, the 96-px ROI tolerates that), and the per-edge ratio MTF_f22 / MTF_f4 is
averaged. The ratio cancels everything both captures share (lens design, OLPF, pixel aperture,
demosaicing-free raw green sampling) and isolates aperture-dependent blur: diffraction at f/22 vs.
aberrations at f/4. A Gaussian sigma (native px) is fitted to the ratio: that is the kernel a
"PSF-matched" metric should apply to predictions of an all-in-focus f/4-like image.

  paired_edge_mtf.py --raw DIR_OR_ZIP --inputs <f4 stems> --targets <f22 stems> [--focus-dir DP] --out x.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import estimate_gt_mtf as em  # noqa: E402


def _raw(path):
    import rawpy
    with rawpy.imread(path) as r:
        raw = r.raw_image_visible.astype(np.float32)
        colors = r.raw_colors_visible
        black = np.array(r.black_level_per_channel, np.float32)[colors]
        white = float(r.white_level)
    raw = np.clip((raw - black) / (white - black), 0, 1)
    return raw, ((colors == 1) | (colors == 3)).astype(np.float32)


def _path(src, stem, tmp):
    if os.path.isdir(src):
        return os.path.join(src, f"{stem}.CR2")
    with zipfile.ZipFile(src) as z:
        return z.extract(f"CR2/{stem}.CR2", tmp)


def _measure(raw, mask, r0, c0, tr, roi=96):
    y0, x0 = max(0, 2 * r0 - roi // 2), max(0, 2 * c0 - roi // 2)
    sub, m = raw[y0:y0 + roi, x0:x0 + roi], mask[y0:y0 + roi, x0:x0 + roi] > 0
    if sub.shape != (roi, roi):
        return None
    yy, xx = np.nonzero(m)
    v = sub[yy, xx]
    if tr:
        xx, yy = yy, xx
    return em.edge_mtf(xx.astype(float), yy.astype(float), v)


def job(args):
    src, s4, s22, focus_dir, k = args
    focus = em.load_focus(focus_dir, s4)
    with tempfile.TemporaryDirectory() as tmp:
        a, ma = _raw(_path(src, s4, tmp))
        b, mb = _raw(_path(src, s22, tmp))
    H, W = a.shape
    g = em._green_halfres(a[: H // 2 * 2, : W // 2 * 2], ma[: H // 2 * 2, : W // 2 * 2])
    out = []
    for r0, c0, tr in em.find_edges(g, k):
        if focus is not None and not em.in_focus(focus, 2 * r0, 2 * c0):
            continue
        e4, e22 = _measure(a, ma, r0, c0, tr), _measure(b, mb, r0, c0, tr)
        if e4 and e22 and abs(e4["angle_deg"] - e22["angle_deg"]) < 2:
            out.append({"mtf4": e4["mtf"], "mtf22": e22["mtf"], "mtf50_4": e4["mtf50"], "mtf50_22": e22["mtf50"]})
    return s4, out


def fit_sigma(f, ratio, fmax=0.3):
    sel = (f >= 0.06) & (f <= fmax) & (ratio > 0.05) & (ratio < 1.5)
    if sel.sum() < 3:
        return float("nan")
    y = -np.log(np.clip(ratio[sel], 1e-3, None))           # = 2 pi^2 sigma^2 f^2
    x = 2 * math.pi ** 2 * f[sel] ** 2
    s2 = float((x @ y) / (x @ x))
    return math.sqrt(max(s2, 0.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--focus-dir")
    ap.add_argument("--k", type=int, default=600)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s4, s22 = a.inputs.split(","), a.targets.split(",")
    with ProcessPoolExecutor(a.procs) as ex:
        res = dict(ex.map(job, [(a.raw, x, y, a.focus_dir, a.k) for x, y in zip(s4, s22)]))
    json.dump(res, open(a.out, "w"))
    edges = [e for v in res.values() for e in v]
    if not edges:
        sys.exit("no paired edges")
    f = em.FREQS
    m4, m22 = np.array([e["mtf4"] for e in edges]), np.array([e["mtf22"] for e in edges])
    ratio = np.median(m22 / np.clip(m4, 1e-3, None), 0)
    th = em.theory_mtf(f) / np.clip(em.theory_mtf(f, fnum=4.0), 1e-6, None)
    n_img = sum(1 for v in res.values() if v)
    sig = fit_sigma(f, ratio)
    # bootstrap over images for the sigma CI
    rng = np.random.default_rng(0)
    keys = [k for k, v in res.items() if v]
    sigs = []
    for _ in range(500):
        ks = rng.choice(keys, len(keys))
        e = [x for k in ks for x in res[k]]
        r = np.median(np.array([x["mtf22"] for x in e]) / np.clip(np.array([x["mtf4"] for x in e]), 1e-3, None), 0)
        sigs.append(fit_sigma(f, r))
    sigs = np.array([s for s in sigs if s == s])
    print(f"{len(edges)} paired edges from {n_img} images; median MTF50 f/4 {np.median([e['mtf50_4'] for e in edges]):.3f}, "
          f"f/22 {np.median([e['mtf50_22'] for e in edges]):.3f} c/px")
    print("freq  median(MTF22/MTF4)  theory Airy22/Airy4")
    for fi, r, t in zip(f, ratio, th):
        if round(fi * 100) % 5 == 0 and fi <= 0.45:
            print(f"{fi:4.2f}  {r:8.3f}            {t:8.3f}")
    print(f"Gaussian fit to the ratio (0.06-0.30 c/px): sigma = {sig:.2f} native px "
          f"[95% CI over images {np.percentile(sigs, 2.5):.2f}, {np.percentile(sigs, 97.5):.2f}]; "
          f"Airy-only ratio fits sigma = {fit_sigma(f, th):.2f}")


if __name__ == "__main__":
    main()
