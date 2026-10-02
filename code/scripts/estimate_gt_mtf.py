"""Measure the real optical blur of the DPDD captures with the slanted-edge method on raw Bayer data.

Question answered: does the f/22 "sharp" target follow the diffraction-limited (Airy) model assumed by the
PSF-matched metrics (uhdd/metrics/optics.py)? Edges are found automatically in the CR2 raw (green
photosites only, linear, no demosaicing), their sub-pixel position/angle is fitted, an oversampled edge
spread function (ESF) is built from the irregular samples, and MTF = |FFT(d ESF/dx)|.

Defocused edges have low MTF, so per image we keep the sharpest edges (envelope) and report their mean
curve. Theory: diffraction (circular aperture, lambda, N, pixel pitch) x pixel-aperture sinc.

  python estimate_gt_mtf.py --zip cr2.zip --names a,b,c --out mtf.json [--procs 8]
  # focal-plane only (relative f/22 vs f/4 blur): masks from dp_maps.py on the official DP views;
  # for f/22 targets pass the paired f/4 stems so both use the same focal-plane regions
  python estimate_gt_mtf.py --zip cr2.zip --names <f4 stems> --focus-dir DP --out f4_focus.json
  python estimate_gt_mtf.py --zip cr2.zip --names <f22 stems> --focus-of <f4 stems> --focus-dir DP --out f22_focus.json
"""
import argparse
import json
import math
import os
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage

FREQS = np.round(np.arange(0.02, 0.62, 0.02), 2)  # cycles / pixel (sensor Nyquist = 0.5)
OVS = 4                                           # ESF bins per pixel
HALF = 16                                         # ESF half-width in pixels


def theory_mtf(f, wavelength_um=0.55, fnum=22.0, pitch_um=36000 / 6720):
    """Diffraction-limited circular-aperture MTF x pixel-aperture sinc, f in cycles/pixel."""
    fc = pitch_um / (wavelength_um * fnum)
    x = np.clip(np.asarray(f, float) / fc, 0, 1)
    diff = 2 / np.pi * (np.arccos(x) - x * np.sqrt(1 - x ** 2))
    return diff * np.abs(np.sinc(np.asarray(f, float)))


def edge_mtf(xs, ys, vals, ovs=OVS, half=HALF):
    """MTF from irregular samples around a near-vertical edge.

    xs, ys, vals: 1-D arrays of sample coordinates (pixels) and linear values. Returns a dict with the
    MTF at FREQS, MTF50, edge contrast and fitted angle, or None if the edge is unusable.
    """
    xs, ys, vals = (np.asarray(a, float) for a in (xs, ys, vals))
    rows = np.unique(ys)
    ex, ey = [], []
    for y in rows:  # per-row edge position = centroid of |derivative| along the row
        m = ys == y
        order = np.argsort(xs[m])
        x, v = xs[m][order], vals[m][order]
        if len(x) < 8:
            continue
        d = np.abs(np.diff(v))
        j = int(np.argmax(np.convolve(d, np.ones(3), "same")))
        lo_j, hi_j = max(0, j - 3), min(len(d), j + 4)   # centroid only near the steepest step (noise elsewhere biases it)
        dd = d[lo_j:hi_j]
        if dd.sum() < 1e-3:
            continue
        ex.append((x[lo_j:hi_j] + x[lo_j + 1:hi_j + 1]) @ dd / (2 * dd.sum()))
        ey.append(y)
    if len(ex) < 16:
        return None
    ex, ey = np.array(ex), np.array(ey)
    keep = np.ones(len(ex), bool)
    for _ in range(3):  # robust line fit x = a*y + b
        a, b = np.polyfit(ey[keep], ex[keep], 1)
        r = np.abs(ex - (a * ey + b))
        keep = r < max(1.0, 2.5 * np.median(r[keep]))
        if keep.sum() < 12:
            return None
    cos_t = 1 / math.sqrt(1 + a * a)
    d = (xs - (a * ys + b)) * cos_t                       # signed distance to the edge line
    sel = np.abs(d) < half
    bins = np.floor((d[sel] + half) * ovs).astype(int)
    n = 2 * half * ovs
    cnt = np.bincount(bins, minlength=n)[:n]
    if (cnt < 3).any():
        return None
    esf = np.bincount(bins, weights=vals[sel], minlength=n)[:n] / cnt
    lo, hi = esf[: 4 * ovs].mean(), esf[-4 * ovs:].mean()
    contrast = abs(hi - lo)
    plateau = max(esf[: 4 * ovs].std(), esf[-4 * ovs:].std())
    if contrast < 0.04 or plateau > 0.05 * contrast:
        return None
    esf = (esf - lo) / (hi - lo)
    lsf = np.diff(esf)
    c = int(np.argmax(np.abs(lsf)))
    w = 12 * ovs
    if c - w < 0 or c + w > len(lsf):
        return None
    seg = lsf[c - w: c + w] * np.hanning(2 * w)
    if seg.sum() == 0:
        return None
    nfft = 2048
    mtf = np.abs(np.fft.rfft(seg, nfft))
    f = np.fft.rfftfreq(nfft, 1 / ovs)                    # cycles / pixel
    mtf = mtf / mtf[0] / np.abs(np.sinc(f / ovs))         # normalize, undo the bin-box smoothing
    curve = np.interp(FREQS, f, mtf)
    if curve[FREQS <= 0.3].max() > 1.05:                  # overshoot = sloped/textured background, not blur
        return None
    below = np.where(mtf < 0.5)[0]
    mtf50 = float(f[below[0]]) if len(below) else float("nan")
    return {"mtf": curve.tolist(), "mtf50": mtf50, "contrast": float(contrast),
            "angle_deg": math.degrees(math.atan(a))}


def _green_halfres(raw, mask):
    s = (raw * mask).reshape(raw.shape[0] // 2, 2, raw.shape[1] // 2, 2).sum((1, 3))
    c = mask.reshape(raw.shape[0] // 2, 2, raw.shape[1] // 2, 2).sum((1, 3))
    return s / np.maximum(c, 1)


def find_edges(g, k=150, win=24, min_sep=40):
    """Candidate straight, unclipped, slightly slanted edges on the half-res green image.
    Returns [(row, col, transpose)] in half-res coordinates (edge made near-vertical by `transpose`)."""
    gs = ndimage.gaussian_filter(g, 1.0)
    gy, gx = np.gradient(gs)
    box = lambda a: ndimage.uniform_filter(a, win)
    Jxx, Jyy, Jxy = box(gx * gx), box(gy * gy), box(gx * gy)
    tr = Jxx + Jyy
    coh = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / np.maximum(tr, 1e-12)
    phi = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)            # gradient direction
    dev_x = np.degrees(np.abs(phi))                       # deviation from the x axis (edge near vertical)
    dev_y = 90 - dev_x
    rng = ndimage.maximum_filter(gs, win) - ndimage.minimum_filter(gs, win)
    ok = (coh > 0.9) & (rng > 0.04) & (ndimage.maximum_filter(gs, win) < 0.92)
    slant = np.where(dev_x < dev_y, dev_x, dev_y)
    ok &= (slant > 2.5) & (slant < 20)
    ok[:win * 2] = ok[-win * 2:] = False
    ok[:, :win * 2] = ok[:, -win * 2:] = False
    rr, cc = np.nonzero(ok)
    order = np.argsort(-rng[rr, cc])
    picked, taken = [], set()
    for i in order:
        r, c = rr[i], cc[i]
        key = (r // min_sep, c // min_sep)
        if key in taken:
            continue
        taken.add(key)
        picked.append((int(r), int(c), bool(dev_y[r, c] < dev_x[r, c])))
        if len(picked) >= k:
            break
    return picked


def load_focus(focus_dir, stem):
    """Focal-plane mask from dp_maps.py (1680x1120, DP views of the f/4 capture) or None."""
    if not focus_dir:
        return None
    import cv2
    p = os.path.join(focus_dir, f"{stem}_focus.png")
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 127


def in_focus(mask, raw_y, raw_x, crop=(12, 12), f=4, r=2):
    """Raw-sensor pixel -> native (minus develop_raw.py CROP offset) -> DP grid (/4); any focus within r."""
    y, x = (raw_y - crop[0]) // f, (raw_x - crop[1]) // f
    if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
        return False
    return bool(mask[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].any())


def analyse_cr2(path, k=150, roi=96, focus=None):
    import rawpy
    with rawpy.imread(path) as r:
        raw = r.raw_image_visible.astype(np.float32)
        colors = r.raw_colors_visible
        black = np.array(r.black_level_per_channel, np.float32)[colors]
        white = float(r.white_level)
    raw = np.clip((raw - black) / (white - black), 0, 1)
    mask = ((colors == 1) | (colors == 3)).astype(np.float32)
    H, W = raw.shape
    g = _green_halfres(raw[: H // 2 * 2, : W // 2 * 2], mask[: H // 2 * 2, : W // 2 * 2])
    res = []
    for r0, c0, tr in find_edges(g, k if focus is None else 4 * k):
        if focus is not None and not in_focus(focus, 2 * r0, 2 * c0):
            continue
        y0, x0 = max(0, 2 * r0 - roi // 2), max(0, 2 * c0 - roi // 2)
        sub = raw[y0:y0 + roi, x0:x0 + roi]
        m = mask[y0:y0 + roi, x0:x0 + roi] > 0
        if sub.shape != (roi, roi):
            continue
        yy, xx = np.nonzero(m)
        v = sub[yy, xx]
        if tr:
            xx, yy = yy, xx                                  # transpose so the edge is near-vertical
        out = edge_mtf(xx.astype(float), yy.astype(float), v)
        if out:
            out.update(row=2 * r0, col=2 * c0, transposed=tr)
            res.append(out)
    return res


def _job(args):
    zf, stem, k, focus_dir, focus_stem = args
    focus = load_focus(focus_dir, focus_stem or stem)
    if os.path.isdir(zf):                                  # --zip may also be a folder of extracted CR2s
        return stem, analyse_cr2(os.path.join(zf, f"{stem}.CR2"), k, focus=focus)
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zf) as z:
            p = z.extract(f"CR2/{stem}.CR2", tmp)
        return stem, analyse_cr2(p, k, focus=focus)


def summarize(edges_per_image, top_frac=0.15):
    """Envelope curve: per image the sharpest `top_frac` edges (by MTF50), then the mean over images."""
    curves, m50 = [], []
    for edges in edges_per_image:
        if len(edges) < 5:
            continue
        e = sorted(edges, key=lambda d: -d["mtf50"] if d["mtf50"] == d["mtf50"] else 0)
        top = e[: max(3, int(len(e) * top_frac))]
        curves.append(np.mean([t["mtf"] for t in top], 0))
        m50.append(np.mean([t["mtf50"] for t in top]))
    return np.mean(curves, 0), float(np.mean(m50)), len(curves)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="CR2 zip (members CR2/<stem>.CR2) or a folder of <stem>.CR2")
    ap.add_argument("--names", required=True, help="comma-separated CR2 stems")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=150, help="max candidate edges per image")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--focus-dir", help="dp_maps.py output: keep only edges on the f/4 focal plane")
    ap.add_argument("--focus-of", help="comma-separated stems whose focus mask applies to --names, in order "
                                       "(e.g. the f/4 stems when --names are the f/22 targets)")
    a = ap.parse_args()
    stems = a.names.split(",")
    fo = a.focus_of.split(",") if a.focus_of else [None] * len(stems)
    assert len(fo) == len(stems), "--focus-of must list one stem per --names entry"
    with ProcessPoolExecutor(a.procs) as ex:
        res = dict(ex.map(_job, [(a.zip, s, a.k, a.focus_dir, f) for s, f in zip(stems, fo)]))
    json.dump(res, open(a.out, "w"))
    curve, m50, n = summarize(list(res.values()))
    print(f"{n} images, edges/image median {int(np.median([len(v) for v in res.values()]))}, envelope MTF50 {m50:.3f} c/px")
    print("freq(c/px)  measured  theory_f22  theory_f4")
    for f, mv in zip(FREQS, curve):
        if abs(f * 50 - round(f * 50)) < 1e-6 and round(f * 50) % 5 == 0 or f in (0.1, 0.3, 0.44, 0.5):
            print(f"{f:8.2f}  {mv:8.3f}  {theory_mtf(f):10.3f}  {theory_mtf(f, fnum=4.0):9.3f}")


if __name__ == "__main__":
    main()
