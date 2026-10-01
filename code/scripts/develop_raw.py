"""Develop DPDD Canon CR2 raws to native-resolution 16-bit PNGs that match the official release.

The official DPDD PNGs were developed with Canon DPP (not reproducible). We develop with libraw
(linear, camera white balance, centered 6720x4480 crop) and fit ONE global polynomial colour/tone map
to the official 1680x1120 PNGs of the same scenes (train scenes), then apply it unchanged to all scenes.

  fit:      python develop_raw.py fit --zip cr2.zip --official DIR [DIR ...] --out map.json
  evaluate: python develop_raw.py eval --zip cr2.zip --official DIR [DIR ...] --map map.json
  develop:  python develop_raw.py develop --zip cr2.zip --map map.json --out DIR [--names a,b | --names-from DIR]
"""
import argparse
import json
import os
import tempfile
import zipfile
import zlib
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

CROP = (12, 12, 4480, 6720)  # y0, x0, h, w inside libraw's 4502x6744 output (= camera crop margins)


def features(x):
    """x: (N, 3) linear rgb in [0, 1] -> (N, F) polynomial features."""
    cols = [np.ones(len(x), x.dtype)]
    for i in range(3):
        v = x[:, i]
        cols += [v, v * v, v ** 3, np.sqrt(v), np.cbrt(v)]
    cols += [x[:, 0] * x[:, 1], x[:, 1] * x[:, 2], x[:, 0] * x[:, 2]]
    return np.stack(cols, 1)


def apply_map(img, W, chunk_rows=256):
    """img: (H, W, 3) float in [0, 1] linear -> (H, W, 3) float32 in [0, 1]."""
    out = np.empty(img.shape, np.float32)
    for y in range(0, img.shape[0], chunk_rows):
        blk = img[y:y + chunk_rows].reshape(-1, 3).astype(np.float32)
        out[y:y + chunk_rows] = np.clip(features(blk) @ W.astype(np.float32), 0, 1).reshape(-1, img.shape[1], 3)
    return out


def develop_linear(path):
    import rawpy
    with rawpy.imread(path) as r:
        x = r.postprocess(output_bps=16, use_camera_wb=True, no_auto_bright=True, gamma=(1, 1))
    y0, x0, h, w = CROP
    return x[y0:y0 + h, x0:x0 + w]  # uint16 RGB


def lowres(x, f=4):
    h, w = x.shape[:2]
    return cv2.resize(x, (w // f, h // f), interpolation=cv2.INTER_AREA)


def read_png(path):  # -> float64 RGB [0,1]
    return cv2.imread(path, -1)[:, :, ::-1].astype(np.float64) / 65535


def psnr(a, b):
    return float(10 * np.log10(1 / np.mean((a - b) ** 2)))


def _find_official(dirs, stem):
    for d in dirs:
        p = os.path.join(d, stem + ".png")
        if os.path.exists(p):
            return p
    return None


def _extract(zf_path, stem, tmp):
    with zipfile.ZipFile(zf_path) as z:
        return z.extract(f"CR2/{stem}.CR2", tmp)


def _sample(args):
    zf_path, stem, off_path, n = args
    with tempfile.TemporaryDirectory() as tmp:
        lin = lowres(develop_linear(_extract(zf_path, stem, tmp))).astype(np.float64) / 65535
    off = read_png(off_path)
    idx = np.random.default_rng(zlib.crc32(stem.encode())).choice(lin.shape[0] * lin.shape[1], n, replace=False)
    return lin.reshape(-1, 3)[idx], off.reshape(-1, 3)[idx]


def _eval_one(args):
    zf_path, stem, off_path, W = args
    with tempfile.TemporaryDirectory() as tmp:
        lin = lowres(develop_linear(_extract(zf_path, stem, tmp))).astype(np.float64) / 65535
    off = read_png(off_path)
    return stem, psnr(apply_map(lin, W), off), psnr(np.clip(lin ** (1 / 2.2), 0, 1), off)


def cr2_stems(zf_path):
    with zipfile.ZipFile(zf_path) as z:
        return sorted(os.path.basename(n)[:-4] for n in z.namelist() if n.lower().endswith(".cr2"))


def scenes(zf_path, official):
    return [(s, p) for s in cr2_stems(zf_path) if (p := _find_official(official, s))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fit", "eval", "develop"])
    ap.add_argument("--zip", required=True)
    ap.add_argument("--official", nargs="*", default=[], help="dirs of official 1680x1120 PNGs (stem-matched)")
    ap.add_argument("--map", default="map.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--names", default=None, help="comma-separated stems")
    ap.add_argument("--names-from", default=None, help="dir whose PNG stems select the scenes")
    ap.add_argument("--max-scenes", type=int, default=0)
    ap.add_argument("--samples", type=int, default=20000, help="pixels per scene for fitting")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()

    if a.mode == "fit":
        sc = scenes(a.zip, a.official)
        if a.max_scenes:
            sc = sc[:a.max_scenes]
        with ProcessPoolExecutor(a.procs) as ex:
            res = list(ex.map(_sample, [(a.zip, s, p, a.samples) for s, p in sc]))
        X = np.concatenate([r[0] for r in res])
        Y = np.concatenate([r[1] for r in res])
        W, *_ = np.linalg.lstsq(features(X), Y, rcond=None)
        json.dump({"W": W.tolist(), "n_scenes": len(sc), "scenes": [s for s, _ in sc]}, open(a.map, "w"))
        print(f"fitted on {len(sc)} scenes, {len(X)} pixels -> {a.map}")
    elif a.mode == "eval":
        W = np.array(json.load(open(a.map))["W"])
        sc = scenes(a.zip, a.official)
        if a.max_scenes:
            sc = sc[:a.max_scenes]
        with ProcessPoolExecutor(a.procs) as ex:
            res = list(ex.map(_eval_one, [(a.zip, s, p, W) for s, p in sc]))
        for s, pm, pg in res:
            print(f"{s}\tmapped {pm:.2f} dB\tplain gamma2.2 {pg:.2f} dB")
        print(f"mean mapped {np.mean([r[1] for r in res]):.2f} dB (n={len(res)}); plain {np.mean([r[2] for r in res]):.2f} dB")
    else:
        W = np.array(json.load(open(a.map))["W"])
        if a.names:
            stems = a.names.split(",")
        elif a.names_from:
            stems = sorted(f[:-4] for f in os.listdir(a.names_from) if f.endswith(".png"))
        else:
            stems = [s for s, _ in scenes(a.zip, a.official)] if a.official else cr2_stems(a.zip)
        os.makedirs(a.out, exist_ok=True)
        with ProcessPoolExecutor(a.procs) as ex:
            list(ex.map(_develop_one, [(a.zip, s, W, a.out) for s in stems]))
        print(f"developed {len(stems)} scenes -> {a.out}")


def _develop_one(args):
    zf_path, stem, W, out = args
    dst = os.path.join(out, stem + ".png")
    if os.path.exists(dst):
        return
    with tempfile.TemporaryDirectory() as tmp:
        lin = develop_linear(_extract(zf_path, stem, tmp)).astype(np.float32) / 65535
    img = np.round(apply_map(lin, W) * 65535).astype(np.uint16)
    cv2.imwrite(dst + ".part.png", img[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1])
    os.replace(dst + ".part.png", dst)


if __name__ == "__main__":
    main()
