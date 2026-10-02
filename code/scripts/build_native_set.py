#!/usr/bin/env python
"""Build the native-resolution DPDD test set (x1, x2, x4) from raw CR2, tightly calibrated to the
official 1680x1120 release.

  build_native_set.py --raw DIR_OR_ZIP --official-inputs DIR --official-targets DIR --out OUT [--procs 5]

Official dirs hold the 1680x1120 PNGs *paired by name*: official-targets/<input stem>.png is the target
of official-inputs/<input stem>.png (e.g. the symlink layout made outside: dataset/dpdd_1680/test/).
The target CR2 stem is read from the symlink name if it is one, else --pairs CSV (input_stem,target_stem).

Per capture (inputs and targets independently):
 1. libraw development: camera white balance, linear 16-bit, crop (12, 12, 4480, 6720) of libraw's
    4502x6744 output. Verified outside: this crop matches the official framing (best integer offset for
    26/37 images, median sub-pixel residual 0.07 native px).
 2. Colour/tone calibration to the official rendering of the same capture: a 19-feature polynomial
    (develop_raw.features) fitted on the 4x area-downscaled linear image vs. the official PNG, applied at
    native resolution; an unsharp mask (sigma, amount) per capture for the converter's sharpening,
    chosen on held-out-split pixels (two captures prefer none); and a smooth additive residual field
    (Gaussian sigma 16 px at 1680, i.e. 64 px native) for the converter's local tone processing.
    Result on the 37 indoor test pairs, 4x area downscale vs. official, held-out pixels: inputs median
    38.8 dB (worst 33.9), targets median 39.0 dB (worst 30.7, fine canvas texture); without sharpening
    37.6 / 27.0 dB for targets; the global map of develop_raw.py gave 28.7 dB. Sharpening fitted for
    28/37 targets (mostly sigma 1.5 px, amount 1.5-2.5) and 12/37 inputs. Task check: blurry-vs-sharp
    PSNR at 1680x1120 is 26.74 dB on our x4 (registered, masked) vs. 26.41 dB on the official files
    (paired +0.33 dB, sd 0.59) -> the x4 level reproduces the standard benchmark.
    Noise reduction and demosaicing differences of the official converter are not reproduced.
 3. Targets registered to inputs at native resolution (translation ECC, uhdd.registration), masks.
 4. x2 / x4 by area downscaling of inputs, registered targets and masks (grid-consistent with x1).
    Note: the official 1680x1120 filter is undetermined between area and bilinear (< 1 dB, mixed).

Outputs: OUT/inputs, OUT/x1/{targets,masks}, OUT/x2/..., OUT/x4/..., OUT/build_report.csv (per image:
calibration PSNR of inputs/targets vs. official at 1/4, fitted sharpening, registration shift/ECC).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from develop_raw import apply_map, features  # noqa: E402

CROP = (12, 12, 4480, 6720)
LF_SIGMA = 16          # px at 1680x1120
USM_SIGMAS = (1.0, 1.5, 2.0, 2.5, 3.0)                    # native px
USM_AMOUNTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)


def develop_linear(path: str) -> np.ndarray:
    import rawpy
    with rawpy.imread(path) as r:
        x = r.postprocess(output_bps=16, use_camera_wb=True, no_auto_bright=True, gamma=(1, 1))
    y0, x0, h, w = CROP
    return x[y0:y0 + h, x0:x0 + w].astype(np.float32) / 65535


def read_rgb(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return img[:, :, ::-1].astype(np.float32) / (65535.0 if img.dtype == np.uint16 else 255.0)


def calibrate(lin: np.ndarray, official: np.ndarray) -> tuple[np.ndarray, float, dict]:
    """Native linear -> native rendering matching `official` (1/4 res).

    Returns (image, held-out PSNR at 1/4 vs official, fitted parameters)."""
    h, w = official.shape[:2]
    lo = cv2.resize(lin, (w, h), interpolation=cv2.INTER_AREA)
    X, Y = lo.reshape(-1, 3), official.reshape(-1, 3)
    idx = np.random.default_rng(0).permutation(len(X))
    fit, ev = idx[::2], idx[1::2]
    W, *_ = np.linalg.lstsq(features(X[fit]).astype(np.float64), Y[fit].astype(np.float64), rcond=None)
    nat = apply_map(lin, W)                                     # tone at native resolution
    down = lambda z: cv2.resize(z, (w, h), interpolation=cv2.INTER_AREA)
    d0 = down(nat)

    # sharpening of the official converter: unsharp mask y = x + k (x - G_s x) at native resolution,
    # (s, k) chosen on the fit pixels; D(USM(x)) = D(x) + k (D(x) - D(G_s x)) needs one blur per s
    def score(d, sel):
        d = d + cv2.GaussianBlur(official - d, (0, 0), LF_SIGMA)
        r = np.clip(d, 0, 1).reshape(-1, 3)[sel] - Y[sel]
        return float(10 * np.log10(1 / np.mean(r ** 2)))
    best = (score(d0, fit), 0.0, 0.0)
    for sg in USM_SIGMAS:
        db = down(cv2.GaussianBlur(nat, (0, 0), sg))
        for k in USM_AMOUNTS[1:]:
            sc = score(d0 + k * (d0 - db), fit)
            if sc > best[0]:
                best = (sc, sg, k)
    _, sg, k = best
    if k > 0:
        nat += k * (nat - cv2.GaussianBlur(nat, (0, 0), sg))
    d = down(nat)
    field = cv2.GaussianBlur(official - d, (0, 0), LF_SIGMA)    # smooth local-tone residual (1/4 res)
    nat += cv2.resize(field, (lin.shape[1], lin.shape[0]), interpolation=cv2.INTER_LINEAR)
    np.clip(nat, 0, 1, out=nat)
    check = down(nat).reshape(-1, 3)[ev]
    psnr = float(10 * np.log10(1 / np.mean((check - Y[ev]) ** 2)))
    return nat, psnr, {"usm_sigma": sg, "usm_amount": k}


def to16(x: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(x, 0, 1) * 65535).astype(np.uint16)


def write(path: Path, rgb: np.ndarray | None = None, gray: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = rgb[:, :, ::-1] if rgb is not None else gray
    if not cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise IOError(path)


def raw_path(src: str, stem: str, tmp: str) -> str:
    if os.path.isdir(src):
        return os.path.join(src, f"{stem}.CR2")
    with zipfile.ZipFile(src) as z:
        return z.extract(f"CR2/{stem}.CR2", tmp)


def job(args) -> dict:
    s_in, s_tg, a = args
    cv2.setNumThreads(2)
    from uhdd.registration import RegConfig, register_pair
    out = Path(a.out)
    with tempfile.TemporaryDirectory() as tmp:
        x, p_in, q_in = calibrate(develop_linear(raw_path(a.raw, s_in, tmp)), read_rgb(Path(a.official_inputs) / f"{s_in}.png"))
        y, p_tg, q_tg = calibrate(develop_linear(raw_path(a.raw, s_tg, tmp)), read_rgb(Path(a.official_targets) / f"{s_in}.png"))
    x16, y16 = to16(x), to16(y)
    del x, y
    reg = register_pair(x16, y16, RegConfig(motion="translation"))
    write(out / "inputs" / f"{s_in}.png", x16)
    write(out / "x1" / "targets" / f"{s_in}.png", reg["aligned"])
    write(out / "x1" / "masks" / f"{s_in}.png", gray=reg["mask"])
    for f in (2, 4):
        size = (x16.shape[1] // f, x16.shape[0] // f)
        write(out / f"x{f}" / "inputs" / f"{s_in}.png", cv2.resize(x16, size, interpolation=cv2.INTER_AREA))
        write(out / f"x{f}" / "targets" / f"{s_in}.png", cv2.resize(reg["aligned"], size, interpolation=cv2.INTER_AREA))
        m = reg["mask"][: size[1] * f, : size[0] * f].reshape(size[1], f, size[0], f).min(axis=(1, 3))
        write(out / f"x{f}" / "masks" / f"{s_in}.png", gray=m)
    return {"name": s_in, "target_cr2": s_tg, "calib_psnr_input": round(p_in, 2), "calib_psnr_target": round(p_tg, 2),
            "usm_input": f"{q_in['usm_sigma']}/{q_in['usm_amount']}", "usm_target": f"{q_tg['usm_sigma']}/{q_tg['usm_amount']}",
            "reg_shift_px": round(reg["corner_shift_px"], 2), "reg_ecc": round(reg["ecc"], 4), "warped": int(reg["warped"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="folder of <stem>.CR2 or the DPDD CR2 zip")
    ap.add_argument("--official-inputs", required=True)
    ap.add_argument("--official-targets", required=True)
    ap.add_argument("--pairs", help="CSV input_stem,target_stem (else from target symlink names)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--only", help="comma-separated input stems")
    ap.add_argument("--overwrite", action="store_true", help="rebuild pairs that already exist")
    a = ap.parse_args()

    pairs = {}
    if a.pairs:
        with open(a.pairs) as f:
            pairs = dict(r[:2] for r in csv.reader(f) if r)
    else:
        for p in Path(a.official_targets).glob("*.png"):
            if p.is_symlink():
                pairs[p.stem] = Path(os.readlink(p)).stem
    have = ({p.stem for p in Path(a.raw).glob("*.CR2")} if os.path.isdir(a.raw)
            else {Path(n).stem for n in zipfile.ZipFile(a.raw).namelist() if n.endswith(".CR2")})
    todo = sorted(s for s, t in pairs.items() if s in have and t in have)
    if a.only:
        todo = [s for s in todo if s in set(a.only.split(","))]
    if not a.overwrite:
        todo = [s for s in todo if not (Path(a.out) / "x4" / "masks" / f"{s}.png").exists()]
    print(f"{len(todo)} pairs to build", flush=True)
    rows = []
    with ProcessPoolExecutor(a.procs) as ex:
        for r in ex.map(job, [(s, pairs[s], a) for s in todo]):
            rows.append(r)
            print(r, flush=True)
    rep = Path(a.out) / "build_report.csv"
    old = list(csv.DictReader(open(rep))) if rep.exists() else []
    rows = sorted({r["name"]: r for r in old + rows}.values(), key=lambda r: r["name"])
    if rows:
        with open(rep, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), restval="")
            w.writeheader()
            w.writerows(rows)
        ci = np.array([float(r["calib_psnr_input"]) for r in rows]); ct = np.array([float(r["calib_psnr_target"]) for r in rows])
        sh = np.array([float(r["reg_shift_px"]) for r in rows])
        print(f"{len(rows)} pairs; calibration vs official @1/4: inputs median {np.median(ci):.2f} dB (min {ci.min():.2f}), "
              f"targets median {np.median(ct):.2f} dB (min {ct.min():.2f}); registration shift median {np.median(sh):.2f} px, max {sh.max():.2f}")


if __name__ == "__main__":
    main()
