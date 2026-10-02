#!/usr/bin/env python
"""Training-free native-resolution baselines built from a low-res deblur, the blurry input and a blur map.

  fuse_baselines.py --method multiscale --inputs NAT/inputs --x4 RES/drb_x4+bicubic_x4 --x2 RES/drb_x2+bicubic_x2 \
      --dp-maps DP --out RES/drb_multiscale [--params tuned.json] [--procs 4]
  fuse_baselines.py --method multiscale ... --tune --targets NAT/x1/targets --masks NAT/x1/masks --params tuned.json

Methods (docs/baselines.md, group B; b = DP blur level in DP px at 1680x1120, ramp(b; a, c) = clip((c-b)/(c-a), 0, 1)):
  composite   w = ramp(b; t0, t1);  y = w x + (1-w) up4           (copy the input where it is in focus)
  multiscale  w_in = ramp(b; t0, t1), w4 = 1 - ramp(b; m0, m1), w2 = max(0, 1 - w_in - w4);
              y = w_in x + w2 up2 + w4 up4                        (blur-adaptive choice of the deblur scale)
  guided      guided filter (He et al.) of up4 with the native input as guide, per channel, radius r, eps
  detail      y = up4 + HB(x), HB = x - up4(down4(x))             (add the input's own high band everywhere)
  exemplar    composite + (1-w) * in-focus exemplar texture (uhdd.exemplar; needs --anchor4 = the 1/4 deblur)
up4 / up2 = the x4 / x2 deblurred images upsampled to native (any upsampler; we use bicubic).

--tune grid-searches the method's parameters on a split with targets (objective: mean per-image PSNR, masked,
border 64 px) and writes them to --params. Tune on **val**, never on test. Without --params, defaults below.
Writes 16-bit PNGs and OUT/meta.json (per-image time_s of the fusion step, excluding the deblur/upsampling).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

DEFAULTS = {
    "composite": {"t0": 0.4, "t1": 1.2},
    # x2 deblurring sees 2x the 1680-px blur; DPDD-trained models cover |disparity| up to ~3.5 DP px at 1680,
    # so x2 is trusted up to b ~ 1.5 and x4 takes over by 2.5 (principled defaults; tune on val)
    "multiscale": {"t0": 0.4, "t1": 1.2, "m0": 1.5, "m1": 2.5},
    "guided": {"r": 8, "eps": 1e-3},
    "detail": {},
    "exemplar": {"t0": 0.4, "t1": 1.2},
}
GRID = {
    "composite": {"t0": [0.2, 0.4, 0.6], "dt": [0.4, 0.8, 1.6]},
    "multiscale": {"t0": [0.2, 0.4, 0.6], "dt": [0.4, 0.8], "m0": [1.0, 1.5, 2.0, 3.0], "dm": [0.5, 1.0, 2.0]},
    "guided": {"r": [4, 8, 16, 32], "eps": [1e-4, 1e-3, 1e-2]},
}


def rd(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(path)
    return img[:, :, ::-1].astype(np.float32) / (65535.0 if img.dtype == np.uint16 else 255.0)


def ramp(b: np.ndarray, a: float, c: float) -> np.ndarray:
    return np.clip((c - b) / max(c - a, 1e-6), 0, 1).astype(np.float32)


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He et al. guided filter, per channel (channel c of src guided by channel c of guide)."""
    k = (2 * r + 1, 2 * r + 1)
    box = lambda z: cv2.boxFilter(z, -1, k, borderType=cv2.BORDER_REFLECT)
    mi, mp = box(guide), box(src)
    a = (box(guide * src) - mi * mp) / (box(guide * guide) - mi * mi + eps)
    b = mp - a * mi
    return box(a) * guide + box(b)


def fuse(method: str, p: dict, x, up4, up2=None, blur=None, anchor4=None, device="cpu") -> tuple[np.ndarray, dict]:
    info = {}
    if method == "composite":
        w = ramp(blur, p["t0"], p["t1"])[..., None]
        y = w * x + (1 - w) * up4
    elif method == "multiscale":
        w_in = ramp(blur, p["t0"], p["t1"])
        w4 = 1 - ramp(blur, p["m0"], p["m1"])
        w2 = np.clip(1 - w_in - w4, 0, 1)
        w4 = 1 - w_in - w2
        y = w_in[..., None] * x + w2[..., None] * up2 + w4[..., None] * up4
        info = {"frac_x2": float(w2.mean()), "frac_x4": float(w4.mean())}
    elif method == "guided":
        y = guided_filter(x, up4, int(p["r"]), float(p["eps"]))
    elif method == "detail":
        from uhdd.exemplar import high_band
        y = up4 + high_band(x)
    elif method == "exemplar":
        from uhdd.exemplar import transfer
        w = ramp(blur, p["t0"], p["t1"])
        tex, info = transfer(x, anchor4, w, "exemplar", device=device)
        y = w[..., None] * x + (1 - w[..., None]) * (up4 + tex[..., None])
    else:
        raise ValueError(method)
    return np.clip(y, 0, 1), info


def masked_psnr(y, gt, m, crop=64) -> float:
    c = slice(crop, -crop)
    se, n = 0.0, 0
    for r0 in range(0, y[c, c].shape[0], 512):
        d = (y[c, c][r0:r0 + 512] - gt[c, c][r0:r0 + 512]).astype(np.float64) ** 2
        mm = m[c, c][r0:r0 + 512]
        se += float(d.sum(-1)[mm].sum())
        n += int(mm.sum()) * 3
    return 10 * np.log10(n / max(se, 1e-30))


def load(a, name):
    from uhdd.dualpixel import load_blur_map
    x = rd(Path(a.inputs) / f"{name}.png")
    d = {"x": x, "up4": rd(Path(a.x4) / f"{name}.png")}
    if a.x2:
        d["up2"] = rd(Path(a.x2) / f"{name}.png")
    if a.dp_maps:
        d["blur"] = load_blur_map(a.dp_maps, name, x.shape[:2])
    if a.anchor4:
        d["anchor4"] = rd(Path(a.anchor4) / f"{name}.png")
    return d


def combos(method):
    g = GRID[method]
    for vals in itertools.product(*g.values()):
        q = dict(zip(g, vals))
        if "dt" in q:
            q["t1"] = q["t0"] + q.pop("dt")
        if "dm" in q:
            q["m1"] = q["m0"] + q.pop("dm")
        yield q


def job_tune(args):
    a, name = args
    cv2.setNumThreads(2)
    d = load(a, name)
    gt = rd(Path(a.targets) / f"{name}.png")
    m = cv2.imread(str(Path(a.masks) / f"{name}.png"), cv2.IMREAD_GRAYSCALE) > 127
    return name, [masked_psnr(fuse(a.method, q, **d)[0], gt, m) for q in combos(a.method)]


def job_run(args):
    a, name, p = args
    cv2.setNumThreads(2)
    import torch
    torch.set_num_threads(2)
    from uhdd.io import write_image
    d = load(a, name)
    t = time.time()
    y, info = fuse(a.method, p, **d, device=a.device)
    dt = time.time() - t
    write_image(Path(a.out) / f"{name}.png", np.rint(y * 65535).astype(np.uint16))
    return name, {"time_s": dt, **info}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=list(DEFAULTS))
    ap.add_argument("--inputs", required=True, help="blurry native inputs")
    ap.add_argument("--x4", required=True, help="x4 deblur upsampled to native")
    ap.add_argument("--x2", help="x2 deblur upsampled to native (multiscale)")
    ap.add_argument("--anchor4", help="x4 deblur at 1/4 res (exemplar)")
    ap.add_argument("--dp-maps")
    ap.add_argument("--out")
    ap.add_argument("--params", help="JSON with parameters (read; written by --tune)")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--targets")
    ap.add_argument("--masks")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--device", default="cpu", help="matching device for 'exemplar'")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", help="comma-separated image names")
    a = ap.parse_args()
    names = sorted(p.stem for p in Path(a.x4).glob("*.png"))[: a.limit or None]
    if a.only:
        names = [n for n in names if n in set(a.only.split(","))]

    if a.tune:
        qs = list(combos(a.method))
        with ProcessPoolExecutor(a.procs) as ex:
            res = dict(ex.map(job_tune, [(a, n) for n in names]))
        mean = np.mean([res[n] for n in names], axis=0)
        best = int(np.argmax(mean))
        for q, v in sorted(zip(qs, mean), key=lambda t: -t[1])[:5]:
            print(f"{v:.3f} dB  {q}")
        json.dump({"method": a.method, "params": qs[best], "psnr": float(mean[best]), "n": len(names)},
                  open(a.params, "w"), indent=1)
        return

    p = dict(DEFAULTS[a.method])
    if a.params:
        p.update(json.load(open(a.params))["params"])
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.procs) as ex:
        images = dict(ex.map(job_run, [(a, n, p) for n in names]))
    ts = [v["time_s"] for v in images.values()]
    meta = {"images": images, "summary": {"method": a.method, "params": p, "time_s_mean": float(np.mean(ts))}}
    (Path(a.out) / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(images)} images -> {a.out}  params {p}  {np.mean(ts):.1f} s/img")


if __name__ == "__main__":
    main()
