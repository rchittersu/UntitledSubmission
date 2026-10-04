"""Pilot for the in-focus exemplar memory (plan/method_plan.md, checkpoint 1): non-learned texture transfer.

Question: does texture that is missing in defocused regions exist in the in-focus regions of the same image,
and can it be found by matching in the *anchor domain* (1/4 res, deblurred)?

Per image (native res): base = DP composite (input where in focus, anchor+bicubic elsewhere, as dp_composite.py).
High band of the input that the 1/4 anchor cannot carry: hf = Y - up4(down4(Y)) (luma). Cells of 32 native px,
stride 16. Bank = in-focus cells (composite weight w > 0.9). Each defocused cell gets the high band of one bank
cell, scaled by the ratio of low-res contrasts (clamped to [0.5, 2]), overlap-added with a Hann window and
weighted by (1 - w). Source selection:
  exemplar  match anchor-domain descriptors: 16x16 px at 1/4 res (64 native) around the cell, zero-mean unit-norm;
            query = anchor, key = 4x downscaled input (in focus, so it is comparable with the anchor)
  random    a random bank cell (control: does relevance matter?)
  oracle    match the target's own high band against bank high bands (upper bound on usable in-focus texture)
Scores: whole image (crop 64, masks) PSNR/HB, and LPIPS/DISTS/PSNR on up to 6 random 512-px tiles per DP blur
bin (tile's median blur; same tiles for all methods) (b0..b3 edges 0.4/1.5/3.5 DP px).

  NATIVE=dataset/dpdd_native RESULTS=dataset/results/local DPMAPS=dataset/dpdd_1680/test/dp_maps \
      [UP4=<anchor+bicubic dir> A4=<anchor x4 dir>] python code/analysis/exemplar_pilot.py [--limit N] [--device cuda]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from uhdd.dualpixel import load_blur_map  # noqa: E402
from uhdd.exemplar import C, cells, high_band, lowres_desc, luma, nn_match, overlap_add, patches  # noqa: E402,F401
from uhdd.metrics import fidelity, iqa  # noqa: E402

D = os.environ.get("NATIVE", "dataset/dpdd_native")
R = os.environ.get("RESULTS", "dataset/results/local")
DP = os.environ.get("DPMAPS", "dataset/dpdd_1680/test/dp_maps")
UP4 = os.environ.get("UP4", "drb_x4+bicubic_x4")     # anchor + bicubic x4 (folder under RESULTS)
A4 = os.environ.get("A4", "drb_x4")                   # anchor at 1/4 (folder under RESULTS)
T0, T1 = 0.4, 1.2          # composite weight ramp (DP px), as dp_composite.py
EDGES = (0.4, 1.5, 3.5)


def rd(p):
    return cv2.imread(p, cv2.IMREAD_UNCHANGED)


def f32(img):
    return img[:, :, ::-1].astype(np.float32) / 65535


def tensor(x, device):
    return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].to(device)


def pick_tiles(mask, bm, crop=64, tile=512, per_bin=6, seed=0):
    """Up to `per_bin` valid 512-px tiles per DP blur bin (same tiles for every method)."""
    H, W = mask.shape
    by = {}
    for y in range(crop, H - crop - tile + 1, tile):
        for x in range(crop, W - crop - tile + 1, tile):
            if mask[y:y + tile, x:x + tile].mean() >= 0.9:
                by.setdefault(int(np.searchsorted(EDGES, np.median(bm[y:y + tile, x:x + tile]))), []).append((y, x))
    rng = np.random.default_rng(seed)
    return {b: [v[i] for i in sorted(rng.permutation(len(v))[:per_bin])] for b, v in by.items()}


def hb_rgb(x):
    """High band per channel (as uhdd.metrics.optics.high_band, s=4), float32 via OpenCV."""
    h, w = x.shape[:2]
    lo = cv2.resize(cv2.resize(x, (w // 4, h // 4), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_CUBIC)
    return x - lo


def score(pred, gt, mask, tiles, device, gt_hb, crop=64, tile=512):
    """Whole-image PSNR / HB-NMSE on the CPU in row blocks (low memory); LPIPS/DISTS/PSNR on GPU tiles."""
    import time
    T = time.time()
    c = slice(crop, -crop)
    se, n, hn = 0.0, 0, 0.0
    m = mask[c, c]
    for y in range(0, m.shape[0], 512):
        d = pred[c, c][y:y + 512] - gt[c, c][y:y + 512]
        mm = m[y:y + 512]
        se += float((d.astype(np.float64) ** 2).sum(-1)[mm].sum())
        n += int(mm.sum()) * 3
    ph = hb_rgb(pred)
    for y in range(0, pred.shape[0], 512):
        hn += float(((ph[y:y + 512] - gt_hb[y:y + 512]).astype(np.float64) ** 2).sum())
    del ph
    print(f"    whole-image {time.time() - T:.0f}s", flush=True)
    r = {"psnr": 10 * np.log10(n / se), "hb_nmse_db": 10 * np.log10(hn / float((gt_hb.astype(np.float64) ** 2).sum()))}
    for b, pos in tiles.items():
        v = []
        for y, x in pos:
            pt, gt_ = tensor(pred[y:y + tile, x:x + tile], device), tensor(gt[y:y + tile, x:x + tile], device)
            v.append((fidelity.psnr(pt, gt_), iqa.full_reference("lpips", pt, gt_, tile=tile),
                      iqa.full_reference("dists", pt, gt_, tile=tile)))
        v = np.array(v)
        r.update({f"psnr_t{b}": float(v[:, 0].mean()), f"lpips_t{b}": float(v[:, 1].mean()),
                  f"dists_t{b}": float(v[:, 2].mean()), f"n_t{b}": len(v)})
    return r


def run(name, device, rng, save=False):
    import time
    t0 = time.time()
    inp, gt = f32(rd(f"{D}/inputs/{name}.png")), f32(rd(f"{D}/x1/targets/{name}.png"))
    aup, a4 = f32(rd(f"{R}/{UP4}/{name}.png")), f32(rd(f"{R}/{A4}/{name}.png"))
    mask = rd(f"{D}/x1/masks/{name}.png") > 127
    H, W = gt.shape[:2]
    bm = load_blur_map(DP, name, (H, W))
    w = np.clip((T1 - bm) / (T1 - T0), 0, 1)
    comp = w[..., None] * inp + (1 - w[..., None]) * aup

    yin = luma(inp)
    hf_in, hf_gt = high_band(yin), high_band(luma(gt))
    pos = cells(H, W)
    wbox = cv2.boxFilter(w, -1, (C, C), anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    wc = wbox[pos[:, 0], pos[:, 1]]
    bank, query = pos[wc > 0.9], pos[wc < 0.9]
    tiles = pick_tiles(mask, bm)
    gt_hb = hb_rgb(gt)
    print(f"  {name} prep {time.time() - t0:.0f}s", flush=True)
    res, saved = {}, {}

    def emit(key, img):                       # score (and optionally keep) one output, then drop it
        res[key] = score(img, gt, mask, tiles, device, gt_hb)
        print(f"  {name} {key} {time.time() - t0:.0f}s", flush=True)
        if save:
            saved[key] = np.rint(np.clip(img, 0, 1) * 65535).astype(np.uint16)
        if device.type == "mps":
            torch.mps.empty_cache()

    emit("input", inp)
    emit("bicubic", aup)
    emit("composite", comp)
    del aup
    stats = {"bank": int(len(bank)), "query": int(len(query))}
    if len(bank) >= 16 and len(query):
        y4_in = cv2.resize(yin, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        kd, ksd = lowres_desc(y4_in, bank)
        qd, qsd = lowres_desc(luma(a4), query)
        bank_hf = patches(hf_in, bank)
        j_ex, sim = nn_match(qd, kd, device)
        bh = bank_hf.reshape(len(bank), -1)
        bh = bh / (np.linalg.norm(bh, axis=1, keepdims=True) + 1e-6)
        gh = patches(hf_gt, query).reshape(len(query), -1)
        j_or, _ = nn_match(gh / (np.linalg.norm(gh, axis=1, keepdims=True) + 1e-6), bh, device)
        del gh, bh
        j_rd = rng.integers(0, len(bank), len(query))
        stats.update(sim_median=float(np.median(sim)), sim_frac_08=float((sim > 0.8).mean()))
        wq = (1 - w)[..., None]
        for key, j in (("exemplar", j_ex), ("random", j_rd), ("oracle", j_or)):
            if key == "oracle":                                    # amplitude of the target's own high band
                gain = patches(hf_gt, query).std((1, 2)) / (bank_hf[j].std((1, 2)) + 1e-6)
            else:
                gain = qsd / (ksd[j] + 1e-6)
            gain = np.clip(gain, 0.5, 2.0)[:, None, None]
            tex = overlap_add((H, W), query, bank_hf[j] * gain)
            emit(key, np.clip(comp + wq * tex[..., None], 0, 1))
            del tex
    swin = f"{R}/drb_x4+swinir_x4_real/{name}.png"
    if os.path.exists(swin):
        emit("composite_swinir", w[..., None] * inp + (1 - w[..., None]) * f32(rd(swin)))
    stats.update(t_total=round(time.time() - t0, 1))
    return res, stats, saved


def summarize(path, ref="composite"):
    """Markdown tables: means per method, and paired differences vs `ref` with 95 % bootstrap CIs."""
    allr = json.load(open(path))
    meths = ["input", "bicubic", "composite", "random", "exemplar", "oracle"]
    cols = ["psnr", "hb_nmse_db"] + [f"{m}_t{b}" for b in (0, 2, 3) for m in ("psnr", "lpips", "dists")]
    rng = np.random.default_rng(0)
    print(f"n = {len(allr)} images; tiles: up to 6 per bin per image\n")
    print("| method | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1))
    for m in meths:
        vals = [np.nanmean([r[m].get(c, np.nan) for r in allr.values() if m in r]) for c in cols]
        print(f"| {m} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    print(f"\nPaired Δ vs {ref} (mean [95 % CI], % images improved)\n")
    print("| method | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1))
    for m in meths:
        if m == ref:
            continue
        cells = []
        for c in cols:
            d = np.array([r[m][c] - r[ref][c] for r in allr.values() if m in r and c in r[m] and c in r[ref]])
            if not len(d):
                cells.append("–")
                continue
            bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
            better = (d > 0) if c.startswith("psnr") else (d < 0)
            cells.append(f"{d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] {100 * better.mean():.0f}%")
        print(f"| {m} | " + " | ".join(cells) + " |")
    st = [r["stats"] for r in allr.values()]
    print(f"\nmatch similarity median {np.median([x.get('sim_median', np.nan) for x in st]):.3f}, "
          f"frac > 0.8 {np.nanmedian([x.get('sim_frac_08', np.nan) for x in st]):.3f}; bank cells median {np.median([x['bank'] for x in st]):.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--summary", action="store_true", help="only print tables from exemplar_pilot.json")
    ap.add_argument("--save", default="", help="comma-separated names whose outputs are saved as 16-bit PNG")
    a = ap.parse_args()
    if a.summary:
        return summarize(f"{R}/exemplar_pilot.json")
    device = torch.device(a.device)
    names = sorted(f[:-4] for f in os.listdir(f"{R}/{A4}") if f.endswith(".png"))[: a.limit or None]
    rng = np.random.default_rng(0)
    allr = {}
    for n in names:
        res, stats, out = run(n, device, rng, save=n in a.save.split(","))
        allr[n] = {"stats": stats, **res}
        print(n, stats, {k: (round(v["psnr"], 2), round(v.get("dists_t3", v.get("dists_t2", np.nan)), 3)) for k, v in res.items()}, flush=True)
        if n in a.save.split(","):
            for k, v in out.items():
                os.makedirs(f"{R}/pilot_{k}", exist_ok=True)
                cv2.imwrite(f"{R}/pilot_{k}/{n}.png", v[:, :, ::-1])
        json.dump(allr, open(f"{R}/exemplar_pilot.json", "w"))


if __name__ == "__main__":
    main()
