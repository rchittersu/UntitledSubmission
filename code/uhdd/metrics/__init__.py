"""Metric registry: `compute(names, pred, gt, mask, ctx)` -> {column: value}.

Names (comma-separated on the CLI):
  psnr, ssim, mae          fidelity                                    (needs gt)
  pm                       PSF-matched psnr/ssim (pm_psnr, pm_ssim)   (needs gt)
  pmg                      as pm + extra Gaussian ctx["pm_sigma"] px at native (pmg_psnr, pmg_ssim)
  hb                       high-band NMSE in dB (hb_nmse_db)           (needs gt)
  lpips, dists             tiled pyiqa FR metrics                      (needs gt)
  musiq, maniqa, clipiqa   pyiqa NR metrics on a crop grid
  drift                    low-frequency drift (lf_drift)              (needs gt)
  seam                     seam_step / seam_ratio on the method grid  (needs ctx["info"])
  stats                    sharpness / noise / slope consistency      (needs gt)
  blurbins                 PSNR per DP-defocus bin (psnr_b0..b3, frac_b0..b3), bins on smoothed
                           |disparity| [0,0.4), [0.4,1.5), [1.5,3.5), [3.5,inf) DP px; b0 = focal plane
                                                                       (needs gt, ctx["blur_map"])
  gridshift                gs_mad, gs_psnr, gs_seam_step, gs_seam_ratio: difference to the same
                           model run on a shifted tile grid           (needs ctx["pred_shift"])

ctx keys: scale (evaluated res = native / scale; sets PSF pixel pitch), fnum (target f-number),
hb_s (high-band factor), info (tiling dict of this image from run_model meta), crop (border
cropped before evaluation), fr_tile, cell.
"""
from __future__ import annotations

import torch

from . import consistency, fidelity, iqa, optics

NEEDS_GT = {"psnr", "ssim", "mae", "pm", "pmg", "hb", "blurbins", "lpips", "dists", "drift", "stats"}
ALL = ["psnr", "ssim", "mae", "pm", "pmg", "hb", "lpips", "dists", "musiq", "maniqa", "clipiqa",
       "drift", "seam", "stats", "gridshift", "blurbins"]
BLUR_EDGES = (0.4, 1.5, 3.5)
# direction for summaries: +1 higher is better, -1 lower is better
DIRECTION = {"psnr": 1, "ssim": 1, "pm_psnr": 1, "pm_ssim": 1, "pmg_psnr": 1, "pmg_ssim": 1, "mae": -1, "hb_nmse_db": -1,
             "lpips": -1, "dists": -1, "musiq": 1, "maniqa": 1, "clipiqa": 1, "lf_drift": -1,
             "seam_step": -1, "seam_ratio": -1, "sharp_lr_abs": -1, "sharp_lr_std": -1,
             "noise_lr_abs": -1, "noise_lr_std": -1, "slope_err_abs": -1, "slope_err_std": -1,
             "psnr_b0": 1, "psnr_b1": 1, "psnr_b2": 1, "psnr_b3": 1, "gs_mad": -1, "gs_psnr": 1, "gs_seam_step": -1, "gs_seam_ratio": -1,
             "time_s": -1, "pipeline_time_s": -1, "peak_mem_gb": -1}


def compute(names: list[str], pred: torch.Tensor, gt: torch.Tensor | None,
            mask: torch.Tensor | None, ctx: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for n in names:
        if n in NEEDS_GT and gt is None:
            continue
        if n == "psnr":
            out["psnr"] = fidelity.psnr(pred, gt, mask)
        elif n == "ssim":
            out["ssim"] = fidelity.ssim(pred, gt, mask)
        elif n == "mae":
            out["mae"] = fidelity.mae(pred, gt, mask)
        elif n == "pm":
            pm = optics.psf_match(pred, ctx.get("fnum", 22.0), scale=ctx.get("scale", 1))
            out["pm_psnr"] = fidelity.psnr(pm, gt, mask)
            out["pm_ssim"] = fidelity.ssim(pm, gt, mask)
        elif n == "pmg":
            pm = optics.psf_match(pred, ctx.get("fnum", 22.0), scale=ctx.get("scale", 1),
                                  extra_sigma_native=ctx.get("pm_sigma", 0.8))
            out["pmg_psnr"] = fidelity.psnr(pm, gt, mask)
            out["pmg_ssim"] = fidelity.ssim(pm, gt, mask)
        elif n == "hb":
            out["hb_nmse_db"] = optics.highband_nmse_db(pred, gt, ctx.get("hb_s", 4))
        elif n in ("lpips", "dists"):
            out[n] = iqa.full_reference(n, pred, gt, tile=ctx.get("fr_tile", 1024))
        elif n in ("musiq", "maniqa", "clipiqa"):
            out[n] = iqa.no_reference(n, pred)
        elif n == "drift":
            out["lf_drift"] = consistency.lf_drift(pred, gt, ctx.get("cell", 256))
        elif n == "seam":
            if ctx.get("info"):
                out.update(consistency.seam_metrics(pred, gt, ctx["info"], ctx.get("crop", 0)))
        elif n == "blurbins":
            bm = ctx.get("blur_map")
            if bm is not None:
                edges = (0.0, *ctx.get("blur_edges", BLUR_EDGES), float("inf"))
                for k in range(len(edges) - 1):
                    m = (bm >= edges[k]) & (bm < edges[k + 1])
                    if mask is not None:
                        m = m & mask
                    out[f"frac_b{k}"] = m.float().mean().item()
                    out[f"psnr_b{k}"] = fidelity.psnr(pred, gt, m) if m.any() else float("nan")
        elif n == "gridshift":
            if ctx.get("pred_shift") is not None and ctx.get("info") and ctx.get("info_shift"):
                out.update(consistency.grid_shift_metrics(pred, ctx["pred_shift"], ctx["info"],
                                                          ctx["info_shift"], ctx.get("crop", 0)))
        elif n == "stats":
            out.update(consistency.stat_consistency(pred, gt, ctx.get("cell", 256)))
        else:
            raise KeyError(f"unknown metric '{n}' (known: {ALL})")
    return out
