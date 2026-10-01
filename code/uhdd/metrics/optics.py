"""Optics-aware metrics.

1. PSF-matched fidelity. At native resolution the DPDD "sharp" target (f/22) is diffraction
   limited: Airy first-zero diameter 2.44*lambda*N ~ 30 um vs. 5.36 um pixel pitch (~5.5 px).
   A prediction sharper than the target is penalized by PSNR/SSIM. We therefore blur the
   prediction with the target's diffraction PSF before comparing ("pm_" metrics), which removes
   the penalty for detail beyond the target's own optical cutoff.
   Measured (secure env, slanted edges on raw): the f/22 targets are blurrier than Airy x pixel
   aperture, extra Gaussian sigma ~0.7-0.8 px at native resolution. Part of that (optical
   low-pass filter, pixel aperture) is shared by the f/4 input, so the right extra blur is
   undecided: `pm` = Airy only (primary), `pmg` = Airy + Gaussian(extra_sigma) (sensitivity).

2. High-band error. Native-resolution detail is exactly what a 1/s-resolution anchor cannot
   contain. H(x) = x - U(D(x)) keeps frequencies above the 1/s Nyquist limit (D = s x s box
   average, U = bicubic). We report normalized MSE in dB: 10 log10(|H(p)-H(g)|^2 / |H(g)|^2).
   0 dB = no better than predicting no high band at all; lower is better.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import j1

# Canon EOS 5D Mark IV: 36 mm / 6720 px
DPDD_PITCH_UM = 36000.0 / 6720.0
RGB_WAVELENGTHS_UM = (0.610, 0.550, 0.465)


@lru_cache(maxsize=16)
def airy_kernel(wavelength_um: float, fnum: float, pitch_um: float,
                radius_zeros: float = 4.0, oversample: int = 11) -> np.ndarray:
    """Discrete Airy intensity PSF, integrated over square pixels, normalized to sum 1."""
    r0_px = 1.22 * wavelength_um * fnum / pitch_um           # first zero radius in pixels
    R = max(1, int(math.ceil(radius_zeros * r0_px)))
    n = (2 * R + 1) * oversample
    c = (np.arange(n) + 0.5) / oversample - (R + 0.5)        # sub-pixel centers, in pixels
    rr = np.hypot(*np.meshgrid(c, c, indexing="ij")) * pitch_um
    v = math.pi * rr / (wavelength_um * fnum)
    with np.errstate(divide="ignore", invalid="ignore"):
        I = np.where(v > 1e-12, (2 * j1(v) / v) ** 2, 1.0)
    k = I.reshape(2 * R + 1, oversample, 2 * R + 1, oversample).sum(axis=(1, 3))
    return (k / k.sum()).astype(np.float32)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    r = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = (k / k.sum()).view(1, 1, 1, -1).repeat(x.shape[1], 1, 1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k, groups=x.shape[1])
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.transpose(-1, -2), groups=x.shape[1])


def psf_match(pred: torch.Tensor, fnum: float = 22.0, pitch_um: float = DPDD_PITCH_UM,
              scale: int = 1, extra_sigma_native: float = 0.0) -> torch.Tensor:
    """Blur 1x3xHxW `pred` with the per-channel diffraction PSF of the target capture.

    `scale` = downsampling factor of the evaluated resolution (pixel pitch grows by it).
    """
    ks = [torch.from_numpy(airy_kernel(w, fnum, pitch_um * scale)) for w in RGB_WAVELENGTHS_UM]
    K = max(k.shape[0] for k in ks)
    ks = [F.pad(k, [(K - k.shape[0]) // 2] * 4) for k in ks]
    weight = torch.stack(ks)[:, None].to(pred.device, pred.dtype)  # 3x1xKxK
    p = K // 2
    x = F.pad(pred, (p, p, p, p), mode="reflect")
    # optional extra Gaussian blur (sigma in native pixels, scaled to the evaluated resolution)
    return gaussian_blur(F.conv2d(x, weight, groups=3), extra_sigma_native / scale)


def high_band(x: torch.Tensor, s: int) -> torch.Tensor:
    H, W = x.shape[-2:]
    x = x[..., : H // s * s, : W // s * s]
    low = F.interpolate(F.avg_pool2d(x, s), scale_factor=s, mode="bicubic", align_corners=False)
    return x - low


def highband_nmse_db(pred: torch.Tensor, gt: torch.Tensor, s: int = 4) -> float:
    hp, hg = high_band(pred.double(), s), high_band(gt.double(), s)
    num = (hp - hg).pow(2).sum().item()
    den = hg.pow(2).sum().item()
    return 10 * math.log10(max(num, 1e-30) / max(den, 1e-30))
