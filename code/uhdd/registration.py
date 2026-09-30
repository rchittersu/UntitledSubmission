"""Registration of target (sharp) to input (blurry) captures.

Aperture-bracketed pairs are captured one after another and are slightly misaligned; at native
resolution the misalignment is s times larger (in pixels) than in the 1/s benchmark. We align
each *target to its input* once, method-independently, and freeze the result.

Coarse-to-fine ECC (OpenCV) on Gaussian-smoothed luminance; the smoothing also reduces the
influence of the defocus difference between the two images. The target is warped only if the
estimated motion moves any image corner by more than `min_shift` pixels, to avoid adding
interpolation blur to already aligned pairs.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MOTION = {"translation": cv2.MOTION_TRANSLATION, "euclidean": cv2.MOTION_EUCLIDEAN,
          "affine": cv2.MOTION_AFFINE, "homography": cv2.MOTION_HOMOGRAPHY}


@dataclass
class RegConfig:
    motion: str = "homography"
    levels: tuple[int, ...] = (16, 8, 4, 2)   # downsampling factors, coarse to fine
    iters: int = 100
    eps: float = 1e-6
    gauss: int = 5                            # ECC gaussFiltSize
    min_shift: float = 0.3                    # px at native res; below -> identity
    erode: int = 4                            # px eroded from the valid mask


def _gray(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return g / (65535.0 if img.dtype == np.uint16 else 255.0)


def _scale_warp(Wm: np.ndarray, f: float) -> np.ndarray:
    """Convert a warp estimated at one scale to another scale (coords multiplied by f)."""
    if Wm.shape[0] == 2:
        Wm = Wm.copy()
        Wm[:, 2] *= f
        return Wm
    S = np.diag([f, f, 1.0]).astype(np.float32)
    return (S @ Wm @ np.linalg.inv(S)).astype(np.float32)


def corner_shift(Wm: np.ndarray, H: int, W: int) -> float:
    pts = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]], np.float32)[None]
    M = Wm if Wm.shape[0] == 3 else np.vstack([Wm, [0, 0, 1]]).astype(np.float32)
    moved = cv2.perspectiveTransform(pts, M)
    return float(np.abs(moved - pts).max())


def estimate(inp: np.ndarray, tgt: np.ndarray, cfg: RegConfig) -> tuple[np.ndarray, float]:
    """Warp W such that tgt(W(x)) ~ inp(x). Returns (W at native scale, final ECC correlation)."""
    motion = MOTION[cfg.motion]
    Wm = np.eye(3 if motion == cv2.MOTION_HOMOGRAPHY else 2, 3, dtype=np.float32)
    gi, gt = _gray(inp), _gray(tgt)
    H, W = gi.shape
    prev_f, cc = None, float("nan")
    for f in cfg.levels:
        size = (W // f, H // f)
        a = cv2.resize(gi, size, interpolation=cv2.INTER_AREA)
        b = cv2.resize(gt, size, interpolation=cv2.INTER_AREA)
        if prev_f is not None:
            Wm = _scale_warp(Wm, prev_f / f)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, cfg.iters, cfg.eps)
        cc, Wm = cv2.findTransformECC(a, b, Wm, motion, crit, None, cfg.gauss)
        prev_f = f
    return _scale_warp(Wm, float(prev_f)), float(cc)


def apply(tgt: np.ndarray, Wm: np.ndarray, erode: int) -> tuple[np.ndarray, np.ndarray]:
    """Warp target onto the input grid (Lanczos, float32) and return (aligned, valid mask)."""
    H, W = tgt.shape[:2]
    flags = cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP
    maxv = 65535.0 if tgt.dtype == np.uint16 else 255.0
    src = tgt.astype(np.float32)
    ones = np.ones((H, W), np.uint8)
    if Wm.shape[0] == 3:
        out = cv2.warpPerspective(src, Wm, (W, H), flags=flags, borderMode=cv2.BORDER_REFLECT)
        m = cv2.warpPerspective(ones, Wm, (W, H), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    else:
        out = cv2.warpAffine(src, Wm, (W, H), flags=flags, borderMode=cv2.BORDER_REFLECT)
        m = cv2.warpAffine(ones, Wm, (W, H), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    if erode > 0:
        m = cv2.erode(m, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
    out = np.clip(np.rint(out), 0, maxv).astype(tgt.dtype)
    return out, (m > 0).astype(np.uint8) * 255


def register_pair(inp: np.ndarray, tgt: np.ndarray, cfg: RegConfig) -> dict:
    """Returns {aligned, mask, warp, corner_shift_px, ecc, warped}."""
    Wm, cc = estimate(inp, tgt, cfg)
    H, W = inp.shape[:2]
    shift = corner_shift(Wm, H, W)
    if shift < cfg.min_shift:
        return {"aligned": tgt, "mask": np.full((H, W), 255, np.uint8), "warp": Wm,
                "corner_shift_px": shift, "ecc": cc, "warped": False}
    aligned, mask = apply(tgt, Wm, cfg.erode)
    return {"aligned": aligned, "mask": mask, "warp": Wm, "corner_shift_px": shift, "ecc": cc,
            "warped": True}
