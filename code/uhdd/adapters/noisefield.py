"""Per-image latent noise shared by all tiles (for one-step generative upsamplers run through our tiler).

The official whole-image / latent-tiled samplers of S3Diff and VOSR draw ONE noise tensor for the whole latent image,
so overlapping tiles see the same noise. Run per tile, the same is achieved by drawing a noise field for the whole
image once (seeded, per image) and cropping each tile's window from it. `factor` = latent px per input px
(x4 upsampling into an /8 VAE latent: 0.5).
"""
from __future__ import annotations

import math

import torch


class NoiseField:
    def __init__(self, channels: int, factor: float, seed: int = 0):
        self.c, self.f, self.seed, self.z = channels, factor, seed, None

    def reset(self) -> None:
        self.z = None

    def crop(self, boxes: list[tuple[int, int, int, int]], full_hw: tuple[int, int], device, dtype=torch.float32):
        """Noise for tiles `boxes` = [(y, x, h, w)] in input px of the (padded) image of size >= full_hw."""
        H = max([full_hw[0]] + [y + h for y, _, h, _ in boxes])
        W = max([full_hw[1]] + [x + w for _, x, _, w in boxes])
        need = (math.ceil(H * self.f) + 1, math.ceil(W * self.f) + 1)
        if self.z is None or self.z.shape[-2] < need[0] or self.z.shape[-1] < need[1] or self.z.device != torch.device(device):
            g = torch.Generator(device=device).manual_seed(self.seed)
            self.z = torch.randn(1, self.c, *need, generator=g, device=device, dtype=torch.float32)
        out = []
        for y, x, h, w in boxes:
            ly, lx, lh, lw = int(y * self.f), int(x * self.f), round(h * self.f), round(w * self.f)
            out.append(self.z[..., ly:ly + lh, lx:lx + lw])
        return torch.cat(out).to(dtype)


def gaussian_color_fix(y: torch.Tensor, src: torch.Tensor, sigma: float = 5.0) -> torch.Tensor:
    """VOSR's `wavelet_color_fix` in torch: low frequencies (Gaussian, sigma 5 px, cv2 kernel size 8 sigma + 1,
    reflect-101 borders) from `src`, high frequencies from `y`."""
    from torchvision.transforms.functional import gaussian_blur
    k = int(round(sigma * 8 + 1)) | 1
    k = min(k, (min(y.shape[-2:]) // 2) * 2 - 1)    # reflect padding needs pad < size (tiny inputs only)
    lo = lambda t: gaussian_blur(t, [k, k], [sigma, sigma])  # noqa: E731
    return (lo(src) + y - lo(y)).clamp(0, 1)
