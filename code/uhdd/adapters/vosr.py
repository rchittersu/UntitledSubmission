"""VOSR one-step (github.com/cswry/VOSR, CVPR 2026): x4 SR with a LightningDiT in a VAE latent, conditioned on the
bicubic-upsampled image latent and DINOv2 features. Generative x4 upsampler after the 1/4-res deblur
(docs/baselines.md, C6). Variants by checkpoint folder: VOSR2 (1.4B, Qwen-Image VAE, DINOv2-L), VOSR_0.5B_os
(SD2 VAE + light decoder, DINOv2-B), VOSR_1.4B_os.

Per tile, the official one-step path (inference_vosr_onestep.py, untiled branch; the official DiT tile is 512 output
px = the training resolution = our 128-LR-px tile):
  bicubic x4 -> [-1, 1] -> VAE encode (deterministic posterior mode, official default) -> DINOv2 features of the tile
  (resized to 448, layer from args.json) -> flow from noise in `infer_steps` (1) DiT steps; the noise is cropped from
  one per-image field so overlapping tiles share it, as in the official full-latent noise -> decode -> colour fix
  (official `wavelet_color_fix`: Gaussian sigma 5 low-pass from the upsampled input).
Model configuration comes from the checkpoint's args.json, like the official script. DiT forward_flexible needs square
inputs: non-square calls (whole-image mode on small inputs) are reflect-padded to a square and cropped back.
"""
from __future__ import annotations

import glob
import json
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

from .noisefield import NoiseField, gaussian_color_fix

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _local(path: str, ckpts: Path) -> Path:
    """args.json paths are relative to the repo ('preset/ckpts/...'): resolve them inside $VOSR_CKPTS."""
    p = Path(path)
    return ckpts / Path(*p.parts[2:]) if p.parts[:2] == ("preset", "ckpts") else p


def _dinov2(args, ckpts: Path, device):
    torch.hub.set_dir(str(ckpts / "torch_cache"))          # cached hub repo + pretrained weights (offline)
    hub = ckpts / "torch_cache" / "facebookresearch_dinov2_main"
    name = {"dinov2b": "dinov2_vitb14", "dinov2l": "dinov2_vitl14", "dinov2g": "dinov2_vitg14"}[args.enc_type]
    enc = torch.hub.load(str(hub), name, source="local") if hub.is_dir() else torch.hub.load("facebookresearch/dinov2", name)
    enc.head = torch.nn.Identity()

    def forward_with_features(self, x, masks=None):     # inference_vosr_onestep.py load_dinov2
        feats = {}
        x = self.prepare_tokens_with_masks(x, masks)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            feats[f"layer_{i}"] = x[:, 1:]
        return feats, self.norm(x)[:, 1:]

    enc.forward_with_features = types.MethodType(forward_with_features, enc)
    return enc.to(device).eval()


class VOSRTiles(torch.nn.Module):
    def __init__(self, dit, vae, venc, light_decoder, args):
        super().__init__()
        self.dit, self.vae, self.venc, self.light_decoder, self.args = dit, vae, venc, light_decoder, args


def build(spec: dict):
    from .compat import disable_xformers
    disable_xformers()
    repo, ckpts = Path(spec["repo"]), Path(spec["ckpts"])
    sys.path.insert(0, str(repo))
    from models.lightningdit import LightningDiT
    cdir = ckpts / spec.get("variant", "VOSR2")
    args = types.SimpleNamespace(**json.loads((cdir / "args.json").read_text()))
    if getattr(args, "auxiliary_time_cond", None) is None:      # official default for older checkpoints
        args.auxiliary_time_cond = True
    args.infer_steps = int(spec.get("infer_steps", getattr(args, "infer_steps", None) or 1))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    light = None
    if args.ae_type == "qwen":
        from models.qwenimage_vae2d import AutoencoderKLQwenImage2D
        vae = AutoencoderKLQwenImage2D.from_pretrained(str(_local(args.ae_path, ckpts)))
    else:
        from diffusers import AutoencoderKL
        from models.light_decoder import LightDecoder
        vae = AutoencoderKL.from_pretrained(str(_local(args.ae_path, ckpts)), subfolder="vae")
        ck = torch.load(ckpts / "sd21_lwdecoder.pth", map_location="cpu")
        c = ck["config"]
        light = LightDecoder(in_channels=c["in_channels"], out_channels=c["out_channels"],
                             block_out_channels=tuple(c["block_out_channels"]), layers_per_block=c["layers_per_block"])
        light.load_state_dict(ck["model_state_dict"])
        light = light.eval()

    base = 4 if args.ae_type == "sd2" else 16
    dit = LightningDiT(input_size=args.resolution // 8, patch_size=args.patch_size, in_channels=2 * base,
                       out_channels=base, hidden_size=args.dim, depth=args.depth, num_heads=args.num_heads,
                       mlp_ratio=args.mlp_ratio, z_dims=args.enc_dim, encdim_ratio=args.encdim_ratio,
                       auxiliary_time_cond=args.auxiliary_time_cond, use_qknorm=args.use_qknorm,
                       use_swiglu=args.use_swiglu, use_rope=args.use_rope, use_rmsnorm=args.use_rmsnorm,
                       wo_shift=args.wo_shift, num_fused_layers=len(args.layer_dinov2b_list))
    hits = [p for d in ("clean_weights", "checkpoints", ".") for n in ("ema_model.safetensors", "model.safetensors")
            if (p := cdir / d / n).is_file()] or sorted(glob.glob(str(cdir / "**" / "*.safetensors"), recursive=True))
    if not hits:
        raise FileNotFoundError(f"no .safetensors under {cdir}")
    from safetensors.torch import load_file
    res = dit.load_state_dict(load_file(str(hits[0])), strict=False)      # strict=False as the official script
    if res.missing_keys:
        print(f"[vosr] {len(res.missing_keys)} missing keys (non-persistent buffers expected): {res.missing_keys[:5]}")
    dit.forward = dit.forward_flexible
    return VOSRTiles(dit.eval(), vae.eval(), _dinov2(args, ckpts, device), light, args)


def wrap(net: VOSRTiles, spec: dict):
    sys.path.insert(0, str(Path(spec["repo"])))
    from tiled_vae import decode_latent, encode_latent
    a = net.args
    lc = 4 if a.ae_type == "sd2" else 16
    noise = NoiseField(lc, 0.5, int(spec.get("seed", 42)))
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)

    def prepare(x: torch.Tensor) -> None:
        noise.reset()

    @torch.no_grad()
    def venc_features(lq: torch.Tensor) -> list:          # get_venc_features + preprocess_raw_image
        r = F.interpolate(lq * 0.5 + 0.5, a.dinov2_size, mode="bicubic").clip(0, 1)
        feats, x_norm = net.venc.forward_with_features((r - mean.to(r)) / std.to(r))
        z = [v for k, v in feats.items() if k.startswith("layer_")]
        z[-1] = x_norm
        return [z[i] for i in a.layer_dinov2b_list]

    @torch.no_grad()
    def fn(x: torch.Tensor, boxes=None, full_hw=None) -> torch.Tensor:
        x = x.float()
        B, _, h, w = x.shape
        boxes = boxes or [(0, 0, h, w)] * B
        s = max(h, w)
        if h != w:      # forward_flexible: square only
            x = F.pad(x, (0, s - w, 0, s - h), mode="reflect")
            boxes = [(y0, x0, s, s) for y0, x0, _, _ in boxes]
        up = F.interpolate(x, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)
        lq = up * 2 - 1
        lat, lmean, lstd = encode_latent(net.vae, lq, a, x.device, posterior_mode=True)
        zf = venc_features(lq)
        z = noise.crop(boxes, full_hw or (h, w), x.device, lat.dtype)
        t = torch.linspace(1.0, 0.0, a.infer_steps + 1, device=x.device)
        for i in range(a.infer_steps):
            u = net.dit(torch.cat([lat, z], 1), t[i].expand(B), t[i + 1].expand(B), zf)
            z = z - (t[i] - t[i + 1]) * u
        y = (decode_latent(net.vae, z, a, lmean, lstd, net.light_decoder).float() * 0.5 + 0.5).clamp(0, 1)
        if spec.get("color_fix", "wavelet") == "wavelet":
            y = gaussian_color_fix(y, up)
        return y[..., : h * 4, : w * 4]

    fn.positional = True
    fn.prepare = prepare
    return fn
