"""S3Diff (Zhang et al., arXiv 2024; github.com/ArcticHare105/S3Diff): one-step x4 SR on SD-Turbo with
degradation-guided LoRA. Generative x4 upsampler after the 1/4-res deblur (docs/baselines.md, C5).

Per tile, the official test path (src/inference_s3diff.py + src/s3diff_tile.py) without the authors' own latent
tiling — our tile of 192 LR px = 768 HR px = 96 latent px is exactly their latent tile, so each tile is one untiled
pass of their model:
  bilinear x4 -> [-1, 1] -> VAE encode (posterior sample; noise cropped from one per-image field, see noisefield.py)
  -> UNet at t = 999 with classifier-free guidance 1.07 between the official positive / negative prompts
  -> one DDPM step -> decode -> wavelet colour fix (StableSR) to the bilinear-upsampled tile.
Degradation score: DEResNet on the WHOLE LR image, once per image (`prepare`), as the official script does.
Runs in the main env: their LoRA forward is replaced by an equivalent one that does not depend on peft internals,
and basicsr (two helpers) is shimmed. CUDA only (the model code hard-codes it). Train resolution 512 HR.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

from .noisefield import NoiseField

POS = "A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting."
NEG = "oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth"


def _shim_basicsr() -> None:
    """S3Diff imports basicsr.archs.arch_util.{default_init_weights, ResidualBlockNoBN} (weights are loaded after)."""
    if "basicsr.archs.arch_util" in sys.modules:
        return
    from torch import nn

    @torch.no_grad()
    def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
        if not isinstance(module_list, list):
            module_list = [module_list]
        for module in module_list:
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, **kwargs)
                    m.weight.data *= scale
                    if m.bias is not None:
                        m.bias.data.fill_(bias_fill)

    class ResidualBlockNoBN(nn.Module):
        def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
            super().__init__()
            self.res_scale = res_scale
            self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
            self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
            self.relu = nn.ReLU(inplace=True)
            if not pytorch_init:
                default_init_weights([self.conv1, self.conv2], 0.1)

        def forward(self, x):
            return x + self.conv2(self.relu(self.conv1(x))) * self.res_scale

    mods = {n: types.ModuleType(n) for n in ("basicsr", "basicsr.archs", "basicsr.archs.arch_util")}
    mods["basicsr.archs.arch_util"].default_init_weights = default_init_weights
    mods["basicsr.archs.arch_util"].ResidualBlockNoBN = ResidualBlockNoBN
    sys.modules.update(mods)


def _lora_fwd(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """S3Diff's my_lora_fwd (src/model.py) without the peft-version-specific checks: base output + per-adapter
    lora_B(de_mod-modulated lora_A(x)) * scaling."""
    kwargs.pop("adapter_names", None)
    result = self.base_layer(x, *args, **kwargs)
    dtype = result.dtype
    for name in self.active_adapters:
        if name not in self.lora_A.keys():
            continue
        a, b = self.lora_A[name], self.lora_B[name]
        x = x.to(a.weight.dtype)
        t = a(self.lora_dropout[name](x))
        if isinstance(a, torch.nn.Conv2d):
            t = torch.einsum("...khw,...kr->...rhw", t, self.de_mod)
        elif isinstance(a, torch.nn.Linear):
            t = torch.einsum("...lk,...kr->...lr", t, self.de_mod)
        else:
            raise NotImplementedError("only conv and linear LoRA layers")
        result = result + b(t) * self.scaling[name]
    return result.to(dtype)


class S3DiffTiles(torch.nn.Module):
    def __init__(self, s3, de_net, pos_enc, neg_enc):
        super().__init__()
        self.s3, self.de_net = s3, de_net
        self.register_buffer("pos_enc", pos_enc, persistent=False)
        self.register_buffer("neg_enc", neg_enc, persistent=False)

    @torch.no_grad()
    def modulate(self, deg: torch.Tensor) -> None:
        """Set the per-layer LoRA modulation from the degradation score (s3diff_tile.py, S3Diff.forward)."""
        s3 = self.s3
        proj = deg[..., None] * s3.W[None, None, :] * 2 * torch.pi
        proj = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        proj = torch.cat([proj[:, 0], proj[:, 1]], dim=-1)
        vae_de, unet_de = s3.vae_de_mlp(proj), s3.unet_de_mlp(proj)
        vae_blk = s3.vae_block_mlp(s3.vae_block_embeddings.weight)
        unet_blk = s3.unet_block_mlp(s3.unet_block_embeddings.weight)
        vae_emb = s3.vae_fuse_mlp(torch.cat([vae_de.unsqueeze(1).repeat(1, vae_blk.shape[0], 1),
                                             vae_blk.unsqueeze(0).repeat(vae_de.shape[0], 1, 1)], -1))
        unet_emb = s3.unet_fuse_mlp(torch.cat([unet_de.unsqueeze(1).repeat(1, unet_blk.shape[0], 1),
                                               unet_blk.unsqueeze(0).repeat(unet_de.shape[0], 1, 1)], -1))
        for name, m in s3.vae.named_modules():
            if name in s3.vae_lora_layers:
                p = name.split(".")
                e = vae_emb[:, int(p[2])] if p[1] == "down_blocks" else vae_emb[:, -2] if p[1] == "mid_block" else vae_emb[:, -1]
                m.de_mod = e.reshape(-1, s3.lora_rank_vae, s3.lora_rank_vae)
        for name, m in s3.unet.named_modules():
            if name in s3.unet_lora_layers:
                p = name.split(".")
                e = (unet_emb[:, int(p[1])] if p[0] == "down_blocks" else unet_emb[:, 4] if p[0] == "mid_block"
                     else unet_emb[:, int(p[1]) + 5] if p[0] == "up_blocks" else unet_emb[:, -1])
                m.de_mod = e.reshape(-1, s3.lora_rank_unet, s3.lora_rank_unet)


def build(spec: dict):
    repo = Path(spec["repo"])
    sys.path[:0] = [str(repo / "src"), str(repo)]
    _shim_basicsr()
    from de_net import DEResNet
    from s3diff import S3Diff
    s3 = S3Diff(sd_path=spec["sd_path"], pretrained_path=spec["weights"], lora_rank_unet=32, lora_rank_vae=16)
    for part, names in ((s3.vae, s3.vae_lora_layers), (s3.unet, s3.unet_lora_layers)):
        for name, m in part.named_modules():
            if name in names:
                m.forward = _lora_fwd.__get__(m, m.__class__)
    s3.set_eval()
    de = DEResNet(num_in_ch=3, num_degradation=2)
    de.load_model(spec.get("de_net") or str(repo / "assets" / "mm-realsr" / "de_net.pth"))

    @torch.no_grad()
    def enc(p: str) -> torch.Tensor:
        ids = s3.tokenizer([p], max_length=s3.tokenizer.model_max_length, padding="max_length", truncation=True,
                           return_tensors="pt").input_ids.cuda()
        return s3.text_encoder(ids)[0]

    return S3DiffTiles(s3, de.cuda().eval(), enc(spec.get("pos_prompt", POS)), enc(spec.get("neg_prompt", NEG)))


def wrap(net: S3DiffTiles, spec: dict):
    from utils.wavelet_color import adaptive_instance_normalization, wavelet_reconstruction
    s3 = net.s3
    g_scale = float(spec.get("guidance", 1.07))
    noise = NoiseField(s3.vae.config.latent_channels, 0.5, int(spec.get("seed", 0)))
    state: dict = {}

    @torch.no_grad()
    def prepare(x: torch.Tensor) -> None:      # whole LR image in [0, 1]
        state["deg"] = net.de_net(x.float())
        noise.reset()

    @torch.no_grad()
    def fn(x: torch.Tensor, boxes=None, full_hw=None) -> torch.Tensor:
        x = x.float()
        B = x.shape[0]
        if "deg" not in state:
            prepare(x[:1])
        net.modulate(state["deg"].expand(B, -1))
        up = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False).clamp(0, 1)
        c_t = up * 2 - 1
        dist = s3.vae.encode(c_t).latent_dist
        eps = noise.crop(boxes or [(0, 0, *x.shape[-2:])] * B, full_hw or x.shape[-2:], x.device, dist.mean.dtype)
        lat = (dist.mean + dist.std * eps) * s3.vae.config.scaling_factor
        pred = s3.unet(lat, s3.timesteps, encoder_hidden_states=net.pos_enc.expand(B, -1, -1)).sample
        if g_scale != 1.0:      # CFG: second UNet pass with the negative prompt (guidance 1 = positive prompt only)
            neg = s3.unet(lat, s3.timesteps, encoder_hidden_states=net.neg_enc.expand(B, -1, -1)).sample
            pred = neg + g_scale * (pred - neg)
        den = s3.sched.step(pred, s3.timesteps, lat, return_dict=True).prev_sample
        y = (s3.vae.decode(den / s3.vae.config.scaling_factor).sample.clamp(-1, 1) * 0.5 + 0.5).float()
        fix = spec.get("color_fix", "wavelet")
        if fix == "wavelet":
            y = wavelet_reconstruction(y, up)
        elif fix == "adain":
            y = adaptive_instance_normalization(y, up)
        return y.clamp(0, 1)

    fn.positional = True
    fn.prepare = prepare
    return fn
