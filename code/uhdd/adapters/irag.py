"""iRAG (Lee et al., ICCV 2025): reference-based SR with a TTSR restoration branch and a StableSR-style latent
diffusion model (github.com/ByeonghunLee12/iRAG, `sr/inference.py`, online mode).

Per tile, as the official script does per image: lq = bicubic x4 of the anchor tile in [-1, 1]; TTSR(lq, ref,
lr = anchor tile) gives the intermediate I_inter; z_lr from the CFW autoencoder (StableSR VQGAN-CFW), z_inter from the
diffusion VAE; DDIM (eta 0) from q_sample(z_lr, t = 999) with empty prompt, conditioned on [z_lr, z_inter]; CFW decode
with fusion weight `dec_w` (0.5); AdaIN colour correction to lq. Steps: `steps` (official default 200, their
evaluation 50; we use 50). `output: inter` returns the TTSR intermediate only (the regression RefSR result, no
diffusion). The retrieval stage of iRAG (hash search in a database) is replaced by our same-image references.

The model code needs pytorch-lightning / taming only for training: `compat.irag_imports()` stubs them when absent,
and the CFW autoencoder's training loss (LPIPS + discriminator) is replaced by Identity before it is built.
"""
from __future__ import annotations

import contextlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class IRAG:
    def __init__(self, spec: dict, device="cuda"):
        from .compat import irag_imports
        irag_imports()
        repo = Path(spec["repo"])
        sys.path.insert(0, str(repo))
        from omegaconf import OmegaConf
        from ldm.util import instantiate_from_config
        from ldm.models.diffusion.ddim import DDIMSampler
        from scripts.wavelet_color_fix import adaptive_instance_normalization, wavelet_reconstruction
        self.spec, self.device = spec, torch.device(device)
        cfg = OmegaConf.load(spec.get("config", str(repo / "configs" / "irag.yaml")))
        cfg.model.params.ckpt_path = spec["ckpt"]                 # merged diffusion + TTSR checkpoint
        model = instantiate_from_config(cfg.model)
        sd = torch.load(spec["ckpt"], map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(sd.get("state_dict", sd), strict=False)   # as inference.py
        self.load_report = {"missing": len(missing), "unexpected": len(unexpected)}
        vq_cfg = OmegaConf.load(spec.get("vq_config", str(repo / "configs" / "autoencoder" / "autoencoder_kl_64x64x4_resi.yaml")))
        vq_cfg.model.params.lossconfig = {"target": "torch.nn.Identity"}
        vq_cfg.model.params.pop("ckpt_path", None)
        vq = instantiate_from_config(vq_cfg.model)
        vsd = torch.load(spec["vqgan_ckpt"], map_location="cpu", weights_only=False)
        vm, vu = vq.load_state_dict(vsd.get("state_dict", vsd), strict=False)
        self.load_report.update(vq_missing=len(vm), vq_unexpected=len(vu))
        vq.decoder.fusion_w = float(spec.get("dec_w", 0.5))
        model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000, linear_start=0.00085,
                                linear_end=0.0120, cosine_s=8e-3)
        model.num_timesteps = 1000
        self.model, self.vq = model.to(self.device).eval(), vq.to(self.device).eval()
        steps = int(spec.get("steps", 50))
        self.sampler = DDIMSampler(self.model)
        self.sampler.make_schedule(ddim_num_steps=steps, ddim_eta=float(spec.get("eta", 0.0)), verbose=False)
        self.timesteps = np.array(sorted(self._space_timesteps(repo, steps)))
        self.adain, self.wavelet = adaptive_instance_normalization, wavelet_reconstruction

    @staticmethod
    def _space_timesteps(repo: Path, steps: int):
        sp = importlib.util.spec_from_file_location("irag_inference", repo / "inference.py")
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
        return mod.space_timesteps(1000, [steps])

    @torch.no_grad()
    def __call__(self, anchor: torch.Tensor, ref: torch.Tensor | None) -> torch.Tensor:
        s, m = self.spec, self.model
        a = anchor.float().to(self.device)
        lq = F.interpolate(a, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1) * 2 - 1
        if ref is None:
            ref = (lq + 1) / 2                                    # no reference: the input itself
        r = ref.float().to(self.device)
        if r.shape[-2:] != lq.shape[-2:]:
            r = F.interpolate(r, size=lq.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
        r = r * 2 - 1
        torch.manual_seed(int(s.get("seed", 42)))
        ctx = torch.autocast("cuda") if self.device.type == "cuda" else contextlib.nullcontext()
        with ctx, m.ema_scope():
            inter = m.restoration_module(lq, r, lr_pixel=a * 2 - 1).clamp(-1, 1)
            if s.get("output") == "inter":
                return ((inter.float() + 1) / 2).clamp(0, 1).cpu()
            init, enc_fea = self.vq.encode(lq)
            z_lr = m.get_first_stage_encoding(init)
            z_inter = m.get_first_stage_encoding(m.encode_first_stage(inter))
            c = m.cond_stage_model([""] * lq.shape[0])
            t = torch.full((lq.shape[0],), 999, device=self.device, dtype=torch.long)
            x_T = m.q_sample(x_start=z_lr, t=t, noise=torch.randn_like(z_lr))
            z, _ = self.sampler.ddim_sampling_sr_t(cond=c, struct_cond=z_lr, struct_cond_ref=z_inter, shape=z_lr.shape,
                                                   unconditional_conditioning=None, unconditional_guidance_scale=None,
                                                   timesteps=self.timesteps, x_T=x_T)
            x = self.vq.decode(z * 1. / m.scale_factor, enc_fea)
        x = x.float()
        fix = s.get("colorfix", "adain")
        if fix == "adain":
            x = self.adain(x, lq.float())
        elif fix == "wavelet":
            x = self.wavelet(x, lq.float())
        return ((x + 1) / 2).clamp(0, 1).cpu()


def build(spec: dict, device="cuda"):
    return IRAG(spec, device)
