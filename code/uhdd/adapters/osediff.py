"""OSEDiff (Wu et al., NeurIPS 2024): one-step diffusion SR on SD2.1-base + LoRA (github.com/cswry/OSEDiff).

Used as a generative x4 upsampler after the 1/4-res deblur (docs/baselines.md, C3). Per LR tile, as the
official test_osediff.py does per image: bicubic x4 upsampling, prompt from DAPE (RAM fine-tuned on degraded
inputs; `prompt: dape`) or empty (`prompt: empty`), one UNet step at t = 999 in latent space, decode, and colour
correction to the upsampled input (`color_fix: adain | wavelet | none`, official default adain).
OSEDiff's code hard-codes CUDA. Train resolution 512: use LR tiles of 128 (HR 512) with overlap.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F


class OSEDiffTiles(torch.nn.Module):
    def __init__(self, model, dape, color_fix: str):
        super().__init__()
        self.model, self.dape, self.color_fix = model, dape, color_fix


def build(spec: dict):
    sys.path.insert(0, spec["repo"])
    from osediff import OSEDiff_test
    args = SimpleNamespace(pretrained_model_name_or_path=spec["sd_path"], osediff_path=spec["weights"],
                           merge_and_unload_lora=False, mixed_precision=spec.get("precision", "fp16"),
                           vae_encoder_tiled_size=1024, vae_decoder_tiled_size=224,
                           latent_tiled_size=96, latent_tiled_overlap=32)
    model = OSEDiff_test(args)
    dape = None
    if spec.get("prompt", "dape") == "dape":
        from ram.models import ram_lora
        from ram.models.ram_lora import ram
        if spec.get("bert_path"):
            # the repo hard-codes a path of the authors' cluster for the BERT tokenizer; use a local copy instead
            from transformers import BertTokenizer

            def _init_tokenizer():
                tok = BertTokenizer.from_pretrained(spec["bert_path"], local_files_only=True)
                tok.add_special_tokens({"bos_token": "[DEC]"})
                tok.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
                tok.enc_token_id = tok.additional_special_tokens_ids[0]
                return tok

            ram_lora.init_tokenizer = _init_tokenizer
        dape = ram(pretrained=spec["ram_path"], pretrained_condition=spec["dape_path"], image_size=384, vit="swin_l")
        dape = dape.eval().to("cuda", dtype=model.weight_dtype)
    return OSEDiffTiles(model, dape, spec.get("color_fix", "adain"))


def wrap(net: OSEDiffTiles, spec: dict):
    from my_utils.wavelet_color_fix import adaptive_instance_normalization, wavelet_reconstruction
    from ram import inference_ram
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    extra = spec.get("extra_prompt", "")

    @torch.no_grad()
    def fn(x: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(x.float(), scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)
        outs = []
        for t in up:
            t = t[None]
            prompt = ""
            if net.dape is not None:
                r = (F.interpolate(t, size=(384, 384), mode="bicubic", align_corners=False) - mean.to(t)) / std.to(t)
                prompt = inference_ram(r.to(net.model.weight_dtype), net.dape)[0]
            prompt = f"{prompt}, {extra}," if extra or prompt else ""
            y = (net.model(t * 2 - 1, prompt=prompt).float() * 0.5 + 0.5).clamp(0, 1)
            if net.color_fix == "adain":
                y = adaptive_instance_normalization(y, t)
            elif net.color_fix == "wavelet":
                y = wavelet_reconstruction(y, t)
            outs.append(y.clamp(0, 1))
        return torch.cat(outs)

    return fn
