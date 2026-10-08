"""ReFIR (Guo et al., NeurIPS 2024) on SeeSR (Wu et al., CVPR 2024): training-free reference injection into a
multi-step diffusion SR model (github.com/csguoh/ReFIR, `seesr/test_real.py`).

Per tile, as the official script does per image: DAPE / RAM tags of the input and of the reference as prompts
(+ "clean, high-resolution, 8k"), bicubic x4 of the anchor tile and the reference denoised together as a batch of 2
with classifier-free guidance (5.5, negative "dotted, noise, blur, lowres, smooth"), 50 steps, start from the LR
latent; in the late decoder self-attention layers of the last 20 steps the input's attention output becomes
(1 - m) self + m cross(input -> reference) (m = per-position mean similarity, min-max normalised), AdaIN colour
correction to the input. Without a reference (or `refir_scale: 0`) it is plain SeeSR on the input.

Deviations from the official script (all switchable in the spec):
  - `reset_editor: true`: the attention editor's step counter is reset before every tile. The official script
    registers it once and never resets it, so from the second image on the injection runs at every step instead of
    the last 20 (`reset_editor: false` reproduces that).
  - xformers calls are replaced by torch scaled-dot-product attention (same maths) so it runs without xformers.
  - the reference is given at the tile size (512): no 2x reference with mirror padding of the input.
"""
from __future__ import annotations

import contextlib
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@contextlib.contextmanager
def _cwd(path: str):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _sdpa_editor(seesr_register) -> None:
    """Replace ReFIR's xformers attention with torch SDPA (inputs [(b h), n, d], scale passed through)."""
    from einops import rearrange

    def attn(q, k, v, scale):
        return F.scaled_dot_product_attention(q, k, v, scale=scale)

    def base_forward(self, q, k, v, sim, a, is_cross, place_in_unet, num_heads, **kw):
        return rearrange(attn(q, k, v, kw.get("scale")), "(b h) n d -> b n (h d)", h=num_heads)

    def attn_batch(self, q, k, v, num_heads, **kw):
        return rearrange(attn(q, k, v, kw.get("scale")), "(b h) n d -> b h n d", h=num_heads)

    seesr_register.AttentionBase.forward = base_forward
    seesr_register.MutualSelfAttentionControl.attn_batch = attn_batch


class ReFIRSeeSR:
    def __init__(self, spec: dict, device="cuda"):
        from .compat import seesr_imports
        seesr_imports()
        repo = spec["repo"]
        sys.path.insert(0, repo)
        self.spec, self.device = spec, torch.device(device)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(spec.get("precision", "fp16"), torch.float32)
        self.dtype = dtype
        with _cwd(repo):
            from diffusers import AutoencoderKL, DDPMScheduler
            from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
            from models.controlnet import ControlNetModel
            from models.unet_2d_condition import UNet2DConditionModel
            from pipelines.pipeline_seesr import StableDiffusionControlNetPipeline
            import seesr_register
            from ram.models import ram_lora
            from ram.models.ram_lora import ram
            from ram import inference_ram
            from utils.wavelet_color_fix import adain_color_fix, wavelet_reconstruction
            _sdpa_editor(seesr_register)
            sd, ss = spec["sd_path"], spec["seesr_path"]
            pipe = StableDiffusionControlNetPipeline(
                vae=AutoencoderKL.from_pretrained(sd, subfolder="vae"),
                text_encoder=CLIPTextModel.from_pretrained(sd, subfolder="text_encoder"),
                tokenizer=CLIPTokenizer.from_pretrained(sd, subfolder="tokenizer"),
                feature_extractor=CLIPImageProcessor.from_pretrained(f"{sd}/feature_extractor"),
                unet=UNet2DConditionModel.from_pretrained(ss, subfolder="unet"),
                controlnet=ControlNetModel.from_pretrained(ss, subfolder="controlnet"),
                scheduler=DDPMScheduler.from_pretrained(sd, subfolder="scheduler"),
                safety_checker=None, requires_safety_checker=False)
            pipe._init_tiled_vae(encoder_tile_size=spec.get("vae_tile", 2048), decoder_tile_size=spec.get("vae_tile", 2048))
            for m in (pipe.text_encoder, pipe.vae, pipe.unet, pipe.controlnet):
                m.requires_grad_(False)
                m.to(self.device, dtype=dtype)
            if spec.get("bert_path"):
                from transformers import BertTokenizer

                def _init_tokenizer():   # the repo hard-codes a path on the authors' cluster
                    tok = BertTokenizer.from_pretrained(spec["bert_path"], local_files_only=True)
                    tok.add_special_tokens({"bos_token": "[DEC]"})
                    tok.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
                    tok.enc_token_id = tok.convert_tokens_to_ids("[ENC]")
                    return tok

                ram_lora.init_tokenizer = _init_tokenizer
            self.tagger = ram(pretrained=spec["ram_path"], pretrained_condition=spec["dape_path"], image_size=384,
                              vit="swin_l").eval().to(self.device)
            self.editor = None
            if float(spec.get("refir_scale", 0.6)) > 0:
                self.editor = seesr_register.MutualSelfAttentionControl(float(spec.get("refir_scale", 0.6)))
                seesr_register.regiter_attention_editor_diffusers(pipe, self.editor)
        self.pipe, self.inference_ram = pipe, inference_ram
        self.adain, self.wavelet = adain_color_fix, wavelet_reconstruction

    @torch.no_grad()
    def _prompt(self, img: torch.Tensor):
        """img 1x3xHxW in [0, 1] -> (tag prompt, RAM image embeddings), as get_validation_prompt()."""
        x = (F.interpolate(img.float(), size=(384, 384), mode="bilinear", align_corners=False) - MEAN.to(img.device)) / STD.to(img.device)
        x = x.to(self.device)
        tags = self.inference_ram(x, self.tagger)[0]
        emb = self.tagger.generate_image_embeds(x)
        return f"{tags}, {self.spec.get('prompt', '')},{self.spec.get('added_prompt', 'clean, high-resolution, 8k')}", emb

    @torch.no_grad()
    def __call__(self, anchor: torch.Tensor, ref: torch.Tensor | None) -> torch.Tensor:
        s = self.spec
        up = F.interpolate(anchor.float(), scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1).to(self.device)
        H, W = up.shape[-2:]
        p_in, e_in = self._prompt(anchor)
        neg = s.get("negative_prompt", "dotted, noise, blur, lowres, smooth")
        # ReFIR's UNet always runs [input, reference] as a pair (its similarity mask indexes batch 0-3). Without a
        # reference the pair is [input, input] with the injection off (scale 0): exactly SeeSR on the input.
        use_ref = self.editor is not None and ref is not None
        if use_ref:
            r = ref.float().to(self.device)
            if r.shape[-2:] != (H, W):
                r = F.interpolate(r, size=(H, W), mode="bicubic", align_corners=False).clamp(0, 1)
            p_ref, e_ref = self._prompt(r)
        else:
            r, p_ref, e_ref = up, p_in, e_in
        img, prompts, embs = torch.cat([up, r]), [p_in, p_ref], torch.cat([e_in, e_ref])
        if self.editor is not None:
            if s.get("reset_editor", True):
                self.editor.reset()
            self.editor.refir_scale = float(s.get("refir_scale", 0.6)) if use_ref else 0.0
        g = torch.Generator(device=self.device).manual_seed(int(s.get("seed", 1234)))
        ctx = torch.autocast("cuda") if self.device.type == "cuda" else contextlib.nullcontext()
        with ctx:
            out = self.pipe(prompts, img, num_inference_steps=int(s.get("steps", 50)), generator=g, height=H, width=W,
                            guidance_scale=float(s.get("guidance", 5.5)), negative_prompt=[neg] * len(prompts),
                            conditioning_scale=float(s.get("conditioning_scale", 1.0)), start_point=s.get("start_point", "lr"),
                            ram_encoder_hidden_states=embs, latent_tiled_size=2048, latent_tiled_overlap=32,
                            args=SimpleNamespace()).images
        out = out.float().cpu()
        align = s.get("align", "adain")
        if align == "adain":
            out = self.adain(out, img.float().cpu())          # official: AdaIN of the batch to [input, reference]
        elif align == "wavelet":
            out = self.wavelet(out, img.float().cpu())
        return out[:1, :, :H, :W].clamp(0, 1)


def build(spec: dict, device="cuda"):
    return ReFIRSeeSR(spec, device)
