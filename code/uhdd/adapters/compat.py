"""Small compatibility shims so every adapter runs in one Python environment (newer diffusers, xformers built for another torch)."""
from __future__ import annotations

import os


def disable_xformers() -> None:
    """Third-party code that probes xformers (VOSR SwiGLU, DINOv2 hub code) falls back to plain PyTorch when this is set.
    Needed when the installed xformers wheel was built for a different torch and cannot load its CUDA extensions."""
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")   # VOSR's fallback SwiGLU uses @torch.compile; Triton may not parse the node's CUDA version


def diffusers_vae_mixin() -> None:
    """OSEDiff's autoencoder imports `diffusers.loaders.FromOriginalVAEMixin`, removed in newer diffusers
    (replaced by `FromOriginalModelMixin`); alias it so the unmodified OSEDiff code imports."""
    import diffusers.loaders as loaders
    if not hasattr(loaders, "FromOriginalVAEMixin") and hasattr(loaders, "FromOriginalModelMixin"):
        loaders.FromOriginalVAEMixin = loaders.FromOriginalModelMixin


def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    """transformers 4.x helper (head pruning, unused at inference) that transformers 5 dropped."""
    import torch
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    return heads, torch.arange(len(mask))[mask].long()


def osediff_imports() -> None:
    """OSEDiff (and its bundled BERT / UNet copies) import names that newer diffusers / transformers moved or renamed.
    Alias them so the unmodified OSEDiff code imports in the one main environment."""
    import sys
    diffusers_vae_mixin()
    import diffusers.models.embeddings as emb
    if not hasattr(emb, "PositionNet") and hasattr(emb, "GLIGENTextBoundingboxProjection"):
        emb.PositionNet = emb.GLIGENTextBoundingboxProjection
    try:
        import diffusers.models.unet_2d_blocks  # noqa: F401
    except ImportError:
        from diffusers.models.unets import unet_2d_blocks
        sys.modules["diffusers.models.unet_2d_blocks"] = unet_2d_blocks
    import transformers.modeling_utils as mu
    import transformers.pytorch_utils as pu
    for n in ("apply_chunking_to_forward", "prune_linear_layer"):
        if not hasattr(mu, n) and hasattr(pu, n):
            setattr(mu, n, getattr(pu, n))
    if not hasattr(mu, "find_pruneable_heads_and_indices"):
        mu.find_pruneable_heads_and_indices = getattr(pu, "find_pruneable_heads_and_indices", _find_pruneable_heads_and_indices)
    # LoRA: in old diffusers `add_adapter` & co. lived on the loader mixins that OSEDiff's UNet / VAE copies inherit;
    # newer diffusers moved them to PeftAdapterMixin. Copy the methods over so the bundled classes get them.
    from diffusers.loaders import PeftAdapterMixin, UNet2DConditionLoadersMixin
    import diffusers.loaders as loaders
    for cls in (UNet2DConditionLoadersMixin, loaders.FromOriginalVAEMixin):
        for name in dir(PeftAdapterMixin):
            if not name.startswith("__") and not hasattr(cls, name):
                setattr(cls, name, getattr(PeftAdapterMixin, name))
    # RAM's bundled (old) BERT copy never sets this attribute, which transformers 5 reads when loading weights
    from transformers import PreTrainedModel
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}
    # mask helpers of transformers 4 `PreTrainedModel` that RAM's bundled BERT calls (removed in transformers 5)
    import torch

    def invert_attention_mask(self, encoder_attention_mask):
        m = encoder_attention_mask[:, None, :, :] if encoder_attention_mask.dim() == 3 else encoder_attention_mask[:, None, None, :]
        m = m.to(dtype=self.dtype)
        return (1.0 - m) * torch.finfo(self.dtype).min

    def get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
        dtype = dtype or self.dtype
        if attention_mask.dim() == 3:
            ext = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            ext = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask {attention_mask.shape}")
        return (1.0 - ext.to(dtype=dtype)) * torch.finfo(dtype).min

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            raise NotImplementedError("head_mask is not used by OSEDiff / RAM")
        return [None] * num_hidden_layers

    for fn in (invert_attention_mask, get_extended_attention_mask, get_head_mask):
        if not hasattr(PreTrainedModel, fn.__name__):
            setattr(PreTrainedModel, fn.__name__, fn)


def seesr_imports() -> None:
    """SeeSR (used by ReFIR) bundles UNet / ControlNet / pipeline copies written against diffusers ~0.21: alias the
    modules and names that newer diffusers moved, plus the OSEDiff set (RAM / DAPE share the same BERT copy)."""
    import sys
    import types
    osediff_imports()
    import diffusers.loaders as loaders
    if not hasattr(loaders, "FromOriginalControlnetMixin"):
        loaders.FromOriginalControlnetMixin = loaders.FromOriginalModelMixin
    import diffusers.models.attention as att
    if not hasattr(att, "AdaGroupNorm"):
        from diffusers.models.normalization import AdaGroupNorm
        att.AdaGroupNorm = AdaGroupNorm
    moved = {"diffusers.pipeline_utils": "diffusers.pipelines.pipeline_utils",
             "diffusers.models.dual_transformer_2d": "diffusers.models.transformers.dual_transformer_2d",
             "diffusers.models.transformer_2d": "diffusers.models.transformers.transformer_2d",
             "diffusers.models.unet_2d_condition": "diffusers.models.unets.unet_2d_condition",
             "diffusers.models.controlnet": "diffusers.models.controlnets.controlnet"}
    import importlib
    for old, new in moved.items():
        if old not in sys.modules:
            try:
                importlib.import_module(old)
            except ImportError:
                sys.modules[old] = importlib.import_module(new)
    if sys.platform == "darwin" and "modules" not in sys.modules:   # SeeSR's utils/devices.py (A1111 heritage), macOS only
        pkg = types.ModuleType("modules")
        pkg.mac_specific = types.SimpleNamespace(has_mps=False)
        sys.modules["modules"] = pkg
        sys.modules["modules.mac_specific"] = pkg.mac_specific


def irag_imports() -> None:
    """iRAG (StableSR-style latent diffusion) imports pytorch-lightning and taming-transformers only for training
    (LightningModule base class, logging decorators, VQ / discriminator losses). For inference in the one main
    environment, install minimal stand-ins when the real packages are absent: LightningModule = nn.Module."""
    import sys
    import types

    import torch.nn as nn

    def _mod(name: str, **attrs) -> types.ModuleType:
        m = sys.modules.get(name) or types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(_mod(parent), child, m)
        return m

    try:
        import pytorch_lightning  # noqa: F401
    except ImportError:
        class LightningModule(nn.Module):
            def log(self, *a, **k):
                pass

            def log_dict(self, *a, **k):
                pass

        ident = lambda f: f  # noqa: E731
        _mod("pytorch_lightning", LightningModule=LightningModule, seed_everything=lambda *a, **k: None,
             Callback=object, Trainer=object)
        _mod("pytorch_lightning.utilities", rank_zero_only=ident)
        _mod("pytorch_lightning.utilities.distributed", rank_zero_only=ident)
        _mod("pytorch_lightning.utilities.rank_zero", rank_zero_only=ident)
    try:
        import taming  # noqa: F401
    except ImportError:
        class _Unavailable(nn.Module):
            def __init__(self, *a, **k):
                raise RuntimeError("taming-transformers is not installed (only needed for iRAG training / VQ models)")

        stub = lambda *a, **k: None  # noqa: E731
        _mod("taming.modules.vqvae.quantize", VectorQuantizer2=_Unavailable, VectorQuantizer=_Unavailable)
        _mod("taming.modules.losses.vqperceptual", hinge_d_loss=stub, vanilla_d_loss=stub, LPIPS=_Unavailable)
        _mod("taming.modules.losses.lpips", LPIPS=_Unavailable)
        _mod("taming.modules.discriminator.model", NLayerDiscriminator=_Unavailable, weights_init=stub)
        _mod("taming.data.imagenet", ImagePaths=_Unavailable, str_to_indices=stub, give_synsets_from_indices=stub,
             download=stub, retrieve=stub)
