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
