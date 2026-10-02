"""DATSR (Cao et al., ECCV 2022, reference-based x4 SR) without mmcv.

DATSR imports `mmcv.ops.modulated_deform_conv2d` (modulated deformable conv, DCNv2) and `mmcv.scandir`.
`install_mmcv_shim` provides both: the conv via torchvision.ops.deform_conv2d (same offset layout:
per kernel point (dy, dx), offset groups inferred from the channel count; runs on CPU and CUDA), so the
original weights run unchanged. Package `__init__` files (which import every model / training dependency)
are bypassed by registering bare namespace modules.

Inputs of the three nets (as in datsr/models/ref_restoration_model.py, test()):
  img_in_lq  LR image (x4 smaller), img_in_up = bicubic x4 of it (for matching), img_ref = HR reference of
  the *same size* as the HR output (the deformable aggregation samples the ref feature map on the output grid).
All in [0, 1], RGB.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


def install_mmcv_shim() -> None:
    if "mmcv" in sys.modules and hasattr(sys.modules["mmcv"], "_uhdd_shim"):
        return
    import torchvision
    from torch.nn.modules.utils import _pair

    def modulated_deform_conv2d(x, offset, mask, weight, bias, stride=1, padding=0, dilation=1, groups=1,
                                deform_groups=1):
        if groups != 1:
            raise NotImplementedError("groups > 1")
        return torchvision.ops.deform_conv2d(x, offset, weight, bias, stride=_pair(stride), padding=_pair(padding),
                                             dilation=_pair(dilation), mask=mask)

    mm, ops = types.ModuleType("mmcv"), types.ModuleType("mmcv.ops")
    ops.modulated_deform_conv2d = modulated_deform_conv2d
    ops.ModulatedDeformConv2d = torch.nn.Module            # imported by dcn_v2.py, never instantiated
    mm.ops, mm.scandir, mm._uhdd_shim = ops, (lambda d, *a, **k: iter(sorted(os.listdir(d)))), True
    sys.modules["mmcv"], sys.modules["mmcv.ops"] = mm, ops


def _namespace(repo: Path) -> None:
    for name, sub in (("datsr", "datsr"), ("datsr.models", "datsr/models"), ("datsr.models.archs", "datsr/models/archs")):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [str(repo / sub)]
            sys.modules[name] = m


def build(repo: str, weights_g: str, weights_extractor: str, device="cpu"):
    """-> (net_g, net_map, net_extractor), eval mode, on `device` (config of options/test/test_restoration_*.yml)."""
    install_mmcv_shim()
    repo = Path(repo)
    sys.path.insert(0, str(repo))
    _namespace(repo)
    import importlib
    g = importlib.import_module("datsr.models.archs.swin_unetv3_ref_restoration_arch")
    fm = importlib.import_module("datsr.models.archs.flow_similarity_corres_generation_arch")
    ce = importlib.import_module("datsr.models.archs.contras_extractor_arch")
    net_g = g.SwinUnetv3RestorationNet(ngf=128, n_blocks=8, groups=8, embed_dim=128, depths=[4, 4], num_heads=[4, 4],
                                       window_size=8, use_checkpoint=False)
    net_g.load_state_dict(torch.load(weights_g, map_location="cpu", weights_only=False), strict=True)
    net_map = fm.FlowSimCorrespondenceGenerationArch(patch_size=3, stride=1, vgg_layer_list=["relu1_1", "relu2_1", "relu3_1"],
                                                    vgg_type="vgg19")
    net_ex = ce.ContrasExtractorSep()
    net_ex.load_state_dict(torch.load(weights_extractor, map_location="cpu", weights_only=False), strict=True)
    nets = tuple(n.eval().to(device) for n in (net_g, net_map, net_ex))
    for n in nets:
        for p in n.parameters():
            p.requires_grad_(False)
    return nets


@torch.inference_mode()
def run(nets, lq: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """lq: Bx3xhxw, ref: Bx3x4hx4w (both [0,1]) -> Bx3x4hx4w."""
    net_g, net_map, net_ex = nets
    up = F.interpolate(lq, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)
    feats = net_ex(up, ref)
    pre_offset, ref_feat = net_map(feats, ref)
    return net_g(lq, pre_offset, ref_feat).clamp(0, 1)
