"""IFAN (Lee et al., CVPR 2021). Repo: github.com/codeslake/IFAN, weights ckpt/IFAN.pytorch.

The network is built from the repo's easydict config (configs/config_IFAN.py) and returns a
dict; 'result' is the deblurred image (clipped to [0,1] in eval mode). Official test: 8-bit
input in [0,1], image cropped to a multiple of 8 (`refine_image(.., 8)`); we pad instead.
The checkpoint also holds the training-only reblurring network (`reblurNet.*`), which is dropped.
Also accepts `IFAN.safetensors` from github.com/jacobsparts/ifan-rs releases (v0.1.0): a
conversion of IFAN.pytorch holding only the deblurring network, verified bit-identical (all 158
tensors, max abs diff 0) to the official checkpoint; public GitHub link, no Dropbox needed.
"""
from __future__ import annotations

import torch


def build(spec: dict) -> torch.nn.Module:
    from configs.config_IFAN import get_config  # repo is on sys.path
    from models.archs.IFAN import Network
    cfg = get_config("IFAN_CVPR2021", "IFAN", spec.get("config", "config_IFAN"))
    cfg.network = "IFAN"
    net = Network(cfg)
    if str(spec["weights"]).endswith(".safetensors"):  # e.g. the ifan-rs conversion (Network only)
        from safetensors.torch import load_file
        sd = load_file(spec["weights"])
    else:
        sd = torch.load(spec["weights"], map_location="cpu")
    # checkpoint = {"module.Network.*": deblurring net, "module.reblurNet.*": training-only reblurring net};
    # the official loader relies on strict=False, we keep only the deblurring net and insist
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    if any(k.startswith("Network.") for k in sd):
        sd = {k[len("Network."):]: v for k, v in sd.items() if k.startswith("Network.")}
    net.load_state_dict(sd, strict=True)
    return net


def wrap(net: torch.nn.Module, spec: dict):
    return lambda x: net(x)["result"].clamp(0, 1)
