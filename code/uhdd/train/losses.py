"""Losses (plan/method_plan.md §4): L1 on the validity mask everywhere; an alignment-tolerant contextual texture loss
(Mechrez et al. 2018) on VGG features, only where the input is defocused (the f/22 target is slightly misregistered
and noisier, so pixel losses alone teach the mean and treat transferred texture as noise)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def l1(pred, target, mask):
    return ((pred - target).abs() * mask).sum() / (mask.sum() * pred.shape[1] + 1e-8)


class VGGFeatures(nn.Module):
    """torchvision VGG19 up to relu3_1 (stride 4). weights: path to torchvision's vgg19 state dict (secure env is
    offline) or None (random init: tests only)."""

    def __init__(self, weights: str | None = None, layer: int = 12):
        super().__init__()
        from torchvision.models import vgg19
        net = vgg19()
        if weights:
            net.load_state_dict(torch.load(weights, map_location="cpu"))
        self.f = net.features[:layer].eval()
        for p in self.f.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.f((x - self.mean) / self.std)


def contextual(fp: torch.Tensor, ft: torch.Tensor, h: float = 0.5) -> torch.Tensor:
    """Contextual loss between feature sets fp (N, C) and ft (M, C): -log mean_j max_i CX_ij."""
    mu = ft.mean(0, keepdim=True)
    p, t = F.normalize(fp - mu, dim=1), F.normalize(ft - mu, dim=1)
    d = 1 - p @ t.T                                         # cosine distance (N, M)
    d = d / (d.min(1, keepdim=True).values + 1e-5)
    w = torch.exp((1 - d) / h)
    cx = w / w.sum(1, keepdim=True)
    return -torch.log(cx.max(0).values.mean() + 1e-5)


class Loss(nn.Module):
    """cfg: l1 (weight), cx (weight), cx_points (features sampled per image), vgg_weights."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.vgg = VGGFeatures(cfg.get("vgg_weights")) if cfg.get("cx", 0) > 0 else None

    def forward(self, pred, b, gen: torch.Generator | None = None) -> tuple[torch.Tensor, dict]:
        c = self.cfg
        logs = {}
        loss = c.get("l1", 1.0) * l1(pred, b["y"], b["m"])
        logs["l1"] = loss.detach()
        if self.vgg is not None:
            fp, ft = self.vgg(pred), self.vgg(b["y"])
            reg = F.avg_pool2d(b["regime"] * b["m"], pred.shape[-1] // fp.shape[-1]) > 0.5
            cx, n = pred.new_zeros(()), 0
            for i in range(pred.shape[0]):
                sel = reg[i, 0].flatten().nonzero()[:, 0]
                if len(sel) < 16:
                    continue
                perm = sel[torch.randperm(len(sel), device=sel.device)[: c.get("cx_points", 1024)]]
                a, t = fp[i].flatten(1)[:, perm].T, ft[i].flatten(1)[:, perm].T
                cx, n = cx + contextual(a.float(), t.float()), n + 1
            if n:
                cx = cx / n
                loss = loss + c["cx"] * cx
                logs["cx"] = cx.detach()
        logs["loss"] = loss.detach()
        return loss, logs
