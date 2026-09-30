"""Model registry: wraps third-party restoration / SR networks behind one interface.

Every model is a `Restorer`: NCHW float in [0,1] -> NCHW float in [0,1], output `scale`x
larger, input H/W must be multiples of `multiple` (handled by tiling/padding).

Models are declared in a YAML file (see configs/models.yaml). `kind: torch` entries are built
generically from the original repository (added to sys.path) + a checkpoint; anything unusual
can point to an `adapter: module:function` that receives the built nn.Module and the spec and
returns a callable. Prefer `import: path/to/arch_file.py:Class` (loaded in isolation); with
dotted package imports load one third-party model per process, since many repos ship a package
with the same name (e.g. `basicsr`), which would clash in `sys.modules`.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
import yaml


@dataclass
class Restorer:
    name: str
    fn: Callable[[torch.Tensor], torch.Tensor]
    multiple: int = 1
    scale: int = 1
    spec: dict = field(default_factory=dict)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


def _expand(v: Any) -> Any:
    """Expand ${ENV_VARS} recursively; unresolved variables are an error (no silent paths)."""
    if isinstance(v, str):
        out = os.path.expandvars(v)
        if re.search(r"\$\{?\w+", out):
            raise KeyError(f"unresolved environment variable in '{v}'")
        return out
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    return v


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)["models"]


def _import(path: str, repo: str | None = None):
    """'pkg.module:Name', or 'relative/file.py:Name' (resolved against `repo`).

    Loading a single architecture file skips the repo's package __init__ files (which often
    pull in training-only dependencies) and avoids clashes between repos that ship packages
    with the same name.
    """
    mod, _, attr = path.rpartition(":")
    if mod.endswith(".py"):
        file = Path(repo or ".") / mod
        name = f"_uhdd_ext_{abs(hash(str(file.resolve())))}"
        spec = importlib.util.spec_from_file_location(name, file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return getattr(module, attr)
    return getattr(importlib.import_module(mod), attr)


def _load_state_dict(model: torch.nn.Module, spec: dict) -> None:
    try:
        ckpt = torch.load(spec["weights"], map_location="cpu", weights_only=True)
    except Exception:  # older checkpoints pickle extra objects; only load trusted weights
        ckpt = torch.load(spec["weights"], map_location="cpu", weights_only=False)
    key = spec.get("state_key")
    sd = ckpt[key] if key else ckpt
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=spec.get("strict", True))
    if missing or unexpected:
        print(f"[{spec['name']}] missing={len(missing)} unexpected={len(unexpected)}", file=sys.stderr)


def _wrap(net: Callable, spec: dict) -> Callable[[torch.Tensor], torch.Tensor]:
    lo, hi = spec.get("input_range", [0.0, 1.0])
    out_idx = spec.get("output_index")

    def fn(x: torch.Tensor) -> torch.Tensor:
        if (lo, hi) != (0.0, 1.0):
            x = x * (hi - lo) + lo
        y = net(x)
        if out_idx is not None:
            y = y[out_idx]
        if (lo, hi) != (0.0, 1.0):
            y = (y - lo) / (hi - lo)
        return y.clamp(0, 1) if spec.get("clamp", True) else y

    return fn


def _builtin(spec: dict) -> Callable:
    kind, s = spec["op"], spec.get("scale", 1)
    if kind == "identity":
        return lambda x: x
    if kind == "bicubic":
        return lambda x: F.interpolate(x, scale_factor=s, mode="bicubic", align_corners=False).clamp(0, 1)
    raise ValueError(f"unknown builtin {kind}")


def build(name: str, config: str | Path, device: torch.device | str,
          channels_last: bool = False, compile: bool = False) -> Restorer:
    spec = _expand(load_config(config)[name])
    spec["name"] = name
    kind = spec.get("kind", "torch")

    if kind == "builtin":
        fn = _builtin(spec)
    elif kind == "torch":
        if "repo" in spec:
            sys.path.insert(0, spec["repo"])
        net = _import(spec["import"], spec.get("repo"))(**spec.get("kwargs", {}))
        if spec.get("weights"):
            _load_state_dict(net, spec)
        net = net.eval().to(device)
        if channels_last:
            net = net.to(memory_format=torch.channels_last)
        for p in net.parameters():
            p.requires_grad_(False)
        if compile:
            net = torch.compile(net)
        fn = _import(spec["adapter"], spec.get("repo"))(net, spec) if "adapter" in spec else _wrap(net, spec)
    else:
        raise ValueError(f"unknown kind {kind}")

    return Restorer(name, fn, int(spec.get("multiple", 1)), int(spec.get("scale", 1)), spec)
