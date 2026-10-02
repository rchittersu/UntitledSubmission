"""HAT (Chen et al., CVPR 2023) without installing basicsr.

hat/archs/hat_arch.py imports `basicsr.utils.registry.ARCH_REGISTRY` (only for its decorator) and
`basicsr.archs.arch_util.{to_2tuple, trunc_normal_}`; both are provided by a minimal shim (timm's versions),
then the single architecture file is loaded. Inputs must be multiples of the window size (16): the
registry entry sets `multiple: 16` so uhdd pads, as HAT's own test code does.
"""
from __future__ import annotations

import sys
import types


def _shim_basicsr() -> None:
    if "basicsr" in sys.modules:
        return
    from timm.layers import to_2tuple, trunc_normal_

    class _Registry:
        def register(self, obj=None):
            return obj if obj is not None else (lambda o: o)

    mods = {n: types.ModuleType(n) for n in ("basicsr", "basicsr.utils", "basicsr.utils.registry", "basicsr.archs",
                                             "basicsr.archs.arch_util")}
    mods["basicsr.utils.registry"].ARCH_REGISTRY = _Registry()
    mods["basicsr.archs.arch_util"].to_2tuple = to_2tuple
    mods["basicsr.archs.arch_util"].trunc_normal_ = trunc_normal_
    sys.modules.update(mods)


def build(spec: dict):
    _shim_basicsr()
    from uhdd.models import _import
    return _import("hat/archs/hat_arch.py:HAT", spec["repo"])(**spec.get("kwargs", {}))
