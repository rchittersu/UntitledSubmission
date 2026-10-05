"""Results layout: one root, organised by stage -> input source -> chain (docs/evaluation.md "Results layout").

    $UHDD_RESULTS/dpdd/
      inputs/<src>/{inputs,targets,masks} -> symlinks into $UHDD_DATA (setup_inputs.py), manifest.json
      inputs/dp_maps                      -> DP blur maps (native)
      deblur/<src>/<model>@<tiling>/                     PNGs + meta.json + launch.json (+ metrics_<tag>.*)
      upsample/<src>/<anchor>@<tiling>/<sr>@<tiling>/    same; <src> = where the anchor's input came from
      fusion/<src>/<anchor>@<tiling>/<method>/           training-free fusions, reference-based SR
      tables/<tag>/
      scratch/                                           8-bit copies, scene subsets (safe to delete)

Metrics live next to the images. Legacy roots (LEGACY) are never written by anything that uses this module.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT_NAME = "dpdd"
LEGACY = ("dpdd_v2", "dpdd_official_x4", "dpdd_p1")   # earlier results roots under $UHDD_RESULTS: read-only

# input sources: scale (native / this resolution) and folders relative to $UHDD_DATA
SOURCES = {
    "ours_x1": {"scale": 1, "inputs": "dpdd_native_v2/inputs", "targets": "dpdd_native_v2/x1/targets",
                "masks": "dpdd_native_v2/x1/masks"},
    "ours_x2": {"scale": 2, "inputs": "dpdd_native_v2/x2/inputs", "targets": "dpdd_native_v2/x2/targets",
                "masks": "dpdd_native_v2/x2/masks"},
    "ours_x4": {"scale": 4, "inputs": "dpdd_native_v2/x4/inputs", "targets": "dpdd_native_v2/x4/targets",
                "masks": "dpdd_native_v2/x4/masks"},
    # the original DPDD test pairs (1680x1120); no masks. Upsampled to native they are scored against ours_x1
    # (rendering mismatch -> diagnostic only)
    "official_x4": {"scale": 4, "inputs": "dpdd_official/inputs", "targets": "dpdd_official/targets", "masks": None},
}
NATIVE = "ours_x1"
DP_MAPS = "dpdd_native/dp_maps"
CROP_NATIVE = 64        # border removed before evaluation, native px (= 64 / scale at lower resolutions)


def root() -> Path:
    r = os.environ.get("UHDD_RESULTS")
    if not r:
        raise SystemExit("set $UHDD_RESULTS")
    return Path(r) / ROOT_NAME


def data_root() -> Path:
    d = os.environ.get("UHDD_DATA")
    if not d:
        raise SystemExit("set $UHDD_DATA")
    return Path(d)


def source(src: str) -> dict:
    if src not in SOURCES:
        raise ValueError(f"unknown source '{src}' (one of {', '.join(SOURCES)})")
    return SOURCES[src]


def input_dir(src: str, kind: str = "inputs") -> Path:
    """inputs / targets / masks of a source, through the symlinks made by setup_inputs.py."""
    return root() / "inputs" / src / kind


def dp_maps_dir() -> Path:
    return root() / "inputs" / "dp_maps"


def tag(model: str, tile: int = 0, overlap: int = 0, extra: str = "") -> str:
    """Folder name of one step: <model>@whole or <model>@t<tile>o<overlap>[-extra]."""
    t = f"{model}@" + (f"t{int(tile)}o{int(overlap)}" if tile else "whole") + (f"-{extra}" if extra else "")
    return re.sub(r"[^A-Za-z0-9_.@+-]", "_", t)


def deblur_dir(src: str, model_tag: str) -> Path:
    source(src)
    return root() / "deblur" / src / model_tag


def upsample_dir(src: str, anchor_tag: str, sr_tag: str) -> Path:
    source(src)
    return root() / "upsample" / src / anchor_tag / sr_tag


def fusion_dir(src: str, anchor_tag: str, method: str) -> Path:
    source(src)
    return root() / "fusion" / src / anchor_tag / method


def scratch_dir(*parts: str) -> Path:
    return root().joinpath("scratch", *parts)


def tables_dir(eval_tag: str) -> Path:
    return root() / "tables" / eval_tag


def legacy_dir(stage: str, src: str, model_tag: str, sr_tag: str | None = None) -> Path | None:
    """Where the same run lives in the legacy roots (handoff 2/3), for comparing fresh results with old ones."""
    res = root().parent
    if src == "official_x4":
        return res / "dpdd_official_x4" / "steps" / f"x1__{model_tag}" if stage == "deblur" else None
    key = f"x{source(src)['scale']}__{model_tag}" + (f"__{sr_tag}" if sr_tag else "")
    return res / "dpdd_v2" / "steps" / key


def stage_of(path: Path) -> tuple[str, str]:
    """(stage, src) of a run folder inside the root."""
    rel = Path(path).resolve().relative_to(root().resolve()).parts
    return rel[0], rel[1]
