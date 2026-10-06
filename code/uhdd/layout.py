"""Results layout: one root, organised by stage -> input source -> chain (docs/evaluation.md "Results layout").

    $UHDD_RESULTS/dpdd/
      inputs/<src>/{inputs,targets,masks} -> symlinks into $UHDD_DATA (setup_inputs.py), manifest.json
      inputs/dp_maps, inputs/<split>_dp_maps -> DP maps (1680x1120) of the test / train / val split
      cache/<split>/                      -> training cache (build_train_cache.py)
      deblur/<src>/<model>@<tiling>/                     PNGs + meta.json + launch.json (+ metrics_<tag>.*)
      upsample/<src>/<anchor>@<tiling>/<sr>@<tiling>/    same; <src> = where the anchor's input came from
      fusion/<src>/<anchor>@<tiling>/<method>/           training-free fusions, reference-based SR
      tables/<tag>/
      scratch/                                           scene subsets (safe to delete)

Metrics live next to the images. Legacy roots (LEGACY) are never written by anything that uses this module.
"""
from __future__ import annotations

import json
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
    # training / validation splits of the native set (method training; val = model selection only)
    "train_x1": {"scale": 1, "inputs": "dpdd_native_train/inputs", "targets": "dpdd_native_train/x1/targets",
                 "masks": "dpdd_native_train/x1/masks"},
    "train_x4": {"scale": 4, "inputs": "dpdd_native_train/x4/inputs", "targets": "dpdd_native_train/x4/targets",
                 "masks": "dpdd_native_train/x4/masks"},
    "val_x1": {"scale": 1, "inputs": "dpdd_native_val/inputs", "targets": "dpdd_native_val/x1/targets",
               "masks": "dpdd_native_val/x1/masks"},
    "val_x4": {"scale": 4, "inputs": "dpdd_native_val/x4/inputs", "targets": "dpdd_native_val/x4/targets",
               "masks": "dpdd_native_val/x4/masks"},
}
# split of every source, its native source (targets of upsampled outputs), DP maps, expected scene count
SPLITS = {
    "test": {"native": "ours_x1", "dp_maps": "dpdd_native/dp_maps", "n": 76,
             "sources": ["ours_x1", "ours_x2", "ours_x4", "official_x4"]},
    "train": {"native": "train_x1", "dp_maps": "dpdd_native_train/dp_maps", "n": 350, "sources": ["train_x1", "train_x4"]},
    "val": {"native": "val_x1", "dp_maps": "dpdd_native_val/dp_maps", "n": 74, "sources": ["val_x1", "val_x4"]},
}
NATIVE = SPLITS["test"]["native"]
DP_MAPS = SPLITS["test"]["dp_maps"]


def _apply_overrides() -> None:
    """Optional $UHDD_DATA/uhdd_sources.json: {"sources": {name: {...}}, "splits": {split: {...}}} replaces folder
    names (relative to $UHDD_DATA) where the secure data tree differs from the defaults above."""
    d = os.environ.get("UHDD_DATA")
    f = Path(d) / "uhdd_sources.json" if d else None
    if f and f.exists():
        o = json.loads(f.read_text())
        for k, v in o.get("sources", {}).items():
            SOURCES.setdefault(k, {}).update(v)
        for k, v in o.get("splits", {}).items():
            SPLITS.setdefault(k, {}).update(v)


_apply_overrides()
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


def split_of(src: str) -> str:
    for k, v in SPLITS.items():
        if src in v["sources"]:
            return k
    raise ValueError(f"source {src} belongs to no split")


def native_of(src: str) -> str:
    """Native-resolution source of the same split (targets for upsampled outputs)."""
    return SPLITS[split_of(src)]["native"]


def dp_maps_dir(split: str = "test") -> Path:
    return root() / "inputs" / ("dp_maps" if split == "test" else f"{split}_dp_maps")


def cache_dir(split: str) -> Path:
    return root() / "cache" / split


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
