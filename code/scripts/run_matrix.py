#!/usr/bin/env python
"""Run a matrix of pipelines (deblur model x resolution x tiling x upsampler) and evaluate them.

  run_matrix.py experiments/dpdd_p1.yaml [--gpus all] [--only REGEX] [--dry-run]

Experiment file (see experiments/dpdd_p1.yaml):
  data:     inputs/targets/masks folder per scale factor (1 = native, 4 = 1/4 resolution)
  results:  output root
  run:      default run_model.py options (tile, overlap, tile_batch, blend, precision, ...).
            Without `tile` (in run or step), or with tile: paper, each step uses its model's paper inference
            setup (uhdd.tiling.default_tiling: registry paper_input / paper_tile) at that step's input size;
            without `overlap`, overlap = tile * run.overlap_ratio (default 1/8). Resolved values are part of the
            cache key, so changing a default creates new step folders.
  eval:     metrics, native border crop (divided by the evaluated scale), extra options
  pipelines: list of {group, name, for, steps}; `for` is a grid of template variables that
             are substituted into name/steps with str.format, e.g.
               - group: g2_lowres_upsample
                 name: "{m}@x4+{up}"
                 for: {m: [restormer_dpdd, ifan], up: [bicubic_x4, swinir_x4]}
                 steps:
                   - {model: "{m}", scale: 4, tile: 0}
                   - {model: "{up}"}
The first step reads the data inputs at `scale`; each later step reads the previous output.
A step's output scale = input scale / model scale (SR models have scale 4). The final output
is evaluated against the targets at its scale.

Evaluations are cached per final step and eval `tag`: change the tag (or delete
metrics_<tag>.json) after changing metrics. Step outputs are cached under <results>/steps/<chain key>, where the key encodes the full
chain (input scale, models, tiling), so shared prefixes are computed once. Finished steps
(meta.json covering all images) and finished evaluations are skipped: the runner is resumable.
Per group, a Markdown table is written to <results>/tables/<group>.md.
If eval.metrics contains `gridshift`, the last step of every tiled pipeline is also run on a tile
grid shifted by half a stride (<step>@shift<k>) and passed to evaluate.py --pred-shift.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from uhdd.io import list_images  # noqa: E402
from uhdd.models import _expand, load_config  # noqa: E402

RUN_KEYS = ["tile", "overlap", "tile_batch", "blend", "precision", "input_bits", "save_bits"]


def expand_pipelines(spec: list[dict]) -> list[dict]:
    out = []
    for p in spec:
        grid = p.get("for", {})
        keys = list(grid)
        for combo in itertools.product(*(grid[k] for k in keys)) if keys else [()]:
            env = dict(zip(keys, combo))
            fmt = lambda v: v.format(**env) if isinstance(v, str) else v
            out.append({"group": p.get("group", "main"), "name": fmt(p["name"]),
                        "steps": [{k: fmt(v) for k, v in s.items()} for s in p["steps"]]})
    return out


def step_tag(step: dict, run: dict) -> str:
    tile = step.get("tile", run.get("tile", 0))
    tag = f"{step['model']}@" + (f"t{tile}o{step.get('overlap', run.get('overlap', 64))}" if tile else "whole")
    for k in ("blend", "precision", "input_bits"):
        if k in step:
            tag += f"-{k}{step[k]}"
    return re.sub(r"[^A-Za-z0-9_.@+-]", "_", tag)



def pipeline_cost(steps: list[dict]) -> dict:
    """Per-image pipeline cost: time summed over steps, peak memory = max over steps; mean/std/n over images
    (+ the worst image's memory), consistent with the per-image columns written by pipeline_csv."""
    import numpy as np
    names = list(steps[-1])
    t = np.array([sum(st.get(k, {}).get("time_s") or 0 for st in steps) for k in names], float)
    m = np.array([max((st.get(k, {}).get("peak_mem_gb") or 0) for st in steps) for k in names], float)
    return {"pipeline_time_s": {"mean": float(t.mean()), "std": float(t.std()), "n": len(t)},
            "peak_mem_gb": {"mean": float(m.mean()), "std": float(m.std()), "n": len(m)},
            "peak_mem_gb_max": {"mean": float(m.max()), "std": 0.0, "n": len(m)}}


def pipeline_csv(src: Path, steps: list[dict], dst: Path) -> Path:
    """Copy of the final step's per-image metrics CSV with pipeline_time_s / peak_mem_gb over all steps."""
    import csv as _csv
    rows = list(_csv.DictReader(open(src)))
    for r in rows:
        k = r["name"]
        r["pipeline_time_s"] = sum(st.get(k, {}).get("time_s") or 0 for st in steps)
        r["peak_mem_gb"] = max((st.get(k, {}).get("peak_mem_gb") or 0) for st in steps)
    with open(dst, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(dict.fromkeys(c for r in rows for c in r)))
        w.writeheader()
        w.writerows(rows)
    return dst

def native_hw(exp: dict) -> tuple[int, int]:
    """(H, W) of the native inputs: data.native_hw if given, else the first native input's header."""
    if "native_hw" in exp["data"]:
        return tuple(exp["data"]["native_hw"])
    from PIL import Image
    first = next(iter(sorted(list_images(exp["data"]["inputs"][1]).values())))
    with Image.open(first) as im:
        return im.size[1], im.size[0]


def resolve_tiling(st: dict, run: dict, spec: dict, hw: tuple[int, int]) -> dict:
    """Fill tile / overlap of a step: explicit values win; else the model's paper setup; overlap = tile * ratio."""
    from uhdd.tiling import OVERLAP_RATIO, default_tiling
    st = dict(st)
    ratio = float(run.get("overlap_ratio", OVERLAP_RATIO))
    tile = st.get("tile", run.get("tile"))
    if tile is None or tile == "paper":
        t, o = default_tiling(spec, hw, ratio)
        st["tile"] = t
        st.setdefault("overlap", run.get("overlap", o))
    else:
        st["tile"] = int(tile)
        if st.get("overlap", run.get("overlap")) is None:
            st["overlap"] = int(round(st["tile"] * ratio / 2)) * 2
    return st


def plan(exp: dict, model_cfg: dict) -> list[dict]:
    """Resolve each pipeline into concrete steps with input/output folders and scales."""
    root = Path(exp["results"])
    run = exp.get("run", {})
    hw1 = native_hw(exp)
    jobs = []
    for p in expand_pipelines(exp["pipelines"]):
        scale = int(p["steps"][0]["scale"])
        src = Path(exp["data"]["inputs"][scale])
        key = f"x{scale}"
        steps = []
        for st in p["steps"]:
            if st["model"] not in model_cfg:
                raise KeyError(f"pipeline {p['name']}: unknown model {st['model']}")
            st = resolve_tiling(st, run, model_cfg[st["model"]], (hw1[0] // scale, hw1[1] // scale))
            key += "__" + step_tag(st, run)
            out = root / "steps" / key
            opts = {k: st.get(k, run.get(k)) for k in RUN_KEYS if st.get(k, run.get(k)) is not None}
            steps.append({"model": st["model"], "inputs": src, "out": out, "opts": opts})
            src = out
            scale = scale // int(model_cfg[st["model"]].get("scale", 1))
            if scale < 1:
                raise ValueError(f"pipeline {p['name']}: output would exceed native resolution")
        jobs.append({**p, "steps": steps, "final": src, "eval_scale": scale})
    return jobs


def done(out: Path, n: int) -> bool:
    meta = out / "meta.json"
    return meta.exists() and len(json.loads(meta.read_text())["images"]) >= n


def sh(cmd: list, dry: bool) -> None:
    print("  $ " + " ".join(map(str, cmd)), flush=True)
    if not dry:
        subprocess.run(list(map(str, cmd)), check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment")
    ap.add_argument("--config", default=str(HERE.parent / "configs" / "models.yaml"))
    ap.add_argument("--gpus", default="all")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", help="regex on pipeline names")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args()

    exp = _expand(yaml.safe_load(Path(a.experiment).read_text()))
    exp["data"] = {k: ({int(s): v for s, v in d.items()} if isinstance(d, dict) else d) for k, d in exp["data"].items()}
    model_cfg = load_config(a.config)
    jobs = plan(exp, model_cfg)
    if a.only:
        jobs = [j for j in jobs if re.search(a.only, j["name"])]
    ev = exp.get("eval", {})
    tag = ev.get("tag", "eval")
    py = sys.executable

    unique_steps = {str(s["out"]) for j in jobs for s in j["steps"]}
    print(f"{len(jobs)} pipelines, {len(unique_steps)} unique steps", flush=True)

    for j in jobs:
        print(f"\n[{j['group']}] {j['name']}", flush=True)
        first = Path(j["steps"][0]["inputs"])
        n = len(list_images(first)) if first.exists() else 0
        steps = list(j["steps"])
        shift = None
        if "gridshift" in ev.get("metrics", "") and int(steps[-1]["opts"].get("tile", 0) or 0) > 0:
            # same last step on a grid shifted by half a stride: content-free tiling metrics
            last = steps[-1]
            t, o = int(last["opts"]["tile"]), int(last["opts"].get("overlap", 64))
            shift = {**last, "out": last["out"].with_name(last["out"].name + f"@shift{(t - o) // 2}"),
                     "opts": {**last["opts"], "grid_offset": (t - o) // 2}}
            steps.append(shift)
        for s in steps:
            if n and done(s["out"], n):
                print(f"  cached: {s['out'].name}")
                continue
            cmd = [py, HERE / "run_model.py", "--model", s["model"], "--config", a.config,
                   "--inputs", s["inputs"], "--out", s["out"], "--gpus", a.gpus,
                   "--workers", a.workers, "--skip-existing"]
            for k, v in s["opts"].items():
                cmd += [f"--{k.replace('_', '-')}", v]
            sh(cmd, a.dry_run)

        if a.no_eval:
            continue
        sc = j["eval_scale"]
        result = j["final"] / f"metrics_{tag}.json"
        if result.exists():
            print(f"  evaluated: {result}")
        else:
            cmd = [py, HERE / "evaluate.py", "--pred", j["final"], "--targets", exp["data"]["targets"][sc],
                   "--scale", sc, "--crop", ev.get("crop_native", 0) // sc, "--metrics", ev["metrics"],
                   "--tag", tag, "--label", j["name"], "--gpus", a.gpus, "--workers", a.workers]
            if sc in exp["data"].get("masks", {}):
                cmd += ["--masks", exp["data"]["masks"][sc]]
            if shift is not None:
                cmd += ["--pred-shift", shift["out"]]
            if "noharm" in ev["metrics"].split(","):        # in-focus preservation needs the input at this scale
                cmd += ["--inputs", exp["data"]["inputs"][sc]]
            for k, v in ev.get("options", {}).items():
                cmd += [f"--{k.replace('_', '-')}", v]
            sh(cmd, a.dry_run)
        j["result"] = result

    if a.dry_run or a.no_eval:
        return
    tables = Path(exp["results"]) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict]] = {}
    for j in jobs:
        groups.setdefault(j["group"], []).append(j)
    pdir = Path(exp["results"]) / "pipelines"
    pdir.mkdir(parents=True, exist_ok=True)
    for g, js in groups.items():
        files = []
        for j in js:  # per-pipeline summary: its own label + total runtime over all steps
            if not j["result"].exists():
                continue
            r = json.loads(j["result"].read_text())
            steps = [json.loads((s["out"] / "meta.json").read_text())["images"] for s in j["steps"]]
            r["label"], r["steps"] = j["name"], [str(s["out"].name) for s in j["steps"]]
            r["csv"] = str(pipeline_csv(j["result"].with_suffix(".csv"), steps,
                                        pdir / (re.sub(r"[^A-Za-z0-9_.@+-]", "_", j["name"]) + f"__{tag}.csv")))
            r["metrics"].update(pipeline_cost(steps))
            f = pdir / (re.sub(r"[^A-Za-z0-9_.@+-]", "_", j["name"]) + f"__{tag}.json")
            f.write_text(json.dumps(r, indent=1))
            files.append(str(f))
        md = subprocess.run([py, HERE / "summarize.py", *files, *(["--cols", ev["table_cols"]] if ev.get("table_cols") else [])],
                            check=True, capture_output=True, text=True).stdout
        (tables / f"{g}.md").write_text(f"## {g}\n\n{md}")
        print(f"\n## {g}\n{md}")


if __name__ == "__main__":
    main()
