#!/usr/bin/env python
"""Manual launcher for single runs on the native DPDD test set, in the results layout of uhdd/layout.py.

Meant to be read and run by hand (shell wrapper with setup notes: code/experiments/launch.sh). Every run lands in its
own folder (PNGs + meta.json + launch.json; --eval adds metrics_dpdd4.csv/json next to the images):

    $UHDD_RESULTS/dpdd/deblur/<src>/<model>@<tiling>/
    $UHDD_RESULTS/dpdd/upsample/<src>/<anchor>@whole/<sr>@<tiling>/        <src> = x4 source of the anchor

USAGE
    python code/scripts/setup_inputs.py                                     # once: link + verify inputs (required)
    python code/scripts/launch.py list                                      # sources, models, runs and their state
    python code/scripts/launch.py deblur --src ours_x4 --model drbnet --eval          # Table 1 row (scored at x4)
    python code/scripts/launch.py deblur --src official_x4 --model drbnet --eval      # same on the original images
    python code/scripts/launch.py deblur --src ours_x1 --model restormer --eval       # native, paper tiling
    python code/scripts/launch.py upsample --src ours_x4 --anchor drbnet --sr vosr2 --gpu 0 --eval   # x4 anchor + x4 SR
    python code/scripts/launch.py upsample --src official_x4 --anchor drbnet --sr vosr2 --eval      # diagnostic
    python code/scripts/launch.py summary                                   # tables (with CIs) of everything scored
    python code/scripts/launch.py compare                                   # fresh results vs legacy (handoff 2/3)
    any run: --scenes 1P0A1046,1P0A2030 (or a file), --dry-run (print the commands only)

WHAT HAPPENS
    deblur   : run_model.py on inputs/<src>, tiling = the model's paper setup at that resolution (or --tile).
    upsample : the anchor = deblur of inputs/<src> (an x4 source) with the anchor's paper setup (whole image at x4; run
               first if missing), then the upsampler. Every upsampler (bicubic, hat_l, hat_real, swinir_real,
               osediff, s3diff, vosr2, vosr_0.5b) is a registry model run through run_model.py in the main env with
               its paper tiling (overlap tile/8, linear blending); S3Diff / VOSR via uhdd/adapters (per-tile
               official path, per-image latent noise shared by overlapping tiles).
    --eval   : evaluate.py, tag dpdd4 (skipped when metrics_dpdd4.json already covers every PNG and is newer;
               --rescore forces it). Native outputs: the frozen headline + diagnostics against inputs/ours_x1 (DP
               maps, masks, 64 px border). x4 outputs: against the targets of the same source, border 16 px;
               official_x4 has no masks -> headline metrics only. An upsampled official anchor is scored against
               our native targets (other rendering): diagnostic only, labelled so.
    --gpu 0,1,2,3 / all: scenes split over the GPUs (run_model.py / evaluate.py: one process per GPU).
    Runs resume: finished images are skipped. Each run refuses to start unless setup_inputs.py verified its sources.

ENVIRONMENT
    UHDD_DATA, UHDD_RESULTS, UHDD_REPOS, UHDD_WEIGHTS     as for every other script (model paths: configs/models.yaml)
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE))

from uhdd import layout  # noqa: E402
from uhdd.tiling import default_tiling  # noqa: E402

PY = sys.executable
TAG = "dpdd4"
DEBLUR = {  # short name -> registry model (anchors for the upsampler study: drbnet, bokehlicious)
    "input": "identity", "drbnet": "drbnet_single", "bokehlicious": "bokehlicious_deblur",
    "restormer": "restormer_dpdd", "lakdnet": "lakdnet_dpdd_l", "ifan": "ifan",
}
REGISTRY_SR = {  # every upsampler runs through run_model.py with its paper tiling (registry, overlap tile/8)
    "bicubic": "bicubic_x4", "hat_l": "hat_l_x4", "hat_real": "hat_x4_real",
    "swinir_real": "swinir_x4_real", "osediff": "osediff_x4",
    "s3diff": "s3diff_x4", "vosr2": "vosr2_x4", "vosr_0.5b": "vosr_0.5b_x4",
    "ours_v0": "ours_v0",                                   # the method (configs/models.yaml, adapters/ours.py)
    "ours_v0_oracle": "ours_v0_oracle",                     # v0 trained with oracle exemplars
}
METRICS_NATIVE = "psnr,ssim,lpips,dists,musiq,clipiqa,msres,hb,blurbins,percbins,noharm,apsnr"
METRICS_LOWRES = "psnr,ssim,lpips,dists,musiq,clipiqa,hb,blurbins,percbins,noharm,apsnr"
METRICS_OFFICIAL = "psnr,ssim,lpips,dists,musiq,clipiqa"
COLS = {"deblur": "psnr,ssim,lpips,dists,musiq,clipiqa,dpsnr_b0,time_s",
        "upsample": "psnr,ssim,lpips,dists,musiq,clipiqa,psnr_s4,ssim_s4,dpsnr_b0,time_s"}


# ---------------------------------------------------------------- helpers
def registry() -> dict:
    import yaml
    return yaml.safe_load((CODE / "configs" / "models.yaml").read_text())["models"]


def model_name(short: str, table: dict) -> str:
    if short in table:
        return table[short]
    if short in table.values():
        return short
    sys.exit(f"unknown model '{short}' (one of {', '.join(table)})")


def manifest() -> dict:
    p = layout.root() / "inputs" / "manifest.json"
    if not p.exists():
        sys.exit("inputs not set up: run code/scripts/setup_inputs.py first")
    return json.loads(p.read_text())


def require(*srcs: str) -> dict:
    """Manifest entry of each source; refuse to run on sources that setup_inputs.py did not verify."""
    man = manifest()
    for s in srcs:
        rec = man["sources"].get(s)
        if rec is None:
            sys.exit(f"source {s} not set up: setup_inputs.py --sources {s},...")
        if not rec.get("ok"):
            sys.exit(f"source {s} failed verification: {rec.get('problems')} (inputs/manifest.json)")
    if not man.get("ok"):
        print(f"warning: manifest has problems outside these sources: {man.get('problems')}")
    return man


def hw_of(man: dict, src: str) -> tuple[int, int]:
    w, h = (int(v) for v in man["sources"][src]["sizes"][0].split("x"))
    return h, w


def step_tag(model: str, src_hw: tuple[int, int], tile: int | None = None, overlap: int | None = None) -> tuple[str, int, int]:
    """(folder tag, tile, overlap): explicit tile, else the model's paper setup at this input size."""
    if tile is None:
        tile, overlap = default_tiling(registry()[model], src_hw)
    elif overlap is None:
        overlap = int(round(tile * 0.125 / 2)) * 2
    return layout.tag(model, tile, overlap), int(tile), int(overlap or 0)


def scene_names(src: str, spec: str | None) -> list[str]:
    allnames = sorted(p.stem for p in layout.input_dir(src).glob("*.png"))
    if not spec or spec == "all":
        return allnames
    want = Path(spec).read_text().split() if Path(spec).exists() else spec.split(",")
    missing = set(want) - set(allnames)
    if missing:
        sys.exit(f"unknown scenes: {sorted(missing)}")
    return want


def todo(out: Path, names: list[str]) -> list[str]:
    return [n for n in names if not (out / f"{n}.png").exists()]


def run(cmd: list, dry: bool, extra_env: dict | None = None) -> float:
    pre = " ".join(f"{k}={v}" for k, v in (extra_env or {}).items())
    print(f"  $ {pre + ' ' if pre else ''}{' '.join(shlex.quote(str(c)) for c in cmd)}", flush=True)
    if dry:
        return 0.0
    t = time.time()
    # UHDD_FAST_EXIT: finished workers exit immediately (a CUDA event-handler thread can segfault at interpreter exit and mp.spawn then kills the siblings)
    subprocess.run([str(c) for c in cmd], check=True, env={"UHDD_FAST_EXIT": "1", **os.environ, **(extra_env or {})})
    return time.time() - t


def subset(src: Path, names: list[str], key: str, dry: bool) -> Path:
    """Folder of symlinks to `names` in `src` (run_model.py and VOSR read whole folders)."""
    dst = layout.scratch_dir("subsets", key)
    if not dry:
        dst.mkdir(parents=True, exist_ok=True)
        for f in dst.glob("*.png"):
            f.unlink()
        for n in names:
            (dst / f"{n}.png").symlink_to((src / f"{n}.png").resolve())
    return dst


def provenance(out: Path, rec: dict, dry: bool) -> None:
    """launch.json: what produced this folder (appended per launch)."""
    if dry:
        return
    try:
        commit = subprocess.run(["git", "-C", str(CODE), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
    except OSError:
        commit = ""
    p = out / "launch.json"
    hist = json.loads(p.read_text()) if p.exists() else []
    hist.append({**rec, "commit": commit, "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    out.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(hist, indent=1))


def scored_current(out: Path) -> bool:
    """metrics_<TAG>.json exists, covers every PNG in `out` and is newer than all of them."""
    m = out / f"metrics_{TAG}.json"
    pngs = list(out.glob("*.png"))
    if not (m.exists() and pngs):
        return False
    return (json.loads(m.read_text()).get("n_images") == len(pngs)
            and m.stat().st_mtime >= max(p.stat().st_mtime for p in pngs))


def evaluate(out: Path, src: str, label: str, gpu: str, dry: bool, native_targets: bool, force: bool = False) -> None:
    """Score `out` (tag dpdd4). native_targets: output at native resolution -> ours_x1 targets. Skipped when the
    metrics are current (all PNGs scored, nothing newer), unless `force`."""
    if not force and not dry and scored_current(out):
        print(f"  scored: {out / f'metrics_{TAG}.json'} (current; --rescore to redo)")
        return
    if native_targets:
        s, scale, metrics = layout.native_of(src), 1, METRICS_NATIVE
    else:
        s, scale = src, layout.source(src)["scale"]
        metrics = METRICS_OFFICIAL if layout.source(src)["masks"] is None else METRICS_LOWRES
    cmd = [PY, HERE / "evaluate.py", "--pred", out, "--targets", layout.input_dir(s, "targets"), "--scale", scale,
           "--crop", layout.CROP_NATIVE // scale, "--metrics", metrics, "--tag", TAG, "--label", label, "--gpus", gpu]
    if layout.source(s)["masks"]:
        cmd += ["--masks", layout.input_dir(s, "masks"), "--inputs", layout.input_dir(s, "inputs"),
                "--dp-maps", layout.dp_maps_dir(layout.split_of(s))]
    run(cmd, dry)


# ---------------------------------------------------------------- stages
def deblur(model: str, src: str, names: list[str], gpu: str, dry: bool, ev: bool,
           tile: int | None = None, overlap: int | None = None, force: bool = False) -> Path:
    man = require(src, layout.native_of(src))
    t, tile, overlap = step_tag(model, hw_of(man, src), tile, overlap)
    out = layout.deblur_dir(src, t)
    label = f"{model} deblur ({src})"
    left = todo(out, names)
    print(f"== deblur {t} on {src}: {len(names)} scenes, {len(left)} to run -> {out}")
    if left:
        inp = layout.input_dir(src)
        if len(left) < len(scene_names(src, None)):
            inp = subset(inp, left, f"deblur_{src}_{t}", dry)
        cmd = [PY, HERE / "run_model.py", "--model", model, "--inputs", inp, "--out", out, "--gpus", gpu,
               "--tile", tile, "--overlap", overlap, "--skip-existing"]
        run(cmd, dry)
        provenance(out, {"stage": "deblur", "src": src, "model": model, "tile": tile, "overlap": overlap,
                         "scenes": left, "cmd": " ".join(map(str, cmd))}, dry)
    if ev:
        evaluate(out, src, label, gpu, dry, native_targets=layout.source(src)["scale"] == 1, force=force)
    return out


def upsample(src: str, anchor: str, sr: str, names: list[str], gpu: str, dry: bool, ev: bool,
             force: bool = False) -> Path:
    if layout.source(src)["scale"] != 4:
        sys.exit(f"upsample needs an x4 source (x4 upsamplers), got {src}")
    official = src.startswith("official")
    man = require(src, layout.native_of(src))
    a_out = deblur(anchor, src, names, gpu, dry, ev=False)
    a_tag = a_out.name
    s_tag = step_tag(REGISTRY_SR[sr], hw_of(man, src))[0]
    out = layout.upsample_dir(src, a_tag, s_tag)
    label = f"{anchor} @x4 ({src}) + {sr}" + (" [diagnostic: official rendering]" if official else "")
    left = todo(out, names)
    print(f"== upsample {a_tag} + {s_tag} ({src}): {len(names)} scenes, {len(left)} to run -> {out}")
    rec = {"stage": "upsample", "src": src, "anchor": anchor, "anchor_dir": str(a_out), "sr": sr, "scenes": left}
    if left:
        _, tile, overlap = step_tag(REGISTRY_SR[sr], hw_of(man, src))
        inp = a_out if len(left) == len(scene_names(src, None)) else subset(a_out, left, f"up_{src}_{a_tag}", dry)
        cmd = [PY, HERE / "run_model.py", "--model", REGISTRY_SR[sr], "--inputs", inp, "--out", out, "--gpus", gpu,
               "--tile", tile, "--overlap", overlap, "--skip-existing"]
        if registry()[REGISTRY_SR[sr]].get("aux"):   # the method: native input + DP maps of the same split
            cmd += ["--aux", f"native={layout.input_dir(layout.native_of(src))}",
                    "--aux", f"dp={layout.dp_maps_dir(layout.split_of(src))}"]
        run(cmd, dry)
        provenance(out, {**rec, "tile": tile, "overlap": overlap, "cmd": " ".join(map(str, cmd))}, dry)
    if ev:
        evaluate(out, src, label, gpu, dry, native_targets=True, force=force)
    return out


# ---------------------------------------------------------------- overview
def runs() -> list[Path]:
    r = layout.root()
    return sorted(p.parent for st in ("deblur", "upsample", "fusion") for p in (r / st).rglob("meta.json"))


def listing() -> None:
    r = layout.root()
    man = r / "inputs" / "manifest.json"
    if man.exists():
        m = json.loads(man.read_text())
        print("inputs   :", ", ".join(f"{s} ({v.get('n')} scenes, {'ok' if v['ok'] else 'NOT OK'})"
                                      for s, v in m["sources"].items()))
    else:
        print("inputs   : not set up (setup_inputs.py)")
    print("deblur   :", ", ".join(f"{k} ({v})" for k, v in DEBLUR.items()))
    print("registry :", ", ".join(f"{k} ({v})" for k, v in REGISTRY_SR.items()))
    print(f"\nruns under {r}:")
    for d in runs():
        n = len(list(d.glob("*.png")))
        ev = "scored" if (d / f"metrics_{TAG}.json").exists() else "not scored"
        print(f"  {str(d.relative_to(r)):75s} {n:3d} images, {ev}")


def scored(stage: str | None = None, src: str | None = None) -> dict[tuple[str, str], list[Path]]:
    out: dict[tuple[str, str], list[Path]] = {}
    for d in runs():
        st, s = layout.stage_of(d)
        if (stage and st != stage) or (src and s != src) or not (d / f"metrics_{TAG}.json").exists():
            continue
        out.setdefault((st, s), []).append(d / f"metrics_{TAG}.json")
    return out


def summary(stage: str | None, src: str | None) -> None:
    groups = scored(stage, src)
    if not groups:
        sys.exit("nothing scored yet (run with --eval)")
    for (st, s), files in groups.items():
        print(f"\n### {st} / {s}")
        args = [f"{p}={json.loads(p.read_text()).get('label', p.parent.name)}" for p in files]
        run([PY, HERE / "summarize.py", *args, "--ci", "--cols", COLS.get(st, COLS["upsample"])], False)


def compare() -> None:
    """Fresh results vs the legacy roots (same harness -> should agree up to nondeterminism)."""
    keys = ("psnr", "ssim", "lpips", "dists")
    print(f"{'run':70s} " + " ".join(f"{k:>16s}" for k in keys))
    for (st, s), files in scored().items():
        for p in files:
            d = p.parent
            if st == "upsample":
                old = layout.legacy_dir(st, s, d.parent.name, d.name)
            else:
                old = layout.legacy_dir(st, s, d.name)
            oj = old / f"metrics_{TAG}.json" if old else None
            new = json.loads(p.read_text())["metrics"]
            if not (oj and oj.exists()):
                print(f"{str(d.relative_to(layout.root())):70s} (no legacy result)")
                continue
            o = json.loads(oj.read_text())["metrics"]
            cells = [f"{new[k]['mean']:.4f} {new[k]['mean'] - o[k]['mean']:+.4f}" if k in new and k in o else "-"
                     for k in keys]
            print(f"{str(d.relative_to(layout.root())):70s} " + " ".join(f"{c:>16s}" for c in cells))
    print("\ncells: fresh mean, then fresh - legacy")


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--scenes", default="all", help="'all' (default), comma list, or a file with one name per line")
    common.add_argument("--gpu", default="0", help="'0', '0,1,3' or 'all': scenes are split over these GPUs "
                                                     "(one process per GPU)")
    common.add_argument("--eval", action="store_true", help=f"also score (tag {TAG}); skipped if already current")
    common.add_argument("--rescore", action="store_true", help="score again even if the metrics are current")
    common.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    d = sub.add_parser("deblur", parents=[common], help="one deblurrer on one input source")
    d.add_argument("--src", required=True, choices=list(layout.SOURCES))
    d.add_argument("--model", required=True, help=f"{', '.join(DEBLUR)} or a registry name")
    d.add_argument("--tile", type=int, help="override the paper tiling (input px; 0 = whole image)")
    d.add_argument("--overlap", type=int, help="default: tile / 8")
    u = sub.add_parser("upsample", parents=[common], help="x4 anchor + x4 upsampler, output at native resolution")
    u.add_argument("--src", required=True, choices=[s for s, v in layout.SOURCES.items() if v["scale"] == 4],
                   help="x4 input of the anchor: ours_x4 (our raw-built rendering) or official_x4 (the original DPDD "
                        "images; diagnostic, scored against our native targets)")
    u.add_argument("--anchor", required=True, help=f"{', '.join(DEBLUR)} or a registry name")
    u.add_argument("--sr", required=True, choices=list(REGISTRY_SR))
    sub.add_parser("list")
    s = sub.add_parser("summary")
    s.add_argument("--stage", choices=["deblur", "upsample", "fusion"])
    s.add_argument("--src", choices=list(layout.SOURCES))
    sub.add_parser("compare")
    a = ap.parse_args()

    if a.cmd == "list":
        return listing()
    if a.cmd == "summary":
        return summary(a.stage, a.src)
    if a.cmd == "compare":
        return compare()
    if a.cmd == "deblur":
        deblur(model_name(a.model, DEBLUR), a.src, scene_names(a.src, a.scenes), a.gpu, a.dry_run, a.eval,
               a.tile, a.overlap, a.rescore)
    else:
        upsample(a.src, model_name(a.anchor, DEBLUR), a.sr, scene_names(a.src, a.scenes), a.gpu, a.dry_run, a.eval,
                 a.rescore)


if __name__ == "__main__":
    main()
