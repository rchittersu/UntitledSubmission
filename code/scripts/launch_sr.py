#!/usr/bin/env python
"""Manual launcher: one low-res deblurrer ("anchor", run at x4) + one x4 upsampler, on the native test set.

Meant to be read and run by hand. Every run lands in its own folder that evaluate.py / summarize.py understand, so
results can be aggregated later by anyone:

    $UHDD_RESULTS/dpdd_v2/sr/<anchor>@x4+<upsampler>/      <- 76 native-resolution PNGs + meta.json (+ metrics_dpdd4.*)

USAGE
    python code/scripts/launch_sr.py --list                                  # what exists, what is done
    python code/scripts/launch_sr.py --anchor drbnet --sr vosr2 --dry-run    # print the exact commands only
    python code/scripts/launch_sr.py --anchor drbnet --sr vosr2 --gpu 0      # run (all 76 scenes)
    python code/scripts/launch_sr.py --anchor bokehlicious --sr s3diff --gpu 1 --scenes 1P0A1046,1P0A2030
    python code/scripts/launch_sr.py --anchor drbnet --sr vosr2 --eval       # run + score with the frozen protocol
    python code/scripts/launch_sr.py --summary                               # one table of everything scored so far

WHAT HAPPENS (per anchor x upsampler)
    1. anchor   : the x4 deblur of the test inputs. Reused from the evaluation matrix cache
                  ($UHDD_RESULTS/dpdd_v2/steps/x4__<model>@whole) if present, else computed with run_model.py.
    2. upsampler: registry models (bicubic, hat_l, hat_real, swinir_real, osediff) run through run_model.py with their
                  paper tiling; external tools (s3diff, vosr2, vosr_0.5b) run their own code in their own Python
                  environment on 8-bit copies of the anchor (both tools read 8-bit sRGB), with their own tiling.
    3. --eval   : evaluate.py with the frozen headline + diagnostics (tag dpdd4), DP blur maps, 64 px border, masks.

ENVIRONMENT (paths; nothing is hard-coded)
    UHDD_DATA, UHDD_RESULTS                  as for every other script
    S3DIFF_REPO, S3DIFF_PY                   S3Diff checkout and the python of its environment
    S3DIFF_SD, S3DIFF_PKL                    sd-turbo snapshot dir, s3diff.pkl
    VOSR_REPO, VOSR_PY, VOSR_CKPTS           VOSR checkout, python of its environment, folder with VOSR2/, VOSR_0.5B_os/ ...
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
PY = sys.executable

ANCHORS = {  # short name -> registry model (visual pick 2026-10-05: drbnet, bokehlicious)
    "drbnet": "drbnet_single", "bokehlicious": "bokehlicious_deblur",
    "restormer": "restormer_dpdd", "lakdnet": "lakdnet_dpdd_l", "ifan": "ifan",
}
REGISTRY_SR = {  # short name -> registry model (tiling: registry paper_tile, overlap tile/8)
    "bicubic": "bicubic_x4", "hat_l": "hat_l_x4", "hat_real": "hat_x4_real",
    "swinir_real": "swinir_x4_real", "osediff": "osediff_x4",
}
EXTERNAL_SR = {  # short name -> description; commands in external_cmd()
    "s3diff": "S3Diff (one-step, SD-Turbo + degradation-guided LoRA); official latent tiling 96/32",
    "vosr2": "VOSR 2.0 one-step 1.4B DiT (CVPR 2026); DiT tiles 512 output px, overlap 64 (= tile/8)",
    "vosr_0.5b": "VOSR 0.5B one-step; same tiling",
}
METRICS = "psnr,ssim,lpips,dists,musiq,clipiqa,msres,hb,blurbins,percbins,noharm,apsnr"


def env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        sys.exit(f"set ${name} (see the header of {Path(__file__).name})")
    return v


def roots():
    data = Path(env("UHDD_DATA")) / "dpdd_native_v2"
    res = Path(env("UHDD_RESULTS")) / "dpdd_v2"
    return data, res


def run(cmd: list, dry: bool, extra_env: dict | None = None) -> float:
    pre = " ".join(f"{k}={v}" for k, v in (extra_env or {}).items())
    print(f"  $ {pre + ' ' if pre else ''}{' '.join(shlex.quote(str(c)) for c in cmd)}", flush=True)
    if dry:
        return 0.0
    t = time.time()
    subprocess.run([str(c) for c in cmd], check=True, env={**os.environ, **(extra_env or {})})
    return time.time() - t


def anchor_dir(anchor: str, scenes: str | None, gpu: str, dry: bool) -> Path:
    data, res = roots()
    model = ANCHORS[anchor]
    cached = res / "steps" / f"x4__{model}@whole"
    if (cached / "meta.json").exists():
        print(f"[anchor] reusing {cached}")
        return cached
    out = res / "sr" / "_anchors" / f"{model}@x4"
    print(f"[anchor] computing {model} at x4 -> {out}")
    run([PY, HERE / "run_model.py", "--model", model, "--inputs", data / "x4" / "inputs", "--out", out,
         "--gpus", gpu, "--skip-existing"], dry)
    return out


def lr8_dir(anchor_path: Path, anchor: str, names: list[str], dry: bool) -> Path:
    """8-bit copies of the anchor (S3Diff and VOSR read 8-bit sRGB through PIL)."""
    _, res = roots()
    out = res / "sr" / "_lr8" / anchor
    print(f"[lr8] 8-bit copies -> {out}")
    if dry:
        return out
    import cv2
    out.mkdir(parents=True, exist_ok=True)
    for n in names:
        dst = out / f"{n}.png"
        if not dst.exists():
            img = cv2.imread(str(anchor_path / f"{n}.png"), cv2.IMREAD_UNCHANGED)
            if img.dtype != "uint8":
                img = (img.astype("float32") / 257.0 + 0.5).clip(0, 255).astype("uint8")
            cv2.imwrite(str(dst), img)
    return out


def external_cmd(sr: str, lr8: Path, out: Path, only: str | None) -> tuple[list, dict]:
    if sr == "s3diff":
        cmd = [env("S3DIFF_PY"), CODE / "external" / "s3diff_run.py", "--repo", env("S3DIFF_REPO"),
               "--inputs", lr8, "--out", out, "--sd-path", env("S3DIFF_SD"), "--pretrained", env("S3DIFF_PKL")]
        return cmd + (["--only", only] if only else []), {}
    ckpt = {"vosr2": "VOSR2", "vosr_0.5b": "VOSR_0.5B_os"}[sr]
    # VOSR reads a whole folder; with --scenes the folder is a symlinked subset
    cmd = [env("VOSR_PY"), Path(env("VOSR_REPO")) / "inference_vosr_onestep.py", "-c", Path(env("VOSR_CKPTS")) / ckpt,
           "-i", lr8, "-o", out, "-u", "4", "--tile_size", "512", "--tile_overlap", "64"]
    return cmd, {"PYTHONPATH": env("VOSR_REPO")}


def scene_names(spec: str | None) -> list[str]:
    data, _ = roots()
    allnames = sorted(p.stem for p in (data / "inputs").glob("*.png"))
    if not spec or spec == "all":
        return allnames
    want = Path(spec).read_text().split() if Path(spec).exists() else spec.split(",")
    missing = set(want) - set(allnames)
    if missing:
        sys.exit(f"unknown scenes: {sorted(missing)}")
    return want


def launch(a) -> None:
    data, res = roots()
    names = scene_names(a.scenes)
    only = None if len(names) == len(scene_names(None)) else ",".join(names)
    out = res / "sr" / f"{ANCHORS[a.anchor]}@x4+{a.sr}"
    label = f"{ANCHORS[a.anchor]} @x4 + {a.sr}"
    print(f"== {label}: {len(names)} scenes -> {out}")
    src = anchor_dir(a.anchor, a.scenes, a.gpu, a.dry_run)

    if a.sr in REGISTRY_SR:
        cmd = [PY, HERE / "run_model.py", "--model", REGISTRY_SR[a.sr], "--inputs", src, "--out", out,
               "--gpus", a.gpu, "--skip-existing"]
        if only:   # run_model reads whole folders: use a symlinked subset
            src = subset(src, names, res / "sr" / "_subsets" / f"{a.anchor}", a.dry_run)
            cmd[cmd.index("--inputs") + 1] = src
        run(cmd, a.dry_run)
    else:
        lr8 = lr8_dir(src, a.anchor, names, a.dry_run)
        if only and a.sr.startswith("vosr"):
            lr8 = subset(lr8, names, res / "sr" / "_subsets" / f"{a.anchor}_lr8", a.dry_run)
        cmd, extra = external_cmd(a.sr, lr8, out, only)
        gpu = a.gpu if a.gpu not in ("all", "cpu") else "0"
        dt = run(cmd, a.dry_run, {"CUDA_VISIBLE_DEVICES": gpu, **extra})
        if not a.dry_run and not (out / "meta.json").exists():   # VOSR writes no meta: mean time per image
            per = dt / max(len(names), 1)
            (out / "meta.json").write_text(json.dumps({"images": {n: {"time_s": round(per, 3)} for n in names},
                                                       "summary": {"method": a.sr, "note": "time = wall / n"}}, indent=1))

    if a.eval:
        run([PY, HERE / "evaluate.py", "--pred", out, "--targets", data / "x1" / "targets", "--masks", data / "x1" / "masks",
             "--inputs", data / "inputs", "--dp-maps", Path(env("UHDD_DATA")) / "dpdd_native" / "dp_maps",
             "--crop", "64", "--metrics", METRICS, "--tag", "dpdd4", "--label", label, "--gpus", a.gpu], a.dry_run)
    print(f"== done: {out}")


def subset(src: Path, names: list[str], dst: Path, dry: bool) -> Path:
    if not dry:
        dst.mkdir(parents=True, exist_ok=True)
        for f in dst.glob("*.png"):
            f.unlink()
        for n in names:
            (dst / f"{n}.png").symlink_to((src / f"{n}.png").resolve())
    return dst


def listing() -> None:
    _, res = roots()
    print("anchors  :", ", ".join(f"{k} ({v})" for k, v in ANCHORS.items()))
    print("registry :", ", ".join(f"{k} ({v})" for k, v in REGISTRY_SR.items()))
    for k, v in EXTERNAL_SR.items():
        print(f"external : {k:10s} {v}")
    print(f"\nruns under {res / 'sr'}:")
    for d in sorted((res / "sr").glob("*@x4+*")) if (res / "sr").exists() else []:
        n = len(list(d.glob("*.png")))
        ev = "scored" if (d / "metrics_dpdd4.json").exists() else "not scored"
        print(f"  {d.name:50s} {n:3d} images, {ev}")


def summary() -> None:
    _, res = roots()
    files = [f"{p}={json.loads(p.read_text()).get('label', p.parent.name)}" for p in sorted((res / "sr").glob("*/metrics_dpdd4.json"))]
    if not files:
        sys.exit("nothing scored yet (run with --eval)")
    run([PY, HERE / "summarize.py", *files, "--ci", "--cols", "psnr,ssim,lpips,dists,musiq,clipiqa,psnr_s4,ssim_s4,dpsnr_b0"], False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchor", choices=list(ANCHORS), help="low-res deblurrer run at x4")
    ap.add_argument("--sr", choices=list(REGISTRY_SR) + list(EXTERNAL_SR), help="x4 upsampler")
    ap.add_argument("--scenes", default="all", help="'all' (default), comma list, or a file with one name per line")
    ap.add_argument("--gpu", default="0", help="GPU index for this run ('all' = every GPU for registry models)")
    ap.add_argument("--eval", action="store_true", help="also score with the frozen protocol (tag dpdd4)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.list:
        return listing()
    if a.summary:
        return summary()
    if not (a.anchor and a.sr):
        ap.error("--anchor and --sr are required (or --list / --summary)")
    launch(a)


if __name__ == "__main__":
    main()
