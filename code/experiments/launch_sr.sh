#!/usr/bin/env bash
# =====================================================================================================================
# launch_sr.sh — manual anchor x upsampler runs (wrapper around code/scripts/launch_sr.py), with setup notes.
#
# WHAT IT DOES
#   One low-res deblurrer ("anchor", run at x4 = the standard DPDD protocol) followed by one x4 upsampler, on all
#   76 native DPDD test scenes. Every run lands in its own folder that evaluate.py / summarize.py understand:
#
#       $UHDD_RESULTS/dpdd_v2/sr/<anchor model>@x4+<upsampler>/     76 native PNGs + meta.json (+ metrics_dpdd4.*)
#
#   Per run:
#     1. anchor    reused from the evaluation-matrix cache ($UHDD_RESULTS/dpdd_v2/steps/x4__<model>@whole) when it
#                  exists, else computed with run_model.py.
#     2. upsampler registry models (bicubic, hat_l, hat_real, swinir_real, osediff) go through run_model.py with their
#                  paper tiling (overlap = tile/8). External tools (s3diff, vosr2, vosr_0.5b) run their own code in
#                  their own Python env on 8-bit copies of the anchor (sr/_lr8/<anchor>), with their own tiling.
#     3. --eval    evaluate.py with the frozen protocol (tag dpdd4: PSNR/SSIM/LPIPS/DISTS native, MUSIQ, CLIPIQA,
#                  PSNR/SSIM at x4, + diagnostics), DP blur maps, 64 px border crop.
#
#   Anchors for the upsampler study (user's visual pick, 2026-10-05): drbnet, bokehlicious.
#   Also available: restormer, lakdnet, ifan.
#
#   Upsamplers:
#     bicubic      bicubic_x4                       whole image
#     hat_l        HAT-L classical x4               512 / 64 LR px
#     hat_real     Real-HAT-GAN x4                  512 / 64
#     swinir_real  SwinIR-M real-world (BSRGAN) x4  400 / 50
#     osediff      OSEDiff one-step (SD2.1)         128 / 16 LR px (HR 512)
#     s3diff       S3Diff one-step (SD-Turbo + degradation-guided LoRA); official latent tiles 96 / 32
#     vosr2        VOSR 2.0 one-step 1.4B DiT (CVPR 2026); DiT tiles 512 / 64 output px
#     vosr_0.5b    VOSR 0.5B one-step; same tiling
#
# USAGE
#   code/experiments/launch_sr.sh help                          this text
#   code/experiments/launch_sr.sh env                           print the environment the launcher will use
#   code/experiments/launch_sr.sh setup s3diff|vosr             one-time: clone, Python env, weights (see SETUP)
#   code/experiments/launch_sr.sh check                         verify paths, weights and that both envs import
#   code/experiments/launch_sr.sh list                          what exists, what is scored
#   code/experiments/launch_sr.sh dry ANCHOR SR                 print the exact commands, run nothing
#   code/experiments/launch_sr.sh run ANCHOR SR [GPU] [args]    run + score (all 76); extra args go to launch_sr.py,
#                                                               e.g. --scenes 1P0A1046,1P0A2030 (smoke test)
#   code/experiments/launch_sr.sh study SR [GPU]                run SR on both study anchors (drbnet, bokehlicious)
#   code/experiments/launch_sr.sh summary                       one table (with CIs) of everything scored so far
#
#   Examples
#     code/experiments/launch_sr.sh dry drbnet vosr2
#     code/experiments/launch_sr.sh run drbnet vosr2 0 --scenes 1P0A1046         # smoke test, 1 scene
#     code/experiments/launch_sr.sh run drbnet vosr2 0
#     code/experiments/launch_sr.sh study s3diff 1                                # drbnet then bokehlicious on GPU 1
#     nohup code/experiments/launch_sr.sh study vosr2 0 > vosr2.log 2>&1 &         # long runs: detach
#
#   Runs are resumable for registry models (--skip-existing). For s3diff / vosr, delete a half-finished output folder
#   (or pass --scenes with the missing ones) before re-running.
#
# ENVIRONMENT (nothing is hard-coded; set these in your shell or an env file, then `source` it)
#   UHDD_DATA, UHDD_RESULTS     as for every other script (required)
#   UHDD_REPOS                  where third-party code is cloned        (default for *_REPO below)
#   UHDD_WEIGHTS                where weights are stored                (default for S3DIFF_SD / S3DIFF_PKL)
#   UHDD_ENVS                   where the per-tool Python envs live     (default for *_PY below)
#
#   S3DIFF_REPO   S3Diff checkout                         default $UHDD_REPOS/S3Diff
#   S3DIFF_PY     python of the S3Diff env                default $UHDD_ENVS/s3diff/bin/python
#   S3DIFF_SD     local snapshot of stabilityai/sd-turbo  default $UHDD_WEIGHTS/sd-turbo
#   S3DIFF_PKL    s3diff.pkl                              default $UHDD_WEIGHTS/s3diff/s3diff.pkl
#   VOSR_REPO     VOSR checkout                           default $UHDD_REPOS/VOSR
#   VOSR_PY       python of the VOSR env                  default $UHDD_ENVS/vosr/bin/python
#   VOSR_CKPTS    folder with VOSR2/, VOSR_0.5B_os/, Qwen-Image-vae-2d/, torch_cache/ ...
#                                                         default $VOSR_REPO/preset/ckpts
#   PY            python of the main uhdd env             default: python on PATH
#
# SETUP (one time; `setup s3diff` / `setup vosr` runs exactly these steps)
#   The two tools pin mutually incompatible torch / diffusers versions, so each gets its own env; the main uhdd env
#   is untouched. Registry upsamplers need nothing beyond the main env (OSEDiff setup: docs/baselines.md, C3).
#
#   S3Diff — github.com/ArcticHare105/S3Diff (Apache-2.0)
#     env      torch 2.1, diffusers 0.25.1, peft 0.10, xformers (the repo's requirements.txt)
#     weights  sd-turbo snapshot (huggingface.co/stabilityai/sd-turbo),
#              s3diff.pkl (huggingface.co/zhangap/S3Diff), de_net.pth ships in the repo (assets/mm-realsr/)
#     run      code/external/s3diff_run.py = the official loop of src/inference_s3diff.py (bilinear x4, latent
#              tiles 96/32, wavelet colour fix) without the official end-of-run pyiqa scoring; per-image time in meta.json
#
#   VOSR — github.com/cswry/VOSR (Apache-2.0, CVPR 2026)
#     env      torch 2.5.1, diffusers 0.35 (the repo's requirements.txt)
#     weights  huggingface.co/CSWRY/VOSR -> $VOSR_CKPTS (VOSR2/, VOSR_0.5B_os/, Qwen-Image VAE, DINOv2 torch cache)
#     run      official inference_vosr_onestep.py -u 4 --tile_size 512 --tile_overlap 64, wavelet colour fix,
#              deterministic VAE posterior; no per-image timing -> meta.json gets wall time / n
#
#   Needs: git, huggingface-cli (pip install -U "huggingface_hub[cli]"), CUDA GPU. Set HF_TOKEN if a repo is gated.
#
# OUTPUT CHECKS (after a run)
#   - 76 PNGs at native size in the run folder; `list` shows the count and whether it was scored.
#   - metrics_dpdd4.csv / .json next to them; `summary` tables psnr, ssim, lpips, dists, musiq, clipiqa,
#     psnr_s4, ssim_s4, dpsnr_b0 (in-focus change vs input; negative = in-focus regions damaged).
#   - Look at the images too: in-focus regions vs the input, invented text / texture, tile seams, colour shifts.
# =====================================================================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCH="$ROOT/code/scripts/launch_sr.py"
PY="${PY:-python}"
STUDY_ANCHORS=(drbnet bokehlicious)

export S3DIFF_REPO="${S3DIFF_REPO:-${UHDD_REPOS:-}/S3Diff}"
export S3DIFF_PY="${S3DIFF_PY:-${UHDD_ENVS:-}/s3diff/bin/python}"
export S3DIFF_SD="${S3DIFF_SD:-${UHDD_WEIGHTS:-}/sd-turbo}"
export S3DIFF_PKL="${S3DIFF_PKL:-${UHDD_WEIGHTS:-}/s3diff/s3diff.pkl}"
export VOSR_REPO="${VOSR_REPO:-${UHDD_REPOS:-}/VOSR}"
export VOSR_PY="${VOSR_PY:-${UHDD_ENVS:-}/vosr/bin/python}"
export VOSR_CKPTS="${VOSR_CKPTS:-$VOSR_REPO/preset/ckpts}"

usage() { sed -n '3,/^# =====/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; }
die() { echo "error: $*" >&2; exit 1; }
need() { for v in "$@"; do [[ -n "${!v:-}" ]] || die "set \$$v (see: $0 help)"; done; }
step() { echo; echo ">> $*"; }

show_env() {
  for v in UHDD_DATA UHDD_RESULTS UHDD_REPOS UHDD_WEIGHTS UHDD_ENVS PY \
           S3DIFF_REPO S3DIFF_PY S3DIFF_SD S3DIFF_PKL VOSR_REPO VOSR_PY VOSR_CKPTS; do
    printf '  %-13s %s\n' "$v" "${!v:-<unset>}"
  done
}

make_env() {  # make_env NAME REPO  -> venv under $UHDD_ENVS/NAME with the repo's requirements
  local name="$1" repo="$2" envdir="$UHDD_ENVS/$1"
  if [[ -x "$envdir/bin/python" ]]; then echo "env exists: $envdir"; return; fi
  "${ENV_PYTHON:-python3.10}" -m venv "$envdir"   # ENV_PYTHON: base interpreter for the tool envs (3.10 works for both)
  "$envdir/bin/pip" install -U pip wheel
  "$envdir/bin/pip" install -r "$repo/requirements.txt"
  "$envdir/bin/pip" install opencv-python-headless pillow
}

setup_s3diff() {
  need UHDD_REPOS UHDD_WEIGHTS UHDD_ENVS
  step "clone S3Diff -> $S3DIFF_REPO"
  [[ -d "$S3DIFF_REPO/.git" ]] || git clone https://github.com/ArcticHare105/S3Diff "$S3DIFF_REPO"
  step "python env -> $UHDD_ENVS/s3diff (torch 2.1, diffusers 0.25.1, peft 0.10, xformers)"
  make_env s3diff "$S3DIFF_REPO"
  step "weights: sd-turbo -> $S3DIFF_SD, s3diff.pkl -> $(dirname "$S3DIFF_PKL")"
  huggingface-cli download stabilityai/sd-turbo --local-dir "$S3DIFF_SD"
  huggingface-cli download zhangap/S3Diff s3diff.pkl --local-dir "$(dirname "$S3DIFF_PKL")"
  [[ -f "$S3DIFF_REPO/assets/mm-realsr/de_net.pth" ]] || echo "WARNING: de_net.pth not found in the repo (assets/mm-realsr/)"
}

setup_vosr() {
  need UHDD_REPOS UHDD_ENVS
  step "clone VOSR -> $VOSR_REPO"
  [[ -d "$VOSR_REPO/.git" ]] || git clone https://github.com/cswry/VOSR "$VOSR_REPO"
  step "python env -> $UHDD_ENVS/vosr (torch 2.5.1, diffusers 0.35)"
  make_env vosr "$VOSR_REPO"
  step "weights: CSWRY/VOSR -> $VOSR_CKPTS"
  huggingface-cli download CSWRY/VOSR --local-dir "$VOSR_CKPTS"
}

check() {
  local ok=1
  chk() { if eval "$2"; then echo "  ok    $1"; else echo "  FAIL  $1"; ok=0; fi; }
  need UHDD_DATA UHDD_RESULTS
  echo "main env"
  chk "native test set       \$UHDD_DATA/dpdd_native_v2/inputs" "[[ -d '$UHDD_DATA/dpdd_native_v2/inputs' ]]"
  chk "DP blur maps          \$UHDD_DATA/dpdd_native/dp_maps"   "[[ -d '$UHDD_DATA/dpdd_native/dp_maps' ]]"
  chk "uhdd imports" "'$PY' -c 'import sys; sys.path.insert(0, \"$ROOT/code\"); import uhdd' 2>/dev/null"
  echo "S3Diff"
  chk "repo        $S3DIFF_REPO" "[[ -f '$S3DIFF_REPO/src/s3diff_tile.py' ]]"
  chk "de_net.pth"               "[[ -f '$S3DIFF_REPO/assets/mm-realsr/de_net.pth' ]]"
  chk "sd-turbo    $S3DIFF_SD"   "[[ -d '$S3DIFF_SD' ]]"
  chk "s3diff.pkl  $S3DIFF_PKL"  "[[ -f '$S3DIFF_PKL' ]]"
  chk "env imports $S3DIFF_PY"   "'$S3DIFF_PY' -c 'import torch, diffusers, peft; assert torch.cuda.is_available()' 2>/dev/null"
  echo "VOSR"
  chk "repo        $VOSR_REPO"   "[[ -f '$VOSR_REPO/inference_vosr_onestep.py' ]]"
  chk "VOSR2 ckpt  $VOSR_CKPTS/VOSR2" "[[ -d '$VOSR_CKPTS/VOSR2' ]]"
  chk "env imports $VOSR_PY"     "'$VOSR_PY' -c 'import torch, diffusers; assert torch.cuda.is_available()' 2>/dev/null"
  (( ok )) && echo "all checks passed" || echo "some checks failed (registry upsamplers only need the main env)"
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  help|-h|--help) usage ;;
  env)     show_env ;;
  setup)   case "${1:-}" in s3diff) setup_s3diff ;; vosr) setup_vosr ;; *) die "setup s3diff|vosr" ;; esac ;;
  check)   check ;;
  list)    "$PY" "$LAUNCH" --list ;;
  summary) "$PY" "$LAUNCH" --summary ;;
  dry)     [[ $# -ge 2 ]] || die "dry ANCHOR SR"
           "$PY" "$LAUNCH" --anchor "$1" --sr "$2" --dry-run "${@:3}" ;;
  run)     [[ $# -ge 2 ]] || die "run ANCHOR SR [GPU] [launch_sr.py args]"
           anchor="$1" sr="$2" gpu="${3:-0}"; shift $(( $# >= 3 ? 3 : 2 ))
           "$PY" "$LAUNCH" --anchor "$anchor" --sr "$sr" --gpu "$gpu" --eval "$@" ;;
  study)   [[ $# -ge 1 ]] || die "study SR [GPU]"
           for a in "${STUDY_ANCHORS[@]}"; do
             "$PY" "$LAUNCH" --anchor "$a" --sr "$1" --gpu "${2:-0}" --eval
           done ;;
  *)       die "unknown command '$cmd' (see: $0 help)" ;;
esac
