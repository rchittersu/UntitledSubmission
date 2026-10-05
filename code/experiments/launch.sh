#!/usr/bin/env bash
# =====================================================================================================================
# launch.sh — manual runs on the native DPDD test set (wrapper around code/scripts/launch.py), with setup notes.
#
# RESULTS LAYOUT (code/uhdd/layout.py; metrics next to the images)
#   $UHDD_RESULTS/dpdd/
#     inputs/<src>/{inputs,targets,masks}   symlinks into $UHDD_DATA + manifest.json (verified by `setup inputs`)
#     inputs/dp_maps                        DP blur maps (native)
#     deblur/<src>/<model>@<tiling>/                    PNGs + meta.json + launch.json (+ metrics_dpdd4.*)
#     upsample/<src>/<anchor>@whole/<sr>@<tiling>/      same; <src> = where the anchor's x4 input came from
#     fusion/<src>/<anchor>@whole/<method>/             (training-free fusions, DATSR: not in the launcher yet)
#     scratch/                                          8-bit copies, scene subsets (safe to delete)
#   Sources <src>: ours_x1 (native), ours_x4 (our raw-built rendering at 1680x1120), official_x4 (the original DPDD
#   test images), ours_x2. Legacy roots ($UHDD_RESULTS/dpdd_v2, dpdd_official_x4, dpdd_p1) are never written;
#   `setup inputs --protect-legacy` makes them read-only.
#
# WHAT A RUN DOES
#   deblur    one deblurrer on one source, tiling = the model's paper setup at that resolution (whole image at x4,
#             1120-px tiles at native), overlap = tile/8. Scored at the source's resolution (x4: border 16 px;
#             official_x4: headline metrics only, it has no masks).
#   upsample  anchor (deblur of SRC, an x4 source; run first if missing) + x4 upsampler -> native, scored at native
#             against our targets (frozen headline + diagnostics, DP maps, 64 px border).
#             SRC = official_x4: anchor from the original DPDD images; scored against our native targets (other
#             rendering) -> labelled diagnostic.
#   Registry upsamplers run through run_model.py; s3diff / vosr run their own code in their own Python env on 8-bit
#   copies of the anchor. Runs resume (finished images are skipped) and refuse sources that failed verification.
#
#   Deblurrers: input (= identity), drbnet, bokehlicious (study anchors, user's visual pick), restormer, lakdnet, ifan.
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
#   code/experiments/launch.sh help                           this text
#   code/experiments/launch.sh env                            print the environment the launcher will use
#   code/experiments/launch.sh setup inputs [--protect-legacy]  link + verify inputs (names, sizes, same scene in
#                                                             official vs ours); REQUIRED before any run
#   code/experiments/launch.sh setup s3diff|vosr              one-time: clone, Python env, weights (see SETUP)
#   code/experiments/launch.sh check                          verify inputs, S3Diff / VOSR paths, weights, envs
#   code/experiments/launch.sh list                           sources, models, every run and whether it is scored
#   code/experiments/launch.sh deblur SRC MODEL [GPU] [args]          one deblurrer on one source, scored
#   code/experiments/launch.sh upsample SRC MODEL SR [GPU] [args]     deblur MODEL on SRC (x4) + upsampler SR, scored
#   code/experiments/launch.sh study SRC SR [GPU] [args]              `upsample` for both study anchors
#   code/experiments/launch.sh fresh [GPU]                    the standard set from scratch (see `fresh` below)
#   code/experiments/launch.sh summary [--stage S] [--src S]  tables (with CIs) of everything scored
#   code/experiments/launch.sh compare                        fresh results vs legacy (same harness -> ~equal)
#   SRC: ours_x1 (native), ours_x2, ours_x4, official_x4 (upsample: ours_x4 or official_x4)
#   GPU: 0 | 0,1,2,3 | all  — scenes are split over the listed GPUs, one process per GPU (registry models via
#        run_model.py / evaluate.py; S3Diff / VOSR: one process per GPU on a round-robin shard, output [gpu N]).
#   [args] go to launch.py: --scenes 1P0A1046,1P0A2030 (or a file), --dry-run, --tile N (deblur only)
#
#   Examples
#     code/experiments/launch.sh setup inputs --protect-legacy
#     code/experiments/launch.sh deblur ours_x4 drbnet 0                          # Table 1 row, ours
#     code/experiments/launch.sh deblur official_x4 drbnet 0                      # Table 1 row, official images
#     code/experiments/launch.sh deblur ours_x1 restormer 0                       # native, paper tiling
#     code/experiments/launch.sh upsample ours_x4 drbnet vosr2 0 --dry-run        # print the exact commands
#     code/experiments/launch.sh upsample ours_x4 drbnet vosr2 0 --scenes 1P0A1046   # smoke test, 1 scene
#     code/experiments/launch.sh upsample ours_x4 drbnet vosr2 0                  # all 76
#     code/experiments/launch.sh upsample official_x4 drbnet vosr2 0              # diagnostic
#     code/experiments/launch.sh study ours_x4 s3diff 1                           # drbnet then bokehlicious, GPU 1
#     code/experiments/launch.sh upsample ours_x4 drbnet vosr2 0,1,2,3            # 76 scenes split over 4 GPUs
#     nohup code/experiments/launch.sh study ours_x4 vosr2 all > vosr2.log 2>&1 &  # long runs: detach
#
#   fresh = in this order, all 76, scored (registry models only; s3diff / vosr via `study`):
#     deblur    {ours_x4, official_x4, ours_x1} x {input, drbnet, bokehlicious, restormer, lakdnet, ifan}
#     upsample  ours_x4 x {drbnet, bokehlicious} x {bicubic, hat_l, hat_real, swinir_real, osediff}
#   then `compare` (should match the handoff-3 numbers) and `summary`.
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
LAUNCH="$ROOT/code/scripts/launch.py"
SETUP_INPUTS="$ROOT/code/scripts/setup_inputs.py"
PY="${PY:-python}"
STUDY_ANCHORS=(drbnet bokehlicious)
FRESH_DEBLUR=(input drbnet bokehlicious restormer lakdnet ifan)
FRESH_SRC=(ours_x4 official_x4 ours_x1)
FRESH_SR=(bicubic hat_l hat_real swinir_real osediff)

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
  echo "inputs"
  chk "uhdd imports" "'$PY' -c 'import sys; sys.path.insert(0, \"$ROOT/code\"); import uhdd.layout' 2>/dev/null"
  chk "manifest ok           \$UHDD_RESULTS/dpdd/inputs/manifest.json" \
      "'$PY' -c 'import json,sys; sys.exit(0 if json.load(open(\"$UHDD_RESULTS/dpdd/inputs/manifest.json\"))[\"ok\"] else 1)' 2>/dev/null"
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

fresh() {
  local gpu="${1:-0}"
  for src in "${FRESH_SRC[@]}"; do
    for m in "${FRESH_DEBLUR[@]}"; do "$PY" "$LAUNCH" deblur --src "$src" --model "$m" --gpu "$gpu" --eval; done
  done
  for a in "${STUDY_ANCHORS[@]}"; do
    for sr in "${FRESH_SR[@]}"; do "$PY" "$LAUNCH" upsample --src ours_x4 --anchor "$a" --sr "$sr" --gpu "$gpu" --eval; done
  done
  "$PY" "$LAUNCH" compare
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  help|-h|--help) usage ;;
  env)     show_env ;;
  setup)   what="${1:-}"; shift || true
           case "$what" in
             inputs) "$PY" "$SETUP_INPUTS" "$@" ;;
             s3diff) setup_s3diff ;;
             vosr)   setup_vosr ;;
             *)      die "setup inputs|s3diff|vosr" ;;
           esac ;;
  check)   check ;;
  list)    "$PY" "$LAUNCH" list ;;
  summary) "$PY" "$LAUNCH" summary "$@" ;;
  compare) "$PY" "$LAUNCH" compare ;;
  deblur)  [[ $# -ge 2 ]] || die "deblur SRC MODEL [GPU] [launch.py args]"
           src="$1" m="$2" gpu="${3:-0}"; shift $(( $# >= 3 ? 3 : 2 ))
           "$PY" "$LAUNCH" deblur --src "$src" --model "$m" --gpu "$gpu" --eval "$@" ;;
  upsample) [[ $# -ge 3 ]] || die "upsample SRC MODEL SR [GPU] [launch.py args]"
           src="$1" m="$2" sr="$3" gpu="${4:-0}"; shift $(( $# >= 4 ? 4 : 3 ))
           "$PY" "$LAUNCH" upsample --src "$src" --anchor "$m" --sr "$sr" --gpu "$gpu" --eval "$@" ;;
  study)   [[ $# -ge 2 ]] || die "study SRC SR [GPU] [launch.py args]"
           src="$1" sr="$2" gpu="${3:-0}"; shift $(( $# >= 3 ? 3 : 2 ))
           for a in "${STUDY_ANCHORS[@]}"; do
             "$PY" "$LAUNCH" upsample --src "$src" --anchor "$a" --sr "$sr" --gpu "$gpu" --eval "$@"
           done ;;
  fresh)   fresh "$@" ;;
  *)       die "unknown command '$cmd' (see: $0 help)" ;;
esac
