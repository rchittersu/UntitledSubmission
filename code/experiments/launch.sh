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
#     scratch/                                          scene subsets (safe to delete)
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
#   Every model (deblurrers and upsamplers, incl. OSEDiff, S3Diff, VOSR) is a registry model (configs/models.yaml)
#   run by run_model.py in the ONE main env, with its paper tiling, overlap = tile/8, linear blending. Runs resume
#   (finished images are skipped) and refuse sources that failed verification.
#
#   Deblurrers: input (= identity), drbnet, bokehlicious (study anchors, user's visual pick), restormer, lakdnet, ifan.
#   Upsamplers:
#     bicubic      bicubic_x4                       whole image
#     hat_l        HAT-L classical x4               512 / 64 LR px
#     hat_real     Real-HAT-GAN x4                  512 / 64
#     swinir_real  SwinIR-M real-world (BSRGAN) x4  400 / 50
#     osediff      OSEDiff one-step (SD2.1)         128 / 16 LR px (HR 512)
#     s3diff       S3Diff one-step (SD-Turbo + degradation LoRA)  192 / 24 LR px (= official latent tile 96)
#     vosr2        VOSR 2.0 one-step 1.4B DiT (CVPR 2026)  128 / 16 LR px (= official DiT tile 512 output px)
#     vosr_0.5b    VOSR 0.5B one-step                       128 / 16
#
# USAGE
#   code/experiments/launch.sh help                           this text
#   code/experiments/launch.sh env                            print the environment the launcher will use
#   code/experiments/launch.sh setup inputs [--protect-legacy]  link + verify inputs (names, sizes, same scene in
#                                                             official vs ours); REQUIRED before any run
#   code/experiments/launch.sh setup s3diff|vosr|refir|irag   one-time: clone, weights (see SETUP)
#   code/experiments/launch.sh tiles select|prep|run|report [args]   reference-SR tile study on selected val tiles
#                                                             (scripts/tile_study.py; see TILE STUDY below)
#   code/experiments/launch.sh check                          verify inputs, S3Diff / VOSR paths, weights, envs
#   code/experiments/launch.sh list                           sources, models, every run and whether it is scored
#   code/experiments/launch.sh deblur SRC MODEL [GPU] [args]          one deblurrer on one source, scored
#   code/experiments/launch.sh upsample SRC MODEL SR [GPU] [args]     deblur MODEL on SRC (x4) + upsampler SR, scored
#   code/experiments/launch.sh study SRC SR [GPU] [args]              `upsample` for both study anchors
#   code/experiments/launch.sh fresh [GPU]                    the standard set from scratch (see `fresh` below)
#   code/experiments/launch.sh summary [--stage S] [--src S]  tables (with CIs) of everything scored
#   code/experiments/launch.sh compare                        fresh results vs legacy (same harness -> ~equal)
#   SRC: ours_x1 (native), ours_x2, ours_x4, official_x4 (upsample: ours_x4 or official_x4)
#   GPU: 0 | 0,1,2,3 | all  — scenes are split over the listed GPUs, one process per GPU (run_model.py /
#        evaluate.py). Each process loads its own copy of the model (VOSR 2.0: 1.4B params in fp32 per GPU).
#   [args] go to launch.py: --scenes 1P0A1046,1P0A2030 (or a file), --dry-run, --tile N (deblur only), --rescore
#   Every run is scored (metrics_dpdd4.csv/json next to the images); scoring is skipped when the metrics are current
#   (cover every PNG, newer than all of them) unless --rescore.
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
#   fresh = in this order, all 76, scored (S3Diff / VOSR via `study`, they are the slow ones):
#     deblur    {ours_x4, official_x4, ours_x1} x {input, drbnet, bokehlicious, restormer, lakdnet, ifan}
#     upsample  ours_x4 x {drbnet, bokehlicious} x {bicubic, hat_l, hat_real, swinir_real, osediff}
#   then `compare` (should match the handoff-3 numbers) and `summary`.
#
# ENVIRONMENT (nothing is hard-coded; set these in your shell or an env file, then `source` it)
#   UHDD_DATA, UHDD_RESULTS     data and results roots (required)
#   UHDD_REPOS, UHDD_WEIGHTS    third-party code / weights (model paths: code/configs/models.yaml)
#   PY                          python of the main env (default: python on PATH)
#
# SETUP (one time; `setup s3diff` / `setup vosr` runs exactly these steps). One Python env for everything:
#   pip install -r code/requirements.txt (diffusers >= 0.35, transformers, peft, fairscale, ...). Both adapters were
#   checked outside on CPU with tiny random models against diffusers 0.40 / peft 0.21 / transformers 5.18.
#
#   S3Diff — github.com/ArcticHare105/S3Diff (Apache-2.0)            registry s3diff_x4, uhdd/adapters/s3diff.py
#     code     $UHDD_REPOS/S3Diff (de_net.pth ships in assets/mm-realsr/)
#     weights  $UHDD_WEIGHTS/sd-turbo (stabilityai/sd-turbo), $UHDD_WEIGHTS/s3diff/s3diff.pkl (zhangap/S3Diff)
#     per tile: bilinear x4, VAE posterior sample (shared per-image noise), UNet t=999 with CFG 1.07 (official
#     prompts), one DDPM step, decode, wavelet colour fix; degradation score from DEResNet on the whole LR image.
#
#   VOSR — github.com/cswry/VOSR (Apache-2.0, CVPR 2026)              registry vosr2_x4 / vosr_0.5b_x4, adapters/vosr.py
#     code     $UHDD_REPOS/VOSR
#     weights  $UHDD_WEIGHTS/vosr = CSWRY/VOSR (VOSR2/, VOSR_0.5B_os/, Qwen-Image-vae-2d/, stable-diffusion-2-1-base/,
#              sd21_lwdecoder.pth, torch_cache/ = DINOv2 hub code + weights, loaded offline)
#     per tile: bicubic x4, VAE posterior mode, DINOv2 features of the tile, one flow step from noise (shared per-image
#     noise, seed 42), decode, wavelet (Gaussian sigma 5) colour fix. Config from the checkpoint's args.json.
#
#   Fidelity reference (optional): code/external/s3diff_run.py reproduces the official S3Diff script (its own env).
#   Needs: git, huggingface-cli (pip install -U "huggingface_hub[cli]"), CUDA GPU. Set HF_TOKEN if a repo is gated.
#
# TILE STUDY (reference-based SR on selected tiles, fast feedback; code/scripts/tile_study.py, configs/refsr.yaml)
#   launch.sh tiles select                                   8 val scenes x 2 tiles per DP bin (b0..b3), fixed
#   launch.sh tiles prep --anchor drbnet                     crops + references: self (native at the tile), retrieved
#                                                            (mosaic of in-focus exemplars, DINOv2 memory)
#   launch.sh tiles run --anchor drbnet --model refir_seesr --ref retrieved [--set steps=20] [--gpu 1] [--shard 0/2]
#   launch.sh tiles run --anchor drbnet --model irag --ref self
#   launch.sh tiles report --anchor drbnet                   per-bin gains over bicubic, raw and anchor-locked,
#                                                            LPIPS / DISTS, contact sheets (tilestudy/val/sheets/)
#   models: bicubic, highband (non-learned), seesr (no reference), refir_seesr, irag, irag_inter (TTSR branch only)
#   refs:   none | self | retrieved
#
#   ReFIR — github.com/csguoh/ReFIR (on SeeSR)          code $UHDD_REPOS/ReFIR, adapter uhdd/adapters/refir.py
#     weights  $UHDD_WEIGHTS/sd2_base (stabilityai/stable-diffusion-2-base, NOT 2.1),
#              $UHDD_WEIGHTS/seesr/{seesr/{unet,controlnet},DAPE.pth} (SeeSR release, Google Drive / OneDrive link in
#              the ReFIR README: download by hand if gdown cannot reach it), RAM + BERT tokenizer as for OSEDiff
#   iRAG — github.com/ByeonghunLee12/iRAG                code $UHDD_REPOS/iRAG, adapter uhdd/adapters/irag.py
#     weights  $UHDD_WEIGHTS/irag/iRAG.ckpt (Google Drive folder in the iRAG README),
#              $UHDD_WEIGHTS/stablesr/vqgan_cfw_00011.ckpt (Iceclear/StableSR), OpenCLIP ViT-H-14 laion2b_s32b_b79k
#              (fetched into the HF cache by `setup irag`; offline afterwards)
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
FRESH_SR=(bicubic hat_l hat_real swinir_real)   # osediff: needs the old-library overlay; left out for now (see report)

usage() { sed -n '3,/^# =====/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; }
die() { echo "error: $*" >&2; exit 1; }
need() { for v in "$@"; do [[ -n "${!v:-}" ]] || die "set \$$v (see: $0 help)"; done; }
step() { echo; echo ">> $*"; }

show_env() {
  for v in UHDD_DATA UHDD_RESULTS UHDD_REPOS UHDD_WEIGHTS PY; do
    printf '  %-13s %s\n' "$v" "${!v:-<unset>}"
  done
}

setup_s3diff() {
  need UHDD_REPOS UHDD_WEIGHTS
  step "clone S3Diff -> $UHDD_REPOS/S3Diff"
  [[ -d "$UHDD_REPOS/S3Diff/.git" ]] || git clone https://github.com/ArcticHare105/S3Diff "$UHDD_REPOS/S3Diff"
  step "weights: sd-turbo -> $UHDD_WEIGHTS/sd-turbo, s3diff.pkl -> $UHDD_WEIGHTS/s3diff"
  huggingface-cli download stabilityai/sd-turbo --local-dir "$UHDD_WEIGHTS/sd-turbo"
  huggingface-cli download zhangap/S3Diff s3diff.pkl --local-dir "$UHDD_WEIGHTS/s3diff"
  [[ -f "$UHDD_REPOS/S3Diff/assets/mm-realsr/de_net.pth" ]] || echo "WARNING: de_net.pth not found in the repo (assets/mm-realsr/)"
}

setup_vosr() {
  need UHDD_REPOS UHDD_WEIGHTS
  step "clone VOSR -> $UHDD_REPOS/VOSR"
  [[ -d "$UHDD_REPOS/VOSR/.git" ]] || git clone https://github.com/cswry/VOSR "$UHDD_REPOS/VOSR"
  step "weights: CSWRY/VOSR -> $UHDD_WEIGHTS/vosr (VOSR2, VOSR_0.5B_os and what they need)"
  huggingface-cli download CSWRY/VOSR --local-dir "$UHDD_WEIGHTS/vosr" --include "VOSR2/*" "VOSR_0.5B_os/*" \
      "Qwen-Image-vae-2d/*" "stable-diffusion-2-1-base/*" "sd21_lwdecoder.pth" "torch_cache/*"
}

setup_refir() {
  need UHDD_REPOS UHDD_WEIGHTS
  step "clone ReFIR -> $UHDD_REPOS/ReFIR"
  [[ -d "$UHDD_REPOS/ReFIR/.git" ]] || git clone https://github.com/csguoh/ReFIR "$UHDD_REPOS/ReFIR"
  step "weights: SD-2-base -> $UHDD_WEIGHTS/sd2_base"
  huggingface-cli download stabilityai/stable-diffusion-2-base --local-dir "$UHDD_WEIGHTS/sd2_base"
  step "weights: SeeSR + DAPE -> $UHDD_WEIGHTS/seesr (Google Drive folder from the ReFIR README)"
  if command -v gdown >/dev/null; then
    gdown --folder https://drive.google.com/drive/folders/12HXrRGEXUAnmHRaf0bIn-S8XSK4Ku0JO -O "$UHDD_WEIGHTS/seesr" || \
      echo "WARNING: gdown failed; download the SeeSR folder by hand into $UHDD_WEIGHTS/seesr (seesr/unet, seesr/controlnet, DAPE.pth)"
  else
    echo "gdown not installed: download the SeeSR folder by hand into $UHDD_WEIGHTS/seesr (seesr/unet, seesr/controlnet, DAPE.pth)"
  fi
  [[ -f "$UHDD_WEIGHTS/ram/ram_swin_large_14m.pth" ]] || echo "WARNING: RAM weights missing ($UHDD_WEIGHTS/ram/ram_swin_large_14m.pth, as for OSEDiff)"
}

setup_irag() {
  need UHDD_REPOS UHDD_WEIGHTS
  step "clone iRAG -> $UHDD_REPOS/iRAG"
  [[ -d "$UHDD_REPOS/iRAG/.git" ]] || git clone https://github.com/ByeonghunLee12/iRAG "$UHDD_REPOS/iRAG"
  step "weights: iRAG checkpoint -> $UHDD_WEIGHTS/irag (Google Drive folder from the iRAG README)"
  if command -v gdown >/dev/null; then
    gdown --folder https://drive.google.com/drive/folders/1-onBC231a5EFVmstBzLx8hYkrN0s1qvJ -O "$UHDD_WEIGHTS/irag" || \
      echo "WARNING: gdown failed; download the iRAG pretrained folder by hand into $UHDD_WEIGHTS/irag (iRAG.ckpt)"
  else
    echo "gdown not installed: download the iRAG pretrained folder by hand into $UHDD_WEIGHTS/irag (iRAG.ckpt)"
  fi
  step "weights: StableSR CFW autoencoder -> $UHDD_WEIGHTS/stablesr"
  huggingface-cli download Iceclear/StableSR vqgan_cfw_00011.ckpt --local-dir "$UHDD_WEIGHTS/stablesr"
  step "weights: OpenCLIP ViT-H-14 (laion2b_s32b_b79k) into the HF cache (text encoder of the SD-2.1 UNet)"
  huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin
}

check() {
  local ok=1
  chk() { if eval "$2"; then echo "  ok    $1"; else echo "  FAIL  $1"; ok=0; fi; }
  need UHDD_DATA UHDD_RESULTS
  echo "inputs"
  chk "uhdd imports" "'$PY' -c 'import sys; sys.path.insert(0, \"$ROOT/code\"); import uhdd.layout' 2>/dev/null"
  chk "manifest ok           \$UHDD_RESULTS/dpdd/inputs/manifest.json" \
      "'$PY' -c 'import json,sys; sys.exit(0 if json.load(open(\"$UHDD_RESULTS/dpdd/inputs/manifest.json\"))[\"ok\"] else 1)' 2>/dev/null"
  chk "main env imports diffusers, transformers, peft, fairscale" \
      "'$PY' -c 'import diffusers, transformers, peft, fairscale' 2>/dev/null"
  chk "CUDA available" "'$PY' -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null"
  echo "S3Diff"
  chk "repo        \$UHDD_REPOS/S3Diff"            "[[ -f '${UHDD_REPOS:-}/S3Diff/src/s3diff.py' ]]"
  chk "de_net.pth"                                  "[[ -f '${UHDD_REPOS:-}/S3Diff/assets/mm-realsr/de_net.pth' ]]"
  chk "sd-turbo    \$UHDD_WEIGHTS/sd-turbo"        "[[ -d '${UHDD_WEIGHTS:-}/sd-turbo/unet' ]]"
  chk "s3diff.pkl  \$UHDD_WEIGHTS/s3diff"          "[[ -f '${UHDD_WEIGHTS:-}/s3diff/s3diff.pkl' ]]"
  echo "VOSR"
  chk "repo        \$UHDD_REPOS/VOSR"              "[[ -f '${UHDD_REPOS:-}/VOSR/models/lightningdit.py' ]]"
  chk "VOSR2       \$UHDD_WEIGHTS/vosr/VOSR2"      "[[ -f '${UHDD_WEIGHTS:-}/vosr/VOSR2/args.json' ]]"
  chk "Qwen VAE    \$UHDD_WEIGHTS/vosr/Qwen-Image-vae-2d" "[[ -d '${UHDD_WEIGHTS:-}/vosr/Qwen-Image-vae-2d' ]]"
  chk "DINOv2 hub  \$UHDD_WEIGHTS/vosr/torch_cache" "[[ -d '${UHDD_WEIGHTS:-}/vosr/torch_cache/facebookresearch_dinov2_main' ]]"
  echo "ReFIR / iRAG (tile study)"
  chk "repo        \$UHDD_REPOS/ReFIR"             "[[ -f '${UHDD_REPOS:-}/ReFIR/seesr/seesr_register.py' ]]"
  chk "SD-2-base   \$UHDD_WEIGHTS/sd2_base"         "[[ -d '${UHDD_WEIGHTS:-}/sd2_base/unet' ]]"
  chk "SeeSR       \$UHDD_WEIGHTS/seesr/seesr"      "[[ -d '${UHDD_WEIGHTS:-}/seesr/seesr/controlnet' ]]"
  chk "DAPE        \$UHDD_WEIGHTS/seesr/DAPE.pth"   "[[ -f '${UHDD_WEIGHTS:-}/seesr/DAPE.pth' ]]"
  chk "repo        \$UHDD_REPOS/iRAG"              "[[ -f '${UHDD_REPOS:-}/iRAG/sr/inference.py' ]]"
  chk "iRAG ckpt   \$UHDD_WEIGHTS/irag/iRAG.ckpt"   "[[ -f '${UHDD_WEIGHTS:-}/irag/iRAG.ckpt' ]]"
  chk "CFW VQGAN   \$UHDD_WEIGHTS/stablesr"         "[[ -f '${UHDD_WEIGHTS:-}/stablesr/vqgan_cfw_00011.ckpt' ]]"
  chk "main env imports open_clip, kornia, omegaconf, accelerate" \
      "'$PY' -c 'import open_clip, kornia, omegaconf, accelerate' 2>/dev/null"
  (( ok )) && echo "all checks passed" || echo "some checks failed"
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
             refir)  setup_refir ;;
             irag)   setup_irag ;;
             *)      die "setup inputs|s3diff|vosr|refir|irag" ;;
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
  tiles)   [[ $# -ge 1 ]] || die "tiles select|prep|run|report [args] (see: $0 help, TILE STUDY)"
           "$PY" "$ROOT/code/scripts/tile_study.py" "$@" ;;
  *)       die "unknown command '$cmd' (see: $0 help)" ;;
esac
