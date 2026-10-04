#!/usr/bin/env bash
# Group B (training-free fusion) and group D (reference-based SR) baselines on the native test set
# (docs/baselines.md). Needs the anchors from code/experiments/dpdd_eval_v3.yaml (run it first) and DP maps.
# Usage: code/experiments/run_baselines_bd.sh [anchor model, default drbnet_single]   (env: UHDD_DATA, UHDD_RESULTS)
set -euo pipefail
M=${1:-drbnet_single}
D=$UHDD_DATA/dpdd_native_v2; DP=$UHDD_DATA/dpdd_native/dp_maps
S=$UHDD_RESULTS/dpdd_v2/steps; O=$UHDD_RESULTS/dpdd_v2/baselines_bd
A4=$S/x4__${M}@whole                                   # 1/4-res deblur (anchor)
UP4=$S/x4__${M}@whole__bicubic_x4@whole              # anchor + bicubic x4
UP2=$S/x2__${M}@t1120o140__bicubic_x2@whole           # x2 deblur + bicubic x2
PARAMS=${PARAMS:-$UHDD_RESULTS/dpdd_v2/fusion_params}  # tuned on val (see docs/baselines.md B, "tuning")
EV="--targets $D/x1/targets --masks $D/x1/masks --inputs $D/inputs --dp-maps $DP --crop 64 --tag dpdd4 --gpus all \
    --metrics psnr,ssim,lpips,dists,musiq,clipiqa,msres,hb,blurbins,percbins,noharm,apsnr"
py=python
mkdir -p $O
for m in composite multiscale guided detail exemplar; do
  # one runner per method: atomic lock, so several nodes can share this script (a stale lock after a crash: rm -r $O/.lock_*)
  [ -f $O/${M}_$m/metrics_dpdd4.json ] && { echo "skip $m: already evaluated"; continue; }
  mkdir $O/.lock_${M}_$m 2>/dev/null || { echo "skip $m: finished or running elsewhere"; continue; }
  P=""; [ -f $PARAMS/$m.json ] && P="--params $PARAMS/$m.json"
  $py code/scripts/fuse_baselines.py --method $m --inputs $D/inputs --x4 $UP4 --x2 $UP2 --anchor4 $A4 \
      --dp-maps $DP --out $O/${M}_$m $P --procs 8 --device cuda
  $py code/scripts/evaluate.py --pred $O/${M}_$m --label "$M fuse:$m" $EV
done
# DATSR is slow (>= 15 min per 6720x4480 image on one GPU): shard the images over all GPUs of the node (one process per GPU).
# DATSR=1 enables the headline variant (MSE weights, mosaic reference); DATSR_FULL=1 adds colocated and the GAN weights.
datsr() {  # $1 ref  $2 weights  $3 out dir  $4 label
  mkdir -p $3
  names=($(ls $D/inputs | sed 's/\.png$//')); G=${NGPU:-$(nvidia-smi -L | wc -l)}
  for ((g = 0; g < G; g++)); do
    sub=""; for ((i = g; i < ${#names[@]}; i += G)); do sub="$sub,${names[$i]}"; done
    CUDA_VISIBLE_DEVICES=$g $py code/scripts/refsr_baseline.py --anchors4 $A4 --inputs $D/inputs --dp-maps $DP --ref $1 \
        --weights $2 --out $3 --only ${sub#,} &
  done
  wait
  $py code/scripts/evaluate.py --pred $3 --label "$4" $EV
}
if [ "${DATSR:-0}" = 1 ]; then   # opt-in: too slow for the default run (see above)
datsr mosaic restoration_mse $O/${M}_datsr_mse_mosaic "$M @x4 + DATSR-mse (mosaic)"
datsr mosaic_focus restoration_mse $O/${M}_datsr_mse_mosaic_focus "$M @x4 + DATSR-mse (in-focus mosaic)"
fi
if [ "${DATSR_FULL:-0}" = 1 ]; then
  datsr colocated restoration_mse $O/${M}_datsr_mse_colocated "$M @x4 + DATSR-mse (colocated)"
  datsr mosaic restoration_gan $O/${M}_datsr_gan_mosaic "$M @x4 + DATSR-gan (mosaic)"
fi
