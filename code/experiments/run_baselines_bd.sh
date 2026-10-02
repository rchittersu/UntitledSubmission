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
EV="--targets $D/x1/targets --masks $D/x1/masks --inputs $D/inputs --dp-maps $DP --crop 64 --tag dpdd3 --gpus all \
    --metrics psnr,ssim,hb,lpips,dists,blurbins,percbins,noharm,apsnr,msres"
py=python
for m in composite multiscale guided detail exemplar; do
  P=""; [ -f $PARAMS/$m.json ] && P="--params $PARAMS/$m.json"
  $py code/scripts/fuse_baselines.py --method $m --inputs $D/inputs --x4 $UP4 --x2 $UP2 --anchor4 $A4 \
      --dp-maps $DP --out $O/${M}_$m $P --procs 8 --device cuda
  $py code/scripts/evaluate.py --pred $O/${M}_$m --label "$M fuse:$m" $EV
done
for ref in mosaic colocated; do
  $py code/scripts/refsr_baseline.py --anchors4 $A4 --inputs $D/inputs --dp-maps $DP --ref $ref \
      --weights restoration_mse --out $O/${M}_datsr_mse_$ref
  $py code/scripts/evaluate.py --pred $O/${M}_datsr_mse_$ref --label "$M @x4 + DATSR-mse ($ref)" $EV
done
$py code/scripts/refsr_baseline.py --anchors4 $A4 --inputs $D/inputs --dp-maps $DP --ref mosaic \
    --weights restoration_gan --out $O/${M}_datsr_gan_mosaic
$py code/scripts/evaluate.py --pred $O/${M}_datsr_gan_mosaic --label "$M @x4 + DATSR-gan (mosaic)" $EV
