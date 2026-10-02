# Secure-env runbook — P2a: training data, anchors, VAE ceiling

Audience: the Claude agent in the secure environment. Read `plan/method_plan.md` first (decided
2026-10-02): we now build the method — a native-resolution guided upsampler in two variants (A feed-forward,
B one-step DiT prior) taking the blurry native input + a 1/4-res deblurred anchor + a blur map + in-focus
exemplars. Training happens here. This runbook prepares everything training needs; the training code
arrives in the next pull (P2b).

**Order with P1b** (`plan/secure_runbook_p1b.md`, not yet reported back): do P1b steps 0–3 first
(IFAN, DP maps, native test set v2 — needed by both). Then run this runbook's CPU-heavy steps 2–3
**in parallel with** P1b step 5 (GPU). One handoff patch for both is fine (two report files).

Same rules: record every step's outcome; deviations in a table; hand off early if blocked;
no internal paths / hostnames in the report (use `$UHDD_DATA` etc.).

---

## 0. Update and test

```bash
git pull
pip install -r code/requirements.txt          # adds diffusers
pytest -q code/tests                          # expect 47 passed
```

## 1. Resources (needed to size P2b)

Report: number and type of GPUs (memory each), max job wall time, free storage under `$UHDD_DATA`,
CPU cores. Whether the SD3.5-Medium licence may be accepted / weights are available here (step 5).

## 2. Official train/val pairing + DP maps

```bash
for s in train val; do
  python code/scripts/pair_official.py --root <dd_dp_dataset_png> --split $s --out $UHDD_DATA/dpdd_1680/$s
  python code/scripts/dp_maps.py --left $UHDD_DATA/dpdd_1680/$s/inputs_l --right $UHDD_DATA/dpdd_1680/$s/inputs_r \
      --out $UHDD_DATA/dpdd_1680/$s/dp_maps --procs 16
done
```
Expected: train 350 pairs, val 74 (checked outside for val and test: test reproduces the pairing used so
far). Report the printed pair counts and the dp_maps focal-plane line per split.

## 3. Native train / val sets (same recipe as the test set)

`build_native_set.py` reads the CR2s from a folder or the DPDD raw zip; run it once per raw zip
(indoor, outdoor) — pairs without both CR2s are skipped and finished pairs are not rebuilt.
```bash
for s in train val; do for z in <indoor raw zip or dir> <outdoor raw zip or dir>; do
  python code/scripts/build_native_set.py --raw $z --official-inputs $UHDD_DATA/dpdd_1680/$s/inputs \
      --official-targets $UHDD_DATA/dpdd_1680/$s/targets --pairs $UHDD_DATA/dpdd_1680/$s/pairs.csv \
      --out $UHDD_DATA/dpdd_native_$s --procs 16
done; done
```
Storage: ≈ 0.4 GB per pair over x1+x2+x4 (≈ 140 GB train, 30 GB val). If short, build val first and
report; we can drop x2 (not used in training).
Also the **outdoor test raws** if present (into the native test set v2, `--out $UHDD_DATA/dpdd_native_v2`).
Report per split: #pairs built vs expected, the summary line (calibration medians/min, registration
shift median/max), the 5 worst rows of `build_report.csv`, any pair with shift > 20 px (describe, do not drop).

## 4. Anchors at 1/4 + the train/test anchor gap

The method's training anchors are off-the-shelf DPDD deblurrers run on the ×4 inputs. These models were
trained on DPDD train at 1680×1120 = our ×4 level, so their train-set outputs are probably better than on
test: measure it.
```bash
for s in train val v2; do D=$UHDD_DATA/dpdd_native_$s; [ $s = v2 ] && D=$UHDD_DATA/dpdd_native_v2
  for m in restormer_dpdd drbnet_single ifan lakdnet_dpdd_l bokehlicious_deblur; do
    python code/scripts/run_model.py --model $m --inputs $D/x4/inputs --out $UHDD_RESULTS/anchors/$s/$m --gpus all --tile 0
    python code/scripts/evaluate.py --pred $UHDD_RESULTS/anchors/$s/$m --targets $D/x4/targets --masks $D/x4/masks \
        --metrics psnr,ssim --scale 4 --crop 16 --gpus all --tag anchor
  done
  python code/scripts/evaluate.py --pred $D/x4/inputs --targets $D/x4/targets --masks $D/x4/masks \
      --metrics psnr,ssim --scale 4 --crop 16 --gpus all --tag anchor        # blurry input reference
done
```
(If a model OOMs whole-image at 1680×1120, use `--tile 1024 --overlap 128` and say so.)
Report a table: rows = models + blurry input, columns = PSNR/SSIM on train / val / test, and the
**gain over the blurry input** per split. A gain on train much larger than on val/test = overfitted
anchors (expected; tells us how strong the synthetic-anchor mix must be).
Keep the anchor outputs: they are training inputs for P2b.

## 5. VAE ceiling (chooses the backbone of variant B)

On the native test set v2 (all available pairs), 1024-px crops, 2 most in-focus + 2 most defocused per image:
```bash
python code/scripts/vae_ceiling.py --targets $UHDD_DATA/dpdd_native_v2/x1/targets \
    --dp-maps $UHDD_DATA/dpdd_native/dp_maps --per-region 2 --dtype fp32 --out vae_ceiling.csv \
    --vae sdxl=madebyollin/sdxl-vae-fp16-fix \
    --vae flux=black-forest-labs/FLUX.1-schnell:vae \
    --vae dcae=mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers \
    --vae sd35=stabilityai/stable-diffusion-3.5-medium:vae      # only if licence/weights allowed (step 1)
```
Report the printed table verbatim (outside is running the same on the M4 for sdxl/flux/dcae; the numbers
should match to ~0.01 dB — a cross-check of both setups).

## 6. Handoff

Report `handoff/from_secure/<date>_p2a.md` (+ the P1b report if done in the same patch) with steps 0–5,
deviations, the tables. `handoff/make_patch.sh`.
