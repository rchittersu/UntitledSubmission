# Secure-env runbook — handoff 2: native data (test 76, val, train), protocol v3, all baselines

**One runbook per handoff.** This file is the only current task for the secure env; it replaces the P1b and P2a
runbooks (in git history). Outside replaces it after your patch is applied.

Audience: the Claude agent in the secure environment. Read first:
- [`docs/evaluation.md`](../docs/evaluation.md) — protocol, metrics, tiling standard (§2.7), how to run (§5);
- [`docs/baselines.md`](../docs/baselines.md) — every baseline, weights, commands;
- [`plan/method_plan.md`](method_plan.md) — why we need train/val native sets and anchors (the method, P2b next).

What changed since your P1 report (applied outside): translation registration, `gridshift` seams, DP blur bins,
PSF matching dropped as primary, per-capture calibrated native set (`build_native_set.py`), protocol v3 metrics
(`percbins`, `noharm`, `apsnr`, `msres`), per-method paper tiling (overlap = tile / 8), new baselines (HAT,
OSEDiff, Restormer+TLC, training-free fusion, DATSR reference-based SR).

Rules as always: record every step's outcome; deviations in a table; no internal paths / hostnames / dataset
names in the report (`$UHDD_*` variables); hand off early if blocked (a partial patch is better than none).

**Priority if time is short**: 0 → 3 (test + val only) → 4 (test 76) → 7 (references, G1, G2 bicubic/HAT) →
8 (composite, multiscale, DATSR mosaic) → 5 → 4 (train) → rest. CPU-heavy steps (3, 4) and GPU steps (5, 7, 8)
can run in parallel.

---

## 0. Update and test
```bash
git pull && git checkout -b handoff/h2          # base = origin/main
pip install -r code/requirements.txt            # adds safetensors (diffusers: only for the deferred VAE step)
pytest -q code/tests                            # expect 57 passed
```

## 1. Resources (report)
GPU count / type / memory, max job wall time, free storage under `$UHDD_DATA`, CPU cores. These size the P2b training runs.

## 2. Weights and repos
```bash
# IFAN (public mirror, bit-identical to the official net)
wget -P $UHDD_WEIGHTS/ifan https://github.com/jacobsparts/ifan-rs/releases/download/v0.1.0/IFAN.safetensors
sha256sum $UHDD_WEIGHTS/ifan/IFAN.safetensors   # baa8ba206149a7e4c0350a9817425cf5e496c2008fcab7e6a92914f5fe96fe82
python code/scripts/check_against_official.py ifan     # expect max|ours-ref| ~1e-6
# DATSR (reference-based SR; no mmcv needed)
git clone https://github.com/caojiezhang/DATSR $UHDD_REPOS/DATSR
gh release download -R caojiezhang/DATSR -p '*.pth' -D $UHDD_WEIGHTS/datsr
# HAT (official Google Drive per the HAT README; or the HF mirrors in docs/baselines.md C2)
git clone https://github.com/XPixelGroup/HAT $UHDD_REPOS/HAT      # weights -> $UHDD_WEIGHTS/hat/
# OSEDiff (CUDA): commands in docs/baselines.md C3 (repo, SD2.1-base, RAM swin-L)
git clone https://github.com/swz30/Restormer $UHDD_REPOS/Restormer   # if not present (TLC baseline reuses it)
```
If a download is blocked, report it and continue without that model.

## 3. Official pairing + DP maps (test, val, train)
```bash
for s in test val train; do
  python code/scripts/pair_official.py --root <dd_dp_dataset_png> --split $s --out $UHDD_DATA/dpdd_1680/$s
  python code/scripts/dp_maps.py --left $UHDD_DATA/dpdd_1680/$s/inputs_l --right $UHDD_DATA/dpdd_1680/$s/inputs_r \
      --out $UHDD_DATA/dpdd_1680/$s/dp_maps --procs 16
done
ln -sfn $UHDD_DATA/dpdd_1680/test/dp_maps $UHDD_DATA/dpdd_native/dp_maps     # path used by the v3 configs
```
Expected 76 / 74 / 350 pairs. Report pair counts and each split's focal-plane line (outside, test: median 0.129,
min 0.007, max 0.721).

## 4. Native sets v2 from raw (test = 76 with outdoor, val, train)
`build_native_set.py` (docs/evaluation.md §2.1): per-capture calibration to the official rendering, translation
registration, ×1/×2/×4 + masks. Run once per raw zip/folder (indoor, outdoor); finished pairs are skipped.
```bash
for s in test val train; do OUT=$UHDD_DATA/dpdd_native_v2; [ $s != test ] && OUT=$UHDD_DATA/dpdd_native_$s
  for z in <indoor raw zip or dir> <outdoor raw zip or dir>; do
    python code/scripts/build_native_set.py --raw $z --official-inputs $UHDD_DATA/dpdd_1680/$s/inputs \
        --official-targets $UHDD_DATA/dpdd_1680/$s/targets --pairs $UHDD_DATA/dpdd_1680/$s/pairs.csv --out $OUT --procs 16
  done
done
```
Storage ≈ 0.4 GB per pair (train ≈ 140 GB, val 30 GB, test 30 GB); if short, train last and drop its x2.
Report per split: pairs built vs expected; the summary line (calibration medians/min, registration shift
median/max; outside on the 37 indoor test pairs: 38.8 / 39.0 dB, shift median 1.5, max 6.6 px); the 5 worst rows of
`build_report.csv`; any shift > 20 px (describe, do not drop). Keep indoor/outdoor identifiable (report both).

## 5. Anchors at ×4 (train / val / test) and the train/test anchor gap
DPDD-trained deblurrers saw DPDD train at 1680×1120 = our ×4 level → train-set anchors are probably too good.
```bash
for s in train val v2; do D=$UHDD_DATA/dpdd_native_$s
  for m in restormer_dpdd drbnet_single ifan lakdnet_dpdd_l bokehlicious_deblur; do
    python code/scripts/run_model.py --model $m --inputs $D/x4/inputs --out $UHDD_RESULTS/anchors/$s/$m --gpus all
    python code/scripts/evaluate.py --pred $UHDD_RESULTS/anchors/$s/$m --targets $D/x4/targets --masks $D/x4/masks \
        --metrics psnr,ssim --scale 4 --crop 16 --gpus all --tag anchor
  done
  python code/scripts/evaluate.py --pred $D/x4/inputs --targets $D/x4/targets --masks $D/x4/masks \
      --metrics psnr,ssim --scale 4 --crop 16 --gpus all --tag anchor        # blurry-input reference
done
```
(`run_model.py` uses each model's paper tiling = whole image at ×4.) Report: rows = models + input, columns =
PSNR/SSIM on train / val / test and the gain over the input per split. Keep the outputs (P2b training inputs).

## 6. (deferred) VAE ceiling
Not in this handoff — skip. (Variant B backbone choice is deferred; `vae_ceiling.py` stays in the repo.)

## 7. Evaluation protocol v3 — registry pipelines
```bash
python code/scripts/run_matrix.py code/experiments/dpdd_eval_v3.yaml --dry-run | less
python code/scripts/run_matrix.py code/experiments/dpdd_eval_v3.yaml --gpus all
```
- Results dir `$UHDD_RESULTS/dpdd_v2` (new: the step cache is keyed by pipeline, not by dataset, so P1 outputs on
  the v1 renderings are **not** reused). Tag `dpdd3`.
- Groups: references · G1 native patch-wise (paper tiling: 1120 tiles; Bokehlicious 1500) · 512-tile ablation ·
  A4 Restormer+TLC · G2 ×4 + {bicubic, SwinIR-real, HAT-L, Real-HAT} · OSEDiff · ×2 + {bicubic, SwinIR-real ×2}.
- Memory: 1120-px tiles at batch 2 (P1: Restormer 1024 px batch 2 peaked at 27 GB); lower `tile_batch` if needed
  and say so.

## 8. Training-free (B) and reference-based SR (D) baselines
1. Val anchors for tuning (DRBNet shown; repeat for restormer_dpdd):
   ```bash
   V=$UHDD_DATA/dpdd_native_val; R=$UHDD_RESULTS/dpdd_v2/val
   python code/scripts/run_model.py --model drbnet_single --inputs $V/x4/inputs --out $R/drb_x4 --gpus all
   python code/scripts/run_model.py --model bicubic_x4 --inputs $R/drb_x4 --out $R/drb_x4+bic --gpus all
   python code/scripts/run_model.py --model drbnet_single --inputs $V/x2/inputs --out $R/drb_x2 --gpus all
   python code/scripts/run_model.py --model bicubic_x2 --inputs $R/drb_x2 --out $R/drb_x2+bic --gpus all
   ```
2. Tune on val (composite, multiscale, guided; exemplar uses the composite thresholds):
   ```bash
   P=$UHDD_RESULTS/dpdd_v2/fusion_params; mkdir -p $P
   for m in composite multiscale guided; do
     python code/scripts/fuse_baselines.py --method $m --inputs $V/inputs --x4 $R/drb_x4+bic --x2 $R/drb_x2+bic \
         --dp-maps $UHDD_DATA/dpdd_1680/val/dp_maps --tune --targets $V/x1/targets --masks $V/x1/masks \
         --params $P/$m.json --procs 8
   done
   cp $P/composite.json $P/exemplar.json
   ```
3. Run + evaluate on test: `code/experiments/run_baselines_bd.sh drbnet_single` and `... restormer_dpdd`
   (needs step 7's anchors; DATSR ≈ 150 tiles of 128 LR px per image).
Report the tuned parameters (top-5 printout), runtimes, and DATSR's number of in-focus candidates per image.

## 9. Tables for the report
```bash
J=$UHDD_RESULTS/dpdd_v2
# main comparison vs the DP composite (docs/evaluation.md §6 columns)
python code/scripts/summarize.py $J/pipelines/*__dpdd3.json $J/baselines_bd/*/metrics_dpdd3.json --ci \
    --ref "drbnet_single fuse:composite" --cols psnr,apsnr,ssim,dists_b3,lpips_b3,psnr_b3,dpsnr_b0,psnr_s4,pipeline_time_s,peak_mem_gb
# per-bin gains over the input
python code/scripts/summarize.py <same jsons> --ci --ref "input (blurry)" --cols psnr_b0,psnr_b1,psnr_b2,psnr_b3
# resolution sweep (figure data)
python code/scripts/summarize.py <same jsons> --cols psnr_s4,psnr_s2,psnr,dists_s4,dists_s2,dists
```
Paste outputs verbatim; indoor-only (37) and all-76 separately.

## 10. Optional
- Paired focal-plane MTF on the outdoor raws: `python code/scripts/paired_edge_mtf.py --raw <outdoor> ...
  --focus-dir $UHDD_DATA/dpdd_native/dp_maps --out paired_focus_outdoor.json` → ratio table and σ.

## 11. Handoff
Report `handoff/from_secure/<date>_h2.md` (template in CLAUDE.md) with steps 0–10, deviations, tables, and
observations in words (where methods fail, by blur bin; visual notes on HAT / OSEDiff / DATSR outputs).
In the same patch, update the status lists and result sections of `docs/evaluation.md` and `docs/baselines.md`
(✅/🔶/⏳ and numbers) — `docs/` is allowed in patches. Commit `code/` fixes with a CPU test where possible.
`handoff/make_patch.sh`.
