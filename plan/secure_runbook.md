# Secure-env runbook — handoff 3: evaluation only (complete the protocol, perspective experiments, data facts)

**One runbook per handoff.** This file replaces the handoff-2 runbook (in git history). Your handoff-2 patch and
addendum notes were applied and reviewed outside — thank you; review notes are in `docs/evaluation.md` §7.6.

Audience: the Claude agent in the secure environment. Read first: `docs/evaluation.md` (§9 decision log, §7.6),
`docs/baselines.md`.

## Standing rules (new, from the user)
- **Always work with all 76 test scenes** — every table, analysis, figure and sensitivity check. Indoor/outdoor
  splits are only an extra breakdown, never the main or only result.
- **This handoff is evaluation only** — no training of any model.
- **Protocol is frozen** (2026-10-04, decision log): headline = out-of-the-box PSNR, SSIM, LPIPS, DISTS at native
  resolution, MUSIQ, CLIPIQA, PSNR/SSIM at ×4, runtime. Per-bin, no-harm, aligned PSNR, sweep = diagnostics only.
- **Report results; keep interpretation short.** The user will look at the results and the images and form their
  own view. Facts, tables, CIs, and anything surprising or broken — no long narratives.
- Visual review is done by the user manually: keep the visual material available in the env (see §6), no written
  visual review needed.

Same rules as always: outcome of every step; deviations table; no internal paths/hostnames; hand off early if blocked.

---

## 0. Update and test
```bash
git pull && git checkout -b handoff/h3        # base = origin/main
pytest -q code/tests                          # expect 63 passed
```
If you have local commits not in the handoff-2 patch (e.g. the ×4 Bokehlicious / LaKDNet / IFAN + {bicubic,
HAT-L} group in `dpdd_eval_v3.yaml`), rebase them onto the new main; `dpdd_eval_v3.yaml` changed outside (tag
`dpdd4`, new `t1_deblur_x4` group) — keep both.

What changed outside (for this handoff):
- `run_matrix.py`: pipeline time / peak memory are now per image (time summed over steps, memory = max over steps),
  written into a per-pipeline CSV so means and CIs agree (fixes the "mean outside its CI" bug). Also reports
  `peak_mem_gb_max`.
- `msres` now includes `ssim_s2/ssim_s4`.
- `refsr_baseline.py --ref mosaic_focus`: DATSR reference without the blurry co-located crop for defocused tiles.
- `exemplar_pilot.py`: anchor folders via env `UP4`, `A4`.
- `dpdd_eval_v3.yaml`: tag **`dpdd4`** with the full frozen headline; group `t1_deblur_x4` (5 deblurrers + input at ×4).
- Report terminology: bin **b0 = focal plane (in focus)**, b3 = most defocused (the handoff-2 report had it reversed
  in two places).

## 1. Complete the frozen protocol on all 76
```bash
python code/scripts/run_matrix.py code/experiments/dpdd_eval_v3.yaml --gpus all      # inference cached; re-evaluates under dpdd4
```
- Adds whole-image LPIPS, MUSIQ, CLIPIQA, SSIM at ×4 for every row, and Table 1 (deblur-only at ×4 for Restormer,
  LaKDNet-L, DRBNet, IFAN, Bokehlicious + input, scored at ×4).
- Re-run `code/experiments/run_baselines_bd.sh drbnet_single` and `... restormer_dpdd` (fusion re-evaluated under
  `dpdd4`; tuned parameters reused).
- Slow SR models (SwinIR-real, Real-HAT, OSEDiff) stay on Restormer and DRBNet only (user decision; no need to add
  them for LaKDNet-L / IFAN). Bicubic and HAT-L for all five anchors if your local group already does it.

## 2. DATSR (reference-based SR) on all 76
```bash
DATSR=1 code/experiments/run_baselines_bd.sh drbnet_single     # runs mosaic and mosaic_focus (MSE weights), sharded
```
Report both variants. Also report, per variant, the median number of in-focus candidates per image and how many
tiles had no in-focus candidate. Restormer anchor only if time permits.

## 3. Timing pass
On an idle node (or a reserved GPU, nothing else running), one timing run per pipeline on 5 fixed test scenes
(same 5 for all; list them): `pipeline_time_s` and `peak_mem_gb` from the new aggregation. State the GPU type.

## 4. Perspective experiments (all 76)
**4a. Upper bounds with a perfect anchor** (how much is lost purely by going through ×4):
```bash
T4=$UHDD_DATA/dpdd_native_v2/x4/targets; R=$UHDD_RESULTS/dpdd_v2/oracle
python code/scripts/run_model.py --model bicubic_x4 --inputs $T4 --out $R/gt_x4+bicubic --gpus all
python code/scripts/run_model.py --model hat_l_x4   --inputs $T4 --out $R/gt_x4+hat_l --gpus all
python code/scripts/fuse_baselines.py --method composite --inputs $UHDD_DATA/dpdd_native_v2/inputs --x4 $R/gt_x4+bicubic \
    --dp-maps $UHDD_DATA/dpdd_native/dp_maps --out $R/gt_x4+composite --params $UHDD_RESULTS/dpdd_v2/fusion_params/composite.json
# evaluate each with the dpdd4 metric list (same EV options as run_baselines_bd.sh)
```
**4b. Anchor quality vs final quality**: per image, PSNR of the ×4 anchor (at ×4) vs PSNR of anchor + bicubic and of
the composite (native), for Restormer and DRBNet — report the Spearman correlation and the per-image CSV
(name, anchor_psnr_x4, final_psnr, frac_b0..b3).

**4c. Exemplar transfer with its controls**, DRBNet anchor:
```bash
NATIVE=$UHDD_DATA/dpdd_native_v2 RESULTS=<dir with the anchor folders> DPMAPS=$UHDD_DATA/dpdd_native/dp_maps \
UP4=<x4 + bicubic folder> A4=<x4 anchor folder> python code/analysis/exemplar_pilot.py --device cuda
python code/analysis/exemplar_pilot.py --summary
```
Paste the summary tables (exemplar / random / oracle vs composite, paired CIs).

**4d. Per-image view**: CSV over the 76 scenes for input, native DRBNet, ×4 + bicubic, composite, exemplar,
OSEDiff: name, indoor/outdoor, frac_b0..b3, psnr, dists, dpsnr_b0. (Plot data for the user; no interpretation.)

**4e. Sensitivity**: main-table rows recomputed without the 3 test scenes with > 10 px registration shift
(`1P0A1526`, `1P0A1696`, `1P0A1772`), as paired Δ vs the full-76 numbers (`subset_results.py`).

**4f. Breakdown** (supplementary only): indoor-37 / outdoor-39 for the main rows.

## 5. Data facts for later (no training)
- Train / val native sets: summary lines of `build_report.csv` (calibration medians/min, registration shift
  median/max), list of pairs > 10 px, indoor/outdoor counts; same for test (outdoor calibration quality).
- Confirm DP blur maps exist for train and val (counts).
- **Val proxy for future model selection**: define a cheap subset (e.g. 15 fixed val scenes, stratified by blur
  fraction) and check it against the full val set on existing pipelines (input, native DRBNet, ×4 + bicubic,
  composite, exemplar): Spearman of the method ranking and per-method PSNR difference proxy vs full. List the scenes.

## 6. Visual material (for the user's manual review)
Keep / extend the visual set in the env: add DATSR (both variants) and the composite/exemplar fusions to the 8
selected captures; same crops as before. Report only where it is (generic description) and what it contains.

## 7. Handoff
Report `handoff/from_secure/<date>_h3.md` (template in CLAUDE.md): steps 0–6, deviations, tables (all 76, CIs),
CSV blocks for 4b/4d. Update the result numbers in `docs/evaluation.md` §7 and `docs/baselines.md` (status lists).
`handoff/make_patch.sh`.
