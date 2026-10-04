# Secure-env runbook — handoff 3: evaluation only (complete the protocol, DATSR, visual review, data facts)

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
- Visual review: you do it (§6) and the user also reviews the same material manually, so keep it available in the
  env and make your notes concrete enough to be checked against the images (scene, crop, what is seen).

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

## 4. Data facts for later (no training)
- Train / val native sets: summary lines of `build_report.csv` (calibration medians/min, registration shift
  median/max), list of pairs > 10 px, indoor/outdoor counts; same for test (outdoor calibration quality).
- Confirm DP blur maps exist for train and val (counts).
- **Val proxy for future model selection**: define a cheap subset (e.g. 15 fixed val scenes, stratified by blur
  fraction) and check it against the full val set on existing pipelines (input, native DRBNet, ×4 + bicubic,
  composite, exemplar): Spearman of the method ranking and per-method PSNR difference proxy vs full. List the scenes.

## 5. Visual review (you and the user, on the same material)
Extend the visual set in the env: add DATSR (both variants) and the composite / exemplar fusions to the 8 selected
captures (same crops: sharp region, high-error region, centre; full-resolution crops, not downscaled montages).
Review **all 8 captures at 100 % zoom** and report per capture and per method, briefly and concretely:
- what is recovered vs. the target (edges, text, texture, grain), what is still blurry;
- any invented content (text/glyph changes, texture that is not in the target, faces), halos, colour shifts, banding,
  tile seams or tile-to-tile texture changes;
- whether in-focus regions are altered vs. the input.
Add a short cross-capture summary table (method × issue, with the captures where it occurs). Keep the material in place
and describe generically where it is, so the user can check the same crops.

## 6. Handoff
Report `handoff/from_secure/<date>_h3.md` (template in CLAUDE.md): steps 0–5, deviations, tables (all 76, CIs), the
visual review. Update the result numbers in `docs/evaluation.md` §7 and `docs/baselines.md` (status lists).
`handoff/make_patch.sh`.
