# Secure-env runbook — P1b: protocol v2, re-registration, complete P1 tables

Audience: the Claude agent in the secure environment. Builds on `plan/secure_runbook_p1.md` and your
report `handoff/from_secure/2026-10-01_p1-eval.md` (applied outside, thank you — very useful).
Read `plan/evaluation.md` → "Status and decisions (2026-10-02)" first: it explains *why* each step below exists.

Summary of what changed outside (all CPU-tested, 42 tests; `git pull` to get them):
- registration default **euclidean** (homography overfits the f/4–f/22 blur difference: your 25–79 px
  shifts are ~0.4 px on the official 1680×1120 pairs with translation/euclidean);
- new metric **`gridshift`** (content-free seams / tiling sensitivity; the old `seam_ratio` is noise);
- **DP defocus maps** (`code/scripts/dp_maps.py`) → metric **`blurbins`** and `estimate_gt_mtf.py --focus-dir`;
- metric **`pmg`** (Airy + Gaussian σ, sensitivity for your MTF finding);
- **`summarize.py --ci --ref`** (bootstrap CIs, paired per-image differences);
- IFAN weights from a public GitHub mirror (`IFAN.safetensors`, bit-identical to the official net);
- `code/experiments/dpdd_p1.yaml` eval tag **`dpdd2`** with the new metrics.

Same rules as before: note the outcome of every step; deviations in a table; hand off early if blocked.

---

## 0. Update and test

```bash
git pull && git checkout -b handoff/p1b      # base = origin/main
pip install -r code/requirements.txt         # adds safetensors
pytest -q code/tests                         # expect 42 passed
```

## 1. IFAN

```bash
wget -P $UHDD_WEIGHTS/ifan https://github.com/jacobsparts/ifan-rs/releases/download/v0.1.0/IFAN.safetensors
sha256sum $UHDD_WEIGHTS/ifan/IFAN.safetensors   # baa8ba206149a7e4c0350a9817425cf5e496c2008fcab7e6a92914f5fe96fe82
python code/scripts/check_against_official.py ifan   # expect max|ours-ref| ~1e-6
```
If the GitHub release is blocked too, report it and continue without IFAN.

## 2. DP defocus maps (official 1680×1120 dual-pixel views of the blurry input)

The views are in the same Hugging Face copy you used (`dd_dp_dataset_png/test_l/source/*_L.png`,
`test_r/source/*_R.png`, ~1.3 GB) or in the authors' `dd_dp_dataset_canon.zip`.

```bash
python code/scripts/dp_maps.py --left <test_l/source> --right <test_r/source> --out $UHDD_DATA/dpdd_native/dp_maps
```
Map names are the **blurry-input stems**, matching the native set. Report the printed focal-plane
fraction line. (Outside on all 76 test images: median 0.129, min 0.007, max 0.721.)

## 3. Re-register the native targets (euclidean)

```bash
D=$UHDD_DATA/dpdd_native
mv $D/x1 $D/x1_homography                     # keep v1 for the comparison below
python code/scripts/register_pairs.py --inputs $D/inputs --targets $D/targets --out $D/x1 --motion euclidean --procs 8
python code/scripts/prepare_scales.py --inputs $D/inputs --targets $D/x1/targets --masks $D/x1/masks \
    --out $D --factors 2 4 --filter area --overwrite
```
Report: the summary line, and a per-image table for the 12 images that had > 10 px with homography:
`name | homography shift (v1) | euclidean shift (v2) | ECC v1 | ECC v2`.
Expected (from the official 1680×1120 pairs): euclidean shifts of a few native px at most. If any
image still has > 20 px, flag it and describe what you see (do not drop it).

**Before/after check (one model)**: evaluate the *existing* `x1__restormer_dpdd@t512o64` outputs
against v1 and v2 targets (`evaluate.py --metrics psnr,ssim,lpips --tag reg_v1/reg_v2`) and report
both rows: this isolates the registration effect on the numbers.

## 4. Focal-plane MTF (is the f/22 target blurrier than in-focus f/4 at native resolution?)

```bash
python code/scripts/estimate_gt_mtf.py --zip cr2.zip --names <f4 stems> --focus-dir $D/dp_maps --out f4_focus.json
python code/scripts/estimate_gt_mtf.py --zip cr2.zip --names <f22 stems> --focus-of <f4 stems> --focus-dir $D/dp_maps --out f22_focus.json
```
(`--names` for the f/22 run = the CR2 stems of the targets, `--focus-of` = their paired f/4 stems in the
same order.) Report the same table as in your P1 report for both, plus edges/image. The question:
on the **same focal-plane regions**, is MTF(f/22) below MTF(f/4)? If yes, fit the Gaussian σ of the
*ratio* (that σ, not 0.8, would go into `pmg`). If MTF(f/4) is lower, say so: then PM blurring is not
justified at all and `psnr` stays primary.

## 5. Complete P1 with protocol v2

```bash
python code/scripts/run_matrix.py code/experiments/dpdd_p1.yaml --dry-run | head -40
python code/scripts/run_matrix.py code/experiments/dpdd_p1.yaml --gpus all
```
- Inference outputs from P1 are reused (cache); new: IFAN rows, the remaining G2 pipelines, the
  `references` row (blurry input), and one shifted-grid twin per tiled pipeline (for `gridshift`).
- All evaluations rerun under tag `dpdd2` (new registration + new metrics).
- If time is short, priority: (1) `references`, (2) G2 remaining with **bicubic** and **SwinIR-real**,
  (3) IFAN, (4) G2 with SwinIR classical.

## 6. Tables for the report

For every group (`t1_resolution_gap`, `g1_native_patchwise`, `g2_lowres_upsample`, `references`):
```bash
python code/scripts/summarize.py $UHDD_RESULTS/dpdd_p1/pipelines/*__dpdd2.json --ci \
    --cols psnr,pm_psnr,pmg_psnr,ssim,lpips,dists,hb_nmse_db,gs_psnr,gs_seam_ratio,pipeline_time_s
```
And these paired comparisons at native resolution (all on the same images):
```bash
# 1) the key one: native patch-wise vs deblur-at-1/4 + upsampling, per model
python code/scripts/summarize.py <"m native t512" json> <"m @x4 + bicubic_x4" json> <"m @x4 + swinir_x4_real" json> \
    --ref "m native t512" --cols psnr,lpips,dists,hb_nmse_db
# 2) per-bin gains over the blurry input
python code/scripts/summarize.py <native rows> <"input (blurry)" json> --ref "input (blurry)" \
    --cols psnr_b0,psnr_b1,psnr_b2,psnr_b3
```
Paste the outputs verbatim.

## 7. Outdoor raws (if they arrived)

Develop with the existing colour map (`develop_raw.py develop`), add to the native set, rerun steps 2–6
on all 76 (`--tag dpdd2_all`). Report indoor-only and all-76 tables separately.

## 8. Handoff

Report `handoff/from_secure/<date>_p1b.md` with: steps 0–7 outcomes; deviations; the tables;
registration v1 vs v2; focal-plane MTF; observations in words (where do the methods fail, by blur bin).
Commit any `code/` fixes with a CPU test where possible. `handoff/make_patch.sh`.
