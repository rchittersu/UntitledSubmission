# Evaluation — native-resolution defocus deblurring

Canonical description of **how we evaluate**: test data, protocol, every metric (concept + exact
definition + code), how to run it, what is done (with numbers), what is priority, what is todo.
Baselines are in [`baselines.md`](baselines.md). The method is in [`../plan/method_plan.md`](../plan/method_plan.md).
History of decisions (dated) is in [§9](#9-decision-log). The current secure-env task is
[`../plan/secure_runbook.md`](../plan/secure_runbook.md).

Legend: ✅ done · 🔶 priority (next 1–2 weeks) · ⏳ todo (later / after phase 1) · 📝 note only ·
**[S]** runs in the secure env (data + GPUs) · **[O]** outside (code, local M4 checks).

---

## 0. Status at a glance

### ✅ Done
| Item | Where | Key numbers |
|---|---|---|
| Native test set v2 from raw (indoor, 37 pairs) | [O] built, [S] to rebuild | calibration vs official @1/4: inputs 38.8 dB, targets 39.0 dB (median); x4 reproduces the standard benchmark (blurry-vs-sharp 26.74 vs 26.41 dB) |
| Translation registration (replaces homography) | [O] | shift median 1.5 px, max 6.6 px native (homography: 6.8 / 114 px, 12/37 > 10 px) |
| DP blur maps + 4 blur bins | [O] | Spearman(\|disparity\|, sharpness gap) median 0.76; focal-plane fraction median 0.129 |
| GT sharpness (paired focal-plane MTF) | [O] | f/22 target as sharp as in-focus f/4 below 0.3 c/px → plain PSNR is primary, PSF matching only sensitivity |
| Grid-shift seam metric | [O] | old `seam_ratio` floor 1.19 on untiled images → replaced |
| Paired bootstrap CIs (`summarize.py --ci --ref`) | [O] | — |
| Resolution-gap table (4 models, 37 pairs, v1 renderings) | [S] | 1/4 → native costs 1.4–1.7 dB PSNR, +0.11–0.16 LPIPS |
| Local study on native set v2 (DRBNet, SwinIR, composite) | [O] | §7.2 |
| New metrics: `percbins`, `noharm`, `apsnr`, `msres` (+ tests) | [O] code | definitions §3; first numbers §7.4 |
| Eval protocol v3 matrix (`code/experiments/dpdd_eval_v3.yaml`, tag `dpdd3`) | [O] code | ready for [S] |

### 🔶 Priority
| Item | Where | Status |
|---|---|---|
| Native test set to **76 pairs** (outdoor raws) and val/train native sets | [S] | `plan/secure_runbook.md` steps 3–4 |
| **Tune every free parameter on val, never on test** (composite/multiscale/guided thresholds, exemplar settings) | [S] | `fuse_baselines.py --tune`; §5.4 |
| Headline perceptual metric = **per-bin LPIPS/DISTS** (`percbins`) | [S] runs | code ✅ |
| **No-harm** (in-focus preservation) column in the main table | [S] runs | code ✅ |
| **Tile-aligned PSNR** (`apsnr`) as fidelity sensitivity | [S] runs | code ✅ |
| **Resolution sweep** (`msres`) | [S] runs | code ✅; presentation §6 |
| Runtime / memory at 6720×4480, same GPU for all methods | [S] | already logged by `run_model.py`; fusion/ref-SR scripts write `meta.json` |
| Full v3 evaluation of all baselines (`dpdd_eval_v3.yaml` + `run_baselines_bd.sh`) | [S] | ready |

### ⏳ Todo
| Item | When |
|---|---|
| Cross-tile semantic / texture **consistency metric** (design in §3.12) | after first method results |
| **Human study** (2AFC) | once main baselines + method v0 exist |
| **Hallucination check** on text/structure crops (OCR / edge fidelity) — supplementary + discussion, not main table | with the qualitative figures |
| **Second camera / dataset** (internal 50 MP smartphone set without GT; own aperture-pair captures; check RealBokeh full-res availability) | after phase 1 of experiments |
| No-reference metrics (MUSIQ / CLIPIQA / MANIQA) on the second-camera set | with the above |

### 📝 Notes (ground-truth limits, keep in mind when writing)
- f/22 target is **softer than f/4 near Nyquist** (MTF ratio 0.64 at 0.4 c/px): genuine native detail above
  ~0.3 c/px is slightly penalized by PSNR. Optional sensitivity: PSNR after a 0.3 c/px low-pass (not implemented).
- f/22 needs a longer exposure → **noise/grain differs** between input and target; affects LPIPS in flat regions.
- **LPIPS on whole native images ranks the blurry input best** (dominated by in-focus grain): never use it alone.
- Residual misalignment after translation registration is ~1–2 native px (→ `apsnr`).
- The native set is **our rendering** of the raws (calibrated to the official look, 39 dB at 1/4): absolute
  numbers are comparable to published DPDD numbers only at ×4 and only up to that calibration error.

---

## 1. The problem the protocol must handle

The paper is about deblurring at **native sensor resolution** (6720×4480 = 30 MP for DPDD's Canon 5D IV).
Every public defocus benchmark is evaluated at 1680×1120 (= ×4 downscaled), where the same captures have 4× smaller
circles of confusion, 4× smaller misalignments and no visible diffraction. At native resolution:
1. Methods behave differently (native patch-wise deblurring ≈ identity; low-res deblur + upsampling loses texture;
   generative SR hallucinates) — the evaluation must separate **fidelity**, **native texture**, **damage to
   already-sharp regions** and **hallucination**.
2. The ground truth is a second capture (f/22) with its own limits (diffraction near Nyquist, different noise,
   1–2 px residual misalignment).
3. Defocus varies across the image; averages over the image are dominated by in-focus or flat regions →
   **stratify by blur level**.
4. Images are 30 MP: perceptual networks need tiling; everything must be efficient.

## 2. Test data

### 2.1 DPDD at native resolution (main paired set)
- **Source**: DPDD (Abuolaim & Brown, ECCV 2020) — 500 scenes, f/4 (blurry) vs f/22 (sharp), tripod, Canon EOS
  5D Mark IV. Splits 350 / 74 / 76 (train / val / test). The processed release (`dd_dp_dataset_canon.zip`, 15.8 GB,
  = Hugging Face `JacobLinCool/DPDD`) is **1680×1120 only**; full resolution exists only as raw CR2 (indoor 29.5 GB,
  outdoor 30.4 GB). So the native set is **developed from raw by us**.
- **Pairing** (`code/scripts/pair_official.py`): official target files carry their own capture number (1–2 above
  the source, e.g. source `1P0A0917` ↔ target `1P0A0916`); pairs are formed by sorted order within the split and
  checked (capture gap ≤ 3). Reproduces the test pairing used so far exactly.
- **Development + calibration** (`code/scripts/build_native_set.py`), per capture:
  1. libraw: camera white balance, linear 16-bit, crop (12, 12, 4480, 6720) of libraw's 4502×6744 output
     (verified: matches the official framing, median sub-pixel residual 0.07 native px).
  2. Colour/tone: 19-feature polynomial fitted on the ×4 area-downscaled linear image vs the official PNG of the
     same capture, applied at native resolution; unsharp mask (σ, amount) fitted per capture (the converter's
     sharpening; fitted for 28/37 targets, 12/37 inputs); smooth local-tone residual field (σ = 16 px at 1680).
  3. Targets registered to inputs (translation, §2.3); validity masks.
  4. ×2 / ×4 by area downscaling (grid-consistent with ×1).
  - Agreement with official @1/4 on held-out pixels: inputs **38.8 dB**, targets **39.0 dB** (median; worst 33.9 /
    30.7). Task check: blurry-vs-sharp PSNR at 1680×1120 is 26.74 dB on our ×4 vs 26.41 dB on the official files
    (paired +0.33 dB, sd 0.59).
- **Coverage**: ✅ 37 indoor test pairs (raws available outside and in [S]). 🔶 76 with the outdoor raws [S].
  Report indoor-only and all-76 tables separately until then.
- **Val split** 🔶: native val set (74 pairs) [S] — used **only** for tuning baseline / method parameters.

### 2.2 Resolution levels
`x1` = native 6720×4480, `x2` = 3360×2240, `x4` = 1680×1120 (= standard benchmark resolution). All from the same
native rendering by area downscaling (inputs, targets, masks by min-pooling). A pipeline is named by the resolution
it *runs* at; it is always **scored at native resolution** unless stated (`msres` adds downscaled scores).

### 2.3 Registration
- Targets are aligned **once** to the blurry input (method-independent), translation-only ECC on luminance
  (`code/uhdd/registration.py`, default `motion="translation"`), masks mark pixels valid in both.
- Why translation: homography/affine **overfit the f/4–f/22 blur difference** (on the 37 indoor raws: homography
  median 6.8 px, max 114 px, 12/37 > 10 px vs translation median 1.5, max 6.6, with identical ECC).
- Residual: ~1–2 native px locally → `apsnr` (§3.6) as sensitivity.
- Border crop: 64 native px for every method (divided by the scale when scoring at ×2/×4).

### 2.4 Blur map and blur bins
- **DP disparity** (`code/scripts/dp_maps.py`, `code/uhdd/dualpixel.py`): DPDD provides the left/right dual-pixel
  views of the f/4 capture (1680×1120). Horizontal disparity ∝ signed defocus; block matching on band-passed luma
  with sub-pixel parabola fit + confidence. Outputs `<stem>_disp.png`, `_conf.png`, `_focus.png`.
- **Blur level** (`uhdd.dualpixel.load_blur_map`): confidence-weighted |disparity| box-smoothed over 31 px at 1680,
  bilinearly resized to the scored image. Unit: DP px at 1680×1120.
- **Bins**: b0 [0, 0.4) focal plane · b1 [0.4, 1.5) · b2 [1.5, 3.5) · b3 [3.5, ∞) (≈ quartiles on the test set).
- Validation: per-image Spearman(|disparity|, log sharpness gap target/input) median **0.76**; focal-plane cells
  have a log-sharpness gap of 0.70 vs 1.99 elsewhere.
- Per-bin PSNR is content-confounded (strongly defocused regions are often smooth: blurry input b0..b3 =
  25.7 / 23.6 / 23.3 / 26.0 dB) → report per-bin **gains over the blurry input** (`summarize.py --ref`).
- The DP map is an **oracle** for the method (DP exists only on DP sensors); the method proper uses a predicted map
  (`plan/method_plan.md` §2.2). For evaluation the DP map is fine: it only defines regions.

### 2.7 Tiling standard (per method, from its paper)
- **Rule** (`uhdd/tiling.py:default_tiling`, used by `run_model.py` and `run_matrix.py` whenever no tile is given):
  a method never processes more pixels per pass than in its own paper's evaluation.
  - `paper_input: [H, W]` (method evaluated on whole images of that size): whole image if the input fits (either
    orientation), else **square tiles of the short side** (rounded down to the model's `multiple`).
  - `paper_tile: T` (the authors' code tiles): tiles of T px (LR px for SR), whole image if the input fits.
  - neither (bicubic, identity): whole image.
- **Overlap = tile / 8** for every method (`--overlap-ratio 0.125`; P1 used 512/64), linear border-aware blending.
- Tile batch: registry `tile_batch`, else 2 for tiles ≥ 1024 px, 4 otherwise.
- Resolved values are in the step cache key (`<model>@t1120o140`) and in each output's `meta.json` (`tiling`).

| method | paper / code inference | default at ×4 (1680×1120) | default at ×2 / ×1 | source |
|---|---|---|---|---|
| Restormer, Restormer+TLC | whole DPDD image 1680×1120 | whole | 1120 / overlap 140 | `test_single_image_defocus_deblur.py` |
| LaKDNet-L/S | whole DPDD image | whole | 1120 / 140 | `run.py` |
| DRBNet | whole DPDD image (cropped to /16) | whole | 1120 / 140 | `run.py` |
| IFAN | whole DPDD image | whole | 1120 / 140 | `eval.py` |
| Bokehlicious (deblur) | whole RealBokeh_3MP image 2000×1500 [V: RealDefocus test size] | whole | 1500 / 188 | `evaluate.py` |
| SwinIR (classical, real) ×4/×2 | `--tile 400` recommended for large inputs | 400 LR / 50 | — | README, `main_test_swinir.py` |
| HAT-L, Real-HAT ×4 | tile mode `tile_size 512` | 512 LR / 64 | — | `options/test/HAT_SRx4_ImageNet-LR.yml` |
| OSEDiff ×4 | `process_size 512` HR | 128 LR / 16 | — | `test_osediff.py` |
| DATSR (script) | CUFED5 whole images (~500×330 HR) | 128 LR / 16 | — | `refsr_baseline.py --tile 128 --overlap 16` |
| bicubic, identity | — | whole | whole | — |

- Supplementary ablation: the P1 512-px tiles for Restormer and DRBNet (`g1_tile_ablation`); P1 showed 512 vs 1024
  changes fidelity by < 0.06 dB.

### 2.5 Ground truth limits (📝)
- Paired focal-plane MTF (`code/scripts/paired_edge_mtf.py`, 336 focal-plane edges on the same 37 raws):
  MTF(f/22)/MTF(f/4) = 1.05 / 1.24 / 0.99 / 0.64 at 0.1 / 0.2 / 0.3 / 0.4 c/px; Gaussian fit σ ≈ 0 (95 % CI 0–0.31 px).
  Airy theory would predict 0.75 / 0.50 / 0.25 / 0.05. → The target is not blurrier than in-focus f/4 except near
  Nyquist; **plain PSNR/SSIM are primary**; PSF-matched `pm` is a sensitivity column only.
- Noise differs (longer f/22 exposure); colour rendering is ours (§2.1).

### 2.6 Other sets (⏳)
- **Second camera, no GT**: the internal 50 MP smartphone set [S] (describe generically in reports) and/or our own
  aperture-pair captures (15–20 tripod scenes at ≥ 45 MP: f/2.8–4 vs f/11–16, gives GT). NR metrics + human study.
- **RealBokeh / RealDefocus** (Bokehlicious): captured at 6000×4000, released at 3 MP — check whether full-res is
  obtainable [V]; RealDOF (IFAN, ~2320×1536): mid-resolution generalization only.
- Synthetic defocus at native resolution (rendered from sharp HR photos + depth): exact GT and per-CoC analysis;
  optional (also training data, `plan/method_plan.md` §8).

## 3. Metrics

All metrics are computed by `code/scripts/evaluate.py` (one process per GPU) through the registry
`code/uhdd/metrics/__init__.py:compute`; per-image rows go to `<pred>/metrics_<tag>.csv`, means to `.json`.
Images are scored in [0, 1] float, masked by the validity mask, after the 64-px border crop.

| # | Name (`--metrics`) | Columns | Dir. | Role | Status |
|---|---|---|---|---|---|
| 3.1 | `psnr`, `ssim` | psnr, ssim | ↑ | **headline fidelity** | ✅ |
| 3.2 | `blurbins` | psnr_b0..b3, frac_b* | ↑ | **headline** (as gains over input) | ✅ |
| 3.3 | `lpips`, `dists` | lpips, dists | ↓ | whole-image perceptual (caveat) | ✅ |
| 3.4 | `percbins` | lpips_b0..b3, dists_b0..b3, ntile_b* | ↓ | **headline perceptual** | ✅ code, 🔶 run |
| 3.5 | `noharm` | keep_psnr_b0, dpsnr_b0 | ↑ | **main table** (damage to sharp regions) | ✅ code, 🔶 run |
| 3.6 | `apsnr` | apsnr, apsnr_shift | ↑ | fidelity sensitivity (misalignment) | ✅ code, 🔶 run |
| 3.7 | `msres` | psnr_s2/s4, lpips_s2/s4, dists_s2/s4 | ↑/↓ | resolution sweep (figure) | ✅ code, 🔶 run |
| 3.8 | `hb` | hb_nmse_db | ↓ | native-detail error | ✅ |
| 3.9 | `gridshift` | gs_psnr, gs_mad, gs_seam_step, gs_seam_ratio | ↑/↓ | seams / tiling sensitivity | ✅ |
| 3.10 | `stats`, `drift` | sharp/noise/slope consistency, lf_drift | ↓ | supplementary | ✅ |
| 3.11 | `musiq`, `maniqa`, `clipiqa` | same | ↑ | NR, second camera only | ✅ code, ⏳ |
| 3.12 | semantic consistency | — | | cross-tile texture consistency | ⏳ design |
| 3.13 | hallucination (text/structure) | — | | supplementary + discussion | ⏳ |
| 3.14 | runtime / memory | time_s, pipeline_time_s, peak_mem_gb | ↓ | main table | ✅ |
| 3.15 | `pm` | pm_psnr, pm_ssim | ↑ | sensitivity only (§2.5) | ✅ |
| 3.16 | human study | — | | ⏳ | ⏳ |

### 3.1 PSNR / SSIM
- PSNR = 10·log10(1/MSE) over valid pixels and channels, float64 (`uhdd/metrics/fidelity.py`; on Apple MPS the
  float64 part runs on CPU).
- SSIM reproduces `skimage.metrics.structural_similarity(data_range=1, channel_axis=-1)` (7×7 uniform window,
  sample covariance, border-cropped map), float64 per channel — the DPDD evaluation convention. With a mask, a
  window counts only if all its pixels are valid.

### 3.2 Per-bin PSNR (`blurbins`)
PSNR restricted to pixels of each blur bin (§2.4) ∩ mask; `frac_b{k}` = fraction of pixels. Because bins differ in
content, the table reports **ΔPSNR per bin vs the blurry input** (paired, with CIs): `summarize.py <rows> <input
row> --ref "input (blurry)" --cols psnr_b0,psnr_b1,psnr_b2,psnr_b3`.

### 3.3 Whole-image LPIPS / DISTS
pyiqa LPIPS (AlexNet) and DISTS on non-overlapping 1024-px tiles, area-weighted mean (`uhdd/metrics/iqa.py`).
**Caveat**: at native resolution LPIPS ranks the blurry input best (local study: input 0.308 vs every deblurring
pipeline ≥ 0.32 on 37 images) because it is dominated by grain/texture in the large in-focus and mildly defocused
area. Kept in the supplementary; the headline perceptual metric is 3.4.

### 3.4 Per-bin LPIPS / DISTS (`percbins`) — headline perceptual metric
- Concept: measure perceptual quality **where deblurring matters** (b2, b3) separately from where the input is
  already the answer (b0).
- Definition (`uhdd/metrics/binned.py:perceptual_bins`): non-overlapping 512-px tiles (`--bin-tile`); a tile is
  used if ≥ 90 % of its pixels are valid; it is assigned to the bin of its **median** blur level; LPIPS / DISTS are
  averaged over the tiles of each bin. `ntile_b{k}` = number of tiles (≈ 100 tiles per 30 MP image).
- Main table: `dists_b3`, `lpips_b3` (and b2); b0 goes to the no-harm block.

### 3.5 No-harm (`noharm`)
- Concept: in focal-plane regions the blurry input already *is* sharp; a method should not change it. Generative SR
  replaced real in-focus detail (local study: SwinIR-real −0.85 dB in b0).
- `keep_psnr_b0` = PSNR(pred, **input**) on b0 pixels (GT-free; higher = less change).
- `dpsnr_b0` = PSNR_b0(pred, GT) − PSNR_b0(input, GT) per image (< 0 = the method damaged in-focus detail).
- Needs `--inputs` (blurry inputs at the scored resolution; `run_matrix.py` adds it automatically).

### 3.6 Tile-aligned PSNR (`apsnr`)
- Concept: separate *content* errors from the ~1–2 px residual misalignment of the registered target.
- Definition (`binned.aligned_psnr`): per valid 512-px tile, the integer shift within ±2 px (`--align-radius`) that
  minimizes the squared error is applied; `apsnr` = PSNR over all tiles' best errors; `apsnr_shift` = mean |dy|+|dx|.
  Same shift search for every method (a method cannot gain from it unless its output is shifted).
- Report as a sensitivity column next to PSNR; if rankings agree, misalignment is not driving conclusions.

### 3.7 Resolution sweep (`msres`)
- Concept: where do a method's gains live? Low-res deblur + bicubic wins at ×4 but loses native detail;
  generative SR may look better at ×1 but be worse at ×4; our method should win at every scale.
- Definition (`binned.resolution_sweep`): prediction, target (area) and mask (all source pixels valid) downscaled
  by f ∈ {2, 4} (`--eval-scales`); `psnr_s{f}`, `lpips_s{f}`, `dists_s{f}`.
- `psnr_s4` connects every native method to the standard 1680×1120 benchmark.
- Presentation (proposal, §6): a small multiple **figure** — x-axis evaluation resolution (×4, ×2, ×1), y-axis PSNR
  (left panel) and DISTS (right panel), one line per method; the crossing of "anchor + bicubic" and "generative SR"
  lines and ours above both is the visual argument. In the main table only `psnr_s4` (comparability).

### 3.8 High-band error (`hb`)
`hb_nmse_db` = 10·log10(‖HB(pred) − HB(gt)‖² / ‖HB(gt)‖²) with HB(x) = x − up4(avgpool4(x)) (bicubic up): error in
exactly the band a 1/4-res method cannot carry. 0 dB ≈ "no native detail" (bicubic), < 0 better, > 0 = wrong
detail (hallucinated or misaligned).

### 3.9 Seams (`gridshift`)
The same tiled pipeline is run twice, the second time with the tile grid shifted by half a stride
(`run_model.py --grid-offset`; `run_matrix.py` creates the twin automatically). `gs_psnr` = PSNR between the two
outputs (content-free tiling sensitivity), `gs_seam_ratio` = difference energy on tile boundaries / elsewhere.
Replaced the old `seam_ratio` (content-driven floor 1.19 on untiled images).

### 3.10 Statistical consistency (`stats`, `drift`) — supplementary
Per-cell (256 px) maps of sharpness (Laplacian variance), noise level and spectral slope; log-ratio error vs GT
(`sharp_lr_*`, `noise_lr_*`, `slope_err_*`). `lf_drift`: low-frequency colour drift vs GT. Local finding: per-tile
texture statistics of SwinIR are not more tile-dependent than untiled bicubic (η² 0.39 vs 0.40).

### 3.11 No-reference metrics
MUSIQ, MANIQA, CLIPIQA on a fixed grid of 16 crops of 512 px (`iqa.no_reference`). Known bias toward
over-sharpening; only for the second-camera set (no GT), never as the main evidence.

### 3.12 Cross-tile semantic / texture consistency (⏳ design)
Claim to test: the exemplar memory gives the **same material the same texture across tiles**. Proposal:
pairs of 256-px patches in *different tiles* that depict the same material in the GT (DINOv2 cosine > τ on GT
patches, both in b2/b3); for each pair compare a texture distance d (Gram-matrix / DISTS-texture term) between the
two predicted patches with the same distance between the two GT patches: relative consistency error
RCE = mean |d(pred_i, pred_j) − d(gt_i, gt_j)|. NR variant for the second camera: Intra-LPIPS over matched pairs.

### 3.13 Hallucination on text and structure (⏳, supplementary)
On crops with text / labels / dials (selected once, list frozen): OCR character accuracy (e.g. PaddleOCR / Tesseract)
of pred vs GT, and an edge-fidelity score (F1 of Canny edges vs GT within 2 px). Discussion + supplementary.

### 3.14 Runtime and memory
`run_model.py` logs per-image `time_s` and `peak_mem_gb` (meta.json); `run_matrix.py` sums steps into
`pipeline_time_s`; fusion / ref-SR scripts write `time_s` in their meta.json. Report GPU type; all methods on the
same GPU type at 6720×4480.

### 3.15 PSF-matched fidelity (`pm`) — sensitivity only
Prediction convolved with the f/22 Airy PSF (+ pixel aperture) before PSNR/SSIM. Not primary (§2.5).

### 3.16 Human study (⏳)
2AFC on 512-px native crops (≥ 40 pairs, ≥ 15 raters), ours vs {DP composite, anchor + SwinIR-real/OSEDiff, best
native patch-wise}; crops sampled stratified by blur bin (half b2/b3), position randomized, full-res display at
100 %. Report preference rate with binomial CIs.

## 4. Statistics
- Every claim is a **paired per-image difference** to a reference row with a 95 % bootstrap CI (2000 resamples)
  and the fraction of images improved: `summarize.py <jsons> --ci --ref "<label>"`.
- n = 37 (indoor) → 76. No claim from the 8-image subsets except as "local preview".

## 5. How to run

### 5.0 Results layout and the manual launcher (2026-10-05)
All new results go to one root, organised by **stage → input source → chain** (`code/uhdd/layout.py`):
```
$UHDD_RESULTS/dpdd/
  inputs/<src>/{inputs,targets,masks}   symlinks into $UHDD_DATA; inputs/manifest.json = verification
  inputs/dp_maps
  deblur/<src>/<model>@<tiling>/                    PNGs, meta.json, launch.json, metrics_dpdd4.{csv,json}
  upsample/<src>/<anchor>@whole/<sr>@<tiling>/      <src> = source of the anchor's x4 input
  fusion/<src>/<anchor>@whole/<method>/             (fusion, DATSR: not ported yet)
  tables/<tag>/   scratch/
```
- Sources: `ours_x1` (native), `ours_x2`, `ours_x4` (our raw-built rendering), `official_x4` (the original DPDD test
  images, no masks). Tiling tag = `whole` or `t<tile>o<overlap>` (paper setup, §2.7). Metrics next to the images.
- `code/scripts/setup_inputs.py` links the sources and verifies them before anything runs: 76 scenes, identical names
  across inputs / targets / masks / DP maps **and across sources**, sizes (×4 = native / 4, official = ours ×4), and
  content (official vs ours ×4 per-scene PSNR ≈ 39 dB; nearest-thumbnail check against swapped names). Every problem
  is attributed to a source; the launcher refuses unverified sources.
- `code/scripts/launch.py` (`deblur`, `upsample --src ours_x4|official_x4`, `list`, `summary`, `compare`) and the
  wrapper `code/experiments/launch.sh` (setup, `fresh` = the standard set, `study`). Scoring: native outputs against
  `ours_x1` (frozen headline + diagnostics, 64 px border); ×4 outputs against their own source (border 16;
  `official_x4`: headline metrics only). An upsampled official anchor is scored against our native targets and
  labelled diagnostic.
- Legacy roots `dpdd_v2`, `dpdd_official_x4`, `dpdd_p1` (handoffs 1–3, `run_matrix.py` layout) are kept read-only;
  `launch.py compare` reads them to check fresh results. `run_matrix.py` / `run_baselines_bd.sh` still use the
  legacy layout (sections 5.3, 5.4).

### 5.1 Environment
```bash
export UHDD_DATA=...  UHDD_RESULTS=...  UHDD_REPOS=...  UHDD_WEIGHTS=...    # never hard-code paths
pip install -r code/requirements.txt
pytest -q code/tests                                                       # 57 tests, CPU
```

### 5.2 Data (once) [S]
```bash
python code/scripts/pair_official.py --root <dd_dp_dataset_png> --split test --out $UHDD_DATA/dpdd_1680/test
python code/scripts/dp_maps.py --left $UHDD_DATA/dpdd_1680/test/inputs_l --right $UHDD_DATA/dpdd_1680/test/inputs_r \
    --out $UHDD_DATA/dpdd_native/dp_maps
python code/scripts/build_native_set.py --raw <cr2 dir or zip> --official-inputs $UHDD_DATA/dpdd_1680/test/inputs \
    --official-targets $UHDD_DATA/dpdd_1680/test/targets --pairs $UHDD_DATA/dpdd_1680/test/pairs.csv \
    --out $UHDD_DATA/dpdd_native_v2 --procs 16
```
(same for `val` → `dpdd_native_val`, used only for tuning.)

### 5.3 Registry pipelines (inference + evaluation + tables) [S]
```bash
python code/scripts/run_matrix.py code/experiments/dpdd_eval_v3.yaml --dry-run | less   # check the plan
python code/scripts/run_matrix.py code/experiments/dpdd_eval_v3.yaml --gpus all
```
Each pipeline's steps are cached under `$UHDD_RESULTS/dpdd_v2/steps/<key>` (keyed by pipeline, **not** by dataset:
use a new `results` dir when the data changes). Evaluations: `<final step>/metrics_dpdd3.{csv,json}`; tables per
group under `$UHDD_RESULTS/dpdd_v2/tables/`.

### 5.4 Script baselines (fusion, reference-based SR) [S]
```bash
# tune fusion parameters on VAL (anchors for val from the same models)
python code/scripts/fuse_baselines.py --method multiscale --inputs $VAL/inputs --x4 <val x4+bicubic> --x2 <val x2+bicubic> \
    --dp-maps <val dp_maps> --tune --targets $VAL/x1/targets --masks $VAL/x1/masks --params $UHDD_RESULTS/dpdd_v2/fusion_params/multiscale.json
# run + evaluate all group B and D baselines on test with the tuned parameters
code/experiments/run_baselines_bd.sh drbnet_single
code/experiments/run_baselines_bd.sh restormer_dpdd
```

### 5.5 Tables
```bash
# main comparison, paired to the DP composite
python code/scripts/summarize.py <jsons...> --ci --ref "drbnet_single fuse:composite" \
    --cols psnr,apsnr,ssim,dists_b3,lpips_b3,dpsnr_b0,psnr_s4,pipeline_time_s
# per-bin gains over the input
python code/scripts/summarize.py <jsons...> <input json> --ref "input (blurry)" --cols psnr_b0,psnr_b1,psnr_b2,psnr_b3
```

### 5.6 Local (M4, outside) checks
Memory is the constraint (16 GB): run one 30 MP job at a time, CPU for float64 metrics (`--gpus cpu`), and
8-image subsets (`dataset/results/local/subset8.txt`). Analysis scripts: `code/analysis/` (`NATIVE`, `RESULTS`,
`DPMAPS` env vars).

## 6. Main table and figures (target layout)

**Main table** (DPDD native test, 76 scenes; mean, paired CIs vs the DP composite in the supplementary):

| block | columns |
|---|---|
| fidelity | PSNR ↑, SSIM ↑, aligned PSNR ↑ |
| where deblurring matters | DISTS_b3 ↓, LPIPS_b3 ↓, ΔPSNR_b3 vs input ↑ |
| no harm | ΔPSNR_b0 vs input ↑ |
| comparability | PSNR @×4 ↑ |
| cost | s / 30 MP image, peak GB |

Rows: blurry input · native patch-wise (best of G1) · retrained-at-native (A) · anchor + {bicubic, HAT-L, SwinIR-real /
Real-HAT, OSEDiff} · training-free {DP composite, multiscale, guided, exemplar} · DATSR (ref-SR) · **ours A** · **ours B**.

**Figures**: (1) resolution sweep (§3.7); (2) per-bin gain bars (b0–b3) for 4–5 key rows; (3) qualitative crops
per bin incl. a text crop; (4) relevance map of the exemplar memory.
**Supplementary**: whole-image LPIPS/DISTS, HB, gridshift, stats, pm, per-bin PSNR table, hallucination crops,
indoor vs outdoor split, all anchors.

## 7. Results so far

### 7.1 Resolution gap [S, 2026-10-01, 37 indoor, v1 renderings — rerun on v2: runbook step 7]
Same model run at ×4 / ×2 / ×1 (512 tiles), each scored at **its own resolution** against the target at that
resolution (excerpt of the secure P1 report):

| model | PSNR ×4 / ×2 / ×1 | LPIPS ×4 / ×2 / ×1 |
|---|---|---|
| Restormer | 26.62 / 26.13 / 25.19 | 0.141 / 0.225 / 0.253 |
| LaKDNet-L | 26.51 / 25.61 / 24.90 | 0.132 / 0.251 / 0.291 |
| DRBNet | 26.70 / 25.90 / 24.96 | 0.121 / 0.206 / 0.272 |
| Bokehlicious | 26.20 / 25.41 / 24.64 | 0.112 / 0.179 / 0.251 |

Tile 512 vs 1024 changes fidelity by < 0.06 dB for every
model → tile context is not the bottleneck; out-of-distribution blur size is.

### 7.2 Local study on native set v2 [O, M4, DRBNet official weights]
37 indoor pairs, scored at native, crop 64, masks:

| pipeline | PSNR ↑ | SSIM ↑ | LPIPS ↓ | HB-NMSE dB ↓ |
|---|---|---|---|---|
| blurry input | 25.28 | 0.628 | **0.308** | +1.61 |
| DRBNet native, 1024 tiles | 25.32 | 0.615 | 0.320 | +1.97 |
| DRBNet @×4 + bicubic | **27.03** | **0.702** | 0.551 | **−0.04** |
| DRBNet @×2 (1024 tiles) + bicubic | 26.17 | 0.675 | 0.394 | +0.46 |
| DP composite (input in focus, ×4+bicubic elsewhere) | 26.89 | 0.693 | 0.476 | +0.28 |

Same 8-image subset (`dataset/results/local/subset8.txt`), adds real-world SwinIR (512-px LR tiles):

| pipeline | PSNR ↑ | SSIM ↑ | LPIPS ↓ | HB-NMSE dB ↓ |
|---|---|---|---|---|
| blurry input | 26.44 | 0.658 | **0.278** | +1.33 |
| DRBNet native, 1024 tiles | 26.29 | 0.651 | 0.280 | +1.59 |
| DRBNet @×4 + bicubic | **27.82** | **0.721** | 0.524 | **−0.07** |
| DRBNet @×4 + SwinIR-real | 26.94 | 0.700 | 0.439 | +1.01 |
| DRBNet @×2 (1024 tiles) + bicubic | 27.14 | 0.707 | 0.357 | +0.19 |
| DRBNet @×2 (1024 tiles) + SwinIR-real ×2 | 26.60 | 0.701 | 0.425 | +1.13 |
| DP composite: input / ×4+bicubic | **27.83** | 0.716 | 0.437 | +0.19 |
| DP composite: input / ×4+SwinIR-real | 27.32 | 0.706 | 0.385 | +0.75 |

Paired on these 8: ×4 + SwinIR-real vs ×4 + bicubic: PSNR −0.88 dB (8/8 images), LPIPS −0.085,
in-focus bin −0.85 dB; ×2 + SwinIR-real is worse than ×2 + bicubic on both PSNR (−0.54) and LPIPS (+0.068).
Findings: native patch-wise ≈ identity (+0.04 dB [−0.16, +0.22]); generative SR hallucinates (text → scribbles,
fabric → painterly strokes) and damages in-focus detail; no tile seams; the region "deblurred **and** native
texture" is empty → the method's target.

### 7.3 Exemplar pilot [O, interim 5 of 37 images, stopped]
Non-learned in-focus texture transfer (`code/analysis/exemplar_pilot.py`, `uhdd/exemplar.py`), paired Δ vs DP
composite on the first 4 images: LPIPS b2 −0.104, b3 −0.134 (4/4 images), DISTS b3 −0.037 (4/4), PSNR −0.09 dB;
random in-focus texture: LPIPS b2 −0.054, PSNR −0.20 dB. Matching beats random; median match similarity 0.69.
Full 37 [O, pending] or as baseline B3 in [S].

### 7.4 Training-free baselines, local preview [O, 8 images]
Full table in [`baselines.md`](baselines.md) §B-results. DP composite 27.83 dB (= ×4 + bicubic 27.82) with no in-focus
damage; multiscale / exemplar −0.14 dB, guided −0.37, detail −0.30; ×4 + SwinIR-real −0.88 dB and ΔPSNR_b0 −0.76.
Aligned PSNR is 0.3–0.6 dB higher for every row with the same ranking.

### 7.5 VAE ceiling (variant B backbone) [O, partial]
SDXL VAE on 1024-px native target crops (29 crops): in-focus 29.5 dB / LPIPS 0.116, defocused 32.8 dB / 0.082 →
a latent model cannot reproduce in-focus native texture; variant B needs the pixel-space copy path. Full comparison
(SDXL, FLUX, SD3.5, DC-AE): **deferred** (not in handoff 2).

### 7.6 Handoff 2 — full matrix on 76 native pairs [S, 2026-10-04]
Full protocol-v3 tables (all 76 and indoor-37, 95 % CIs): `handoff/from_secure/2026-10-04_h2.md`.
- ×4 deblur + bicubic beats native-resolution inference of every deblurrer by 1.1–1.8 dB (indoor-37: 27.03 vs
  25.2–25.9; input 25.28); native deblurrers stay within ±0.2 dB of the input (Bokehlicious +0.6). Tile size and
  TLC change nothing (≤ 0.06 dB). Indoor numbers reproduce the local M4 values to 0.01 dB (GPU/CPU agreement).
- Generative upsamplers damage in-focus regions: ΔPSNR_b0 vs input OSEDiff −1.1/−1.6 dB, SwinIR-real up to −0.8,
  Real-HAT up to −0.5 (all 76). **Note: the report calls b0 "the blurriest bin" — b0 is the focal plane (in focus).**
- Exemplar fusion (tuned on val): DISTS_b3 0.299 → 0.262 (DRBNet), 0.338 → 0.274 (Restormer) vs composite, −0.12…−0.18 dB.
- Deblur-only at ×4 (indoor-37, addendum): input 26.71, Restormer 28.36, DRBNet 28.48 dB (LPIPS 0.164, DISTS 0.130):
  the low-res models work; the native failure is about resolution.
- Not run: DATSR (>15 min/image on one GPU; opt-in), VAE ceiling (deferred).

**Metric issues found in review (pending decision, §8):**
- Per-bin DISTS/LPIPS at native still reward grain/noise: in b3 the *blurry input* scores DISTS 0.234 vs ×4 + bicubic
  0.302 (indoor), and "detail" fusion wins perceptual columns by re-adding input grain. At ×4-downscaled scoring the
  order is sensible (input 0.191 vs 0.137).
- In focus, ×4 + bicubic is +0.59 dB *above* the input (DRBNet, indoor): the f/22 target is smoother / less noisy, so
  ΔPSNR_b0 rewards smoothing; report `keep_psnr_b0` alongside.
- Peak-memory aggregation for chained pipelines is wrong (means outside their CIs); timings came from shared nodes.

**First visual pass [S, 3 of 8 captures, downscaled montages]:** native Restormer ≈ input; ×4 + bicubic visibly
recovers edges (lock hinges, poster lettering); HAT-L ≈ bicubic; SwinIR-real / Real-HAT add contrast and halos
(SwinIR-real a mild colour shift); **OSEDiff rewrites caption text into different glyphs, shifts faces and texture,
and adds horizontal banding in a dark scene**; guided ≈ smooth bicubic, detail visibly grainier. No ×4 pipeline
recovers the target's fine grain in the dark scene → part of the PSNR/perceptual gap is PSNR rewarding smoothness.
Limit for the paper: ×4 + bicubic is never sharper than the target's detail level — **the method must show it adds
real detail, not only that it does no harm.** Visual set (8 captures × 12 variants, crops) is held in [S].

## 8. Open items (owner)

### Decisions taken 2026-10-04 (were pending)
Protocol frozen (standard headline metrics); all 76 scenes always; handoff 3 evaluation-only with DATSR (mosaic and
in-focus-only mosaic); slow SR models stay on Restormer/DRBNet; visual review by the secure agent and the user; no perspective experiments in handoff 3; report fixes
(b0 wording, per-image cost aggregation) in handoff 3. See §9 and `plan/secure_runbook.md`.

- [S] `plan/secure_runbook.md` (76 pairs, val/train native sets, anchors, v3 evaluation, baselines B/D; VAE ceiling deferred).
- [S] v3 evaluation of all registry baselines + group B/D scripts.
- [O] consistency metric (3.12) implementation once method outputs exist.
- [O] human-study tooling (crop sampler + 2AFC page).
- [O] hallucination crop list + OCR script (supplementary).
- [You] second camera / captures (after phase 1).

## 9. Decision log
| date | decision | evidence |
|---|---|---|
| 2026-10-02 | registration = translation (was homography) | §2.3 |
| 2026-10-02 | seams by grid shift (`gridshift`), old `seam_ratio` dropped | floor 1.19 on untiled images |
| 2026-10-02 | plain PSNR/SSIM primary; PSF matching sensitivity only | paired focal-plane MTF §2.5 |
| 2026-10-02 | blur stratification from DP views, 4 bins | Spearman 0.76 |
| 2026-10-02 | paired bootstrap CIs for every claim | — |
| 2026-10-02 | native set v2 = per-capture calibrated development | 38.8 / 39.0 dB @1/4 |
| 2026-10-03 | headline perceptual = per-bin LPIPS/DISTS; whole-image LPIPS supplementary | LPIPS ranks the blurry input best |
| 2026-10-03 | add no-harm, aligned PSNR, resolution sweep, runtime to the protocol (v3, tag `dpdd3`) | user decision |
| 2026-10-03 | all free parameters tuned on val only | user decision |
| 2026-10-03 | hallucination/OCR → supplementary + discussion; consistency metric, human study, second camera → todo | user decision |
| 2026-10-03 | per-method default tiling from each paper's inference setup; overlap = tile / 8 for all | user decision; §2.7 |
| 2026-10-04 | **Protocol frozen before any method result exists.** Headline (main table): out-of-the-box PSNR, SSIM, LPIPS, DISTS at native resolution (pyiqa / skimage defaults, whole image), no-reference MUSIQ + CLIPIQA, runtime; plus PSNR/SSIM at ×4 (the standard DPDD benchmark resolution); human study. Per-blur-bin breakdown, no-harm, aligned PSNR, resolution sweep, hallucination crops = diagnostics (analysis section / supplementary), computed identically for all methods, never used to rank. No custom headline metric (band-limited PSNR, downscaled DISTS, denoise-before-scoring dropped). | user decision: avoid looking like metric tuning |
| 2026-10-04 | **All 76 test scenes in all cases** (main tables, analyses, figures); indoor/outdoor splits only as supplementary breakdowns. | user decision |
| 2026-10-04 | **Handoff 3 stays evaluation-only** (no training): goal is to build a clear picture of the problem before method work. | user decision |
| 2026-10-04 | DATSR in handoff 3 (mosaic + in-focus-only mosaic); slow SR models not extended to LaKDNet-L / IFAN; visual review by the secure agent and by the user; no perspective experiments (upper bounds, anchor-vs-final, exemplar controls) in handoff 3; eval tag `dpdd4` adds MUSIQ, CLIPIQA, SSIM at ×4. | user decision |
| 2026-10-05 | **×4 column: report both** our raw-built renderings and the original 1680×1120 images; which one leads is deferred until the story is set. | user decision |
| 2026-10-05 | Anchors for the upsampler study: **DRBNet and Bokehlicious** (user's visual assessment). New upsamplers S3Diff and VOSR 2.0 run manually via `launch_sr.py` (now `launch.py`); secure aggregates later. | user decision |
| 2026-10-05 | **New results root `dpdd/`** organised by stage → source → chain, metrics next to the images (§5.0). **Fresh start**: all runs redone through the launcher and re-checked against the legacy results (`compare`); legacy roots kept read-only. Official vs ours scene names are verified (names + content) before any run. | user decision |
| 2026-10-05 | **S3Diff and VOSR integrated** as registry models in the one main env (adapters, our tiling, overlap tile/8) instead of their own scripts and envs: same tiling treatment for every upsampler. | user decision |
| 2026-10-06 | **Anchor rendering: undecided, track both.** Run every anchor × upsampler on `ours_x4` and `official_x4` (the official-anchor rows score higher at native, e.g. Bokehlicious + bicubic 25.20 vs 24.79 dB; not investigated for now). | user decision |
| 2026-10-06 | **Official vs ours ×4 targets** differ (median 32 dB) mostly by translation (our targets are registered to the input): accepted; revisit only if it costs too much. | user decision |
| 2026-10-06 | **Runtime**: ignore cold-start timings (first images / model loading, shared GPUs); runtime comes from a warm, idle-GPU timing pass. | user decision |
| 2026-10-06 | **OSEDiff paused**: not run in the single environment; its handoff-3 row stays; no separate environment for now. | user decision |
| 2026-10-06 | HAT-L / SwinIR-real / LaKDNet / IFAN anchors in the new layout and the HAT-L anomaly check: not now (handoff-3 rows stand). Handoff 5: wait. | user decision |

### 7.7 Handoff-3 results [S, 2026-10-05, all 76 test scenes, protocol dpdd4]
Complete tables (Table 1 deblur-only at x4, native-resolution main table with CIs, per-bin diagnostics, timing pass on 5 scenes, control run on the original 1680x1120 DPDD pairs, data facts) are in `handoff/from_secure/2026-10-05_h3.md`. Control: the harness reproduces the published numbers on the original images (input, Restormer, DRBNet, IFAN within ~0.06 dB); our raw-built renderings shift per-model PSNR by -0.58..+0.16 dB and reorder the models. DP maps exist for all splits (test 76, val 74, train 350). Open: complete per-shard DATSR stats (candidate counts, tiles without candidate), visual review (user), val proxy (not run).

### 7.8 Handoff-4 results [S, 2026-10-06, all 76 test scenes, protocol dpdd4, new results layout]
Fresh runs through the launcher (deblur on `official_x4` / `ours_x4` / `ours_x1`; DRBNet and Bokehlicious x4 + bicubic / Real-HAT / S3Diff / VOSR2 on `ours_x4` and, as diagnostic, `official_x4`): tables in `handoff/from_secure/2026-10-06_h4.md`. All fresh rows with a legacy counterpart agree to 4 decimals. Open: HAT-L / SwinIR-real / LaKDNet / IFAN anchors in the new layout, the HAT-L in-focus anomaly check, fusion and DATSR in the new layout, OSEDiff in the single environment.
