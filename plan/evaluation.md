# Evaluation plan — main comparisons and metrics

Status: **v2 (2026-10-02)** — updated with the first secure-env results (`handoff/from_secure/2026-10-01_p1-eval.md`) and outside checks on the official 1680×1120 data. Outside env. Secure env: comment/amend via patch (edit this file or answer in the report).
Legend: **[D]** decision needed from user, **[V]** fact to verify, **[P1..P4]** priority.

---

## Status and decisions (2026-10-02)

**Findings so far** (secure P1 partial: 4 models, 37 indoor native scenes; outside: official 1680×1120 test/val):
- **Resolution gap confirmed** (all 4 models): 1/4 → native with 512 tiles costs 1.4–1.7 dB PSNR and 0.11–0.16 LPIPS; high-band error goes from −1.8…−3.2 dB to +1.0…+3.1 dB.
- **Fidelity vs. perception split at native res**: Restormer ×4 + bicubic beats native patch-wise Restormer in PSNR (26.12 vs 25.19) but is far worse in LPIPS (0.449 vs 0.253). Neither existing route is good — this is the paper's gap; candidate teaser numbers.
- **Tile context is not the bottleneck**: 512 vs 1024 tiles change fidelity by < 0.06 dB for every model. The dominant cause of the gap is out-of-distribution blur size (CoC ×4), not missing context. Seams are a secondary effect (see grid-shift metric). → Intro should lead with blur-size OOD; do not sell Stage 2 mainly on seam removal.

**Decisions** (protocol v2, eval tag `dpdd2` in `code/experiments/dpdd_p1.yaml`):
1. **Registration = translation** (was homography). Official 1680×1120 test pairs: translation/euclidean
   ECC median 0.41/0.58 px (max 3.8/4.3); affine/homography median 1.3/2.1 px, max 18/30 px for a
   negligible correlation gain (e.g. `1P0A2513`: 0.5 vs 19.5 px, ECC 0.953 vs 0.962). **Native, on the raws of
   the 37 indoor test pairs** (outside, plain libraw development): translation median 1.5 px, max 6.6,
   none > 10 px; euclidean median 2.1, max 11.9; homography median 6.8, max 114, **12/37 > 10 px** (=
   the secure P1 count) — with identical ECC. The 25–79 px P1 shifts were homography overfitting the
   f/4–f/22 blur difference, not motion. Native targets must be re-registered.
2. **Seam metric = grid-shift consistency** (`gridshift`: same model on a second tile grid offset by half a stride; `gs_psnr`, `gs_mad`, `gs_seam_ratio`). The old `seam_ratio` has a content-driven floor of 1.19 (median 0.99, p10–p90 0.62–1.76) on *untiled* images, so P1's 1.09–1.42 values carry no signal. On real DPDD (DRBNet, 1680×1120, 512 tiles), `gs_seam_ratio` = 2.9 vs old `seam_ratio` = 1.06.
3. **PSF matching is not used as primary — plain PSNR/SSIM are.** Measured outside on the raw
   CR2s of the 37 indoor test pairs (`code/scripts/paired_edge_mtf.py`): on the *same* 336 focal-plane
   edges (DP focus mask) the ratio MTF(f/22)/MTF(f/4) is 1.05 / 1.24 / 0.99 / 0.64 at 0.1 / 0.2 / 0.3 /
   0.4 c/px (Airy theory: 0.75 / 0.50 / 0.25 / 0.05); Gaussian fit σ ≈ 0 px (95 % CI 0–0.31) vs 0.91 px
   for Airy-only. The f/22 target is as sharp as in-focus f/4 below 0.3 c/px (wide-aperture aberrations
   and residual defocus offset diffraction), so blurring predictions by the Airy PSF would penalize
   correct outputs. `pm` stays as a sensitivity column; `pmg` dropped. (The secure P1 finding that the
   target is blurrier than Airy×pixel in absolute terms is reproduced here: all-edge MTF50 0.128 vs
   their 0.130 c/px — the extra blur is shared with the f/4 input.)
4. **Blur-level stratification from DP views** (`dp_maps.py`, metric `blurbins`): bins on smoothed |DP disparity| at 1680×1120: [0,0.4) focal plane, [0.4,1.5), [1.5,3.5), [3.5,∞) DP px (≈ quartiles on the test set). Validated: per-image Spearman(|disparity|, log sharpness gap target/input) median 0.76. Per-bin PSNR is content-confounded (blurry input: 25.7 / 23.6 / 23.3 / 26.0 dB — strongly defocused regions are often smooth), so report per-bin **gains over the blurry input** (`summarize.py --ref`).
5. **Statistics**: per-image paired differences with 95 % bootstrap CIs (`summarize.py --ci --ref`) for every claim; n = 37 (→ 76 with outdoor raw).
6. **Native test set**: the only full-resolution DPDD data is raw CR2 (indoor 29.5 GB, outdoor 30.4 GB). The authors' processed release (`dd_dp_dataset_canon.zip`, 15.8 GB) is 1680×1120 — the same files as the Hugging Face copy used in P1 (identical layout and sizes). So the native set must be developed from raw (D3); absolute numbers are not comparable to published DPDD numbers, comparisons within our protocol are.

---

## 0. Key facts that shape the protocol

- **DPDD full resolution exists only as raw CR2** (correction 2026-10-02; the processed 16-bit PNG release is 1680×1120): 500 scenes, Canon EOS 5D Mark IV, 6720×4480 (30 MP), split 70/15/15 → test = 76 scenes. The standard benchmark resolution 1680×1120 is **exactly 1/4 of native** → Stage 1 at 4× down = the standard DPDD protocol, so every published pretrained model is a valid anchor out of the box. (f/4 blurry, f/22 sharp — verified in the paper)
- **How DPDD 1680×1120 was made is not documented.** The paper only says images are downscaled to 1680×1120 before training. The official DPDNet code (`DPDNet/data.py`, test mode) applies `cv2.resize(img, (1680, 1120))` = cv2 default **INTER_LINEAR, no antialiasing** (aliases fine detail; if the release was made this way the standard benchmark itself contains aliasing). Verified facts from the paper: blurry f/4, sharp f/22, tripod + remote trigger, same focus distance/focal length, 16-bit sRGB PNG processed from CR2. Metrics in their code: PSNR/SSIM on float [0,1] 16-bit, `compare_ssim(..., multichannel=True)` → our `fidelity.ssim` matches it. Resolve empirically: `prepare_scales.py --reference` compares all filters against the official release (>~60 dB = identical pipeline). If none matches (e.g. resized during raw export in Canon DPP), use our own `area` downscaling for all resolution-gap experiments (grid-consistent with native) and the official files only for reproducing published numbers.
- **Ground truth is diffraction-limited at native res.** Airy disk diameter ≈ 2.44·λ·N = 2.44·0.55µm·22 ≈ 30µm; 5D IV pixel pitch ≈ 5.36µm → **~5–6 px blur in the "sharp" GT** at native res (≈1.4 px at 1/4 res, hence invisible in the standard protocol). A sharper-than-GT output is penalized by PSNR/SSIM. Must be handled explicitly (§3.1).
- **Misalignment** between the two captures scales ×4 at native res (a ~1 px error at 1680×1120 becomes ~4 px). Needs registration (§3.2).
- **Other real paired sets are not native**: RealDOF (IFAN, 50 scenes, ~2320×1536, beam-splitter, test-only) and RealDefocus/RealBokeh (released at 3 MP; captured at 6000×4000 but full-res not public). Usable only as mid-resolution / generalization checks.
- Hence: **real paired native = DPDD only** → complement with a synthetic native set (clean GT) and a real unpaired native set (no GT).

## 1. Test sets

| ID | Set | Res. | GT | Purpose | Priority |
|---|---|---|---|---|---|
| T1 | DPDD test (76) | 6720×4480 | real f/22, registered | Main paired table | **P1** |
| T2 | Synthetic UHD defocus test (~100) | 24–50 MP | exact (rendered from sharp HR photo + depth) | Clean fidelity, per-CoC analysis, no diffraction/misalignment | P2 |
| T3 | Real unpaired UHD captures (DSLR 24–60 MP + phone 50–200 MP) | native | none | NR metrics, consistency, qualitative, user study | P2 [D: do we have captures / can we shoot?] |
| T4 | RealDOF, RealDefocus-3MP | 3–3.5 MP | real | Generalization / "does not hurt at moderate res" | P4 |

T2 must not overlap synthetic training sources; source images must be genuinely sharp at native res (select by focus/sharpness score). [D: source of sharp HR photos]

## 2. Compared methods (main table)

All baselines use publicly released DPDD-trained weights, no retraining, unless noted.

**G1 — Native patch-wise deblurring** (motivation: CoC exceeds receptive field/patch, seams)
- Restormer, NAFNet, LaKDNet, DRBNet, IFAN, + 1–2 recent strong ones (FFTFormer / EVSSM / EAMamba — pick those with DPDD weights) [V: weights availability]
- Tiling: 512 (train size) and 1024 tiles, 1/8 overlap, linear/Gaussian blending; TLC for Restormer/NAFNet.

**G2 — Low-res deblur (1680×1120) + ×4 upsampling** (motivation: blind to native observation, hallucination)
- Upsamplers: bicubic; regression SR (SwinIR or HAT, real-world weights); generative SR multi-step tiled (StableSR or SUPIR); one-step generative SR (OSEDiff; + a 2025/26 one-step SR if code exists).
- Anchor for G2 = same LR method(s) we use as Stage 1 (e.g. Restormer), so the only difference is the upsampler.

**G3 — Guided upsampling with the blurry native input as guide** (key ablation-style baseline for copy/generate)
- Joint bilateral / guided filter / Deep Guided Filter using native blurry input as guide.

**G4 — Ours**
- Ours (native Stage 1 signals); Ours + off-the-shelf anchors with surrogates (the plug-in table, §5).

Optional [D]: one UHD-restoration architecture (e.g. UHDformer) retrained on our data; a diffusion defocus-deblur method run natively patch-wise if code is available.

## 3. Protocol

### 3.1 Diffraction-limited GT (T1) [D: choose]
Options (can report more than one):
- (a) **PSF-matched fidelity**: convolve every prediction with the f/22 diffraction PSF (Airy, per-channel λ) before PSNR/SSIM → does not penalize outputs sharper than GT. Cheap, physically motivated, a protocol contribution. [V: check GT residual MTF matches Airy; lens aberrations may add more]
- (b) Report fidelity additionally at 1/2 res (3360×2240, diffraction ≈ 2.8 px).
- (c) Rely on T2 for exact fidelity; on T1 lead with LPIPS/DISTS + NR + consistency.
Proposal: **(a) primary for PSNR/SSIM on T1, raw (unmatched) numbers in supp, (c) for per-CoC analysis.**

### 3.2 Registration (T1)
- Align GT to the **blurry input** once (not to each prediction, to keep it method-independent): global homography (ECC on luminance) + low-magnitude dense flow check; report residual. Crop a fixed border (e.g. 64 px native) for all methods.
- Freeze aligned GT + masks, version them, and publish the procedure.
- Sanity check: metrics of standard 1680×1120 protocol reproduced on our downsampled registered GT within ±0.1 dB of published numbers.

### 3.3 Blur-level stratification
- Per-pixel defocus magnitude for T1 from **DPDD dual-pixel views** (DP disparity ∝ CoC) or from T2 render maps. Report metrics in 3–4 CoC bins (in-focus / mild / strong / extreme). This is where the copy/generate claim is tested: ours should match G3 in in-focus bins and beat G2 in strong bins.

## 4. Metrics

**4.1 Fidelity (T1, T2)**: PSNR, SSIM (PSF-matched on T1), LPIPS, DISTS; **high-band PSNR** (on Laplacian-pyramid levels above the 1/4-res Nyquist) — directly measures native-resolution detail, which whole-image PSNR hides.

**4.2 No-reference (T1–T3)**: MUSIQ, MANIQA, CLIPIQA (computed on 512 crops, averaged). Secondary only; note known bias toward over-sharpening.

**4.3 Consistency** (from notes.txt taxonomy; the paper's distinctive metrics). Must be **tiling-agnostic** where possible, since methods tile differently.
1. *Low-frequency (tone/color)*: std over a fixed grid of cells of the per-cell mean color error (pred − GT) after heavy low-pass; NR variant on T3: drift vs. the LR anchor.
2. *Seams*: seam ratio = gradient energy of (pred − GT) on each method's own tile boundaries / at random control lines. NR variant on T3: same on pred alone. Reported for tiled methods only.
3. *Semantic/texture consistency*: for patch pairs that depict the same content (pairs chosen by DINOv2 similarity **on GT**), compare d(pred_i, pred_j) with d(GT_i, GT_j) using LPIPS and Gram/style distance → "relative consistency error". NR variant on T3: Intra-LPIPS/Intra-Style (SyncDiffusion) over matched pairs.
4. *Statistical*: per-cell maps of sharpness (Laplacian variance), noise level, spectral slope; report correlation / log-ratio error vs. GT maps. Detects patches that are "differently sharp" without a hard seam.

**4.4 Efficiency**: time per 30 MP image, peak GPU memory, GPU type; same hardware for all.

**4.5 Human study** [D]: 2AFC on T1+T3 crops, ours vs. best G1 / best G2 (~20 participants).

## 5. Tables / figures (target)

1. **Resolution-gap analysis** (motivation, contribution 1): G1 methods evaluated on T1 at 1/4, 1/2, 1× native (and on T2 per CoC bin) → PSNR/LPIPS drop with resolution. **P1, cheap, pretrained only.**
2. **Main table**: T1 (+T2) — G1, G2, G3, Ours; fidelity + consistency + time.
3. **Plug-in table**: for N LR methods (Restormer, NAFNet, DRBNet, LaKDNet, IFAN, …; ≥2 held out from surrogate/anchor training): bicubic ↑ vs. best G2 ↑ vs. Ours ↑ (surrogate). Plus native vs. surrogate signals for our Stage 1.
4. Per-CoC-bin plot (§3.3).
5. Consistency metrics table + qualitative crops (distant same-texture regions).

## 6. First tasks for the secure env (in order)

1. **[P1] Data**: DPDD test at native res; run `prepare_scales.py --reference <official 1680x1120>` and report the PSNR per filter; build 1×, 1/2, 1/4 versions.
2. **[P1] Registration** (§3.2) + frozen masks; reproduce published 1680×1120 numbers for 2–3 methods.
3. **[P1] Metric suite**: fidelity (+PSF-matched, high-band), NR, consistency 1–4, as one script with per-image CSV output.
4. **[P1] Resolution-gap analysis** (Table 1) with G1 methods.
5. **[P2] G2 / G3 baselines** on T1.
6. **[P2] T2 synthetic test set** generation.

Report back: registration residual stats, reproduction check, Table 1 numbers, per-image metric CSV summaries (not full CSVs if large), runtime per method.

## 7. Open decisions [D]
1. Diffraction handling (§3.1).
2. T3: existing native unpaired captures? which devices?
3. Source of sharp HR photos for T2 (and training synthesis).
4. Generative SR baselines to include (compute: SUPIR at 30 MP is slow).
5. Human study yes/no.
6. Tile size/overlap for G1 (fix one main setting, others in supp).
