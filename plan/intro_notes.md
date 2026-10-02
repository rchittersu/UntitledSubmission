# Intro / framing notes from the first results (2026-10-02)

Not applied to `Template/sec/1_intro.tex` yet — waiting for the complete P1b tables (G2 with all
upsamplers, IFAN, re-registered targets, CIs). Line numbers refer to `sec/1_intro.tex` at commit 381e8b0.

## Supported by data

- **Both straightforward strategies fail, in different metrics** (intro ¶ "Two straightforward strategies",
  l. 70–94). P1: native patch-wise Restormer PSNR 25.19 / LPIPS 0.253 vs. Restormer at 1/4 + bicubic
  26.12 / 0.449 (37 native indoor scenes). The "complementary failure" story holds and is quantifiable:
  patch-wise loses fidelity, deblur-then-upsample loses detail (high-band error, LPIPS). Candidate
  teaser numbers once the G2 rows with SwinIR-real exist and CIs are in.
- **The gap grows with resolution** for every model (Table 1: −1.4…−1.7 dB, +0.11…0.16 LPIPS from
  1/4 to native). Supports contribution 1 ("we identify and analyze the gap").

## Needs rewording

- **Cause of the patch-wise failure** (l. 76–83): the draft gives two "compounding problems" with equal
  weight — (1) CoC exceeds the patch / never-seen blur extent, (2) seams between patches. Data: tile
  512 → 1024 changes fidelity by < 0.06 dB, and seams are small (DRBNet at 1680: 0.1/255 mean
  difference between two tile grids, though concentrated 3× at seams). → Lead with **out-of-distribution
  blur size** (networks trained at ≤ 1680×1120 never saw 4× larger CoC); present seams/inconsistency
  as a secondary, measurable effect. Do not claim that more context per patch would fix it.
- "the CoC ... can exceed the patch size itself" (l. 76–78): only true for very large tiles' worth of
  blur; with DP disparity we can now quantify CoC distribution at native resolution (|disparity| p90
  ≈ 5 DP px at 1680 → ~20 native px DP disparity; convert to CoC diameter before claiming).
- **Seam/consistency claims for Stage 2** (cross-patch attention, l. 135–139, contribution 2): keep, but
  motivate mainly by *consistency of synthesized texture across distant regions* (semantic inconsistency),
  which the planned DINOv2-pair metric measures, not by visible seams. Re-check after `gridshift` numbers
  for Restormer / generative upsamplers (expected larger than DRBNet's).
- **Evaluation-protocol paragraph** (l. 194–207): now has concrete content: euclidean registration
  (homography overfits blur differences), grid-shift consistency (content-free), DP-based blur-level
  stratification, diffraction-aware fidelity (pm) with measured target MTF. "introduce metrics that
  directly quantify discontinuities along patch boundaries" → "a content-free grid-shift consistency
  measure".
- **DPDD capture resolution** (l. 40–43, `% VERIFY`): verified — captured 6720×4480, released processed
  PNGs at 1680×1120; full resolution only as raw CR2. Native test images must be developed from raw:
  say so in the protocol section.

## Open

- Diffraction-aware PSNR: **not** primary (paired focal-plane MTF: f/22 target as sharp as in-focus f/4
  below 0.3 c/px). The intro's "diffraction-limited at native resolution" (l. 197–198) must be softened:
  the target is diffraction-limited only near Nyquist; relative to the f/4 input it is not blurrier.
- Teaser panel choice: needs a scene where (a) patch-wise leaves strong blur and (b) bicubic is soft,
  ideally with repeated texture for the consistency point. Ask the secure side for 2–3 candidates by name.
