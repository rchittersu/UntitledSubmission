# Method plan — native-resolution guided upsampler (2026-10-02)

Decided with the user on 2026-10-02 after the local study (`docs/evaluation.md` §7.2):
- **Deadline**: CVPR 2027, mid-November 2026 (exact date + paper-registration deadline to verify when
  announced). About 6 weeks from today.
- **Two variants, both in the paper**: **A** a strong feed-forward model; **B** a one-step DiT prior.
- **Inputs (both)**: the blurry native-resolution input, the low-res deblurred anchor (any off-the-shelf
  deblurrer at 1/4), and a blur map. **Plus in-focus exemplar guidance** (below): Stage 2 is still limited by
  the tile size, so texture and structure for defocused regions (walls, fabric, repeated structure) are
  taken from *in-focus regions anywhere in the image*, selected by a relevance map.
- **Blur map**: dual-pixel disparity = oracle / upper bound; the method proper uses a predicted /
  re-blur-surrogate map (works for any camera). Both are reported.
- **Training in the secure env** on DPDD-native pairs built from the raw CR2s (≈ 60 GB).
- **Headline metrics**: PSNR / SSIM overall and as per-blur-bin gains over the input, DISTS, per-bin LPIPS,
  plus a small human study. LPIPS alone is not a headline metric (it ranks the blurry input best at native res).

## 1. What the local study says the method must do

| Observation (37 indoor native pairs, DRBNet) | Consequence for the design |
|---|---|
| Native patch-wise deblurring ≈ identity (+0.04 dB) | Do not deblur at native res; deblur at 1/4 (anchor) and *upsample with guidance*. |
| Anchor + bicubic: +1.7 dB but LPIPS 0.31 → 0.55 | Anchor gives structure; native texture must come from elsewhere. |
| Generative SR (SwinIR-real) hallucinates and destroys in-focus detail, worse than bicubic at ×2 | Never re-synthesize where the input is sharp → **copy path**; generate only where needed, and from evidence, not from a generic prior alone. |
| DP composite (copy input where in focus): keeps PSNR, LPIPS 0.52 → 0.44 | Blur-aware copy/generate is necessary; the remaining gap is texture in *defocused* regions. |
| No tile seams / tile-dependent texture found | Cross-patch attention is not sold on seams; it is sold on **where texture comes from** (in-focus exemplars) and consistency of the same material across tiles. |

Target region (8-image subset numbers): PSNR ≥ 27.8 dB **and** LPIPS ≲ 0.35, with DISTS and per-bin LPIPS
better than every baseline in the defocused bins b2/b3.

## 2. Shared formulation

Inputs at native resolution H×W (6720×4480): blurry `x`; anchor `a` = deblurrer(`x`↓4) at H/4×W/4;
blur map `b` (|CoC| proxy in native px, from DP or predicted); focus confidence `f` = P(b < τ).

Output, per tile T (512–1024 native px, overlap, border-aware blend as in `uhdd.tiling`):

    y_T = g ⊙ x_T + (1 − g) ⊙ G(x_T, a↑_T, b_T, f_T ; M)

- `g` = learned copy gate (initialized from `f`): copy real detail where it survives.
- `G` = the generator (variant A or B), conditioned on the tile and on a **global exemplar memory M**.
- `a↑` = bicubic-upsampled anchor (structure / colour reference, residual base).

### 2.1 In-focus exemplar memory (the new component)

Why: inside a 512-px tile of a defocused wall there is no native texture to copy, and the anchor only has
it at 1/4 res. The same material is often in focus elsewhere in the image (another part of the wall,
the same fabric nearer the focal plane). Texture transfer from those regions is evidence-based, unlike
SwinIR-style synthesis.

Built once per image (global, independent of the tile grid):
1. **Candidate cells**: native 64×64 cells with high focus confidence `f` *and* agreement between `x`↓4 and
   the anchor (the cell really is sharp), ranked by texture energy; keep top K (K ≈ 512–2048), at scales
   {1, 0.75, 0.5} to cover perspective changes of texture scale.
2. **Keys** in the *anchor domain*: encode the cell's 4× downscaled version with the same low-res encoder that
   encodes the anchor (both are deblurred / in focus at 1/4, so matching is reliable across focus).
3. **Values** in the *native domain*: features of the full-res input cell (tokens, e.g. 8×8 per cell).

Per tile: queries = anchor features at the tile; relevance r = softmax(q·k/τ) over top-k candidates plus a
learned **"no match"** token (so it can abstain and fall back to the prior). The **relevance map**
(max similarity per location) is an output: used for gating, visualization and analysis.
Injection: cross-attention at 1/4 and 1/8 native feature scales (A) or extra joint-attention context
tokens (B, MMDiT). Every tile sees the same memory → same material gets the same texture source across
tiles (consistency without explicit tile-to-tile attention).

This replaces the "Stage-1 affinity map routes cross-patch attention" idea of the draft with a concrete,
measurable version: affinity = anchor-domain relevance, restricted to in-focus sources.
Related work to position against: reference-based SR (TTSR, C2-Matching, MASA-SR, DATSR, ...) — here the
reference is internal and selected by focus; internal / self-similarity SR (ZSSR, cross-scale
self-similarity); exemplar-guided restoration. [V: verify citations]

Training details: memory sampled from the **whole** training image while the loss is on a crop; with
p = 0.5 cells overlapping the crop are removed (forces transfer instead of copying from itself).

### 2.2 Blur map

- Oracle: DP disparity (`dp_maps.py`), smoothed |d| → native px.
- Method: tiny head predicting the blur bin / level from re-blur agreement between `x`↓4 and the anchor
  (+ local input sharpness), supervised by the DP map on DPDD train. At train time the DP and predicted
  maps are swapped at random (p = 0.5) so the generator is robust to the predicted one.

### 2.3 Anchors (plug-in)

Training anchors from {Restormer, DRBNet, IFAN, LaKDNet, Bokehlicious} at 1/4, plus synthetic anchors
(target↓4 degraded by residual blur ∝ blur map, noise, mild ringing). One model held out at test time
for the plug-in claim.
**Risk**: those deblurrers were trained on DPDD train at 1680×1120 = exactly our ×4 level, so train-set
anchors are better than test-set anchors. Measure anchor PSNR train vs test (runbook step 5); the
synthetic-anchor mix is the mitigation; fallback: 2-fold retraining of one Stage-1 model.

## 3. Variant A — feed-forward

- U-Net (NAFNet-style blocks), 4 levels, input concat [`x`, `a↑`, `b`, `f`] → residual over `a↑`, copy gate
  `g`; exemplar cross-attention at levels 3–4; ≈ 20–40 M params.
- Losses: L1 + 0.1·LPIPS(vgg) + FFT-L1; stage 2 of training adds a light patch-GAN (weight ≈ 5e-3) only on
  pixels with `f` < 0.5 (do not invent texture where the input is sharp).
- Crops 512 native, batch ≥ 8/GPU, AdamW 2e-4 cosine, ~100k iters; tiles 1024 at inference.

## 4. Variant B — one-step DiT prior

- Backbone choice by **VAE ceiling** (deferred; not in handoff 2): reconstruct native DPDD targets through each
  candidate VAE and keep the one that preserves in-focus texture best. Candidates: SD3.5-Medium (MMDiT,
  16-ch f8 VAE), FLUX.1-schnell VAE (16-ch f8), PixArt-Σ (SDXL 4-ch f8), Sana / Sana-Sprint (DC-AE f32,
  one-step natively, likely too lossy). Licences to check before use.
- One-step: start latent = anchor latent (`a↑` encoded) noised to a fixed t*; predict x0 in one step
  (OSEDiff-style); LoRA (rank 64) + condition embedder for [`x` latent, `b`, `f`]; exemplar tokens appended
  to the joint-attention context. Losses: latent L2 + LPIPS/DISTS on decoded crops + adversarial / VSD
  regularizer for realism (masked to defocused pixels, as in A).
- Final pixel-space copy fusion with `g` (VAE cannot reproduce in-focus grain exactly).
- Prior work to check for one-step DiT SR: OSEDiff, TSD-SR, DiT4SR, DreamClear, HYPIR, PiSA-SR [V].

## 5. Experiments

Main table (76 native test pairs once outdoor raws are in; 37 indoor until then), protocol v2:
blurry input · native patch-wise (best model) · anchor + bicubic · anchor + SwinIR-real · DP composite ·
**A** · **B**, each with DRBNet and Restormer anchors (+ held-out anchor).
Ablations (on A, cheap): no exemplar memory · random exemplars (no relevance) · no blur map · DP vs predicted
map · no blurry input (anchor-only SR) · tile 256/512/1024 (memory should make tile size irrelevant) ·
anchor model swap.
Figures: teaser (input / bicubic / SwinIR / ours with relevance arrows from in-focus source), relevance
maps, per-bin gain plot, runtime and memory at 6720×4480.
Human study: 2AFC, ≈ 15 raters × 40 crop pairs, ours vs {composite, SwinIR, native patch-wise}.

## 6. Go / no-go checkpoints

1. **Exemplar pilot (local, non-learned, by Oct 6)**: on the 37 indoor native pairs, transfer the high
   band of the best-matching in-focus input cell (match in the anchor domain) onto `a↑` in defocused cells.
   Report LPIPS / DISTS / HB / PSNR in bins b2–b3 vs bicubic and SwinIR, plus an **oracle** (match on the
   target's high band) = how much usable texture exists in-focus at all, and match coverage. If the oracle
   does not help in b2/b3 the memory is not worth it → A without memory, B carries texture.
2. **A v0 (by ≈ Oct 16)**: beats the DP composite on PSNR *and* LPIPS on the indoor set; memory ablation
   shows a significant b2/b3 DISTS/LPIPS gain.
3. **B v0 (by ≈ Oct 30)**: better perceptual metrics than A at ≤ 0.5 dB PSNR cost, no hallucinated text
   (checked on the crops of `code/analysis/comparison_crops.py`).

## 7. Timeline (from 2026-10-02)

| Week | Outside (code + paper) | Secure (data + GPUs) |
|---|---|---|
| 1 (Oct 2–9) | exemplar pilot; training code A (data cache, memory, model, DDP trainer) + CPU tests; this plan | **handoff 2** (`plan/secure_runbook.md`): native train/val sets, DP maps, anchors, anchor gap, VAE ceiling |
| 2 (Oct 9–16) | B code (backbone per VAE ceiling); intro reframing | A v0 training + ablations |
| 3 (Oct 16–23) | analysis of A; related work + bib | B training; A plug-in anchors; outdoor test raws |
| 4 (Oct 23–30) | figures; method section | B tuning; full evaluation on 76; human-study crops |
| 5 (Oct 30–Nov 6) | experiments section; supplement | final ablations, runtime |
| 6 (Nov 6–deadline) | polish | reruns for reviewers' obvious questions |

## 8. Open items

- GPU budget in the secure env (number and type) → sets batch sizes and whether B uses SD3.5-Medium.
- Storage for the native training set: ≈ 90 GB PNG (x1+x2+x4 of 350+74 scenes) + ≈ 130 GB uncompressed
  training cache (can be the cache only).
- Synthetic defocus training data (sharp HR photos + rendered defocus): not needed for v0; consider if
  DPDD-native (350 scenes) overfits.
