# Method plan — native-resolution guided upsampler (revised 2026-10-06)

Replaces the 2026-10-02 plan (in git history). Decided with the user on 2026-10-06 after handoffs 2–4
(baseline study complete on all 76 test scenes; ceilings in `docs/evaluation.md` §7 and the paper's
"Ceilings" paragraph / frontier figure).

- **Deadline**: CVPR 2027, mid-November 2026 (≈ 5½ weeks from today).
- **Training in the secure env** on DPDD-native pairs (350 train / 74 val scenes, built from raw).
- **Evaluation**: frozen protocol `dpdd4` (out-of-the-box PSNR, SSIM, LPIPS, DISTS at native, MUSIQ, CLIPIQA,
  PSNR/SSIM at ×4; diagnostics never rank). All 76 test scenes.

## 0. Decisions (2026-10-06)

| Topic | Decision |
|---|---|
| Unit of work | **512×512 native tiles** (overlap tile/8, linear blend, `uhdd.tiling`) |
| Inputs | native blurry crop `x` · bicubic-upsampled ×4 anchor `a↑` · blur map · same-image in-focus exemplars |
| Blur map input | **DP disparity map directly** (no calibration). CoC radius / per-band MTF availability maps = **ablation** |
| Blur map at test | DP oracle first; predicted map = the deployable version, later |
| Exemplars | **same image only**; retrieved with **DINOv2** features in the anchor domain (§3) |
| Anchors for training | **Bokehlicious = primary** (not trained on DPDD → its train-set anchors are test-like) + DPDD-trained deblurrers + simulated anchors |
| Structural errors of the ×4 deblur | **out of scope** (inherited by design through the anchor lock, §2) |
| Base model | **validate the formulation with a big pretrained model first (v0), then slim** (§4) |
| Variant B (one-step DiT) | **conditional** on the post-v0 measurement of what is left to generate (§6) |

## 1. What the baselines say the method must do

| Finding (76 scenes, handoffs 2–4) | Consequence |
|---|---|
| Native inference ≈ identity (4/5 networks < 0.2 dB; Bokehlicious +0.64) | Deblur at ×4 (anchor), upsample with guidance |
| ×4 deblur + bicubic = fidelity ceiling (24.79 dB, Bokehlicious), perceptually worse than the input | Keep the anchor's structure exactly; add native detail on top |
| Bicubic keeps the anchor at ×4 (26.10 vs anchor 26.08); generative SR loses there (S3Diff −0.8 dB) | **Lock the low band to the anchor** (§2) |
| SD-based one-step SR damages the focal plane (OSEDiff −1.1/−1.6, S3Diff −1.0 dB); VOSR safe but ≈ bicubic | Never re-synthesize where the input is sharp; generation only where nothing real exists |
| Non-learned exemplar transfer: DISTS 0.232 (better than every generative upsampler) within 0.2 dB of bicubic, focal plane untouched | Real same-image texture is the main source of detail |
| Bicubic gains +0.44 dB in the focal plane (f/22 target less noisy than f/4 input) | Copy path = copy **+ light denoise**, not a hard copy |

**Target region**: PSNR ≥ the fidelity ceiling of the same anchor (24.79 dB for Bokehlicious) **and** DISTS at or
below the exemplar transfer (≤ 0.232), focal-plane change ≥ 0, ×4 PSNR = anchor. Empty for every baseline.

## 2. Formulation

Per 512 native tile T; `x_T` native input crop, `a↑_T` = bicubic ×4 of the 128-px anchor crop, `d_T` DP disparity,
memory `M` (§3):

    Δ = G(x_T, a↑_T, d_T ; M)
    y_T = a↑_T + (Δ − up(down(Δ)))          # anchor lock (back-projection): down(y_T) = anchor exactly

- `down` = the area ×4 downscale used everywhere in the protocol, `up` = bicubic. The network only adds content
  in the null space of the downscaler → ×4 fidelity is guaranteed, native PSNR depends only on the added high
  band, and any better anchor directly gives a better result (plug-in by construction).
- Copy behaviour is learned (no hard gate): where `x` is sharp the best high band is `x`'s own (lightly denoised).
- Three regimes the model should learn (stated in disparity terms; the CoC/MTF version is the ablation):
  in focus → **copy**; mild/moderate defocus (high frequencies attenuated, not gone) → **deconvolve from `x`
  itself** (no baseline uses this regime); strong defocus (high frequencies gone) → **exemplar transfer**, then
  the prior.
- Inherits structural errors of the anchor by design (decision: out of scope). Possible later extension: soft
  lock with a per-pixel predicted lock strength.

## 3. Exemplar memory (same image, DINOv2)

Built once per image:
1. **DINOv2 on the whole ×4 anchor** (1680×1120; patch 14 → ≈ 120×80 token grid, each token ≈ 56×56 native px).
   Weights: DINOv2-L, already shipped for VOSR (`$UHDD_WEIGHTS/vosr/torch_cache`).
2. **Keys** = anchor tokens in in-focus regions (DP disparity below a threshold), minus flat / low-texture tokens.
   Keys and queries both come from the anchor → same domain (comparing smooth deblurred regions with sharp native
   texture would be a domain gap).
3. **Values** = the native input crops under the key tokens (64–128 px), at scales {1, 0.75, 0.5} for perspective.
4. **Queries** = anchor tokens covering the defocused part of the current tile.
5. **Retrieval**: top-K by cosine similarity (K ≈ 16) + the similarity scores + a learned **null** token. Retrieval
   only needs recall@K; the model's cross-attention learns relevance (precision).
6. **Training**: exemplars overlapping the tile's own location are dropped (p = 0.5–1) so the model learns to
   transfer, not copy in place. Memory built from the whole training image, loss on the crop.

Relevance (max attention / similarity) is an output map: used for analysis and figures.
Risk: DINO is partly semantic (same class, wrong texture scale) → multi-scale values, scores, null; measured by
recall (§5, M2). Related work to position against: reference-based SR (TTSR, C2-Matching, MASA-SR, DATSR — our
DATSR baseline uses the observation as reference), internal self-similarity SR.

## 4. Models

### v0 — validate the formulation (big, pretrained)
- **Backbone**: HAT-L initialised from its pretrained ×4 SR weights (`hat_l_x4`), which already maps a 128-px
  low-res image to 512 px = the anchor → native path. Added: an encoder for [`x`, `d`] injected into the HAT
  features (per stage, zero-initialised), and cross-attention to the exemplar tokens at the deepest stages. Output
  through the anchor lock.
- Slow (HAT-L ≈ 140 s per 30 MP image in our runs): acceptable for v0.
- Two runs: **oracle exemplars** (chosen with the target's high band) and **retrieved exemplars** → the gap says
  whether to work on retrieval or on the model.

### v1 — slim (after v0 works)
- U-Net, 4 levels, ≈ 20–40 M params:
  - full and 1/2 resolution: NAFNet-style conv blocks (no attention);
  - 1/4 (128²): conv + Restormer-style channel attention (global over the tile, linear cost);
  - 1/8 (64² = 4096 tokens): full spatial self-attention + **cross-attention to the exemplar memory**.
- No window attention: it only adds local mixing that convolutions already provide; global context and exemplars
  are cheap at the coarse levels.
- Optionally distilled from v0.

### Losses (both)
- L1 on the full output (dominates in focus and in the low band — the lock already fixes the ×4 band).
- Alignment-tolerant texture loss (contextual / DISTS-like) and, in a second stage, a light patch-GAN **only where
  the disparity is large** (the target is slightly misregistered and noisier; pixel losses alone teach the mean and
  make exemplar texture look like noise).
- Exemplar contribution scales with relevance: wrong high-band energy costs PSNR roughly by its power.

### Training data
- Real pairs: 512 crops of the 350 native train scenes (≈ 40k distinct tiles), val 74.
- Anchors: Bokehlicious ×4 on train inputs (primary), the DPDD-trained deblurrers (diversity), simulated anchors
  (target ↓4 + residual blur ∝ disparity + ringing + noise, matched to real anchor error statistics per bin).
- Optional pretraining: synthetic native pairs (f/22 target re-blurred with a spatially varying disc from the DP
  map) — exactly aligned, teaches the deconvolution regime; then fine-tune on real pairs.
- Blur map at train: DP; later mix DP and predicted (p = 0.5) for robustness to the predicted map.

## 5. Measurements before training (secure, cheap)

- **M1 — regime shares**: fraction of pixels per regime (in focus / mild–moderate / strong defocus) on train and
  test, from a rough disparity → CoC factor (analysis only; fitted from the existing sharpness-loss correlation and
  focal-plane MTF data). If most defocused pixels are mild–moderate, the deconvolution regime dominates.
- **M2 — exemplar retrieval recall**: per defocused cell, oracle exemplar = in-focus native crop whose high band
  best matches the target's; recall@K of DINO retrieval vs random vs the pixel-feature matching of the non-learned
  exemplar transfer; coverage (fraction of strong-defocus cells with a relevant match).
- **M3 — anchor check (light)**: same model, real vs simulated anchors on val; no formal train/test gap study needed
  (Bokehlicious is not DPDD-trained).

## 6. Variant B (one-step DiT) — conditional

After v0: measure the error left in strong-defocus regions with low exemplar relevance ("nothing to transfer").
- Large and visible → B: **VOSR as the prior** (the one generative baseline that did not damage the focal plane),
  LoRA, conditioned on [`x`, `d`] + exemplar tokens; anchor lock and copy path in pixel space, so the prior only acts
  on the high band of the "nothing to transfer" region. Shared per-image noise across tiles (as in the adapters).
- Small → B is an ablation or dropped; the paper is A. (Saves the VAE-ceiling question.)

## 7. Experiments

- Main table (latest-run convention): input, native deblurrers, Bokehlicious / DRBNet ×4 + {bicubic, Real-HAT,
  S3Diff, VOSR}, **ours (v1, and v0 if different)**; both ×4 renderings tracked (decision 2026-10-06).
- Plug-in: ours with every anchor (Bokehlicious, DRBNet, Restormer, LaKDNet, IFAN), at least one held out of
  training.
- Ablations: no exemplars · random exemplars · oracle exemplars · DINO vs pixel features for retrieval · no blur
  map · disparity vs CoC radius / MTF maps · no native input (anchor-only SR) · no anchor lock · tile 256/512/1024 ·
  DP vs predicted blur map · real vs simulated training anchors.
- Figures: frontier with our points; per-regime gains; relevance maps with arrows to in-focus sources; teaser crops.
- Runtime from a warm, idle-GPU pass.

## 8. Checkpoints and timeline (from 2026-10-06)

| When | Outside (code + paper) | Secure (data + GPUs) |
|---|---|---|
| Oct 6–10 | data pipeline (tiles, anchors, DP, DINO memory), v0 model + trainer, CPU tests | M1–M3; anchors on train/val (Bokehlicious + others), DINO features cached |
| Oct 10–17 | v1 code; method section | **v0** training: oracle vs retrieved exemplars → go/no-go 1 |
| Oct 17–24 | analysis; figures | v1 training + ablations; plug-in anchors → go/no-go 2 |
| Oct 24–31 | experiments section | B only if §6 says so; full 76-scene evaluation; human-study crops |
| Oct 31–Nov 7 | supplement | final ablations, warm timing pass |
| Nov 7–deadline | polish | reruns |

- **Go/no-go 1** (≈ Oct 17): v0 with oracle exemplars inside the target region on val; retrieved within a clear
  margin of oracle. If oracle does not help → exemplars are not the story; fall back to deconvolution + prior.
- **Go/no-go 2** (≈ Oct 24): v1 reaches v0's numbers (or close) at practical runtime; plug-in works with a held-out
  anchor.

## 9. Open items

- GPU budget per run (80 GB-class GPUs available; how many for training).
- Predicted blur map (deployable version): small head on [`x`↓4, anchor] supervised by DP; when.
- Whether to report v0 (HAT-based) or only v1 in the paper.
- Human study (2AFC) logistics.
