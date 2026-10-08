# Method plan — reference-guided native upsampler (revised 2026-10-07)

Replaces the 2026-10-06 plan (in git history). Decided with the user on 2026-10-07 after handoff 5
(`handoff/from_secure/2026-10-07_h5.md`): training on the 350 DPDD-native pairs did not overfit (targets misaligned by
1–2 px, so L1 learns the mean; too few scenes), and retrieval quality could not be measured (no ground truth for which
exemplar helps; the spectral oracle failed).

- **Deadline**: CVPR 2027, mid-November 2026 (≈ 5 weeks from today).
- **Training on synthetic pairs** from sharp high-resolution datasets (DIV2K, DIV8K, Flickr2K; LSDIR if more is
  needed): exact targets, exact exemplar labels, unlimited data.
- **DPDD native** (350 / 74 / 76) is used for validation (model selection) and test, never for training in v0.
- **Evaluation**: frozen protocol `dpdd4` (out-of-the-box PSNR, SSIM, LPIPS, DISTS at native, MUSIQ, CLIPIQA,
  PSNR/SSIM at ×4; diagnostics never rank). All 76 test scenes.

**2026-10-08 — first step changed (user):** before building our own model, adapt the most recent reference-based
diffusion SR with code + weights — ReFIR on SeeSR (NeurIPS 2024) and iRAG (ICCV 2025, 50 DDIM steps) — to our setting
(x4 deblurred anchor + same-image references: `self`, `retrieved`), on selected val tiles first (`scripts/tile_study.py`,
`docs/baselines.md` §D3, handoff 6). Find what breaks, fix it, and decide then whether the synthetic-training plan
below is still needed.

## 0. Decisions (2026-10-07)

| Topic | Decision |
|---|---|
| What the model is | a **generic ×4 upsampler**: degraded ×4 image + reference patches → native resolution |
| Anchor at test | the output of any existing ×4 deblur / denoise method (unchanged); not tied to Bokehlicious / DRBNet |
| Anchor in training | **simple generic degradation** of the downscaled target (§3.1) — not a simulation of real defocus or of a specific deblurrer; simple degradation is what makes it generalise |
| Native observation | enters **only through the attention route**: native crops are the reference memory (values). No native-input encoder, no DP input channel to the network |
| Regimes | one mechanism: the query's own native patch in the memory (in focus → copy), mildly blurred references with their blur level (partial use), other in-focus references (transfer), learned null (abstain) |
| Blur map at test | selects and labels the references (DP oracle first; predicted map later); never an image input |
| Reference selection | DINOv2 (+ colour) candidate retrieval, then a **retrieval head trained on transfer-gain labels** from the synthetic ground truth (§4) |
| Attention | **masked cross-attention**: only the retrieved top-K per query token + null, bias = learned retrieval score; further masks by key blur, query flatness, scale (§5) |
| Anchor lock | kept: `down4(y) = anchor` exactly |
| Unit of work | 512×512 native tiles at test (overlap tile/8, linear blend, global lock after blending) |
| Base model | v0 = HAT-L ×4 pretrained backbone + exemplar cross-attention + anchor lock; slim later (§6) |
| Variant B (one-step DiT) | conditional, as before (§8) |

## 1. What the baselines and handoff 5 say

| Finding | Consequence |
|---|---|
| Native inference ≈ identity; ×4 deblur + bicubic = fidelity ceiling but smooth | Upsample the ×4 deblur; add native detail on top |
| Generative SR loses at ×4 and damages the focal plane | **Lock the low band to the anchor**; never re-synthesize sharp regions |
| Non-learned exemplar transfer: DISTS 0.232, within 0.2 dB of bicubic | Real same-image texture is the main source of detail |
| h5: L1 on DPDD pairs plateaus at bicubic; contextual loss 0.1 dominates the gradient | Train on exactly aligned synthetic pairs; perceptual weight small (≤ 0.01) or ramped |
| h5: DINOv2 recall only slightly above pixels; ignores colour; flat queries retrieve noise; edges not retrieved | Learn retrieval from gain labels; add colour; mask flat queries to null; structure comes from the anchor, references only supply texture |
| h5: Restormer / DRBNet are 0.75–1.8 dB better on train than val/test | Do not train on real-deblurrer anchors of DPDD; generic degradation instead |

**Target region** (unchanged): PSNR ≥ bicubic of the same anchor and DISTS ≤ the non-learned exemplar transfer
(≤ 0.232), focal-plane change ≥ 0, ×4 PSNR = anchor.

## 2. Formulation

For a native tile T with anchor crop `a_T` (×4) and reference set `R` (native patches with blur levels):

    Δ = G(a_T ; R, mask)
    y_T = Δ + U(a_T − down4(Δ))      # anchor lock (exact: down4(y_T) = a_T)

- `G` = ×4 SR backbone on `a_T` with masked cross-attention to the reference tokens (+ null).
- Test time: `R` = native crops of the blurry input. Keys in focus (DP |d| < τ₀) are references at blur level ≈ 0;
  mildly defocused crops (τ₀ ≤ |d| < τ₁) enter with their blur level as a key feature; strongly defocused crops are
  never references. The query's own native crop is in `R` exactly when it qualifies — that is the copy path.
- The network never sees the blurry image directly: everything native comes through attention.

## 3. Synthetic training data

Sources: DIV2K (800 train / 100 val), DIV8K (1,500, up to 8K), Flickr2K (2,650); LSDIR (≈ 85k) only if needed.
Everything is generated on the fly on the GPU from the sharp HR image `y` (only HR images + precomputed DINOv2
tokens are cached).

### 3.1 Anchor (generic degradation of `down4(y)`)
Random per sample, simple and spatially smooth: residual Gaussian / disc blur (σ field smooth over the image, 0–1.5
LR px), over-sharpening / ringing, Gaussian + Poisson noise, JPEG, small global colour / contrast / gamma shift.
Ranges set so that bicubic of the anchor spans the PSNR range of real ×4 deblurrer anchors (calibrated once on DPDD
val: real anchors vs targets).

### 3.2 References (same image)
Per training image a memory of native patches at scales {1, 0.75, 0.5}, each with a **blur level** b:
- **copy** case (p ≈ 0.3): the query's own HR patch is in the memory at b = 0;
- **partial** case (p ≈ 0.2): the query's own patch is in the memory blurred with a random small disc / Gaussian
  (b > 0, the mild-defocus regime);
- **transfer** case (p ≈ 0.5): the query's own patch and its neighbourhood are removed; only other locations remain.
Other references get b = 0 or a random mild blur (p ≈ 0.3) so the model learns to discount by blur level. Noise is
added to references like the native input has it.

### 3.3 Labels for retrieval (exact, free)
For a query token q and candidate key k: **transfer gain** g(q, k) = improvement over bicubic of the anchor when k's
native high band (energy-matched, best shift within a small radius) is added at q, measured against the HR target
(PSNR and a texture distance). Gives: positives / negatives for the retrieval head, the **gain oracle** (best k),
and **abstain labels** (no k with g > 0). The same computation on DPDD val (alignment-tolerant) replaces the spectral
M2 of handoff 5.

## 4. Reference selection

1. **Candidates**: DINOv2-L tokens of the anchor (frozen, cached) + a colour / brightness descriptor; cosine top-K₀
   (K₀ ≈ 64) per query token over the whole-image memory. Cheap, recall-oriented.
2. **Retrieval head** (small, trained): re-scores the K₀ candidates from [DINO features, colour, key blur level,
   key scale, query texture]; trained with a contrastive / ranking loss on g(q, k) (§3.3) plus an abstain logit.
   Keeps the top-K (K ≈ 16) with scores. Trained jointly with the upsampler or pre-trained, decided by v0 results.
3. **Fine placement** happens inside the attention (each reference contributes 4×4 tokens; local correlation if
   needed).

## 5. Masked cross-attention

Per query token: attends only to its top-K references + null; bias = α · retrieval score (learned α per head),
−∞ elsewhere (already in `ExemplarAttention`). Additional masks / biases:
- **key blur**: learned bias from the key's blur level (strongly blurred keys never present; mild ones discounted);
- **query flatness**: flat queries (low anchor texture) attend to null only;
- **scale**: optional, keys at the scale implied by the query's depth (later);
- **training**: random key dropout so the model does not over-trust rank 1.

Relevance (attention mass on non-null keys) is an output map for analysis and figures.

## 6. Models

### v0 — validate the formulation (big, pretrained)
HAT-L initialised from its ×4 SR weights on the anchor tile; masked exemplar cross-attention after the body (and at
1–2 intermediate stages if needed); light refinement at native resolution on [backbone output, attended reference
features]; anchor lock. Removed vs the 2026-10-06 v0: native-input encoder, DP input, SFT conditioning from them.
Activation checkpointing (handoff 5: OOM without it). Two runs: **gain-oracle references** vs **retrieved
references** → go / no-go 1.

### v1 — slim (after v0 works)
U-Net, 4 levels, 20–40 M params, NAF blocks, channel attention at 1/4, self + masked cross-attention at 1/8.

### Losses
L1 (exact targets now) + small perceptual / contextual term (≤ 0.01, ramped in after L1 converges); later a light
patch-GAN only on transfer-case tokens. Retrieval head: ranking loss on gain labels.

## 7. Measurements and checks

- **Synthetic**: memorisation test (fixed tiles) must clearly beat bicubic; per-case gains (copy / partial /
  transfer / null) on held-out synthetic images.
- **Retrieval**: recall@K and nDCG against the gain oracle; DINOv2 vs pixels vs learned head; abstain calibration;
  harm rate (top-1 makes it worse).
- **DPDD val** (zero-shot): per-bin gains vs bicubic of the same anchor, with Bokehlicious and DRBNet anchors;
  gain-oracle vs retrieved on real data; coverage (defocused area with a positive-gain key).
- **Alignment of DPDD pairs** (evaluation only): focal-plane residual-shift field per scene; re-estimate on
  focal-plane tiles if structured.

## 8. Variant B (one-step DiT) — conditional
Unchanged: after v0, measure what is left in strong-defocus regions with no relevant reference; only if large, a
VOSR-based prior conditioned on the reference tokens, anchor lock in pixel space.

## 9. Experiments

- Main table: input, native deblurrers, ×4 deblurrers + {bicubic, Real-HAT, S3Diff, VOSR}, **ours with several
  anchors** (no anchor is special; plug-in by construction).
- Plug-in: every available ×4 deblurrer as anchor, zero-shot.
- Ablations: no references · random · gain-oracle · DINOv2 vs learned retrieval vs pixels · masking (score bias,
  blur bias, flatness) · without partial / blurred-reference augmentation · no anchor lock · synthetic only vs +DPDD
  fine-tune (if done) · DP vs predicted blur map.
- Hallucination check on our outputs (text crops, harm rate), runtime from a warm, idle-GPU pass.

## 10. Timeline (from 2026-10-07)

| When | Outside (code + paper) | Secure (data + GPUs) |
|---|---|---|
| Oct 7–10 | synthesis pipeline (anchor degradation, same-image references, copy / partial / transfer mixing), gain labels, retrieval head, masks; CPU tests | get DIV2K / DIV8K / Flickr2K; DINOv2 token cache; anchor-degradation calibration on DPDD val; DPDD alignment measurement |
| Oct 10–17 | method section rewrite | **v0** on synthetic: memorisation test, then gain-oracle vs retrieved → go/no-go 1 (≈ Oct 17) |
| Oct 17–24 | v1 code; analysis | v1 + ablations; zero-shot DPDD val / plug-in → go/no-go 2 (≈ Oct 24) |
| Oct 24–31 | experiments section | B only if §8 says so; full 76-scene test; human-study crops |
| Oct 31–Nov 7 | supplement | final ablations, timing pass |

- **Go/no-go 1**: on synthetic held-out and DPDD val, gain-oracle references clearly beat bicubic of the anchor in
  the transfer case; retrieved within a clear margin. If even the oracle does not help on DPDD val, references are
  not the story.
- **Go/no-go 2**: zero-shot DPDD val in the target region with at least two anchors.

## 11. Carry-over from the 2026-10-06 setup
Reused: memory / retrieval code (`uhdd/train/memory.py`), `ExemplarAttention`, anchor lock, HAT backbone + grad
checkpointing, trainer (`train.py`, torchrun, EMA, resume, val, logs), `train.sh` / `train_status.py`, `ours_*`
registry inference (aux native + DP now feed only the memory). Replaced: DPDD cache as training data, spectral oracle,
native-input encoder / DP input, `sim_anchor` defocus-proportional blur (→ generic degradation).

## 12. Open items
- DPDD fine-tune after synthetic training: decide after go/no-go 1.
- Predicted blur map (deployable): small head on the anchor + downscaled input, supervised by DP; when.
- Human study (2AFC) logistics; scope cuts (review 2026-10-06 #1, to revisit).
