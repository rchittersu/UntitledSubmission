# Baselines — what we compare against, how each is set up and run

Companion to [`evaluation.md`](evaluation.md) (protocol and metrics). Every baseline answers one reviewer question;
the table says which. Legend: ✅ done · 🔶 priority · ⏳ todo · [S] secure env (GPUs) · [O] outside (code / M4 checks).

## 0. Overview

| ID | Baseline | Reviewer question it answers | Code | Status |
|---|---|---|---|---|
| **A1** | Restormer fine-tuned on native crops | "Why not just train at native resolution?" | P2b trainer (shared with ours) | 🔶 [S] after the native train set (runbook step 4) data |
| **A2** | DRBNet fine-tuned on native crops | same, light model | P2b trainer | 🔶 [S] |
| **A3** | Restormer, scale-augmented training (×1/×2/×4 crops) | "Would multi-scale training fix it?" | P2b trainer | ⏳ [S] |
| **A4** | TLC (test-time local converter) on Restormer, native patch-wise | "Is it just train/test statistics mismatch?" | registry `restormer_dpdd_tlc` (`uhdd/adapters/tlc.py`) | ✅ code + test, 🔶 [S] run |
| **B1** | DP composite: input where in focus, anchor + bicubic elsewhere | "Is copying the input enough?" | `fuse_baselines.py --method composite` | ✅ code, local preview |
| **B2** | Blur-adaptive multi-scale composite (input / ×2 / ×4 deblur by blur level) | "Why not pick the right deblur scale per pixel?" | `--method multiscale` | ✅ code, local preview |
| **B3** | Training-free exemplar transfer (in-focus texture) | "Does learning the exemplar memory matter?" | `--method exemplar` | ✅ code, local preview |
| **B4** | Guided-filter upsampling (anchor guided by the native input) | "Does classical guided upsampling suffice?" | `--method guided` | ✅ code, local preview |
| **B5** | Detail transfer (anchor + input high band) | "Is adding the input's own detail enough?" | `--method detail` | ✅ code, local preview |
| **C0** | Anchor + bicubic ×4 / ×2 | upsampling floor | registry `bicubic_x4/x2` | ✅ |
| **C1** | Anchor + SwinIR (classical / real-world) ×4, ×2 | regression / GAN SR | registry `swinir_*` | ✅ |
| **C2** | Anchor + HAT-L (classical) / Real-HAT-GAN ×4 | strongest regression SR | registry `hat_l_x4`, `hat_x4_real` | ✅ code + sanity, 🔶 [S] run |
| **C3** | Anchor + OSEDiff ×4 (one-step diffusion SR) | closest prior to variant B | registry `osediff_x4` | ✅ code (CUDA only), 🔶 [S] run |
| **C4** | Anchor + SUPIR or SeeSR (multi-step generative SR) | strongest generative prior | external scripts | ⏳ [S] |
| **D1** | Anchor + DATSR (reference-based SR, ref = blurry native input) | closest prior to the exemplar memory | `refsr_baseline.py` | ✅ code + sanity, 🔶 [S] run |
| **D2** | C2-Matching / MASA-SR / TTSR | other Ref-SR | — | ⏳ (weights on Google Drive, CUDA DCN) |
| **E1** | Native patch-wise: Restormer, LaKDNet-L, DRBNet, Bokehlicious | existing defocus models at native res | registry | ✅ [S] P1 |
| **E2** | IFAN native patch-wise | — | registry `ifan` | 🔶 [S] runbook step 7 |
| **E3** | NRKNet, GKMNet, UHD-restoration models (e.g. UHDformer), DPDNet-dual (DP oracle) | completeness | — | ⏳ [V weights] |
| **E4** | Standard ×4 table (anchors at 1680×1120) | connects Stage 1 to published numbers | registry | ✅ [S] P1 (standardization check) |

Fairness rules (all groups):
1. **Same anchors** for every "anchor + X" row: DRBNet and Restormer (DPDD weights, whole image at ×4). The upsampler is
   the only difference.
2. **Free parameters tuned on the native val split** (never on test); defaults documented below are only used before
   tuning and are marked "provisional" in any table.
3. **Tiling from each method's paper** (`evaluation.md` §2.7: whole image up to the paper's test size, else tiles of
   its short side; SR models use their code's tile size), **overlap = tile / 8** for all; same GPU type for timing,
   same border crop and masks (`evaluation.md` §2).
4. Pretrained baselines use the authors' released weights (checksums in the runbook / handoff reports), no retraining except group A.

---

## A. Retrained at native resolution (🔶 [S], with P2b)

**Why**: the natural objection to the whole paper. Our local and P1 results show off-the-shelf DPDD models are
≈ identity at native resolution because native blur is 4× larger than anything they saw. Retraining at native
resolution is the direct fix; if it worked, the guided upsampler would be unnecessary.

**Data**: the DPDD-native train set (350 pairs) built by `build_native_set.py` (runbook step 4) — exactly the data our method
uses. Model selection on the native val set (74).

| ID | model | init | crops | schedule (proposal) | test |
|---|---|---|---|---|---|
| A1 | Restormer (26 M) | DPDD weights | 512 native, batch 8 | AdamW 3e-4 → cosine, 60k it, L1 | default tiling (1120 / 140) |
| A2 | DRBNet | DPDD weights | 512 native, batch 16 | as A1, 60k it | default tiling (1120 / 140) |
| A3 | Restormer | DPDD weights | mixed ×1 / ×2 / ×4 crops (p = 0.5 / 0.25 / 0.25) | as A1 | tiled at native |
| A4 | Restormer + TLC (no training) | DPDD weights | — | — | native patch-wise |

Notes:
- A1–A3 share the P2b data cache and DDP trainer with our method (same crop sampler, same augmentation) → identical
  data budget.
- A4: Restormer's channel attention (MDTA) aggregates over the whole input; at test time on 512–1024 tiles the
  statistics differ from training crops (128–384). TLC (Chu et al., ECCV 2022) computes them over local windows:
  `uhdd/adapters/tlc.py` patches every MDTA module to run on overlapping windows of 576 input px (1.5 × the largest
  training crop, scaled to each layer's resolution; stride ½ window; outputs averaged), as the official TLC "grids"
  implementation. Registry `restormer_dpdd_tlc`; matrix group `a4_tlc` (default tiling) in `dpdd_eval_v3.yaml`.
  No training needed.
- Report A rows in the main table (best of A1–A3) and all in the supplementary.

## B. Training-free fusion baselines (✅ code, 🔶 [S] full run)

Script: `code/scripts/fuse_baselines.py`. Inputs: the native blurry input `x`, the anchor deblurred at ×4 and upsampled
to native `up4` (anchor + bicubic), optionally the ×2 deblur upsampled `up2`, the ×4 anchor itself `anchor4`, and the
DP blur level `b` (DP px at 1680, `evaluation.md` §2.4). `ramp(b; a, c) = clip((c − b)/(c − a), 0, 1)`.

| ID | method | formula | parameters (default → tuning grid) |
|---|---|---|---|
| B1 | composite | `w = ramp(b; t0, t1)`; `y = w·x + (1−w)·up4` | t0 0.4, t1 1.2 → t0 ∈ {0.2, 0.4, 0.6}, t1−t0 ∈ {0.4, 0.8, 1.6} |
| B2 | multiscale | `w_in = ramp(b; t0, t1)`, `w4 = 1 − ramp(b; m0, m1)`, `w2 = max(0, 1 − w_in − w4)`; `y = w_in·x + w2·up2 + w4·up4` | t0 0.4, t1 1.2, m0 1.5, m1 2.5 → + m0 ∈ {1, 1.5, 2, 3}, m1−m0 ∈ {0.5, 1, 2} |
| B3 | exemplar | B1 + `(1−w)·tex`, `tex` = high band of the best-matching in-focus cell (below) | t0, t1 as B1; cell 32 px, stride 16, descriptor 16×16 at 1/4, gain clip [0.5, 2] |
| B4 | guided | guided filter (He et al.) of `up4` with guide `x`, per channel | r 8, ε 1e-3 → r ∈ {4, 8, 16, 32}, ε ∈ {1e-4, 1e-3, 1e-2} |
| B5 | detail | `y = up4 + HB(x)`, `HB(x) = x − up4(down4(x))` | none |

- **B2 rationale (default thresholds)**: DPDD-trained models cover |disparity| up to ≈ 3.5 DP px at 1680×1120. At ×2
  the same blur is 2× larger, so ×2 deblurring is trusted up to b ≈ 1.5; ×4 takes over by 2.5.
- **B3 details** (`code/uhdd/exemplar.py:transfer`): cells of 32 native px, stride 16. Bank = cells with mean copy
  weight > 0.9 (in focus). Each defocused cell takes one bank cell by cosine similarity of zero-mean unit-norm
  16×16 windows at 1/4 resolution — query from the **anchor** (deblurred), key from the **4× downscaled input**
  (sharp there, so comparable). The bank cell's luma high band (`x − up4(down4(x))`) is scaled by the ratio of the
  two windows' std (clipped to [0.5, 2]), overlap-added with a Hann window, added to all channels, weighted by `1 − w`.
  Controls in the pilot: random bank cell; oracle (match on the target's own high band).
- **Tuning** (objective: mean per-image masked PSNR on val; `--tune` prints the top-5 settings and writes JSON):
  ```bash
  python code/scripts/fuse_baselines.py --method composite --inputs $VAL/inputs --x4 $VALUP4 --dp-maps $VALDP \
      --tune --targets $VAL/x1/targets --masks $VAL/x1/masks --params $UHDD_RESULTS/dpdd_v2/fusion_params/composite.json
  ```
  PSNR is the objective because these are fidelity baselines; their perceptual columns are reported as they come.
- **Run + evaluate on test**: `code/experiments/run_baselines_bd.sh <anchor model>` (reads tuned params if present).
- Runtime (M4 CPU, per 30 MP image, fusion step only): composite/multiscale < 1 s, detail 0.2 s, guided 1.8 s,
  exemplar 4.3 s.

### B — local preview (✅ [O], 8 images, DRBNet anchor, provisional parameters)
See §B-results at the end of this file (filled from `dataset/results/local/*/metrics_local3.json`).

## C. Low-res deblur + upsampler (G2)

Pipeline: anchor model at ×4 (whole image, = standard DPDD protocol) → upsampler ×4 to native (tiled). ×2 variants:
anchor at ×2 (1024 tiles) → ×2 upsampler.

| ID | upsampler | registry | weights | default tile / overlap (LR px) | status |
|---|---|---|---|---|---|
| C0 | bicubic ×4 / ×2 | `bicubic_x4`, `bicubic_x2` | — | whole image | ✅ |
| C1 | SwinIR-M classical ×4 / ×2 | `swinir_x4`, `swinir_x2` | SwinIR GitHub release | 400 / 50 | ✅ [S] (classical ×4), [O] |
| C1 | SwinIR-M real-world (BSRGAN, GAN) ×4 / ×2 | `swinir_x4_real`, `swinir_x2_real` | SwinIR GitHub release | 400 / 50 | ✅ [O] 8 images |
| C2 | HAT-L classical ×4 | `hat_l_x4` | official: Google Drive (HAT README); mirror `huggingface.co/anchuang/HAT-L_SRx4_ImageNet-pretrain` | 512 / 64 | ✅ code; 🔶 [S] |
| C2 | Real-HAT-GAN ×4 | `hat_x4_real` | official: Google Drive `Real_HAT_GAN_SRx4.pth` (HF mirror has only the "sharper" variant) | 512 / 64 | ✅ code; 🔶 [S] |
| C3 | OSEDiff ×4 | `osediff_x4` | LoRA + DAPE in the repo (`preset/models`); SD2.1-base (HF); RAM swin-L (HF) | 128 / 16 (HR 512) | ✅ code; 🔶 [S] (CUDA only) |
| C4 | SUPIR / SeeSR | — | SDXL + SUPIR ckpts / SD2-base + SeeSR | their own tiled samplers | ⏳ [S] |

Implementation notes:
- **HAT** (`code/uhdd/adapters/hat.py`): the architecture file imports `basicsr` only for a registry decorator and two
  helpers; a minimal shim (timm's `to_2tuple`, `trunc_normal_`) avoids installing basicsr. `multiple: 16` (window size).
  Sanity [O]: weights load strictly; on a bicubic-downscaled native target crop (64 → 256) HAT-L gives 38.49 dB vs
  bicubic 38.31 dB.
- **OSEDiff** (`code/uhdd/adapters/osediff.py`): per LR tile exactly the official test path — bicubic ×4, DAPE prompt
  per tile (`prompt: dape`; `empty` for an ablation), one UNet step at t = 999, VAE decode, AdaIN colour fix to the
  upsampled input. OSEDiff's code hard-codes CUDA → [S] only. Setup [S]:
  ```bash
  git clone https://github.com/cswry/OSEDiff $UHDD_REPOS/OSEDiff     # osediff.pkl + DAPE.pth are in preset/models
  huggingface-cli download Manojb/stable-diffusion-2-1-base --local-dir $UHDD_WEIGHTS/sd21_base   # mirror named in the OSEDiff README
  wget -P $UHDD_WEIGHTS/ram https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth
  pip install peft transformers fairscale   # OSEDiff / RAM requirements (see their requirements.txt)
  ```
- **C4** (⏳): run the authors' scripts on the anchor folder with their own tiling (SUPIR: `--no_llava` to avoid the
  13B captioner, tiled sampler; SeeSR: sd-turbo 2-step mode is feasible at 30 MP), record time; add as an external step.
- Anchors: DRBNet and Restormer (both in `code/experiments/dpdd_eval_v3.yaml`). LaKDNet / IFAN / Bokehlicious anchors
  for the plug-in table come with the method.

### C — results so far
- [O, 8 images, DRBNet anchor] ×4 + SwinIR-real vs ×4 + bicubic: PSNR −0.88 dB (8/8), LPIPS −0.085 (8/8), HB +1.08 dB,
  in-focus bin −0.85 dB; ×2 + SwinIR-real vs ×2 + bicubic: PSNR −0.54, LPIPS +0.068 (worse on both, 8/8).
  Visual: garbled text, painterly fabric, "cracked paint" on smooth surfaces, re-invented wood grain.
- [S, P1, 37 indoor, v1] Restormer ×4 + bicubic 26.12 dB / LPIPS 0.449; + SwinIR classical 25.62 / 0.458.

## D. Reference-based SR (closest prior work to the exemplar memory)

**Why**: our exemplar memory transfers in-focus texture from elsewhere in the image. Reference-based SR (TTSR,
C2-Matching, MASA-SR, DATSR) transfers texture from a reference image. With the blurry native input as the
reference, Ref-SR is the obvious alternative; ours must beat it.

**D1 — DATSR** (Cao et al., ECCV 2022; `github.com/caojiezhang/DATSR`) — chosen because it is the strongest of the
four with weights on GitHub releases (`restoration_mse.pth`, `restoration_gan.pth`, `feature_extraction.pth`).
- **Setup without mmcv** (`code/uhdd/adapters/datsr.py`): DATSR needs `mmcv.ops.modulated_deform_conv2d` (DCNv2);
  a shim maps it to `torchvision.ops.deform_conv2d` (same offset layout, runs on CPU/CUDA/MPS), and the package
  `__init__` files are bypassed. ImageNet VGG16/VGG19 weights are downloaded by torchvision on first use.
- **How it is applied** (`code/scripts/refsr_baseline.py`): LR = the ×4 anchor, tiles of 128 LR px (HR 512), overlap 16
  (= tile / 8, the standard).
  DATSR requires a reference of the same size as the HR tile, so per tile:
  - `colocated`: the blurry native input at the tile's own location (in-focus tiles: exact sharp content).
  - `mosaic` (main): 2×2 mosaic of 256-px native crops = the co-located crop + the 3 in-focus crops of the whole image
    most similar to the anchor tile (in-focus = mean copy weight > 0.8 on a stride-128 grid; similarity = Euclidean
    distance of standardized 15-d colour / gradient-orientation statistics at 1/4 res). DATSR's own patch matching
    then chooses what to use.
  - Weights: `restoration_mse` (fidelity) and `restoration_gan` (perceptual).
- **Sanity** [O]: with the ground truth as reference (input = GT ×4 downscaled), a 512-px tile is reconstructed at
  **45.96 dB** → shim and weights are correct. On a real tile (DRBNet anchor, co-located blurry input as ref):
  bicubic 34.34, DATSR 34.01, input 33.43 dB (single tile, no conclusion).
- **Runtime**: 128 → 512 tile 20 s on M4 CPU, ~5 s on M4 GPU; ~150 tiles per image → [S] GPU only.
- Run [S]: `code/experiments/run_baselines_bd.sh drbnet_single` (mosaic + colocated with mse; mosaic with gan).

**D2 — C2-Matching, MASA-SR, TTSR** (⏳): weights only on Google Drive; C2-Matching and MASA-SR use compiled CUDA DCN
ops (the same shim approach may work for C2-Matching), TTSR is pure PyTorch but older. Add one if DATSR turns out to
be competitive with ours.

## E. Existing defocus deblurring models at native resolution (G1)

| model | weights | default native tile / overlap (§2.7) | status / numbers [S, P1, 37 indoor, v1 renderings, 512 tiles] |
|---|---|---|---|
| Restormer (DPDD single) | official | 1120 / 140 | ✅ 25.19 dB / LPIPS 0.253 (t512) |
| LaKDNet-L (DPDD) | official | 1120 / 140 | ✅ 24.90 / 0.291 |
| DRBNet | official | 1120 / 140 | ✅ 24.96 / 0.272; [O, v2] 1024 tiles: 25.32 vs input 25.28 (≈ identity) |
| Bokehlicious (RealDefocus) | official | 1500 / 188 | ✅ 24.64 / 0.251 (cross-dataset) |
| IFAN | GitHub mirror `jacobsparts/ifan-rs` (bit-identical) | 1120 / 140 | 🔶 runbook step 7 |
| NRKNet, GKMNet | [V] | — | ⏳ |
| UHD restoration (e.g. UHDformer) | [V: defocus checkpoint] | — | ⏳ |
| DPDNet-dual (uses DP views) | official | — | ⏳ only in the DP-oracle comparison |

All rerun on the v2 renderings under tag `dpdd3` (`dpdd_eval_v3.yaml`, group `g1_native_patchwise`).

---

## Status lists

### ✅ Done
- B1–B5 implemented (`fuse_baselines.py`, `uhdd/exemplar.py`), tested, local preview on 8 images.
- C0/C1 run ([S] P1, [O] local); C2 HAT adapter + sanity; C3 OSEDiff adapter (untested on GPU).
- D1 DATSR adapter (mmcv-free) + reference builder + sanity (45.96 dB with GT reference).
- A4 TLC adapter for Restormer + CPU test.
- E1 four models native patch-wise ([S] P1, v1 renderings).
- Experiment config `dpdd_eval_v3.yaml` (C/E groups) + `run_baselines_bd.sh` (B/D groups).

### 🔶 Priority
1. [S] Tune B1–B4 on native val; run B, C2, C3, D1 on test (v3 tag `dpdd3`), DRBNet + Restormer anchors.
2. [S] A1/A2 (native fine-tuning) with the P2b trainer; A4 TLC run (`a4_tlc` group).
3. [S] E2 IFAN; rerun E1 on v2 renderings.

### ⏳ Todo
- A3 scale-augmented training; C4 SUPIR / SeeSR; D2 other Ref-SR; E3 NRKNet / GKMNet / UHD models / DPDNet-dual.
- Plug-in table anchors (LaKDNet, IFAN, Bokehlicious) once the method exists.

---

## B-results
_(filled below from the local evaluation)_
