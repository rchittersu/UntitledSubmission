# Training — the native upsampler (method v0 / v1)

Canonical description of the training setup: data, cache, code, commands. Design: `plan/method_plan.md`; paper:
`Template/sec/3_method.tex`. Written outside and CPU-tested (`code/tests/test_train.py`); run in the secure env.

## 0. Status

| Item | Status |
|---|---|
| Train / val sources in the results layout (`train_x1`, `train_x4`, `val_x1`, `val_x4`), per-split DP maps | ✅ code |
| Training cache builder + measurements M1 (regime shares), M2 (retrieval recall vs oracle) | ✅ code, CPU-tested |
| Exemplar memory (DINOv2 in the anchor domain, native values, multi-scale, oracle) | ✅ code, CPU-tested (pixel features) |
| Simulated anchors | ✅ code; ranges to calibrate (M3) |
| v0 network (HAT-L backbone + native-input encoder + exemplar cross-attention + refinement + anchor lock) | ✅ code, CPU-tested with the tiny backbone |
| Trainer (torchrun, bf16, EMA, resume, val tiles, CSV + TensorBoard) | ✅ code, CPU-tested |
| Manual launcher `code/experiments/train.sh` + `train_status.py` (progress / ETA / val table) | ✅ tested on CPU (start, stop, resume, status) |
| Inference as a registry model (`ours_v0`) through the launcher, global anchor lock after blending | ✅ code, CPU-tested |
| v1 (slim U-Net), GAN stage, predicted blur map | ⏳ after v0 |

## 1. Data flow

```
$UHDD_DATA/dpdd_native_{train,val}/  ──setup_inputs──▶  $UHDD_RESULTS/dpdd/inputs/{train,val}_x{1,4}/, {train,val}_dp_maps
                       launch.sh deblur train_x4 <model>  ──▶  dpdd/deblur/train_x4/<model>@whole/   (real anchors)
                       build_train_cache.py --split train ──▶  dpdd/cache/train/<scene>/ + manifest.json
                       torchrun train.py --config v0.yaml ──▶  dpdd/train/v0/ (model_ema.pt, log.csv, val.csv, tb/)
                       launch.sh upsample ours_x4 bokehlicious ours_v0  ──▶  dpdd/upsample/ours_x4/…/ours_v0@t128o16/ (scored)
```
Folder names of the train / val native sets and DP maps default to `dpdd_native_train`, `dpdd_native_val`,
`dpdd_native_{train,val}/dp_maps` (`uhdd/layout.py` SOURCES / SPLITS). If the secure tree differs, put the real
relative paths in `$UHDD_DATA/uhdd_sources.json`, e.g. `{"splits": {"train": {"dp_maps": "<dir>"}}}`.

## 2. Cache (`code/scripts/build_train_cache.py`)

Per scene under `$UHDD_RESULTS/dpdd/cache/<split>/<scene>/`:

| File | Content | Size (30 MP) |
|---|---|---|
| `x.npy`, `y.npy` | native input / target, H×W×3, native bit depth, memory-mapped (random 512 crops without PNG decoding) | 2 × 180 MB |
| `mask.npy` | validity mask (uint8) | 30 MB |
| `blur.npy` | |DP disparity| at the anchor resolution, confidence-weighted, box 15 (float16): regimes, tile weights, memory keys, loss masks | 4 MB |
| `dp.npy` | network input, 2 channels: |DP disparity| box 3 (keeps depth edges) + DP confidence (float16) | 8 MB |
| `anchor_<model>.npy` | ×4 anchors of each deblurrer (default Bokehlicious, DRBNet, Restormer) | 3 × 11 MB |
| `mem.npz` | memory on the **primary anchor** (Bokehlicious): key boxes (native) and scales, top-32 retrieval per query token, oracle top-4 | ≈ 2 MB |
| `meta.json` | per-scene M1 / M2 numbers | — |

Total ≈ 155 GB for train + val. Retrieval is computed once on the primary anchor and reused for every anchor variant
(in-focus regions look the same in every anchor).

**Memory** (`uhdd/train/memory.py`, also used at inference): DINOv2-L (`dinov2_vitl14`, offline hub cache
`$UHDD_WEIGHTS/vosr/torch_cache`) on the anchor at scales {1, 0.75, 0.5}; token = 14 px of the scaled anchor =
56 / s native px. Keys: tokens with mean |d| < 0.4 and anchor texture above the 30th percentile of in-focus tokens.
Values: native input crops of 64 / s px (resized to 64). Queries: s = 1 tokens. Oracle: keys whose native crop
matches the **target's** high-band spectrum (radial × angular log-power, shift-invariant) at the query.

**Measurements** (in `manifest.json`): `m1_blur_bins_b0_b3` (pixel share per DP blur bin), `m2` (keys per scene,
coverage of defocused tokens, recall@16 of the oracle top-1 among retrieved keys, chance level);
`--compare-pixels` adds `m2_pixels` (the pixel-feature ablation).

## 3. Training sample (`uhdd/train/data.py`)

512-px native tile at a multiple of 4 (its 128-px anchor tile is exact), sampled with weight 0.25 + defocused
fraction. Anchor: real (Bokehlicious 0.5, DRBNet 0.15, Restormer 0.15) or simulated (0.2; `uhdd/train/sim_anchor.py`:
target ↓4 + residual Gaussian blur ∝ |d|, ringing, noise). Exemplars: the tile's query tokens' top-16, keys overlapping
the tile dropped (no copying in place), union capped at 32 by score; per-token score matrix (−∞ where an exemplar is
not among the token's top-16). Same random flip / transpose on everything, including the exemplar crops.

## 4. Network (`uhdd/net/ours.py`, config `code/configs/train/v0.yaml`)

- Backbone: HAT-L ×4 (registry `hat_l_x4` weights) on the anchor tile.
- Conditioning trunk (`CondEncoder`): [native tile pixel-unshuffled ×4, DP input (|d| box 3, confidence)] → features
  added after `conv_first` (zero-init) **and an SFT (per-pixel scale + shift, zero-init) after every backbone stage**
  (HAT-L: all 12 residual groups), so the regime information reaches every depth. Untrained, the model is exactly
  HAT-L ×4 + anchor lock (checked in the tests on the real HAT architecture).
- `ExemplarAttention` after the backbone body: anchor-grid positions attend to exemplar tokens (4×4 per crop) and 4
  null tokens; bias = α · retrieval score of the position's token, −∞ otherwise; zero-init output.
- Refinement at native resolution (NAF blocks) on [backbone output, native input, DP input↑] (zero-init).
- **Anchor lock** `y = Δ + U(a − down4(Δ))` with `down4` = area, `U` = bicubic + nearest-neighbour correction so that
  `down4(U(e)) = e` exactly → `down4(y) = a` to float precision. At inference the lock is applied again to the
  blended image (`finalize`), alternated with clamping to [0, 1].
- Ablation switches in the model config: `exemplars`, `native_input`, `anchor_lock`, `sft` (off = additive only); data: `oracle`, anchor mix,
  `drop_colocated`, `tile`.

## 5. Training (`code/scripts/train.py`)

AdamW (2e-4; backbone 5e-5), warm-up 2k, cosine to 1e-6, 100k steps, batch 4 / GPU × 8, bf16, grad clip 1, EMA 0.999.
Loss: L1 on the validity mask + 0.1 × contextual loss (VGG19 relu3_1, `$UHDD_WEIGHTS/vgg/vgg19-dcbb9e9d.pth`) on
1024 defocused positions per tile. Validation at step 0 (untrained = locked HAT-L, the reference row) and every 2k steps on 256 fixed val tiles (EMA): PSNR, defocused-region
PSNR, ×4 PSNR, and the same for bicubic (reference). Model selection on full val images: launcher on `val_x4`.

## 6. Commands (secure): `code/experiments/train.sh`

One human-readable command per step, run by hand; `train.sh help` lists them in order. Every step is safe to re-run
(finished work is skipped, training resumes from `last.pt`).
```bash
code/experiments/train.sh inputs                 # 1 train / val sources + DP maps
code/experiments/train.sh check                  # 2 offline weights (DINOv2-L, VGG19, HAT-L), GPUs, inputs, caches
code/experiments/train.sh anchors all            # 3 x4 anchors on train_x4 / val_x4 (bokehlicious, drbnet, restormer)
code/experiments/train.sh cache val all          # 4 caches (+ M1 / M2, pixel-feature ablation)
code/experiments/train.sh cache train all
code/experiments/train.sh step0                  # 5 untrained model = locked HAT-L (reference val row)
code/experiments/train.sh overfit auto           # 6 2 most defocused train scenes, 3k steps
code/experiments/train.sh smoke                  # 7 300 steps: it/s, memory, projected time
code/experiments/train.sh start v0_oracle        # 8 background runs (all visible GPUs; NGPU as 2nd argument)
code/experiments/train.sh start v0
code/experiments/train.sh eval v0 val_x4 bokehlicious all     # 9 full images via the launcher (test: ours_x4, at the end)
```
Watching: `train.sh status` (one line per run: state, progress, it/s, ETA, latest val PSNR vs bicubic), `train.sh
status v0` (detail: ETA + finish time, loss, lr, grad norm, peak GPU memory, checkpoints, val table vs bicubic and vs
step 0), `train.sh watch v0` (refreshes every 60 s), `train.sh log v0` (raw output), TensorBoard on `dpdd/train`.
`train.sh stop v0` stops a background run; `train.sh start v0` resumes it with the run's saved `config.yaml`.
Overrides: `--set k=v ...` (e.g. `train.sh start v0 --set name=v0_noex model.exemplars=false` = new run folder).
Underneath: `train.py --config … --set … --out dpdd/train/<name>` (torchrun for > 1 GPU), `build_train_cache.py`,
`train_status.py`.

Weights needed offline: HAT-L (`$UHDD_WEIGHTS/hat/…`, already used), DINOv2-L hub cache (shipped with VOSR), VGG19
(torchvision `vgg19-dcbb9e9d.pth`, to copy over).

## 7. Tests
`pytest -q code/tests/test_train.py`: synthetic scenes (in-focus textures A | B, defocused A) → cache (pixel features)
→ retrieval finds the same texture (> 80 %), M2 recall above chance → dataset shapes → 3 training steps + resume →
trained model as a registry model through the tiler; `down4(y) = anchor` on the blended image.

### Secure-env status [S, 2026-10-07] (handoff 5)
Validated: train / val inputs + DP maps (V1, no scene overlap with test), offline weights (V2), real anchors on train / val (V3), cache + M1 / M2 and a visual retrieval check (V4), step-0 reference (V5: HAT-L x4 + lock, val PSNR 24.38, -0.42 vs bicubic). Open: V6 overfit does not pass yet (L1 + 0.1 contextual plateaus at 32.96 on the two scenes; L1 only reaches bicubic, 33.85, and stays there); V7-V9 and the full runs not started. Fixes found on the way: activation checkpointing (`model.grad_checkpoint`; HAT-L at batch 4 / 512 px otherwise needs > 79 GB), `train.sh` stale pid after foreground runs, `train.sh overfit` ignoring `--set`. Details and tables: `handoff/from_secure/2026-10-07_h5.md`.
