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
| `blur.npy` | |DP disparity| at the anchor resolution, confidence-weighted, box 15 (float16) | 4 MB |
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
- `CondEncoder`: native tile (pixel-unshuffled ×4) + blur → backbone features after `conv_first` (zero-init).
- `ExemplarAttention` after the backbone body: anchor-grid positions attend to exemplar tokens (4×4 per crop) and 4
  null tokens; bias = α · retrieval score of the position's token, −∞ otherwise; zero-init output.
- Refinement at native resolution (NAF blocks) on [backbone output, native input, blur↑] (zero-init).
- **Anchor lock** `y = Δ + U(a − down4(Δ))` with `down4` = area, `U` = bicubic + nearest-neighbour correction so that
  `down4(U(e)) = e` exactly → `down4(y) = a` to float precision. At inference the lock is applied again to the
  blended image (`finalize`), alternated with clamping to [0, 1].
- Ablation switches in the model config: `exemplars`, `native_input`, `anchor_lock`; data: `oracle`, anchor mix,
  `drop_colocated`, `tile`.

## 5. Training (`code/scripts/train.py`)

AdamW (2e-4; backbone 5e-5), warm-up 2k, cosine to 1e-6, 100k steps, batch 4 / GPU × 8, bf16, grad clip 1, EMA 0.999.
Loss: L1 on the validity mask + 0.1 × contextual loss (VGG19 relu3_1, `$UHDD_WEIGHTS/vgg/vgg19-dcbb9e9d.pth`) on
1024 defocused positions per tile. Validation every 2k steps on 256 fixed val tiles (EMA): PSNR, defocused-region
PSNR, ×4 PSNR, and the same for bicubic (reference). Model selection on full val images: launcher on `val_x4`.

## 6. Commands (secure)

```bash
# 1. sources + DP maps of train / val (once)
code/experiments/launch.sh setup inputs --sources train_x1,train_x4,val_x1,val_x4
# 2. real anchors on train / val (x4, whole image)
for s in train_x4 val_x4; do for m in bokehlicious drbnet restormer; do code/experiments/launch.sh deblur $s $m all; done; done
# 3. cache (+ M1, M2)
python code/scripts/build_train_cache.py --split train --gpus all --compare-pixels
python code/scripts/build_train_cache.py --split val --gpus all --compare-pixels
# 4. smoke run, then v0 with oracle and with retrieved exemplars
torchrun --nproc_per_node 8 code/scripts/train.py --config code/configs/train/v0.yaml --set optim.steps=300 --out <tmp>
torchrun --nproc_per_node 8 code/scripts/train.py --config code/configs/train/v0_oracle.yaml
torchrun --nproc_per_node 8 code/scripts/train.py --config code/configs/train/v0.yaml
# 5. full-image evaluation (val for selection, then test)
code/experiments/launch.sh upsample val_x4 bokehlicious ours_v0 all
code/experiments/launch.sh upsample ours_x4 bokehlicious ours_v0 all
```
Weights needed offline: HAT-L (`$UHDD_WEIGHTS/hat/…`, already used), DINOv2-L hub cache (shipped with VOSR), VGG19
(torchvision `vgg19-dcbb9e9d.pth`, to copy over).

## 7. Tests
`pytest -q code/tests/test_train.py`: synthetic scenes (in-focus textures A | B, defocused A) → cache (pixel features)
→ retrieval finds the same texture (> 80 %), M2 recall above chance → dataset shapes → 3 training steps + resume →
trained model as a registry model through the tiler; `down4(y) = anchor` on the blended image.
