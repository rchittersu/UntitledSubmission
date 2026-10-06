# Secure-env runbook — handoff 5: training data, cache, validation of the setup, first v0 runs

**One runbook per handoff.** Replaces the handoff-4 runbook (in git history). Your handoff-4 patch was applied and
reviewed outside (`handoff/LOG.md`, `docs/evaluation.md` §7.8).

Audience: the Claude agent in the secure environment. **The user runs the steps by hand with
`code/experiments/train.sh`** (`train.sh help` lists them in order) and watches them (`train.sh status`, `watch`,
`log`); you check each step's outputs against its pass criterion, as in handoff 4. Do not replace the script calls with
your own command lines; if a step needs different settings, give the user the `train.sh … --set k=v` line.
Read first: `docs/training.md` (canonical: data flow, cache, sample, network, trainer, commands), `plan/method_plan.md`
§4 / §8b (design and why), `Template/sec/3_method.tex` + Fig. 2 (`fig/method.tex`).

## Standing rules
- This handoff is the first one **with training**. The test set (76) is **never** used for training, tuning or model
  selection: selection on val (74), test only at the end of §8.
- Evaluation protocol unchanged (dpdd4, frozen). Same rules as always: outcome of every step, deviations table, no
  internal paths / hostnames / proprietary names, hand off early if blocked.
- **Validate before you scale.** Each step below has a pass criterion; if one fails, investigate, report, and do not
  start the 100k-step runs on a broken setup. A short, well-diagnosed report beats a long training run on a bug.
- Code fixes are welcome in `code/` (with a CPU test where possible); describe each in the report.

## What changed outside
- Results layout gains train / val sources: `train_x1`, `train_x4`, `val_x1`, `val_x4`, per-split DP maps
  (`inputs/{train,val}_dp_maps`), `cache/<split>/`, `train/<name>/` (`code/uhdd/layout.py` SOURCES / SPLITS). If the
  secure folder names differ from the defaults, put the real relative paths in `$UHDD_DATA/uhdd_sources.json`
  (format: `docs/training.md` §1) — do not edit `layout.py` with internal names.
- New code: `code/uhdd/train/` (memory, data, sim_anchor, losses), `code/uhdd/net/ours.py` (v0), `code/scripts/
  build_train_cache.py`, `code/scripts/train.py`, configs `code/configs/train/{v0,v0_oracle,tiny}.yaml`, registry model
  `ours_v0` (adapter `code/uhdd/adapters/ours.py`, aux inputs native + DP passed by the launcher).
- `uhdd/dualpixel.py`: `load_dp_input` (network DP input: |d| box 3 + confidence).
- **`code/experiments/train.sh`**: one human-readable command per step (inputs, check, anchors, cache, step0, overfit,
  smoke, start / stop / status / watch / log, eval); background runs, resume with the run's saved settings;
  `code/scripts/train_status.py` prints progress, ETA, loss and the val table (vs bicubic, vs step 0).
- `train.py` validates at step 0 (`val.at_start`, default on): the untrained model is HAT-L ×4 + anchor lock.
  It now expands `${UHDD_*}` in config values (before: the default cache / VGG paths were taken literally) and
  prints the parameter count.

---

## 0. Update and test
```bash
git pull && git checkout -b handoff/h5        # base = origin/main
pip install -r code/requirements.txt
pytest -q code/tests                          # expect 83 passed, 1 skipped outside; here the HAT test should RUN
```
`test_sft_on_hat_arch` skips outside (no `$UHDD_REPOS/HAT`); in the secure env it must pass (84 passed). Report if not.

## 1. Train / val inputs and DP maps (V1)
```bash
code/experiments/train.sh inputs
code/experiments/launch.sh list
```
Pass: 350 train / 74 val scenes; inputs / targets / masks / DP maps carry the same names; ×4 sources are native / 4;
the manifest merges (the test sources are still listed). **No overlap** of scene names (or near-identical thumbnails)
between train, val and test — check and report explicitly.
DP map check: for 3 train and 3 val scenes, overlay |d| (as `blur.npy` will see it, `load_blur_map(..., win=15)`) on
the input and describe: in-focus regions low, defocused high, orientation / registration correct (no transpose,
flip, offset), units (DP px at 1680; typical range). Give the per-scene 5/50/95th percentiles of |d| and of the
confidence channel. Same check for a test scene against the handoff-2/3 DP maps (must be identical).

## 2. Offline weights (V2)
- DINOv2-L: `$UHDD_WEIGHTS/vosr/torch_cache/facebookresearch_dinov2_main` + `checkpoints/dinov2_vitl14_pretrain.pth`
  (shipped with VOSR). Load it with `uhdd.train.memory.Features("dinov2", hub_dir=...)` on one anchor; report token
  grid and time.
- VGG19: `$UHDD_WEIGHTS/vgg/vgg19-dcbb9e9d.pth` (torchvision). If it is missing, ask the user to copy it in; do not
  download from inside unless that is allowed. `uhdd.train.losses.VGGFeatures` must load it without network.
- HAT-L ×4: the registry weights already used for the baselines.
`code/experiments/train.sh check` loads all three (plus GPUs, inputs, caches) and prints ok / FAIL per item.
Pass: all ok with networking off.

## 3. Real anchors on train / val (V3)
```bash
code/experiments/train.sh anchors all       # bokehlicious, drbnet, restormer on train_x4 and val_x4, scored, then a summary
```
Metrics are computed by the launcher as for test. Report the ×4 PSNR / SSIM of each anchor on **train vs val vs
test (ours_x4, handoff 4)**. Expected: DRBNet / Restormer (trained on DPDD) higher on train than val/test — that is the
train/test gap the anchor mix must cover; Bokehlicious (not trained on DPDD) should be similar on all three. A large
train-val gap for Bokehlicious would mean something else is wrong (data, names).

## 4. Cache + M1 / M2 (V4)
```bash
code/experiments/train.sh cache val all --limit 4                # smoke first (4 scenes)
code/experiments/train.sh cache val all --overwrite
code/experiments/train.sh cache train all
```
Report: time per scene, size per scene and total (expected ≈ 155 GB train + val), the printed manifest summary.
- **M1** (pixel share per blur bin b0–b3, train and val): tells how much of the image is copy / deconvolve /
  exemplar regime.
- **M2**: keys per scene, coverage of defocused tokens, recall@16 of the oracle top-1 vs chance, DINOv2 vs pixel
  features. Pass: DINOv2 recall clearly above chance and above pixels. If DINOv2 is not better than pixels, say so —
  it changes a paper claim.
- **Visual check of retrieval** (most important): for 3 val scenes, pick 5 defocused query tokens each and describe
  the anchor patch, the top-4 retrieved native exemplars and the oracle top-1 (same material? same scale? plausible
  source of detail?). Describe in words; put the token coordinates in the report so outside can refer to them.
  Note failure modes (e.g. retrieval of edges instead of texture, wrong scale, sky / flat keys).

## 5. Step-0 check (V5)
```bash
code/experiments/train.sh step0             # -> train/_step0 (val.csv, printed table)
```
`val.csv` step 0 is the untrained model = HAT-L ×4 + anchor lock on Bokehlicious anchors. Pass: `psnr_x4` ≈ the
anchor itself (lock exact → ×4 PSNR of the anchor vs target), `psnr` within ~0.1 dB of HAT-L on the same tiles
(and ≥ `psnr_bicubic` is NOT required — handoff 3 saw HAT-L below bicubic on some anchors; report what you get).
Also report the parameter count (total / trainable) printed at start.

## 6. Overfit (V6)
```bash
code/experiments/train.sh overfit auto      # 3k steps on the 2 most defocused train scenes (M1), val on the same scenes
``` Pass: train loss falls steadily, val PSNR on the same
scenes rises well above step 0 (several dB on defocused tiles), no NaN / inf, grad norm sane. If the loss does not
move, check the zero-init paths actually receive gradient (`grad_norm`), then the LR groups.

## 7. Smoke run: throughput (V7)
```bash
code/experiments/train.sh smoke                         # 300 steps; status shows it/s, peak memory, projected ETA
code/experiments/train.sh smoke --set workers=12        # dataloader check
```
Report it/s, peak GPU memory, GPU utilisation (`nvidia-smi`), whether the dataloader is the bottleneck (workers 4 / 8 / 12),
and the projected wall time of 100k steps. If memory does not fit batch 4 at 512, report and use the largest batch
that fits (record it); do not shrink the tile. If 100k steps take more than ~3 days, propose a step count.

## 8. Training runs (user launches; check them)
In this order — the oracle run is the go / no-go for the exemplar path:
```bash
code/experiments/train.sh start v0_oracle    # background; then: train.sh watch v0_oracle
code/experiments/train.sh start v0           # after v0_oracle (or on other GPUs: CUDA_VISIBLE_DEVICES=... train.sh start v0 4)
```
While they run: `train.sh status` / `status v0` (the val table is `val.csv`: EMA, every 2k): PSNR / defocused PSNR / ×4 PSNR, each vs bicubic of the same
anchor. Report curves as tables (step, val metrics) for both runs.
Full-image evaluation (registry `ours_v0` / `ours_v0_oracle` read `dpdd/train/<name>/model_ema.pt`):
```bash
code/experiments/train.sh eval v0 val_x4 bokehlicious all --scenes <3 val scenes>   # V8: full-image smoke
code/experiments/train.sh eval v0 val_x4 bokehlicious all
code/experiments/train.sh eval v0_oracle val_x4 bokehlicious all
code/experiments/train.sh eval v0 ours_x4 bokehlicious all       # test, once, at the end
code/experiments/train.sh eval v0 ours_x4 drbnet all             # plug-in, other anchor
```
V8 pass: no seams, `down4(output) = anchor` (the run's `meta.json` / a check by hand: ×4 PSNR vs the anchor > 60 dB),
time and memory per 30 MP image. Look at the images: tile seams, exemplar copy artifacts (repeated patches), in-focus
regions unchanged vs input.
V9: v0_oracle vs v0 on val (tiles + full images). Interpretation for the report (do not overclaim): oracle ≫ v0 →
retrieval is the bottleneck; oracle ≈ v0 ≈ no gain over HAT-L → the exemplar path is not used (check the attention to
the null tokens); both ≫ HAT-L → go.

If time remains (lower priority): the ablation without exemplars on val only —
`code/experiments/train.sh start v0 --set name=v0_noex model.exemplars=false` (its own run folder `train/v0_noex`; compare on the val tiles in `train.sh status v0_noex`).

## 9. Handoff
Report `handoff/from_secure/<date>_h5.md` (template in CLAUDE.md): steps 0–8 with pass / fail per V-check, the
deviations table, M1 / M2 tables, anchor table (train / val / test), retrieval descriptions, throughput, training
curves (tables), val / test results with CIs, visual notes. Update `docs/training.md` §0 status and `docs/evaluation.md`
results if test rows exist. Nothing from the cache or checkpoints leaves the environment. `handoff/make_patch.sh`.
