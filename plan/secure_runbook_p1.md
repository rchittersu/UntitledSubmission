# Secure-env runbook — P1 evaluation (DPDD at native resolution)

Audience: the Claude agent in the secure environment. Read `CLAUDE.md` (handoff protocol) and
`code/README.md` first. Goal of this round: get the evaluation pipeline running on GPUs and
produce the first tables of `plan/evaluation.md` (resolution gap, G1, G2).

Everything in `code/` was written outside and tested **on CPU only** (36 tests). The GPU paths
(multi-GPU spawn, CUDA memory, AMP) have never run. Expect small fixes; make them in `code/`,
add a CPU test when possible, and send them back in the patch.

Work top to bottom. **After each numbered step, note the outcome** (it goes into the report).
If a step blocks, stop, write the report with what you have, and hand off.

---

## 0. Environment

```bash
git pull                                  # base for the handoff patch
git checkout -b handoff/p1-eval
python -m venv .venv && source .venv/bin/activate      # or conda; Python >= 3.10
pip install -r code/requirements.txt       # torch (CUDA build!), opencv, scipy, pyiqa, timm, einops, easydict, torchvision
pytest -q code/tests                       # expect: 36 passed
python -c "import torch; print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

Environment variables used everywhere (never hard-code paths in committed files):

```bash
export UHDD_REPOS=<dir for third-party repos>
export UHDD_WEIGHTS=<dir for checkpoints>
export UHDD_DATA=<dir for datasets>
export UHDD_RESULTS=<dir for outputs>
```

Also needed: `zpaq` (system package) for the Bokehlicious checkpoint.

## 1. Baseline repos and weights

```bash
cd $UHDD_REPOS
git clone https://github.com/swz30/Restormer
git clone https://github.com/TimSeizinger/Bokehlicious
git clone https://github.com/lingyanruan/LaKDNet
git clone https://github.com/lingyanruan/DRBNet
git clone https://github.com/codeslake/IFAN
git clone https://github.com/JingyunLiang/SwinIR

cd $UHDD_WEIGHTS
mkdir -p restormer bokehlicious lakdnet drbnet ifan swinir
wget -P restormer https://github.com/swz30/Restormer/releases/download/v1.0/single_image_defocus_deblurring.pth
(cd bokehlicious && zpaq x $UHDD_REPOS/Bokehlicious/checkpoints/defocus_deblur.zpaq)
wget -P lakdnet https://lakdnet.mpi-inf.mpg.de/Weights/Defocus/train_on_dpdd_l/train_on_dpdd_l.pth
wget -P lakdnet https://lakdnet.mpi-inf.mpg.de/Weights/Defocus/train_on_dpdd_s/train_on_dpdd_s.pth
(cd drbnet && gdown 1vGImev9LdagttXE_nN1gZGVstVTRVQHt -O ckpts.zip && unzip -q ckpts.zip)   # -> drbnet/ckpts/single/...
(cd ifan && wget -O checkpoints.zip 'https://www.dropbox.com/s/qohhmr9p81u0syi/checkpoints.zip?dl=1' \
         && unzip -q checkpoints.zip IFAN.pytorch)                                            # 961 MB zip, only IFAN.pytorch needed
wget -P swinir https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth
wget -P swinir https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth
```

If the secure env has no internet, transfer these files in; verify with `sha256sum` (checksums of
the files verified outside, relative to `$UHDD_WEIGHTS`):

```
7dce451f33f8f5e0faf7c4e3996e5dcc1bd425ecd1ada99b0f9750e490fd4c9e  restormer/single_image_defocus_deblurring.pth
ef1ce51c4cafd82b63d91048f1589048308cfbae98470e07d0fc16614d255557  bokehlicious/defocus_deblur.pt
8f954fe18c4f8f4f03302cf6adaccb586faf5816160c407ca3a1a6aa5f3121f6  lakdnet/train_on_dpdd_l.pth
5f32974e06605ef7723cd2d8bdf22d586626b8de82eb620709744b655ec374b1  lakdnet/train_on_dpdd_s.pth
7e711340686e9ee728829d60e443d2ee28b5a45f6c0415632c22b7f099712382  drbnet/ckpts/single/single_image_defocus_deblurring.pth
9770cb367301330301badb4d0c52eabae6e685ca989d18ea7cc873ea0f6768ca  ifan/IFAN.pytorch
4e78e33f22c1aa8a773db0cf4a7381bae97c2362c717f155439ebc690cbd9215  swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth
b9afb61e65e04eb7f8aba5095d070bbe9af28df76acd0c9405aeb33b814bcfc6  swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth
```

Check the integrations against the official code paths (now on GPU hardware):

```bash
python code/scripts/check_against_official.py lakdnet_dpdd_l lakdnet_dpdd_s drbnet_single ifan swinir_x4 swinir_x4_real
```

Expected: `max|ours-ref|` ≈ 1e-6 for every line (outside, CPU: ≤ 1.7e-6; SwinIR exactly 0).
Report the printed lines.

## 2. Data layout

Native DPDD test pairs (76 scenes, 6720×4480, 16-bit), matched by identical file stems:

```
$UHDD_DATA/dpdd_native/inputs/<name>.png      # f/4 blurry (the "inputs" folder you have)
$UHDD_DATA/dpdd_native/targets/<name>.png     # f/22 sharp (the "targets" folder you have)
```

Also needed for reproduction checks: the official 1680×1120 test set (any of DPDD's
`test_c/source` + `test_c/target`, or Restormer's `Datasets/test/DPDD/inputC` + `target`),
at `$UHDD_DATA/dpdd_official/{inputs,targets}` with the same file stems as the native set
(rename/symlink if the release uses different names; report what you did).

Report: number of pairs, resolution(s), bit depth, any portrait-orientation images.

## 3. How was the official 1680×1120 made?

```bash
python code/scripts/prepare_scales.py --inputs $UHDD_DATA/dpdd_native/inputs \
    --out /tmp/unused --factors 4 --reference $UHDD_DATA/dpdd_official/inputs --reference-n 20
```

Report the six `filter=... PSNR` lines. Decision rule: a filter with > ~60 dB is the official
pipeline → use it as `--filter` below. If all are < ~45 dB, use `area` (our default) and say so.

## 4. Registration and scaled sets

```bash
D=$UHDD_DATA/dpdd_native
python code/scripts/register_pairs.py --inputs $D/inputs --targets $D/targets --out $D/x1 --procs 8
python code/scripts/prepare_scales.py --inputs $D/inputs --targets $D/x1/targets --masks $D/x1/masks \
    --out $D --factors 2 4 --filter <from step 3>
```

Report the final line of `register_pairs.py` (median / p90 / max corner shift in px, #warped)
and the few largest rows of `$D/x1/registration.csv` (name, shift, ecc). If any shift is > 10 px
or ECC < 0.9, flag the image; do not drop it silently.

Result layout (what `code/experiments/dpdd_p1.yaml` expects):
`$D/inputs`, `$D/x1/{targets,masks}`, `$D/x2/{inputs,targets,masks}`, `$D/x4/{inputs,targets,masks}`.

## 5. Reproduce published Restormer numbers (official 1680×1120 set)

```bash
R=$UHDD_RESULTS
python code/scripts/run_model.py --model restormer_dpdd --inputs $UHDD_DATA/dpdd_official/inputs \
    --out $R/official/restormer --gpus all
python code/scripts/evaluate.py --pred $R/official/restormer --targets $UHDD_DATA/dpdd_official/targets \
    --target-bits 8 --metrics psnr,ssim,mae,lpips --tag official --gpus all
```

Expected (Restormer paper, DPDD single-image, combined): PSNR 25.98, SSIM 0.811, MAE 0.038,
LPIPS 0.178 — must match within ~0.01. If it does not, **stop here**: the metric or data path is
wrong, and everything downstream would be too. Report ours vs. published. Optionally repeat for
`lakdnet_dpdd_l` (README: 26.25 / 0.813) and `ifan`.

## 6. Smoke test of the matrix

```bash
python code/scripts/run_matrix.py code/experiments/dpdd_p1.yaml --dry-run | head -30
python code/scripts/run_matrix.py code/experiments/dpdd_p1.yaml --only "^restormer_dpdd" --gpus all
cat $UHDD_RESULTS/dpdd_p1/tables/*.md
```

Check: runs on all GPUs, no OOM, sensible numbers (blurry input row < deblurred rows at x4),
per-image time and peak memory in the tables. On OOM: lower `tile_batch`, or set a tile for the
failing step in the YAML (e.g. `tile: 1024` for whole-image x4 steps); report what you changed.

## 7. Full P1 matrix

```bash
python code/scripts/run_matrix.py code/experiments/dpdd_p1.yaml --gpus all
python code/scripts/run_matrix.py code/experiments/dpdd_p1.yaml --only "input" --gpus all   # reference row
```

41 pipelines. The runner is resumable; rerun the same command after any interruption.

## 8. Handoff

1. Write `handoff/from_secure/<YYYY-MM-DD>_p1-eval.md` using the template in `CLAUDE.md`. Include:
   - environment: GPU type/count, torch/CUDA versions, total wall time;
   - step 0: test result; step 1: `check_against_official` lines;
   - step 2: data facts; step 3: six filter lines + chosen filter;
   - step 4: registration statistics + flagged images;
   - step 5: Restormer (and optional others) ours vs. published;
   - steps 6–7: paste `results/tables/*.md` verbatim (t1_resolution_gap, g1_native_patchwise,
     g2_lowres_upsample, references);
   - every code change you made (what and why) and every deviation from this runbook;
   - observations in words: visible seams, color shifts, over-sharpening, failure cases
     (name the image and region), anything surprising in the metrics.
2. Commit the report and any `code/` fixes on the branch (no data, no results folders, no images).
3. `handoff/make_patch.sh` → copy the printed `.patch` file out verbatim.

If a step fails and you cannot fix it, hand off early with a partial report; do not wait for
the full matrix.
