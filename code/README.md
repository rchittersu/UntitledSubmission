# uhdd — native-resolution defocus deblurring evaluation

Evaluation harness for the paper (see `docs/evaluation.md`, `docs/baselines.md`). Written outside the secure env and
tested there only on CPU (`pytest code/tests`, 34 tests, incl. the multi-process path via
`--gpus cpu:N`). **GPU paths are untested until run in the secure env**; report any issue in the
handoff.

## Setup

```bash
pip install -r code/requirements.txt
export UHDD_REPOS=/path/to/third_party_repos     # git clones of baseline repos
export UHDD_WEIGHTS=/path/to/checkpoints
pytest -q code/tests
```

## Layout

| | |
|---|---|
| `uhdd/io.py` | 8/16-bit image I/O, input/target pairing by file stem, prefetching dataset |
| `uhdd/padding.py` | pad-to-multiple (reflect) + crop |
| `uhdd/tiling.py` | batched tiled inference with border-aware blending; returns the tile grid |
| `uhdd/models.py` | YAML model registry (`configs/models.yaml`) |
| `uhdd/parallel.py` | one process per GPU, prefetch readers, async writers |
| `uhdd/registration.py` | coarse-to-fine ECC target→input alignment |
| `uhdd/metrics/` | fidelity (PSNR, skimage-exact SSIM), optics (PSF-matched, high-band), pyiqa (LPIPS, DISTS, NR), consistency (drift, seams, statistics) |
| `scripts/` | `prepare_scales`, `register_pairs`, `run_model`, `evaluate`, `summarize` |

## Size constraints (multiples of 8/16/32/64/128)

Each model declares `multiple` in `configs/models.yaml` (U-Nets: 2^#downsamplings, e.g.
Restormer 8, NAFNet 16; windowed transformers: window × 2^#downsamplings, e.g. 128).
- Whole-image mode (`--tile 0`): the image is **reflect-padded** bottom/right to the next multiple,
  processed, and **cropped back**. Never resized: resizing changes the CoC size and pixel grid.
- Tiled mode: the tile size is rounded up to the multiple, so every tile is valid and all tiles
  have the same shape (batched, cuDNN autotuned). The last row/column of tiles is shifted
  inwards instead of padded; only images smaller than one tile are padded.
- Downscaling (`prepare_scales`) crops a remainder of < s pixels so the low-res grid stays
  exactly aligned with the native one (DPDD 6720×4480 is divisible by 2 and 4, nothing is cropped).

## Baselines

All entries live in `configs/models.yaml`. Each was checked on CPU by running the authors' own
preprocessing + model call next to ours on the same images (max abs difference reported).

| Config name | Method | Trained on | Weights source (→ `$UHDD_WEIGHTS/...`) | Extra deps | Check vs official code |
|---|---|---|---|---|---|
| `restormer_dpdd` | Restormer (CVPR'22) | DPDD | GitHub release `swz30/Restormer` v1.0 `single_image_defocus_deblurring.pth` → `restormer/` | einops | reproduces official demo outputs (≤1/255 on ≤21 px) |
| `bokehlicious_deblur` | Bokehlicious, RealDefocus deblur variant | RealDefocus | in repo: `checkpoints/defocus_deblur.zpaq` (`zpaq x`) → `bokehlicious/defocus_deblur.pt` | timm, zpaq | ≤ 7e-7 |
| `lakdnet_dpdd_l`, `lakdnet_dpdd_s` | LaKDNet (L / S) | DPDD | `lakdnet.mpi-inf.mpg.de/Weights/Defocus/train_on_dpdd_{l,s}/…pth` → `lakdnet/` | einops | ≤ 1.7e-6 |
| `drbnet_single` | DRBNet (CVPR'22) | LFDOF+DPDD | `python download_ckpts.py` in repo (gdrive id `1vGImev9LdagttXE_nN1gZGVstVTRVQHt`), unzip → `drbnet/ckpts/single/` | torchvision | ≤ 4.5e-7 |
| `ifan` | IFAN (CVPR'21) | DPDD | `github.com/jacobsparts/ifan-rs/releases/download/v0.1.0/IFAN.safetensors` → `ifan/` (bit-identical conversion; or official `checkpoints.zip` → `IFAN.pytorch`) | easydict, safetensors | ≤ 1.0e-6 (checkpoint: keep `module.Network.*`, drop training-only `reblurNet`) |
| `swinir_x4` | SwinIR-M x4 classical (upsampler) | DF2K | GitHub release `JingyunLiang/SwinIR` v0.0 `001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth` → `swinir/` | timm | exact (0) |
| `swinir_x4_real` | SwinIR-M x4 real-world GAN (upsampler) | BSRGAN degr. | same release, `003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth` → `swinir/` | timm | exact (0) |

Repos go to `$UHDD_REPOS/<Name>`: `swz30/Restormer`, `TimSeizinger/Bokehlicious`,
`lingyanruan/LaKDNet`, `lingyanruan/DRBNet`, `codeslake/IFAN`, `JingyunLiang/SwinIR`.

Early observation (demo images, ~0.2 MP, 128 px tiles / 32 overlap vs. whole image): tiling
alone changes outputs by 35–45 dB PSNR (DRBNet least, LaKDNet/IFAN/Restormer most), i.e. the
patch-wise gap already appears at low resolution.

Protocol details reproduced from each official test script: 8-bit inputs (`input_bits: 8`) for
Restormer/LaKDNet/DRBNet/IFAN; DRBNet works in [-1, 1]; IFAN returns a dict (`result`);
Bokehlicious needs full-image position maps (handled per tile) and the blurry input's f-number
(`av`, DPDD = 4). Official scripts that *crop* to a multiple (DRBNet 16, IFAN 8) are equivalent
to our padding on DPDD (1680×1120 is divisible by 16).

## Restormer setup and reproduction check

```bash
git clone https://github.com/swz30/Restormer $UHDD_REPOS/Restormer
mkdir -p $UHDD_WEIGHTS/restormer && wget -P $UHDD_WEIGHTS/restormer \
    https://github.com/swz30/Restormer/releases/download/v1.0/single_image_defocus_deblurring.pth

# official protocol: whole 1680x1120 images, 8-bit input (from config) and 8-bit targets
python code/scripts/run_model.py --model restormer_dpdd --inputs $OFF/inputC --out $R/official/restormer --gpus all
python code/scripts/evaluate.py --pred $R/official/restormer --targets $OFF/target --target-bits 8 \
    --metrics psnr,ssim,mae,lpips --tag official --gpus all
# expected (paper): PSNR 25.98, SSIM 0.811, MAE 0.038, LPIPS 0.178  -> must match to ~0.01
```
`$OFF` = official DPDD test set (Restormer's `Datasets/test/DPDD`: `inputC/`, `target/`, same file names).
Outputs are saved as 16-bit PNG; the tiny difference from Restormer's float evaluation is
< 0.001 dB.

## Running combinations: `run_matrix.py`

All comparisons are declared in one experiment file and run with one command:

```bash
export UHDD_DATA=/data UHDD_RESULTS=/results
python code/scripts/run_matrix.py code/experiments/dpdd_p1.yaml --dry-run      # print the plan
python code/scripts/run_matrix.py code/experiments/dpdd_p1.yaml --gpus all     # run (resumable)
python code/scripts/run_matrix.py code/experiments/dpdd_p1.yaml --only "restormer"  # subset
```

- A pipeline is a chain of steps (`deblur @ scale` → optional upsampler); `for:` grids expand
  templates (`{m}`, `{s}`, `{up}`, `{t}`) into all combinations.
- Step outputs are cached by their full chain, e.g. `steps/x4__restormer_dpdd@whole__swinir_x4@t256o32`,
  so shared prefixes run once (the x4 Restormer result feeds all three upsamplers).
- Every step uses the multi-GPU `run_model.py`; every final output the multi-GPU `evaluate.py`,
  at the pipeline's output scale (border crop scaled accordingly).
- Finished steps/evaluations are skipped; after changing metrics, change `eval.tag`.
- Outputs: `results/pipelines/<name>__<tag>.json` (metrics + `pipeline_time_s` summed over steps +
  peak memory) and `results/tables/<group>.md` — paste these into handoff reports.

`experiments/dpdd_p1.yaml` = P1 plan: resolution-gap table (5 models × x4/x2/x1, 512 tiles),
G1 native patch-wise (512 / 1024 tiles), G2 x4 whole-image deblur + {bicubic, SwinIR, SwinIR-real},
and the blurry input as reference: 41 pipelines / 41 unique steps.

## Pipeline (P1 tasks)

```bash
D=/data/dpdd_native   # inputs/ targets/ (same file stems)
R=/results

# 1. check which filter reproduces the official 1680x1120 release (compares only, writes nothing)
python code/scripts/prepare_scales.py --inputs $D/inputs --targets $D/targets --out $D --factors 4 \
    --reference /data/dpdd_official/test_c/source
# 2. register targets to inputs at native res, then build scales from the aligned set
#    (pass --filter chosen in step 1; existing outputs are skipped unless --overwrite)
python code/scripts/register_pairs.py --inputs $D/inputs --targets $D/targets --out $D/x1
python code/scripts/prepare_scales.py --inputs $D/inputs --targets $D/x1/targets --masks $D/x1/masks \
    --out $D --factors 2 4
#   -> $D/x1/{targets,masks}, $D/x2/{inputs,targets,masks}, $D/x4/...  (x1 inputs = $D/inputs)

# 3. (manual equivalent of run_matrix) resolution-gap table: same model at x4 / x2 / x1
for s in 4 2 1; do
  IN=$D/x$s/inputs; [ $s = 1 ] && IN=$D/inputs
  python code/scripts/run_model.py --model restormer_dpdd --inputs $IN --out $R/x$s/restormer_t512 \
      --tile 512 --overlap 64 --tile-batch 8 --gpus all
  python code/scripts/evaluate.py --pred $R/x$s/restormer_t512 --targets $D/x$s/targets \
      --masks $D/x$s/masks --scale $s --crop $((64 / s)) --tag dpdd --gpus all \
      --metrics psnr,ssim,mae,pm,lpips,dists,drift,seam,stats
done
python code/scripts/summarize.py $R/x*/restormer_t512/metrics_dpdd.json --cols psnr,pm_psnr,ssim,lpips,seam_ratio,time_s

# 4. low-res deblur + x4 upsampling (G2)
python code/scripts/run_model.py --model bicubic_x4 --inputs $R/x4/restormer_t512 --out $R/x1/restormer+bicubic
```

Paste `summarize.py` tables (and `register_pairs.py` shift statistics) into the handoff report.

## Native test set: `build_native_set.py` (preferred)

Builds x1/x2/x4 (+ masks, `build_report.csv`) from raw CR2, calibrated per capture to the official
1680×1120 rendering (crop (12,12) verified; polynomial colour/tone + fitted unsharp mask + smooth
local-tone field; translation registration; area downscaling). On the 37 indoor test pairs: 38.8 / 39.0 dB
agreement with the official images (inputs / targets, median), and the x4 level reproduces the standard
benchmark's blurry-vs-sharp PSNR within +0.33 dB. See the script docstring for all numbers.

```bash
python code/scripts/build_native_set.py --raw <cr2 dir or zip> --official-inputs <1680 inputs> \
    --official-targets <1680 targets named by input stem> [--pairs pairs.csv] --out $UHDD_DATA/dpdd_native_v2
```

## Native test set from raw (older global map) and optics checks

```bash
# CR2 -> native 16-bit PNG matching the official release's look: fit once on train scenes, apply to any scenes
python code/scripts/develop_raw.py fit --zip cr2.zip --official <official_train_source_dir> <official_train_target_dir> --map map.json
python code/scripts/develop_raw.py eval --zip cr2.zip --official <official_test_dirs> --map map.json      # held-out PSNR at 1/4 scale
python code/scripts/develop_raw.py develop --zip cr2.zip --map map.json --out <dir> --names <stem,stem,...>
# slanted-edge MTF on the raw green photosites (is the f/22 target diffraction-limited?)
python code/scripts/estimate_gt_mtf.py --zip cr2.zip --names <stem,...> --out mtf.json
```
Official DPDD pairs have *different* file stems (blurry `1P0A0917` <-> sharp `1P0A0916`); pair them by sorted order and
rename the targets to the blurry stems before using `run_model`/`evaluate`. Needs `rawpy`.

## Efficiency notes

- `--gpus all`: one process per GPU over a size-balanced shard of images; no inter-GPU sync.
- Per GPU: `--workers` reader processes prefetch images (kept 8/16-bit until on the GPU),
  `--write-threads` write PNGs asynchronously (compression level 1).
- `--tile-batch`: raise until the GPU is full; `--precision bf16/fp16` and `--compile` for speed
  (fp32 default for reported numbers; state precision in reports).
- Metrics run on the GPU (SSIM in float64 per channel, ~1.5 GB extra at 30 MP).
- CPU steps (`prepare_scales`, `register_pairs`) use process pools (`--procs`).

## Known gaps / TODO

- Semantic consistency metric (DINOv2-matched patch pairs), guided-upsampling baseline (G3),
  blur-level stratification from DP views, synthetic test set (T2).
- A recent DPDD-trained transformer/Mamba baseline with public weights is still to be added.
- Diffusion SR upsamplers (StableSR/SUPIR/OSEDiff) are not configured yet (large SD backbones).

## Protocol v3 and script baselines (2026-10-03)

Full description: `docs/evaluation.md` (metrics, how to run) and `docs/baselines.md` (every baseline).
- New metrics in `evaluate.py`: `percbins` (LPIPS/DISTS per DP blur bin), `noharm` (needs `--inputs`), `apsnr`
  (tile-aligned PSNR), `msres` (resolution sweep). Matrix: `code/experiments/dpdd_eval_v3.yaml` (tag `dpdd3`).
- Training-free baselines: `scripts/fuse_baselines.py --method composite|multiscale|guided|detail|exemplar`
  (`--tune` on val). Reference-based SR: `scripts/refsr_baseline.py` (DATSR, mmcv-free via `uhdd/adapters/datsr.py`).
  Both: `code/experiments/run_baselines_bd.sh <anchor model>`.
- New registry models: `hat_l_x4`, `hat_x4_real` (basicsr-free adapter), `osediff_x4` (CUDA only), `restormer_dpdd_tlc`.
- Tiling defaults per method from its paper (registry `paper_input` / `paper_tile`, `uhdd.tiling.default_tiling`);
  overlap = tile / 8. `run_model.py` without `--tile` and `run_matrix.py` steps without `tile` use them.
- Helpers: `scripts/pair_official.py` (name-paired official splits), `scripts/vae_ceiling.py` (variant B backbone).
- Manual anchor × upsampler runs (incl. S3Diff, VOSR 2.0 in their own environments): `scripts/launch_sr.py` (shell wrapper + setup: `experiments/launch_sr.sh help`)
  (`--list`, `--dry-run`, `--eval`, `--summary`); see `docs/baselines.md` "Manual SR launcher".
