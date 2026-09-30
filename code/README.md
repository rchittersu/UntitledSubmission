# uhdd — native-resolution defocus deblurring evaluation

Evaluation harness for the paper (see `plan/evaluation.md`). Written outside the secure env and
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

## Pipeline (P1 tasks of plan/evaluation.md)

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

# 3. resolution-gap table: same model at x4 / x2 / x1, tiled at the training crop size
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
- Only Restormer is configured (architecture checked against its official config; loading +
  inference tested on CPU with random weights only); other baselines are templates [V].
