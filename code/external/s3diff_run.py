#!/usr/bin/env python
"""S3Diff inference on a folder, run INSIDE S3Diff's own Python environment (its pinned torch/diffusers versions).

Reproduces the official loop of S3Diff/src/inference_s3diff.py (github.com/ArcticHare105/S3Diff) step by step:
bilinear x4 upsampling of the LR image, [-1, 1], reflect-pad to a multiple of 64, degradation score from DEResNet
on the LR image, one forward pass of the tiled model (official latent tiling: 96 latent px, overlap 32; VAE encoder
tiles 1024, decoder 224), crop, wavelet colour fix to the upsampled LR. Differences: no metric computation at the
end (the official script scores every output with pyiqa NR metrics), no accelerate wrapper, per-image timing in
OUT/meta.json, optional --only.

  $S3DIFF_PY code/external/s3diff_run.py --repo $S3DIFF_REPO --inputs LR8_DIR --out OUT \
      --sd-path SD_TURBO_DIR --pretrained S3DIFF_PKL [--de-net $S3DIFF_REPO/assets/mm-realsr/de_net.pth]
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--inputs", required=True, help="folder of LR PNGs (8-bit RGB)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sd-path", required=True, help="local snapshot of stabilityai/sd-turbo")
    ap.add_argument("--pretrained", required=True, help="s3diff.pkl (huggingface.co/zhangap/S3Diff)")
    ap.add_argument("--de-net", default=None, help="default: <repo>/assets/mm-realsr/de_net.pth")
    ap.add_argument("--align", default="wavelet", choices=["wavelet", "adain", "nofix"])
    ap.add_argument("--only", help="comma-separated names")
    a = ap.parse_args()

    repo = Path(a.repo).resolve()
    sys.path[:0] = [str(repo / "src"), str(repo)]
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms
    from de_net import DEResNet
    from s3diff_tile import S3Diff
    from my_utils.testing_utils import parse_args_paired_testing
    from utils.wavelet_color import wavelet_color_fix, adain_color_fix

    de_net = a.de_net or str(repo / "assets" / "mm-realsr" / "de_net.pth")
    # the official argument parser supplies every model default (LoRA ranks, tiling sizes, prompts)
    sys.argv = [sys.argv[0], "--sd_path", a.sd_path, "--de_net_path", de_net, "--pretrained_path", a.pretrained,
                "--align_method", a.align, "--output_dir", a.out]
    args = parse_args_paired_testing()
    sf = 4

    net_sr = S3Diff(lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae, sd_path=a.sd_path,
                    pretrained_path=a.pretrained, args=args)
    net_sr.set_eval()
    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model(de_net)
    net_sr, net_de = net_sr.cuda(), net_de.cuda().eval()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(a.inputs).glob("*.png"))
    if a.only:
        keep = set(a.only.split(","))
        files = [p for p in files if p.stem in keep]
    meta = {"images": {}, "summary": {"method": "s3diff", "align": a.align, "official": "src/inference_s3diff.py",
                                      "latent_tiled_size": args.latent_tiled_size,
                                      "latent_tiled_overlap": args.latent_tiled_overlap}}
    for p in files:
        im_lr = transforms.ToTensor()(Image.open(p).convert("RGB"))[None].cuda().float()
        t0 = time.time()
        h, w = im_lr.shape[2:]
        im_up = F.interpolate(im_lr, size=(h * sf, w * sf), mode="bilinear", align_corners=False).contiguous()
        x = torch.clamp(im_up * 2 - 1.0, -1.0, 1.0)
        H, W = x.shape[2:]
        x = F.pad(x, pad=(0, math.ceil(W / 64) * 64 - W, 0, math.ceil(H / 64) * 64 - H), mode="reflect")
        with torch.no_grad():
            deg = net_de(im_lr)
            y = net_sr(x, deg, pos_prompt=[args.pos_prompt], neg_prompt=[args.neg_prompt])[:, :, :H, :W]
            y = (y * 0.5 + 0.5).cpu()
        torch.cuda.synchronize()
        pil = transforms.ToPILImage()(y[0].clamp(0, 1))
        if a.align != "nofix":
            ref = transforms.ToPILImage()(im_up[0].cpu())
            pil = wavelet_color_fix(pil, ref) if a.align == "wavelet" else adain_color_fix(pil, ref)
        pil.save(out / f"{p.stem}.png")
        dt = time.time() - t0
        meta["images"][p.stem] = {"time_s": round(dt, 3), "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3)}
        print(f"{p.stem}: {dt:.1f} s", flush=True)
        (out / f".meta_{os.getpid()}.json").write_text(json.dumps(meta, indent=1))
    merged = {"images": {}, "summary": meta["summary"]}
    for f in sorted(out.glob(".meta_*.json")):
        merged["images"].update(json.loads(f.read_text())["images"])
    (out / "meta.json").write_text(json.dumps(merged, indent=1))


if __name__ == "__main__":
    main()
