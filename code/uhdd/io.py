"""Image I/O and input/target folder pairing.

Images are kept in their native integer dtype (uint8/uint16) on the CPU side, which halves
(or quarters) host memory and inter-process transfer compared with float32 for 30 MP images.
Conversion to float in [0, 1] happens on the GPU (`to_tensor`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

IMG_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".webp"}

# Let OpenCV use its own threads only when we are not already parallel over processes.
cv2.setNumThreads(int(os.environ.get("UHDD_CV2_THREADS", "0")))


def list_images(folder: str | Path) -> dict[str, Path]:
    """Map file stem -> path for all images directly inside `folder`."""
    folder = Path(folder)
    out: dict[str, Path] = {}
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            if p.stem in out:
                raise ValueError(f"duplicate stem '{p.stem}' in {folder}")
            out[p.stem] = p
    if not out:
        raise FileNotFoundError(f"no images in {folder}")
    return out


@dataclass(frozen=True)
class Pair:
    name: str
    input: Path
    target: Path | None


def pair_folders(inputs: str | Path, targets: str | Path | None, strict: bool = True) -> list[Pair]:
    """Pair images by file stem. `targets=None` gives unpaired (no-GT) items.

    strict=True raises if any input has no target or vice versa (lists the offenders).
    """
    ins = list_images(inputs)
    if targets is None:
        return [Pair(k, v, None) for k, v in ins.items()]
    tgs = list_images(targets)
    only_in, only_tg = sorted(ins.keys() - tgs.keys()), sorted(tgs.keys() - ins.keys())
    if strict and (only_in or only_tg):
        raise ValueError(
            f"unpaired files: {len(only_in)} inputs without target {only_in[:10]}, "
            f"{len(only_tg)} targets without input {only_tg[:10]}"
        )
    return [Pair(k, ins[k], tgs[k]) for k in sorted(ins.keys() & tgs.keys())]


def read_image(path: str | Path) -> np.ndarray:
    """Read as HxWx3 RGB in native dtype (uint8 or uint16)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {path}")
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    img = img[..., :3]
    if img.dtype not in (np.uint8, np.uint16):
        raise TypeError(f"{path}: unsupported dtype {img.dtype}")
    return np.ascontiguousarray(img[..., ::-1])  # BGR -> RGB


def write_image(path: str | Path, img: np.ndarray, png_compression: int = 1) -> None:
    """Write RGB uint8/uint16 HxWx3. Low PNG compression: 30 MP writes are I/O-bound."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), np.ascontiguousarray(img[..., ::-1]),
                     [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
    if not ok:
        raise IOError(f"cannot write {path}")


def dtype_max(dtype) -> float:
    return float(np.iinfo(dtype).max)


def to_transfer(img: np.ndarray) -> torch.Tensor:
    """Zero-copy CPU tensor for cheap shared-memory / PCIe transfer.

    uint16 is reinterpreted as int16 (torch lacks full uint16 support); `to_tensor` undoes it.
    """
    if img.dtype == np.uint16:
        return torch.from_numpy(img.view(np.int16))
    return torch.from_numpy(img)


def to_tensor(img: np.ndarray | torch.Tensor, device: torch.device | str,
              dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """HxWx3 image (numpy uint8/uint16, or a `to_transfer` tensor) -> 1x3xHxW float [0,1] on device."""
    t = to_transfer(img) if isinstance(img, np.ndarray) else img
    t = t.to(device, non_blocking=True)
    if t.dtype == torch.int16:  # reinterpreted uint16
        t, maxv = t.to(torch.int32).bitwise_and_(0xFFFF), 65535.0
    elif t.dtype == torch.uint8:
        maxv = 255.0
    else:
        raise TypeError(f"unexpected transfer dtype {t.dtype}")
    return t.permute(2, 0, 1).unsqueeze(0).to(dtype).div_(maxv)


def to_numpy(x: torch.Tensor, out_dtype=np.uint16) -> np.ndarray:
    """1x3xHxW float [0,1] -> HxWx3 integer array (rounded, clamped)."""
    maxv = dtype_max(out_dtype)
    x = x.detach().squeeze(0).clamp(0, 1).mul(maxv).round_().permute(1, 2, 0).to(torch.int32)
    if out_dtype == np.uint16:  # transfer 2 bytes/px: int32 -> int16 wraps, viewed back as uint16
        return x.to(torch.int16).contiguous().cpu().numpy().view(np.uint16)
    return x.to(torch.uint8).contiguous().cpu().numpy()


def read_mask(path: str | Path) -> np.ndarray:
    """Validity mask (HxW bool) stored as 8-bit PNG, 255 = valid."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f"cannot read {path}")
    return m > 127


class ImageDataset(torch.utils.data.Dataset):
    """Reads a list of records {"name": str, <key>: path | None, ...}.

    Image keys are returned as `to_transfer` tensors (native bit depth), key "mask" as a bool
    tensor, "name" and other non-path fields unchanged. Runs inside DataLoader workers.
    """

    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        out = {}
        for k, v in self.records[i].items():
            if isinstance(v, Path):
                out[k] = torch.from_numpy(read_mask(v)) if k == "mask" else to_transfer(read_image(v))
            else:
                out[k] = v
        return out
