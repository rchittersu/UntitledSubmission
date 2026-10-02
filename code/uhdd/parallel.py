"""Multi-GPU / multi-process helpers.

Evaluation work is embarrassingly parallel over images, so we use one process per GPU with no
inter-GPU communication (no NCCL / torchrun needed). Each process takes a strided shard of the
image list (sorted by file size so shards are balanced), prefetches reads with DataLoader
workers, and writes outputs from a thread pool so GPU compute overlaps disk I/O.
"""
from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Sequence

import torch
import torch.multiprocessing as mp


def parse_gpus(spec: str) -> list[int]:
    """'all' -> every visible GPU, 'cpu' or '' -> [], '0,2' -> [0, 2],
    'cpu:N' -> N CPU processes (exercises the multi-process path without GPUs), 'mps' -> Apple GPU."""
    spec = (spec or "").strip().lower()
    if spec in ("", "cpu", "none"):
        return []
    if spec == "mps":
        return [-2]
    if spec == "all" and not torch.cuda.is_available() and torch.backends.mps.is_available():
        return [-2]
    if spec.startswith("cpu:"):
        return [-1] * int(spec[4:])
    if spec == "all":
        return list(range(torch.cuda.device_count()))
    return [int(g) for g in spec.split(",")]


def shard(items: Sequence, rank: int, world: int, key: Callable | None = None) -> list:
    """Strided shard; with `key` (e.g. file size) items are sorted descending first to balance load."""
    items = sorted(items, key=key, reverse=True) if key else list(items)
    return items[rank::world]


def _entry(rank: int, gpus: list[int], worker: Callable, args: tuple) -> None:
    if gpus[rank] == -2:
        return worker(rank, len(gpus), torch.device("mps"), *args)
    if gpus[rank] < 0:
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // len(gpus)))
        return worker(rank, len(gpus), torch.device("cpu"), *args)
    torch.cuda.set_device(gpus[rank])
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False  # keep fp32 numerics exact unless asked
    worker(rank, len(gpus), torch.device(f"cuda:{gpus[rank]}"), *args)


def launch(worker: Callable, gpus: list[int], *args: Any) -> None:
    """Run worker(rank, world, device, *args) once per GPU (or once on CPU)."""
    if not gpus:
        torch.set_num_threads(os.cpu_count() or 1)
        worker(0, 1, torch.device("cpu"), *args)
    elif len(gpus) == 1:
        _entry(0, gpus, worker, args)
    elif -2 in gpus:
        raise ValueError("mps runs as a single process")
    else:
        mp.spawn(_entry, args=(gpus, worker, args), nprocs=len(gpus), join=True)


class AsyncWriter:
    """Bounded thread pool for writing results (cv2.imwrite releases the GIL)."""

    def __init__(self, workers: int = 4, max_pending: int = 8):
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.pending: list[Future] = []
        self.max_pending = max_pending

    def submit(self, fn: Callable, *args) -> None:
        self.pending = [f for f in self.pending if not f.done()]
        while len(self.pending) >= self.max_pending:
            self.pending.pop(0).result()
        self.pending.append(self.pool.submit(fn, *args))

    def close(self) -> None:
        for f in self.pending:
            f.result()  # re-raise write errors
        self.pool.shutdown()


def loader(dataset, workers: int) -> torch.utils.data.DataLoader:
    """Prefetching loader yielding single items (images differ in size, no batching)."""
    return torch.utils.data.DataLoader(
        dataset, batch_size=None, shuffle=False, num_workers=workers,
        pin_memory=torch.cuda.is_available(), prefetch_factor=2 if workers > 0 else None,
        persistent_workers=False,
    )
