"""Benchmark of the training step (largest batch, speed, memory, GPU utilisation) and cost estimate of the training runs."""
import gc
import time

import numpy as np
import pandas as pd
import torch

from .evaluate import validate
from .model import set_checkpointing
from .train import make_rng, train_step
from .utils import GpuMonitor, gpu_name

MIB = 1024 ** 2


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _fits(model, cohort, indices, batch, patch, sampling_cfg, loss_cfg, steps=2):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6)
    rng = make_rng(0)
    ok = True
    try:
        for _ in range(steps):
            train_step(model, opt, cohort, indices, batch, patch, sampling_cfg, loss_cfg, rng)
    except torch.OutOfMemoryError:
        ok = False
    del opt
    _free()
    return ok


def largest_batch(model, cohort, indices, patch, sampling_cfg, loss_cfg, candidates=(1, 2, 3, 4, 6, 8, 12, 16)):
    """Largest batch of the candidates that completes two training steps without running out of memory (0: none)."""
    best = 0
    for batch in candidates:
        if not _fits(model, cohort, indices, batch, patch, sampling_cfg, loss_cfg):
            break
        best = batch
    return best


def time_training(model, cohort, indices, batch, patch, sampling_cfg, loss_cfg, seconds=75.0, warmup=3):
    """Real training steps for about `seconds`: iterations per second, time share of sampling and targets, memory, utilisation."""
    device = cohort.device
    cuda = device.type == "cuda"
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    rng = make_rng(0)
    model.train()
    for _ in range(warmup):
        train_step(model, opt, cohort, indices, batch, patch, sampling_cfg, loss_cfg, rng)
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    steps, data_seconds, losses, norms = 0, 0.0, [], []
    with GpuMonitor(interval=1.0) as monitor:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            loss, parts, dt = train_step(model, opt, cohort, indices, batch, patch, sampling_cfg, loss_cfg, rng,
                                         timed=True)
            losses.append(loss)
            norms.append(parts["grad_norm"])
            data_seconds += dt
            steps += 1
            if cuda:
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    row = dict(steps=steps, it_per_s=steps / elapsed, patches_per_s=steps * batch / elapsed,
               Mvoxel_per_s=steps * batch * float(np.prod(patch)) / elapsed / 1e6,
               data_share=data_seconds / elapsed,
               peak_allocated_MiB=torch.cuda.max_memory_allocated() / MIB if cuda else float("nan"),
               peak_reserved_MiB=torch.cuda.max_memory_reserved() / MIB if cuda else float("nan"),
               loss_first=float(torch.stack(losses[:5]).mean()), loss_last=float(torch.stack(losses[-5:]).mean()),
               nonfinite_steps=int((~torch.isfinite(torch.stack(norms)) | ~torch.isfinite(torch.stack(losses))).sum()),
               grad_norm_median=float(torch.stack(norms).float().nanmedian()),
               **monitor.summary())
    del opt
    _free()
    return row


def run_benchmark(model, cohort, patches, sampling_cfg, loss_cfg, seconds=75.0, checkpointing=(True, False),
                  candidates=(1, 2, 3, 4, 6, 8, 12, 16)):
    """One row per patch size and checkpointing setting, at the largest batch that fits. The model weights are changed."""
    indices = cohort.split_indices("train")
    rows = []
    for patch in patches:
        for enabled in checkpointing:
            set_checkpointing(model, enabled)
            batch = largest_batch(model, cohort, indices, patch, sampling_cfg, loss_cfg, candidates)
            timing = None
            while batch and timing is None:
                try:
                    timing = time_training(model, cohort, indices, batch, patch, sampling_cfg, loss_cfg, seconds)
                except torch.OutOfMemoryError:
                    batch = max([c for c in candidates if c < batch], default=0)
                _free()
            row = dict(gpu=gpu_name(), patch="x".join(str(p) for p in patch), checkpointing=enabled, batch=batch,
                       **(timing or {}))
            rows.append(row)
            print(" | ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
    return pd.DataFrame(rows)


def time_validation(model, cohort, patch, loss_cfg, sw_batch=2, overlap=0.25):
    """Seconds of one full validation pass (sliding windows on every validation phantom, loss on all its draws)."""
    indices = cohort.split_indices("val")
    t0 = time.perf_counter()
    val = validate(model, cohort, indices, patch, loss_cfg, sw_batch, overlap)
    if cohort.device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0, val


def patches_per_pass(cohort, patch):
    """Patches needed to see every training voxel once, counting only the part of a patch that lies inside the phantom."""
    shapes = [cohort.shape[int(i)] for i in cohort.split_indices("train")]
    return float(sum(np.prod(s) / np.prod([min(p, n) for p, n in zip(patch, s)]) for s in shapes))


def estimate_cost(table, cohort, validation_seconds, val_every, passes=(300, 600), finetune_fraction=0.25,
                  cu_per_hour=None):
    """
    Hours (and compute units, if cu_per_hour is given) of scenario A for each assumed number of passes over the training
    voxels, and of one B fine-tune at finetune_fraction of it, for every benchmark row. The number of passes is an
    assumption: the benchmark measures speed, not convergence.
    """
    rows = []
    for row in table[table.batch > 0].itertuples(index=False):
        patch = tuple(int(p) for p in row.patch.split("x"))
        per_pass = patches_per_pass(cohort, patch)
        for n in passes:
            iterations = n * per_pass / row.batch
            hours = (iterations / row.it_per_s + iterations / val_every * validation_seconds) / 3600.0
            out = dict(gpu=row.gpu, patch=row.patch, checkpointing=row.checkpointing, batch=row.batch, passes=n,
                       iterations=int(round(iterations)), hours_A=hours, hours_each_B=hours * finetune_fraction,
                       hours_A_and_4B=hours * (1 + 4 * finetune_fraction))
            if cu_per_hour:
                out.update(cu_A=hours * cu_per_hour, cu_each_B=hours * finetune_fraction * cu_per_hour,
                           cu_A_and_4B=hours * (1 + 4 * finetune_fraction) * cu_per_hour)
            rows.append(out)
    return pd.DataFrame(rows)
