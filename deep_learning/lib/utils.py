"""General utilities: device, mixed precision, scheduler, checkpoints in the background, GPU monitor, Colab runtime."""
import copy
import gc
import hashlib
import math
import os
import random
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    """CUDA device with TF32 and cuDNN autotuning enabled, or the CPU."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Device in use: {gpu_name()} ({total:.1f} GB) | torch {torch.__version__}")
    else:
        print(f"Device in use: cpu | torch {torch.__version__}")
    return device


def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def autocast_bf16(device):
    """bfloat16 mixed-precision context, valid on both GPU and CPU."""
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"VRAM allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB")


def copy_files_to_local(src_dir, names, dst_dir):
    """Copies the named files from Drive to the local disk once per session and returns the local folder."""
    os.makedirs(dst_dir, exist_ok=True)
    for name in names:
        dst = os.path.join(dst_dir, name)
        if not os.path.isfile(dst):
            print(f"Copying {name} -> {dst_dir} ...")
            shutil.copyfile(os.path.join(src_dir, name), dst + ".tmp")
            os.rename(dst + ".tmp", dst)
    return dst_dir


def get_warmup_cosine_scheduler(optimizer, warmup_iters, total_iters, min_lr_ratio=0.01):
    """Linear warm-up then cosine decay to min_lr_ratio, stepped once per iteration."""
    def lr_lambda(step):
        if step < warmup_iters:
            return (step + 1) / max(1, warmup_iters)
        progress = min(1.0, (step - warmup_iters) / max(1, total_iters - warmup_iters))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def load_state_dict_checked(model, state_dict, source):
    """load_state_dict(strict=False) that raises if no key matches and warns about skipped keys."""
    if not set(state_dict) & set(model.state_dict()):
        raise RuntimeError(f"No key of {source} matches the model: nothing would be loaded")
    result = model.load_state_dict(state_dict, strict=False)
    for kind, keys in (("missing", result.missing_keys), ("unexpected", result.unexpected_keys)):
        if keys:
            shown = ", ".join(keys[:5]) + (", ..." if len(keys) > 5 else "")
            print(f"[WARNING] {source}: {len(keys)} {kind} keys ({shown})")


def _to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return copy.deepcopy(obj)


def _atomic_save(obj, path, keep_previous=False):
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    if keep_previous and os.path.exists(path):
        os.replace(path, f"{path}.prev")
    os.replace(tmp, path)


def load_checkpoint(path, device):
    """Loads path, or path.prev if path is missing or unreadable (interrupted write). Returns (object, file) or (None, None)."""
    for candidate in (path, f"{path}.prev"):
        if os.path.exists(candidate):
            try:
                return torch.load(candidate, map_location=device, weights_only=False), candidate
            except Exception as e:
                print(f"[WARNING] {candidate} not readable ({type(e).__name__}: {e})")
    return None, None


class BackgroundSaver:
    """
    Writes checkpoints in a background thread (training waits only for the CPU copy); close() waits. Every write goes to
    a temporary file and is renamed; keep_previous keeps the file it replaces as <path>.prev.
    """

    def __init__(self, max_pending=2):
        self.max_pending = max_pending
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending = []

    def save(self, obj, path, keep_previous=False):
        while len(self._pending) >= self.max_pending:
            self._pending.pop(0).result()
        self._pending.append(self._executor.submit(_atomic_save, _to_cpu(obj), path, keep_previous))

    def close(self):
        try:
            while self._pending:
                self._pending.pop(0).result()
        finally:
            self._executor.shutdown(wait=True)


class GpuMonitor:
    """Samples GPU utilisation (%) and used memory (MiB) with nvidia-smi in a background thread."""

    def __init__(self, interval=1.0):
        self.interval = interval
        self.utilisation, self.memory = [], []
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        query = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits", "-i", "0"]
        while not self._stop.is_set():
            try:
                used, memory = subprocess.run(query, capture_output=True, text=True, timeout=5).stdout.split(",")
                self.utilisation.append(float(used))
                self.memory.append(float(memory))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def summary(self):
        """Mean and minimum utilisation, maximum used memory; NaN without nvidia-smi."""
        if not self.utilisation:
            return dict(gpu_util_mean=float("nan"), gpu_util_min=float("nan"), smi_memory_max_MiB=float("nan"))
        return dict(gpu_util_mean=float(np.mean(self.utilisation)), gpu_util_min=float(np.min(self.utilisation)),
                    smi_memory_max_MiB=float(np.max(self.memory)))


def _md5(paths):
    digest = hashlib.md5()
    for path in paths:
        with open(path, "rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()


def lib_digest(code_dir):
    """md5 of the lib/*.py files of code_dir, in name order."""
    lib = os.path.join(code_dir, "lib")
    return _md5(os.path.join(lib, name) for name in sorted(os.listdir(lib)) if name.endswith(".py"))


def write_version(code_dir):
    """Writes code_dir/VERSION (commit of the code and digest of lib/); run after committing the code, then commit VERSION."""
    commit = subprocess.run(["git", "-C", code_dir, "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    with open(os.path.join(code_dir, "VERSION"), "w") as handle:
        handle.write(f"{commit} {lib_digest(code_dir)}\n")
    return commit


def provenance(code_dir, src_dir):
    """
    code_commit (from code_dir/VERSION, there is no git repository on Drive; '+modified' if lib/ no longer matches the
    digest stored with it, 'unknown' without the file) and net_data_md5, to be written into every saved table.
    """
    path = os.path.join(code_dir, "VERSION")
    commit = "unknown"
    if os.path.isfile(path):
        with open(path) as handle:
            commit, digest = handle.read().split()
        if digest != lib_digest(code_dir):
            commit += "+modified"
    return dict(code_commit=commit, net_data_md5=_md5([os.path.join(src_dir, "net_data.py")]))


def save_table(table, path, stamp):
    """Writes the table as CSV with the provenance columns appended to every row."""
    table.assign(**stamp).to_csv(path, index=False)


def save_curves(train_log, val_log, path, title, stamp):
    """PNG of the training NLL (mean between two log lines) and the validation NLL against the optimiser step."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if train_log:
        ax.plot([r["step"] for r in train_log], [r["train_nll"] for r in train_log], lw=1, label="training NLL (patches)")
    if val_log:
        ax.plot([r["step"] for r in val_log], [r["val_nll"] for r in val_log], "o-", ms=4,
                label="validation NLL (whole phantoms)")
    ax.set_xlabel("optimiser step")
    ax.set_ylabel("NLL")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.text(0.01, 0.01, " | ".join(f"{k} {v}" for k, v in stamp.items()), fontsize=6, alpha=0.7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(f"{path}.tmp", format="png", dpi=120, metadata={k: str(v) for k, v in stamp.items()})
    plt.close(fig)
    os.replace(f"{path}.tmp", path)


def stop_runtime():
    """Flushes Drive and releases the Colab runtime (stops the billing of compute units); does nothing outside Colab."""
    try:
        from google.colab import drive, runtime
    except ImportError:
        print("Not on Colab: runtime left running")
        return
    drive.flush_and_unmount()
    runtime.unassign()
