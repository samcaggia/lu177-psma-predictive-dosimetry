#!/usr/bin/env python3
"""Full Monte Carlo run: 10 PET-weighted sources per phantom, training + test, resumable. Check from outside: run.log, status.json, progress.csv, failures.csv. Create the file STOP to end after the current run. A lock on run.lock refuses a second launch while one is alive (released by the kernel on exit or crash)."""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import psma_mc as M  # noqa: E402

OUT = M.WORKSPACE / "data" / "processed" / "psma" / "mc_full"
N_PER_SOURCE = 500_000
DIRECT_HALF = 1_000_000
CHECK_STUDIES = ["psma_bc63defabeea583e_2017-04-22", "psma_9abd0bb6d4259d4e_2015-10-22"]
MIN_FREE_GB = 5.0


def log(message):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with open(OUT / "run.log", "a") as handle:
        handle.write(line + "\n")


def append(name, row):
    path = OUT / name
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def save(study, name, dose):
    image = nib.Nifti1Image(dose.astype(np.float32), nib.load(M.ph.PHANTOMS / study / "ct.nii.gz").affine)
    image.header.set_data_dtype(np.float32)
    temporary = OUT / study / f"{name}.tmp.nii.gz"
    nib.save(image, temporary)
    os.replace(temporary, OUT / study / f"{name}.nii.gz")


class Status:
    def __init__(self, total, threads):
        self.total, self.threads, self.started, self.done_now, self.seconds_now, self.failed = total, threads, time.time(), 0, 0.0, 0
        self.done_before = sum(1 for _ in OUT.glob("psma_*/D_*_pet.nii.gz"))

    def write(self, current):
        done = self.done_before + self.done_now
        rate = self.seconds_now / self.done_now if self.done_now else None
        data = dict(pid=os.getpid(), threads=self.threads, n_per_source=N_PER_SOURCE, updated=time.strftime("%Y-%m-%d %H:%M:%S"), current=current, source_runs_done=done, source_runs_total=self.total, percent=round(100 * done / self.total, 2), failed_this_launch=self.failed, seconds_per_source_run=rate, eta_days=None if rate is None else round((self.total - done) * rate / 86400, 2), free_disk_gb=round(shutil.disk_usage(OUT).free / 1e9, 1))
        temporary = OUT / "status.tmp.json"
        temporary.write_text(json.dumps(data, indent=1))
        os.replace(temporary, OUT / "status.json")


def run_source(study, name, n, weights, seed, work, masks, threads, kind, scale=1.0):
    folder = work / name
    folder.mkdir(parents=True, exist_ok=True)
    M.write_image(weights.astype(np.float32), M.spacing_of(study), folder / "source.mhd")
    start = time.time()
    M.run_gate(work / "ct_materials.mhd", work / "materials.json", folder / "source.mhd", folder, n, seed, threads)
    wall = time.time() - start
    info = json.loads((folder / "run.json").read_text())
    dose = M.read_image(folder / "mc_dose.mhd").astype(np.float64) / info["events"] * M.DECAYS_PER_MBQ_H * scale
    means = M.mass_weighted_on_targets(dose, masks["density"], masks["targets"])
    save(study, name, dose)
    shutil.rmtree(folder)
    append("targets.csv", dict(study_id=study, run=name, kind=kind, n=info["events"], seed=seed, **{t: means[t] for t in means}))
    append("progress.csv", dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), study_id=study, run=name, kind=kind, n=info["events"], gate_seconds=info["seconds"], wall_seconds=wall))
    return wall


def should_stop():
    if (OUT / "STOP").exists():
        log("STOP file found: ending after the current run")
        return True
    free = shutil.disk_usage(OUT).free / 1e9
    if free < MIN_FREE_GB:
        log(f"only {free:.1f} GB free: stopping")
        return True
    return False


def process(study, index, status, threads):
    folder = OUT / study
    folder.mkdir(exist_ok=True)
    pending = [c for c in M.COMPARTMENTS_V2 if not (folder / f"D_{c}_pet.nii.gz").exists() and not (folder / f"skip_{c}").exists()]
    check = study in CHECK_STUDIES
    pending_direct = [h for h in "ab" if check and not (folder / f"D_direct_{h}_gy.nii.gz").exists()]
    if not pending and not pending_direct:
        return True
    work = folder / "work"
    work.mkdir(exist_ok=True)
    M.write_material_inputs(study, work)
    masks = dict(targets=M.target_masks(study), density=M.density_map(M.load(study, "ct"), M.load(study, "organs")))
    for j, c in enumerate(M.COMPARTMENTS_V2):
        if c not in pending:
            continue
        if should_stop():
            return False
        weights = M.source_weights_v2(study, c, "pet")
        if weights is None:
            (folder / f"skip_{c}").write_text("empty source\n")
            status.total -= 1
            log(f"{study} {c}: empty source, skipped")
            continue
        status.write(f"{study}/{c}")
        try:
            wall = run_source(study, f"D_{c}_pet", N_PER_SOURCE, weights, 200000 + 100 * index + j, work, masks, threads, "source")
        except Exception as error:
            status.failed += 1
            detail = getattr(error, "stderr", b"") or b""
            append("failures.csv", dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), study_id=study, run=c, error=str(error)[:200], stderr_tail=detail[-500:].decode(errors="replace")))
            log(f"{study} {c}: FAILED {str(error)[:200]}")
            continue
        status.done_now += 1
        status.seconds_now += wall
        status.write(f"{study}/{c} done")
        log(f"{study} {c} {wall:.0f} s; {status.done_before + status.done_now}/{status.total}")
    if pending_direct:
        tia, lesion_tia, fractions = M.test_tia_v2(study)
        weights = sum(tia[c] * M.source_weights_v2(study, c, "pet") for c in tia if c != "lesions" and tia[c] > 0)
        for i, (idx, w) in M.lesion_weights(study).items():
            weights[idx] += lesion_tia[i] * w
        total = float(weights.sum())
        pd.DataFrame([dict(compartment=c, tia_mbq_h=v, fraction=fractions[c]) for c, v in tia.items()] + [dict(compartment=f"lesion_{i}", tia_mbq_h=v) for i, v in lesion_tia.items()]).to_csv(folder / "test_tia.csv", index=False)
        for h in pending_direct:
            if should_stop():
                return False
            wall = run_source(study, f"D_direct_{h}_gy", DIRECT_HALF, weights / total, 900000 + 100 * index + ord(h), work, masks, threads, "direct", scale=total)
            log(f"{study} direct {h} {wall:.0f} s")
    shutil.rmtree(work)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = open(OUT / "run.lock", "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.seek(0)
        sys.exit(f"another run_mc_full.py is running (lock {OUT / 'run.lock'}: {lock.read().strip()}); not launching")
    lock.seek(0)
    lock.truncate()
    lock.write(f"pid {os.getpid()} since {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lock.flush()
    split =pd.read_csv(M.ph.psma_data.PROCESSED / "psma_68ga_split.csv")
    train = sorted(split[split.split == "train"].study_id)
    test = sorted(split[split.split == "test"].study_id)
    assert all(s in train for s in CHECK_STUDIES)
    order = CHECK_STUDIES + [s for s in train if s not in CHECK_STUDIES] + test
    skipped = sum(1 for _ in OUT.glob("psma_*/skip_*"))
    status = Status(10 * len(order) - skipped, args.threads)
    log(f"launch: {len(order)} phantoms, {N_PER_SOURCE} decays per source, {args.threads} threads, pid {os.getpid()}, {status.done_before} source runs already on disk")
    (OUT / "STOP").unlink(missing_ok=True)
    for index, study in enumerate(order):
        if not process(study, index, status, args.threads):
            status.write("stopped")
            log("stopped; launch again to resume")
            return
        if study == CHECK_STUDIES[-1] and not (OUT / "independent_check.csv").exists():
            log("independent check: computing")
            result = subprocess.run([sys.executable, str(M.WORKSPACE / "scripts" / "monte_carlo" / "check_mc_full.py")], capture_output=True, text=True)
            log("independent check: " + ("written" if result.returncode == 0 else "FAILED " + result.stderr[-400:]))
    status.write("finished")
    missing = [s for s in order if any(not (OUT / s / f"D_{c}_pet.nii.gz").exists() and not (OUT / s / f"skip_{c}").exists() for c in M.COMPARTMENTS_V2)]
    log(f"finished; phantoms with missing sources: {len(missing)}; launch again to retry them")


if __name__ == "__main__":
    main()
