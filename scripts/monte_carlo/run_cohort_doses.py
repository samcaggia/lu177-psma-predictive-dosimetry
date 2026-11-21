#!/usr/bin/env python3
"""Monte Carlo doses of the frozen virtual cohort (notebook 21): per phantom, voxel basis B_c on disk, its target and lesion means, and the lesion part of every virtual patient and scenario. Resumable: a phantom is done when its two parquet files exist.

--metal-basis rebuilds only the organ basis of the phantoms with metal voxels (cohort_doses.METAL_HU), after metal was taken out of the long-range low-pass; the lesion parts do not depend on it and are kept. Resumable through parts/<study>_metal.done."""
import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import pandas as pd  # noqa: E402

import cohort_doses as D  # noqa: E402

PARTS = D.OUT / "parts"


def log(message):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with open(D.OUT / "run.log", "a") as handle:
        handle.write(line + "\n")


def write(frame, study_id, name):
    temporary = PARTS / f"{study_id}_{name}.tmp.parquet"
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, PARTS / f"{study_id}_{name}.parquet")


def work(args):
    study_id, tia, workers = args
    start = time.time()
    basis, lesion_part, n_maps = D.phantom_doses(study_id, tia, workers)
    write(basis, study_id, "basis")
    if lesion_part is None:
        (PARTS / f"{study_id}_metal.done").touch()
    else:
        write(lesion_part, study_id, "lesion_part")
    return study_id, n_maps, time.time() - start


def has_metal(study_id):
    return bool(D.metal_mask(study_id).any())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--metal-basis", action="store_true")
    args = parser.parse_args()
    PARTS.mkdir(parents=True, exist_ok=True)
    tia = pd.concat([D.load_tia("train"), D.load_tia("test")])
    studies = sorted(tia.study_id.unique())
    if args.metal_basis:
        with Pool(args.processes) as pool:
            metal = [s for s, m in zip(studies, pool.map(has_metal, studies)) if m]
        todo = [s for s in metal if not (PARTS / f"{s}_metal.done").exists()]
        jobs = [(s, None, args.workers) for s in todo]
        log(f"launch metal basis: {len(metal)} of {len(studies)} phantoms with metal, {len(todo)} to do, {args.processes} processes x {args.workers} FFT threads, pid {os.getpid()}")
    else:
        todo = [s for s in studies if not (PARTS / f"{s}_lesion_part.parquet").exists()]
        jobs = [(s, tia[tia.study_id == s], args.workers) for s in todo]
        log(f"launch: {len(studies)} phantoms, {len(todo)} to do, {args.processes} processes x {args.workers} FFT threads, pid {os.getpid()}")
    if args.limit:
        jobs = jobs[:args.limit]
    done = 0
    with Pool(args.processes, maxtasksperchild=4) as pool:
        for study_id, n_maps, seconds in pool.imap_unordered(work, jobs):
            done += 1
            log(f"{study_id}: {n_maps} lesion maps, {seconds:.0f} s; {done}/{len(jobs)}")
    log("finished")


if __name__ == "__main__":
    main()
