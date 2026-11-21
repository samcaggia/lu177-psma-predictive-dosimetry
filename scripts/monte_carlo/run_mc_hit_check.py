#!/usr/bin/env python3
"""Paired rerun of selected Monte Carlo sources of the full run (run_mc_full.py) with two scorings in the same simulation: hit type split (gamma steps at the post-step point, corrected) and random (as in the full run). Same decays and seeds as the full run. Resumable: a source is done when both maps exist.

Output: data/processed/psma/mc_hit_check/<study>/D_<source>_pet.nii.gz (split) and D_<source>_pet_random.nii.gz, timing.csv."""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import psma_mc as M  # noqa: E402

OUT = M.ph.psma_data.PROCESSED / "mc_hit_check"
FULL = M.ph.psma_data.PROCESSED / "mc_full"
STUDIES = ["psma_9b733a4106c0fe37_2016-10-14", "psma_24f62f21c04402ec_2015-04-26", "psma_d7c4cf294221ea45_2017-03-03"]
SOURCES = ["kidneys_functional", "liver", "spleen", "bladder", "blood_pool", "rest_of_body", "spine"]


def save(study, name, dose):
    image = nib.Nifti1Image(dose.astype(np.float32), nib.load(M.ph.PHANTOMS / study / "ct.nii.gz").affine)
    image.header.set_data_dtype(np.float32)
    temporary = OUT / study / f"{name}.tmp.nii.gz"
    nib.save(image, temporary)
    os.replace(temporary, OUT / study / f"{name}.nii.gz")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=7)
    p.add_argument("--studies", nargs="*", default=STUDIES)
    p.add_argument("--sources", nargs="*", default=SOURCES)
    args = p.parse_args()
    seeds = pd.read_csv(FULL / "targets.csv").query("kind == 'source'").set_index(["study_id", "run"])
    for study in args.studies:
        folder = OUT / study
        folder.mkdir(parents=True, exist_ok=True)
        todo = [c for c in args.sources if not ((folder / f"D_{c}_pet.nii.gz").exists() and (folder / f"D_{c}_pet_random.nii.gz").exists())]
        if not todo:
            continue
        work = folder / "work"
        work.mkdir(exist_ok=True)
        M.write_material_inputs(study, work)
        for c in todo:
            row = seeds.loc[(study, f"D_{c}_pet")]
            weights = M.source_weights_v2(study, c, "pet")
            run = work / c
            run.mkdir(exist_ok=True)
            M.write_image(weights.astype(np.float32), M.spacing_of(study), run / "source.mhd")
            start = time.time()
            M.run_gate(work / "ct_materials.mhd", work / "materials.json", run / "source.mhd", run, 500_000, int(row.seed), args.threads,
                       extra=("--hit-type", "split", "--random-reference"))
            wall = time.time() - start
            info = json.loads((run / "run.json").read_text())
            scale = M.DECAYS_PER_MBQ_H / info["events"]
            save(study, f"D_{c}_pet_random", M.read_image(run / "mcref_dose.mhd").astype(np.float64) * scale)
            save(study, f"D_{c}_pet", M.read_image(run / "mc_dose.mhd").astype(np.float64) * scale)
            shutil.rmtree(run)
            row_out = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), study_id=study, source=c, n=info["events"], seed=int(row.seed),
                           threads=args.threads, gate_seconds=info["seconds"], wall_seconds=wall)
            path = OUT / "timing.csv"
            pd.DataFrame([row_out]).to_csv(path, mode="a", header=not path.exists(), index=False)
            print(row_out, flush=True)
        shutil.rmtree(work)


if __name__ == "__main__":
    main()
