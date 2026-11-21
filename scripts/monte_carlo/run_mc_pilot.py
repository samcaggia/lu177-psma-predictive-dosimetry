#!/usr/bin/env python3
"""Runs the GATE batches of notebook 14: one process per batch, at most 4 threads, resumable.

Per phantom and compartment: PET-weighted source in BATCHES[study] independent batches, uniform source in one. Then the combined source
on the first phantom. Target means go to mc_pilot/targets.csv after every batch; pooled dose images are written as Gy/(MBq h).
"""
import json
import shutil
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import psma_mc as M  # noqa: E402

BATCH_SIZE = 200_000
STUDIES = {"psma_18f866bfa3d793d4_2016-08-01": 5, "psma_505ebb983b71c700_2015-11-07": 3, "psma_576326cfba8460b8_2016-02-12": 3}
COMBINED_STUDY = "psma_18f866bfa3d793d4_2016-08-01"
COMBINED_N = 2_000_000
TIMING_N = 10_000
THREADS = 4
TEST_T_EFF_H = {"kidneys_functional": 50, "liver": 70, "spleen": 70, "salivary_glands": 30, "bladder": 20, "gi_tract": 40, "rest_of_body": 40, "lesions": 77}
TEST_ACTIVITY_MBQ = 7400
TABLE = M.PILOT / "targets.csv"


def done():
    if not TABLE.exists():
        return set()
    t = pd.read_csv(TABLE)
    return set(zip(t.study_id, t.compartment, t.variant, t.batch))


def run_one(study, compartment, variant, batch, n, weights, work, masks):
    seed = 1000 + 97 * list(STUDIES).index(study) + 1009 * batch + sum(map(ord, compartment + variant))
    folder = work / f"{compartment}_{variant}_{batch}"
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "source.mhd"
    M.write_image(weights.astype(np.float32), M.spacing_of(study), source)
    start = time.time()
    M.run_gate(work / "ct_materials.mhd", work / "materials.json", source, folder, n, seed, THREADS)
    wall = time.time() - start
    info = json.loads((folder / "run.json").read_text())
    dose = M.read_image(folder / "mc_dose.mhd").astype(np.float64) / info["events"] * M.DECAYS_PER_MBQ_H
    relative = M.read_image(folder / "mc_dose_uncertainty.mhd")
    rows = []
    for target, mask in masks.items():
        values = relative[mask]
        rows.append(dict(study_id=study, compartment=compartment, variant=variant, batch=batch, seed=seed, n=info["events"], threads=THREADS, seconds=info["seconds"],
                         wall_seconds=wall, target=target, gy_per_mbq_h=float(dose[mask].mean()) if mask.any() else np.nan,
                         voxel_relative_uncertainty_median=float(np.median(values)) if mask.any() else np.nan))
    pd.DataFrame(rows).to_csv(TABLE, mode="a", header=not TABLE.exists(), index=False)
    shutil.rmtree(folder)
    return dose


def save_dose(study, name, dose):
    affine = nib.load(M.ph.PHANTOMS / study / "ct.nii.gz").affine
    image = nib.Nifti1Image(dose.astype(np.float32), affine)
    image.header.set_data_dtype(np.float32)
    nib.save(image, M.PILOT / study / f"{name}.nii.gz")


def pooled(study, compartment, variant, batches, work, masks, finished):
    name = f"D_{compartment}_{variant}"
    weights = M.source_weights(study, compartment, variant)
    if weights is None:
        print(study, compartment, variant, "empty", flush=True)
        return
    if (M.PILOT / study / f"{name}.nii.gz").exists():
        return
    total_path = work / f"{name}_sum.npy"
    total = np.load(total_path) if total_path.exists() else np.zeros(weights.shape)
    for batch in range(batches):
        if (study, compartment, variant, batch) in finished:
            continue
        total += run_one(study, compartment, variant, batch, BATCH_SIZE, weights, work, masks)
        np.save(total_path, total)
        print(time.strftime("%H:%M:%S"), study, compartment, variant, batch, flush=True)
    save_dose(study, name, total / batches)
    total_path.unlink()


def combined(study, work, masks, finished):
    if (study, "combined", "pet", 0) in finished:
        return
    pet, organs = M.load(study, "pet_pct_ia_ml").astype(np.float64), M.load(study, "organs")
    voxel_ml = np.prod(M.spacing_of(study)) / 1000
    rows, weights = [], np.zeros(pet.shape)
    for compartment, labels in M.COMPARTMENTS.items():
        mask = np.isin(organs, labels)
        fraction = pet[mask].sum() * voxel_ml / 100
        tia = TEST_ACTIVITY_MBQ * fraction * TEST_T_EFF_H[compartment] / np.log(2)
        rows.append(dict(compartment=compartment, fraction_ia=fraction, t_eff_h=TEST_T_EFF_H[compartment], tia_mbq_h=tia))
        weights += tia * M.source_weights(study, compartment, "pet")
    pd.DataFrame(rows).to_csv(M.PILOT / study / "combined_tia.csv", index=False)
    total_tia = sum(r["tia_mbq_h"] for r in rows)
    dose = run_one(study, "combined", "pet", 0, COMBINED_N, weights / total_tia, work, masks)
    save_dose(study, "D_combined_direct_gy", dose * total_tia)


def main():
    M.PILOT.mkdir(parents=True, exist_ok=True)
    for study, batches in STUDIES.items():
        folder = M.PILOT / study
        folder.mkdir(exist_ok=True)
        work = folder / "work"
        work.mkdir(exist_ok=True)
        M.write_material_inputs(study, work)
        masks = M.target_masks(study)
        for variant, n_batches in (("pet", batches), ("uniform", 1)):
            for compartment in M.COMPARTMENTS:
                pooled(study, compartment, variant, n_batches, work, masks, done())
        if study == COMBINED_STUDY:
            combined(study, work, masks, done())
        if (study, "liver", "timing", 0) not in done():
            run_one(study, "liver", "timing", 0, TIMING_N, M.source_weights(study, "liver", "pet"), work, masks)
        shutil.rmtree(work)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
