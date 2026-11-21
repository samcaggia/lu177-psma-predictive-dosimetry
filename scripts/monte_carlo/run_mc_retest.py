#!/usr/bin/env python3
"""GATE runs of notebook 15, resumable, 4 threads: native kernels, the 3 new sources, a low-budget set, direct runs, per-lesion checks. Does not touch the pilot results."""
import json
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

STUDIES = {"psma_18f866bfa3d793d4_2016-08-01": 5, "psma_505ebb983b71c700_2015-11-07": 3, "psma_576326cfba8460b8_2016-02-12": 3}
TYPICAL, HIGH_TLV = list(STUDIES)[:2]
THIRD_GRID = "psma_de1f4300eda3bef3_2017-04-23"
NEW = ["blood_pool", "spine", "rest_of_body"]
BATCH = 200_000
DIRECT_HALF = 1_000_000
LESION_DIRECT = 2_000_000
SINGLE_LESION = 200_000
THREADS = 4
KERNEL_N = 20_000_000
KERNELS = {(2.734375, 2.734375, 3.2699997425079346), (4.0728302001953125, 4.0728302001953125, 5.0), (4.0728302001953125, 4.0728302001953125, 2.0)}
TABLE = M.V2 / "targets.csv"


def done():
    if not TABLE.exists():
        return set()
    t = pd.read_csv(TABLE)
    return set(t.run_id)


def affine(study):
    return nib.load(M.ph.PHANTOMS / study / "ct.nii.gz").affine


def save(study, name, dose):
    image = nib.Nifti1Image(dose.astype(np.float32), affine(study))
    image.header.set_data_dtype(np.float32)
    nib.save(image, M.V2 / study / f"{name}.nii.gz")


def run_one(study, run_id, kind, n, weights, seed, work, masks):
    folder = work / run_id.replace("/", "_")
    folder.mkdir(parents=True, exist_ok=True)
    M.write_image(weights.astype(np.float32), M.spacing_of(study), folder / "source.mhd")
    start = time.time()
    M.run_gate(work / "ct_materials.mhd", work / "materials.json", folder / "source.mhd", folder, n, seed, THREADS)
    wall = time.time() - start
    info = json.loads((folder / "run.json").read_text())
    dose = M.read_image(folder / "mc_dose.mhd").astype(np.float64) / info["events"] * M.DECAYS_PER_MBQ_H
    means = M.mass_weighted_on_targets(dose, masks["density"], masks["targets"])
    plain = {t: float(dose[m].mean()) if m.any() else np.nan for t, m in masks["targets"].items()}
    pd.DataFrame([dict(run_id=run_id, study_id=study, kind=kind, n=info["events"], seed=seed, seconds=info["seconds"], wall_seconds=wall, target=t, gy_per_mbq_h=means[t], plain_mean=plain[t]) for t in means]).to_csv(TABLE, mode="a", header=not TABLE.exists(), index=False)
    shutil.rmtree(folder)
    print(time.strftime("%H:%M:%S"), run_id, round(wall), flush=True)
    return dose


def kernels():
    M.NATIVE_KERNELS.mkdir(parents=True, exist_ok=True)
    for spacing in sorted(KERNELS):
        name = "kernel_{:.2f}x{:.2f}x{:.2f}".format(*spacing).replace(".", "p")
        if (M.NATIVE_KERNELS / f"{name}_raw.npy").exists():
            continue
        subprocess.run([sys.executable, str(M.WORKSPACE / "scripts" / "gate" / "lu177_native_kernel.py"), "--voxel-mm", *map(str, spacing), "--n", str(KERNEL_N), "--threads", str(THREADS), "--name", name, "--output", str(M.NATIVE_KERNELS)], check=True, cwd=str(M.NATIVE_KERNELS))
        shutil.rmtree(M.NATIVE_KERNELS / f"{name}_work")
        print(time.strftime("%H:%M:%S"), name, flush=True)


def main():
    M.V2.mkdir(parents=True, exist_ok=True)
    kernels()
    for k, (study, batches) in enumerate({**STUDIES, THIRD_GRID: 3}.items()):
        folder = M.V2 / study
        if study != THIRD_GRID and (folder / "D_direct_b_gy.nii.gz").exists():
            continue
        folder.mkdir(exist_ok=True)
        work = folder / "work"
        work.mkdir(exist_ok=True)
        M.write_material_inputs(study, work)
        masks = dict(targets=M.target_masks(study), density=M.density_map(M.load(study, "ct"), M.load(study, "organs")))
        tia, lesion_tia, fractions = M.test_tia_v2(study)
        seed = 50000 + 1000 * k
        for compartment in (M.COMPARTMENTS_V2 if study == THIRD_GRID else NEW):
            seed += 100
            for variant, n_batches in ((("pet", batches),) if study == THIRD_GRID else (("pet", batches), ("uniform", 1))):
                name = f"D_{compartment}_{variant}"
                if (folder / f"{name}.nii.gz").exists():
                    continue
                weights = M.source_weights_v2(study, compartment, variant)
                if weights is None:
                    continue
                total = 0.0
                for batch in range(n_batches):
                    total = total + run_one(study, f"{study}/{name}/{batch}", variant, BATCH, weights, seed + batch + (50 if variant == "uniform" else 0), work, masks)
                save(study, name, total / n_batches)
        weights = sum(tia[c] * M.source_weights_v2(study, c, "pet") for c in tia if c != "lesions" and M.source_weights_v2(study, c, "pet") is not None)
        lesions = M.lesion_weights(study)
        for i, (idx, w) in lesions.items():
            weights[idx] += lesion_tia[i] * w
        total_tia = float(weights.sum())
        pd.DataFrame([dict(compartment=c, tia_mbq_h=v, fraction=fractions[c]) for c, v in tia.items()] + [dict(compartment=f"lesion_{i}", tia_mbq_h=v) for i, v in lesion_tia.items()]).to_csv(folder / "test_tia.csv", index=False)
        for half in "ab":
            name = f"D_direct_{half}_gy"
            if not (folder / f"{name}.nii.gz").exists():
                save(study, name, run_one(study, f"{study}/{name}", "direct", DIRECT_HALF, weights / total_tia, seed + 500 + ord(half), work, masks) * total_tia)
        order = sorted(lesions, key=lambda i: -lesions[i][0][0].size)
        chosen = {order[0], order[len(order) // 2], order[-1]} if len(order) >= 3 else set()
        if study == TYPICAL:
            chosen = set(order)
        if study == HIGH_TLV:
            mid = [i for i in order if 1.5 <= lesions[i][0][0].size * np.prod(M.spacing_of(study)) / 1000 <= 2.5][:1]
            small = [i for i in order if 0.3 <= lesions[i][0][0].size * np.prod(M.spacing_of(study)) / 1000 <= 0.6][:1]
            chosen = {order[0], order[6]} | set(mid) | set(small)
        for i in sorted(chosen):
            name = f"D_lesion{i}_single"
            if not (folder / f"{name}.nii.gz").exists():
                w = np.zeros(M.load(study, "organs").shape)
                w[lesions[i][0]] = lesions[i][1]
                save(study, name, run_one(study, f"{study}/{name}", "single_lesion", SINGLE_LESION, w, seed + 700 + i, work, masks))
        if study == HIGH_TLV and not (folder / "D_lesions_only_direct_gy.nii.gz").exists():
            w = np.zeros(weights.shape)
            for i, (idx, value) in lesions.items():
                w[idx] += lesion_tia[i] * value
            total = float(w.sum())
            save(study, "D_lesions_only_direct_gy", run_one(study, f"{study}/lesions_only", "lesions_only", LESION_DIRECT, w / total, seed + 900, work, masks) * total)
        if study == TYPICAL:
            for j, c in enumerate(M.COMPARTMENTS_V2):
                name = f"D_low_{c}"
                if not (folder / f"{name}.nii.gz").exists():
                    save(study, name, run_one(study, f"{study}/{name}", "low", BATCH, M.source_weights_v2(study, c, "pet"), seed + 800 + j, work, masks))
        shutil.rmtree(work)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
