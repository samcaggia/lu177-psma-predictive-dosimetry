#!/usr/bin/env python3
"""Independent check of the chosen voxel target against direct runs on 2 phantoms not used to choose it. Writes independent_check.csv and independent_check_voxel.csv in mc_full."""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import psma_mc as M  # noqa: E402
from run_mc_full import CHECK_STUDIES, OUT  # noqa: E402


def read(path):
    return np.asanyarray(nib.load(path).dataobj).astype(np.float64)


def main():
    means, voxels, lesion_rows = [], [], []
    for s in CHECK_STUDIES:
        organs = M.load(s, "organs")
        density, masks, spacing, body = M.density_map(M.load(s, "ct"), organs), M.target_masks(s), M.spacing_of(s), organs > 0
        rho = np.maximum(density, 0.1)
        tia, lesion_tia, _ = M.test_tia_v2(s)
        weights = M.lesion_weights(s)
        amounts = dict(tia, lesions=sum(lesion_tia[i] for i in weights))
        present = [c for c in M.COMPARTMENTS_V2 if amounts.get(c, 0) > 0]
        source = sum(tia[c] * M.source_weights_v2(s, c, "pet") for c in present if c != "lesions")
        for i, (idx, w) in weights.items():
            source[idx] += lesion_tia[i] * w
        split, native = M.SplitVsv(organs.shape, spacing), M.NativeVsv(organs.shape, spacing)
        vsv = native(source) / rho
        chosen = split(source, density)
        specified = vsv.copy()
        raw = np.zeros(organs.shape)
        for c in present:
            w = M.source_weights_v2(s, c, "pet")
            dose = read(OUT / s / f"D_{c}_pet.nii.gz")
            raw += amounts[c] * dose
            specified += amounts[c] * M.long_range_term(dose, native(w), density, body, 20.0, spacing)
            if c != "lesions":
                chosen += amounts[c] * M.long_range_by_material(dose, split(w, density), density, body, 40.0 if c == "rest_of_body" else 10.0, spacing)
        a, b = read(OUT / s / "D_direct_a_gy.nii.gz"), read(OUT / s / "D_direct_b_gy.nii.gz")
        direct = (a + b) / 2
        reference, half_a, half_b = [M.mass_weighted_on_targets(x, density, masks) for x in (direct, a, b)]
        estimates = {"chosen": chosen, "specified sigma 20": specified, "VSV/rho only": vsv, "raw MC D_c, lesions grouped": raw}
        for name, estimate in estimates.items():
            found = M.mass_weighted_on_targets(estimate, density, masks)
            for t in M.TARGETS:
                if reference[t] == reference[t]:
                    means.append(dict(study_id=s, estimate=name, target=t, gy=found[t], direct_gy=reference[t], ratio=found[t] / reference[t], half_difference_pct=100 * abs(half_a[t] - half_b[t]) / (half_a[t] + half_b[t])))
            for region, mask in {**masks, "body": body}.items():
                m = mask & (density > 0.5)
                if m.sum() >= 20:
                    half_r = float(np.corrcoef(a[m], b[m])[0, 1])
                    voxels.append(dict(study_id=s, estimate=name, region=region, voxels=int(m.sum()), r=float(np.corrcoef(estimate[m], direct[m])[0, 1]), half_r=half_r, ceiling=np.sqrt(2 * half_r / (1 + half_r)) if half_r > 0 else np.nan))
        lesions = M.load(s, "lesions")
        total = float(source.sum())
        for i in weights:
            m = lesions == i
            mean = lambda x: float((x[m] * density[m]).sum() / density[m].sum())
            lesion_rows.append(dict(study_id=s, lesion=i, volume_ml=m.sum() * np.prod(spacing) / 1000, expected_decays=2e6 * lesion_tia[i] / total, direct_gy=mean(direct), chosen_ratio=mean(chosen) / mean(direct), grouped_ratio=mean(raw) / mean(direct)))
    pd.DataFrame(means).to_csv(OUT / "independent_check.csv", index=False)
    pd.DataFrame(voxels).to_csv(OUT / "independent_check_voxel.csv", index=False)
    pd.DataFrame(lesion_rows).to_csv(OUT / "independent_check_lesions.csv", index=False)


if __name__ == "__main__":
    main()
