#!/usr/bin/env python3
"""Studies whose NIfTI PET must not be used, or only in part, in anything based on the SUV.

Applies to every training or analysis that reads `*_0001.nii.gz`. CT-only uses are not affected.
Running this file writes the table read by notebook 08 and by fetch_psma_pet_headers.py.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data/processed/psma/psma_pet_exclusions.csv"
INVENTORY = ROOT / "data/processed/psma/psma_inventory.csv"

IMPLAUSIBLE_UPTAKE = "uptake time < 30 min in the header: the SUV decay correction uses a wrong injection time"
WRONG_WEIGHT = "header weight 180 kg, height 0.90 m (89-92 kg, 1.80 m in the other studies of the subject): the SUV uses 180 kg"
IMPLAUSIBLE_TOTAL = "143 %IA in the FOV; uptake time 132 min likely wrong in the header, not verifiable"
INTERPOLATED = "k 2-83 interpolated between DICOM slices 1 of 6 (69 slices), k 0 empty; real consecutive PET from k 84"

SUV_EXCLUSIONS = {
    "psma_6d5c002d629eb131_2022-06-04": IMPLAUSIBLE_UPTAKE,
    "psma_838c9340e85ca431_2015-09-11": IMPLAUSIBLE_UPTAKE,
    "psma_a96814a79aa26c8f_2022-09-23": IMPLAUSIBLE_UPTAKE,
    "psma_ad7bc1aa5b586751_2018-09-14": IMPLAUSIBLE_UPTAKE,
    "psma_b5a4134e681683c0_2018-10-05": IMPLAUSIBLE_UPTAKE,
    "psma_e3c4639102fe5c51_2020-09-25": IMPLAUSIBLE_UPTAKE,
    "psma_f77011649a56cf08_2021-06-26": IMPLAUSIBLE_UPTAKE,
    "psma_c80781b803041d12_2018-07-14": WRONG_WEIGHT,
    "psma_54d1613921a2d2ae_2021-02-01": IMPLAUSIBLE_TOTAL,
}
PET_CROPS = {"psma_3959d1c381a5bcd6_2015-11-12": (84, INTERPOLATED)}


def filter_pet_studies(table, column="study_id"):
    """Rows of a study table without the SUV-excluded studies."""
    return table[~table[column].isin(SUV_EXCLUSIONS)]


def pet_slice_range(study_id, nz):
    """Axial slices (k_min, k_max_exclusive) of the NIfTI PET, CT and label to use; raises for an excluded study."""
    if study_id in SUV_EXCLUSIONS:
        raise ValueError(f"{study_id} is excluded from SUV-based use: {SUV_EXCLUSIONS[study_id]}")
    return (PET_CROPS[study_id][0] if study_id in PET_CROPS else 0), nz


def build_table():
    rows = [dict(study_id=s, action="exclude", valid_k_min="", reason=r) for s, r in SUV_EXCLUSIONS.items()]
    rows += [dict(study_id=s, action="crop", valid_k_min=k, reason=r) for s, (k, r) in PET_CROPS.items()]
    meta = pd.read_csv(INVENTORY, usecols=["study_id", "subject", "tracer", "nz"])
    table = pd.DataFrame(rows).merge(meta, on="study_id", how="left", validate="one_to_one")
    if table["subject"].isna().any():
        raise ValueError("study not in the inventory")
    return table[["study_id", "subject", "tracer", "nz", "action", "valid_k_min", "reason"]].sort_values(["action", "study_id"])


if __name__ == "__main__":
    build_table().to_csv(TABLE, index=False)
    print(TABLE)
