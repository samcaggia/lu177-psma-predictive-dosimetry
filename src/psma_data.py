"""Single loader of the PSMA-PET-CT-Lesions studies.

Every use of the PET goes through this module, so the global exclusion list and the slice crop of
psma_exclusions.py are always applied and all volumes come back on the same (PET) grid.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from paths import DATASET as RAW_DATA
from paths import WORKSPACE
from psma_exclusions import SUV_EXCLUSIONS, filter_pet_studies, pet_slice_range

DATASET = RAW_DATA / "PSMA-PET-CT-Lesions_v3"
IMAGES = DATASET / "imagesTr"
LABELS = DATASET / "labelsTr"
SEGMENTATIONS = DATASET / "psma_segmentations"
PROCESSED = WORKSPACE / "data/processed/psma"
INVENTORY = PROCESSED / "psma_inventory.csv"
HEADERS = PROCESSED / "pet_dicom_headers.csv"

HEADER_COLUMNS = [
    "study_id", "weight_kg", "height_m", "activity_mbq", "half_life_s", "uptake_series_min",
    "pbpk_cohort", "suv_to_pct_ia_per_ml_decay_corrected",
]
AFFINE_TOLERANCE = 1e-2


def study_table():
    """One row per usable study (excluded studies removed): inventory plus the DICOM header quantities."""
    inventory = pd.read_csv(INVENTORY)
    headers = pd.read_csv(HEADERS, usecols=HEADER_COLUMNS)
    table = inventory.merge(headers, on="study_id", how="left", validate="one_to_one")
    return filter_pet_studies(table).reset_index(drop=True)


def labelmap(task):
    """Label -> class name of the segmentation masks; task is 'organs' or 'salivary'."""
    content = json.loads((SEGMENTATIONS / f"labelmap_{task}.json").read_text())
    return {int(k): v for k, v in content["labels"].items()}


@dataclass
class PsmaStudy:
    study_id: str
    ct_hu: np.ndarray
    pet_pct_ia_ml: np.ndarray
    lesion_label: np.ndarray
    organs: np.ndarray
    salivary: np.ndarray
    affine: np.ndarray
    spacing_mm: tuple
    k_min: int
    info: dict

    @property
    def voxel_ml(self):
        return float(np.prod(self.spacing_mm)) / 1000.0


def _read(path, k_min, reference=None):
    image = nib.load(path)
    if reference is not None:
        if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=AFFINE_TOLERANCE):
            raise ValueError(f"{path.name} is not on the PET grid")
    return image, np.asanyarray(image.dataobj)[:, :, k_min:]


def load_study(study_id, table=None):
    """CT (HU), PET (decay-corrected %IA/mL), lesion label, organ and salivary masks, all on the PET grid.

    Raises ValueError for a study in the exclusion list; a cropped study comes back cropped, with its affine updated.
    """
    if study_id in SUV_EXCLUSIONS:
        raise ValueError(f"{study_id} is excluded from SUV-based use: {SUV_EXCLUSIONS[study_id]}")
    table = study_table() if table is None else table
    row = table.loc[table["study_id"] == study_id].iloc[0]

    pet_image = nib.load(IMAGES / f"{study_id}_0001.nii.gz")
    k_min, _ = pet_slice_range(study_id, pet_image.shape[2])
    suv = np.asanyarray(pet_image.dataobj)[:, :, k_min:].astype(np.float32)
    factor = float(row["suv_to_pct_ia_per_ml_decay_corrected"])
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError(f"{study_id}: no SUV to %IA/mL factor")

    _, ct = _read(IMAGES / f"{study_id}_0000.nii.gz", k_min, pet_image)
    _, label = _read(LABELS / f"{study_id}.nii.gz", k_min, pet_image)
    _, organs = _read(SEGMENTATIONS / "organs" / f"{study_id}.nii.gz", k_min, pet_image)
    _, salivary = _read(SEGMENTATIONS / "salivary" / f"{study_id}.nii.gz", k_min, pet_image)

    affine = pet_image.affine.copy()
    affine[:3, 3] = pet_image.affine[:3, :3] @ np.array([0, 0, k_min]) + pet_image.affine[:3, 3]
    info = dict(
        subject=row["subject"], tracer=row["tracer"], scanner=row["scanner"], ct_contrast=bool(row["ct_contrast"]),
        weight_kg=float(row["weight_kg"]), height_m=None if pd.isna(row["height_m"]) else float(row["height_m"]),
        activity_mbq=float(row["activity_mbq"]), half_life_s=float(row["half_life_s"]),
        uptake_min=float(row["uptake_series_min"]), pbpk_cohort=bool(row["pbpk_cohort"]),
        suv_to_pct_ia_per_ml=factor, nz_original=int(pet_image.shape[2]),
    )
    return PsmaStudy(
        study_id=study_id, ct_hu=ct.astype(np.float32), pet_pct_ia_ml=suv * np.float32(factor),
        lesion_label=(label > 0), organs=organs.astype(np.uint8), salivary=salivary.astype(np.uint8),
        affine=affine, spacing_mm=tuple(float(z) for z in pet_image.header.get_zooms()[:3]), k_min=k_min, info=info,
    )


def lesion_components(lesion_label):
    """Lesions as 26-connected components, numbered from 1 (same connectivity as notebook 08)."""
    return ndi.label(lesion_label, structure=np.ones((3, 3, 3), dtype=bool))
