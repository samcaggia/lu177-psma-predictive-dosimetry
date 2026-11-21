"""Builds one phantom per PSMA study: CT, PET in decay-corrected %IA/mL, compartment labels, numbered lesions, meta.json."""
import json
import multiprocessing
import os
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import psma_data

PHANTOMS = psma_data.PROCESSED / "phantoms"
QC_DIR = psma_data.SEGMENTATIONS / "qc"
SCHNEIDER = psma_data.WORKSPACE / ".venv/lib/python3.12/site-packages/opengate/data/Schneider2000DensitiesTable.txt"

LABELS = {
    1: "kidney_functional", 2: "kidney_nonfunctional", 3: "liver", 4: "spleen",
    5: "parotid_gland_left", 6: "parotid_gland_right", 7: "submandibular_gland_left", 8: "submandibular_gland_right",
    9: "urinary_bladder", 10: "gi_tract", 11: "blood_pool", 12: "spine_L2_L4", 13: "spine_other",
    14: "rest_of_body", 15: "lesion",
}
LABEL_OF = {name: label for label, name in LABELS.items()}
GLANDS = ["parotid_gland_left", "parotid_gland_right", "submandibular_gland_left", "submandibular_gland_right"]
GI_CLASSES = ["stomach", "duodenum", "small_bowel", "colon"]
MARROW_LEVELS = ["vertebrae_L2", "vertebrae_L3", "vertebrae_L4"]
SPINE_LEVELS = [f"vertebrae_C{i}" for i in range(1, 8)] + [f"vertebrae_T{i}" for i in range(1, 13)] + \
    [f"vertebrae_L{i}" for i in range(1, 6)] + ["vertebrae_S1", "sacrum"]

# g/mL, ICRP 110 Table A.1 (adult male, AM_organs.dat). Fluid = urine, the closest listed medium.
NOMINAL_DENSITY = {
    "kidney_functional": 1.05, "kidney_nonfunctional": 1.04, "liver": 1.05, "spleen": 1.04,
    "parotid_gland_left": 1.03, "parotid_gland_right": 1.03,
    "submandibular_gland_left": 1.03, "submandibular_gland_right": 1.03,
    "urinary_bladder": 1.04, "blood_pool": 1.06,
}
GI_HU_CLIP = 50.0

ENHANCED_KIDNEY_HU = 60.0
THRESHOLD_ENHANCED_HU = 20.0
THRESHOLD_UNENHANCED_HU = 10.0
MIN_FLUID_ML_ENHANCED = 0.25
MIN_FLUID_ML_UNENHANCED = 2.0
TEST_THRESHOLDS_HU = (0, 5, 10, 15, 20, 30, 40)
CYST_MIN_ML = 2.0
CYST_COLD_RATIO = 0.35
SINGLE_KIDNEY_ML = 20.0
SMALL_KIDNEY_ML = 60.0
KIDNEY_DILATION_MM = 10.0
MIN_BLOOD_POOL_ML = 2.0
MIN_MARROW_VERTEBRA_ML = 10.0
OUTSIDE_BODY_FLAG_PCT = 5.0
BODY_HU = -300.0
BODY_OPENING_MM = 6.0
BODY_MIN_LIMB_ML = 50.0
LIMB_MIN_PET_RATIO = 0.2
BODY_SLICE_WINDOW = 15
BODY_SLICE_MIN_RATIO = 0.8
BODY_REPAIR_CLOSING_MM = 15.0
SIX = ndi.generate_binary_structure(3, 1)


def density_from_hu(hu):
    """Schneider 2000 HU-to-density table, the one used by the GATE phantom of lu177_patient_mc.py."""
    table = np.loadtxt(SCHNEIDER, comments="#")
    return np.interp(hu, table[:, 0], table[:, 1])


def _disk(radius_voxels):
    r = int(np.ceil(radius_voxels))
    x, y = np.mgrid[-r:r + 1, -r:r + 1]
    return (x ** 2 + y ** 2 <= radius_voxels ** 2 + 1e-9)[:, :, None]


def body_mask(ct_hu, pet, spacing_mm):
    """Patient body from the CT: threshold, in-plane opening against the table, 3D components, slice-wise hole filling.

    The largest component is the trunk. Other large components (arms) are kept only if their PET is not empty:
    the couch has no activity. A slice much smaller than its neighbours has an open chest wall, so its lungs were not
    filled: it is closed in-plane before filling.
    """
    opened = ndi.binary_opening(ct_hu > BODY_HU, structure=_disk(max(1.0, BODY_OPENING_MM / spacing_mm[0])))
    labels, n = ndi.label(opened)
    if n == 0:
        raise ValueError("empty body mask")
    voxel_ml = float(np.prod(spacing_mm)) / 1000.0
    volumes = ndi.sum_labels(opened, labels, index=np.arange(1, n + 1)) * voxel_ml
    largest = int(np.argmax(volumes)) + 1
    trunk_pet = float(np.mean(pet[labels == largest]))
    keep = [largest]
    for i, volume in enumerate(volumes, start=1):
        if i != largest and volume >= BODY_MIN_LIMB_ML and np.mean(pet[labels == i]) >= LIMB_MIN_PET_RATIO * trunk_pet:
            keep.append(i)
    mask = np.isin(labels, keep)
    for k in range(mask.shape[2]):
        if mask[:, :, k].any():
            mask[:, :, k] = ndi.binary_fill_holes(mask[:, :, k])
    area = mask.sum(axis=(0, 1)).astype(float)
    reference = ndi.median_filter(area, size=BODY_SLICE_WINDOW, mode="nearest")
    closing = _disk(BODY_REPAIR_CLOSING_MM / spacing_mm[0])[:, :, 0]
    for k in np.flatnonzero((area < BODY_SLICE_MIN_RATIO * reference) & (area > 0)):
        mask[:, :, k] = ndi.binary_fill_holes(ndi.binary_closing(mask[:, :, k], structure=closing))
    return mask


def _crop_box(mask, margin=2):
    where = np.where(mask)
    return tuple(slice(max(0, w.min() - margin), w.max() + margin + 1) for w in where)


def _smoothed_inside(ct_hu, mask):
    """3x3x3 median of the CT using only voxels of the organ: outside voxels take the value of the nearest organ voxel."""
    indices = ndi.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    return ndi.median_filter(ct_hu[tuple(indices)], size=3, mode="nearest")


def fluid_mask(ct_smooth, kidney, threshold_hu, min_ml, voxel_ml):
    """Fluid-density voxels of a kidney (cysts, pelvis, sinus).

    Rim voxels are partial-volume mixtures with perirenal fat, so they count only next to a fluid voxel of the interior.
    """
    low = (ct_smooth < threshold_hu) & kidney
    interior = ndi.binary_erosion(kidney, structure=SIX)
    fluid = ndi.binary_dilation(low & interior, structure=SIX, iterations=1, mask=low)
    labels, n = ndi.label(fluid, structure=np.ones((3, 3, 3), dtype=bool))
    if n == 0:
        return fluid
    volumes = ndi.sum_labels(fluid, labels, index=np.arange(1, n + 1)) * voxel_ml
    return np.isin(labels, np.flatnonzero(volumes >= min_ml) + 1)


def kidney_parenchyma_hu(ct_hu, kidney):
    core = ndi.binary_erosion(kidney, structure=SIX)
    values = ct_hu[core] if core.sum() >= 20 else ct_hu[kidney]
    return float(np.median(values)) if values.size else float("nan")


def kidney_fluid(ct_hu, kidney, voxel_ml, enhanced):
    """Fluid mask at the operating threshold of the CT type, plus functional volumes at every test threshold."""
    fluid = np.zeros_like(kidney)
    if kidney.sum() == 0:
        return fluid, {}
    box = _crop_box(kidney, margin=3)
    smooth = _smoothed_inside(ct_hu[box], kidney[box])
    threshold = THRESHOLD_ENHANCED_HU if enhanced else THRESHOLD_UNENHANCED_HU
    min_ml = MIN_FLUID_ML_ENHANCED if enhanced else MIN_FLUID_ML_UNENHANCED
    fluid[box] = fluid_mask(smooth, kidney[box], threshold, min_ml, voxel_ml)
    by_threshold = {str(t): float((kidney[box] & ~fluid_mask(smooth, kidney[box], t, min_ml, voxel_ml)).sum() * voxel_ml)
                    for t in TEST_THRESHOLDS_HU}
    return fluid, by_threshold


def largest_cold_component_ml(fluid, pet, reference, voxel_ml):
    """Largest fluid component of at least CYST_MIN_ML with a mean PET below CYST_COLD_RATIO of the reference."""
    if not fluid.any() or reference <= 0:
        return 0.0
    box = _crop_box(fluid)
    labels, n = ndi.label(fluid[box], structure=np.ones((3, 3, 3), dtype=bool))
    largest = 0.0
    for i in range(1, n + 1):
        component = labels == i
        volume = float(component.sum() * voxel_ml)
        if volume >= CYST_MIN_ML and float(np.mean(pet[box][component])) < CYST_COLD_RATIO * reference:
            largest = max(largest, volume)
    return largest


def kidney_summary(ct_hu, pet, kidney, fluid, by_threshold, reference, voxel_ml):
    functional = kidney & ~fluid
    cold_ml = largest_cold_component_ml(fluid, pet, reference, voxel_ml)
    return dict(
        present=bool(kidney.sum() > 0), total_ml=float(kidney.sum() * voxel_ml),
        functional_ml=float(functional.sum() * voxel_ml), nonfunctional_ml=float(fluid.sum() * voxel_ml),
        parenchyma_hu=kidney_parenchyma_hu(ct_hu, kidney) if kidney.sum() >= 50 else None,
        functional_mean_pct_ia_ml=float(np.mean(pet[functional])) if functional.any() else None,
        fluid_mean_pct_ia_ml=float(np.mean(pet[fluid])) if fluid.any() else None,
        cyst=bool(cold_ml > 0), largest_cold_fluid_ml=cold_ml, functional_ml_by_threshold=by_threshold,
    )


def blood_pool_mask(aorta):
    """Aorta lumen surrogate: the TotalSegmentator aorta eroded by one voxel, removing wall and partial volume."""
    return ndi.binary_erosion(aorta, structure=SIX)


def compartment_map(study):
    """Integer label image (lesions on top), the same before lesions, kidney summaries and the source masks."""
    organ_names = {v: k for k, v in psma_data.labelmap("organs").items()}
    gland_names = {v: k for k, v in psma_data.labelmap("salivary").items()}
    organs, salivary, ct, pet = study.organs, study.salivary, study.ct_hu, study.pet_pct_ia_ml
    voxel_ml = study.voxel_ml

    def organ(name):
        return organs == organ_names[name]

    left, right = organ("kidney_left"), organ("kidney_right")
    parenchyma = [kidney_parenchyma_hu(ct, m) for m in (left, right) if m.sum() >= 50]
    enhanced = bool(parenchyma) and max(parenchyma) >= ENHANCED_KIDNEY_HU
    fluid_left, thresholds_left = kidney_fluid(ct, left, voxel_ml, enhanced)
    fluid_right, thresholds_right = kidney_fluid(ct, right, voxel_ml, enhanced)
    means = [float(np.mean(pet[k & ~f])) for k, f in ((left, fluid_left), (right, fluid_right)) if (k & ~f).any()]
    reference = max(means) if means else 0.0
    kidneys = dict(
        left=kidney_summary(ct, pet, left, fluid_left, thresholds_left, reference, voxel_ml),
        right=kidney_summary(ct, pet, right, fluid_right, thresholds_right, reference, voxel_ml),
    )

    labels = np.zeros(organs.shape, dtype=np.uint8)
    labels[body_mask(ct, pet, study.spacing_mm)] = LABEL_OF["rest_of_body"]
    labels[np.isin(organs, [organ_names[n] for n in SPINE_LEVELS])] = LABEL_OF["spine_other"]
    labels[np.isin(organs, [organ_names[n] for n in MARROW_LEVELS])] = LABEL_OF["spine_L2_L4"]
    labels[np.isin(organs, [organ_names[n] for n in GI_CLASSES])] = LABEL_OF["gi_tract"]
    labels[blood_pool_mask(organ("aorta"))] = LABEL_OF["blood_pool"]
    labels[organ("urinary_bladder")] = LABEL_OF["urinary_bladder"]
    labels[organ("spleen")] = LABEL_OF["spleen"]
    labels[organ("liver")] = LABEL_OF["liver"]
    labels[left | right] = LABEL_OF["kidney_functional"]
    labels[fluid_left | fluid_right] = LABEL_OF["kidney_nonfunctional"]
    for gland in GLANDS:
        labels[salivary == gland_names[gland]] = LABEL_OF[gland]
    before_lesions = labels.copy()
    labels[study.lesion_label] = LABEL_OF["lesion"]
    sources = dict(kidney_left=left, kidney_right=right, **{c: organ(c) for c in GI_CLASSES},
                   **{c: organ(c) for c in SPINE_LEVELS})
    return labels, before_lesions, enhanced, kidneys, sources


def _region(mask, study, density=None, hu_clip=None):
    """Volume, mass, mean and total %IA of a mask; nominal density if given, otherwise Schneider density of the CT."""
    n = int(mask.sum())
    volume = n * study.voxel_ml
    if n == 0:
        return dict(volume_ml=0.0, mass_g=0.0, mean_pct_ia_ml=None, total_pct_ia=0.0, mean_suv=None)
    pet = study.pet_pct_ia_ml[mask]
    if density is not None:
        mass = volume * density
    else:
        hu = study.ct_hu[mask]
        mass = float(np.sum(density_from_hu(hu if hu_clip is None else np.minimum(hu, hu_clip)))) * study.voxel_ml
    mean = float(np.mean(pet))
    return dict(volume_ml=float(volume), mass_g=float(mass), mean_pct_ia_ml=mean,
                total_pct_ia=float(np.sum(pet, dtype=np.float64)) * study.voxel_ml,
                mean_suv=mean / study.info["suv_to_pct_ia_per_ml"])


def _combine(regions, suv_factor):
    regions = [r for r in regions if r["volume_ml"] > 0]
    if not regions:
        return dict(volume_ml=0.0, mass_g=0.0, mean_pct_ia_ml=None, total_pct_ia=0.0, mean_suv=None)
    volume = sum(r["volume_ml"] for r in regions)
    total = sum(r["total_pct_ia"] for r in regions)
    mean = total / volume
    return dict(volume_ml=volume, mass_g=sum(r["mass_g"] for r in regions), mean_pct_ia_ml=mean,
                total_pct_ia=total, mean_suv=mean / suv_factor)


def _region_for(name, mask, study):
    if name in NOMINAL_DENSITY:
        return dict(_region(mask, study, density=NOMINAL_DENSITY[name]), mass_method=f"ICRP 110 density {NOMINAL_DENSITY[name]} g/mL")
    if name == "gi_tract":
        return dict(_region(mask, study, hu_clip=GI_HU_CLIP), mass_method=f"Schneider density of the CT, HU clipped at {GI_HU_CLIP:g}")
    return dict(_region(mask, study), mass_method="Schneider density of the CT")


def _dilated_kidney_pct_ia(study, labels):
    """Kidney %IA in the mask grown by KIDNEY_DILATION_MM into rest of body only: PET/CT misregistration and spill-out."""
    kidney = np.isin(labels, [LABEL_OF["kidney_functional"], LABEL_OF["kidney_nonfunctional"]])
    if not kidney.any():
        return 0.0
    distance = ndi.distance_transform_edt(~kidney, sampling=study.spacing_mm)
    grown = (distance <= KIDNEY_DILATION_MM) & (kidney | (labels == LABEL_OF["rest_of_body"]))
    return float(np.sum(study.pet_pct_ia_ml[grown], dtype=np.float64)) * study.voxel_ml


def load_qc_tables():
    """Reads the QC tables of notebook 09, indexed by study."""
    flags = pd.read_csv(QC_DIR / "qc_flags.csv").set_index("study_id")["qc_flags"].to_dict()
    volumes = pd.read_csv(QC_DIR / "qc_volumes.csv").set_index("study_id")
    glands = pd.read_csv(QC_DIR / "qc_salivary_glands.csv")
    glands = {study: group.set_index("gland") for study, group in glands.groupby("study_id")}
    return dict(flags=flags, volumes=volumes, glands=glands)


def _salivary_entry(name, study_id, region, qc):
    row = qc["glands"][study_id].loc[name]
    truncated = bool(row["truncated"])
    present = bool(row["present"])
    has_pet = bool(region["total_pct_ia"] > 0)
    return dict(
        region, gland_type="parotid" if name.startswith("parotid") else "submandibular", present=present,
        truncated=truncated, top_gap_mm=None if pd.isna(row["top_gap_mm"]) else float(row["top_gap_mm"]),
        use_for_pbpk=bool(present and not truncated and has_pet),
    )


def build_meta(study, labels, before_lesions, enhanced, kidneys, sources, components, n_lesions, qc):
    voxel_ml = study.voxel_ml
    pet = study.pet_pct_ia_ml
    compartments = {name: dict(label=label, **_region_for(name, labels == label, study)) for label, name in LABELS.items()
                    if name != "lesion"}
    salivary = {g: _salivary_entry(g, study.study_id, compartments[g], qc) for g in GLANDS}

    factor = study.info["suv_to_pct_ia_per_ml"]
    groups = dict(
        kidneys_total=_combine([compartments["kidney_functional"], compartments["kidney_nonfunctional"]], factor),
        salivary_glands_all=_combine([compartments[g] for g in GLANDS], factor),
        salivary_glands_pbpk=_combine([compartments[g] for g in GLANDS if salivary[g]["use_for_pbpk"]], factor),
        parotid_glands=_combine([compartments[g] for g in GLANDS[:2]], factor),
        submandibular_glands=_combine([compartments[g] for g in GLANDS[2:]], factor),
        spine_whole=_combine([compartments["spine_L2_L4"], compartments["spine_other"]], factor),
    )
    lesion_free = labels != LABEL_OF["lesion"]
    gi_parts = {c: _region(sources[c] & lesion_free, study, hu_clip=GI_HU_CLIP) for c in GI_CLASSES}
    groups["kidneys_total"]["dilated_10mm_total_pct_ia"] = _dilated_kidney_pct_ia(study, labels)
    for side in ("left", "right"):
        region = _region(sources[f"kidney_{side}"] & lesion_free, study, density=NOMINAL_DENSITY["kidney_functional"])
        kidneys[side].update(after_lesions_ml=region["volume_ml"], total_pct_ia=region["total_pct_ia"])

    volumes = ndi.sum_labels(components > 0, components, index=np.arange(1, n_lesions + 1)) * voxel_ml if n_lesions else np.array([])
    lesion_totals = ndi.sum_labels(pet, components, index=np.arange(1, n_lesions + 1)) * voxel_ml if n_lesions else np.array([])
    lesion_means = lesion_totals / volumes if n_lesions else np.array([])
    overlap = {LABELS[int(label)]: float(np.sum(before_lesions[study.lesion_label] == label) * voxel_ml)
               for label in np.unique(before_lesions[study.lesion_label]) if label != 0}
    outside = before_lesions == 0
    outside &= ~study.lesion_label
    fov_total = float(np.sum(pet, dtype=np.float64)) * voxel_ml
    no_body = ~labels.any(axis=(0, 1))
    outside_total = float(np.sum(pet[outside], dtype=np.float64)) * voxel_ml
    lesion_region = dict(
        label=LABEL_OF["lesion"], n=int(n_lesions), n_ge_1ml=int(np.sum(volumes >= 1.0)),
        total_volume_ml=float(volumes.sum()), total_pct_ia=float(lesion_totals.sum()),
        mean_pct_ia_ml=float(lesion_totals.sum() / volumes.sum()) if n_lesions else None,
        volume_ml_by_id=[round(float(v), 3) for v in volumes],
        total_pct_ia_by_id=[round(float(v), 5) for v in lesion_totals],
        mean_pct_ia_ml_by_id=[round(float(v), 6) for v in lesion_means],
        overlap_ml_by_compartment=overlap,
    )
    body_mass_g = sum(c["mass_g"] for c in compartments.values()) + float(
        np.sum(density_from_hu(study.ct_hu[study.lesion_label]))) * voxel_ml

    nb09_flags = qc["flags"].get(study.study_id, "")
    nb09_flags = [] if not isinstance(nb09_flags, str) else [f.strip() for f in nb09_flags.split(";") if f.strip()]
    row = qc["volumes"].loc[study.study_id]
    cut = [c[:-4] for c in row.index if c.endswith("_cut") and bool(row[c])]
    vertebrae_ml = {c: float(sources[c].sum() * voxel_ml) for c in SPINE_LEVELS}
    functional = {side: kidneys[side]["functional_ml"] for side in ("left", "right")}
    flags = dict(
        single_kidney=[side for side, v in functional.items() if v < SINGLE_KIDNEY_ML],
        small_kidney=[side for side, v in functional.items() if SINGLE_KIDNEY_ML <= v < SMALL_KIDNEY_ML],
        kidney_absent_in_segmentation=[side for side in functional if not kidneys[side]["present"]],
        kidney_cyst=[side for side in functional if kidneys[side]["cyst"]],
        kidney_mostly_nonfunctional=[side for side in functional if kidneys[side]["total_ml"] > 0
                                     and kidneys[side]["nonfunctional_ml"] > 0.5 * kidneys[side]["total_ml"]],
        kidney_outlier=[side for side in functional if f"kidney_{side} outlier" in nb09_flags],
        salivary_truncated=[g for g in GLANDS if salivary[g]["truncated"]],
        salivary_absent=[g for g in GLANDS if not salivary[g]["present"]],
        salivary_no_pet=[g for g in GLANDS if salivary[g]["present"] and salivary[g]["total_pct_ia"] == 0],
        contrast_metadata_mismatch=bool(enhanced != study.info["ct_contrast"]),
        blood_pool_small=bool(compartments["blood_pool"]["volume_ml"] < MIN_BLOOD_POOL_ML),
        marrow_L2_L4_incomplete=[c for c in MARROW_LEVELS if vertebrae_ml[c] < MIN_MARROW_VERTEBRA_ML],
        spine_levels_absent=[c for c in SPINE_LEVELS if vertebrae_ml[c] == 0],
        organ_cut_by_fov=cut,
        compartment_empty=[name for name, c in compartments.items() if c["volume_ml"] < 1.0 and name != "kidney_nonfunctional"],
        pet_outside_body=bool(outside_total > OUTSIDE_BODY_FLAG_PCT),
        cropped_not_whole_body=bool(study.k_min > 0),
        nb09_flags=nb09_flags,
    )
    return dict(
        study_id=study.study_id, subject=study.info["subject"], tracer=study.info["tracer"],
        pbpk_cohort=study.info["pbpk_cohort"], scanner=study.info["scanner"], weight_kg=study.info["weight_kg"],
        height_m=study.info["height_m"], injected_activity_mbq=study.info["activity_mbq"],
        half_life_s=study.info["half_life_s"], uptake_time_min=study.info["uptake_min"],
        ct_contrast_metadata=study.info["ct_contrast"], ct_kidney_enhanced=enhanced,
        suv_to_pct_ia_per_ml=study.info["suv_to_pct_ia_per_ml"],
        grid=dict(shape=[int(n) for n in labels.shape], spacing_mm=list(study.spacing_mm), voxel_ml=voxel_ml,
                  k_min_of_original=study.k_min, nz_original=study.info["nz_original"]),
        compartments=compartments, groups=groups, gi_tract_parts=gi_parts, kidneys=kidneys, salivary_glands=salivary,
        lesions=lesion_region,
        totals=dict(fov_pct_ia=fov_total, body_pct_ia=fov_total - outside_total, outside_body_pct_ia=outside_total,
                    slices_without_body=int(no_body.sum()),
                    pct_ia_in_slices_without_body=float(np.sum(pet[:, :, no_body], dtype=np.float64)) * voxel_ml,
                    body_mass_kg=body_mass_g / 1000.0, body_mass_over_weight=body_mass_g / 1000.0 / study.info["weight_kg"]),
        qc_flags=flags,
    )


def _save(array, affine, path, dtype):
    image = nib.Nifti1Image(array.astype(dtype), affine)
    image.header.set_data_dtype(dtype)
    image.set_sform(affine, code=1)
    image.set_qform(affine, code=1)
    nib.save(image, path)


def build_phantom(study_id, table, qc, root=PHANTOMS, overwrite=False):
    """Writes phantoms/<study_id>/ through a temporary folder: a final folder is always complete."""
    final = Path(root) / study_id
    if (final / "meta.json").exists() and not overwrite:
        return "exists"
    temporary = Path(root) / f".tmp_{study_id}_{os.getpid()}"
    temporary.mkdir(parents=True, exist_ok=True)
    study = psma_data.load_study(study_id, table)
    labels, before_lesions, enhanced, kidneys, sources = compartment_map(study)
    components, n_lesions = psma_data.lesion_components(study.lesion_label)
    meta = build_meta(study, labels, before_lesions, enhanced, kidneys, sources, components, n_lesions, qc)
    _save(np.rint(study.ct_hu), study.affine, temporary / "ct.nii.gz", np.int16)
    _save(study.pet_pct_ia_ml, study.affine, temporary / "pet_pct_ia_ml.nii.gz", np.float32)
    _save(labels, study.affine, temporary / "organs.nii.gz", np.uint8)
    _save(components, study.affine, temporary / "lesions.nii.gz", np.uint16)
    (temporary / "meta.json").write_text(json.dumps(meta, indent=1))
    if final.exists():
        for f in final.iterdir():
            f.unlink()
        final.rmdir()
    os.replace(temporary, final)
    return "built"


_shared = {}


def _init_worker(root):
    _shared.update(table=psma_data.study_table(), qc=load_qc_tables(), root=root)


def _worker(study_id):
    start = time.time()
    try:
        status = build_phantom(study_id, _shared["table"], _shared["qc"], _shared["root"])
        return dict(study_id=study_id, status=status, seconds=time.time() - start, error="")
    except Exception as error:
        return dict(study_id=study_id, status="error", seconds=time.time() - start, error=f"{type(error).__name__}: {error}")


def build_all(study_ids, root=PHANTOMS, processes=6):
    """Builds the missing phantoms in parallel; a failure is logged and does not stop the others."""
    write_labelmap(root)
    with multiprocessing.Pool(processes, initializer=_init_worker, initargs=(root,)) as pool:
        return pd.DataFrame(list(pool.imap_unordered(_worker, list(study_ids))))


def write_labelmap(root=PHANTOMS):
    Path(root).mkdir(parents=True, exist_ok=True)
    groups = dict(kidneys=[1, 2], salivary_glands=[5, 6, 7, 8], whole_spine=[12, 13], red_marrow_surrogate=[12])
    content = dict(background=0, labels={str(k): v for k, v in LABELS.items()}, label_groups=groups,
                   lesion_priority="lesion voxels have label 15 here and their component number in lesions.nii.gz")
    (Path(root) / "labelmap.json").write_text(json.dumps(content, indent=1))


def summary_table(root=PHANTOMS):
    """One row per phantom, flattened from the meta.json files."""
    rows = []
    for path in sorted(Path(root).glob("psma_*/meta.json")):
        meta = json.loads(path.read_text())
        row = {k: meta[k] for k in ("study_id", "subject", "tracer", "pbpk_cohort", "scanner", "weight_kg",
                                    "injected_activity_mbq", "uptake_time_min", "ct_contrast_metadata", "ct_kidney_enhanced")}
        for section in ("compartments", "groups", "gi_tract_parts"):
            for name, values in meta[section].items():
                for key in ("volume_ml", "mass_g", "mean_pct_ia_ml", "total_pct_ia", "mean_suv"):
                    row[f"{name}__{key}"] = values[key]
        row["kidneys_total__dilated_10mm_total_pct_ia"] = meta["groups"]["kidneys_total"]["dilated_10mm_total_pct_ia"]
        for side, values in meta["kidneys"].items():
            for key in ("total_ml", "functional_ml", "nonfunctional_ml", "parenchyma_hu", "largest_cold_fluid_ml", "total_pct_ia",
                        "functional_mean_pct_ia_ml", "fluid_mean_pct_ia_ml"):
                row[f"kidney_{side}__{key}"] = values.get(key)
            for threshold, volume in values["functional_ml_by_threshold"].items():
                row[f"kidney_{side}__functional_ml_thr{threshold}"] = volume
        for gland, values in meta["salivary_glands"].items():
            row[f"{gland}__truncated"] = values["truncated"]
            row[f"{gland}__use_for_pbpk"] = values["use_for_pbpk"]
        lesions = meta["lesions"]
        row.update(n_lesions=lesions["n"], n_lesions_ge_1ml=lesions["n_ge_1ml"], lesions_volume_ml=lesions["total_volume_ml"],
                   lesions_total_pct_ia=lesions["total_pct_ia"])
        row.update({f"totals__{k}": v for k, v in meta["totals"].items()})
        for flag, value in meta["qc_flags"].items():
            row[f"flag__{flag}"] = ";".join(value) if isinstance(value, list) else value
        rows.append(row)
    return pd.DataFrame(rows)
