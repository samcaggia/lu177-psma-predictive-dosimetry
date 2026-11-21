#!/usr/bin/env python3
"""Compare the SUV of the PSMA NIfTI release with Bq/mL from full DICOM PET instances of the IDC bucket.

Instances are parsed in memory. Slices are paired voxel by voxel through the patient coordinates.
ratio = SUV_nifti / (C[Bq/mL] * weight[g] / injected activity[Bq]); 1 means no decay correction of the
injected activity, exp(lambda * t_uptake) means decay correction to the scan start.
"""
import argparse
import io
import math
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_psma_pet_headers import BUCKET_URL, Throttle, http_get

ROOT = Path(__file__).resolve().parents[2]
HEADERS = ROOT / "data/processed/psma/pet_dicom_headers.csv"
sys.path.insert(0, str(ROOT / "src"))
from paths import DATASET  # noqa: E402

IMAGES = DATASET / "PSMA-PET-CT-Lesions_v3" / "imagesTr"
OUTPUT = ROOT / "data/processed/psma/suv_definition_check.csv"
COMBINATIONS = [("68Ga", "Biograph64"), ("68Ga", "Biograph20_mCT"), ("68Ga", "Discovery690"),
                ("18F", "Biograph64"), ("18F", "Biograph20_mCT"), ("18F", "Discovery690")]
ANOMALOUS = ["psma_838c9340e85ca431_2015-09-11", "psma_6d5c002d629eb131_2022-06-04",
             "psma_c80781b803041d12_2018-07-14", "psma_3959d1c381a5bcd6_2015-11-12"]
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])


def list_keys(session, throttle, prefix):
    params = {"list-type": "2", "prefix": f"{prefix}/"}
    xml = http_get(session, throttle, BUCKET_URL + "/", params=params).text
    return re.findall(r"<Key>([^<]+)</Key>", xml)


def compare_slice(ds, nii, study, inverse):
    bqml = ds.pixel_array.astype(float) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    rows, cols = bqml.shape
    origin = np.array(ds.ImagePositionPatient, dtype=float)
    orientation = np.array(ds.ImageOrientationPatient, dtype=float)
    dy, dx = (float(v) for v in ds.PixelSpacing)
    jj, ii = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    lps = origin + ii[..., None] * dx * orientation[:3] + jj[..., None] * dy * orientation[3:]
    ras = lps @ LPS_TO_RAS
    voxel = ras @ inverse[:3, :3].T + inverse[:3, 3]
    index = np.rint(voxel).astype(int)
    grid_error = float(np.abs(voxel - index).max())
    k = int(np.unique(index[..., 2])[0])
    if np.ptp(index[..., 2]) or not 0 <= k < nii.shape[2]:
        return dict(nifti_k=k, grid_error_vox=grid_error, status="no_slice")
    inside = (index[..., 0] >= 0) & (index[..., 0] < nii.shape[0]) & (index[..., 1] >= 0) & (index[..., 1] < nii.shape[1])
    suv_slice = np.asarray(nii.dataobj[..., k], dtype=float)
    suv = np.full(bqml.shape, np.nan)
    suv[inside] = suv_slice[index[..., 0][inside], index[..., 1][inside]]

    predicted = bqml * study.weight_kg * 1000.0 / (study.activity_mbq * 1e6)
    mask = inside & (predicted > max(0.05 * np.nanmax(predicted), 0.05))
    ratio = suv[mask] / predicted[mask]
    fit = float(np.sum(suv[mask] * predicted[mask]) / np.sum(predicted[mask] ** 2))
    return dict(nifti_k=k, grid_error_vox=grid_error, status="ok", z_mm=float(origin[2]), n_voxels=int(mask.sum()),
                suv_max=float(np.nanmax(suv)), ratio_median=float(np.median(ratio)), ratio_p01=float(np.quantile(ratio, 0.01)),
                ratio_p99=float(np.quantile(ratio, 0.99)), ratio_lstsq=fit, slope=float(ds.RescaleSlope))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=5)
    parser.add_argument("--min-interval", type=float, default=0.2)
    args = parser.parse_args()

    studies = pd.read_csv(HEADERS)
    regular = studies[studies["uptake_series_min"].between(55, 75) & studies["plausibility_flags"].isna()]
    chosen = [regular[(regular["tracer"] == t) & (regular["scanner"] == s)].iloc[0]["study_id"] for t, s in COMBINATIONS]
    session = requests.Session()
    throttle = Throttle(args.min_interval)
    rows = []
    for study_id in chosen + ANOMALOUS:
        study = studies[studies["study_id"] == study_id].iloc[0]
        nii = nib.load(IMAGES / f"{study_id}_0001.nii.gz")
        inverse = np.linalg.inv(nii.affine)
        keys = sorted(list_keys(session, throttle, study["series_aws_url"].split("/")[3]))
        picks = [keys[i] for i in np.linspace(0, len(keys) - 1, args.instances).astype(int)]
        for key in picks:
            ds = pydicom.dcmread(io.BytesIO(http_get(session, throttle, f"{BUCKET_URL}/{key}").content))
            row = dict(study_id=study_id, group="anomalous" if study_id in ANOMALOUS else "regular", tracer=study["tracer"],
                       scanner=study["scanner"], weight_kg=study["weight_kg"], activity_mbq=study["activity_mbq"],
                       uptake_series_min=study["uptake_series_min"], half_life_s=study["half_life_s"], object_key=key)
            lam = math.log(2) / study["half_life_s"]
            row["expected_decay_ratio"] = math.exp(lam * study["uptake_series_min"] * 60)
            row.update(compare_slice(ds, nii, study, inverse))
            rows.append(row)
            print({k: row.get(k) for k in ("study_id", "status", "nifti_k", "grid_error_vox", "ratio_median", "ratio_p01", "ratio_p99", "expected_decay_ratio")}, flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT, index=False)
    summary = result[result["status"] == "ok"].groupby(["group", "study_id", "tracer", "scanner"], sort=False).agg(
        slices=("ratio_median", "size"), ratio=("ratio_median", "median"), ratio_min=("ratio_p01", "min"), ratio_max=("ratio_p99", "max"),
        expected=("expected_decay_ratio", "first"), weight=("weight_kg", "first"), uptake=("uptake_series_min", "first"), grid_error=("grid_error_vox", "max"))
    summary["ratio_over_expected"] = summary["ratio"] / summary["expected"]
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
