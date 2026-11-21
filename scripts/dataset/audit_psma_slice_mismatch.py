#!/usr/bin/env python3
"""Classify every NIfTI slice of the study whose DICOM series has fewer instances than the NIfTI has slices."""
import io
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_psma_pet_headers import BUCKET_URL, ROOT, Throttle, http_get

STUDY = "psma_3959d1c381a5bcd6_2015-11-12"
sys.path.insert(0, str(ROOT / "src"))
from paths import DATASET  # noqa: E402

DATA = DATASET / "PSMA-PET-CT-Lesions_v3"
OUTPUT = ROOT / "data/processed/psma/psma_slice_audit_3959.csv"
INTERPOLATION_TOLERANCE_SUV = 1e-3


def main():
    study = pd.read_csv(ROOT / "data/processed/psma/pet_dicom_headers.csv").query("study_id == @STUDY").iloc[0]
    nii = nib.load(DATA / f"imagesTr/{STUDY}_0001.nii.gz")
    pet = np.asanyarray(nii.dataobj).astype(float)
    label = np.asanyarray(nib.load(DATA / f"labelsTr/{STUDY}.nii.gz").dataobj)
    inverse = np.linalg.inv(nii.affine)
    session, throttle = requests.Session(), Throttle(0.2)
    prefix = study["series_aws_url"].split("/")[3]
    xml = http_get(session, throttle, BUCKET_URL + "/", params={"list-type": "2", "prefix": f"{prefix}/"}).text
    dicom = {}
    for key in re.findall(r"<Key>([^<]+)</Key>", xml):
        ds = pydicom.dcmread(io.BytesIO(http_get(session, throttle, f"{BUCKET_URL}/{key}").content))
        x, y, z = (float(v) for v in ds.ImagePositionPatient)
        k = int(np.rint((inverse[:3, :3] @ np.array([-x, -y, z]) + inverse[:3, 3])[2]))
        bqml = ds.pixel_array.astype(float) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        dicom[k] = dict(instance=int(ds.InstanceNumber), z_mm=z, bqml_sum=float(bqml.sum()))
    have = sorted(dicom)
    rows = []
    for k in range(pet.shape[2]):
        slice_k = pet[..., k]
        row = dict(k=k, z_mm=float(nii.affine[2, 2] * k + nii.affine[2, 3]), suv_max=float(slice_k.max()),
                   suv_sum=float(slice_k.sum()), label_voxels=int(label[..., k].sum()), max_error_interpolation="", duplicate_of="")
        row.update(dicom.get(k, {}))
        if k in dicom:
            row["class"] = "dicom"
        elif not slice_k.any():
            row["class"] = "empty"
        else:
            below = max((h for h in have if h < k), default=None)
            above = min((h for h in have if h > k), default=None)
            error = np.inf
            if below is not None and above is not None:
                weight = (k - below) / (above - below)
                error = float(np.abs(slice_k - (1 - weight) * pet[..., below] - weight * pet[..., above]).max())
                row["max_error_interpolation"] = error
            twins = [o for o in range(pet.shape[2]) if o != k and np.abs(slice_k - pet[..., o]).max() < INTERPOLATION_TOLERANCE_SUV]
            row["duplicate_of"] = ";".join(map(str, twins))
            row["class"] = "duplicate" if twins else "interpolated" if error < INTERPOLATION_TOLERANCE_SUV else "unexplained"
        rows.append(row)
    audit = pd.DataFrame(rows)
    audit.to_csv(OUTPUT, index=False)
    print(audit["class"].value_counts().to_dict())
    print("first k of the contiguous DICOM block:", next(k for k in have if all(o in dicom for o in range(k, pet.shape[2]))))
    print("label voxels below it:", int(audit.loc[audit["k"] < 84, "label_voxels"].sum()), "of", int(audit["label_voxels"].sum()))
    print("max interpolation error:", pd.to_numeric(audit["max_error_interpolation"]).max())
    print("share of SUV sum below k 84:", round(audit.loc[audit["k"] < 84, "suv_sum"].sum() / audit["suv_sum"].sum(), 4))


if __name__ == "__main__":
    main()
