#!/usr/bin/env python3
"""Fetch PET DICOM header fields of the PSMA-PET-CT-Lesions studies from the IDC public bucket.

One instance per PT series is read in memory (HTTP Range first, full object as fallback); nothing is
written to disk except the output CSV, which is checkpointed atomically and reused on restart.
IDC study dates are shifted from those of the NIfTI release by one constant, derived and checked here.
"""
import argparse
import io
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import requests
from idc_index import IDCClient

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "data/processed/psma/psma_inventory.csv"
OUTPUT = ROOT / "data/processed/psma/pet_dicom_headers.csv"
EXCLUSIONS = ROOT / "data/processed/psma/psma_pet_exclusions.csv"
BUCKET_URL = "https://idc-open-data.s3.amazonaws.com"
COLLECTION = "psma_pet_ct_lesions"
PIXEL_DATA_TAG = b"\xe0\x7f\x10\x00"
EXPECTED_HALF_LIFE_S = {"18F": 6586.2, "68Ga": 4062.6}
EXPECTED_ACTIVITY_MBQ = {"18F": 246.0, "68Ga": 214.0}
EXPECTED_UPTAKE_MIN = 74.0
RANGES = {
    "weight_kg": (30.0, 200.0),
    "height_m": (1.2, 2.2),
    "activity_mbq": (20.0, 1000.0),
    "uptake_series_min": (30.0, 240.0),
}
PBPK_TRACER = "68Ga"

HEADER_COLUMNS = [
    "PatientWeight", "PatientSize", "PatientAge", "RadionuclideTotalDose",
    "RadiopharmaceuticalStartDateTime", "RadiopharmaceuticalStartTime", "RadionuclideHalfLife",
    "RadionuclidePositronFraction", "Radiopharmaceutical", "RadionuclideCode",
    "AcquisitionDate", "AcquisitionTime", "SeriesDate", "SeriesTime", "StudyDate_header", "StudyTime",
    "Units", "DecayCorrection", "DecayFactor", "CorrectedImage", "FrameReferenceTime",
    "ActualFrameDuration", "InstanceNumber", "ManufacturerModelName_header",
]
FETCH_COLUMNS = ["SeriesInstanceUID", "fetch_status", "fetch_error", "object_key", "object_size",
                 "bytes_read", "used_full_file"]


class Throttle:
    def __init__(self, min_interval):
        self.min_interval = min_interval
        self.last = 0.0

    def wait(self):
        delay = self.last + self.min_interval - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.last = time.monotonic()


def http_get(session, throttle, url, headers=None, params=None, retries=5):
    for attempt in range(retries):
        throttle.wait()
        try:
            response = session.get(url, headers=headers, params=params, timeout=60)
        except requests.RequestException:
            if attempt == retries - 1:
                raise
        else:
            if response.status_code in (200, 206):
                return response
            if response.status_code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                response.raise_for_status()
        time.sleep(2.0 ** (attempt + 1))
    raise RuntimeError(f"no response from {url}")


def first_object(session, throttle, prefix):
    params = {"list-type": "2", "prefix": f"{prefix}/", "max-keys": "1"}
    xml = http_get(session, throttle, BUCKET_URL + "/", params=params).text
    key = re.search(r"<Key>([^<]+)</Key>", xml)
    size = re.search(r"<Size>(\d+)</Size>", xml)
    if key is None or size is None:
        raise ValueError(f"empty prefix {prefix}")
    return key.group(1), int(size.group(1))


def text(value):
    if value is None:
        return ""
    if isinstance(value, pydicom.multival.MultiValue):
        return "|".join(str(v) for v in value)
    return str(value)


def parse_header(buffer):
    ds = pydicom.dcmread(io.BytesIO(buffer), stop_before_pixels=True)
    sequence = ds.get("RadiopharmaceuticalInformationSequence")
    item = sequence[0] if sequence else pydicom.Dataset()
    codes = item.get("RadionuclideCodeSequence")
    return {
        "PatientWeight": text(ds.get("PatientWeight")),
        "PatientSize": text(ds.get("PatientSize")),
        "PatientAge": text(ds.get("PatientAge")),
        "RadionuclideTotalDose": text(item.get("RadionuclideTotalDose")),
        "RadiopharmaceuticalStartDateTime": text(item.get("RadiopharmaceuticalStartDateTime")),
        "RadiopharmaceuticalStartTime": text(item.get("RadiopharmaceuticalStartTime")),
        "RadionuclideHalfLife": text(item.get("RadionuclideHalfLife")),
        "RadionuclidePositronFraction": text(item.get("RadionuclidePositronFraction")),
        "Radiopharmaceutical": text(item.get("Radiopharmaceutical")),
        "RadionuclideCode": text(codes[0].get("CodeMeaning")) if codes else "",
        "AcquisitionDate": text(ds.get("AcquisitionDate")),
        "AcquisitionTime": text(ds.get("AcquisitionTime")),
        "SeriesDate": text(ds.get("SeriesDate")),
        "SeriesTime": text(ds.get("SeriesTime")),
        "StudyDate_header": text(ds.get("StudyDate")),
        "StudyTime": text(ds.get("StudyTime")),
        "Units": text(ds.get("Units")),
        "DecayCorrection": text(ds.get("DecayCorrection")),
        "DecayFactor": text(ds.get("DecayFactor")),
        "CorrectedImage": text(ds.get("CorrectedImage")),
        "FrameReferenceTime": text(ds.get("FrameReferenceTime")),
        "ActualFrameDuration": text(ds.get("ActualFrameDuration")),
        "InstanceNumber": text(ds.get("InstanceNumber")),
        "ManufacturerModelName_header": text(ds.get("ManufacturerModelName")),
    }


def fetch_series(session, throttle, series_url, range_bytes):
    prefix = series_url.split("/")[3]
    key, size = first_object(session, throttle, prefix)
    url = f"{BUCKET_URL}/{key}"
    buffer = http_get(session, throttle, url, headers={"Range": f"bytes=0-{range_bytes - 1}"}).content
    truncated = len(buffer) < size
    used_full = False
    fields = None
    if not truncated or PIXEL_DATA_TAG in buffer:
        try:
            fields = parse_header(buffer)
        except Exception:
            fields = None
    if fields is None or (truncated and not (fields["PatientWeight"] and fields["RadionuclideTotalDose"])):
        if truncated:
            buffer = http_get(session, throttle, url).content
            used_full = True
        fields = parse_header(buffer)
    fields.update(object_key=key, object_size=size, bytes_read=len(buffer), used_full_file=used_full)
    return fields


def parse_dt(value):
    match = re.match(r"^(\d{8})(\d{2})(\d{2})(\d{2})(\.\d+)?", str(value).strip())
    if match is None:
        return pd.NaT
    date, hour, minute, second, fraction = match.groups()
    return pd.to_datetime(f"{date} {hour}:{minute}:{second}{fraction or ''}", format="mixed", errors="coerce")


def combine(date, tm):
    date, tm = str(date).strip(), str(tm).strip()
    return parse_dt(date + tm) if re.fullmatch(r"\d{8}", date) and re.match(r"^\d{6}", tm) else pd.NaT


def positive(series):
    values = pd.to_numeric(series, errors="coerce")
    return values.where(values > 0)


def load_index():
    index = IDCClient().index
    pt = index[(index["collection_id"] == COLLECTION) & (index["Modality"] == "PT")].copy()
    pt["subject"] = pt["PatientID"].str.lower()
    pt["idc_date"] = pd.to_datetime(pt["StudyDate"])
    return pt[["SeriesInstanceUID", "PatientID", "subject", "StudyDate", "idc_date", "series_aws_url",
               "instanceCount", "ManufacturerModelName"]].reset_index(drop=True)


def date_offset(pt, inventory):
    pairs = pt[["subject", "idc_date"]].merge(
        inventory.assign(nifti_date=pd.to_datetime(inventory["study_date"]))[["subject", "nifti_date"]],
        on="subject")
    return int((pairs["nifti_date"] - pairs["idc_date"]).dt.days.mode().iloc[0])


def assemble(raw, pt, inventory, offset):
    df = pt.merge(raw, on="SeriesInstanceUID", how="left")
    df = df.rename(columns={"StudyDate": "idc_study_date", "instanceCount": "idc_instance_count",
                            "ManufacturerModelName": "idc_model"})
    df["date_offset_days"] = offset
    df["study_id"] = df["subject"] + "_" + (df["idc_date"] + pd.Timedelta(days=offset)).dt.strftime("%Y-%m-%d")
    meta = inventory[["study_id", "tracer", "nz", "scanner", "age"]]
    df = df.merge(meta, on="study_id", how="left", indicator=True)
    df["map_status"] = np.where(df["_merge"] == "both", "matched", "unmatched")
    df = df.drop(columns=["_merge", "idc_date"])

    df["weight_kg"] = positive(df["PatientWeight"])
    df["height_m"] = positive(df["PatientSize"])
    df["activity_mbq"] = positive(df["RadionuclideTotalDose"]) / 1e6
    df["half_life_s"] = positive(df["RadionuclideHalfLife"])

    series_dt = pd.Series([combine(a, b) for a, b in zip(df["SeriesDate"], df["SeriesTime"])], dtype="datetime64[ns]")
    acq_dt = pd.Series([combine(a, b) for a, b in zip(df["AcquisitionDate"], df["AcquisitionTime"])], dtype="datetime64[ns]")
    inj_full = pd.Series([parse_dt(v) for v in df["RadiopharmaceuticalStartDateTime"]], dtype="datetime64[ns]")
    inj_time = pd.Series([combine(a, b) for a, b in zip(df["SeriesDate"], df["RadiopharmaceuticalStartTime"])], dtype="datetime64[ns]")
    injection = inj_full.fillna(inj_time)
    df["injection_source"] = np.select([inj_full.notna(), inj_time.notna()], ["start_datetime", "start_time"], "none")
    df["uptake_series_min"] = (series_dt - injection).dt.total_seconds() / 60
    df["uptake_acquisition_min"] = (acq_dt - injection).dt.total_seconds() / 60
    df["uptake_time_of_day_min"] = (series_dt - inj_time).dt.total_seconds() / 60

    frame_s = pd.to_numeric(df["ActualFrameDuration"], errors="coerce") / 1000
    log_factor = np.log(positive(df["DecayFactor"]))
    decay_reference = (acq_dt - series_dt).dt.total_seconds() + frame_s / 2 - log_factor * df["half_life_s"] / math.log(2)
    df["decay_reference_minus_series_s"] = decay_reference

    low_uptake = RANGES["uptake_series_min"][0]
    df["flag_implausible_uptake"] = df["uptake_series_min"] < low_uptake
    exclusions = pd.read_csv(EXCLUSIONS)
    df["suv_excluded"] = df["study_id"].isin(exclusions.loc[exclusions["action"] == "exclude", "study_id"])
    df["pet_valid_k_min"] = df["study_id"].map(exclusions.dropna(subset=["valid_k_min"]).set_index("study_id")["valid_k_min"]).fillna(0).astype(int)
    df["pbpk_cohort"] = (df["tracer"] == PBPK_TRACER) & ~df["suv_excluded"]
    df["suv_decay_factor"] = np.exp(np.log(2) * df["uptake_series_min"] * 60 / df["half_life_s"])
    df["suv_to_pct_ia_per_ml"] = 100 / (df["weight_kg"] * 1000 * df["suv_decay_factor"])
    df["suv_to_pct_ia_per_ml_decay_corrected"] = (100 / (df["weight_kg"] * 1000)).where(~df["flag_implausible_uptake"])

    df["has_weight"] = df["weight_kg"].notna()
    df["has_activity"] = df["activity_mbq"].notna()

    flags = []
    for row in df.itertuples():
        row_flags = []
        for column, (low, high) in RANGES.items():
            value = getattr(row, column)
            if pd.notna(value) and not low <= value <= high:
                row_flags.append(f"{column}_range")
        expected = EXPECTED_HALF_LIFE_S.get(row.tracer)
        if expected and pd.notna(row.half_life_s) and abs(row.half_life_s / expected - 1) > 0.02:
            row_flags.append("half_life_vs_tracer")
        if row.Units != "BQML":
            row_flags.append("units")
        if row.DecayCorrection != "START":
            row_flags.append("decay_correction")
        if "DECY" not in str(row.CorrectedImage) or "ATTN" not in str(row.CorrectedImage):
            row_flags.append("corrected_image")
        if pd.notna(row.nz) and row.idc_instance_count != row.nz:
            row_flags.append("instances_vs_nz")
        flags.append(";".join(row_flags))
    df["plausibility_flags"] = flags
    return df.sort_values(["study_id", "SeriesInstanceUID"]).reset_index(drop=True)


def save(df, path):
    temporary = path.with_suffix(".csv.part")
    df.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_raw(path):
    columns = FETCH_COLUMNS + HEADER_COLUMNS
    if not path.exists():
        return pd.DataFrame(columns=columns, dtype=str)
    return pd.read_csv(path, dtype=str, keep_default_na=False)[columns]


def report(df, inventory):
    print(f"\nPT series in IDC: {len(df)}; offset IDC -> NIfTI dates: {df['date_offset_days'].iloc[0]} days")
    print("Fetch status:", df["fetch_status"].fillna("pending").value_counts().to_dict())
    print("Map status:", df["map_status"].value_counts().to_dict())
    print("Unmatched series:", df.loc[df["map_status"] == "unmatched", ["PatientID", "idc_study_date"]].values.tolist())
    print("Inventory studies without a series:", sorted(set(inventory["study_id"]) - set(df["study_id"])))
    print("Studies with more than one series:", int((df["study_id"].value_counts() > 1).sum()))
    fetched = df[df["fetch_status"] == "ok"]
    if fetched.empty:
        return
    print(f"Fetched {len(fetched)}; full-file fallback {(fetched['used_full_file'].astype(str) == 'True').sum()}")
    print(f"Weight: {fetched['has_weight'].sum()}; activity: {fetched['has_activity'].sum()}; "
          f"both: {(fetched['has_weight'] & fetched['has_activity']).sum()}; height: {fetched['height_m'].notna().sum()}")
    print("Injection source:", fetched["injection_source"].value_counts().to_dict())
    print("Units:", fetched["Units"].value_counts().to_dict(), "DecayCorrection:", fetched["DecayCorrection"].value_counts().to_dict())
    print("Radiopharmaceutical:", fetched["Radiopharmaceutical"].value_counts(dropna=False).to_dict())
    print("Tracer vs radionuclide code:\n", pd.crosstab(fetched["tracer"], fetched["RadionuclideCode"]).to_string())
    print("Half-life by tracer:\n", fetched.groupby("tracer")["half_life_s"].describe().round(1).to_string())
    columns = ["activity_mbq", "uptake_series_min", "uptake_acquisition_min", "uptake_time_of_day_min", "weight_kg"]
    print(fetched.groupby("tracer")[columns].agg(["count", "mean", "median", "std", "min", "max"]).round(1).T.to_string())
    print("By tracer and scanner:\n", fetched.groupby(["tracer", "scanner"])[columns[:2] + ["weight_kg"]].agg(["count", "mean"]).round(1).to_string())
    for tracer, activity in EXPECTED_ACTIVITY_MBQ.items():
        observed = fetched.loc[fetched["tracer"] == tracer, "activity_mbq"].mean()
        print(f"Activity {tracer}: mean {observed:.1f} MBq, description {activity:.0f} MBq, ratio {observed / activity:.3f}")
    print(f"Uptake (series time): mean {fetched['uptake_series_min'].mean():.1f} min, median {fetched['uptake_series_min'].median():.1f}, description {EXPECTED_UPTAKE_MIN:.0f} min")
    print("Decay reference minus series time (s):", fetched["decay_reference_minus_series_s"].describe().round(1).to_dict())
    print("By scanner:\n", fetched.groupby("scanner")["decay_reference_minus_series_s"].describe().round(1).to_string())
    print("Age difference header - inventory:", (pd.to_numeric(fetched["PatientAge"].str.extract(r"(\d+)")[0]) - fetched["age"]).value_counts().to_dict())
    print("Model header == IDC index:", (fetched["ManufacturerModelName_header"] == fetched["idc_model"]).sum())
    flagged = fetched[fetched["plausibility_flags"] != ""]
    print(f"Flagged studies: {len(flagged)}", flagged["plausibility_flags"].str.split(";").explode().value_counts().to_dict())
    print(flagged[["study_id", "tracer", "weight_kg", "height_m", "activity_mbq", "uptake_series_min", "uptake_time_of_day_min", "plausibility_flags"]].head(40).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--limit", type=int, help="maximum series to fetch in this run")
    parser.add_argument("--min-interval", type=float, default=0.2, help="seconds between requests")
    parser.add_argument("--range-bytes", type=int, default=128 * 1024)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()

    inventory = pd.read_csv(INVENTORY)
    pt = load_index()
    offset = date_offset(pt, inventory)
    raw = load_raw(args.output)
    done = set(raw.loc[raw["fetch_status"] == "ok", "SeriesInstanceUID"])
    raw = raw[raw["SeriesInstanceUID"].isin(done)]
    todo = pt[~pt["SeriesInstanceUID"].isin(done)].sort_values("SeriesInstanceUID")
    if args.limit is not None:
        todo = todo.head(args.limit)
    print(f"{len(pt)} PT series, {len(done)} already fetched, {len(todo)} to fetch", flush=True)

    session = requests.Session()
    throttle = Throttle(args.min_interval)
    rows = []

    def checkpoint():
        current = pd.concat([raw, pd.DataFrame(rows, columns=FETCH_COLUMNS + HEADER_COLUMNS, dtype=object)], ignore_index=True)
        save(assemble(current.astype(str).replace({"None": "", "nan": ""}), pt, inventory, offset), args.output)

    for number, series in enumerate(todo.itertuples(), start=1):
        row = {"SeriesInstanceUID": series.SeriesInstanceUID}
        try:
            row.update(fetch_series(session, throttle, series.series_aws_url, args.range_bytes))
            row.update(fetch_status="ok", fetch_error="")
        except Exception as error:
            row.update(fetch_status="error", fetch_error=f"{type(error).__name__}: {error}"[:300])
            print(f"ERROR {series.SeriesInstanceUID}: {row['fetch_error']}", flush=True)
        rows.append(row)
        if number % args.checkpoint_every == 0:
            checkpoint()
            print(f"{number}/{len(todo)}", flush=True)
    checkpoint()
    report(pd.read_csv(args.output, dtype={c: str for c in FETCH_COLUMNS + HEADER_COLUMNS}, keep_default_na=False, na_values=[""]).assign(
        fetch_status=lambda d: d["fetch_status"].fillna("pending"),
        PatientAge=lambda d: d["PatientAge"].fillna(""),
        Units=lambda d: d["Units"].fillna(""),
        DecayCorrection=lambda d: d["DecayCorrection"].fillna(""),
        plausibility_flags=lambda d: d["plausibility_flags"].fillna(""),
    ), inventory)


if __name__ == "__main__":
    main()
