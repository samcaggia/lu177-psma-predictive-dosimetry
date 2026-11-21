"""Activity of every Monte Carlo source and lesion of the frozen virtual cohort at the SPECT times, by re-running the therapy simulation of each virtual patient from its stored draw (the cohort stores the TIAC and the kinetic parameters, not the curves)."""
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))
import pbpk_cohort as Co  # noqa: E402

COHORT = Path(os.environ.get("LU177_COHORT") or WORKSPACE / "data" / "processed" / "psma" / "cohort")
NETDATA = Path(os.environ.get("LU177_NETDATA") or WORKSPACE / "data" / "processed" / "psma" / "netdata")
SPECT_TIMES_H = (24.0, 48.0, 96.0, 168.0)
_STATE = {}


def therapy_curves(geometry, x, rel, l_lesions, spine_ml, blood_ml):
    """pbpk_cohort.simulate_therapy of one virtual patient with its curves captured: returns (time_h, {source: fraction of the injected activity over time, physical decay included}, the simulate_therapy output). The curves are the ones the cohort fitted its kinetic parameters on, taken from the same call."""
    captured = []
    original = Co._kinetic

    def record(t_h, activity, tia):
        captured.append((t_h, np.asarray(activity, float)))
        return original(t_h, activity, tia)

    Co._kinetic = record
    try:
        out = Co.simulate_therapy(geometry, x["values"], x["r_lesions"], x["f_lesions"], l_lesions, rel, x["factors"], spine_ml, blood_ml, nmol=x["therapy_nmol"])
    finally:
        Co._kinetic = original
    names = list(out["tia"]) + [f"lesion_{j}" for j in range(len(out["lesion_tia"]))]
    assert len(names) == len(captured)
    return captured[0][0], {n: c[1] for n, c in zip(names, captured)}, out


def _job(job):
    x, rel, l_lesions, times = job
    geometry = _STATE["geometry"][x["study_id"]]
    spine_ml, blood_ml = Co._spine_ml(x["study_id"])
    try:
        t_h, curves, out = therapy_curves(geometry, x, rel, l_lesions, spine_ml, blood_ml)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        return dict(error=repr(error)[:200], study_id=x["study_id"], draw=x["draw"])
    vid = f"{x['study_id']}_d{x['draw']:02d}"
    tia = {**out["tia"], **{f"lesion_{j}": v for j, v in enumerate(out["lesion_tia"])}}
    rows = []
    for source, curve in curves.items():
        values = np.exp(np.interp(times, t_h, np.log(curve))) if (curve[t_h >= 1.0] > 0).all() and (curve[t_h < 1.0] > 0).all() else np.interp(times, t_h, curve)
        rows.extend(dict(virtual_id=vid, source=source, t_h=t, activity=float(v), tiac_h_rerun=float(tia[source])) for t, v in zip(times, values))
    return rows


def _build(split, scenario, arm, times, processes, studies=None):
    import multiprocessing as mp
    bundles = [b for b in pickle.loads((COHORT / f"bundles_{split}.pkl").read_bytes()) if "error" not in b]
    _STATE["geometry"] = {b["study_id"]: b["geometry"] for b in bundles}
    virtual = [x for x in pickle.loads((COHORT / f"virtual_{split}_{arm}.pkl").read_bytes()) if "error" not in x]
    if studies is not None:
        virtual = [x for x in virtual if x["study_id"] in studies]
    organs = pd.read_parquet(COHORT / f"lambda_rel_{split}.parquet")
    organs = organs[(organs.arm == arm) & (organs.scenario == scenario)].set_index("virtual_id")
    lesions = pd.read_parquet(COHORT / f"lesion_lambda_rel_{split}.parquet")
    lesions = lesions[(lesions.arm == arm) & (lesions.scenario == scenario)].sort_values("lesion")
    by_patient = {vid: g.rel.to_numpy() for vid, g in lesions.groupby("virtual_id")}
    jobs = []
    for x in virtual:
        vid = f"{x['study_id']}_d{x['draw']:02d}"
        row = organs.loc[vid]
        rel = dict(kidneys=float(row.rel_kidneys), liver=float(row.rel_liver), salivary=float(row.rel_salivary))
        jobs.append((x, rel, by_patient.get(vid, np.zeros(0)), np.asarray(times, float)))
    with mp.get_context("fork").Pool(processes) as workers:
        results = workers.map(_job, jobs, chunksize=6)
    failed = [r for r in results if isinstance(r, dict)]
    table = pd.DataFrame([row for r in results if not isinstance(r, dict) for row in r])
    table.insert(0, "arm", arm)
    table["scenario"] = scenario
    return table, failed


def add_checks(table, split):
    """Adds the cohort TIAC (the re-run must reproduce it) and the activity of the cohort's uptake-washout curve at each time."""
    tia = pd.read_parquet(COHORT / f"tia_{split}.parquet").rename(columns={"compartment": "source"})
    out = table.merge(tia[["arm", "virtual_id", "scenario", "source", "tiac_h", "a0", "lw", "lu"]], on=["arm", "virtual_id", "scenario", "source"], how="left")
    uptake = np.where(np.isfinite(out.lu), np.exp(-out.lu.where(np.isfinite(out.lu), 0.0) * out.t_h), 0.0)
    out["curve_activity"] = out.a0 * (np.exp(-out.lw * out.t_h) - uptake)
    return out.drop(columns=["a0", "lw", "lu"])


def build(scenario="A", arm="main", times=SPECT_TIMES_H, processes=3, studies=None):
    """Training and validation virtual patients (cohort split 'train'). Returns (table, failures)."""
    table, failed = _build("train", scenario, arm, times, processes, studies)
    return add_checks(table, "train"), failed


def build_test(scenario="A", arm="main", times=SPECT_TIMES_H, processes=3):
    """The test virtual patients, for the frozen test evaluation only: writes netdata/spect_activity[_<scenario>]_test.parquet. Called only by deep_learning/test_run.py (step spect-data); refuses to overwrite."""
    if activity_path(scenario, test=True).exists():
        raise FileExistsError(f"{activity_path(scenario, test=True)} exists: built once")
    table, failed = _build("test", scenario, arm, times, processes)
    table = add_checks(table, "test")
    table.to_parquet(activity_path(scenario, test=True), index=False)
    return len(table), failed


def activity_path(scenario, test=False):
    """netdata/spect_activity[_<scenario>][_test].parquet: scenario A keeps the name without a scenario."""
    return NETDATA / f"spect_activity{'' if scenario == 'A' else '_' + scenario}{'_test' if test else ''}.parquet"


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "A"
    table, failed = build(scenario, processes=int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    worst = float((table.tiac_h_rerun / table.tiac_h - 1).abs().max())
    table.to_parquet(activity_path(scenario), index=False)
    print(f"Scenario {scenario}: {table.virtual_id.nunique()} virtual patients, {len(table)} rows, {len(failed)} failures; largest |TIAC re-run / cohort - 1| {worst:.2e}")
