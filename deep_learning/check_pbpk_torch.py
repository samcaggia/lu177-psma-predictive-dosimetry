"""
Check of the torch PBPK (lib/pbpk_torch.py) against the frozen cohort, noise-free, with the true parameters of each
virtual patient: source activities at the SPECT times (netdata/spect_activity.parquet) and TIACs. Local, needs torch
and src/. Usage: python check_pbpk_torch.py [phantoms] [substeps]
"""
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import pbpk as P  # noqa: E402
import pbpk_calibration as C  # noqa: E402
import pbpk_cohort as Co  # noqa: E402
import pbpk_fit as F  # noqa: E402
from lib.pbpk_torch import TorchPBPK  # noqa: E402

NETDATA = ROOT / "data" / "processed" / "psma" / "netdata"
COHORT = ROOT / "data" / "processed" / "psma" / "cohort"
OUT = HERE / "runs" / "eval_A" / "spect"


def true_model(x, geometry, organs, lesions, substeps):
    """The torch PBPK of a virtual patient at its true parameters, built as pbpk_cohort.simulate_therapy builds it."""
    vid = f"{x['study_id']}_d{x['draw']:02d}"
    rel = organs.loc[vid]
    r_den = {k: v * x["factors"].get(k, 1.0) for k, v in x["values"]["r_den"].items()}
    patient = F.make_patient(geometry, r_den, x["values"]["f_sal"], x["values"]["gfr"], x["values"]["hematocrit"],
                             x["r_lesions"], x["f_lesions"], lesions.get(vid, np.zeros(0)), float(rel.rel_kidneys),
                             float(rel.rel_salivary))
    patient.lambda_rel.update(liver=float(rel.rel_liver), spleen=float(rel.rel_liver))
    spine_ml, blood_ml = Co._spine_ml(x["study_id"])
    return TorchPBPK(P, patient, P.Injection("Lu177", C.THERAPY_MBQ, x["therapy_nmol"]), spine_ml, blood_ml, substeps), vid


def main():
    n_phantoms = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    substeps = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    phantoms = pd.read_parquet(NETDATA / "phantoms.parquet")
    val = phantoms[phantoms.net_split == "val"].sort_values("tlv_ml")
    studies = set(val.study_id.iloc[np.linspace(0, len(val) - 1, n_phantoms).round().astype(int)])
    geometry = {b["study_id"]: b["geometry"] for b in pickle.loads((COHORT / "bundles_train.pkl").read_bytes())
                if "error" not in b and b["study_id"] in studies}
    virtual = [x for x in pickle.loads((COHORT / "virtual_train_main.pkl").read_bytes())
               if "error" not in x and x["study_id"] in studies and x["draw"] == 0]
    organs = pd.read_parquet(COHORT / "lambda_rel_train.parquet")
    organs = organs[(organs.arm == "main") & (organs.scenario == "A")].set_index("virtual_id")
    lesions = pd.read_parquet(COHORT / "lesion_lambda_rel_train.parquet")
    lesions = lesions[(lesions.arm == "main") & (lesions.scenario == "A")].sort_values("lesion")
    lesions = {vid: g.rel.to_numpy() for vid, g in lesions.groupby("virtual_id")}
    activity = pd.read_parquet(NETDATA / "spect_activity.parquet")
    rows = []
    for x in virtual:
        model, vid = true_model(x, geometry[x["study_id"]], organs, lesions, substeps)
        start = time.perf_counter()
        with torch.no_grad():
            curves, tiac = model.solve()
        seconds = time.perf_counter() - start
        for r in activity[activity.virtual_id == vid].itertuples():
            rows.append(dict(virtual_id=vid, source=r.source, t_h=r.t_h, states=model.n, coupled=model.coupled,
                             seconds=seconds, activity=r.activity, torch_activity=float(model.at(curves[r.source], r.t_h)),
                             tiac_h=r.tiac_h, torch_tiac_h=float(tiac[r.source])))
        print(f"{vid}: {model.n} states, {model.n_lesions} lesions, coupled {model.coupled}, {seconds:.1f} s", flush=True)
    table = pd.DataFrame(rows)
    table["structure"] = np.where(table.source.str.startswith("lesion_"), "lesion", table.source)
    table["tac_error"] = np.log(table.torch_activity / table.activity.where(table.activity > 0))
    table["tiac_error"] = np.log(table.torch_tiac_h / table.tiac_h)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "pbpk_torch_check.csv", index=False)
    pd.set_option("display.width", 200)
    worst = lambda v: v.abs().max()
    print("\nLargest |log(torch / cohort)| of the activity, by time:\n"
          + table.pivot_table(index="structure", columns="t_h", values="tac_error", aggfunc=worst).round(4).to_string())
    print("\nLargest |log(torch / cohort)| of the TIAC:\n" + table.groupby("structure").tiac_error.agg(worst).round(4).to_string())


if __name__ == "__main__":
    main()
