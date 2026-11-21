"""
Single-SPECT update, method M5 (local, CPU torch, needs src/): the differentiable PBPK of each validation virtual
patient, started at its prior (receptor densities, nuisance and lesion parameters from the other draws of the same
phantom, i.e. the PET-fit posterior; agent factors and release rates from the cohort population), fitted to the VOI
activities of the simulated SPECT, with the relative model discrepancy of tto_choice_<scenario>.json in the VOI
likelihood. Usage: [SCENARIO=<scenario>] python spect_update_pbpk.py [tuning|reporting|test] [draws] [times] [steps] [mismatch|shape|exact] [max phantoms] [mismatch SD] [k/n]
(k/n: only the phantoms k, k + n, ... of the list, for parallel runs; the output name ends in .k)
The scenario comes from the environment variable SCENARIO (default A). In a B scenario the prior of M5 is unchanged:
release rates from the cohort population, independent of the PET (the PBPK does not know the PET-washout link).
Condition 'mismatch' (main): the truth has per-organ variability that the PBPK cannot represent (update.mismatch);
'shape': the same plus a late washout rate that changes in time; 'exact': the truth is the PBPK itself (inverse crime,
for comparison only). A run that was interrupted resumes from its .partial.csv.
`test` (frozen test run, scenario A): every test phantom, the cohort test tables (bundles_test.pkl,
virtual_test_main.pkl), params_test and spect_activity_test; the frozen files must match test_freeze_A.json; refuses
to run if its final output exists (a .partial.csv of an interrupted test run is resumed).
"""
import json
import os
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
from lib import pbpk_torch as T  # noqa: E402
from lib import frozen, spect, update  # noqa: E402
from lib.calibration import build_test_cohort  # noqa: E402
from lib.data import GpuCohort, import_net_data  # noqa: E402
from lib.gating import subject_halves  # noqa: E402
from lib.structures import phantom_structures  # noqa: E402
from lib.utils import provenance  # noqa: E402

SCENARIO = os.environ.get("SCENARIO", "A")
NETDATA = frozen.netdata_dir()
COHORT = frozen.cohort_dir()
PBPK = ROOT / "data" / "processed" / "psma" / "pbpk"
EVAL = frozen.eval_dir(SCENARIO)
OUT = EVAL / "spect"


def prior_patient(target, others, geometry, population):
    """
    The patient of M5 before the SPECT and the SD of its log multipliers. Densities, f_SAL, GFR, hematocrit and the
    lesion densities and flows: log-mean over the other draws of the phantom; the SD of the multipliers of the
    densities, the GFR and the lesion densities and flows is the SD of those logs, widened by sqrt(1 + 1/n). Agent factors and release rates: the cohort population. Amount: as injected.
    """
    theta = population["main"]["theta"]
    log_phi = dict(zip(C.AGENT_ORGANS, theta[:4]))
    logs = {k: np.log([x["values"]["r_den"][k] for x in others]) for k in T.DENSITY_ORGANS}
    widen = np.sqrt(1 + 1 / len(others))
    r_den = {k: float(np.exp(logs[k].mean() + log_phi[k])) for k in T.DENSITY_ORGANS}
    nuisance = {k: float(np.exp(np.mean(np.log([x["values"][k] for x in others])))) for k in ("f_sal", "gfr", "hematocrit")}
    n = len(target["r_lesions"])
    log_r = np.log(np.array([x["r_lesions"] for x in others])) if n else np.zeros((len(others), 0))
    r_lesions = np.exp(log_r.mean(axis=0))
    f_lesions = np.exp(np.log(np.array([x["f_lesions"] for x in others])).mean(axis=0)) if n else np.zeros(0)
    rel = dict(kidneys=float(np.exp(theta[4])), liver=float(np.exp(theta[5])), salivary=float(np.exp(theta[6])))
    patient = F.make_patient(geometry, r_den, nuisance["f_sal"], nuisance["gfr"], nuisance["hematocrit"], r_lesions,
                             f_lesions, np.full(n, np.exp(theta[7])), rel["kidneys"], rel["salivary"])
    patient.lambda_rel.update(liver=rel["liver"], spleen=rel["liver"])
    phi_sd, rel_sd = population["main"]["phi_sd"], population["rel_sd"]
    sd = [np.hypot(logs[k].std(ddof=1) * widen, phi_sd[k]) for k in T.DENSITY_ORGANS]
    sd += [rel_sd[k] for k in T.RELEASE_GROUPS] + [np.std(np.log([x["values"]["gfr"] for x in others]), ddof=1) * widen]
    log_f = np.log(np.array([x["f_lesions"] for x in others])) if n else np.zeros((len(others), 0))
    sd += list(log_r.std(axis=0, ddof=1) * widen) + [np.hypot(population["lesion_epistemic_sd"], rel_sd["lesions"])] * n
    sd += list(log_f.std(axis=0, ddof=1) * widen)
    return patient, torch.tensor(np.maximum(sd, 1e-3), dtype=torch.float64)


def late_kinetics(time_h, curve):
    """A0 and lambda_w of a source curve as the cohort defines them (pbpk_cohort._kinetic): log-linear fit over 24-168 h."""
    window = (time_h >= C.TEFF_WINDOW_H[0]) & (time_h <= C.TEFF_WINDOW_H[1]) & (curve > 0)
    if window.sum() < 2:
        return np.nan, np.nan
    slope, intercept = np.polyfit(time_h[window], np.log(curve[window]), 1)
    return float(np.exp(intercept)), float(-slope)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "tuning"
    n_draws = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    times = [float(t) for t in sys.argv[3].split(",")] if len(sys.argv) > 3 else [48.0, 96.0]
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    condition = sys.argv[5] if len(sys.argv) > 5 else "mismatch"
    limit = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    sd_mismatch = float(sys.argv[7]) if len(sys.argv) > 7 else update.MISMATCH_SD
    part, parts = (int(v) for v in sys.argv[8].split("/")) if len(sys.argv) > 8 else (0, 1)
    stored = json.loads((EVAL / f"tto_choice_{SCENARIO}.json").read_text())
    test = which == "test"
    tag = f"{which}_{condition}{'' if sd_mismatch == update.MISMATCH_SD else f'_sd{sd_mismatch:g}'}_{SCENARIO}{'' if parts == 1 else f'.{part}'}"
    freeze_stamp = {}
    if test:
        if SCENARIO != "A":
            raise ValueError("The test run is scenario A")
        frozen.refuse_if_done([OUT / f"pbpk_rows_{tag}.csv"])
        freeze_stamp = {**frozen.check_manifest(EVAL, NETDATA), **frozen.check_inputs(EVAL, NETDATA)}
    nd = import_net_data(str(ROOT / "src"))
    cfg = spect.SpectConfig()
    if test:
        cohort = build_test_cohort(GpuCohort, nd, NETDATA, NETDATA / "shards", torch.device("cpu"), scenario=SCENARIO)
        studies = list(cohort.phantoms.study_id)
    else:
        cohort = GpuCohort(nd, NETDATA, NETDATA / "shards", torch.device("cpu"), scenario=SCENARIO, splits=("val",))
        if stored["subject_half"] != subject_halves(cohort.phantoms, cohort.phantoms.study_id):
            raise RuntimeError(f"tto_choice_{SCENARIO}.json was chosen on another subject split")
        half_of_study = cohort.phantoms.set_index("study_id").subject.astype(str).map(stored["subject_half"])
        studies = [s for s in cohort.phantoms.study_id if half_of_study[s] == which]
    studies = sorted(studies, key=lambda s: cohort.n_lesions[cohort.index[s]])[:limit or None][part::parts]
    split = "test" if test else "train"
    population = pickle.loads((PBPK / "nb19_cohort_population_v2.pkl").read_bytes())
    geometry = {b["study_id"]: b["geometry"] for b in pickle.loads((COHORT / f"bundles_{split}.pkl").read_bytes())
                if "error" not in b and b["study_id"] in studies}
    virtual = {}
    for x in pickle.loads((COHORT / f"virtual_{split}_main.pkl").read_bytes()):
        if "error" not in x and x["study_id"] in studies:
            virtual.setdefault(x["study_id"], {})[x["draw"]] = x
    truth, activity = update.case_data(NETDATA, studies, SCENARIO, condition, sd=sd_mismatch, test=test)
    crosstalk = OUT / ("crosstalk_test" if test else "crosstalk")
    crosstalk.mkdir(parents=True, exist_ok=True)
    rows, histories = [], []
    partial = OUT / f"pbpk_rows_{tag}.partial.csv"
    if partial.exists():
        rows = pd.read_csv(partial).to_dict("records")
    done = {(r["virtual_id"], r["t_h"]) for r in rows}
    for study in studies:
        i = cohort.index[study]
        st = phantom_structures(cohort, i, 1)
        R, gain = update.structure_cache(cohort, i, st, cfg, crosstalk)
        n_voxels = torch.tensor(st.n_voxels, dtype=torch.float64)
        discrepancy = torch.tensor([stored["discrepancy"][s] for s in st.structures], dtype=torch.float64)
        spine_ml, blood_ml = Co._spine_ml(study)
        for draw in range(n_draws):
            target = virtual[study][draw]
            vid = f"{study}_d{draw:02d}"
            patient, prior_sd = prior_patient(target, [x for d, x in virtual[study].items() if d != draw], geometry[study], population)
            model = T.TorchPBPK(P, patient, P.Injection("Lu177", C.THERAPY_MBQ, target["therapy_nmol"]), spine_ml, blood_ml, substeps=1)
            for t_h in times:
                if (vid, t_h) in done:
                    continue
                start = time.perf_counter()
                measured, _, _ = update.measure(cohort, i, st, activity[(vid, t_h)], vid, t_h, cfg)
                voi_sd = lambda expected: update.voi_sd(expected, n_voxels, gain, cfg, discrepancy)
                vector, history = T.fit(model, measured, R, st.sources, t_h, prior_sd, voi_sd, steps)
                log_tiac, sd = T.laplace(model, vector, measured, R, st.sources, t_h, prior_sd, voi_sd)
                with torch.no_grad():
                    curves, _ = model.solve(**T.Parameters(model.n_lesions).unpack(vector))
                seconds = time.perf_counter() - start
                time_h = model.time_h.numpy()
                for k, source in enumerate(st.sources):
                    if (vid, source) not in truth.index:
                        continue
                    key = dict(virtual_id=vid, study_id=study, half=which, source=source, structure=st.structures[k], t_h=t_h, n=np.nan)
                    a0, lw = late_kinetics(time_h, curves[source].numpy())
                    values = {"log_tiac": (log_tiac[source], sd[source]), "log_a0": (np.log(a0), np.nan),
                              "log_lambda_bio": (np.log(max(lw - nd.LAMBDA_PHYS_H, nd.LAMBDA_BIO_FLOOR)), np.nan)}
                    for q, (mean, spread) in values.items():
                        rows.append({**key, "method": "M5 PBPK", "quantity": q, "mean": mean, "sd": spread,
                                     "truth": float(truth.loc[(vid, source), q]), "seconds": seconds})
                    rows.append({**key, "method": "M5 PBPK at the prior", "quantity": "log_tiac", "mean": history[0][source],
                                 "sd": np.nan, "truth": float(truth.loc[(vid, source), "log_tiac"]), "seconds": seconds})
                    for h in history + [dict(history[-1], step=m) for m in range(len(history), steps)]:
                        histories.append(dict(virtual_id=vid, source=source, structure=st.structures[k], t_h=t_h, step=h["step"],
                                              loss=h["loss"], error=h[source] - float(truth.loc[(vid, source), "log_tiac"])))
                print(f"{vid} @{t_h:g} h: {model.n} states, {model.n_lesions} lesions, {seconds:.0f} s, loss {history[0]['loss']:.1f} -> {history[-1]['loss']:.1f}", flush=True)
            pd.DataFrame(rows).to_csv(partial, index=False)
    stamp = {**provenance(str(HERE), str(ROOT / "src")), "steps": steps, "condition": condition, "mismatch_sd": 0.0 if condition == "exact" else sd_mismatch,
             **freeze_stamp}
    table = pd.DataFrame(rows)
    table.assign(**stamp).to_csv(OUT / f"pbpk_rows_{tag}.csv", index=False)
    partial.unlink(missing_ok=True)
    if done:
        return
    history = pd.DataFrame(histories)
    by_step = history.assign(abs_error=history.error.abs()).groupby(["structure", "t_h", "step"]).abs_error.median().reset_index()
    by_step.assign(**stamp).to_csv(OUT / f"pbpk_error_by_step_{tag}.csv", index=False)
    pd.set_option("display.width", 250)
    table["error"] = (table["mean"] - table.truth).abs()
    table["case"] = table.method + " @" + table.t_h.astype(int).astype(str) + "h"
    for q in ("log_tiac", "log_a0", "log_lambda_bio"):
        print(f"\n{q}, median |error|\n" + table[table.quantity == q].pivot_table(index="structure", columns="case", values="error", aggfunc="median").round(2).to_string())
    final = by_step[by_step.step.isin([0, 5, 10, 20, 30, steps - 1])]
    print("\nlog TIAC error against the step\n" + final.pivot_table(index="structure", columns=["t_h", "step"], values="abs_error").round(2).to_string())


if __name__ == "__main__":
    main()
