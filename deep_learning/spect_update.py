"""
Single-SPECT update, local part (CPU torch, no network): M0 (gated prior), M1 (classical single-time-point) and M2
(Bayesian update per structure) on the validation virtual patients, from the simulated SPECT VOI activities.
Usage: [SCENARIO=<scenario>] python spect_update.py [tuning|reporting|test] [draws] [times, comma separated]
       [mismatch|shape|exact] [mismatch SD] [gated|population|pbpk_fit|ceiling] [scenario of the choices]
On the tuning half, main condition (mismatch, SD 0.15), three things are chosen and stored in tto_choice_<scenario>.json:
the relative model discrepancy of the VOI likelihood per structure group (coverage matching), the per-structure gates
of M1 and M2 (update, or keep the prior), and the post-hoc SD factor of M2 without discrepancy (comparison row only).
Every other run reads them from the file; at a SPECT time that was not tuned, the gate of the nearest tuned time is used.
With `population` the prior of every structure and quantity is the population baseline (no network anywhere): the
ablation of the pre-therapy network, same gates, discrepancy and seeds; only its M0 and M2 rows are written.
`pbpk_fit` (the scenario A draws of the phantom: the PET-fit posterior without the PET-washout link) and `ceiling` (the
other draws of the phantom in the scenario: everything the phantom can tell, link included) are run the same way.
The scenario comes from the environment variable SCENARIO (default A); its tables are netdata/<stem>_<scenario>.parquet
and its outputs runs/eval_<scenario>/. The halves are those of the subject split of scenario A (same seed). The last
argument reads the discrepancy and the gates from the tto_choice file of another scenario (a pre-check before the
network of the scenario exists; the output name then ends in _choice<scenario>); nothing is chosen in that case.
`test` (frozen test run, scenario A, gated or population prior only): every test phantom, the test structure
predictions of the notebook (section 10), params_test and spect_activity_test; discrepancy and gates read from
tto_choice_A.json, whose md5 (and that of every other frozen file) must match test_freeze_A.json; refuses to run if
its output exists.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from lib import frozen, spect, update  # noqa: E402
from lib.calibration import build_test_cohort  # noqa: E402
from lib.data import GpuCohort, import_net_data  # noqa: E402
from lib.gating import subject_halves  # noqa: E402
from lib.scenario import CEILING, scenario_file  # noqa: E402
from lib.scoring import with_sd  # noqa: E402
from lib.structures import phantom_structures  # noqa: E402
from lib.utils import provenance  # noqa: E402

SCENARIO = os.environ.get("SCENARIO", "A")
NETDATA = frozen.netdata_dir()
EVAL = frozen.eval_dir(SCENARIO)
OUT = EVAL / "spect"
SCORED = ("log_tiac", "log_a0", "log_lambda_bio")
PRIORS = {"population": ("baseline_population", "population prior", "_popprior"),
          "pbpk_fit": ("baseline_pbpk_fit", "PBPK-fit prior", "_pbpkprior"),
          "ceiling": (f"baseline_{CEILING}", "same-phantom-draws prior", "_ceilingprior")}


def score(nd, table, by=("half", "t_h", "method", "structure", "quantity")):
    """n, bias, median |error|, coverage of the 68% and 95% intervals (where an SD is given) per group."""
    rows = []
    for key, g in table.groupby(list(by), sort=False, dropna=False):
        error = g["mean"] - g.truth
        row = dict(zip(by, key), n=len(g), bias=float(error.mean()), median_abs_error=float(error.abs().median()))
        if g.sd.notna().any():
            for level in (0.68, 0.95):
                half = nd.predictive_halfwidth(g.sd.to_numpy(float), g.n.to_numpy(float), level)
                row[f"coverage_{int(level * 100)}"] = float((error.abs().to_numpy() < half).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "tuning"
    n_draws = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    times = [float(t) for t in sys.argv[3].split(",")] if len(sys.argv) > 3 else [48.0, 96.0]
    condition = sys.argv[4] if len(sys.argv) > 4 else "mismatch"
    sd = float(sys.argv[5]) if len(sys.argv) > 5 else update.MISMATCH_SD
    prior_kind = sys.argv[6] if len(sys.argv) > 6 else "gated"
    choice_from = sys.argv[7] if len(sys.argv) > 7 else SCENARIO
    choosing = (which == "tuning" and condition == "mismatch" and sd == update.MISMATCH_SD and prior_kind == "gated"
                and choice_from == SCENARIO)
    test = which == "test"
    prior_tag = "" if prior_kind == "gated" else PRIORS[prior_kind][2]
    tag = (f"{which}_{condition}{'' if sd == update.MISMATCH_SD else f'_sd{sd:g}'}{prior_tag}"
           f"{'' if choice_from == SCENARIO else f'_choice{choice_from}'}_{SCENARIO}")
    freeze_stamp = {}
    if test:
        if SCENARIO != "A" or choice_from != SCENARIO or prior_kind not in ("gated", "population"):
            raise ValueError("The test run is scenario A with its own choices, gated or population prior")
        frozen.refuse_if_done([OUT / f"update_rows_{tag}.csv"])
        freeze_stamp = {**frozen.check_manifest(EVAL, NETDATA), **frozen.check_inputs(EVAL, NETDATA)}
    nd = import_net_data(str(ROOT / "src"))
    cfg = spect.SpectConfig()
    OUT.mkdir(parents=True, exist_ok=True)
    choice_path = EVAL.parent / f"eval_{choice_from}" / f"tto_choice_{choice_from}.json"
    stored = json.loads(choice_path.read_text()) if choice_path.exists() else {}
    grid = update.DISCREPANCY_GRID if choosing else sorted({0.0, *stored["discrepancy"].values()})
    if test:
        cohort = build_test_cohort(GpuCohort, nd, NETDATA, NETDATA / "shards", torch.device("cpu"), scenario=SCENARIO)
        studies = list(cohort.phantoms.study_id)
    else:
        cohort = GpuCohort(nd, NETDATA, NETDATA / "shards", torch.device("cpu"), scenario=SCENARIO, splits=("val",))
        subject_half = subject_halves(cohort.phantoms, cohort.phantoms.study_id)
        reference = EVAL.parent / "eval_A" / "gated_prior_A.json"
        if reference.exists() and json.loads(reference.read_text())["subject_half"] != subject_half:
            raise RuntimeError("The subject split differs from that of scenario A")
        half_of_study = cohort.phantoms.set_index("study_id").subject.astype(str).map(subject_half)
        studies = [s for s in cohort.phantoms.study_id if half_of_study[s] == which]
    test_suffix = "_test" if test else ""
    if prior_kind == "gated":
        choice = json.loads((EVAL / f"gated_prior_{SCENARIO}.json").read_text())
        if not test and choice["subject_half"] != subject_half:
            raise RuntimeError(f"gated_prior_{SCENARIO}.json was chosen on another subject split")
        population = pd.read_parquet(NETDATA / "baseline_population_test.parquet" if test
                                     else scenario_file(NETDATA, "baseline_population", SCENARIO))
        network = EVAL / f"{'test_' if test else ''}structure_prediction_{SCENARIO}.csv"
        m0 = update.gated_table(choice, with_sd(pd.read_csv(network)), population[population.arm == "main"])
    else:
        stem, prior_label, prior_tag = PRIORS[prior_kind]
        m0 = pd.read_parquet(NETDATA / f"{stem}{test_suffix}.parquet" if prior_kind == "pbpk_fit" or test
                             else scenario_file(NETDATA, stem, SCENARIO))
        m0 = m0[(m0.arm == "main") & m0.study_id.isin(set(studies))]
    truth, activity = update.case_data(NETDATA, studies, SCENARIO, condition, sd=sd, test=test)
    rates = update.population_rates(pd.read_parquet(scenario_file(NETDATA, "population_prior", SCENARIO)))
    rows = []
    for study in studies:
        i = cohort.index[study]
        st = phantom_structures(cohort, i, 1)
        R, gain = update.structure_cache(cohort, i, st, cfg, OUT / f"crosstalk{test_suffix}")
        n_voxels = torch.tensor(st.n_voxels, dtype=torch.float64)
        lam_pop = torch.tensor([rates[s] for s in st.structures], dtype=torch.float64)
        for draw in range(n_draws):
            vid = f"{study}_d{draw:02d}"
            prior_rows = m0[m0.virtual_id == vid]
            prior = update.prior_tensors(prior_rows, st.sources)
            n_prior = prior_rows.set_index("source").reindex(st.sources)
            key = dict(virtual_id=vid, study_id=study, half=which)

            def add(method, t_h, quantity, mean, sd_=None, n=None, **extra):
                for k, source in enumerate(st.sources):
                    if (vid, source) not in truth.index:
                        continue
                    rows.append({**key, "source": source, "structure": st.structures[k], "t_h": t_h, "method": method,
                                 "quantity": quantity, "mean": float(mean[k]), "sd": float(sd_[k]) if sd_ is not None else np.nan,
                                 "n": float(n[k]) if n is not None else np.nan, "truth": float(truth.loc[(vid, source), quantity]),
                                 **{name: (float(v[k]) if hasattr(v, "__len__") else v) for name, v in extra.items()}})

            for q in SCORED:
                add("M0 gated prior", np.nan, q, n_prior[f"{q}_mean"].to_numpy(), n_prior[f"{q}_sd"].to_numpy(),
                    n_prior[f"{q}_n"].to_numpy())
            for t_h in times:
                measured, _, _ = update.measure(cohort, i, st, activity[(vid, t_h)], vid, t_h, cfg)
                positive = measured.clamp_min(1e-30)
                add("M1 Hanscheid", t_h, "log_tiac", torch.log(update.hanscheid(positive, t_h)))
                add("M1 population rate", t_h, "log_tiac", torch.log(update.madsen(positive / torch.diagonal(R), t_h, lam_pop)))
                for value in grid:
                    posterior, ess, used = update.bayes_update(prior, measured, R, n_voxels, gain, t_h, cfg, value,
                                                               seed=i * 1000 + draw)
                    for q in SCORED:
                        add("M2 grid", t_h, q, posterior[q][0], posterior[q][1], discrepancy=value, ess=ess, samples=used)
        print(f"{study} ({which}): {len(st)} structures", flush=True)
    table = pd.DataFrame(rows)
    stamp = {**provenance(str(HERE), str(ROOT / "src")), "condition": condition, "mismatch_sd": 0.0 if condition == "exact" else sd}
    stamp["prior"] = prior_kind
    stamp.update(freeze_stamp)
    grid_rows = table[table.method == "M2 grid"]
    table = table[table.method != "M2 grid"]
    if choosing:
        stored["discrepancy"], coverage = update.choose_discrepancy(grid_rows)
        stored["discrepancy_rule"] = ("relative SD added to the VOI likelihood, per structure group; the grid value whose M2 68% and "
                                      "95% intervals of log TIAC cover closest to nominal on the tuning half")
        coverage.assign(**stamp).to_csv(OUT / f"discrepancy_coverage_{tag}.csv", index=False)
    chosen = grid_rows[np.isclose(grid_rows.discrepancy, grid_rows.structure.map(stored["discrepancy"]))].assign(method="M2 Bayes")
    plain = grid_rows[grid_rows.discrepancy == 0.0].assign(method="M2 Bayes, no discrepancy")
    if choosing:
        z2 = ((plain["mean"] - plain.truth) / plain.sd) ** 2
        fitted = z2.groupby([plain.structure, plain.t_h, plain.quantity]).mean() ** 0.5
        stored["m2_sd_scale"] = {" | ".join(map(str, k)): float(v) for k, v in fitted.items()}
        stored.pop("m2_sd_scale_rule", None)
    keys = plain.structure + " | " + plain.t_h.astype(str) + " | " + plain.quantity
    scaled = plain.assign(sd=plain.sd * keys.map(stored["m2_sd_scale"]).fillna(1.0), method="M2 Bayes, no discrepancy, SD scaled")
    table = pd.concat([table, chosen, plain, scaled], ignore_index=True)
    gated_methods = ["M1 population rate", "M2 Bayes"]
    if choosing:
        stored["gate"] = update.gate_choice(score(nd, table), gated_methods)
        stored["gate_rule"] = "update where the median |log TIAC error| on the tuning half is below that of M0, per structure and SPECT time"
        stored["subject_half"] = subject_half
        stored["mismatch_sd"] = sd
        choice_path.write_text(json.dumps(stored, indent=1))
    prior_rows = table[table.method == "M0 gated prior"]
    for method in gated_methods:
        gate = stored["gate"][method]
        rows_m = table[table.method == method]
        keep = np.array([update.gate_at(gate.get(s, {}), t) for s, t in zip(rows_m.structure, rows_m.t_h)])
        fallback = rows_m[~keep][["virtual_id", "source", "t_h", "quantity"]].merge(
            prior_rows.drop(columns=["t_h"]), on=["virtual_id", "source", "quantity"])
        table = pd.concat([table, pd.concat([rows_m[keep], fallback]).assign(method=method + ", gated")], ignore_index=True)
    if prior_kind != "gated":
        names = {"M0 gated prior": f"M0 {prior_label}", "M2 Bayes": f"M2 Bayes, {prior_label}",
                 "M2 Bayes, gated": f"M2 Bayes, {prior_label}, gated"}
        table = table[table.method.isin(names)].assign(method=lambda d: d.method.map(names))
    table.assign(**stamp).to_csv(OUT / f"update_rows_{tag}.csv", index=False)
    summary = score(nd, table)
    summary.assign(**stamp).to_csv(OUT / f"update_summary_{tag}.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 400)
    if prior_kind != "gated":
        d = summary[summary.quantity == "log_tiac"]
        print(d.pivot_table(index="structure", columns=["method", "t_h"], values="median_abs_error", dropna=False).round(2).to_string())
        return
    summary["case"] = summary.method + np.where(summary.t_h.notna(), " @" + summary.t_h.fillna(0).astype(int).astype(str) + "h", "")
    shown = ["M0 gated prior", "M1 Hanscheid", "M1 population rate", "M1 population rate, gated", "M2 Bayes", "M2 Bayes, gated"]
    for q in SCORED:
        d = summary[(summary.quantity == q) & summary.method.isin(shown)]
        print(f"\n== {which}: {q}, median |error| ==\n" + d.pivot(index="structure", columns="case", values="median_abs_error").round(2).to_string())
    d = summary[(summary.quantity == "log_tiac") & summary.method.isin(["M0 gated prior", "M2 Bayes", "M2 Bayes, no discrepancy", "M2 Bayes, no discrepancy, SD scaled"])]
    for column in ("coverage_68", "coverage_95"):
        print(f"\n== {which}: log_tiac, {column} ==\n" + d.pivot(index="structure", columns="case", values=column).round(2).to_string())
    print("\nDiscrepancy:", stored["discrepancy"])
    print("Gate (update at each time):\n" + pd.DataFrame({m: {s: " / ".join("yes" if v else "no" for v in g.values()) for s, g in gate.items()}
                                                           for m, gate in stored["gate"].items()}).to_string())
    m2 = table[table.method == "M2 Bayes"]
    print("\nM2 effective sample size: min", round(m2.ess.min()), "median", round(m2.ess.median()), "| share below 200:", round(float((m2.ess < 200).mean()), 4))


if __name__ == "__main__":
    main()
