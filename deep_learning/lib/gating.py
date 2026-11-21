"""
Gated prior (no torch): per structure group and quantity, the calibrated network or the population prior, chosen on
the tuning half of the validation subjects and reported on the other half. The split is stored with the choice, to be
reused by the SPECT update.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from .scoring import KEYS, QUANTITIES, apply_scales, fit_scales

HALVES = ("tuning", "reporting")


def subject_halves(phantoms, studies, seed=20261114):
    """{subject: 'tuning' | 'reporting'}: subjects in sorted order, shuffled with the seed, dealt in turn."""
    subjects = np.array(sorted(phantoms.subject[phantoms.study_id.isin(studies)].unique()))
    np.random.default_rng(seed).shuffle(subjects)
    return {str(s): HALVES[k % 2] for k, s in enumerate(subjects)}


def residuals(nd, prediction, truth, method):
    """
    One row per (virtual patient, source, quantity) with a finite truth and prediction: error = mean - truth, the
    half-widths of the 68% and 95% predictive intervals (net_data.predictive_halfwidth) and the negative log
    predictive density (Gaussian, or Student t with n - 1 degrees of freedom when <q>_n is given).
    """
    truth = truth.assign(log_lambda_u=truth.log_lambda_u.where(truth.lu_target))
    columns = [c for q in QUANTITIES for c in (f"{q}_mean", f"{q}_sd", f"{q}_n") if c in prediction]
    merged = truth[KEYS + ["study_id", "structure", *QUANTITIES]].merge(prediction[KEYS + columns], on=KEYS)
    parts = []
    for q in QUANTITIES:
        if f"{q}_mean" not in merged:
            continue
        ok = (np.isfinite(merged[q]) & np.isfinite(merged[f"{q}_mean"]) & np.isfinite(merged[f"{q}_sd"])).to_numpy()
        d = merged.loc[ok, KEYS + ["study_id", "structure"]].copy()
        error = (merged[f"{q}_mean"] - merged[q]).to_numpy(float)[ok]
        sd = merged[f"{q}_sd"].to_numpy(float)[ok]
        n = merged[f"{q}_n"].to_numpy(float)[ok] if f"{q}_n" in merged else np.full(len(sd), np.nan)
        student = np.isfinite(n) & (n > 1)
        scale = np.where(student, sd * np.sqrt(1 + 1 / np.where(student, n, 1.0)), sd)
        gaussian = 0.5 * np.log(2 * np.pi * scale ** 2) + 0.5 * (error / scale) ** 2
        with np.errstate(invalid="ignore"):
            t = np.log(scale) - stats.t.logpdf(error / scale, np.where(student, n - 1, 1.0))
        d = d.assign(quantity=q, method=method, error=error, sd=sd, n=n, nlpd=np.where(student, t, gaussian),
                     half_68=nd.predictive_halfwidth(sd, n, 0.68), half_95=nd.predictive_halfwidth(sd, n, 0.95))
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def summarise(res, by=("method", "structure", "quantity")):
    """n, bias, median absolute error, mean negative log predictive density and coverage of the two intervals."""
    rows = []
    for key, g in res.groupby(list(by), sort=False):
        rows.append(dict(zip(by, key), n=len(g), bias=float(g.error.mean()),
                         median_abs_error=float(g.error.abs().median()), nlpd=float(g.nlpd.mean()),
                         coverage_68=float((g.error.abs() < g.half_68).mean()),
                         coverage_95=float((g.error.abs() < g.half_95).mean())))
    return pd.DataFrame(rows)


def gated_prior(nd, network, truth_network, population, truth_population, phantoms, seed=20261114, criterion="nlpd"):
    """
    network: structure predictions of the network (raw SD, <q>_mean and <q>_sd), scored against truth_network (lesions
    on the dilated mask); population: the population baseline, scored against truth_population (cohort values).
    The SD factors of the network are fitted on the tuning half only. Per structure group and quantity the method with
    the lower `criterion` on the tuning half is chosen ('nlpd', the mean negative log predictive density, scores mean
    and width together; 'median_abs_error' the mean only). Returns (choice dict for the JSON, table of both halves).
    """
    half_of_subject = subject_halves(phantoms, network.study_id.unique(), seed)
    half_of_study = phantoms.set_index("study_id").subject.astype(str).map(half_of_subject)
    tuning = lambda table: table[table.study_id.map(half_of_study) == "tuning"]
    scales = fit_scales(tuning(network), tuning(truth_network))
    res = pd.concat([residuals(nd, apply_scales(network, scales), truth_network, "network (calibrated)"),
                     residuals(nd, population, truth_population, "population")], ignore_index=True)
    res["half"] = res.study_id.map(half_of_study)
    table = summarise(res, ("half", "method", "structure", "quantity"))
    tuned = table[table.half == "tuning"].pivot(index=["structure", "quantity"], columns="method", values=criterion)
    chosen = (tuned["network (calibrated)"] < tuned["population"]).map({True: "network (calibrated)", False: "population"})
    pick = res.set_index(["structure", "quantity"]).index.map(chosen)
    gated = res[res.method.to_numpy() == np.asarray(pick)].assign(method="gated")
    table = pd.concat([table, summarise(gated, ("half", "method", "structure", "quantity"))], ignore_index=True)
    choice = dict(seed=int(seed), criterion=criterion, subject_half=half_of_subject,
                  choice={s: {q: chosen[(s, q)] for (s2, q) in chosen.index if s2 == s}
                          for s in sorted({s for s, _ in chosen.index})},
                  network_sd_scale={s: {r.quantity: float(r.scale) for r in g.itertuples()}
                                    for s, g in scales.groupby("structure")})
    return choice, table


def apply_choice(nd, choice, network, truth_network, population, truth_population):
    """
    The gated prior of a stored choice applied unchanged (test split): network SD factors and per-group choice from the
    JSON, nothing fitted. Same scoring as gated_prior: calibrated network against truth_network (lesions on the dilated
    mask), population against truth_population. Returns the summary by method, structure and quantity, with the rows
    of the network (calibrated), the population and the gated prior.
    """
    scales = pd.DataFrame([dict(structure=s, quantity=q, scale=v) for s, g in choice["network_sd_scale"].items()
                           for q, v in g.items()])
    res = pd.concat([residuals(nd, apply_scales(network, scales), truth_network, "network (calibrated)"),
                     residuals(nd, population, truth_population, "population")], ignore_index=True)
    pick = np.array([choice["choice"].get(s, {}).get(q, "population") for s, q in zip(res.structure, res.quantity)])
    gated = res[res.method.to_numpy() == pick].assign(method="gated")
    return summarise(pd.concat([res, gated], ignore_index=True))


def save_choice(choice, path, stamp):
    with open(path, "w") as handle:
        json.dump({**choice, "provenance": {k: (v if isinstance(v, str) else float(v)) for k, v in stamp.items()}},
                  handle, indent=1)
