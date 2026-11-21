"""
Single-SPECT update at structure level: the VOI measurement of a simulated SPECT, the cross-talk matrix of the forward
model between sources and VOIs, the gated prior M0 as a table, classical single-time-point dosimetry (M1) and the
Bayesian update by importance sampling (M2).
"""
import math

import numpy as np
import pandas as pd
import torch

from . import spect
from .kinetics import LAMBDA_PHYS_H
from .scoring import KEYS, QUANTITIES
from .structures import EARLY_SHARE_MAX

PRIOR_QUANTITIES = ("log_a0", "log_lambda_bio", "early_share", "log_tiac")
PROPOSAL_WIDTH = 1.5


def gated_table(choice, network, population):
    """
    M0 as a prediction table: per structure group and quantity the calibrated network (SD x the factor stored with the
    choice, no <q>_n) or the population baseline (<q>_n kept), as chosen in gated_prior_<scenario>.json. Adds
    <q>_method. Rows: the (arm, virtual_id, source) of the network table.
    """
    out = network[KEYS + ["study_id", "structure"]].merge(population[KEYS + [c for q in QUANTITIES for c in (
        f"{q}_mean", f"{q}_sd", f"{q}_n")]], on=KEYS, how="left", suffixes=("", "_pop"))
    for q in QUANTITIES:
        pick = out.structure.map(lambda s: choice["choice"].get(s, {}).get(q, "population"))
        scale = out.structure.map(lambda s: choice["network_sd_scale"].get(s, {}).get(q, 1.0)).to_numpy()
        use = (pick == "network (calibrated)").to_numpy() & np.isfinite(network[f"{q}_mean"].to_numpy())
        out[f"{q}_mean"] = np.where(use, network[f"{q}_mean"].to_numpy(), out[f"{q}_mean"].to_numpy())
        out[f"{q}_sd"] = np.where(use, network[f"{q}_sd"].to_numpy() * scale, out[f"{q}_sd"].to_numpy())
        out[f"{q}_n"] = np.where(use, np.nan, out[f"{q}_n"].to_numpy())
        out[f"{q}_method"] = np.where(use, "network", "population")
    return out


@torch.no_grad()
def crosstalk(cohort, i, st, cfg, chunk=6):
    """
    R[v, s]: share of the activity of source s that the forward model (resolution only, calibration 1) puts in the VOI
    of structure v; sources and VOIs in the order of st (a lesion source is its own labels, its VOI the dilated mask).
    The expected VOI measurement is R @ source activities.
    """
    weight = cohort.targets(i, 0)["w"].float()
    lesion, code = cohort.lesion[i].long(), cohort.compartment[i].long()
    grid = float(cohort.phantoms.grid_mm.iloc[0])
    columns = []
    for start in range(0, len(st), chunk):
        maps = []
        for source in st.sources[start:start + chunk]:
            mask = (lesion == int(source.split("_")[1]) + 1) if source.startswith("lesion_") else \
                ((code == cohort.nd.COMPARTMENT_CODES[source]) & (lesion == 0))
            maps.append(torch.where(mask, weight, torch.zeros_like(weight)))
        columns.append(spect.voi_sums(spect.blur(torch.stack(maps), cfg.fwhm_mm, grid), st).double())
    return torch.cat(columns).T.contiguous()


@torch.no_grad()
def measure(cohort, i, st, activity_of_source, virtual_id, t_h, cfg):
    """VOI activities of the simulated SPECT of one virtual patient at t_h: (measured (structures,), acquisition draws, image)."""
    grid = float(cohort.phantoms.grid_mm.iloc[0])
    image, info = spect.simulate(spect.activity_map(cohort, i, activity_of_source), cfg, grid, virtual_id, t_h)
    return spect.voi_sums(image, st).double(), info, image


def voi_sd(expected, n_voxels, gain, cfg, discrepancy=0.0):
    """
    SD of a VOI measurement at its expected value: calibration (relative), the correlated voxel noise summed over the
    VOI, sqrt(gain) x voxels x the voxel SD at the mean voxel activity (gain from spect.voi_noise_gain), and a
    relative model discrepancy (what the forward model does not describe: the resolution of the patient, the
    misregistration, biology outside the kinetic model), a number or one value per structure.
    """
    noise = torch.sqrt(gain) * n_voxels * cfg.noise_sd(expected / n_voxels)
    return torch.sqrt((cfg.calibration_sd * expected) ** 2 + noise ** 2 + (discrepancy * expected) ** 2)


def drift_integral(rate, drift, t, t0):
    """Integral from 0 to t of rate x exp(drift (u - t0) / t0) du; rate x t for drift 0."""
    small = np.abs(drift) < 1e-9
    safe = np.where(small, 1.0, drift)
    return rate * np.where(small, t, t0 / safe * np.exp(-safe) * np.expm1(safe * t / t0))


def mismatch(truth, activity, sd_amplitude, sd_rate, seed=20261115, sd_drift=0.0, t0=96.0):
    """
    Biological variability that the PBPK does not represent: for every virtual patient and source, independent
    log-normal factors on the uptake amplitude (SD sd_amplitude) and on the late biological washout rate (SD sd_rate).
    truth: params rows; activity: rows (virtual_id, source, t_h, activity). The activity becomes
    A(t) e^a exp(-lambda_bio (e^b - 1) t); the truth: A0 e^a, lambda_bio e^b, late TIA = A0' / lambda_w', early TIA
    scaled by e^a, TIAC their sum. Returns (truth, activity) with the same columns; the draws depend on
    (seed, virtual_id, source) only.
    Shape mismatch (sd_drift > 0): the late washout is not one exponential. Its biological rate changes in time,
    lambda_bio e^b exp(s (t - t0) / t0), s normal with SD sd_drift (a third draw; the first two are unchanged), so no
    method's kinetic model contains the truth. The activity is A(t) e^a exp(-(L(t) - lambda_bio t)), L the integral of
    the rate; the late TIA is A0 e^a times the integral of exp(-lambda_phys t - L(t)) (trapezoid, 0-4000 h); A0 and
    lambda_bio of the truth are the amplitude and the rate of the tangent exponential at t0.
    """
    import zlib
    keys = truth[["virtual_id", "source"]].drop_duplicates()
    draws = np.array([np.random.default_rng([seed, zlib.crc32(f"{v}|{s}".encode())]).normal(size=3)
                      for v, s in zip(keys.virtual_id, keys.source)])
    keys = keys.assign(shift_amplitude=draws[:, 0] * sd_amplitude, shift_rate=draws[:, 1] * sd_rate, drift=draws[:, 2] * sd_drift)
    out = truth.merge(keys, on=["virtual_id", "source"])
    rate = np.maximum(out.lw - LAMBDA_PHYS_H, 1e-5)
    new_rate = rate * np.exp(out.shift_rate)
    early = out.tiac_early_h * np.exp(out.shift_amplitude)
    if sd_drift > 0:
        grid = np.concatenate([np.linspace(0.0, 400.0, 801), np.linspace(400.0, 4000.0, 1801)[1:]])
        drift, moved_rate = out.drift.to_numpy()[:, None], new_rate.to_numpy()[:, None]
        curve = np.exp(-LAMBDA_PHYS_H * grid - np.minimum(drift_integral(moved_rate, drift, grid, t0), 700.0))
        late = out.a0 * np.exp(out.shift_amplitude) * np.trapezoid(curve, grid, axis=1)
        log_a0 = out.log_a0 + out.shift_amplitude - (drift_integral(new_rate, out.drift, t0, t0) - new_rate * t0)
    else:
        late = out.a0 * np.exp(out.shift_amplitude) / (new_rate + LAMBDA_PHYS_H)
        log_a0 = out.log_a0 + out.shift_amplitude
    out = out.assign(log_a0=log_a0, log_lambda_bio=np.log(new_rate),
                     log_tiac_late=np.log(late), log_tiac=np.log(late + early), early_share=early / (late + early))
    moved = activity.merge(out[["virtual_id", "source", "shift_amplitude", "shift_rate", "drift"]].assign(rate=rate.to_numpy()),
                           on=["virtual_id", "source"])
    if sd_drift > 0:
        exponent = drift_integral(moved.rate * np.exp(moved.shift_rate), moved.drift, moved.t_h, t0) - moved.rate * moved.t_h
    else:
        exponent = moved.rate * np.expm1(moved.shift_rate) * moved.t_h
    moved["activity"] = moved.activity * np.exp(moved.shift_amplitude - exponent)
    return out.drop(columns=["drift"]), moved.drop(columns=["shift_amplitude", "shift_rate", "rate", "drift"])


def hanscheid(measured, t_h):
    """M1a: TIA = A(t) 2 t / ln 2 (Hanscheid et al. 2018), from the VOI activity."""
    return measured * 2.0 * t_h / math.log(2.0)


def madsen(measured, t_h, lambda_eff):
    """M1b: mono-exponential with a population effective rate: TIA = A(t) exp(lambda_eff t) / lambda_eff (Madsen et al. 2018)."""
    return measured * torch.exp(lambda_eff * t_h) / lambda_eff


def bayes_update(prior, measured, R, n_voxels, gain, t_h, cfg, discrepancy=0.0, n_samples=4096, seed=0, student_nu=None, lesion=None,
                 min_ess=200.0, max_samples=65536):
    """
    M2, per structure. prior: {quantity: (mean, sd)} tensors over the structures for log_a0, log_lambda_bio and
    early_share (independent Gaussians; log_lambda_bio Student-t with student_nu degrees of freedom on the structures
    flagged in `lesion`). For every prior sample of a structure, its expected VOI measurement = R[v, v] x A0
    exp(-lambda_w t) + the spill-in of the other sources at their prior mean activity; Gaussian likelihood with
    voi_sd. log lambda_bio and the early share are drawn from the prior; log A0 from an equal mixture of its prior and
    of a Gaussian around the value the measurement implies for the sampled lambda_bio (width PROPOSAL_WIDTH x the
    relative SD of the measurement), with the weights corrected for the proposal: the likelihood is much narrower
    than the prior, and the prior alone leaves too few effective samples. Importance weights give the posterior mean and SD of log A0, log lambda_bio and
    log TIAC = log(A0 / lambda_w / (1 - early share)); the early share keeps its prior. The number of samples is
    doubled while the smallest effective sample size is below min_ess. Returns ({quantity: (mean, sd)}, ess, samples).
    """
    n = measured.numel()
    mean = {q: prior[q][0].double() for q in ("log_a0", "log_lambda_bio", "early_share")}
    sd = {q: prior[q][1].double() for q in ("log_a0", "log_lambda_bio", "early_share")}
    prior_activity = torch.exp(mean["log_a0"] - (torch.exp(mean["log_lambda_bio"]) + LAMBDA_PHYS_H) * t_h)
    spill = R @ prior_activity - torch.diagonal(R) * prior_activity
    diagonal = torch.diagonal(R)
    while True:
        generator = torch.Generator().manual_seed(int(seed))
        z = torch.randn((4, n_samples, n), generator=generator, dtype=torch.float64)
        step = z[1]
        if student_nu is not None and lesion is not None:
            chi = torch.distributions.Chi2(torch.tensor(float(student_nu), dtype=torch.float64))
            torch.manual_seed(int(seed) + 1)
            heavy = z[1] / torch.sqrt(chi.sample((n_samples, n)) / student_nu)
            step = torch.where(lesion, heavy, z[1])
        log_lambda = mean["log_lambda_bio"] + sd["log_lambda_bio"] * step
        share = (mean["early_share"] + sd["early_share"] * z[2]).clamp(max=EARLY_SHARE_MAX)
        rate = torch.exp(log_lambda) + LAMBDA_PHYS_H
        signal = (measured - spill).clamp_min(1e-3 * measured.abs().clamp_min(1e-30))
        centre = torch.log(signal / diagonal) + rate * t_h
        width = (PROPOSAL_WIDTH * voi_sd(signal, n_voxels, gain, cfg, discrepancy) / signal).clamp(0.02, 2.0)
        from_data = torch.rand((n_samples, n), generator=generator, dtype=torch.float64) < 0.5
        log_a0 = torch.where(from_data, centre + width * z[3], mean["log_a0"] + sd["log_a0"] * z[0])
        log_prior = -0.5 * ((log_a0 - mean["log_a0"]) / sd["log_a0"]) ** 2 - torch.log(sd["log_a0"])
        log_data = -0.5 * ((log_a0 - centre) / width) ** 2 - torch.log(width)
        log_proposal = torch.logsumexp(torch.stack([log_prior, log_data]), dim=0)
        expected = diagonal * torch.exp(log_a0 - rate * t_h) + spill
        noise = voi_sd(expected, n_voxels, gain, cfg, discrepancy)
        log_weight = log_prior - log_proposal - 0.5 * ((measured - expected) / noise) ** 2 - torch.log(noise)
        weight = torch.softmax(log_weight, dim=0)
        ess = 1.0 / (weight ** 2).sum(dim=0)
        if float(ess.min()) >= min_ess or n_samples >= max_samples:
            break
        n_samples *= 2
    values = {"log_a0": log_a0, "log_lambda_bio": log_lambda,
              "log_tiac": log_a0 - torch.log(rate) - torch.log1p(-share),
              "log_tiac_late": log_a0 - torch.log(rate), "early_share": share}
    posterior = {}
    for q, x in values.items():
        m = (weight * x).sum(dim=0)
        posterior[q] = (m, torch.sqrt((weight * (x - m) ** 2).sum(dim=0)))
    return posterior, ess, n_samples


def prior_tensors(table, sources, quantities=PRIOR_QUANTITIES):
    """{quantity: (mean, sd)} of one virtual patient from an M0 table, in the order of `sources`; the SD of a
    population entry is its predictive one, sd sqrt(1 + 1/n)."""
    rows = table.set_index("source").reindex(sources)
    out = {}
    for q in quantities:
        n = rows[f"{q}_n"].to_numpy(float)
        widen = np.where(np.isfinite(n) & (n > 1), np.sqrt(1 + 1 / np.where(np.isfinite(n) & (n > 1), n, 1.0)), 1.0)
        out[q] = (torch.from_numpy(rows[f"{q}_mean"].to_numpy(float)), torch.from_numpy(rows[f"{q}_sd"].to_numpy(float) * widen))
    return out


def population_rates(prior_table):
    """Population effective rate lambda_phys + exp(mean log lambda_bio) per structure, from population_prior.parquet."""
    rows = prior_table[prior_table.arm == "main"].set_index("structure")
    return (np.exp(rows.log_lambda_bio_mean) + LAMBDA_PHYS_H).to_dict()


MISMATCH_SD = 0.15
SHAPE_DRIFT_SD = 0.3
DISCREPANCY_GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4)


def case_data(netdata, studies, scenario, condition="mismatch", arm="main", sd=MISMATCH_SD, test=False):
    """
    Truth and source activities of the update experiments for the given phantoms. condition 'mismatch' (main): the
    cohort values perturbed by `mismatch` with SD `sd` on both factors (per-organ variability outside the PBPK); 'shape':
    the same plus a late washout rate that changes in time (SD SHAPE_DRIFT_SD of its log change per 96 h); 'exact':
    the cohort values. Returns (truth indexed by (virtual_id, source), {(virtual_id, t_h): {source: activity}}).
    test: params_test.parquet and spect_activity_test.parquet (frozen test run only); same functions and seeds.
    """
    from pathlib import Path
    params = pd.read_parquet(Path(netdata) / ("params_test.parquet" if test else "params.parquet"))
    truth = params[(params.arm == arm) & (params.scenario == scenario) & params.has_source & params.study_id.isin(set(studies))]
    activity = spect.load_activity(netdata, studies, scenario, arm, test=test)
    if condition == "mismatch":
        truth, activity = mismatch(truth, activity, sd, sd)
    elif condition == "shape":
        truth, activity = mismatch(truth, activity, sd, sd, sd_drift=SHAPE_DRIFT_SD)
    elif condition != "exact":
        raise ValueError(condition)
    table = {key: g.set_index("source").activity.to_dict() for key, g in activity.groupby(["virtual_id", "t_h"])}
    return truth.set_index(["virtual_id", "source"]), table


def structure_cache(cohort, i, st, cfg, folder):
    """Cross-talk matrix and VOI noise gain of a phantom, computed once and stored in folder/<study>.npz."""
    from pathlib import Path
    path = Path(folder) / f"{cohort.phantoms.study_id[i]}.npz"
    if path.exists():
        with np.load(path) as data:
            if "gain" in data.files and list(data["sources"]) == list(st.sources):
                return torch.from_numpy(data["R"]), torch.from_numpy(data["gain"])
    R, gain = crosstalk(cohort, i, st, cfg), spect.voi_noise_gain(cohort, i, st, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    import os
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(temporary, R=R.numpy(), gain=gain.numpy(), sources=np.array(st.sources))
    os.replace(temporary, path)
    return R, gain


def gate_choice(summary, methods, prior="M0 gated prior", quantity="log_tiac", column="median_abs_error"):
    """
    Per structure and SPECT time: True where the method has a lower `column` than the prior on the given summary
    (the tuning half). summary: rows (t_h, method, structure, quantity, column), the prior with t_h NaN.
    Returns {method: {structure: {t_h: bool}}}.
    """
    d = summary[summary.quantity == quantity]
    base = d[d.method == prior].set_index("structure")[column]
    out = {}
    for method in methods:
        rows = d[d.method == method]
        out[method] = {s: {f"{t:g}": bool(v < base[s]) for t, v in zip(g.t_h, g[column])} for s, g in rows.groupby("structure")}
    return out


def gate_at(gate, t_h):
    """Gate of one structure ({time: bool}) at t_h: that of the nearest tuned time (the earlier one on ties); False if none."""
    if not gate:
        return False
    return bool(gate[min(gate, key=lambda k: (abs(float(k) - t_h), float(k)))])


def choose_discrepancy(rows, grid=DISCREPANCY_GRID, quantity="log_tiac"):
    """
    Relative model discrepancy per structure group by matching coverage on the tuning half: rows of the update at
    every value of the grid (columns structure, discrepancy, quantity, mean, sd, truth); the value whose 68% and 95%
    Gaussian intervals of `quantity` cover closest to nominal (sum of the two absolute deviations; the smallest value
    on ties). Returns ({structure: value}, table of the coverages).
    """
    d = rows[rows.quantity == quantity]
    z = ((d["mean"] - d.truth) / d.sd).abs()
    table = pd.DataFrame(dict(structure=d.structure, discrepancy=d.discrepancy, c68=z < 0.9945, c95=z < 1.96))
    table = table.groupby(["structure", "discrepancy"])[["c68", "c95"]].mean().reset_index()
    table["distance"] = (table.c68 - 0.683).abs() + (table.c95 - 0.95).abs()
    best = table.sort_values(["structure", "distance", "discrepancy"]).groupby("structure").first()
    return {s: float(v) for s, v in best.discrepancy.items()}, table
