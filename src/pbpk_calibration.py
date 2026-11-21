"""Population calibration of plan (b): therapy-ligand K_D and lambda_int, lambda_rel per structure and the lesion density population, against published 177Lu-PSMA-617 half-lives and doses.

Targets, tolerances and results are in notebooks/19_pbpk_population_calibration.ipynb. Patients come from the fits of nb18 on the training split.
"""
from dataclasses import dataclass, replace

import numpy as np

import pbpk as P
import pbpk_fit as F

K_ON = 0.046
THERAPY_MBQ = 7400.0
THERAPY_NMOL = 100.0
PLUVICTO_UG_PER_ML = (6.31, 0.72)
PLUVICTO_ML = (7.5, 12.5)
PSMA617_G_PER_MOL = 1042.1
S_EANM = dict(kidneys=(0.204, 422.0), liver=(0.0376, 2360.0), spleen=(0.377, 228.4), salivary=(0.947, 88.98))
DELTA_E = 85.3
DENSITY = dict(kidneys=1.05, liver=1.05, spleen=1.04, salivary=1.03, lesions=1.04)
MIN_LESION_ML = 0.5
TEFF_WINDOW_H = (24.0, 168.0)
WHOLE_BODY_WINDOW_H = (24.0, 96.0)
NAMES = ("log_kd", "log_lambda_int", "log_rel_kidneys", "log_rel_liver", "log_rel_salivary", "log_rel_lesions", "log_median_r")
BOX = np.log([[0.01, 30.0], [1e-4, 3e-2], [2e-5, 3e-3], [2e-5, 3e-3], [2e-5, 3e-3], [2e-5, 3e-3], [10.0, 3000.0]])
BASE = dict(log_kd=np.log(1.0), log_lambda_int=np.log(1e-3), log_rel_kidneys=np.log(F.PRIORS["l_kidney"][0]), log_rel_liver=np.log(F.PRIORS["l_kidney"][0]), log_rel_salivary=np.log(F.PRIORS["l_salivary"][0]), log_rel_lesions=np.log(F.PRIORS["l_tumor"][0]), log_median_r=np.log(F.TUMOUR_MEDIAN))
OUTPUTS = ("teff_kidneys", "teff_liver", "teff_salivary", "teff_lesions", "teff_whole_body", "dose_kidneys", "dose_liver", "dose_spleen", "dose_salivary", "dose_lesions")


@dataclass
class Base:
    """One patient of the real fit: what the therapy simulation keeps from the PET."""
    geometry: object
    r_den: dict
    f_sal: float
    gfr: float
    hematocrit: float
    likelihood: np.ndarray
    seed: int


def measured_amount(seed):
    """Therapy ligand of one patient in nmol: PSMA-617 concentration measured in clinical Pluvicto doses (Wang 2025) times an injected volume within the range of the package insert."""
    rng = np.random.default_rng([seed, 88])
    concentration = max(rng.normal(*PLUVICTO_UG_PER_ML), 0.5)
    return concentration * rng.uniform(*PLUVICTO_ML) * 1000 / PSMA617_G_PER_MOL


def self_dose(tiac, structure, volume_ml):
    """Gy/GBq from the TIAC in hours: organ S-value of EANM 2022 scaled by mass, local deposition for lesions."""
    mass = volume_ml * DENSITY[structure]
    if structure == "lesions":
        return tiac * DELTA_E / mass
    s, reference = S_EANM[structure]
    return tiac * s * reference / mass


def _lesion_hyper(median_log):
    start = F.LesionHyper()
    return replace(start, mu=float(median_log))


def forward(base, theta, lesions=True, nmol=THERAPY_NMOL, factors=None, tau_scale=1.0):
    """Therapy of one patient under a candidate population. Densities, GFR, hematocrit and f_SAL come from the PET fit under the PET ligand; the therapy ligand differs. Lesions: density and flow from the joint posterior given the candidate population, same random numbers for every candidate."""
    t = dict(zip(NAMES, theta))
    g = base.geometry
    n = len(g.lesions_ml)
    rng = np.random.default_rng(base.seed)
    if n and lesions:
        hyper = _lesion_hyper(t["log_median_r"])
        r_lesions, f_lesions = F.draw_lesions(base.likelihood, replace(hyper, tau_between=hyper.tau_between * tau_scale, tau_within=hyper.tau_within * tau_scale), rng)
    else:
        r_lesions, f_lesions = np.full(n, float(np.exp(t["log_median_r"]))), np.full(n, F.PRIORS["f_lesion"][0])
    rel_kidneys, rel_liver, rel_salivary, rel_lesions = (float(np.exp(t[k])) for k in ("log_rel_kidneys", "log_rel_liver", "log_rel_salivary", "log_rel_lesions"))
    r_den = {k: v * (factors or {}).get(k, 1.0) for k, v in base.r_den.items()}
    patient = F.make_patient(g, r_den, base.f_sal, base.gfr, base.hematocrit, r_lesions, f_lesions, np.full(n, rel_lesions), rel_kidneys, rel_salivary)
    patient.lambda_rel.update(liver=rel_liver, spleen=rel_liver)
    ligand = P.Ligand(k_off=float(np.exp(t["log_kd"])) * K_ON, lambda_int=float(np.exp(t["log_lambda_int"])))
    result = P.simulate(patient, P.Injection("Lu177", THERAPY_MBQ, nmol), ligand=ligand, lesion_mode="satellite", topology="full")
    out = {f"teff_{k}": result.t_eff_h(k, *TEFF_WINDOW_H) for k in ("kidneys", "liver", "salivary")}
    volumes = dict(kidneys=g.kidneys_ml, liver=g.liver_ml, spleen=g.spleen_ml, salivary=g.salivary_ml or 1.0)
    for key, volume in volumes.items():
        out[f"dose_{key}"] = self_dose(result.tiac_h(key, include_tail=True), key, volume)
    excreted = result.fraction("urine") + result.fraction("released")
    remaining = np.exp(-result.injection.lambda_phys * result.time_min) - excreted
    window = (result.time_h >= WHOLE_BODY_WINDOW_H[0]) & (result.time_h <= WHOLE_BODY_WINDOW_H[1])
    out["teff_whole_body"] = float(np.log(2) / -np.polyfit(result.time_h[window], np.log(remaining[window]), 1)[0])
    out["tumour_sites_nmol"] = float(np.sum(r_lesions * g.lesions_ml) / 1000)
    keep = np.where(g.lesions_ml >= MIN_LESION_ML)[0] if n else np.zeros(0, int)
    if len(keep):
        out["dose_lesions"] = float(np.mean([self_dose(result.tiac_h(f"lesion_{j}", include_tail=True), "lesions", g.lesions_ml[j]) for j in keep]))
        out["teff_lesions"] = float(np.median([result.t_eff_h(f"lesion_{j}", *TEFF_WINDOW_H) for j in keep]))
    return out


def population(rows):
    """Median half-life, mean dose over patients: how the sources report them."""
    out = {}
    for key in OUTPUTS:
        values = np.array([r[key] for r in rows if key in r and np.isfinite(r[key])])
        out[key] = float(np.median(values) if key.startswith("teff") else np.mean(values))
    return out


def latin_hypercube(n, rng, box=BOX):
    d = box.shape[0]
    cut = (np.argsort(rng.uniform(size=(n, d)), axis=0) + rng.uniform(size=(n, d))) / n
    return box[:, 0] + cut * (box[:, 1] - box[:, 0])


def stretch_sampler(log_density, start, n_steps, rng, a=2.0):
    """Affine-invariant ensemble sampler: log_density takes an array of points. start has an even number of walkers."""
    walkers = start.copy()
    n, d = walkers.shape
    half = n // 2
    lp = log_density(walkers)
    chain = np.zeros((n_steps, n, d))
    accepted = 0
    for step in range(n_steps):
        for first, second in ((np.arange(half), np.arange(half, n)), (np.arange(half, n), np.arange(half))):
            partners = walkers[rng.choice(second, len(first))]
            z = ((a - 1) * rng.uniform(size=len(first)) + 1) ** 2 / a
            trial = partners + z[:, None] * (walkers[first] - partners)
            new = log_density(trial)
            keep = np.log(rng.uniform(size=len(first))) < (d - 1) * np.log(z) + new - lp[first]
            walkers[first[keep]] = trial[keep]
            lp[first[keep]] = new[keep]
            accepted += keep.sum()
        chain[step] = walkers
    return chain, accepted / (2 * n_steps * half)


AGENT_ORGANS = ("kidneys", "salivary", "liver", "spleen")
AGENT_NAMES = ("log_phi_kidneys", "log_phi_salivary", "log_phi_liver", "log_phi_spleen", "log_rel_kidneys", "log_rel_liver", "log_rel_salivary", "log_rel_lesions", "log_median_r")
AGENT_BOX = np.log([[0.03, 5.0], [0.03, 5.0], [0.03, 5.0], [0.03, 5.0], [5e-5, 1e-3], [5e-5, 1e-3], [5e-5, 1e-3], [2e-5, 5e-4], [10.0, 3000.0]])


def agent_factors(base, log_phi, spread):
    """Therapy over PET binding capacity per organ. Spread: log SD between patients, same random numbers for every candidate."""
    z = np.random.default_rng([base.seed, 77]).normal(size=len(AGENT_ORGANS))
    return {k: float(np.exp(m + spread * x)) for k, m, x in zip(AGENT_ORGANS, log_phi, z)}


def forward_agent(base, theta, spread=0.0):
    """Therapy under the agent factors, with K_D and lambda_int at the base: the factors act on the densities fitted under the PET ligand, so mass balance and the plasma curve follow."""
    theta = np.asarray(theta)
    full = (BASE["log_kd"], BASE["log_lambda_int"], *theta[4:9])
    return forward(base, full, factors=agent_factors(base, theta[:4], spread))
