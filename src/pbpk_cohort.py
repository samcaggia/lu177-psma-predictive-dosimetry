"""Virtual cohort of plan (d): posterior draws per phantom, calibrated population, 177Lu TIA, 68Ga PET values, kinetic parameters.

Decisions, targets and checks are in notebooks/19_pbpk_population_calibration.ipynb (sections 13-15) and notebooks/20_virtual_cohort.ipynb.
"""
import json
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize

import pbpk as P
import pbpk_calibration as C
import pbpk_fit as F

VOID_INTERVAL_H = 4.0
COMPARTMENTS = ("kidneys_functional", "liver", "spleen", "salivary_glands", "bladder", "gi_tract", "blood_pool", "spine", "rest_of_body")
THERAPY_LESION_MODE = "auto"


def kidney_population(results, tau):
    """Median of kidney density and an undeclared error of the kidney datum, by marginal likelihood on the grids of the final pass, with the population SD fixed at tau. Studies of a subject share one value."""
    fine = np.linspace(F.ORGAN_GRID_LOG[0], F.ORGAN_GRID_LOG[-1], 241)
    step = fine[1] - fine[0]
    rows = [r for r in results if "kidneys" in r["organ_grid"]]
    names = list(dict.fromkeys(r["subject"] for r in rows))
    group = np.array([names.index(r["subject"]) for r in rows])
    model = np.array([PchipInterpolator(F.ORGAN_GRID_LOG, r["organ_grid"]["kidneys"]["model"])(fine) for r in rows])
    data = np.array([r["organ_grid"]["kidneys"]["data"] for r in rows])
    variance = np.array([r["organ_grid"]["kidneys"]["variance"] for r in rows])
    base = np.log(F.PRIORS["kidneys"][0])
    se = F.PRIORS["kidneys"][1] / np.sqrt(F.FIVE_PATIENTS)

    def objective(theta):
        total_variance = variance + np.exp(2 * theta[1])
        residual = data[:, None] - model
        per_row = -0.5 * residual**2 / total_variance[:, None] - 0.5 * np.log(total_variance[:, None])
        per_group = np.zeros((len(names), len(fine)))
        np.add.at(per_group, group, per_row)
        joint = per_group + np.log(np.maximum(F._normal(fine + base, theta[0], tau) * step, 1e-300))
        peak = joint.max(axis=1, keepdims=True)
        return -(peak[:, 0] + np.log(np.exp(joint - peak).sum(axis=1))).sum() + 0.5 * ((theta[0] - base) / se) ** 2

    best = minimize(objective, [np.log(40.0), np.log(0.2)], method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-3, maxiter=600))
    return dict(tau=tau, median=float(np.exp(best.x[0])), datum_sigma=float(np.exp(best.x[1])), nll=float(best.fun))


def stage_bundle(args):
    """Posterior of one real phantom under a frozen population: organs and nuisance by importance sampling, lesions kept as a censored likelihood. No population parameter is learned here."""
    study_id, settings, hyper, n_keep = args
    geometry = F.load_geometry(study_id)
    options = F.scanner_options(geometry, settings)
    fit = F.fit_joint(geometry, F.real_measure(geometry), options, hyper)
    rng = np.random.default_rng([F.hash_id(study_id), 91])
    sample = F.posterior_sample(geometry, fit, options, rng, n_proposal=300)
    keep = sample["x"][rng.choice(len(sample["weights"]), n_keep, p=sample["weights"])]
    linear = fit["linear"]
    return dict(study_id=study_id, subject=geometry.subject, scanner=geometry.scanner, geometry=geometry, names=sample["names"], n_r=sample["n_r"], r_keys=sample["r_keys"], draws=keep, ess=sample["ess"], likelihood=fit["likelihood"], obs={k: v for k, v in fit["obs"].items()}, map_nuisance=fit["nuisance"], z={k: float(m / s) for k, m, s in zip(linear["keys"], linear["misfit"], linear["sigma"])}, map_r=dict(fit["r_den"]))


def draw_values(bundle, index, priors):
    """Organ densities and nuisance of one posterior draw."""
    y = bundle["draws"][index]
    n_r = bundle["n_r"]
    r_den = dict(zip(bundle["r_keys"], np.exp(y[:n_r])))
    r_den.setdefault("salivary", {**F.PRIORS, **(priors or {})}["salivary"][0])
    extra = dict(zip(bundle["names"][n_r:], np.exp(y[n_r:])))
    nuisance = bundle["map_nuisance"]
    return dict(r_den=r_den, f_sal=extra.get("f_sal", nuisance.f_sal), gfr=extra.get("gfr", nuisance.gfr), hematocrit=extra.get("hematocrit", nuisance.hematocrit), amount_nmol=extra.get("amount_nmol", nuisance.amount_nmol))


def calibration_bases(bundles, priors, per_patient=3):
    out = []
    for b in bundles:
        for j in range(min(per_patient, len(b["draws"]))):
            v = draw_values(b, j, priors)
            out.append(C.Base(b["geometry"], v["r_den"], v["f_sal"], v["gfr"], v["hematocrit"], b["likelihood"], (F.hash_id(b["study_id"]) + 7919 * j) % 2**31))
    return out


def forward_spread(base, theta, phi_sd, rel_sd, tau_scale=1.0, nmol=C.THERAPY_NMOL):
    """Calibration forward with between-patient spread of the agent factors and of lambda_rel. phi_sd and rel_sd follow the order of AGENT_ORGANS and of kidneys, liver and spleen, salivary, lesions."""
    theta = np.asarray(theta)
    rng = np.random.default_rng([base.seed, 77])
    z_phi, z_rel = rng.normal(size=4), rng.normal(size=4)
    factors = {k: float(np.exp(m + s * x)) for k, m, s, x in zip(C.AGENT_ORGANS, theta[:4], phi_sd, z_phi)}
    rel = theta[4:8] + np.asarray(rel_sd) * z_rel
    full = (C.BASE["log_kd"], C.BASE["log_lambda_int"], *rel, theta[8])
    amount = C.measured_amount(base.seed) if nmol == "measured" else nmol
    return C.forward(base, full, factors=factors, tau_scale=tau_scale, nmol=amount)


@dataclass
class Population:
    """What a virtual patient draws from, apart from the posterior of its phantom."""
    theta: np.ndarray
    phi_sd: dict
    rel_sd: dict
    lesion_epistemic_sd: float
    priors: dict
    hyper: object
    name: str = "main"

    def log_phi(self):
        return dict(zip(C.AGENT_ORGANS, self.theta[:4]))

    def log_rel(self):
        return dict(kidneys=self.theta[4], liver=self.theta[5], salivary=self.theta[6], lesions=self.theta[7])


def _kinetic(t_h, activity, tia):
    """Uptake-washout parameters of a curve: A(t) = A0 (exp(-lw t) - exp(-lu t)), lw from the slope over 24-168 h, A0 from its intercept, lu so that the integral equals the TIA."""
    window = (t_h >= C.TEFF_WINDOW_H[0]) & (t_h <= C.TEFF_WINDOW_H[1]) & (activity > 0)
    if window.sum() < 2:
        return dict(a0=np.nan, lw=np.nan, lu=np.nan)
    slope, intercept = np.polyfit(t_h[window], np.log(activity[window]), 1)
    lw, a0 = -slope, float(np.exp(intercept))
    inverse = 1 / lw - tia / a0 if lw > 0 else np.nan
    return dict(a0=a0, lw=float(lw), lu=float(1 / inverse) if inverse and inverse > 0 else np.inf)


def _spine_ml(study_id):
    meta = json.loads((F.PHANTOMS / study_id / "meta.json").read_text())
    c = meta["compartments"]
    return float(sum(c[k]["volume_ml"] for k in c if k.startswith("spine"))), float(c["blood_pool"]["volume_ml"]) if "blood_pool" in c else 0.0


def simulate_pet(geometry, draw, r_lesions, f_lesions, l_lesions, rel):
    """Decay-corrected PET values of the virtual patient at its uptake time, PET ligand, no agent factor."""
    patient = F.make_patient(geometry, draw["r_den"], draw["f_sal"], draw["gfr"], draw["hematocrit"], r_lesions, f_lesions, l_lesions, rel["kidneys"], rel["salivary"])
    patient.lambda_rel.update(liver=rel["liver"], spleen=rel["liver"])
    injection = P.Injection("Ga68", geometry.activity_mbq, draw["amount_nmol"])
    result = P.simulate(patient, injection, stop_min=geometry.uptake_min + 1, topology="full", times_min=[geometry.uptake_min], lesion_mode=THERAPY_LESION_MODE)
    t = geometry.uptake_min
    out = {k: float(result.percent_ia(k, t, True)) for k in ("kidneys", "liver", "spleen", "salivary", "gi")}
    out["blood_conc"] = F.serum_concentration(result, t) * (1 - draw["hematocrit"])
    out["urine"] = float(result.percent_ia("urine", t, True))
    out["lesions"] = np.array([float(result.percent_ia(f"lesion_{j}", t, True)) for j in range(len(geometry.lesions_ml))])
    return out


def simulate_therapy(geometry, draw, r_lesions, f_lesions, l_lesions, rel, factors, spine_ml, blood_ml, lesion_mode=THERAPY_LESION_MODE, nmol=C.THERAPY_NMOL):
    """177Lu therapy of the virtual patient: TIAC per Monte Carlo source and per lesion, kinetic parameters, doses by S-value for the checks only."""
    r_den = {k: v * factors.get(k, 1.0) for k, v in draw["r_den"].items()}
    patient = F.make_patient(geometry, r_den, draw["f_sal"], draw["gfr"], draw["hematocrit"], r_lesions, f_lesions, l_lesions, rel["kidneys"], rel["salivary"])
    patient.lambda_rel.update(liver=rel["liver"], spleen=rel["liver"])
    result = P.simulate(patient, P.Injection("Lu177", C.THERAPY_MBQ, nmol), topology="full", lesion_mode=lesion_mode)
    t_h, lam = result.time_h, result.injection.lambda_phys
    decay = np.exp(-lam * result.time_min)
    groups = [k for k in result.net.groups if k not in ("urine", "released", "tumour") and not k.startswith("lesion_")]
    tissue_tia = {k: result.tiac_h(k, include_tail=True) for k in groups}
    tissue_curve = {k: result.fraction(k) for k in groups}
    amount = result.amount_nmol("arteries") if "arteries" in result.net.groups else None
    index = result.net.groups["blood"]
    serum_fraction = (result.y[:, index["arteries"]] + result.y[:, index["veins"]]) / result.injection.total_nmol * decay / ((result.net.body["Vart"] + result.net.body["Vven"]) * 1000)
    blood_curve = serum_fraction * (1 - draw["hematocrit"]) * blood_ml
    blood_tia = float(np.trapezoid(blood_curve, t_h))
    skeleton = [k for k in ("red_marrow", "bone") if k in tissue_tia]
    tissues, _, _ = P.tissue_table(patient)
    skeleton_ml = sum(tissues[k]["V"] for k in skeleton) * 1000
    spine_share = min(spine_ml / skeleton_ml, 1.0) if skeleton_ml else 0.0
    spine_curve, spine_tia = sum(tissue_curve[k] for k in skeleton) * spine_share, sum(tissue_tia[k] for k in skeleton) * spine_share
    urine = result.amount_nmol("urine") / result.injection.total_nmol
    urine_decays = float(np.sum(np.diff(urine) * 0.5 * (decay[1:] + decay[:-1])))
    bladder_tia = VOID_INTERVAL_H / 2 * urine_decays
    bladder_curve = np.gradient(urine, t_h) * decay * VOID_INTERVAL_H / 2
    source = dict(kidneys_functional=("kidneys",), liver=("liver",), spleen=("spleen",), salivary_glands=("salivary",), gi_tract=("gi",))
    tia, curve = {}, {}
    for name, labels in source.items():
        tia[name], curve[name] = sum(tissue_tia[k] for k in labels), sum(tissue_curve[k] for k in labels)
    tia.update(bladder=bladder_tia, blood_pool=blood_tia, spine=spine_tia)
    curve.update(bladder=bladder_curve, blood_pool=blood_curve, spine=spine_curve)
    total_tia, total_curve = sum(tissue_tia.values()), sum(tissue_curve.values())
    tia["rest_of_body"] = total_tia - sum(tia[k] for k in ("kidneys_functional", "liver", "spleen", "salivary_glands", "gi_tract", "blood_pool", "spine"))
    curve["rest_of_body"] = total_curve - sum(curve[k] for k in ("kidneys_functional", "liver", "spleen", "salivary_glands", "gi_tract", "blood_pool", "spine"))
    kinetics = {k: _kinetic(t_h, curve[k], tia[k]) for k in tia}
    n = len(geometry.lesions_ml)
    lesion_tia = np.array([result.tiac_h(f"lesion_{j}", include_tail=True) for j in range(n)])
    lesion_kinetics = [_kinetic(t_h, result.fraction(f"lesion_{j}"), lesion_tia[j]) for j in range(n)]
    remaining = decay - result.fraction("urine") - result.fraction("released")
    window = (t_h >= C.WHOLE_BODY_WINDOW_H[0]) & (t_h <= C.WHOLE_BODY_WINDOW_H[1])
    check = dict(teff_whole_body=float(np.log(2) / -np.polyfit(t_h[window], np.log(remaining[window]), 1)[0]), excreted_24h=float(np.interp(24.0, t_h, urine)))
    for key, volume in dict(kidneys=geometry.kidneys_ml, liver=geometry.liver_ml, spleen=geometry.spleen_ml, salivary=geometry.salivary_ml or 1.0).items():
        check[f"dose_{key}"] = C.self_dose(tissue_tia[key], key, volume)
    for key in ("kidneys", "liver", "salivary"):
        check[f"teff_{key}"] = result.t_eff_h(key, *C.TEFF_WINDOW_H)
    keep = np.where(geometry.lesions_ml >= C.MIN_LESION_ML)[0] if n else np.zeros(0, int)
    if len(keep):
        check["dose_lesions"] = float(np.mean([C.self_dose(lesion_tia[j], "lesions", geometry.lesions_ml[j]) for j in keep]))
        check["teff_lesions"] = float(np.median([np.log(2) / lesion_kinetics[j]["lw"] for j in keep if lesion_kinetics[j]["lw"] > 0]))
    return dict(tia=tia, kinetics=kinetics, lesion_tia=lesion_tia, lesion_kinetics=lesion_kinetics, check=check)


def whole_body_check(base, theta, gfr_scale=1.0):
    """Excretion and whole-body half-life of one patient at the calibrated population, for the comparison with Kurth 2018."""
    theta = np.asarray(theta)
    g = base.geometry
    n = len(g.lesions_ml)
    rng = np.random.default_rng(base.seed)
    r_lesions, f_lesions = F.draw_lesions(base.likelihood, replace(F.LesionHyper(), mu=float(theta[8])), rng) if n else (np.zeros(0), np.zeros(0))
    r_den = {k: v * float(np.exp(dict(zip(C.AGENT_ORGANS, theta[:4]))[k])) for k, v in base.r_den.items()}
    patient = F.make_patient(g, r_den, base.f_sal, base.gfr * gfr_scale, base.hematocrit, r_lesions, f_lesions, np.full(n, float(np.exp(theta[7]))), float(np.exp(theta[4])), float(np.exp(theta[6])))
    patient.lambda_rel.update(liver=float(np.exp(theta[5])), spleen=float(np.exp(theta[5])))
    result = P.simulate(patient, P.Injection("Lu177", C.THERAPY_MBQ, C.THERAPY_NMOL), topology="full", lesion_mode="satellite")
    t_h, decay = result.time_h, np.exp(-result.injection.lambda_phys * result.time_min)
    urine = result.fraction("urine") / decay
    remaining = decay - result.fraction("urine") - result.fraction("released")
    window = (t_h >= C.WHOLE_BODY_WINDOW_H[0]) & (t_h <= C.WHOLE_BODY_WINDOW_H[1])
    out = dict(tlv_ml=float(g.lesions_ml.sum()), teff_whole_body=float(np.log(2) / -np.polyfit(t_h[window], np.log(remaining[window]), 1)[0]))
    out.update({f"excreted_{h}h": float(np.interp(h, t_h, urine)) for h in (2, 4, 12, 24, 48, 72)})
    return out
