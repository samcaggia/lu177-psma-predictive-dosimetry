"""Fit of src/pbpk.py to a single 68Ga PET, tested on simulated PETs painted on the nb11 phantoms.

Priors, assumptions and results are in notebooks/13_pbpk_fit.ipynb. Real PETs are read from nb17 on, training split only (`real_measure`, `stage_diagnostic_real`).
"""
import json
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares, minimize
from scipy.stats import norm

import pbpk as P

WORKSPACE = Path(__file__).resolve().parents[1]
PHANTOMS = WORKSPACE / "data/processed/psma/phantoms"
SINGLE_KIDNEY_ML = 20.0
KIDNEY_GROW_MM = 10.0
SINUS_CLOSING_MM = 15.0
LABEL = dict(kidney=1, fluid=2, liver=3, spleen=4, glands=(5, 6, 7, 8), bladder=9, gi=10, blood=11, spine=(12, 13), rest=14, lesion=15)
ORGANS = ("kidneys", "liver", "spleen", "salivary")

S2_R_KIDNEY = np.array([31.0, 14, 22, 24, 46])
S2_R_SALIVARY = np.array([41.0, 38, 108, 41, 80])
S2_R_TUMOR_PAIRS = np.array([[520.0, 2412], [57, 19], [282, 83], [52, 51], [136, 190]])
S2_F_SAL = np.array([0.075, 0.074, 0.53, 0.16, 0.18])
TABLE1_TER = np.array([198.0, 201, 136, 252, 176])
S2_L_KIDNEY = np.array([3.7e-4, 2.9e-4, 2.4e-4, 2.3e-4, 3.1e-4])
S2_L_SALIVARY = np.array([3.7e-4, 4.2e-4, 3.5e-4, 3.3e-4, 4.8e-4])
S2_L_TUMOR = np.array([2.4e-4, 1.5e-4, 0.88e-4, 2.0e-4, 2.0e-4])


def _lognormal(values):
    logs = np.log(values)
    return float(np.exp(logs.mean())), float(logs.std(ddof=1))


def tumour_prior():
    """Median and between-patient and within-patient log SD from the five lesion pairs of the S2 File."""
    logs = np.log(S2_R_TUMOR_PAIRS)
    within = np.sqrt(np.mean(np.diff(logs, axis=1) ** 2) / 2)
    total = logs.std(ddof=1)
    return float(np.exp(logs.mean())), float(np.sqrt(max(total**2 - within**2, 0.0))), float(within)


PRIORS = dict(
    kidneys=_lognormal(S2_R_KIDNEY), salivary=_lognormal(S2_R_SALIVARY), liver=(1.33, 1.0), spleen=(0.532, 1.2),
    f_lesion=(0.2, float(np.log(1.6 / 0.02) / 3.92)), f_sal=_lognormal(S2_F_SAL), gfr=(_lognormal(TABLE1_TER / 3 * 20 / 15)),
    hematocrit=(0.42, 0.04), amount_nmol=(2.0, 0.5), gland_volume_factor=(1.7, 0.2),
    l_kidney=_lognormal(S2_L_KIDNEY), l_salivary=_lognormal(S2_L_SALIVARY), l_tumor=_lognormal(S2_L_TUMOR),
)
TUMOUR_MEDIAN, TUMOUR_BETWEEN, TUMOUR_WITHIN = tumour_prior()
TUMOUR_SD = float(np.hypot(TUMOUR_BETWEEN, TUMOUR_WITHIN))
SIGMA = dict(kidneys=0.08, liver=0.08, spleen=0.10, salivary=0.12, lesion=0.15, blood=0.10)
FWHM_MM = dict(Discovery690=6.5, Biograph64=7.5, Biograph20_mCT=6.5)
FWHM_NOMINAL_MM = 7.0
HEMATOCRIT_BOUNDS = (0.2, 0.65)
SCANNER_EFFECT = dict(Discovery690=-0.05, Biograph64=0.06, Biograph20_mCT=0.03)
MISREGISTRATION_MM = 4.0
BLADDER_RETAINED = 0.3
RING_MM = (8.0, 16.0)
ORGAN_IMAGE_SD = dict(liver=0.13, spleen=0.15)
SPLEEN_T = (3.0, 0.12)
R_GRID = np.geomspace(1.0, 3.0e4, 61)
F_GRID_Z = np.linspace(-2.0, 2.0, 9)


@dataclass
class LesionHyper:
    mu: float = float(np.log(TUMOUR_MEDIAN))
    tau_between: float = TUMOUR_BETWEEN
    tau_within: float = TUMOUR_WITHIN
    mu_flow: float = float(np.log(PRIORS["f_lesion"][0]))
    tau_flow: float = PRIORS["f_lesion"][1]
    pi_low: float = 0.0
    mu_low: float = float(np.log(3.0))
    tau_low: float = 0.7
    flow_cap: float = np.inf


@dataclass
class Geometry:
    study_id: str
    subject: str
    scanner: str
    weight_kg: float
    height_cm: float
    activity_mbq: float
    uptake_min: float
    n_kidneys: int
    kidneys_ml: float
    liver_ml: float
    spleen_ml: float
    salivary_ml: float
    has_glands: bool
    lesions_ml: np.ndarray
    voxel_ml: float
    spacing_mm: tuple
    shell_ml: float = 0.0
    rc: dict = field(default_factory=dict)
    rc_lesions: np.ndarray = None
    gland_labels: tuple = LABEL["glands"]


@dataclass
class Truth:
    r_den: dict
    r_lesions: np.ndarray
    f_lesions: np.ndarray
    l_lesions: np.ndarray
    f_sal: float
    gfr: float
    hematocrit: float
    amount_nmol: float
    gland_factor: float
    l_kidney: float
    l_salivary: float
    scanner_effect: float
    fwhm_mm: float
    shift_mm: np.ndarray


def _bbox(mask, margin):
    index = np.where(mask)
    return tuple(slice(max(i.min() - margin, 0), min(i.max() + margin + 1, n)) for i, n in zip(index, mask.shape))


def recovery(mask, spacing_mm, fwhm_mm):
    """Mean over the mask of the PSF-blurred mask: the recovery coefficient of a uniform structure in zero background."""
    sigma = np.array([fwhm_mm / 2.3548 / s for s in spacing_mm])
    box = _bbox(mask, int(np.ceil(4 * sigma.max())) + 1)
    crop = mask[box].astype(np.float32)
    return float(ndi.gaussian_filter(crop, sigma)[mask[box]].mean())


def grown_kidney(labels, spacing_mm):
    kidney = np.isin(labels, [LABEL["kidney"], LABEL["fluid"]])
    box = _bbox(kidney, int(np.ceil(KIDNEY_GROW_MM / min(spacing_mm))) + 2)
    distance = ndi.distance_transform_edt(~kidney[box], sampling=spacing_mm)
    out = np.zeros(labels.shape, bool)
    out[box] = (distance <= KIDNEY_GROW_MM) & (kidney[box] | (labels[box] == LABEL["rest"]))
    return out, kidney


def renal_sinus(kidney, grown, spacing_mm):
    """Part of the grown mask inside the concavity of the kidney: morphological closing of the CT kidney with a 15 mm radius, minus the kidney. Pelvis, calyces and hilum lie here."""
    pad = int(np.ceil(2 * SINUS_CLOSING_MM / min(spacing_mm))) + 2
    box = _bbox(kidney, pad)
    dilated = ndi.distance_transform_edt(~kidney[box], sampling=spacing_mm) <= SINUS_CLOSING_MM
    closed = ndi.distance_transform_edt(dilated, sampling=spacing_mm) > SINUS_CLOSING_MM
    out = np.zeros(kidney.shape, bool)
    out[box] = closed & ~kidney[box]
    return out & grown


def load_geometry(study_id, psf="scanner"):
    """Masks, volumes and recovery coefficients of a phantom. psf: scanner (FWHM_MM of its scanner) or nominal (the same FWHM for all, as up to nb17)."""
    folder = PHANTOMS / study_id
    meta = json.loads((folder / "meta.json").read_text())
    image = nib.load(folder / "organs.nii.gz")
    labels = np.asanyarray(image.dataobj)
    lesions = np.asanyarray(nib.load(folder / "lesions.nii.gz").dataobj)
    spacing = tuple(float(x) for x in meta["grid"]["spacing_mm"])
    groups = meta["groups"]
    glands_used = [g for g, v in meta["salivary_glands"].items() if v["use_for_pbpk"]]
    gland_labels = [meta["salivary_glands"][g]["label"] for g in glands_used]
    n_ids = int(meta["lesions"]["n"])
    volumes = np.array(meta["lesions"]["volume_ml_by_id"], float) if n_ids else np.zeros(0)
    single = [s for s in ("left", "right") if not meta["kidneys"][s]["present"] or meta["kidneys"][s]["functional_ml"] < SINGLE_KIDNEY_ML]
    fwhm = FWHM_MM[meta["scanner"]] if psf == "scanner" else FWHM_NOMINAL_MM
    geometry = Geometry(
        study_id, meta["subject"], meta["scanner"], meta["weight_kg"], (meta["height_m"] or 1.75) * 100, meta["injected_activity_mbq"], meta["uptake_time_min"],
        1 if single else 2, meta["compartments"]["kidney_functional"]["volume_ml"], meta["compartments"]["liver"]["volume_ml"], meta["compartments"]["spleen"]["volume_ml"],
        groups["salivary_glands_pbpk"]["volume_ml"] if glands_used else groups["salivary_glands_all"]["volume_ml"], bool(glands_used), volumes, meta["grid"]["voxel_ml"], spacing)
    geometry.gland_labels = tuple(gland_labels) or LABEL["glands"]
    grown, kidney = grown_kidney(labels, spacing)
    geometry.shell_ml = float((grown & ~kidney).sum()) * geometry.voxel_ml
    geometry.rc = dict(liver=recovery(labels == LABEL["liver"], spacing, fwhm), spleen=recovery(labels == LABEL["spleen"], spacing, fwhm))
    if (labels == LABEL["blood"]).any():
        geometry.rc["blood"] = recovery(labels == LABEL["blood"], spacing, fwhm)
    if gland_labels:
        geometry.rc["salivary"] = float(np.mean([recovery(labels == g, spacing, fwhm) for g in gland_labels]))
    if n_ids:
        boxes = ndi.find_objects(lesions, max_label=n_ids)
        rc = []
        sigma = np.array([fwhm / 2.3548 / s for s in spacing])
        margin = int(np.ceil(4 * sigma.max())) + 1
        for j, box in enumerate(boxes, start=1):
            big = tuple(slice(max(b.start - margin, 0), min(b.stop + margin, n)) for b, n in zip(box, lesions.shape))
            crop = lesions[big] == j
            rc.append(float(ndi.gaussian_filter(crop.astype(np.float32), sigma)[crop].mean()) if crop.any() else np.nan)
        geometry.rc_lesions = np.array(rc)
    else:
        geometry.rc_lesions = np.zeros(0)
    return geometry


def draw_truth(geometry, rng, priors=None, hyper=None):
    """Parameters of one simulated patient. priors and hyper replace the organ and lesion populations, for the empirical Bayes test."""
    table = {**PRIORS, **(priors or {})}
    hyper = hyper or LesionHyper()
    lognormal = lambda key: float(table[key][0] * np.exp(rng.normal(0, table[key][1])))
    r_den = {k: lognormal(k) for k in ORGANS}
    n = len(geometry.lesions_ml)
    patient_effect = rng.normal(0, hyper.tau_between)
    r_lesions = np.exp(hyper.mu + patient_effect + rng.normal(0, hyper.tau_within, n))
    if hyper.pi_low > 0:
        low = rng.uniform(size=n) < hyper.pi_low
        r_lesions = np.where(low, np.exp(hyper.mu_low + rng.normal(0, hyper.tau_low, n)), r_lesions)
    f_lesions = np.exp(hyper.mu_flow + rng.normal(0, hyper.tau_flow, n))
    l_lesions = PRIORS["l_tumor"][0] * np.exp(rng.normal(0, PRIORS["l_tumor"][1], n))
    return Truth(r_den, r_lesions, f_lesions, l_lesions, lognormal("f_sal"), lognormal("gfr"), float(np.clip(rng.normal(*PRIORS["hematocrit"]), 0.25, 0.55)), lognormal("amount_nmol"),
                 lognormal("gland_volume_factor"), lognormal("l_kidney"), lognormal("l_salivary"), SCANNER_EFFECT[geometry.scanner], FWHM_MM[geometry.scanner], rng.normal(0, MISREGISTRATION_MM, 3))


def make_patient(geometry, r_den, f_sal, gfr, hematocrit, r_lesions, f_lesions, l_lesions, l_kidney=None, l_salivary=None, salivary_ml=None):
    lambda_rel = {}
    if l_kidney is not None:
        lambda_rel.update(kidneys=l_kidney, liver=l_kidney, spleen=l_kidney, gi=l_kidney, prostate=l_kidney)
    if l_salivary is not None:
        lambda_rel["salivary"] = l_salivary
    lesions = [P.Lesion(float(v), float(r), float(l), float(f), name=f"lesion_{j}") for j, (v, r, l, f) in enumerate(zip(geometry.lesions_ml, r_lesions, l_lesions, f_lesions))]
    return P.Patient(geometry.study_id, geometry.weight_kg, geometry.kidneys_ml, geometry.liver_ml, geometry.spleen_ml, salivary_ml or geometry.salivary_ml or 1.0, height_cm=geometry.height_cm, hematocrit=hematocrit,
                     gfr_ml_min=gfr * (1 if geometry.n_kidneys == 2 else 0.5), n_kidneys=geometry.n_kidneys, f_sal=f_sal, r_den=dict(r_den), lambda_rel=lambda_rel, lesions=lesions)


def truth_patient(geometry, truth):
    return make_patient(geometry, truth.r_den, truth.f_sal, truth.gfr, truth.hematocrit, truth.r_lesions, truth.f_lesions, truth.l_lesions, truth.l_kidney, truth.l_salivary, (geometry.salivary_ml or 1.0) * truth.gland_factor)


def simulate_truth(geometry, truth, ligand=None):
    """Full topology, lesions automatic: the fit runs the reduced model, so model reduction is part of the test."""
    injection = P.Injection("Ga68", geometry.activity_mbq, truth.amount_nmol)
    return P.simulate(truth_patient(geometry, truth), injection, stop_min=geometry.uptake_min + 1, ligand=ligand, topology="full", times_min=[geometry.uptake_min])


def serum_concentration(result, t_min):
    """%IA/mL of decay-corrected ligand in the serum of the large vessels."""
    _, _, body = P.tissue_table(result.patient)
    index = result.net.groups["blood"]
    amount = np.interp(t_min, result.time_min, result.y[:, index["arteries"]] + result.y[:, index["veins"]])
    return float(amount / result.injection.total_nmol * 100 / ((body["Vart"] + body["Vven"]) * 1000))


def true_percent(result, geometry):
    t = geometry.uptake_min
    out = {k: float(result.percent_ia(k, t, True)) for k in ORGANS}
    out["lesions"] = np.array([float(result.percent_ia(f"lesion_{j}", t, True)) for j in range(len(geometry.lesions_ml))])
    known = [k for k in result.net.groups if k not in ("urine", "released", "blood", *ORGANS) and not k.startswith("lesion_") and k != "tumour"]
    out["blood_total"] = float(result.percent_ia("blood", t, True))
    out["gi"] = float(result.percent_ia("gi", t, True))
    out["other"] = sum(float(result.percent_ia(k, t, True)) for k in known if k != "gi")
    out["serum_conc"] = serum_concentration(result, t)
    out["urine"] = float(result.percent_ia("urine", t, True))
    return out


def paint_and_measure(geometry, truth, result, labels=None, lesions=None):
    """Noise-free measurement: uniform concentrations on the labels, PET/CT shift, Gaussian PSF, then the masks of nb11."""
    folder = PHANTOMS / geometry.study_id
    labels = np.asanyarray(nib.load(folder / "organs.nii.gz").dataobj) if labels is None else labels
    lesions = np.asanyarray(nib.load(folder / "lesions.nii.gz").dataobj) if lesions is None else lesions
    p = true_percent(result, geometry)
    voxel = geometry.voxel_ml
    volume = lambda codes: float(np.isin(labels, codes).sum()) * voxel
    conc_blood = p["serum_conc"] * (1 - truth.hematocrit)
    rest_codes = [LABEL["rest"], *LABEL["spine"]]
    conc_rest = p["other"] / max(volume(rest_codes), 1.0)
    image = np.zeros(labels.shape, np.float32)
    image[labels == LABEL["kidney"]] = p["kidneys"] / geometry.kidneys_ml
    image[labels == LABEL["liver"]] = p["liver"] / geometry.liver_ml
    image[labels == LABEL["spleen"]] = p["spleen"] / geometry.spleen_ml
    gland_ml = (geometry.salivary_ml or 1.0) * truth.gland_factor
    image[np.isin(labels, list(LABEL["glands"]))] = p["salivary"] / gland_ml
    bladder = volume([LABEL["bladder"]])
    if bladder > 0:
        image[labels == LABEL["bladder"]] = BLADDER_RETAINED * p["urine"] / bladder
    gi = volume([LABEL["gi"]])
    if gi > 0:
        image[labels == LABEL["gi"]] = p["gi"] / gi
    image[labels == LABEL["blood"]] = conc_blood
    image[np.isin(labels, rest_codes)] = conc_rest
    if len(geometry.lesions_ml):
        table = np.zeros(len(geometry.lesions_ml) + 1, np.float32)
        table[1:] = p["lesions"] / geometry.lesions_ml
        hit = lesions > 0
        image[hit] = table[lesions[hit]]
    sigma = [truth.fwhm_mm / 2.3548 / s for s in geometry.spacing_mm]
    blurred = ndi.gaussian_filter(image, sigma)
    blurred *= float(np.exp(truth.scanner_effect))
    moved = ndi.shift(blurred, truth.shift_mm / np.array(geometry.spacing_mm), order=1, mode="nearest")
    return measure_image(geometry, blurred, moved, labels, lesions), p


def ring_median(image, mask, spacing_mm, exclude):
    """Median of the image in a shell around a mask, measured beyond the reach of the structure's own signal."""
    inner, outer = RING_MM
    pad = [int(np.ceil(outer / s)) + 1 for s in spacing_mm]
    where = np.nonzero(mask)
    box = tuple(slice(max(int(w.min()) - m, 0), min(int(w.max()) + m + 1, n)) for w, m, n in zip(where, pad, mask.shape))
    distance = ndi.distance_transform_edt(~mask[box], sampling=spacing_mm)
    ring = (distance > inner) & (distance <= outer) & ~exclude[box]
    return float(np.median(image[box][ring])) if ring.any() else 0.0


def measure_image(geometry, unshifted, shifted, labels, lesions):
    """Every measure of a PET in %IA/mL, simulated or real: the masks of nb11 and the local background of the small structures."""
    voxel = geometry.voxel_ml
    grown, kidney = grown_kidney(labels, geometry.spacing_mm)
    spacing = geometry.spacing_mm
    hot = lesions > 0
    measure = dict(
        kidney_grown=float(shifted[grown].sum()) * voxel, kidney_ct=float(shifted[kidney].sum()) * voxel, liver=float(shifted[labels == LABEL["liver"]].mean()),
        spleen=float(shifted[labels == LABEL["spleen"]].mean()), rest=float(unshifted[labels == LABEL["rest"]].mean()),
        blood=float(unshifted[labels == LABEL["blood"]].mean()) if (labels == LABEL["blood"]).any() else np.nan)
    fluid = labels == LABEL["fluid"]
    functional = labels == LABEL["kidney"]
    measure.update(kidney_fluid_pct=float(shifted[fluid].sum()) * voxel, kidney_fluid_ml=float(fluid.sum()) * voxel, kidney_functional_pct=float(shifted[functional].sum()) * voxel, kidney_functional_ml=float(functional.sum()) * voxel)
    if functional.any():
        sinus = renal_sinus(kidney, grown, spacing)
        parenchyma = float(np.median(shifted[functional]))
        measure.update(kidney_sinus_pct=float(shifted[sinus].sum()) * voxel, kidney_sinus_ml=float(sinus.sum()) * voxel, kidney_parenchyma=parenchyma, kidney_sinus_excess_pct=float(np.maximum(shifted[sinus] - parenchyma, 0.0).sum()) * voxel, kidney_shell_excess_pct=float(np.maximum(shifted[grown & ~kidney] - parenchyma, 0.0).sum()) * voxel)
    measure["blood_background"] = ring_median(unshifted, labels == LABEL["blood"], spacing, hot) if np.isfinite(measure["blood"]) else 0.0
    gland_mask = np.isin(labels, list(geometry.gland_labels))
    measure["salivary_conc"] = float(unshifted[gland_mask].mean()) if geometry.has_glands and gland_mask.any() else np.nan
    measure["salivary_pct"] = measure["salivary_conc"] * geometry.salivary_ml
    measure["salivary_background"] = ring_median(unshifted, gland_mask, spacing, hot) if geometry.has_glands and gland_mask.any() else 0.0
    n = len(geometry.lesions_ml)
    measure["lesion_conc"] = ndi.mean(unshifted, lesions, np.arange(1, n + 1)) if n else np.zeros(0)
    background = np.zeros(n)
    if n:
        for j, box in enumerate(ndi.find_objects(lesions, max_label=n)):
            if box is None:
                continue
            pad = [int(np.ceil(RING_MM[1] / s)) + 1 for s in spacing]
            big = tuple(slice(max(b.start - m, 0), min(b.stop + m, size)) for b, m, size in zip(box, pad, lesions.shape))
            background[j] = ring_median(unshifted[big], lesions[big] == j + 1, spacing, hot[big])
    measure["lesion_background"] = background
    return measure


def real_measure(geometry):
    """The same measures on a real phantom. The PET has one position, so the masks of kidney, liver and spleen see the same image as the others."""
    folder = PHANTOMS / geometry.study_id
    labels = np.asanyarray(nib.load(folder / "organs.nii.gz").dataobj)
    lesions = np.asanyarray(nib.load(folder / "lesions.nii.gz").dataobj)
    pet = np.asanyarray(nib.load(folder / "pet_pct_ia_ml.nii.gz").dataobj).astype(np.float32)
    return measure_image(geometry, pet, pet, labels, lesions)


def direct_measure(geometry, truth, p):
    """Measurement without the image step: the model quantities themselves."""
    return dict(kidney_grown=p["kidneys"], kidney_ct=p["kidneys"], liver=p["liver"] / geometry.liver_ml, spleen=p["spleen"] / geometry.spleen_ml, rest=0.0, blood=p["serum_conc"] * (1 - truth.hematocrit),
                salivary_conc=p["salivary"] / ((geometry.salivary_ml or 1.0) * truth.gland_factor) if geometry.has_glands else np.nan,
                salivary_pct=p["salivary"] * geometry.salivary_ml / ((geometry.salivary_ml or 1.0) * truth.gland_factor) if geometry.has_glands else np.nan,
                lesion_conc=p["lesions"] / geometry.lesions_ml if len(geometry.lesions_ml) else np.zeros(0))


def add_noise(measure, rng):
    out = dict(measure)
    for key, sigma in [("kidney_grown", SIGMA["kidneys"]), ("kidney_ct", SIGMA["kidneys"]), ("liver", SIGMA["liver"]), ("spleen", SIGMA["spleen"]), ("blood", SIGMA["blood"])]:
        out[key] = measure[key] * np.exp(rng.normal(0, sigma))
    factor = np.exp(rng.normal(0, SIGMA["salivary"]))
    out["salivary_conc"] = measure["salivary_conc"] * factor
    out["salivary_pct"] = measure["salivary_pct"] * factor
    out["lesion_conc"] = measure["lesion_conc"] * np.exp(rng.normal(0, SIGMA["lesion"], len(measure["lesion_conc"])))
    return out


@dataclass
class FitOptions:
    known_nuisance: bool = False
    image: bool = True
    scanner_offset: float = 0.0
    kidney_mask: str = "grown"
    gland_target: str = "concentration"
    correct_recovery: bool = True
    subtract_shell: bool = True
    fit_topology: str = "reduced"
    fit_nuisance: tuple = ()
    ligand: object = None
    priors: dict = None
    background_recovery: bool = False
    extra_sigma: dict = None
    robust: dict = None
    lesion_offset: object = None
    blood_offset: float = 0.0
    kidney_excess_floor: float = None


@dataclass
class Nuisance:
    f_sal: float
    gfr: float
    hematocrit: float
    amount_nmol: float
    f_lesions: np.ndarray
    l_lesions: np.ndarray
    l_kidney: float
    l_salivary: float
    salivary_ml: float = None


def population_nuisance(geometry):
    n = len(geometry.lesions_ml)
    return Nuisance(PRIORS["f_sal"][0], PRIORS["gfr"][0], PRIORS["hematocrit"][0], PRIORS["amount_nmol"][0], np.full(n, PRIORS["f_lesion"][0]), np.full(n, PRIORS["l_tumor"][0]), PRIORS["l_kidney"][0], PRIORS["l_salivary"][0])


def truth_nuisance(geometry, truth):
    return Nuisance(truth.f_sal, truth.gfr, truth.hematocrit, truth.amount_nmol, truth.f_lesions, truth.l_lesions, truth.l_kidney, truth.l_salivary, (geometry.salivary_ml or 1.0) * truth.gland_factor)


def observations(geometry, measure, options):
    """Decay-corrected %IA or %IA/mL per structure, after scanner, shell and recovery corrections."""
    scale = np.exp(-options.scanner_offset)
    obs = {}
    kidney = measure["kidney_grown"] if options.kidney_mask == "grown" else measure["kidney_ct"]
    if options.kidney_mask == "grown" and options.subtract_shell:
        kidney = kidney - measure["rest"] * geometry.shell_ml
    if options.kidney_excess_floor is not None:
        kidney = kidney - max(measure["kidney_shell_excess_pct"] - options.kidney_excess_floor * measure["kidney_grown"], 0.0)
    obs["kidneys"] = kidney * scale
    rc = geometry.rc if options.correct_recovery else {}
    local = lambda key: max(measure[key], 0.0) * scale if options.background_recovery else 0.0
    corrected = lambda value, factor, background: np.maximum((value - (1 - factor) * background) / factor, 0.1 * value)
    obs["liver"] = measure["liver"] * scale / rc.get("liver", 1.0)
    obs["spleen"] = measure["spleen"] * scale / rc.get("spleen", 1.0)
    if geometry.has_glands and np.isfinite(measure["salivary_conc"]):
        key = "salivary_conc" if options.gland_target == "concentration" else "salivary_pct"
        obs["salivary"] = corrected(measure[key] * scale, rc.get("salivary", 1.0), local("salivary_background")) if key == "salivary_conc" else measure[key] * scale / rc.get("salivary", 1.0)
    obs["blood"] = corrected(measure["blood"] * scale, rc.get("blood", 1.0), local("blood_background")) * np.exp(-options.blood_offset)
    obs["lesions"] = corrected(measure["lesion_conc"] * scale, geometry.rc_lesions, (np.maximum(measure["lesion_background"], 0.0) * scale if options.background_recovery else 0.0)) if options.correct_recovery else measure["lesion_conc"] * scale
    return obs


def _predict(geometry, nuisance, r_den, r_lesions, gland_target, topology="reduced", ligand=None):
    patient = make_patient(geometry, r_den, nuisance.f_sal, nuisance.gfr, nuisance.hematocrit, r_lesions, nuisance.f_lesions, nuisance.l_lesions, nuisance.l_kidney, nuisance.l_salivary, nuisance.salivary_ml)
    injection = P.Injection("Ga68", geometry.activity_mbq, nuisance.amount_nmol)
    net = P.build_network(patient, ligand, topology=topology, lesion_mode="satellite")
    trajectory, _ = P.solve_network(net, injection, geometry.uptake_min + 1)
    y = trajectory(np.array([geometry.uptake_min]))[0]
    fraction = lambda label: sum(y[i] for i in net.groups[label].values()) * 100 / injection.total_nmol
    out = dict(kidneys=fraction("kidneys"), liver=fraction("liver") / patient.liver_ml, spleen=fraction("spleen") / patient.spleen_ml)
    gland = fraction("salivary")
    out["salivary"] = gland / patient.salivary_ml if gland_target == "concentration" else gland
    serum = (y[net.index["arteries"]] + y[net.index["veins"]]) * 100 / injection.total_nmol / ((net.body["Vart"] + net.body["Vven"]) * 1000)
    out["blood"] = serum * (1 - nuisance.hematocrit)
    return out, net, trajectory, injection


def lesion_table(geometry, net, trajectory, injection, f_values, r_values=R_GRID, l_rel=None):
    """Lesion concentration at the scan as a function of density and flow. Per-volume equations do not depend on the volume."""
    l_rel = PRIORS["l_tumor"][0] if l_rel is None else l_rel
    lesions = [P.Lesion(1.0, float(r), float(l_rel), float(f), name=f"g{i}_{j}") for j, f in enumerate(f_values) for i, r in enumerate(r_values)]
    grid = np.array([0.0, geometry.uptake_min])
    out = P.solve_satellites(net, trajectory, injection, lesions, grid)
    values = np.array([sum(c[-1] for c in out[x.name].values()) for x in lesions]) * 100 / injection.total_nmol
    return values.reshape(len(f_values), len(r_values))


NUISANCE_PRIOR = dict(gfr=PRIORS["gfr"], f_sal=PRIORS["f_sal"], amount_nmol=PRIORS["amount_nmol"], hematocrit=(PRIORS["hematocrit"][0], PRIORS["hematocrit"][1] / PRIORS["hematocrit"][0]))


def _student(z, nu):
    """Residual whose square is the Student-t deviance: least squares then minimises the t likelihood."""
    return np.where(np.isfinite(nu), np.sign(z) * np.sqrt((nu + 1) * np.log1p(z**2 / np.where(np.isfinite(nu), nu, 1.0))), z)


def _student_inverse(rho, nu):
    return np.where(np.isfinite(nu), np.sign(rho) * np.sqrt(np.where(np.isfinite(nu), nu, 1.0) * np.expm1(rho**2 / (np.where(np.isfinite(nu), nu, 1.0) + 1))), rho)


def _student_slope(z, nu):
    finite = np.isfinite(nu)
    n = np.where(finite, nu, 1.0)
    small = np.abs(z) < 1e-6
    rho = np.where(small, 1.0, _student(z, nu))
    slope = np.where(small, np.sqrt((n + 1) / n), (n + 1) / n * np.where(small, 1.0, z) / (1 + z**2 / n) / rho)
    return np.where(finite, slope, 1.0)


def fit_organs(geometry, obs, nuisance, r_lesions, start, gland_target, r_keys, topology="reduced", fit_nuisance=(), ligand=None, priors=None, extra_sigma=None, robust=None):
    table = {**PRIORS, **(priors or {})}
    extra = [k for k in fit_nuisance]
    nuisance_prior = {**NUISANCE_PRIOR, **{k: v for k, v in (priors or {}).items() if k in NUISANCE_PRIOR}}
    medians = np.array([table[k][0] for k in r_keys] + [nuisance_prior[k][0] for k in extra])
    sds = np.array([table[k][1] for k in r_keys] + [nuisance_prior[k][1] for k in extra])
    data_keys = list(r_keys) + (["blood"] if extra else [])
    robust = robust or {}
    sigma = np.array([robust[k][1] if k in robust else np.hypot(SIGMA[k], (extra_sigma or {}).get(k, 0.0)) for k in data_keys])
    nu = np.array([robust[k][0] if k in robust else np.inf for k in data_keys])
    data = np.array([np.log(max(obs[k], 1e-9)) for k in data_keys])
    n_r = len(r_keys)

    def residual(x):
        r_den = dict(zip(r_keys, np.exp(x[:n_r])))
        r_den.setdefault("salivary", table["salivary"][0])
        current = replace(nuisance, **{k: float(np.exp(v)) for k, v in zip(extra, x[n_r:])})
        prediction, *_ = _predict(geometry, current, r_den, r_lesions, gland_target, topology, ligand)
        model = np.array([np.log(prediction[k]) for k in data_keys])
        return np.r_[_student((data - model) / sigma, nu), (x - np.log(medians)) / sds]

    lower, upper = np.full(len(medians), -np.inf), np.full(len(medians), np.inf)
    if "hematocrit" in extra:
        i = n_r + extra.index("hematocrit")
        lower[i], upper[i] = np.log(HEMATOCRIT_BOUNDS)
    x0 = np.clip(np.log(start), lower + 1e-6, upper - 1e-6)
    solution = least_squares(residual, x0, bounds=(lower, upper), method="trf", diff_step=0.03, xtol=1e-3, ftol=1e-4, max_nfev=30)
    jac = solution.jac
    covariance = np.linalg.pinv(jac.T @ jac)
    z = _student_inverse(solution.fun[: len(data_keys)], nu)
    data_part = jac[: len(data_keys)] * -sigma[:, None] / _student_slope(z, nu)[:, None]
    fitted = replace(nuisance, **{k: float(np.exp(v)) for k, v in zip(extra, solution.x[n_r:])})
    linear = dict(keys=data_keys, extra=extra, jacobian=data_part, misfit=z * sigma, x=solution.x.copy(), sigma=sigma, nu=nu, n_r=n_r, r_keys=list(r_keys))
    return np.exp(solution.x[:n_r]), np.sqrt(np.diag(covariance))[:n_r], data_part, solution.nfev, solution.fun[: n_r], fitted, solution.x, linear


def posterior_lesions(geometry, obs, table, f_values, f_weights, use_prior=True):
    """Marginal posterior of log R per lesion on R_GRID, flow integrated over its prior. Returns median, 16 and 84 and 2.5 and 97.5 percentiles."""
    n = len(geometry.lesions_ml)
    if not n:
        return dict(median=np.zeros(0), q16=np.zeros(0), q84=np.zeros(0), q025=np.zeros(0), q975=np.zeros(0))
    log_table = np.log(table)
    data = np.log(np.maximum(obs["lesions"], 1e-12))
    log_like = -0.5 * ((data[:, None, None] - log_table[None]) / SIGMA["lesion"]) ** 2
    prior = -0.5 * ((np.log(R_GRID) - np.log(TUMOUR_MEDIAN)) / TUMOUR_SD) ** 2 if use_prior else 0 * R_GRID
    weights = np.exp(log_like) * f_weights[None, :, None]
    posterior = weights.sum(axis=1) * np.exp(prior)[None] * np.gradient(np.log(R_GRID))[None]
    posterior /= np.maximum(posterior.sum(axis=1, keepdims=True), 1e-300)
    cumulative = np.cumsum(posterior, axis=1)
    quantile = lambda q: np.array([np.exp(np.interp(q, c, np.log(R_GRID))) for c in cumulative])
    return dict(median=quantile(0.5), q16=quantile(0.16), q84=quantile(0.84), q025=quantile(0.025), q975=quantile(0.975))


def flow_grid():
    median, sd = PRIORS["f_lesion"]
    values = median * np.exp(sd * F_GRID_Z)
    weights = np.exp(-0.5 * F_GRID_Z**2)
    return values, weights / weights.sum()


def fit_study(geometry, measure, nuisance, options, max_loops=3):
    obs = observations(geometry, measure, options)
    r_keys = [k for k in ("kidneys", "liver", "spleen", "salivary") if k in obs]
    fit_nuisance = options.fit_nuisance if np.isfinite(obs["blood"]) else ()
    start = np.array([PRIORS[k][0] for k in r_keys] + [NUISANCE_PRIOR[k][0] for k in fit_nuisance])
    r_lesions = np.full(len(geometry.lesions_ml), TUMOUR_MEDIAN)
    f_values, f_weights = flow_grid()
    nfev = 0
    current = nuisance
    profile = None
    for _ in range(max_loops):
        x, sd, sens, n, residual, current, raw, _ = fit_organs(geometry, obs, current, r_lesions, start, options.gland_target, r_keys, options.fit_topology, fit_nuisance)
        nfev += n
        if not len(geometry.lesions_ml):
            lesion_out = posterior_lesions(geometry, obs, None, None, None)
            break
        r_den = dict(zip(r_keys, x))
        r_den.setdefault("salivary", PRIORS["salivary"][0])
        _, net, trajectory, injection = _predict(geometry, current, r_den, r_lesions, options.gland_target, options.fit_topology)
        if options.known_nuisance:
            table = known_flow_table(geometry, net, trajectory, injection, current.f_lesions)
            lesion_out = known_flow_posterior(geometry, obs, table)
            profile = profile_known_flow(geometry, obs, table)
        else:
            table = lesion_table(geometry, net, trajectory, injection, f_values)
            lesion_out = posterior_lesions(geometry, obs, table, f_values, f_weights)
            profile = profile_lesions(geometry, obs, table, f_values, f_weights)
        new = lesion_out["median"]
        change = np.abs(np.log(new / r_lesions)).max()
        r_lesions = new
        start = np.exp(raw)
        if change < 0.05:
            break
    return dict(r_keys=r_keys, r=x, sd=sd, sensitivity=sens, nfev=nfev, lesions=lesion_out, profile=profile, obs=obs, residual=residual, gfr=current.gfr, nuisance=current)


def known_flow_table(geometry, net, trajectory, injection, f_lesions):
    lesions = [P.Lesion(1.0, float(r), float(PRIORS["l_tumor"][0]), float(f), name=f"k{j}_{i}") for j, f in enumerate(f_lesions) for i, r in enumerate(R_GRID)]
    out = P.solve_satellites(net, trajectory, injection, lesions, np.array([0.0, geometry.uptake_min]))
    return np.array([sum(c[-1] for c in out[x.name].values()) for x in lesions]).reshape(len(f_lesions), len(R_GRID)) * 100 / injection.total_nmol


def known_flow_posterior(geometry, obs, table):
    """Posterior with each lesion's flow known: one curve of R per lesion, for the arm with known nuisance."""
    data = np.log(np.maximum(obs["lesions"], 1e-12))
    log_like = -0.5 * ((data[:, None] - np.log(table)) / SIGMA["lesion"]) ** 2
    prior = -0.5 * ((np.log(R_GRID) - np.log(TUMOUR_MEDIAN)) / TUMOUR_SD) ** 2
    posterior = np.exp(log_like + prior) * np.gradient(np.log(R_GRID))[None]
    posterior /= np.maximum(posterior.sum(axis=1, keepdims=True), 1e-300)
    cumulative = np.cumsum(posterior, axis=1)
    quantile = lambda q: np.array([np.exp(np.interp(q, c, np.log(R_GRID))) for c in cumulative])
    return dict(median=quantile(0.5), q16=quantile(0.16), q84=quantile(0.84), q025=quantile(0.025), q975=quantile(0.975))


def _interval(nll, threshold):
    best = nll.min(axis=1, keepdims=True)
    inside = nll - best <= threshold
    lower = np.array([R_GRID[np.argmax(row)] for row in inside])
    upper = np.array([R_GRID[len(R_GRID) - 1 - np.argmax(row[::-1])] for row in inside])
    return dict(lower=lower, upper=upper, open_top=inside[:, -1], open_bottom=inside[:, 0], best_nll=best[:, 0], r_best=R_GRID[nll.argmin(axis=1)])


def profile_known_flow(geometry, obs, table, threshold=1.92):
    data = np.log(np.maximum(obs["lesions"], 1e-12))
    return _interval(0.5 * ((data[:, None] - np.log(table)) / SIGMA["lesion"]) ** 2, threshold)


def profile_lesions(geometry, obs, table, f_values, f_weights, threshold=1.92):
    """Profile likelihood of log R per lesion, flow profiled with its prior as penalty, no prior on R. 1.92 is the 95% limit."""
    data = np.log(np.maximum(obs["lesions"], 1e-12))
    penalty = -np.log(f_weights / f_weights.max())
    nll = 0.5 * ((data[:, None, None] - np.log(table)[None]) / SIGMA["lesion"]) ** 2 + penalty[None, :, None]
    return _interval(nll.min(axis=1), threshold)


def predicted_rest(geometry):
    """Regressors of the rest-of-body concentration, used for the scanner covariate. No PBPK model involved."""
    return dict(log_weight=np.log(geometry.weight_kg), uptake_min=geometry.uptake_min, log_tlv=np.log1p(geometry.lesions_ml.sum()))


def stage_measure(args):
    """One simulated PET of one phantom: geometry, truth, noise-free image measurement and direct measurement."""
    study_id, draw, seed = args
    rng = np.random.default_rng([seed, draw, hash_id(study_id)])
    geometry = load_geometry(study_id)
    truth = draw_truth(geometry, rng)
    result = simulate_truth(geometry, truth)
    measure, percent = paint_and_measure(geometry, truth, result)
    return dict(study_id=study_id, draw=draw, seed=seed, geometry=geometry, truth=truth, measure=measure, direct=direct_measure(geometry, truth, percent), percent=percent)


def hash_id(text):
    return zlib.crc32(text.encode())


ARMS = {
    "known nuisance, no image": dict(options=FitOptions(known_nuisance=True, image=False, correct_recovery=False, subtract_shell=False), known=True),
    "known nuisance, image": dict(options=FitOptions(known_nuisance=True), known=True, offset=True),
    "realistic": dict(options=FitOptions(), known=False, offset=True),
    "no scanner covariate": dict(options=FitOptions(), known=False),
    "kidney in the CT mask": dict(options=FitOptions(kidney_mask="ct"), known=False, offset=True),
    "grown mask, shell not subtracted": dict(options=FitOptions(subtract_shell=False), known=False, offset=True),
    "glands, %IA with published volume": dict(options=FitOptions(gland_target="percent"), known=False, offset=True, published_gland=True),
    "no recovery correction": dict(options=FitOptions(correct_recovery=False), known=False, offset=True),
    "full topology in the fit": dict(options=FitOptions(fit_topology="full"), known=False, offset=True),
    "known nuisance, no image, full topology": dict(options=FitOptions(known_nuisance=True, image=False, correct_recovery=False, subtract_shell=False, fit_topology="full"), known=True),
    "blood pool, GFR fitted": dict(options=FitOptions(fit_nuisance=("gfr",)), known=False, offset=True),
    "blood pool, nuisance marginalised": dict(options=FitOptions(fit_nuisance=("gfr", "f_sal", "hematocrit", "amount_nmol")), known=False, offset=True),
}


def therapy_tiac(geometry, r_den, r_lesions, f_sal, gfr, hematocrit, f_lesions, l_lesions, l_kidney, l_salivary, salivary_ml=None, ligand=None, topology="reduced"):
    patient = make_patient(geometry, r_den, f_sal, gfr, hematocrit, r_lesions, f_lesions, l_lesions, l_kidney, l_salivary, salivary_ml)
    result = P.simulate(patient, P.Injection("Lu177", 7400, 100), ligand=ligand, lesion_mode="satellite", topology=topology)
    out = {k: result.tiac_h(k, include_tail=True) for k in ORGANS}
    out["lesions"] = np.array([result.tiac_h(f"lesion_{j}", include_tail=True) for j in range(len(geometry.lesions_ml))])
    return out


def stage_fit(args):
    """All arms on one simulated PET. Returns one row per organ and arm, one row per lesion and arm, and the therapy errors."""
    record, offsets = args
    geometry, truth = record["geometry"], record["truth"]
    organ_rows, lesion_rows, therapy_rows = [], [], []
    rng_seed = [record["seed"], record["draw"], hash_id(record["study_id"]), 7]
    pop = population_nuisance(geometry)
    true_therapy = None
    for arm, spec in ARMS.items():
        rng = np.random.default_rng(rng_seed)
        options = spec["options"]
        measure = add_noise(record["measure"] if options.image else record["direct"], rng)
        if spec.get("offset"):
            options = replace(options, scanner_offset=offsets[geometry.scanner])
        if spec.get("published_gland"):
            nuisance = replace(pop, salivary_ml=(geometry.salivary_ml or 1.0) * PRIORS["gland_volume_factor"][0])
        else:
            nuisance = truth_nuisance(geometry, truth) if spec["known"] else pop
        fit = fit_study(geometry, measure, nuisance, options)
        for key, value, sd, residual in zip(fit["r_keys"], fit["r"], fit["sd"], fit["residual"]):
            index = fit["r_keys"].index(key)
            organ_rows.append(dict(study_id=geometry.study_id, draw=record["draw"], arm=arm, structure=key, truth=truth.r_den[key], estimate=value, sd_log=sd, prior_sd=PRIORS[key][1], sensitivity=fit["sensitivity"][index, index], scanner=geometry.scanner, tlv_ml=geometry.lesions_ml.sum(), residual=residual, nfev=fit["nfev"], gfr_truth=truth.gfr, gfr_fit=fit["gfr"]))
        lesions = fit["lesions"]
        profile = fit["profile"]
        for j in range(len(geometry.lesions_ml)):
            row = dict(study_id=geometry.study_id, draw=record["draw"], arm=arm, lesion=j, volume_ml=geometry.lesions_ml[j], rc=geometry.rc_lesions[j], truth=truth.r_lesions[j], flow=truth.f_lesions[j], estimate=lesions["median"][j], q16=lesions["q16"][j], q84=lesions["q84"][j], q025=lesions["q025"][j], q975=lesions["q975"][j], tlv_ml=geometry.lesions_ml.sum(), scanner=geometry.scanner)
            if profile is not None:
                row.update(lower=profile["lower"][j], upper=profile["upper"][j], open_top=bool(profile["open_top"][j]), open_bottom=bool(profile["open_bottom"][j]), r_best=profile["r_best"][j])
            lesion_rows.append(row)
        n = len(geometry.lesions_ml)
        flows = np.full(n, PRIORS["f_lesion"][0])
        releases = np.full(n, PRIORS["l_tumor"][0])
        r_den = dict(zip(fit["r_keys"], fit["r"]))
        r_den.setdefault("salivary", PRIORS["salivary"][0])
        cases = {}
        if arm == "realistic":
            median_den = {k: PRIORS[k][0] for k in ORGANS}
            gland_true = (geometry.salivary_ml or 1.0) * truth.gland_factor
            true_therapy = therapy_tiac(geometry, truth.r_den, truth.r_lesions, truth.f_sal, truth.gfr, truth.hematocrit, truth.f_lesions, truth.l_lesions, truth.l_kidney, truth.l_salivary, gland_true)
            cases["PET fit, population release"] = therapy_tiac(geometry, r_den, lesions["median"], PRIORS["f_sal"][0], PRIORS["gfr"][0], 0.4, flows, releases, PRIORS["l_kidney"][0], PRIORS["l_salivary"][0])
            cases["PET fit, true release"] = therapy_tiac(geometry, r_den, lesions["median"], PRIORS["f_sal"][0], PRIORS["gfr"][0], 0.4, flows, truth.l_lesions, truth.l_kidney, truth.l_salivary)
            cases["no PET, population medians"] = therapy_tiac(geometry, median_den, np.full(n, TUMOUR_MEDIAN), PRIORS["f_sal"][0], PRIORS["gfr"][0], 0.4, flows, releases, PRIORS["l_kidney"][0], PRIORS["l_salivary"][0])
        if arm == "blood pool, nuisance marginalised":
            fitted = fit["nuisance"]
            cases["PET fit, nuisance marginalised, population release"] = therapy_tiac(geometry, r_den, lesions["median"], fitted.f_sal, fitted.gfr, fitted.hematocrit, flows, releases, PRIORS["l_kidney"][0], PRIORS["l_salivary"][0])
        for case, values in cases.items():
            for key in ORGANS:
                ratio = values[key] / true_therapy[key]
                if key == "salivary":
                    ratio *= truth.gland_factor
                therapy_rows.append(dict(study_id=geometry.study_id, draw=record["draw"], case=case, structure=key, ratio=ratio))
            for j in range(n):
                therapy_rows.append(dict(study_id=geometry.study_id, draw=record["draw"], case=case, structure="lesions", ratio=values["lesions"][j] / true_therapy["lesions"][j]))
    return organ_rows, lesion_rows, therapy_rows


LAMBDA_GRID = np.geomspace(1e-5, 3e-2, 19)


def lambda_profile(record, seed=11):
    """What the 1 h PET says about the release rate: for each fixed release rate, the density that best fits the noise-free PET.

    Organs change only their own release rate, other parameters at truth. Returns one row per structure and release rate.
    """
    from scipy.optimize import minimize_scalar
    geometry, truth = record["geometry"], record["truth"]
    nuisance = truth_nuisance(geometry, truth)
    rows = []
    base_den = dict(truth.r_den)
    injection = P.Injection("Ga68", geometry.activity_mbq, truth.amount_nmol)

    def model(structure, density, l_rel):
        den = {**base_den, structure: float(density)}
        patient = make_patient(geometry, den, nuisance.f_sal, nuisance.gfr, nuisance.hematocrit, truth.r_lesions, truth.f_lesions, truth.l_lesions, truth.l_kidney, truth.l_salivary, nuisance.salivary_ml)
        patient.lambda_rel[structure] = float(l_rel)
        net = P.build_network(patient, lesion_mode="satellite")
        trajectory, _ = P.solve_network(net, injection, geometry.uptake_min + 1, rtol=1e-7, atol_factor=1e-10)
        y = trajectory(np.array([geometry.uptake_min]))[0]
        return sum(y[i] for i in net.groups[structure].values()) * 100 / injection.total_nmol

    def therapy(structure, density, l_rel):
        den = {**base_den, structure: float(density)}
        patient = make_patient(geometry, den, nuisance.f_sal, nuisance.gfr, nuisance.hematocrit, truth.r_lesions, truth.f_lesions, truth.l_lesions, truth.l_kidney, truth.l_salivary, nuisance.salivary_ml)
        patient.lambda_rel[structure] = float(l_rel)
        result = P.simulate(patient, P.Injection("Lu177", 7400, 100), lesion_mode="satellite")
        return result.tiac_h(structure, include_tail=True), result.t_eff_h(structure)

    for structure in ORGANS:
        if structure == "salivary" and not geometry.has_glands:
            continue
        true_rel = {"kidneys": truth.l_kidney, "salivary": truth.l_salivary}.get(structure, truth.l_kidney)
        target = model(structure, base_den[structure], true_rel)
        for l_rel in LAMBDA_GRID:
            objective = lambda log_r: 0.5 * ((np.log(model(structure, np.exp(log_r), l_rel)) - np.log(target)) / SIGMA[structure]) ** 2
            best = minimize_scalar(objective, bounds=(np.log(base_den[structure]) - 4, np.log(base_den[structure]) + 4), method="bounded", options=dict(xatol=0.02, maxiter=20))
            tiac, t_eff = therapy(structure, np.exp(best.x), l_rel)
            rows.append(dict(study_id=geometry.study_id, draw=record["draw"], structure=structure, l_rel=l_rel, l_true=true_rel, nll_refit=best.fun, density_refit=float(np.exp(best.x)), density_true=base_den[structure],
                             pet_change_fixed_density=float(model(structure, base_den[structure], l_rel) / target), tiac_h=tiac, t_eff_h=t_eff))
    if len(geometry.lesions_ml):
        net = P.build_network(truth_patient(geometry, truth), lesion_mode="satellite")
        trajectory, _ = P.solve_network(net, injection, geometry.uptake_min + 1, rtol=1e-7, atol_factor=1e-10)
        j = int(np.argmax(geometry.lesions_ml))
        flow = np.array([truth.f_lesions[j]])
        target = lesion_table(geometry, net, trajectory, injection, flow, np.array([truth.r_lesions[j]]), truth.l_lesions[j])[0, 0]
        for l_rel in LAMBDA_GRID:
            curve = lesion_table(geometry, net, trajectory, injection, flow, R_GRID, l_rel)[0]
            nll = 0.5 * ((np.log(curve) - np.log(target)) / SIGMA["lesion"]) ** 2
            best = int(np.argmin(nll))
            lesions = np.array(truth.r_lesions, float)
            releases = np.array(truth.l_lesions, float)
            lesions[j], releases[j] = R_GRID[best], l_rel
            patient = make_patient(geometry, base_den, nuisance.f_sal, nuisance.gfr, nuisance.hematocrit, lesions, truth.f_lesions, releases, truth.l_kidney, truth.l_salivary, nuisance.salivary_ml)
            result = P.simulate(patient, P.Injection("Lu177", 7400, 100), lesion_mode="satellite")
            rows.append(dict(study_id=geometry.study_id, draw=record["draw"], structure="lesion", l_rel=l_rel, l_true=truth.l_lesions[j], nll_refit=float(nll[best]), density_refit=float(R_GRID[best]), density_true=float(truth.r_lesions[j]),
                             pet_change_fixed_density=float(np.interp(np.log(truth.r_lesions[j]), np.log(R_GRID), curve) / target), tiac_h=result.tiac_h(f"lesion_{j}", include_tail=True), t_eff_h=result.t_eff_h(f"lesion_{j}")))
    return rows


def safe_measure(args):
    try:
        return stage_measure(args)
    except Exception as error:
        return dict(study_id=args[0], draw=args[1], error=repr(error))


def safe_fit(args):
    try:
        return stage_fit(args)
    except Exception as error:
        return dict(study_id=args[0]["study_id"], draw=args[0]["draw"], error=repr(error))


LESION_R = np.geomspace(1.0, 3.0e4, 41)
LESION_F = np.geomspace(0.002, 20.0, 25)
LOG_R = np.log(LESION_R)
LOG_F = np.log(LESION_F)
DELTA_R = float(LOG_R[1] - LOG_R[0])
DELTA_F = float(LOG_F[1] - LOG_F[0])
PATIENT_Z = np.linspace(-3.0, 3.0, 21)
PROFILE_Z = np.linspace(-3.0, 3.0, 61)
MAIN_NUISANCE = ("gfr", "f_sal", "hematocrit", "amount_nmol")
IMAGE_SD = (0.27, 0.11, 0.07)
ORGAN_GRID_LOG = np.linspace(-4.0, 4.0, 11)
COMMON_SD = 0.15
COMMON_Z, COMMON_W = np.polynomial.hermite_e.hermegauss(5)
COMMON_W = COMMON_W / COMMON_W.sum()
FIVE_PATIENTS = 5
LOG_SD_ERROR = 0.35


def _normal(x, mean, sd):
    return np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def lesion_sigma(rc):
    """Per-lesion log error: noise plus the image step (spill-in, wrong PSF), larger at low recovery."""
    image = np.select([rc < 0.3, rc < 0.5], [IMAGE_SD[0], IMAGE_SD[1]], IMAGE_SD[2])
    return np.hypot(SIGMA["lesion"], image)


def lesion_likelihood(log_data, table, rc):
    """L[common, lesion, flow, density]. The common node shifts all lesions of a patient together. Flat above a lower bound where the table saturates: this is the censoring."""
    sigma = lesion_sigma(rc)[None, :, None, None]
    shift = (COMMON_SD * COMMON_Z)[:, None, None, None]
    return np.exp(-0.5 * ((log_data[None, :, None, None] - shift - np.log(table)[None, None]) / sigma) ** 2).astype(np.float32)


def prior_low(hyper):
    return _normal(LOG_R, hyper.mu_low, hyper.tau_low) * DELTA_R


def flow_weights(hyper):
    weights = _normal(LOG_F, hyper.mu_flow, hyper.tau_flow) * DELTA_F
    weights = np.where(LESION_F <= hyper.flow_cap, weights, 0.0)
    return weights / weights.sum()


def patient_integrals(likelihood, hyper, patient_z=PATIENT_Z):
    """Log p(common, u) up to a constant, A[common, lesion, density], the density prior per u and the lesion marginals."""
    per_density = np.einsum("djfr,f->djr", likelihood, flow_weights(hyper))
    effects = hyper.tau_between * patient_z
    prior_u = _normal(patient_z, 0.0, 1.0)
    prior_u = prior_u / prior_u.sum()
    prior_r = _normal(LOG_R[None], (hyper.mu + effects)[:, None], hyper.tau_within) * DELTA_R
    if hyper.pi_low > 0:
        prior_r = (1 - hyper.pi_low) * prior_r + hyper.pi_low * prior_low(hyper)[None]
    marginal = np.maximum(per_density @ prior_r.T, 1e-300)
    log_joint = np.log(COMMON_W)[:, None] + np.log(prior_u)[None] + np.log(marginal).sum(axis=1)
    return log_joint, per_density, prior_r, marginal


def patient_log_marginal(likelihood, hyper, patient_z=PATIENT_Z):
    if not likelihood.shape[1]:
        return 0.0
    log_joint, *_ = patient_integrals(likelihood, hyper, patient_z)
    peak = log_joint.max()
    return float(peak + np.log(np.exp(log_joint - peak).sum()))


def _quantiles(probabilities, axis_values, levels=(0.025, 0.16, 0.5, 0.84, 0.975)):
    cumulative = np.cumsum(probabilities, axis=1)
    return [np.array([np.exp(np.interp(q, c, axis_values)) for c in cumulative]) for q in levels]


def lesion_posterior(likelihood, hyper):
    """Joint posterior of density and flow per lesion. Pooled: with the patient effect and the common error. Alone: each lesion with its total prior only."""
    if not likelihood.shape[1]:
        return None
    log_joint, per_density, prior_r, marginal = patient_integrals(likelihood, hyper)
    p_joint = np.exp(log_joint - log_joint.max())
    p_joint = p_joint / p_joint.sum()
    pooled = np.einsum("du,djr,ur,dju->jr", p_joint, per_density, prior_r, 1.0 / marginal)
    weights = flow_weights(hyper)
    flow_part = np.einsum("djfr,ur->djfu", likelihood, prior_r) * weights[None, None, :, None] / marginal[:, :, None, :]
    pooled_flow = np.einsum("du,djfu->jf", p_joint, flow_part)
    total_sd = float(np.hypot(hyper.tau_between, hyper.tau_within))
    prior_total = (1 - hyper.pi_low) * _normal(LOG_R, hyper.mu, total_sd) * DELTA_R + hyper.pi_low * prior_low(hyper)
    prior_total = prior_total / prior_total.sum()
    alone = np.einsum("d,djr->jr", COMMON_W, per_density) * prior_total[None]
    alone = alone / np.maximum(alone.sum(axis=1, keepdims=True), 1e-300)
    pooled = pooled / np.maximum(pooled.sum(axis=1, keepdims=True), 1e-300)
    pooled_flow = pooled_flow / np.maximum(pooled_flow.sum(axis=1, keepdims=True), 1e-300)
    return dict(pooled=pooled, alone=alone, flow=pooled_flow, prior_total=prior_total, p_joint=p_joint, r_q=_quantiles(pooled, LOG_R), r_alone_q=_quantiles(alone, LOG_R), f_q=_quantiles(pooled_flow, LOG_F))


def draw_lesions(likelihood, hyper, rng):
    """One draw of (density, flow) per lesion from the joint posterior, sharing one patient effect and one common error: the draw for the virtual cohort."""
    log_joint, _, prior_r, _ = patient_integrals(likelihood, hyper)
    p_joint = np.exp(log_joint - log_joint.max())
    d, u = np.unravel_index(rng.choice(p_joint.size, p=(p_joint / p_joint.sum()).ravel()), p_joint.shape)
    joint = likelihood[d] * flow_weights(hyper)[None, :, None] * prior_r[u][None, None, :]
    joint = joint.reshape(likelihood.shape[1], -1).astype(np.float64)
    joint = joint / np.maximum(joint.sum(axis=1, keepdims=True), 1e-300)
    pick = np.array([rng.choice(joint.shape[1], p=row / row.sum()) for row in joint])
    i_f, i_r = np.unravel_index(pick, (len(LESION_F), len(LESION_R)))
    jitter = rng.uniform(-0.5, 0.5, (2, len(pick)))
    return np.exp(LOG_R[i_r] + jitter[0] * DELTA_R), np.exp(LOG_F[i_f] + jitter[1] * DELTA_F)


def lesion_profile(log_data, table, hyper, rc, threshold=1.92):
    """Profile over flow with the per-lesion and common errors in quadrature: the bound one lesion alone can claim."""
    weights = flow_weights(hyper)
    penalty = -np.log(np.maximum(weights / weights.max(), 1e-300))
    sigma = np.hypot(lesion_sigma(rc), COMMON_SD)[:, None, None]
    nll = (0.5 * ((log_data[:, None, None] - np.log(table)[None]) / sigma) ** 2 + penalty[None, :, None]).min(axis=1)
    best = nll.min(axis=1, keepdims=True)
    inside = nll - best <= threshold
    return dict(lower=LESION_R[inside.argmax(axis=1)], upper=LESION_R[len(LESION_R) - 1 - inside[:, ::-1].argmax(axis=1)], open_top=inside[:, -1], open_bottom=inside[:, 0])


def lesion_classes(profile):
    return np.select([profile["open_top"] & profile["open_bottom"], profile["open_top"], profile["open_bottom"]], ["unbounded", "lower bound only", "upper bound only"], "two-sided")


def fit_joint(geometry, measure, options=None, hyper=None, max_loops=3, base_nuisance=None):
    """Main fit: organs with the nuisance marginalised, then density and flow of every lesion jointly, through the plasma curve."""
    options = options or FitOptions(fit_nuisance=MAIN_NUISANCE)
    hyper = hyper or LesionHyper()
    table_prior = {**PRIORS, **(options.priors or {})}
    obs = observations(geometry, measure, options)
    r_keys = [k for k in ("kidneys", "liver", "spleen", "salivary") if k in obs]
    fit_nuisance = options.fit_nuisance if np.isfinite(obs["blood"]) else ()
    nuisance_prior = {**NUISANCE_PRIOR, **{k: v for k, v in (options.priors or {}).items() if k in NUISANCE_PRIOR}}
    start = np.array([table_prior[k][0] for k in r_keys] + [nuisance_prior[k][0] for k in fit_nuisance])
    n = len(geometry.lesions_ml)
    r_lesions = np.full(n, float(np.exp(hyper.mu)))
    current = base_nuisance or population_nuisance(geometry)
    current = replace(current, f_lesions=np.full(n, float(np.exp(hyper.mu_flow))))
    posterior, profile, likelihood = None, None, np.zeros((len(COMMON_Z), 0, len(LESION_F), len(LESION_R)), np.float32)
    log_data = np.zeros(0)
    for _ in range(max_loops):
        x, sd, sens, _, residual, current, raw, linear = fit_organs(geometry, obs, current, r_lesions, start, options.gland_target, r_keys, options.fit_topology, fit_nuisance, options.ligand, options.priors, options.extra_sigma, options.robust)
        start = np.exp(raw)
        if not n:
            break
        r_den = dict(zip(r_keys, x))
        r_den.setdefault("salivary", table_prior["salivary"][0])
        _, net, trajectory, injection = _predict(geometry, current, r_den, r_lesions, options.gland_target, options.fit_topology, options.ligand)
        table = lesion_table(geometry, net, trajectory, injection, LESION_F, LESION_R).astype(np.float64)
        log_data = np.log(np.maximum(obs["lesions"], 1e-12)) - (options.lesion_offset(geometry.rc_lesions, geometry.lesions_ml) if options.lesion_offset else 0.0)
        likelihood = lesion_likelihood(log_data, table, geometry.rc_lesions)
        posterior = lesion_posterior(likelihood, hyper)
        profile = lesion_profile(log_data, table, hyper, geometry.rc_lesions)
        new_r, new_f = posterior["r_q"][2], posterior["f_q"][2]
        change = np.abs(np.log(new_r / r_lesions)).max()
        r_lesions = new_r
        current = replace(current, f_lesions=new_f)
        if change < 0.05:
            break
    r_den = dict(zip(r_keys, x))
    r_den.setdefault("salivary", table_prior["salivary"][0])
    return dict(r_keys=r_keys, r=x, sd=sd, r_den=r_den, nuisance=current, posterior=posterior, profile=profile, likelihood=likelihood, linear=linear, obs=obs, log_data=log_data, table=table if n else None)


def main_arm(record, options=None, hyper=None):
    rng = np.random.default_rng([record["seed"], record["draw"], hash_id(record["study_id"]), 7])
    return fit_joint(record["geometry"], add_noise(record["measure"], rng), options, hyper)


def _lesion_rows(record, fit, hyper):
    geometry, truth = record["geometry"], record["truth"]
    n = len(geometry.lesions_ml)
    if not n:
        return []
    post, profile = fit["posterior"], fit["profile"]
    classes = lesion_classes(profile)
    total_sd = float(np.hypot(hyper.tau_between, hyper.tau_within))
    prior = post["prior_total"]
    rows = []
    for j in range(n):
        alone, pooled = post["alone"][j], post["pooled"][j]
        keep = alone > 0
        kl = float(np.sum(alone[keep] * np.log(alone[keep] / prior[keep])))
        kl_pooled = float(np.sum(pooled[pooled > 0] * np.log(pooled[pooled > 0] / prior[pooled > 0])))
        at_truth = lambda p: float(np.interp(np.log(truth.r_lesions[j]), LOG_R, p / DELTA_R))
        lower = profile["lower"][j]
        cut = float(prior[LESION_R < lower].sum())
        truncated = np.where(LESION_R >= lower, prior, 0.0)
        truncated = truncated / truncated.sum()
        truncated_median = float(np.exp(np.interp(0.5, np.cumsum(truncated), LOG_R)))
        sd_alone = float(np.sqrt(np.sum(alone * LOG_R**2) - np.sum(alone * LOG_R) ** 2))
        rows.append(dict(study_id=geometry.study_id, draw=record["draw"], lesion=j, volume_ml=geometry.lesions_ml[j], rc=geometry.rc_lesions[j], tlv_ml=geometry.lesions_ml.sum(), scanner=geometry.scanner, truth_r=truth.r_lesions[j], truth_f=truth.f_lesions[j],
                         r_median=post["r_q"][2][j], r_q025=post["r_q"][0][j], r_q16=post["r_q"][1][j], r_q84=post["r_q"][3][j], r_q975=post["r_q"][4][j], r_alone_median=post["r_alone_q"][2][j], r_alone_q025=post["r_alone_q"][0][j], r_alone_q975=post["r_alone_q"][4][j],
                         f_median=post["f_q"][2][j], f_q025=post["f_q"][0][j], f_q16=post["f_q"][1][j], f_q84=post["f_q"][3][j], f_q975=post["f_q"][4][j], kl_alone=kl, kl_pooled=kl_pooled, prior_mass_cut=cut, truncated_median=truncated_median, sd_alone=sd_alone, sd_prior=total_sd,
                         score_prior=np.log(max(at_truth(prior), 1e-300)), score_alone=np.log(max(at_truth(alone), 1e-300)), score_pooled=np.log(max(at_truth(pooled), 1e-300)), lower=lower, upper=profile["upper"][j], lesion_class=classes[j]))
    return rows


def _organ_rows(record, fit, tag):
    geometry, truth = record["geometry"], record["truth"]
    rows = []
    for key, value, sd in zip(fit["r_keys"], fit["r"], fit["sd"]):
        rows.append(dict(study_id=geometry.study_id, draw=record["draw"], tag=tag, structure=key, truth=truth.r_den[key], estimate=value, sd_log=sd, scanner=geometry.scanner, tlv_ml=geometry.lesions_ml.sum()))
    return rows


def _therapy_rows(record, values, reference, tag, case):
    geometry, truth = record["geometry"], record["truth"]
    rows = []
    for key in ORGANS:
        ratio = values[key] / reference[key]
        if key == "salivary":
            ratio *= truth.gland_factor
        rows.append(dict(study_id=geometry.study_id, draw=record["draw"], tag=tag, case=case, structure=key, ratio=ratio))
    rows += [dict(study_id=geometry.study_id, draw=record["draw"], tag=tag, case=case, structure="lesions", ratio=a / b) for a, b in zip(values["lesions"], reference["lesions"])]
    return rows


def _truth_therapy(record, l_kidney=None, l_salivary=None, l_lesions=None, ligand=None, **override):
    geometry, truth = record["geometry"], record["truth"]
    args = dict(r_den=truth.r_den, r_lesions=truth.r_lesions, f_sal=truth.f_sal, gfr=truth.gfr, hematocrit=truth.hematocrit, f_lesions=truth.f_lesions, l_lesions=truth.l_lesions if l_lesions is None else l_lesions,
                l_kidney=truth.l_kidney if l_kidney is None else l_kidney, l_salivary=truth.l_salivary if l_salivary is None else l_salivary, salivary_ml=(geometry.salivary_ml or 1.0) * truth.gland_factor, ligand=ligand)
    args.update(override)
    return therapy_tiac(geometry, **args)


WIDE_RELEASE_SD = 0.5


def stage_main(record):
    """Main arm on one simulated PET: lesion rows, organ rows, therapy under the default and a wide release prior, and the spread decomposition."""
    geometry, truth = record["geometry"], record["truth"]
    n = len(geometry.lesions_ml)
    hyper = LesionHyper()
    fit = main_arm(record, hyper=hyper)
    rng = np.random.default_rng([record["seed"], record["draw"], hash_id(record["study_id"]), 16])
    flows_prior = np.exp(hyper.mu_flow + rng.normal(0, hyper.tau_flow, n))
    density_prior = np.exp(hyper.mu + rng.normal(0, hyper.tau_between) + rng.normal(0, hyper.tau_within, n))
    population = {k: PRIORS[k][0] for k in ORGANS}
    releases = np.full(n, PRIORS["l_tumor"][0])
    base = dict(l_lesions=releases, l_kidney=PRIORS["l_kidney"][0], l_salivary=PRIORS["l_salivary"][0])
    medians = dict(f_sal=PRIORS["f_sal"][0], gfr=PRIORS["gfr"][0], hematocrit=PRIORS["hematocrit"][0])
    fitted = dict(f_sal=fit["nuisance"].f_sal, gfr=fit["nuisance"].gfr, hematocrit=fit["nuisance"].hematocrit)
    if n:
        joint_r, joint_f = draw_lesions(fit["likelihood"], hyper, rng)
        flow_alone = fit["posterior"]["f_q"][2]
        median_r = fit["posterior"]["r_q"][2]
    else:
        joint_r, joint_f, flow_alone, median_r = np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
    cases = {
        "no PET": therapy_tiac(geometry, population, density_prior, f_lesions=flows_prior, **medians, **base),
        "PET organs, lesion density and flow from the prior": therapy_tiac(geometry, fit["r_den"], density_prior, f_lesions=flows_prior, **fitted, **base),
        "PET flow, density from the prior": therapy_tiac(geometry, fit["r_den"], density_prior, f_lesions=flow_alone, **fitted, **base),
        "PET joint draw of density and flow": therapy_tiac(geometry, fit["r_den"], joint_r, f_lesions=joint_f, **fitted, **base),
        "PET joint, posterior medians": therapy_tiac(geometry, fit["r_den"], median_r, f_lesions=flow_alone, **fitted, **base),
    }
    reference = _truth_therapy(record)
    therapy = []
    for case, values in cases.items():
        therapy += _therapy_rows(record, values, reference, "default release prior", case)
    wide = dict(l_kidney=float(PRIORS["l_kidney"][0] * np.exp(rng.normal(0, WIDE_RELEASE_SD))), l_salivary=float(PRIORS["l_salivary"][0] * np.exp(rng.normal(0, WIDE_RELEASE_SD))), l_lesions=PRIORS["l_tumor"][0] * np.exp(rng.normal(0, WIDE_RELEASE_SD, n)))
    reference_wide = _truth_therapy(record, **wide)
    for case, values in cases.items():
        therapy += _therapy_rows(record, values, reference_wide, "wide release prior", case)
    decomposition = []
    sources = {
        "density": dict(r_den=population, r_lesions=np.full(n, TUMOUR_MEDIAN)),
        "lesion flow": dict(f_lesions=np.full(n, PRIORS["f_lesion"][0])),
        "GFR": dict(gfr=PRIORS["gfr"][0]),
        "release rate": dict(l_kidney=PRIORS["l_kidney"][0], l_salivary=PRIORS["l_salivary"][0], l_lesions=releases),
        "hematocrit": dict(hematocrit=PRIORS["hematocrit"][0]),
        "f_SAL": dict(f_sal=PRIORS["f_sal"][0]),
    }
    for source, change in sources.items():
        values = _truth_therapy(record, **change)
        for key in ORGANS:
            decomposition.append(dict(study_id=geometry.study_id, draw=record["draw"], source=source, structure=key, ratio=values[key] / reference[key]))
        decomposition += [dict(study_id=geometry.study_id, draw=record["draw"], source=source, structure="lesions", ratio=a / b) for a, b in zip(values["lesions"], reference["lesions"])]
    return dict(organs=_organ_rows(record, fit, "main"), lesions=_lesion_rows(record, fit, hyper), therapy=therapy, decomposition=decomposition, fitted_gfr=fit["nuisance"].gfr, true_gfr=truth.gfr)


def ligand_cell(kd_nm, lambda_int):
    return P.Ligand(k_off=kd_nm * 0.046, lambda_int=lambda_int)


def stage_ligand(args):
    """One simulated patient in one cell of the ligand grid. Truth and PET under the cell's ligand; refit under the same ligand and under the base one."""
    study_id, kd_nm, lambda_int, seed = args
    rng = np.random.default_rng([seed, hash_id(study_id)])
    geometry = load_geometry(study_id)
    truth = draw_truth(geometry, rng)
    ligand = ligand_cell(kd_nm, lambda_int)
    result = simulate_truth(geometry, truth, ligand)
    measure, _ = paint_and_measure(geometry, truth, result)
    record = dict(study_id=study_id, draw=0, seed=seed, geometry=geometry, truth=truth, measure=measure)
    rows_o, rows_l, rows_t = [], [], []
    n = len(geometry.lesions_ml)
    releases = np.full(n, PRIORS["l_tumor"][0])
    base = dict(l_lesions=releases, l_kidney=PRIORS["l_kidney"][0], l_salivary=PRIORS["l_salivary"][0])
    population = {k: PRIORS[k][0] for k in ORGANS}
    medians = dict(f_sal=PRIORS["f_sal"][0], gfr=PRIORS["gfr"][0], hematocrit=PRIORS["hematocrit"][0])
    reference = _truth_therapy(record, ligand=ligand)
    hyper = LesionHyper()
    draw_rng = np.random.default_rng([seed, hash_id(study_id), 3])
    density_prior = np.exp(hyper.mu + draw_rng.normal(0, hyper.tau_between) + draw_rng.normal(0, hyper.tau_within, n))
    flows_prior = np.exp(hyper.mu_flow + draw_rng.normal(0, hyper.tau_flow, n))
    no_pet = therapy_tiac(geometry, population, density_prior, f_lesions=flows_prior, ligand=ligand, **medians, **base)
    rows_t += _therapy_rows(record, no_pet, reference, "cell", "no PET")
    for tag, fit_ligand in [("refit, correct ligand", ligand), ("refit, base ligand assumed", None)]:
        fit = main_arm(record, FitOptions(fit_nuisance=MAIN_NUISANCE, ligand=fit_ligand), hyper)
        fitted = dict(f_sal=fit["nuisance"].f_sal, gfr=fit["nuisance"].gfr, hematocrit=fit["nuisance"].hematocrit)
        if n:
            joint_r, joint_f = draw_lesions(fit["likelihood"], hyper, draw_rng)
            classes = lesion_classes(fit["profile"])
            rows_l += [dict(study_id=study_id, tag=tag, lesion=j, lesion_class=classes[j], lower=fit["profile"]["lower"][j], truth_r=truth.r_lesions[j], truth_f=truth.f_lesions[j], f_median=fit["posterior"]["f_q"][2][j], r_median=fit["posterior"]["r_q"][2][j], r_q025=fit["posterior"]["r_q"][0][j], r_q975=fit["posterior"]["r_q"][4][j]) for j in range(n)]
        else:
            joint_r, joint_f = np.zeros(0), np.zeros(0)
        values = therapy_tiac(geometry, fit["r_den"], joint_r, f_lesions=joint_f, ligand=fit_ligand, **fitted, **base)
        rows_t += _therapy_rows(record, values, reference, "cell", tag)
        rows_o += _organ_rows(record, fit, tag)
    measure_row = dict(study_id=study_id, kidney=measure["kidney_grown"], liver=measure["liver"] * geometry.liver_ml, spleen=measure["spleen"] * geometry.spleen_ml, salivary_conc=measure["salivary_conc"], blood=measure["blood"])
    for rows in (rows_o, rows_l, rows_t):
        for row in rows:
            row.update(kd_nm=kd_nm, lambda_int=lambda_int)
    measure_row.update(kd_nm=kd_nm, lambda_int=lambda_int)
    return dict(organs=rows_o, lesions=rows_l, therapy=rows_t, measure=measure_row)


def stage_measure_population(args):
    """Simulated PET with truth from another population, for the empirical Bayes test."""
    study_id, seed, priors, hyper = args
    rng = np.random.default_rng([seed, 1, hash_id(study_id)])
    geometry = load_geometry(study_id)
    truth = draw_truth(geometry, rng, priors, hyper)
    result = simulate_truth(geometry, truth)
    measure, percent = paint_and_measure(geometry, truth, result)
    return dict(study_id=study_id, draw=0, seed=seed, geometry=geometry, truth=truth, measure=measure, percent=percent)


def stage_eb_fit(args):
    record, priors, hyper, offsets, *rest = args
    settings = rest[0] if rest else dict(fit_nuisance=MAIN_NUISANCE)
    options = FitOptions(**settings, priors=priors, scanner_offset=(offsets or {}).get(record["geometry"].scanner, 0.0))
    fit = main_arm(record, options, hyper)
    geometry = record["geometry"]
    hyper = hyper or LesionHyper()
    n = len(geometry.lesions_ml)
    table = {**PRIORS, **(priors or {})}
    rng = np.random.default_rng([record["seed"], hash_id(record["study_id"]), 21])
    base = dict(l_lesions=np.full(n, PRIORS["l_tumor"][0]), l_kidney=PRIORS["l_kidney"][0], l_salivary=PRIORS["l_salivary"][0])
    reference = _truth_therapy(record)
    density_prior = np.exp(hyper.mu + rng.normal(0, hyper.tau_between) + rng.normal(0, hyper.tau_within, n))
    flows_prior = np.exp(hyper.mu_flow + rng.normal(0, hyper.tau_flow, n))
    no_pet = therapy_tiac(geometry, {k: table[k][0] for k in ORGANS}, density_prior, f_lesions=flows_prior, f_sal=table["f_sal"][0], gfr=PRIORS["gfr"][0], hematocrit=PRIORS["hematocrit"][0], **base)
    joint_r, joint_f = draw_lesions(fit["likelihood"], hyper, rng) if n else (np.zeros(0), np.zeros(0))
    with_pet = therapy_tiac(geometry, fit["r_den"], joint_r, f_lesions=joint_f, f_sal=fit["nuisance"].f_sal, gfr=fit["nuisance"].gfr, hematocrit=fit["nuisance"].hematocrit, **base)
    therapy = _therapy_rows(record, no_pet, reference, "eb", "no PET") + _therapy_rows(record, with_pet, reference, "eb", "PET joint draw")
    return dict(study_id=record["study_id"], subject=geometry.subject, f_sal_truth=record["truth"].f_sal, f_sal_fit=fit["nuisance"].f_sal, draw=record["draw"], scanner=geometry.scanner, likelihood=fit["likelihood"], organ_grid=organ_likelihood_grid(geometry, fit), organs=_organ_rows(record, fit, "eb"), lesions=_lesion_rows(record, fit, hyper), therapy=therapy)


def organ_likelihood_grid(geometry, fit, topology="reduced", priors=None):
    """Model log organ value on a density grid, other organs at the fit and nuisance at the population, plus the data and its variance with the nuisance marginalised."""
    linear = fit["linear"]
    n_r = linear["n_r"]
    nuisance_prior = {**NUISANCE_PRIOR, **{k: v for k, v in (priors or {}).items() if k in NUISANCE_PRIOR}}
    nuisance = replace(fit["nuisance"], **{k: nuisance_prior[k][0] for k in linear["extra"]})
    r_lesions = fit["posterior"]["r_q"][2] if fit["posterior"] else np.zeros(0)
    extra_var = np.array([nuisance_prior[k][1] ** 2 for k in linear["extra"]])
    out = {}
    for i, key in enumerate(linear["r_keys"]):
        values = np.exp(np.log(PRIORS[key][0]) + ORGAN_GRID_LOG)
        model = []
        for value in values:
            r_den = {**fit["r_den"], key: float(value)}
            prediction, *_ = _predict(geometry, nuisance, r_den, r_lesions, "concentration", topology)
            model.append(np.log(prediction[key]))
        J_n = linear["jacobian"][i, n_r:]
        data = np.log(max(fit["obs"][key], 1e-9))
        out[key] = dict(model=np.array(model), data=data, variance=linear["sigma"][i] ** 2 + float(np.sum(J_n**2 * extra_var)), nu=float(linear.get("nu", np.full(i + 1, np.inf))[i]))
    if "salivary" in linear["r_keys"] and "f_sal" in linear["extra"]:
        i = linear["r_keys"].index("salivary")
        which = linear["extra"].index("f_sal")
        model = []
        for value in np.exp(np.log(PRIORS["f_sal"][0]) + ORGAN_GRID_LOG):
            try:
                prediction, *_ = _predict(geometry, replace(nuisance, f_sal=float(value)), {**fit["r_den"], "salivary": PRIORS["salivary"][0]}, r_lesions, "concentration", topology)
                model.append(np.log(prediction["salivary"]))
            except ValueError:
                model.append(model[-1])
        J_n = np.delete(linear["jacobian"][i, n_r:], which)
        others = np.delete(extra_var, which)
        J_r = linear["jacobian"][i, i]
        out["f_sal"] = dict(model=np.array(model), data=np.log(max(fit["obs"]["salivary"], 1e-9)), variance=linear["sigma"][i] ** 2 + float(np.sum(J_n**2 * others)) + float(J_r**2 * PRIORS["salivary"][1] ** 2), nu=np.inf)
    return out


def eb_organs(results, priors, scanners, fit_scanner=True, keys=None, by_subject=False):
    """Population median and SD per key and scanner scale, by marginal likelihood on the grids, with the five-patient priors as hyperprior. Saturation gives a flat likelihood, as for lesions. With by_subject, the studies of a subject share one value."""
    keys = list(keys or ORGANS)
    table = {**PRIORS, **(priors or {})}
    fine = np.linspace(ORGAN_GRID_LOG[0], ORGAN_GRID_LOG[-1], 241)
    step = fine[1] - fine[0]
    prepared = {}
    for key in keys:
        rows = [r for r in results if key in r["organ_grid"]]
        names = [r["subject"] if by_subject else r["study_id"] for r in rows]
        unique = {name: i for i, name in enumerate(dict.fromkeys(names))}
        grids = [r["organ_grid"][key] for r in rows]
        prepared[key] = dict(model=np.array([PchipInterpolator(ORGAN_GRID_LOG, g["model"])(fine) for g in grids]), data=np.array([g["data"] for g in grids]), variance=np.array([g["variance"] for g in grids]), nu=np.array([g.get("nu", np.inf) for g in grids]), scanner=np.array([scanners.index(r["scanner"]) for r in rows]), group=np.array([unique[n] for n in names]), n_groups=len(unique))
    base = np.log([PRIORS[k][0] for k in keys])
    base_sd = np.array([PRIORS[k][1] for k in keys])
    n_keys = len(keys)
    weights = np.bincount([scanners.index(r["scanner"]) for r in results], minlength=len(scanners)) / len(results)

    def unpack(theta):
        mu, log_tau = theta[:n_keys], theta[n_keys:2 * n_keys]
        free = theta[2 * n_keys:] if fit_scanner else np.zeros(len(scanners) - 1)
        scale = np.r_[free, -(weights[:-1] @ free) / weights[-1]]
        return mu, np.exp(log_tau), scale

    def objective(theta):
        mu, tau, scale = unpack(theta)
        total = 0.0
        for i, key in enumerate(keys):
            p = prepared[key]
            log_prior = np.log(np.maximum(_normal(fine + base[i], mu[i], tau[i]) * step, 1e-300))
            residual = (p["data"] - scale[p["scanner"]])[:, None] - p["model"]
            scaled = residual**2 / p["variance"][:, None]
            nu = p["nu"][:, None]
            per_row = np.where(np.isfinite(nu), -0.5 * (nu + 1) * np.log1p(scaled / np.where(np.isfinite(nu), nu, 1.0)), -0.5 * scaled) - 0.5 * np.log(p["variance"][:, None])
            per_group = np.zeros((p["n_groups"], len(fine)))
            np.add.at(per_group, p["group"], per_row)
            joint = per_group + log_prior
            peak = joint.max(axis=1, keepdims=True)
            total -= (peak[:, 0] + np.log(np.exp(joint - peak).sum(axis=1))).sum()
        total += 0.5 * np.sum(((mu - base) / (base_sd / np.sqrt(FIVE_PATIENTS))) ** 2) + 0.5 * np.sum(((theta[n_keys:2 * n_keys] - np.log(base_sd)) / LOG_SD_ERROR) ** 2)
        return total

    start = np.r_[np.log([table[k][0] for k in keys]), np.log([table[k][1] for k in keys]), np.zeros(len(scanners) - 1) if fit_scanner else []]
    solution = minimize(objective, start, method="L-BFGS-B", options=dict(maxiter=300))
    mu, tau, scale = unpack(solution.x)
    hyperprior_pull = (mu - base) / (base_sd / np.sqrt(FIVE_PATIENTS))
    return dict(priors={k: (float(np.exp(m)), float(t)) for k, m, t in zip(keys, mu, tau)}, scale=dict(zip(scanners, scale)), nll=float(solution.fun), theta=solution.x, pull=dict(zip(keys, hyperprior_pull)))


def eb_lesions(results, start=None, hyperprior=True, fixed_flow=None):
    """Population of lesion density and flow with a patient effect, by marginal likelihood on the censored lesion likelihoods, with the five lesion pairs as hyperprior on density."""
    tensors = [r["likelihood"] for r in results if r["likelihood"].shape[1]]
    start = start or LesionHyper()
    base = LesionHyper()
    unpack = lambda t: LesionHyper(t[0], float(np.exp(t[1])), float(np.exp(t[2])), t[3], float(np.exp(t[4])))
    density_se = float(np.hypot(base.tau_between, base.tau_within) / np.sqrt(FIVE_PATIENTS))

    def objective(theta):
        if fixed_flow is not None:
            theta = theta.copy()
            theta[3], theta[4] = fixed_flow
        hyper = unpack(theta)
        penalty = 0.0 if not hyperprior else ((theta[0] - base.mu) / density_se) ** 2 + ((theta[1] - np.log(base.tau_between)) / LOG_SD_ERROR) ** 2 + ((theta[2] - np.log(base.tau_within)) / LOG_SD_ERROR) ** 2 + ((theta[3] - base.mu_flow) / 1.0) ** 2 + ((theta[4] - np.log(base.tau_flow)) / 1.0) ** 2
        return -sum(patient_log_marginal(x, hyper) for x in tensors) + 0.5 * penalty

    theta0 = np.array([start.mu, np.log(start.tau_between), np.log(start.tau_within), start.mu_flow, np.log(start.tau_flow)])
    solution = minimize(objective, theta0, method="Nelder-Mead", options=dict(maxiter=300, xatol=0.02, fatol=0.1))
    if fixed_flow is not None:
        solution.x[3], solution.x[4] = fixed_flow
    return dict(hyper=unpack(solution.x), nll=float(solution.fun), n_patients=len(tensors), n_lesions=int(sum(x.shape[1] for x in tensors)), objective=objective, theta=solution.x, unpack=unpack)


def noisy_arm(record, measure, options, base_nuisance=None):
    rng = np.random.default_rng([record["seed"], record["draw"], hash_id(record["study_id"]), 7])
    return fit_joint(record["geometry"], add_noise(measure, rng), options, LesionHyper(), base_nuisance=base_nuisance)


def _prediction(record, fit, rng, release="median", lesions="draw"):
    geometry, truth = record["geometry"], record["truth"]
    n = len(geometry.lesions_ml)
    joint_r, joint_f = draw_lesions(fit["likelihood"], LesionHyper(), rng) if n else (np.zeros(0), np.zeros(0))
    base = dict(l_lesions=np.full(n, PRIORS["l_tumor"][0]), l_kidney=PRIORS["l_kidney"][0], l_salivary=PRIORS["l_salivary"][0])
    return therapy_tiac(geometry, fit["r_den"], joint_r, f_lesions=joint_f, f_sal=fit["nuisance"].f_sal, gfr=fit["nuisance"].gfr, hematocrit=fit["nuisance"].hematocrit, **base)


def _ratio_rows(record, values, reference, arm, kind, per_volume=False):
    geometry, truth = record["geometry"], record["truth"]
    rows = []
    for key in ORGANS:
        ratio = values[key] / reference[key] * (truth.gland_factor if per_volume and key == "salivary" else 1.0)
        rows.append(dict(study_id=geometry.study_id, arm=arm, kind=kind, structure=key, ratio=ratio))
    return rows


def stage_excess(record):
    """Median TIAC excess: each fit variant, and for the main fit each group of fitted parameters put in the truth one at a time."""
    geometry, truth = record["geometry"], record["truth"]
    n = len(geometry.lesions_ml)
    measure, percent = record["measure"], record["percent"]
    main_options = FitOptions(fit_nuisance=MAIN_NUISANCE)
    blood_true = percent["serum_conc"] * (1 - truth.hematocrit) * geometry.rc.get("blood", 1.0) * float(np.exp(truth.scanner_effect))
    local = FitOptions(fit_nuisance=MAIN_NUISANCE, background_recovery=True)
    variants = {
        "main": (measure, main_options, None),
        "local background": (measure, local, None),
        "no image step": (direct_measure(geometry, truth, percent), FitOptions(fit_nuisance=MAIN_NUISANCE, correct_recovery=False, subtract_shell=False), None),
        "true blood datum": ({**measure, "blood": blood_true}, main_options, None),
        "local background, nuisance known": (measure, FitOptions(background_recovery=True), truth_nuisance(geometry, truth)),
        "local background, no PET/CT shift": (record["measure_no_shift"], local, None),
        "local background, organ image error in sigma": (measure, FitOptions(fit_nuisance=MAIN_NUISANCE, background_recovery=True, extra_sigma=ORGAN_IMAGE_SD), None),
    }
    reference = _truth_therapy(record)
    therapy, parameters = [], []
    main_fit = None
    for name, (m, options, base) in variants.items():
        fit = noisy_arm(record, m, options, base)
        rng = np.random.default_rng([record["seed"], hash_id(record["study_id"]), 31])
        values = _prediction(record, fit, rng)
        therapy += _ratio_rows(record, values, reference, name, "fit variant", per_volume=True)
        nuisance = fit["nuisance"]
        parameters.append(dict(study_id=geometry.study_id, arm=name, gfr=np.log(nuisance.gfr / truth.gfr), f_sal=np.log(nuisance.f_sal / truth.f_sal), hematocrit=np.log(nuisance.hematocrit / truth.hematocrit), amount=np.log(nuisance.amount_nmol / truth.amount_nmol), blood=float(np.log(fit["obs"]["blood"] / (percent["serum_conc"] * (1 - truth.hematocrit)))) if np.isfinite(fit["obs"]["blood"]) else np.nan, **{f"r_{k}": float(np.log(fit["r_den"][k] / truth.r_den[k])) for k in ORGANS}, **{f"z_{k}": float(np.log(v / truth.r_den[k]) / d) for k, v, d in zip(fit["r_keys"], fit["r"], fit["sd"])}))
        if name == "main":
            main_fit = fit
    fitted = dict(gfr=main_fit["nuisance"].gfr, f_sal=main_fit["nuisance"].f_sal, hematocrit=main_fit["nuisance"].hematocrit)
    rng = np.random.default_rng([record["seed"], hash_id(record["study_id"]), 32])
    joint_r, joint_f = draw_lesions(main_fit["likelihood"], LesionHyper(), rng) if n else (np.zeros(0), np.zeros(0))
    groups = {
        "organ densities": dict(r_den=main_fit["r_den"]),
        "GFR": dict(gfr=fitted["gfr"]),
        "hematocrit, f_SAL": dict(hematocrit=fitted["hematocrit"], f_sal=fitted["f_sal"]),
        "release at median": dict(l_kidney=PRIORS["l_kidney"][0], l_salivary=PRIORS["l_salivary"][0], l_lesions=np.full(n, PRIORS["l_tumor"][0])),
        "lesions": dict(r_lesions=joint_r, f_lesions=joint_f),
    }
    for name, change in groups.items():
        therapy += _ratio_rows(record, _truth_therapy(record, **change), reference, name, "one group at the fit")
    for name in groups:
        change = {}
        for other, values in groups.items():
            if other != name:
                change.update(values)
        therapy += _ratio_rows(record, _truth_therapy(record, **change), reference, name, "all but one group at the fit")
    all_change = {}
    for values in groups.values():
        all_change.update(values)
    therapy += _ratio_rows(record, _truth_therapy(record, **all_change), reference, "all", "all groups at the fit")
    return dict(therapy=therapy, parameters=parameters)


def stage_repaint(record):
    """Same truth, same image model, measures redone with the local background; and a second paint without the PET/CT shift."""
    geometry = load_geometry(record["study_id"])
    truth = record["truth"]
    result = simulate_truth(geometry, truth)
    measure, percent = paint_and_measure(geometry, truth, result)
    no_shift, _ = paint_and_measure(geometry, replace(truth, shift_mm=np.zeros(3)), result)
    return {**record, "geometry": geometry, "measure": measure, "measure_no_shift": no_shift, "percent": percent}


def stage_sampling(args):
    """Metropolis on organ densities and nuisance, lesions at their posterior medians, against the Laplace approximation at the MAP."""
    record, name, n_steps, n_tiac = args
    geometry, truth = record["geometry"], record["truth"]
    if name == "no image step":
        measure, options = direct_measure(geometry, truth, record["percent"]), FitOptions(fit_nuisance=MAIN_NUISANCE, correct_recovery=False, subtract_shell=False)
    else:
        measure, options = record["measure"], main_options()
    fit = noisy_arm(record, measure, options)
    linear = fit["linear"]
    r_keys, extra, keys, n_r = linear["r_keys"], linear["extra"], linear["keys"], linear["n_r"]
    obs = fit["obs"]
    n = len(geometry.lesions_ml)
    r_lesions = fit["posterior"]["r_q"][2] if n else np.zeros(0)
    base = replace(fit["nuisance"])
    data = np.array([np.log(max(obs[k], 1e-9)) for k in keys])
    medians = np.log([PRIORS[k][0] for k in r_keys] + [NUISANCE_PRIOR[k][0] for k in extra])
    sds = np.array([PRIORS[k][1] for k in r_keys] + [NUISANCE_PRIOR[k][1] for k in extra])
    sigma = linear["sigma"]

    def log_post(x):
        r_den = dict(zip(r_keys, np.exp(x[:n_r])))
        r_den.setdefault("salivary", PRIORS["salivary"][0])
        current = replace(base, **{k: float(np.exp(v)) for k, v in zip(extra, x[n_r:])})
        try:
            prediction, *_ = _predict(geometry, current, r_den, r_lesions, "concentration")
        except ValueError:
            return -np.inf
        model = np.array([np.log(prediction[k]) for k in keys])
        return -0.5 * np.sum(((data - model) / sigma) ** 2) - 0.5 * np.sum(((x - medians) / sds) ** 2)

    x_map = linear["x"]
    J = linear["jacobian"] / sigma[:, None]
    laplace = np.linalg.inv(J.T @ J + np.diag(1.0 / sds**2))
    rng = np.random.default_rng([record["seed"], hash_id(record["study_id"]), 41])
    d = len(x_map)
    proposal = np.linalg.cholesky(laplace * 2.38**2 / d + 1e-8 * np.eye(d))
    x, lp = x_map.copy(), log_post(x_map)
    chain, accepted = [], 0
    burn = n_steps // 4
    for step in range(n_steps):
        trial = x + proposal @ rng.normal(size=d)
        value = log_post(trial)
        if np.log(rng.uniform()) < value - lp:
            x, lp = trial, value
            accepted += step >= burn
        chain.append(x.copy())
        if step == burn // 2:
            proposal = np.linalg.cholesky(np.cov(np.array(chain).T) * 2.38**2 / d + 1e-6 * np.eye(d))
    chain = np.array(chain[burn:])
    reference = _truth_therapy(record)
    release = dict(l_lesions=np.full(n, PRIORS["l_tumor"][0]), l_kidney=PRIORS["l_kidney"][0], l_salivary=PRIORS["l_salivary"][0])
    f_lesions = fit["nuisance"].f_lesions

    def tiac(y, l_args):
        r_den = dict(zip(r_keys, np.exp(y[:n_r])))
        r_den.setdefault("salivary", PRIORS["salivary"][0])
        pars = dict(zip(extra, np.exp(y[n_r:])))
        return therapy_tiac(geometry, r_den, r_lesions, f_sal=pars["f_sal"], gfr=pars["gfr"], hematocrit=pars["hematocrit"], f_lesions=f_lesions, **l_args)

    at_map = tiac(x_map, release)
    picks = chain[np.linspace(0, len(chain) - 1, n_tiac).astype(int)]
    draws = [tiac(y, release) for y in picks]
    predictive = []
    for y in picks:
        spread = dict(l_lesions=PRIORS["l_tumor"][0] * np.exp(rng.normal(0, PRIORS["l_tumor"][1], n)), l_kidney=float(PRIORS["l_kidney"][0] * np.exp(rng.normal(0, PRIORS["l_kidney"][1]))), l_salivary=float(PRIORS["l_salivary"][0] * np.exp(rng.normal(0, PRIORS["l_salivary"][1]))))
        predictive.append(tiac(y, spread))
    rows = []
    for key in ORGANS:
        factor = (truth.gland_factor if key == "salivary" else 1.0) / reference[key]
        values = np.array([v[key] for v in draws]) * factor
        pred = np.array([v[key] for v in predictive]) * factor
        rows.append(dict(study_id=geometry.study_id, arm=name, structure=key, at_map=at_map[key] * factor, median_post=np.median(values), mean_post=values.mean(), geo_post=np.exp(np.log(values).mean()), median_pred=np.median(pred), mean_pred=pred.mean(), inside_95=bool(np.quantile(pred, 0.025) <= 1.0 <= np.quantile(pred, 0.975))))
    names = list(r_keys) + list(extra)
    truth_values = {**truth.r_den, "gfr": truth.gfr, "f_sal": truth.f_sal, "hematocrit": truth.hematocrit, "amount_nmol": truth.amount_nmol}
    parameters = [dict(study_id=geometry.study_id, arm=name, parameter=k, truth=np.log(truth_values[k]), map=x_map[i], post_median=np.median(chain[:, i]), post_sd=chain[:, i].std(), laplace_sd=np.sqrt(laplace[i, i]), q025=np.quantile(chain[:, i], 0.025), q975=np.quantile(chain[:, i], 0.975), acceptance=accepted / (n_steps - burn)) for i, k in enumerate(names)]
    return dict(therapy=rows, parameters=parameters)


def main_options(**change):
    """Main arm from nb18: nuisance marginalised, local background, full topology, image error of the liver in the likelihood, Student-t for the spleen. nb17 ran it with fit_topology="reduced" and a Gaussian spleen."""
    return FitOptions(**{**dict(fit_nuisance=MAIN_NUISANCE, background_recovery=True, extra_sigma=dict(liver=ORGAN_IMAGE_SD["liver"]), robust=dict(spleen=SPLEEN_T), fit_topology="full"), **change})


def stage_topology(record):
    """Is the residual excess the reduced model? The PET truth has the full topology; compare fit and therapy reference in both."""
    geometry, truth = record["geometry"], record["truth"]
    direct = direct_measure(geometry, truth, record["percent"])
    no_image = dict(fit_nuisance=MAIN_NUISANCE, correct_recovery=False, subtract_shell=False)
    reduced_ref = _truth_therapy(record)
    full_ref = _truth_therapy(record, topology="full")
    rows = []
    n = len(geometry.lesions_ml)

    def predict(fit, topology):
        release = dict(l_lesions=np.full(n, PRIORS["l_tumor"][0]), l_kidney=PRIORS["l_kidney"][0], l_salivary=PRIORS["l_salivary"][0])
        r_lesions = fit["posterior"]["r_q"][2] if n else np.zeros(0)
        return therapy_tiac(geometry, fit["r_den"], r_lesions, f_sal=fit["nuisance"].f_sal, gfr=fit["nuisance"].gfr, hematocrit=fit["nuisance"].hematocrit, f_lesions=fit["nuisance"].f_lesions, topology=topology, **release)

    arms = {
        "no image, reduced fit": (direct, FitOptions(**no_image)),
        "no image, full fit": (direct, FitOptions(**no_image, fit_topology="full")),
        "image, reduced fit": (record["measure"], main_options()),
        "image, full fit": (record["measure"], main_options(fit_topology="full")),
    }
    for name, (measure, options) in arms.items():
        fit = noisy_arm(record, measure, options)
        topology = options.fit_topology
        values = predict(fit, topology)
        for key in ORGANS:
            factor = truth.gland_factor if key == "salivary" else 1.0
            rows.append(dict(study_id=geometry.study_id, arm=name, structure=key, against_reduced=values[key] * factor / reduced_ref[key], against_full=values[key] * factor / full_ref[key], r_error=float(np.log(fit["r_den"][key] / truth.r_den[key]))))
    rows += [dict(study_id=geometry.study_id, arm="truth: full over reduced", structure=key, against_reduced=full_ref[key] / reduced_ref[key]) for key in ORGANS]
    return rows


def stage_lesion_arm(args):
    """Censored lesion likelihood of one patient under one measurement arm, for the empirical Bayes of the lesion population."""
    record, arm = args
    geometry, truth = record["geometry"], record["truth"]
    if arm == "main":
        fit = noisy_arm(record, record["measure"], main_options())
    elif arm == "no local background":
        fit = noisy_arm(record, record["measure"], FitOptions(fit_nuisance=MAIN_NUISANCE))
    elif arm == "no image step":
        fit = noisy_arm(record, direct_measure(geometry, truth, record["percent"]), FitOptions(fit_nuisance=MAIN_NUISANCE, correct_recovery=False, subtract_shell=False))
    elif arm == "no image, nuisance known":
        fit = noisy_arm(record, direct_measure(geometry, truth, record["percent"]), FitOptions(correct_recovery=False, subtract_shell=False), truth_nuisance(geometry, truth))
    return dict(study_id=record["study_id"], arm=arm, likelihood=fit["likelihood"], truth_r=truth.r_lesions, truth_f=truth.f_lesions, log_data=fit["log_data"])


def stage_measure_subject(args):
    """Simulated PET where organ densities and f_SAL belong to the subject, shared by all of its studies: the population of the real training split."""
    study_id, seed, priors, *rest = args
    table = {**PRIORS, **(priors or {})}
    geometry = load_geometry(study_id)
    truth = draw_truth(geometry, np.random.default_rng([seed, 2, hash_id(study_id)]), priors, rest[0] if rest else None)
    shared = np.random.default_rng([seed, 3, hash_id(geometry.subject)])
    truth.r_den = {k: float(table[k][0] * np.exp(shared.normal(0, table[k][1]))) for k in ORGANS}
    truth.f_sal = float(table["f_sal"][0] * np.exp(shared.normal(0, table["f_sal"][1])))
    measure, percent = paint_and_measure(geometry, truth, simulate_truth(geometry, truth))
    return dict(study_id=study_id, draw=0, seed=seed, geometry=geometry, truth=truth, measure=measure, percent=percent)


def lesion_loo(likelihood, log_data, table, rc, hyper):
    """Leave-one-out predictive of each lesion within its patient: PIT of the datum, and the posterior mean of the common error."""
    log_joint, per_density, prior_r, marginal = patient_integrals(likelihood, hyper)
    weights_f = flow_weights(hyper)
    sigma = lesion_sigma(rc)
    log_table = np.log(table)
    pit = np.zeros(len(log_data))
    for j in range(len(log_data)):
        rest = log_joint - np.log(marginal[:, j, :])
        p = np.exp(rest - rest.max())
        p = p / p.sum()
        shift = (COMMON_SD * COMMON_Z)[:, None, None]
        cdf = norm.cdf((log_data[j] - shift - log_table[None]) / sigma[j])
        pit[j] = np.einsum("du,dfr,f,ur->", p, cdf, weights_f, prior_r)
    p_joint = np.exp(log_joint - log_joint.max())
    p_joint = p_joint / p_joint.sum()
    return pit, float((p_joint.sum(axis=1) * COMMON_SD * COMMON_Z).sum()), float(np.exp((p_joint.sum(axis=0) * (hyper.tau_between * PATIENT_Z)).sum()))


def blood_against_population(geometry, fit, options):
    """Log of the blood datum over the model with GFR, hematocrit and amount at the population: what the patient's nuisance would have to absorb."""
    population = replace(fit["nuisance"], gfr=NUISANCE_PRIOR["gfr"][0], hematocrit=NUISANCE_PRIOR["hematocrit"][0], amount_nmol=NUISANCE_PRIOR["amount_nmol"][0])
    r_now = fit["posterior"]["r_q"][2] if fit["posterior"] else np.zeros(0)
    return float(np.log(fit["obs"]["blood"]) - np.log(_predict(geometry, population, fit["r_den"], r_now, "concentration", options.fit_topology)[0]["blood"]))


def study_diagnostic(geometry, measure, hyper=None, options=None, predict=0):
    """Residuals of one PET under the current priors: the same quantities for a real and for a simulated study."""
    hyper = hyper or LesionHyper()
    options = options or main_options()
    fit = fit_joint(geometry, measure, options, hyper)
    linear = fit["linear"]
    row = dict(study_id=geometry.study_id, subject=geometry.subject, scanner=geometry.scanner, tlv_ml=float(geometry.lesions_ml.sum()), n_lesions=len(geometry.lesions_ml), weight_kg=geometry.weight_kg, uptake_min=geometry.uptake_min, rc_blood=geometry.rc.get("blood", np.nan))
    for key, misfit, sigma in zip(linear["keys"], linear["misfit"], linear["sigma"]):
        row[f"misfit_{key}"] = float(misfit)
        row[f"z_{key}"] = float(misfit / sigma)
    if np.isfinite(fit["obs"]["blood"]):
        row["blood_vs_population"] = blood_against_population(geometry, fit, options)
    row.update(blood=measure["blood"], blood_background=measure["blood_background"], blood_corrected=fit["obs"]["blood"], salivary=measure["salivary_conc"], salivary_background=measure["salivary_background"], rest=measure["rest"])
    for key, value in zip(linear["r_keys"], fit["r"]):
        row[f"r_{key}"] = float(value)
    for key, value in [("gfr", fit["nuisance"].gfr), ("f_sal", fit["nuisance"].f_sal), ("hematocrit", fit["nuisance"].hematocrit), ("amount", fit["nuisance"].amount_nmol)]:
        row[key] = value
    lesions = []
    if len(geometry.lesions_ml):
        pit, common, patient = lesion_loo(fit["likelihood"], fit["log_data"], fit["table"], geometry.rc_lesions, hyper)
        row.update(common_error=common, patient_factor=patient, lesion_within_sd=float(np.std(fit["log_data"])) if len(pit) > 1 else np.nan)
        classes = lesion_classes(fit["profile"])
        lesions = [dict(study_id=geometry.study_id, lesion=j, pit=pit[j], volume_ml=geometry.lesions_ml[j], rc=geometry.rc_lesions[j], background=measure["lesion_background"][j], raw=measure["lesion_conc"][j], corrected=fit["obs"]["lesions"][j], n=len(pit), lesion_class=classes[j], lower=fit["profile"]["lower"][j]) for j in range(len(pit))]
    row.update({k: measure[k] for k in ("kidney_grown", "kidney_fluid_pct", "kidney_fluid_ml", "kidney_functional_pct", "kidney_functional_ml", "kidney_sinus_pct", "kidney_sinus_ml", "kidney_parenchyma", "kidney_sinus_excess_pct", "kidney_shell_excess_pct") if k in measure})
    row.update(kidneys_ml=geometry.kidneys_ml, shell_ml=geometry.shell_ml, datum_kidneys=fit["obs"]["kidneys"])
    out = dict(row=row, lesions=lesions, likelihood=fit["likelihood"], table=fit["table"], log_data=fit["log_data"], rc=geometry.rc_lesions, volume=geometry.lesions_ml, organ_grid=organ_likelihood_grid(geometry, fit, options.fit_topology, options.priors), subject=geometry.subject, scanner=geometry.scanner, study_id=geometry.study_id)
    if predict:
        draws = predictive_tiac(geometry, fit, options, hyper, np.random.default_rng([hash_id(geometry.study_id), 51]), predict)
        out["therapy"] = {k: float(np.median(draws[k])) for k in ORGANS} | dict(ess=draws["ess"], lesions=np.median(draws["lesions"], axis=0))
    return out


def stage_diagnostic_simulated(record):
    rng = np.random.default_rng([record["seed"], record["draw"], hash_id(record["study_id"]), 7])
    return study_diagnostic(record["geometry"], add_noise(record["measure"], rng))


def scanner_options(geometry, settings):
    """Main options with the scanner scale and the blood offset of this patient's scanner."""
    settings = dict(settings or {})
    offsets, blood = settings.pop("offsets", None) or {}, settings.pop("blood_offsets", None) or {}
    return main_options(scanner_offset=offsets.get(geometry.scanner, 0.0), blood_offset=blood.get(geometry.scanner, 0.0), **settings)


def stage_diagnostic_real(args):
    """Real PET of the training split. Refuses any study of the test split."""
    study_id, test_studies, settings = args
    if study_id in test_studies:
        raise ValueError(f"{study_id} is in the test split")
    settings = dict(settings or {})
    geometry = load_geometry(study_id, settings.pop("psf", "scanner"))
    hyper, predict = settings.pop("hyper", None), settings.pop("predict", 0)
    return study_diagnostic(geometry, real_measure(geometry), hyper, scanner_options(geometry, settings), predict)


def lesion_population_nll(results, hyper):
    """-log L of a lesion population on the fine patient-effect grid."""
    return -sum(patient_log_marginal(r["likelihood"], hyper, PROFILE_Z) for r in results if r["likelihood"].shape[1])


def lesion_mu_profile(results, mu):
    """Profile of the population median of lesion density at one value: all other population parameters refitted from three starts, spreads bounded so the density grid resolves them."""
    tensors = [r["likelihood"] for r in results if r["likelihood"].shape[1]]
    bounds = [(np.log(0.02), np.log(2.5)), (np.log(DELTA_R), np.log(3.0)), (LOG_F[2], LOG_F[-3]), (np.log(0.2), np.log(3.0))]
    unpack = lambda t: LesionHyper(float(mu), float(np.exp(t[0])), float(np.exp(t[1])), float(t[2]), float(np.exp(t[3])))
    objective = lambda t: -sum(patient_log_marginal(x, unpack(t), PROFILE_Z) for x in tensors)
    solutions = [minimize(objective, np.array([np.log(tb), np.log(tw), np.log(0.3), 0.0]), method="L-BFGS-B", bounds=bounds, options=dict(eps=1e-3, ftol=1e-7, maxiter=60)) for tb, tw in ((0.4, 0.5), (0.8, 0.6), (1.4, 0.7))]
    best = min(solutions, key=lambda x: x.fun)
    hyper = unpack(best.x)
    return dict(median=float(np.exp(mu)), nll=float(best.fun), spread_of_starts=float(max(x.fun for x in solutions) - best.fun), tau_between=hyper.tau_between, tau_within=hyper.tau_within, flow=float(np.exp(hyper.mu_flow)), tau_flow=hyper.tau_flow, q10=float(np.exp(mu - 1.2816 * np.hypot(hyper.tau_between, hyper.tau_within))))


def stage_groups(record):
    """Median TIAC excess of the main arm: each group of fitted parameters put in the truth one at a time, and all but one."""
    geometry = record["geometry"]
    n = len(geometry.lesions_ml)
    fit = noisy_arm(record, record["measure"], main_options())
    nuisance = fit["nuisance"]
    groups = {
        "organ densities": dict(r_den=fit["r_den"]),
        "GFR": dict(gfr=nuisance.gfr),
        "hematocrit": dict(hematocrit=nuisance.hematocrit),
        "f_SAL": dict(f_sal=nuisance.f_sal),
        "release at median": dict(l_kidney=PRIORS["l_kidney"][0], l_salivary=PRIORS["l_salivary"][0], l_lesions=np.full(n, PRIORS["l_tumor"][0])),
    }
    reference = _truth_therapy(record)
    rows = []
    for name, change in groups.items():
        rows += _ratio_rows(record, _truth_therapy(record, **change), reference, name, "one group at the fit")
        others = {}
        for other, values in groups.items():
            if other != name:
                others.update(values)
        rows += _ratio_rows(record, _truth_therapy(record, **others), reference, name, "all but one group at the fit")
    everything = {}
    for values in groups.values():
        everything.update(values)
    rows += _ratio_rows(record, _truth_therapy(record, **everything), reference, "all", "all groups at the fit")
    return rows


def safe_call(job):
    """Run one stage in a worker; a failure comes back as a record instead of stopping the cohort."""
    name, args = job
    try:
        return globals()[name](args)
    except (ValueError, RuntimeError) as error:
        return dict(error=repr(error)[:300], stage=name)


def organ_log_posterior(geometry, fit, options):
    """Log posterior of organ densities and nuisance of one patient, lesions at their posterior medians: the target of sampling."""
    linear = fit["linear"]
    r_keys, extra, keys, n_r = linear["r_keys"], linear["extra"], linear["keys"], linear["n_r"]
    table = {**PRIORS, **(options.priors or {})}
    nuisance_prior = {**NUISANCE_PRIOR, **{k: v for k, v in (options.priors or {}).items() if k in NUISANCE_PRIOR}}
    medians = np.log([table[k][0] for k in r_keys] + [nuisance_prior[k][0] for k in extra])
    sds = np.array([table[k][1] for k in r_keys] + [nuisance_prior[k][1] for k in extra])
    data = np.array([np.log(max(fit["obs"][k], 1e-9)) for k in keys])
    sigma, nu = linear["sigma"], linear["nu"]
    r_lesions = fit["posterior"]["r_q"][2] if fit["posterior"] else np.zeros(0)
    base = replace(fit["nuisance"])
    salivary = table["salivary"][0]
    h_index = n_r + extra.index("hematocrit") if "hematocrit" in extra else None

    def log_post(x):
        if h_index is not None and not np.log(HEMATOCRIT_BOUNDS[0]) <= x[h_index] <= np.log(HEMATOCRIT_BOUNDS[1]):
            return -np.inf
        r_den = dict(zip(r_keys, np.exp(x[:n_r])))
        r_den.setdefault("salivary", salivary)
        current = replace(base, **{k: float(np.exp(v)) for k, v in zip(extra, x[n_r:])})
        try:
            prediction, *_ = _predict(geometry, current, r_den, r_lesions, "concentration", options.fit_topology, options.ligand)
        except ValueError:
            return -np.inf
        z = (data - np.array([np.log(prediction[k]) for k in keys])) / sigma
        return -0.5 * np.sum(_student(z, nu) ** 2) - 0.5 * np.sum(((x - medians) / sds) ** 2)

    J = linear["jacobian"] / sigma[:, None] * np.where(np.isfinite(nu), np.sqrt((nu + 1) / np.where(np.isfinite(nu), nu, 1.0)), 1.0)[:, None]
    laplace = np.linalg.inv(J.T @ J + np.diag(1.0 / sds**2))
    return log_post, linear["x"].copy(), laplace, list(r_keys) + list(extra)


def posterior_sample(geometry, fit, options, rng, n_proposal=200, nu_proposal=5.0, inflate=1.2):
    """Self-normalised importance sampling from a Student-t around the MAP with the Laplace scale. Returns draws, weights and the effective sample size."""
    log_post, x_map, laplace, names = organ_log_posterior(geometry, fit, options)
    d = len(x_map)
    chol = np.linalg.cholesky(laplace * inflate**2 + 1e-9 * np.eye(d))
    normal = rng.normal(size=(n_proposal, d))
    scale = np.sqrt(rng.chisquare(nu_proposal, n_proposal) / nu_proposal)
    steps = normal / scale[:, None]
    draws = x_map + steps @ chol.T
    log_q = -0.5 * (nu_proposal + d) * np.log1p(np.sum(normal**2, axis=1) / scale**2 / nu_proposal)
    log_p = np.array([log_post(x) for x in draws])
    log_w = np.where(np.isfinite(log_p), log_p - log_q, -np.inf)
    weights = np.exp(log_w - log_w.max())
    weights = weights / weights.sum()
    return dict(x=draws, weights=weights, ess=float(1 / np.sum(weights**2)), names=names, n_r=len(fit["linear"]["r_keys"]), r_keys=fit["linear"]["r_keys"])


def _therapy_from_draw(geometry, fit, options, hyper, sample, y, rng, drawn_release=True):
    n = len(geometry.lesions_ml)
    table = {**PRIORS, **(options.priors or {})}
    n_r = sample["n_r"]
    r_den = dict(zip(sample["r_keys"], np.exp(y[:n_r])))
    r_den.setdefault("salivary", table["salivary"][0])
    extra = dict(zip(sample["names"][n_r:], np.exp(y[n_r:])))
    joint_r, joint_f = draw_lesions(fit["likelihood"], hyper, rng) if n else (np.zeros(0), np.zeros(0))
    if drawn_release:
        release = dict(l_lesions=PRIORS["l_tumor"][0] * np.exp(rng.normal(0, PRIORS["l_tumor"][1], n)), l_kidney=float(PRIORS["l_kidney"][0] * np.exp(rng.normal(0, PRIORS["l_kidney"][1]))), l_salivary=float(PRIORS["l_salivary"][0] * np.exp(rng.normal(0, PRIORS["l_salivary"][1]))))
    else:
        release = dict(l_lesions=np.full(n, PRIORS["l_tumor"][0]), l_kidney=PRIORS["l_kidney"][0], l_salivary=PRIORS["l_salivary"][0])
    return therapy_tiac(geometry, r_den, joint_r, f_sal=extra.get("f_sal", fit["nuisance"].f_sal), gfr=extra.get("gfr", fit["nuisance"].gfr), hematocrit=extra.get("hematocrit", fit["nuisance"].hematocrit), f_lesions=joint_f, topology=options.fit_topology, **release)


def predictive_tiac(geometry, fit, options, hyper, rng, n_draws=30, drawn_release=True):
    """Therapy TIAC of one patient from the posterior: organ densities and nuisance by importance sampling, lesions by the joint draw, release from its prior."""
    hyper = hyper or LesionHyper()
    sample = posterior_sample(geometry, fit, options, rng)
    picks = sample["x"][rng.choice(len(sample["weights"]), n_draws, p=sample["weights"])]
    values = [_therapy_from_draw(geometry, fit, options, hyper, sample, y, rng, drawn_release) for y in picks]
    out = {k: np.array([v[k] for v in values]) for k in ORGANS}
    out["lesions"] = np.array([v["lesions"] for v in values])
    out["ess"] = sample["ess"]
    return out


def posterior_therapy_rows(record, draws, reference, tag):
    geometry, truth = record["geometry"], record["truth"]
    rows = []
    for key in ORGANS:
        factor = (truth.gland_factor if key == "salivary" else 1.0) / reference[key]
        v = draws[key] * factor
        rows.append(dict(study_id=geometry.study_id, tag=tag, structure=key, ratio=float(np.median(v)), mean_ratio=float(v.mean()), inside_95=bool(np.quantile(v, 0.025) <= 1.0 <= np.quantile(v, 0.975)), inside_68=bool(np.quantile(v, 0.16) <= 1.0 <= np.quantile(v, 0.84)), log_sd=float(np.log(v).std())))
    if len(geometry.lesions_ml):
        v = draws["lesions"] / reference["lesions"]
        for j in range(v.shape[1]):
            rows.append(dict(study_id=geometry.study_id, tag=tag, structure="lesions", ratio=float(np.median(v[:, j])), mean_ratio=float(v[:, j].mean()), inside_95=bool(np.quantile(v[:, j], 0.025) <= 1.0 <= np.quantile(v[:, j], 0.975)), inside_68=bool(np.quantile(v[:, j], 0.16) <= 1.0 <= np.quantile(v[:, j], 0.84)), log_sd=float(np.log(v[:, j]).std())))
    return rows


def stage_sampling_full(args):
    """Metropolis on organ densities and nuisance against the Laplace approximation and importance sampling, full topology, priors of the fit."""
    record, n_steps, n_tiac, change = args
    geometry, truth = record["geometry"], record["truth"]
    options = main_options(**(change or {}))
    fit = noisy_arm(record, record["measure"], options)
    log_post, x_map, laplace, names = organ_log_posterior(geometry, fit, options)
    rng = np.random.default_rng([record["seed"], hash_id(record["study_id"]), 41])
    d = len(x_map)
    proposal = np.linalg.cholesky(laplace * 2.38**2 / d + 1e-8 * np.eye(d))
    x, lp = x_map.copy(), log_post(x_map)
    chain, accepted = [], 0
    burn = n_steps // 4
    for step in range(n_steps):
        trial = x + proposal @ rng.normal(size=d)
        value = log_post(trial)
        if np.log(rng.uniform()) < value - lp:
            x, lp = trial, value
            accepted += step >= burn
        chain.append(x.copy())
        if step == burn // 2:
            proposal = np.linalg.cholesky(np.cov(np.array(chain).T) * 2.38**2 / d + 1e-6 * np.eye(d))
    chain = np.array(chain[burn:])
    sample = posterior_sample(geometry, fit, options, rng)
    resampled = sample["x"][rng.choice(len(sample["weights"]), 400, p=sample["weights"])]
    reference = _truth_therapy(record, topology=options.fit_topology)
    hyper = LesionHyper()
    meta = dict(n_r=sample["n_r"], r_keys=sample["r_keys"], names=names)
    picks = chain[np.linspace(0, len(chain) - 1, n_tiac).astype(int)]
    therapy = lambda y, drawn: _therapy_from_draw(geometry, fit, options, hyper, meta, y, rng, drawn)
    at_map = _therapy_from_draw(geometry, fit, options, hyper, meta, x_map, rng, False)
    mcmc = [therapy(y, False) for y in picks]
    importance = [therapy(y, False) for y in resampled[: n_tiac]]
    predictive = [therapy(y, True) for y in picks]
    rows = []
    for key in ORGANS:
        factor = (truth.gland_factor if key == "salivary" else 1.0) / reference[key]
        a, b, c = (np.array([v[key] for v in group]) * factor for group in (mcmc, importance, predictive))
        rows.append(dict(study_id=geometry.study_id, structure=key, at_map=at_map[key] * factor, median_post=np.median(a), mean_post=a.mean(), geo_post=np.exp(np.log(a).mean()), median_importance=np.median(b), median_pred=np.median(c), inside_95=bool(np.quantile(c, 0.025) <= 1 <= np.quantile(c, 0.975)), inside_68=bool(np.quantile(c, 0.16) <= 1 <= np.quantile(c, 0.84))))
    truth_values = {**truth.r_den, "gfr": truth.gfr, "f_sal": truth.f_sal, "hematocrit": truth.hematocrit, "amount_nmol": truth.amount_nmol}
    parameters = [dict(study_id=geometry.study_id, parameter=k, truth=np.log(truth_values[k]), map=x_map[i], post_median=np.median(chain[:, i]), post_sd=chain[:, i].std(), laplace_sd=np.sqrt(laplace[i, i]), importance_median=np.median(resampled[:, i]), importance_sd=resampled[:, i].std(), q025=np.quantile(chain[:, i], 0.025), q975=np.quantile(chain[:, i], 0.975), acceptance=accepted / (n_steps - burn), ess=sample["ess"]) for i, k in enumerate(names)]
    return dict(therapy=rows, parameters=parameters)


def stage_fit_full(args):
    """Empirical Bayes pass on one simulated patient under the main arm. Therapy from the posterior when predict is set."""
    record, change, hyper, predict = args
    geometry, truth = record["geometry"], record["truth"]
    options = scanner_options(geometry, change)
    hyper = hyper or LesionHyper()
    rng = np.random.default_rng([record["seed"], record["draw"], hash_id(record["study_id"]), 7])
    fit = fit_joint(geometry, add_noise(record["measure"], rng), options, hyper)
    study_row = dict(study_id=geometry.study_id, subject=geometry.subject, scanner=geometry.scanner, tlv_ml=float(geometry.lesions_ml.sum()), gfr=fit["nuisance"].gfr, hematocrit=fit["nuisance"].hematocrit, f_sal=fit["nuisance"].f_sal, blood_vs_population=blood_against_population(geometry, fit, options), **{f"z_{k}": float(m / d) for k, m, d in zip(fit["linear"]["keys"], fit["linear"]["misfit"], fit["linear"]["sigma"])})
    out = dict(study_id=record["study_id"], subject=geometry.subject, scanner=geometry.scanner, f_sal_truth=truth.f_sal, f_sal_fit=fit["nuisance"].f_sal, gfr_error=float(np.log(fit["nuisance"].gfr / truth.gfr)), hematocrit_error=float(np.log(fit["nuisance"].hematocrit / truth.hematocrit)), likelihood=fit["likelihood"], table=fit["table"], log_data=fit["log_data"], rc=geometry.rc_lesions, volume=geometry.lesions_ml, truth_r=truth.r_lesions, truth_f=truth.f_lesions, organ_grid=organ_likelihood_grid(geometry, fit, options.fit_topology, options.priors), organs=_organ_rows(record, fit, "fit"), row=study_row)
    if predict:
        reference = _truth_therapy(record, topology=options.fit_topology)
        p_rng = np.random.default_rng([record["seed"], hash_id(record["study_id"]), 61])
        draws = predictive_tiac(geometry, fit, options, hyper, p_rng, predict)
        table = {**PRIORS, **(options.priors or {})}
        n = len(geometry.lesions_ml)
        density = np.exp(hyper.mu + p_rng.normal(0, hyper.tau_between) + p_rng.normal(0, hyper.tau_within, n))
        if hyper.pi_low > 0:
            density = np.where(p_rng.uniform(size=n) < hyper.pi_low, np.exp(hyper.mu_low + p_rng.normal(0, hyper.tau_low, n)), density)
        no_pet = therapy_tiac(geometry, {k: table[k][0] for k in ORGANS}, density, f_sal=table["f_sal"][0], gfr=PRIORS["gfr"][0], hematocrit=PRIORS["hematocrit"][0], f_lesions=np.exp(hyper.mu_flow + p_rng.normal(0, hyper.tau_flow, n)), l_lesions=np.full(n, PRIORS["l_tumor"][0]), l_kidney=PRIORS["l_kidney"][0], l_salivary=PRIORS["l_salivary"][0], topology=options.fit_topology)
        out["therapy"] = posterior_therapy_rows(record, draws, reference, "PET posterior") + _therapy_rows(record, no_pet, reference, "no PET", "no PET")
        out["ess"] = draws["ess"]
    return out


@dataclass
class LesionOffset:
    """Shift of the log datum of a lesion. viable: a necrotic core lowers the mean over a large mask. mask: a thresholded small mask selects hot voxels, so y/RC is too high. bins: free by recovery class."""
    kind: str = "none"
    value: tuple = (0.0,)
    reference_ml: float = 10.0

    def __call__(self, rc, volume):
        if self.kind == "viable":
            return -self.value[0] * np.maximum(np.log(volume / self.reference_ml), 0.0)
        if self.kind == "mask":
            return self.value[0] * np.log(1.0 / rc)
        if self.kind == "bins":
            return np.select([rc < 0.3, rc < 0.5, rc < 0.7], [self.value[0], self.value[1], 0.0], self.value[2])
        return np.zeros(len(rc))


def lesion_offset_test(results, kind, hyper, reference_ml=10.0, hyperprior=False):
    """Likelihood of one offset model against none: offset fitted with the population fixed, then the population refitted. Returns the gain in -log L."""
    size = 3 if kind == "bins" else 1
    bounds = [(-1.5, 1.5)] * 3 if kind == "bins" else [(0.0, 1.0)]

    def nll(theta):
        tensors = lesion_offset_tensors(results, LesionOffset(kind, tuple(theta), reference_ml))
        return -sum(patient_log_marginal(x, hyper, PROFILE_Z) for x in tensors)

    start = np.zeros(size)
    solution = minimize(nll, start, method="L-BFGS-B", bounds=bounds, options=dict(eps=1e-2, maxiter=25))
    offset = LesionOffset(kind, tuple(float(x) for x in solution.x), reference_ml)
    refit = eb_lesions_bounded(lesion_offset_tensors(results, offset), hyper, hyperprior, hyper.pi_low > 0)
    return dict(kind=kind, value=offset.value, reference_ml=reference_ml, nll_fixed_population=float(solution.fun), nll=refit["nll"], hyper=refit["hyper"])


def lesion_offset_tensors(results, offset):
    """Censored likelihoods rebuilt with a shift of the log datum per lesion: offset(rc, volume)."""
    return [lesion_likelihood(r["log_data"] - offset(r["rc"], r["volume"]), r["table"], r["rc"]) for r in results if len(r["log_data"])]


def eb_lesions_bounded(tensors, start=None, hyperprior=True, low_tail=False, fixed_mu=None, maxiter=60, flow_cap=np.inf):
    """Lesion population by marginal likelihood on the censored likelihoods with the patient effect: bounded spreads so the grid resolves them, optional low component for non-avid lesions at least a factor e below the median, optional hyperprior from the five lesion pairs. One start; call from several."""
    start = start or LesionHyper()
    base = LesionHyper()
    density_se = float(np.hypot(base.tau_between, base.tau_within) / np.sqrt(FIVE_PATIENTS))
    names = ["mu", "tau_between", "tau_within", "mu_flow", "tau_flow"] + (["pi_low", "mu_low"] if low_tail else [])
    x0 = [start.mu, np.log(start.tau_between), np.log(start.tau_within), start.mu_flow, np.log(start.tau_flow)]
    bounds = [(np.log(5.0), np.log(2000.0)), (np.log(0.02), np.log(2.5)), (np.log(DELTA_R), np.log(3.0)), (LOG_F[2], LOG_F[-3]), (np.log(0.2), np.log(3.0))]
    if low_tail:
        x0 += [np.log(max(start.pi_low, 0.05)), max(start.mu - start.mu_low, 1.0)]
        bounds += [(np.log(0.002), np.log(0.5)), (1.0, 7.0)]
    if fixed_mu is not None:
        x0[0] = fixed_mu
        bounds[0] = (fixed_mu, fixed_mu)
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

    def unpack(t):
        extra = dict(pi_low=float(np.exp(t[5])), mu_low=float(max(t[0] - t[6], LOG_R[0]))) if low_tail else {}
        return LesionHyper(float(t[0]), float(np.exp(t[1])), float(np.exp(t[2])), float(t[3]), float(np.exp(t[4])), flow_cap=flow_cap, **extra)

    def objective(t):
        hyper = unpack(t)
        penalty = 0.0
        if hyperprior:
            penalty = ((t[0] - base.mu) / density_se) ** 2 + ((t[1] - np.log(base.tau_between)) / LOG_SD_ERROR) ** 2 + ((t[2] - np.log(base.tau_within)) / LOG_SD_ERROR) ** 2 + ((t[3] - base.mu_flow) / 1.0) ** 2 + ((t[4] - np.log(base.tau_flow)) / 1.0) ** 2
        return -sum(patient_log_marginal(x, hyper, PROFILE_Z) for x in tensors) + 0.5 * penalty

    solution = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options=dict(eps=1e-3, ftol=1e-7, maxiter=maxiter))
    return dict(hyper=unpack(solution.x), nll=float(solution.fun), names=names, theta=solution.x)


def stage_lesion_edge(args):
    """Is the mask of a real lesion its true extent? Activity in the first voxel layer outside the mask over the mask, background removed, against a uniform lesion of that mask blurred by the PSF. Lesions touching another lesion in that layer are left out."""
    study_id, test_studies = args
    if study_id in test_studies:
        raise ValueError(f"{study_id} is in the test split")
    geometry = load_geometry(study_id)
    folder = PHANTOMS / study_id
    lesions = np.asanyarray(nib.load(folder / "lesions.nii.gz").dataobj)
    pet = np.asanyarray(nib.load(folder / "pet_pct_ia_ml.nii.gz").dataobj).astype(np.float32)
    n = len(geometry.lesions_ml)
    if not n:
        return []
    spacing = geometry.spacing_mm
    sigma = np.array([FWHM_MM[geometry.scanner] / 2.3548 / s for s in spacing])
    margin = int(np.ceil(4 * sigma.max())) + 2
    pad = [int(np.ceil(RING_MM[1] / s)) + 1 for s in spacing]
    rows = []
    hot = lesions > 0
    for j, box in enumerate(ndi.find_objects(lesions, max_label=n)):
        if box is None:
            continue
        reach = [max(m, p) for m, p in zip([margin] * 3, pad)]
        big = tuple(slice(max(b.start - m, 0), min(b.stop + m, size)) for b, m, size in zip(box, reach, lesions.shape))
        mask = lesions[big] == j + 1
        layer = ndi.binary_dilation(mask) & ~mask
        if not layer.any() or (hot[big] & layer).any():
            continue
        blur = ndi.gaussian_filter(mask.astype(np.float32), sigma)
        background = ring_median(pet[big], mask, spacing, hot[big])
        inside, outside = float(pet[big][mask].mean()), float(pet[big][layer].mean())
        if inside <= background or outside <= background:
            continue
        rows.append(dict(study_id=study_id, lesion=j, volume_ml=geometry.lesions_ml[j], rc=geometry.rc_lesions[j], scanner=geometry.scanner, observed=(outside - background) / (inside - background), expected=float(blur[layer].mean() / blur[mask].mean())))
    return rows
