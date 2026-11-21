"""
Tables of structure predictions without torch (usable in the local environment): SD variants, variance scales and
their cross-validation, comparison of methods through net_data.evaluate, side counts of the evaluation.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

QUANTITIES = ("log_a0", "log_lambda_bio", "log_lambda_u", "log_tiac", "log_tiac_late", "early_share")
KEYS = ["arm", "virtual_id", "source"]
HALO_MM = 10.0
HALO_SOURCES = ("kidneys_functional", "salivary_glands", "bladder")


def flattened_voxels(nd, code, lesion, grid_mm):
    """
    Rest-of-body voxels whose prediction is replaced by the median of the others: within HALO_MM of kidneys, glands or
    bladder, or with a 6-neighbour of another source (or of the outside). NumPy, from the labels only.
    """
    rest = code == nd.COMPARTMENT_CODES["rest_of_body"]
    halo = np.isin(code, [nd.COMPARTMENT_CODES[s] for s in HALO_SOURCES])
    near = ndi.distance_transform_edt(~halo, sampling=grid_mm) <= HALO_MM if halo.any() else np.zeros(code.shape, bool)
    label = np.where(lesion > 0, 100 + lesion, code)
    edge = np.zeros(code.shape, bool)
    for axis in range(3):
        a, b = [slice(None)] * 3, [slice(None)] * 3
        a[axis], b[axis] = slice(1, None), slice(None, -1)
        different = label[tuple(a)] != label[tuple(b)]
        edge[tuple(a)] |= different
        edge[tuple(b)] |= different
    return rest & (near | edge)


def with_sd(prediction, suffix="sd"):
    """Prediction table for net_data.evaluate with <q>_sd taken from <q>_<suffix>."""
    out = prediction[KEYS + ["study_id", "structure"]].copy()
    for q in QUANTITIES:
        out[f"{q}_mean"], out[f"{q}_sd"] = prediction[f"{q}_mean"], prediction[f"{q}_{suffix}"]
    return out


def fit_scales(prediction, truth):
    """
    Closed-form NLL-optimal SD factor per quantity and structure group (each compartment, lesions together):
    sqrt(mean z^2), z = (truth - mean) / sd. log_lambda_u only on lambda_u targets.
    """
    truth = truth.assign(log_lambda_u=truth.log_lambda_u.where(truth.lu_target))
    merged = truth[KEYS + ["structure", *QUANTITIES]].merge(prediction.drop(columns=["structure"]), on=KEYS)
    rows = []
    for structure, g in merged.groupby("structure"):
        for q in QUANTITIES:
            z = ((g[q] - g[f"{q}_mean"]) / g[f"{q}_sd"]).to_numpy(float)
            z = z[np.isfinite(z)]
            if len(z):
                rows.append(dict(structure=structure, quantity=q, n=len(z), scale=float(np.sqrt(np.mean(z ** 2))),
                                 mean_z=float(z.mean())))
    return pd.DataFrame(rows)


def apply_scales(prediction, scales):
    """Multiplies <q>_sd by the scale of its structure group; groups without a scale are left unchanged."""
    out = prediction.copy()
    for q in QUANTITIES:
        factor = out.structure.map(scales[scales.quantity == q].set_index("structure").scale).fillna(1.0)
        out[f"{q}_sd"] = out[f"{q}_sd"] * factor.to_numpy()
    return out


def subject_folds(phantoms, studies, folds=5, seed=20261114):
    """Fold of every study, by subject: subjects in sorted order, shuffled with the seed, dealt in turn."""
    table = phantoms[phantoms.study_id.isin(studies)]
    subjects = np.array(sorted(table.subject.unique()))
    np.random.default_rng(seed).shuffle(subjects)
    fold = {s: k % folds for k, s in enumerate(subjects)}
    return dict(zip(table.study_id, table.subject.map(fold)))


def cross_validated(prediction, truth, fold_of_study):
    """Every fold scaled with the scales fitted on the other folds; returns the scaled table and the scales of each fold."""
    fold = prediction.study_id.map(fold_of_study)
    truth_fold = truth.study_id.map(fold_of_study)
    parts, fitted = [], []
    for k in sorted(fold.unique()):
        scales = fit_scales(prediction[fold != k], truth[truth_fold != k])
        parts.append(apply_scales(prediction[fold == k], scales))
        fitted.append(scales.assign(held_out_fold=k))
    return pd.concat(parts, ignore_index=True), pd.concat(fitted, ignore_index=True)


def compare_methods(nd, methods, scenario, reference="population"):
    """
    net_data.evaluate by structure for every method {name: (prediction, truth)}; adds the error as a factor
    (exp(median |log error|) - 1, %; not for the early share) and skill = 1 - MAE / MAE of the reference method.
    """
    tables = []
    for name, (prediction, truth) in methods.items():
        table = nd.evaluate(prediction, truth, by=("structure",), scenarios=(scenario,))
        tables.append(table.assign(method=name))
    out = pd.concat(tables, ignore_index=True)
    out["error_factor_pct"] = np.where(out.quantity == "early_share", np.nan, 100 * np.expm1(out.median_abs_error))
    base = out[out.method == reference].set_index(["structure", "quantity"]).median_abs_error
    out["skill"] = 1 - out.median_abs_error / base.reindex(pd.MultiIndex.from_frame(out[["structure", "quantity"]])).to_numpy()
    return out[["method", "structure", "quantity", "n", "bias", "median_abs_error", "error_factor_pct", "rmse",
                "coverage_68", "coverage_95", "skill"]]


def washout_check(comparison, network="network", pbpk="pbpk_fit", margin=0.05, ceiling=None, rho=0.0):
    """
    Skill on log lambda_bio of the network and of the PBPK reference and their difference, by structure. In scenario A
    the washout is drawn independently of the PET: alarm where the network skill exceeds max(PBPK skill, 0) + margin.
    In a B scenario (ceiling: the method holding the other draws of the same phantom in that scenario) the network
    should have skill: skill_ceiling is what a predictor that knows the phantom perfectly reaches (estimated from 9
    draws), skill_gaussian_bound = 1 - sqrt(1 - rho^2) what the uptake of the draw itself would give for a Gaussian
    variable (structures whose washout follows one release rate), share_of_ceiling the network skill over the ceiling;
    the alarm is then a network skill above the ceiling + margin (more than the phantom can tell).
    """
    d = comparison[comparison.quantity == "log_lambda_bio"].pivot(index="structure", columns="method", values="skill")
    out = d[[network, pbpk]].rename(columns={network: "skill_network", pbpk: "skill_pbpk_fit"}).reset_index()
    out["network_minus_pbpk_fit"] = out.skill_network - out.skill_pbpk_fit
    if ceiling is None or ceiling not in d:
        out["alarm"] = out.skill_network > np.maximum(out.skill_pbpk_fit, 0.0) + margin
        return out
    out["skill_ceiling"] = d[ceiling].to_numpy()
    out["skill_gaussian_bound"] = 1.0 - np.sqrt(1.0 - rho ** 2)
    out["share_of_ceiling"] = np.where(out.skill_ceiling > margin, out.skill_network / out.skill_ceiling, np.nan)
    out["alarm"] = out.skill_network > np.maximum(out.skill_ceiling, 0.0) + margin
    return out


def lesion_overlap(prediction):
    """Number and share of lesions whose dilated mask holds voxels of another lesion, overall and per phantom."""
    lesions = prediction[prediction.structure == "lesion"].drop_duplicates(["study_id", "source"])
    rows = [dict(study_id="all", lesions=len(lesions), with_other_lesion=int(lesions.mask_has_other_lesion.sum()))]
    for study, g in lesions.groupby("study_id"):
        rows.append(dict(study_id=study, lesions=len(g), with_other_lesion=int(g.mask_has_other_lesion.sum())))
    out = pd.DataFrame(rows)
    out["share"] = out.with_other_lesion / out.lesions.clip(lower=1)
    return out


def early_share_cap(prediction):
    """
    Per structure group: share of the voxels where the mean early share is above the cap (voxel-weighted over the
    structures), number of structures with any such voxel, and the largest share within one structure.
    """
    d = prediction.drop_duplicates(["study_id", "source"])
    rows = []
    for structure, g in d.groupby("structure"):
        rows.append(dict(structure=structure, structures=len(g), voxels=int(g.n_voxels.sum()),
                         share_of_voxels_capped=float((g.early_share_capped * g.n_voxels).sum() / g.n_voxels.sum()),
                         structures_with_capped_voxels=int((g.early_share_capped > 0).sum()),
                         largest_share_in_one_structure=float(g.early_share_capped.max())))
    return pd.DataFrame(rows)


def side_by_side(comparison, column, methods=None, quantities=("log_a0", "log_lambda_bio", "log_tiac")):
    """One column of the comparison table with the methods side by side, rows (quantity, structure)."""
    d = comparison[comparison.quantity.isin(quantities)]
    if methods is not None:
        d = d[d.method.isin(methods)]
    return d.pivot(index=["quantity", "structure"], columns="method", values=column)
