"""Training data for the network (notebook 22): phantoms on a common grid, source weights of the Monte Carlo code, per-draw kinetic parameters, rebuild of the voxel targets.

Spatial weights come from psma_mc.source_weights_v2 and psma_mc.lesion_weights (the sources of the full Monte Carlo run), never recomputed here.
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import sparse
from scipy import stats

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))
try:
    import psma_mc as M  # noqa: E402
except ImportError:
    M = None

PROCESSED = WORKSPACE / "data" / "processed" / "psma"
COHORT = PROCESSED / "cohort"
NETDATA = PROCESSED / "netdata"
LAMBDA_PHYS_H = float(np.log(2) / (6.647 * 24))
SOURCE_COMPARTMENTS = ["kidneys_functional", "liver", "spleen", "salivary_glands", "bladder", "gi_tract", "blood_pool", "spine", "rest_of_body"]
assert M is None or SOURCE_COMPARTMENTS == [c for c in M.COMPARTMENTS_V2 if c != "lesions"]
LESION_CODE = len(SOURCE_COMPARTMENTS) + 1
COMPARTMENT_CODES = {**{c: i + 1 for i, c in enumerate(SOURCE_COMPARTMENTS)}, "lesion": LESION_CODE}
PET_REF = 1e-3
CT_CLIP = (-1000.0, 2000.0)
CT_SCALE = 1000.0
BODY_FRACTION = 0.5
CROP_MARGIN = 1
LAMBDA_BIO_FLOOR = 1e-5
SCENARIOS = ("A", "B-0.6", "B-0.3", "B+0.3", "B+0.6")
CURVE_TOLERANCE = 0.05
EARLY_DOMINANT = 0.20
TARGET_KINDS = ("curve", "late_early", "tia")


def axis_overlap(n, s, g, lo, hi):
    """Overlap lengths (mm) between native voxels lo..hi-1 of size s and the target voxels of size g that cover them, target grid anchored at the native corner."""
    j0, j1 = int(np.floor(lo * s / g)), int(np.ceil(hi * s / g))
    rows, cols, vals = [], [], []
    for i in range(lo, hi):
        a, b = i * s, (i + 1) * s
        for j in range(int(np.floor(a / g)), int(np.ceil(b / g))):
            o = min(b, (j + 1) * g) - max(a, j * g)
            if o > 1e-9:
                rows.append(j - j0)
                cols.append(i - lo)
                vals.append(o)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(j1 - j0, hi - lo)), j0


class Resampler:
    """Separable overlap resampling of a native box onto an isotropic grid: 'sum' conserves totals, 'mean' is the volume-weighted average."""

    def __init__(self, shape, spacing, grid_mm, box):
        self.spacing, self.grid_mm, self.box = tuple(spacing), float(grid_mm), box
        mats = [axis_overlap(n, s, grid_mm, b.start, b.stop) for n, s, b in zip(shape, spacing, box)]
        self.offset = tuple(j0 for _, j0 in mats)
        self.sum_ops = [sparse.csr_matrix(m.multiply(1 / s)) for (m, _), s in zip(mats, spacing)]
        self.mean_ops = [sparse.csr_matrix(sparse.diags(1 / np.asarray(m.sum(axis=1)).ravel()) @ m) for m, _ in mats]
        self.shape = tuple(m.shape[0] for m, _ in mats)

    @staticmethod
    def _apply(array, ops):
        out = np.asarray(array, dtype=np.float64)
        for axis, op in enumerate(ops):
            moved = np.moveaxis(out, axis, 0)
            rest = moved.shape[1:]
            out = np.moveaxis((op @ moved.reshape(moved.shape[0], -1)).reshape((op.shape[0],) + rest), 0, axis)
        return out

    def sum(self, array):
        return self._apply(array[self.box], self.sum_ops)

    def mean(self, array):
        return self._apply(array[self.box], self.mean_ops)

    def sub(self, local_box):
        """Resampler of a sub-box (native slices inside self.box), with its offset in the target grid of self."""
        child = Resampler.__new__(Resampler)
        child.spacing, child.grid_mm = self.spacing, self.grid_mm
        child.box = tuple(slice(b.start + l.start, b.start + l.stop) for b, l in zip(self.box, local_box))
        mats = [axis_overlap(None, s, self.grid_mm, b.start, b.stop) for s, b in zip(self.spacing, child.box)]
        child.offset = tuple(j0 - o for (_, j0), o in zip(mats, self.offset))
        child.sum_ops = [sparse.csr_matrix(m.multiply(1 / s)) for (m, _), s in zip(mats, self.spacing)]
        child.shape = tuple(m.shape[0] for m, _ in mats)
        return child


def body_box(organs, spacing, grid_mm):
    """Native slices of the body bounding box, widened so that it starts and ends on target voxel edges, plus CROP_MARGIN target voxels."""
    idx = np.nonzero(organs > 0)
    box = []
    for axis, (n, s) in enumerate(zip(organs.shape, spacing)):
        lo_mm = max(idx[axis].min() * s - CROP_MARGIN * grid_mm, 0.0)
        hi_mm = min((idx[axis].max() + 1) * s + CROP_MARGIN * grid_mm, n * s)
        lo_mm = np.floor(lo_mm / grid_mm) * grid_mm
        hi_mm = min(np.ceil(hi_mm / grid_mm) * grid_mm, n * s)
        box.append(slice(int(np.floor(lo_mm / s + 1e-9)), int(np.ceil(hi_mm / s - 1e-9))))
    return tuple(box)


class _CachedLoad:
    """Reads each phantom image once while the Monte Carlo weight functions run (they reload organs and PET at every call); arrays are only read."""

    def __enter__(self):
        self.original, self.cache = M.load, {}
        M.load = lambda study_id, name: self.cache.setdefault((study_id, name), self.original(study_id, name))
        return self

    def __exit__(self, *args):
        M.load = self.original


def native_sources(study_id):
    """Source map on the native grid: compartment code, lesion id (lesions.nii.gz, cohort lesion j = id j + 1) and unit-sum weight of each source, from the Monte Carlo code."""
    with _CachedLoad():
        organs = M.load(study_id, "organs")
        lesions = M.load(study_id, "lesions").astype(np.int32)
        code = np.zeros(organs.shape, np.uint8)
        weight = np.zeros(organs.shape, np.float64)
        status = {}
        for c in SOURCE_COMPARTMENTS:
            mask = np.isin(organs, M.COMPARTMENTS_V2[c])
            code[mask] = COMPARTMENT_CODES[c]
            w = M.source_weights_v2(study_id, c, "pet") if mask.any() else None
            if w is not None:
                weight[mask] = w[mask]
            status[c] = "ok" if w is not None else ("absent" if not mask.any() else "zero_pet")
        code[organs == 15] = LESION_CODE
        per_lesion = M.lesion_weights(study_id)
    lesion_status = {}
    for i in range(1, int(lesions.max()) + 1):
        if i in per_lesion:
            idx, w = per_lesion[i]
            weight[idx] = w
            lesion_status[i] = "ok"
        else:
            mask = lesions == i
            weight[mask] = 1.0 / mask.sum()
            lesion_status[i] = "uniform_fallback"
    return organs, lesions, code, weight, status, lesion_status


RULES = ("hybrid", "volume", "activity")


def _label_sources(res, organs, lesions, code, weight, pet):
    """Target-grid labels under three rules. A voxel is body if at least BODY_FRACTION of its volume is body. Inside it: 'volume' = source with the largest volume, 'activity' = largest PET activity, 'hybrid' = volume, except where the activity winner is a lesion. Returns {rule: (code, lesion id)} and the resampled weights of each source."""
    voxel_ratio = np.prod(res.spacing) / res.grid_mm ** 3
    body = res.sum((organs > 0).astype(np.float32)) * voxel_ratio >= BODY_FRACTION
    best = {r: np.full(res.shape, -1.0) for r in ("volume", "activity")}
    out = {r: (np.zeros(res.shape, np.uint8), np.zeros(res.shape, np.int32)) for r in ("volume", "activity")}
    resampled = {}
    for c in SOURCE_COMPARTMENTS:
        mask = code == COMPARTMENT_CODES[c]
        if not mask.any():
            continue
        resampled[(COMPARTMENT_CODES[c], 0)] = res.sum(np.where(mask, weight, 0.0))
        for r, score in (("volume", res.sum(mask.astype(np.float32))), ("activity", res.sum(np.where(mask, pet, 0.0)))):
            better = score > best[r]
            best[r][better], out[r][0][better] = score[better], COMPARTMENT_CODES[c]
    for i, box in enumerate(ndi.find_objects(lesions[res.box]), start=1):
        if box is None:
            continue
        sub = res.sub(box)
        mask = lesions[sub.box] == i
        target = tuple(slice(o, o + n) for o, n in zip(sub.offset, sub.shape))
        resampled[(LESION_CODE, i)] = (target, sub._apply(np.where(mask, weight[sub.box], 0.0), sub.sum_ops))
        for r, score in (("volume", sub._apply(mask.astype(np.float32), sub.sum_ops)), ("activity", sub._apply(np.where(mask, pet[sub.box], 0.0), sub.sum_ops))):
            region = best[r][target]
            better = score > region
            region[better] = score[better]
            out[r][0][target][better] = LESION_CODE
            out[r][1][target][better] = i
    lesion_wins = out["activity"][0] == LESION_CODE
    out["hybrid"] = (np.where(lesion_wins, LESION_CODE, out["volume"][0]).astype(np.uint8), np.where(lesion_wins, out["activity"][1], out["volume"][1]).astype(np.int32))
    rescued = {}
    for r in out:
        out[r][0][~body] = 0
        out[r][1][~body] = 0
        rescued[r] = _rescue_lesions(out[r], resampled, body)
    return out, resampled, rescued


def _rescue_lesions(labels, resampled, body):
    """A lesion that wins no voxel gets its voxel of largest weight inside the body, so no source vanishes; returns the ids rescued."""
    code, lesion = labels
    present = set(np.unique(lesion).tolist())
    rescued = []
    for (c, i), value in resampled.items():
        if c != LESION_CODE or i in present:
            continue
        target, w_l = value
        w_in = np.where(body[target], w_l, -1.0)
        if w_in.max() <= 0:
            continue
        k = np.unravel_index(np.argmax(w_in), w_in.shape)
        code[target][k], lesion[target][k] = LESION_CODE, i
        rescued.append(i)
    return rescued


def _retention(resampled, out_code, out_lesion, lesions, code, status, lesion_status, spacing, grid_mm, study_id, rule, relative=None, rescued=()):
    voxel_ml = grid_mm ** 3 / 1000
    native_ml = float(np.prod(spacing)) / 1000
    native_counts = np.bincount(lesions.ravel())
    rows = []
    for (c, i), value in resampled.items():
        if c == LESION_CODE:
            target, w_l = value
            own = (out_code[target] == c) & (out_lesion[target] == i)
            retained, n = float(w_l[own].sum()), int(own.sum())
            if relative is not None and n:
                relative[target][own] = w_l[own] / retained * n
            name, native_n, src_status = f"lesion_{i - 1}", int(native_counts[i]), lesion_status[i]
        else:
            own = (out_code == c) & (out_lesion == 0)
            retained, n = float(value[own].sum()), int(own.sum())
            if relative is not None and n and retained > 0:
                relative[own] = value[own] / retained * n
            name = SOURCE_COMPARTMENTS[c - 1]
            native_n, src_status = int((code == c).sum()), status[name]
        rows.append(dict(study_id=study_id, rule=rule, source=name, code=c, lesion_id=i, n_voxels=n, volume_ml=n * voxel_ml, native_volume_ml=native_n * native_ml, retained=retained, vanished=n == 0, rescued=c == LESION_CODE and i in rescued, status=src_status))
    return pd.DataFrame(rows)


def prepare(study_id, grid_mm, rule="hybrid", native=None):
    """One phantom on the target grid: inputs, labels, relative weights and the per-source table (retained = share of the source weight in voxels labelled as that source, before renormalisation). Retention is also returned for the other rules."""
    organs, lesions, code, weight, status, lesion_status = native if native is not None else native_sources(study_id)
    spacing = M.spacing_of(study_id)
    pet = M.load(study_id, "pet_pct_ia_ml").astype(np.float64)
    ct = M.load(study_id, "ct").astype(np.float64)
    res = Resampler(organs.shape, spacing, grid_mm, body_box(organs, spacing, grid_mm))
    labels, resampled, rescued = _label_sources(res, organs, lesions, code, weight, pet)
    relative = np.zeros(res.shape, np.float64)
    others = pd.concat([_retention(resampled, *labels[r], lesions, code, status, lesion_status, spacing, grid_mm, study_id, r, rescued=rescued[r]) for r in RULES if r != rule])
    sources = _retention(resampled, *labels[rule], lesions, code, status, lesion_status, spacing, grid_mm, study_id, rule, relative, rescued=rescued[rule])
    out_code, out_lesion = labels[rule]
    organs_grid = _majority(res, organs)
    organs_grid[out_code == 0] = 0
    arrays = dict(
        pet=np.log1p(np.maximum(res.mean(pet), 0.0) / PET_REF).astype(np.float16),
        ct=(np.clip(res.mean(ct), *CT_CLIP) / CT_SCALE).astype(np.float16),
        compartment=out_code, lesion=out_lesion.astype(np.uint16), organs=organs_grid,
        weight=relative.astype(np.float16),
    )
    rel16 = arrays["weight"].astype(np.float64)
    sums = _source_sums(rel16, out_code, out_lesion)
    sources["weight_sum"] = [sums.get((c, i), 0.0) for c, i in zip(sources.code, sources.lesion_id)]
    native_box = [(b.start, b.stop) for b in res.box]
    meta = dict(study_id=study_id, grid_mm=grid_mm, rule=rule, n_lesions=int(lesions.max()), shape=list(res.shape), offset_vox=list(res.offset), native_spacing_mm=list(spacing), native_box=native_box, affine=_affine(study_id, grid_mm, res.offset).tolist())
    exact = dict(pet=np.maximum(res.mean(pet), 0.0), ct=np.clip(res.mean(ct), *CT_CLIP), weight=relative)
    return arrays, sources, meta, exact, others


def _majority(res, labels):
    best = np.full(res.shape, -1.0)
    out = np.zeros(res.shape, np.uint8)
    for value in np.unique(labels):
        score = res.sum((labels == value).astype(np.float32))
        better = score > best
        best[better], out[better] = score[better], value
    return out


def _source_sums(relative, code, lesion):
    key = code.astype(np.int64) * 100000 + lesion.astype(np.int64)
    uniq, inverse = np.unique(key.ravel(), return_inverse=True)
    totals = np.bincount(inverse, weights=relative.ravel())
    return {(int(k // 100000), int(k % 100000)): float(t) for k, t in zip(uniq, totals) if k // 100000 > 0}


def _affine(study_id, grid_mm, offset):
    import nibabel as nib
    native = nib.load(M.ph.PHANTOMS / study_id / "ct.nii.gz").affine
    zooms = np.sqrt((native[:3, :3] ** 2).sum(axis=0))
    direction = native[:3, :3] / zooms
    affine = np.eye(4)
    affine[:3, :3] = direction * grid_mm
    corner = native[:3, 3] - direction @ (zooms / 2)
    affine[:3, 3] = corner + direction @ ((np.asarray(offset) + 0.5) * grid_mm)
    return affine


def kinetic_table(split):
    """Cohort TIA table of one split (train or test) with the network parametrisation and the phantom id."""
    tia = pd.read_parquet(COHORT / f"tia_{split}.parquet")
    out = parameter_table(tia)
    out["study_id"] = out.virtual_id.str.rsplit("_d", n=1).str[0]
    out["draw"] = out.virtual_id.str.rsplit("_d", n=1).str[1].astype(int)
    out["structure"] = np.where(out.source.str.startswith("lesion_"), "lesion", out.source)
    return out


def structure_volumes(sources):
    """Native volume (mL) of each source of each phantom, from the cache tables; the volume of the Monte Carlo source mask."""
    return sources.drop_duplicates(["study_id", "source"]).set_index(["study_id", "source"]).native_volume_ml


def mark_sources(params, sources):
    """has_source: the source has voxels on the grid. False where the cohort gives a TIAC to a source the phantom does not hold (psma_3959d1c381a5bcd6_2015-11-12 bladder, cut off by the PET crop, no Monte Carlo source either): excluded from the rebuild and its checks."""
    present = set(zip(sources.study_id[sources.n_voxels > 0], sources.source[sources.n_voxels > 0]))
    out = params.copy()
    out["has_source"] = [(s, c) in present for s, c in zip(out.study_id, out.source)]
    return out


PRIOR_COLUMNS = ("log_a0_per_ml", "log_lambda_bio", "log_lambda_u", "log_tiac_per_ml", "log_tiac_late_per_ml", "early_share")
EVAL_QUANTITIES = ("log_a0", "log_lambda_bio", "log_lambda_u", "log_tiac", "log_tiac_late", "early_share")
PER_ML = {"log_a0": "log_a0_per_ml", "log_tiac": "log_tiac_per_ml", "log_tiac_late": "log_tiac_late_per_ml"}


def population_prior(params, volumes, train_studies):
    """Population prior per arm and structure, scenario A of the training phantoms only: mean and SD of log A0, log TIAC and log late TIAC per mL, log lambda_bio, log lambda_u (lambda_u targets only), early share; share of degenerate lambda_u."""
    d = params[(params.scenario == "A") & params.study_id.isin(train_studies) & params.has_source].copy()
    log_v = np.log(volumes.reindex(pd.MultiIndex.from_arrays([d.study_id, d.source])).to_numpy())
    for q, c in PER_ML.items():
        d[c] = d[q] - log_v
    d["log_lambda_u"] = d.log_lambda_u.where(d.lu_target)
    rows = []
    for (arm, structure), g in d.groupby(["arm", "structure"]):
        row = dict(arm=arm, structure=structure, target_kind=g.target_kind.iloc[0], n=len(g), n_phantoms=g.study_id.nunique(), p_degenerate=float(g.lu_degenerate.mean()))
        for c in PRIOR_COLUMNS:
            x = g[c].dropna()
            row[f"{c}_mean"], row[f"{c}_sd"] = (float(x.mean()), float(x.std())) if len(x) > 1 else (np.nan, np.nan)
            row[f"{c}_n"] = len(x)
        rows.append(row)
    return pd.DataFrame(rows)


def population_prediction(params, volumes, prior):
    """The population prior as a prediction for every virtual patient and source, same keys and columns as the PBPK baseline."""
    keys = params[["split", "arm", "virtual_id", "study_id", "draw", "source", "structure"]].drop_duplicates()
    out = keys.merge(prior, on=["arm", "structure"], how="left")
    log_v = np.log(volumes.reindex(pd.MultiIndex.from_arrays([out.study_id, out.source])).to_numpy())
    for q, c in PER_ML.items():
        out[f"{q}_mean"], out[f"{q}_sd"], out[f"{q}_n"] = out[f"{c}_mean"] + log_v, out[f"{c}_sd"], out[f"{c}_n"]
    keep = ["split", "arm", "virtual_id", "study_id", "draw", "source", "structure", "target_kind", "p_degenerate"] + [f"{q}_{m}" for q in EVAL_QUANTITIES for m in ("mean", "sd", "n")]
    return out[keep].assign(baseline="population")


def pbpk_leave_one_out(params):
    """PBPK-fit prediction from the real PET: the posterior predictive of the phantom, i.e. the scenario A draws of the main arm of the same phantom, leaving out the target draw (it shares the posterior draw of the densities). Same prediction for every scenario of that draw."""
    d = params[(params.arm == "main") & (params.scenario == "A") & params.has_source]
    rows = []
    for (study, source), g in d.groupby(["study_id", "source"], sort=False):
        g = g.sort_values("draw")
        values = {q: g[q].where(g.lu_target).to_numpy() if q == "log_lambda_u" else g[q].to_numpy() for q in EVAL_QUANTITIES}
        degenerate = g.lu_degenerate.to_numpy()
        for k, (vid, draw) in enumerate(zip(g.virtual_id, g.draw)):
            others = np.arange(len(g)) != k
            row = dict(split=g.split.iloc[0], arm="main", virtual_id=vid, study_id=study, draw=draw, source=source, structure=g.structure.iloc[0], target_kind=g.target_kind.iloc[0], n_draws=int(others.sum()), p_degenerate=float(degenerate[others].mean()))
            for name, x in values.items():
                x = x[others]
                x = x[np.isfinite(x)]
                row[f"{name}_mean"], row[f"{name}_sd"] = (float(x.mean()), float(x.std(ddof=1))) if len(x) > 1 else (np.nan, np.nan)
                row[f"{name}_n"] = len(x)
            rows.append(row)
    return pd.DataFrame(rows).assign(baseline="pbpk_fit")


Z68, Z95 = 0.9945, 1.96


def predictive_halfwidth(sd, n, level):
    """Half-width of the central predictive interval for a new draw. With n (number of draws the mean and SD were estimated from): Student t with n - 1 degrees of freedom, scale sd sqrt(1 + 1/n). Without n (NaN): Gaussian, sd times the normal quantile (a method that predicts a distribution directly)."""
    sd, n = np.asarray(sd, float), np.asarray(n, float)
    z = stats.norm.ppf(0.5 + level / 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(n > 1, stats.t.ppf(0.5 + level / 2, np.maximum(n - 1, 1)) * np.sqrt(1 + 1 / np.maximum(n, 1)), z)
    return sd * np.where(np.isfinite(n), t, z)


def evaluate(prediction, params, by=("structure",), scenarios=None):
    """Error of a prediction table (columns <quantity>_mean, <quantity>_sd and, for methods that estimate mean and SD from draws, <quantity>_n; keyed by arm, virtual_id, source) against the cohort parameters, for every scenario of each draw. Coverage from predictive_halfwidth: Student t when <quantity>_n is given, Gaussian otherwise. The same function serves the network and the baselines. Logs except early_share (linear); log_lambda_u only on lambda_u targets where both are finite; sources without voxels left out."""
    truth = params[params.has_source] if scenarios is None else params[params.has_source & params.scenario.isin(scenarios)]
    truth = truth.assign(log_lambda_u=truth.log_lambda_u.where(truth.lu_target))
    keep = ["arm", "virtual_id", "source"] + [c for q in EVAL_QUANTITIES for c in (f"{q}_mean", f"{q}_sd", f"{q}_n") if c in prediction]
    merged = truth.merge(prediction[keep], on=["arm", "virtual_id", "source"], how="inner")
    rows = []
    for key, g in merged.groupby(list(by)):
        for q in EVAL_QUANTITIES:
            if f"{q}_mean" not in g:
                continue
            ok = np.isfinite(g[q]) & np.isfinite(g[f"{q}_mean"])
            if not ok.any():
                continue
            err = (g[f"{q}_mean"] - g[q])[ok]
            sd = g[f"{q}_sd"][ok].to_numpy()
            n = g[f"{q}_n"][ok].to_numpy() if f"{q}_n" in g else np.full(len(sd), np.nan)
            cover = {lv: float((err.abs().to_numpy() < predictive_halfwidth(sd, n, lv)).mean()) for lv in (0.68, 0.95)}
            rows.append(dict(zip(by, key if isinstance(key, tuple) else (key,)), quantity=q, n=int(ok.sum()), bias=float(err.mean()), median_abs_error=float(err.abs().median()), rmse=float(np.sqrt((err ** 2).mean())), coverage_68=cover[0.68], coverage_95=cover[0.95]))
    return pd.DataFrame(rows)


def parameter_table(tia):
    """Network parametrisation of the cohort kinetics: log A0, log lambda_bio (floored), log lambda_u where finite, late TIA A0/lambda_w and early TIA = TIAC - late (signed), and the error of the uptake-washout curve against the TIAC."""
    out = tia.copy()
    out["source"] = out.compartment
    out["lambda_bio"] = out.lw - LAMBDA_PHYS_H
    out["lambda_bio_floored"] = out.lambda_bio < LAMBDA_BIO_FLOOR
    out["log_a0"] = np.log(out.a0)
    out["log_lambda_bio"] = np.log(np.maximum(out.lambda_bio, LAMBDA_BIO_FLOOR))
    out["lu_degenerate"] = ~np.isfinite(out.lu)
    out["log_lambda_u"] = np.where(out.lu_degenerate, np.nan, np.log(out.lu.where(~out.lu_degenerate, 1.0)))
    out["tiac_curve_h"] = out.a0 * (1 / out.lw - np.where(out.lu_degenerate, 0.0, 1 / out.lu.where(~out.lu_degenerate, 1.0)))
    out["curve_error"] = out.tiac_curve_h / out.tiac_h - 1
    out["log_tiac"] = np.log(out.tiac_h)
    out["log_tiac_late"] = np.log(out.tiac_late_h)
    out["early_share"] = out.tiac_early_h / out.tiac_h
    return out


def curve_validity(params):
    """Median |TIA of the uptake-washout curve / TIAC - 1| per structure, main arm, scenario A, and the target kind it implies: 'curve' (A0, lambda_bio, lambda_u, early) at most CURVE_TOLERANCE, 'late_early' (A0, lambda_bio, early) up to EARLY_DOMINANT, 'tia' (TIAC, early, late) above."""
    d = params[(params.arm == "main") & (params.scenario == "A")]
    error = d.curve_error.abs().groupby(d.structure).median()
    kind = np.where(error <= CURVE_TOLERANCE, "curve", np.where(error <= EARLY_DOMINANT, "late_early", "tia"))
    return pd.DataFrame({"median_abs_curve_error": error, "target_kind": kind})


def add_target_kind(params, validity):
    """target_kind per row, and lu_target: lambda_u is a target only for 'curve' structures and only where finite."""
    out = params.copy()
    out["target_kind"] = out.structure.map(validity.target_kind)
    out["lu_target"] = (out.target_kind == "curve") & ~out.lu_degenerate
    return out


def parameter_arrays(params, n_lesions, xp=np):
    """Lookup tables of one draw, indexed by compartment code (0..LESION_CODE) and by lesion id (0..n_lesions): a0, lambda_bio, lambda_u (inf if degenerate), early TIA (signed)."""
    comp = np.zeros((LESION_CODE + 1, 4))
    les = np.zeros((n_lesions + 1, 4))
    comp[:, 2] = les[:, 2] = np.inf
    for row in params.itertuples(index=False):
        values = (row.a0, row.lw - LAMBDA_PHYS_H, row.lu, row.tiac_early_h)
        if row.source.startswith("lesion_"):
            les[int(row.source.split("_")[1]) + 1] = values
        else:
            comp[COMPARTMENT_CODES[row.source]] = values
    return xp.asarray(comp), xp.asarray(les)


def rebuild(compartment, lesion, weight, weight_sum_comp, weight_sum_lesion, comp_table, lesion_table, xp=np):
    """Voxel targets from labels, relative weights and the lookup tables of one draw. Works on NumPy arrays or torch tensors (xp = numpy or torch; indices must be integer arrays/long tensors).
    weight_sum_*: sum of the relative weight of each source over the whole phantom (same indexing as the tables). Returns w (unit-sum per source), A0 (fraction of injected activity per voxel), lambda_bio and lambda_u (1/h), degenerate mask, late TIA a0/lambda_w, early TIA (signed), TIA = late + early (h, as a fraction of injected activity)."""
    is_lesion = lesion > 0
    comp_rows, lesion_rows = comp_table[compartment], lesion_table[lesion]
    rows = xp.where(is_lesion[..., None], lesion_rows, comp_rows)
    norm = xp.where(is_lesion, weight_sum_lesion[lesion], weight_sum_comp[compartment])
    inside = (compartment > 0) & (norm > 0)
    w = xp.where(inside, weight / xp.where(inside, norm, 1.0), 0.0)
    a0, lambda_bio, lambda_u, early = rows[..., 0] * w, rows[..., 1], rows[..., 2], rows[..., 3] * w
    late = xp.where(inside, a0 / (lambda_bio + LAMBDA_PHYS_H), 0.0)
    early = xp.where(inside, early, 0.0)
    degenerate = ~xp.isfinite(lambda_u)
    return dict(w=w, a0=a0, lambda_bio=lambda_bio, lambda_u=lambda_u, degenerate=degenerate & inside, tia_late=late, tia_early=early, tia=late + early, inside=inside)


def source_sum_tables(sources, n_lesions):
    comp = np.zeros(LESION_CODE + 1)
    les = np.zeros(n_lesions + 1)
    for row in sources.itertuples(index=False):
        if row.code == LESION_CODE:
            les[row.lesion_id] = row.weight_sum
        else:
            comp[row.code] = row.weight_sum
    return comp, les


def write_shard(path, phantoms):
    """phantoms: {study_id: {field: array}}; one npz, keys '<study_id>/<field>', stored uncompressed or deflated per member."""
    path = Path(path)
    tmp = path.with_suffix(".tmp.npz")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for study, fields in phantoms.items():
            for name, array in fields.items():
                with z.open(f"{study}/{name}.npy", "w", force_zip64=True) as handle:
                    np.lib.format.write_array(handle, np.ascontiguousarray(array), allow_pickle=False)
    tmp.replace(path)


def read_shard(path, studies=None):
    """{study_id: {field: array}} from one shard."""
    out = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            study, name = key.split("/")
            if studies is None or study in studies:
                out.setdefault(study, {})[name] = data[key]
    return out


NB08_TLV_EDGES_ML = (5.0, 35.8)


def split_validation(phantoms, fraction=0.15, seed=20261110):

    """Validation subjects out of the training phantoms, stratified as nb08: lesion-free or tertile of the subject median TLV (nb08 edges), single or multi study."""
    inventory = pd.read_csv(PROCESSED / "psma_inventory.csv", usecols=["study_id", "subject", "tlv_ml", "n_lesions"])
    table = inventory[inventory.study_id.isin(phantoms)]
    subj = table.groupby("subject").agg(n=("study_id", "size"), any_lesion=("n_lesions", lambda s: bool((s > 0).any())), tlv_ml=("tlv_ml", "median"))
    edges = NB08_TLV_EDGES_ML
    burden = np.where(~subj.any_lesion, "lesion-free", np.where(subj.tlv_ml <= edges[0], "TLV T1", np.where(subj.tlv_ml <= edges[1], "TLV T2", "TLV T3")))
    subj["stratum"] = pd.Series(burden, index=subj.index) + np.where(subj.any_lesion & (subj.n > 1), " / multi", np.where(subj.any_lesion, " / single", ""))
    rng = np.random.default_rng(seed)
    subj["split"] = "train"
    for _, group in subj.groupby("stratum"):
        names = np.array(sorted(group.index))
        rng.shuffle(names)
        subj.loc[names[:int(round(fraction * len(names)))], "split"] = "val"
    return subj


CACHE = NETDATA / "_cache"
GRIDS_MM = (3.0, 4.0)


def float16_errors(arrays, exact):
    """Worst errors of the float16 storage: PET in %IA/mL relative (voxels above PET_REF) and absolute, CT in HU, relative weight relative (voxels with weight above 1e-3)."""
    pet16 = np.expm1(arrays["pet"].astype(np.float64)) * PET_REF
    hot = exact["pet"] > PET_REF
    w16 = arrays["weight"].astype(np.float64)
    w = exact["weight"] > 1e-3
    return dict(pet_rel_max=float(np.max(np.abs(pet16[hot] / exact["pet"][hot] - 1))) if hot.any() else np.nan, pet_abs_max=float(np.max(np.abs(pet16 - exact["pet"]))),
                ct_hu_max=float(np.max(np.abs(arrays["ct"].astype(np.float64) * CT_SCALE - exact["ct"]))), weight_rel_max=float(np.max(np.abs(w16[w] / exact["weight"][w] - 1))) if w.any() else np.nan)


def cache_one(args):
    """Native sources once, then every grid; one npz of arrays and one json (meta, sources, retention under the other rules, float16 errors) per grid. Test phantoms keep only what the rebuild needs."""
    import time
    study_id, split, grids = args
    paths = {g: CACHE / f"g{g:g}" / f"{study_id}.npz" for g in grids}
    if all(p.exists() and p.with_suffix(".json").exists() for p in paths.values()):
        return study_id, "cached", 0.0
    start = time.time()
    native = native_sources(study_id)
    for g, path in paths.items():
        arrays, sources, meta, exact, others = prepare(study_id, g, native=native)
        sums = _source_sums(arrays["weight"].astype(np.float64), arrays["compartment"], arrays["lesion"])
        sources["weight_sum"] = [sums.get((c, i), 0.0) for c, i in zip(sources.code, sources.lesion_id)]
        info = dict(meta=meta, sources=sources.to_dict(orient="list"))
        if split != "test":
            info.update(others=others.to_dict(orient="list"), float16=float16_errors(arrays, exact))
        else:
            info["sources"] = sources[["study_id", "rule", "source", "code", "lesion_id", "n_voxels", "volume_ml", "native_volume_ml", "rescued", "status", "weight_sum"]].to_dict(orient="list")

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path.with_suffix(".tmp.npz"), **arrays)
        path.with_suffix(".tmp.npz").replace(path)
        path.with_suffix(".json").write_text(json.dumps(info))
    return study_id, "built", time.time() - start


FIELDS = ("pet", "ct", "compartment", "lesion", "organs", "weight")
PHANTOMS_PER_SHARD = 40


def load_cached(study_id, grid_mm):
    path = CACHE / f"g{grid_mm:g}" / f"{study_id}.npz"
    with np.load(path) as data:
        arrays = {k: data[k] for k in FIELDS}
    return arrays, json.loads(path.with_suffix(".json").read_text())


def pack_shards(assignment, grid_mm, out=None):
    """assignment: {study_id: net split}. Shards of PHANTOMS_PER_SHARD phantoms per split, sorted by study id; returns {study_id: shard file name}."""
    out = Path(out or NETDATA / "shards")
    out.mkdir(parents=True, exist_ok=True)
    where = {}
    for split in ("train", "val", "test"):
        studies = sorted(s for s, v in assignment.items() if v == split)
        for k in range(0, len(studies), PHANTOMS_PER_SHARD):
            name = f"{split}_{k // PHANTOMS_PER_SHARD:02d}.npz"
            chunk = studies[k:k + PHANTOMS_PER_SHARD]
            write_shard(out / name, {s: load_cached(s, grid_mm)[0] for s in chunk})
            where.update({s: name for s in chunk})
    return where


def build_cache(studies, processes=2, grids=GRIDS_MM, log=None):

    """studies: list of (study_id, split). Resumable; one phantom per task, fresh process every task to free memory."""
    import multiprocessing as mp
    import time
    log = log or (CACHE / "build.log")
    CACHE.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(processes, maxtasksperchild=1) as pool:
        for study_id, state, seconds in pool.imap_unordered(cache_one, [(s, sp, tuple(grids)) for s, sp in studies]):
            with open(log, "a") as handle:
                handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {study_id} {state} {seconds:.0f}\n")


DILATION_RADII = (0, 1, 2, 3)


def _ball(r):
    g = np.arange(-r, r + 1)
    return (g[:, None, None] ** 2 + g[None, :, None] ** 2 + g[None, None, :] ** 2) <= r * r


def lesion_dilation(args):
    """Per lesion of one phantom: share of its native weight (psma_mc.lesion_weights, sum-resampled) inside its grid voxels dilated by r voxels (ball, DILATION_RADII), and, if the parameters of one draw are given, the voxel TIA of the other sources inside the same dilated voxels (rebuild), for lesion evaluation on the grid."""
    study_id, grid_mm, draw_params = args
    arrays, info = load_cached(study_id, grid_mm)
    meta = info["meta"]
    lesion_grid = arrays["lesion"].astype(np.int64)
    tia_grid, lesion_tia = None, None
    if draw_params is not None:
        src = pd.DataFrame(info["sources"])
        n_les = int(meta["n_lesions"])
        g = draw_params[draw_params.source.isin(src.source[src.n_voxels > 0])]
        wc, wl = source_sum_tables(src, n_les)
        tia_grid = rebuild(arrays["compartment"], lesion_grid, arrays["weight"], wc, wl, *parameter_arrays(g, n_les))["tia"]
        lesion_tia = {int(x.source.split("_")[1]) + 1: x.tiac_h for x in g.itertuples(index=False) if x.source.startswith("lesion_")}
    lesions = M.load(study_id, "lesions").astype(np.int32)
    res = Resampler(lesions.shape, M.spacing_of(study_id), grid_mm, tuple(slice(a, b) for a, b in meta["native_box"]))
    weight = np.zeros(lesions.shape, np.float64)
    for i, (idx, w) in M.lesion_weights(study_id).items():
        weight[idx] = w
    rmax = max(DILATION_RADII)
    shape = np.array(lesion_grid.shape)
    rows = []
    for i, box in enumerate(ndi.find_objects(lesions[res.box]), start=1):
        if box is None:
            continue
        sub = res.sub(box)
        w_l = sub._apply(np.where(lesions[sub.box] == i, weight[sub.box], 0.0), sub.sum_ops)
        lo = np.maximum(np.array(sub.offset) - rmax, 0)
        hi = np.minimum(np.array(sub.offset) + np.array(sub.shape) + rmax, shape)
        window = tuple(slice(a, b) for a, b in zip(lo, hi))
        w_win = np.zeros(hi - lo)
        inner = tuple(slice(o - a, o - a + n) for o, a, n in zip(sub.offset, lo, sub.shape))
        clip = tuple(slice(0, s.stop - s.start) for s in inner)
        w_win[inner] = w_l[clip]
        own = lesion_grid[window] == i
        row = dict(study_id=study_id, source=f"lesion_{i - 1}", lesion_id=i, n_voxels=int(own.sum()))
        for r in DILATION_RADII:
            region = ndi.binary_dilation(own, _ball(r)) if r and own.any() else own
            row[f"retained_r{r}"] = float(w_win[region].sum())
            row[f"voxels_r{r}"] = int(region.sum())
            if tia_grid is not None:
                row[f"other_tia_r{r}"] = float(tia_grid[window][region & ~own].sum())
        if lesion_tia is not None:
            row["lesion_tia_h"] = float(lesion_tia.get(i, np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


SHELL_VOXELS = 1


def unit_weights(arrays, sources):
    """Unit-sum weight of every voxel within its source (lesions within their own labels), from the stored relative weights."""
    n_les = int(sources.lesion_id.max()) if (sources.code == LESION_CODE).any() else 0
    wc, wl = source_sum_tables(sources, n_les)
    comp, les = arrays["compartment"].astype(np.int64), arrays["lesion"].astype(np.int64)
    norm = np.where(les > 0, wl[les], wc[comp])
    inside = (comp > 0) & (norm > 0)
    return np.where(inside, arrays["weight"].astype(np.float64) / np.where(inside, norm, 1.0), 0.0)


def separate_lesion(tia_map, code, lesion_grid, i, r_eval, window=None):
    """TIA of lesion i alone from a voxel TIA map: sum over its voxels dilated by r_eval (voxels of other lesions left out), minus the local background in the ring = for each ring voxel, the median map value of its own source in a shell SHELL_VOXELS thick just outside the dilated mask (lesion voxels left out); a source absent from the shell uses its median over the window. Returns (separated, plain dilated sum)."""
    window = window or tuple(slice(0, n) for n in tia_map.shape)
    x, c, lab = tia_map[window], code[window], lesion_grid[window]
    own = lab == i
    dil = ndi.binary_dilation(own, _ball(r_eval)) if r_eval else own
    region = dil & ~((lab > 0) & ~own)
    ring = region & ~own
    shell = ndi.binary_dilation(own, _ball(r_eval + SHELL_VOXELS)) & ~dil & (lab == 0)
    plain = float(x[region].sum())
    background = 0.0
    for s in np.unique(c[ring]):
        if s == 0:
            continue
        ref = x[shell & (c == s)]
        if ref.size == 0:
            ref = x[(c == s) & (lab == 0)]
        background += float(np.median(ref)) * int((ring & (c == s)).sum()) if ref.size else 0.0
    return plain - background, plain


def lesion_separation(args):
    """Separation test on one phantom, lesions on the grid. Maps: 'label' = rebuild of the cohort TIAC (each lesion inside its own labels); 'spill' = the same organs plus each lesion distributed by its native weight sum-resampled without renormalisation (part of it in the neighbours' voxels, as a network that follows the PET would put it); 'baseline' = the 'spill' map of a predicted TIAC per source. Per lesion: true and predicted TIAC, plain dilated sum and separated value of each map."""
    study_id, grid_mm, r_eval, truth, predicted = args
    arrays, info = load_cached(study_id, grid_mm)
    meta = info["meta"]
    sources = pd.DataFrame(info["sources"])
    sources = sources[sources.n_voxels > 0]
    w = unit_weights(arrays, sources)
    code = arrays["compartment"].astype(np.int64)
    lesion_grid = arrays["lesion"].astype(np.int64)
    lesions = M.load(study_id, "lesions").astype(np.int32)
    res = Resampler(lesions.shape, M.spacing_of(study_id), grid_mm, tuple(slice(a, b) for a, b in meta["native_box"]))
    native_w = np.zeros(lesions.shape, np.float64)
    for i, (idx, wl) in M.lesion_weights(study_id).items():
        native_w[idx] = wl
    spread = {}
    for i, box in enumerate(ndi.find_objects(lesions[res.box]), start=1):
        if box is not None:
            sub = res.sub(box)
            spread[i] = (tuple(slice(o, o + n) for o, n in zip(sub.offset, sub.shape)), sub._apply(np.where(lesions[sub.box] == i, native_w[sub.box], 0.0), sub.sum_ops))

    def organ_map(tiac):
        values = np.zeros(LESION_CODE + 1)
        for name, v in tiac.items():
            if not name.startswith("lesion_"):
                values[COMPARTMENT_CODES[name]] = v
        return np.where(lesion_grid > 0, 0.0, values[code] * w)

    def lesion_ids(tiac):
        return {int(k.split("_")[1]) + 1: v for k, v in tiac.items() if k.startswith("lesion_")}

    def spill_map(tiac):
        out = organ_map(tiac)
        for i, v in lesion_ids(tiac).items():
            if i in spread:
                target, s = spread[i]
                out[target] += v * s[tuple(slice(0, t.stop - t.start) for t in target)]
        return out

    truth_les, pred_les = lesion_ids(truth), lesion_ids(predicted)
    label = organ_map(truth)
    for i, v in truth_les.items():
        label[lesion_grid == i] += v * w[lesion_grid == i]
    maps = dict(label=label, spill=spill_map(truth), baseline=spill_map(predicted))
    pad = r_eval + SHELL_VOXELS + 1
    rows = []
    for i, box in enumerate(ndi.find_objects(lesion_grid), start=1):
        if box is None or i not in truth_les:
            continue
        window = tuple(slice(max(b.start - pad, 0), min(b.stop + pad, n)) for b, n in zip(box, lesion_grid.shape))
        row = dict(study_id=study_id, source=f"lesion_{i - 1}", lesion_id=i, tiac_true=truth_les[i], tiac_pred=pred_les.get(i, np.nan))
        for name, m in maps.items():
            row[f"{name}_separated"], row[f"{name}_plain"] = separate_lesion(m, code, lesion_grid, i, r_eval, window)
        rows.append(row)
    return pd.DataFrame(rows)
