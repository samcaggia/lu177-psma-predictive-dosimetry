"""
Local diagnostics of an evaluation (NumPy only, no torch): structure summaries of the rebuilt targets and of saved
network maps, lesion errors against lesion properties, decomposition of the lesion TIAC error by head, alternative
aggregations of the organ quantities, and where the predicted rest-of-body activity sits.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .scoring import QUANTITIES, flattened_voxels

HEADS = ("log_a0", "log_lambda_bio", "early_share", "log_lambda_u", "log_tiac")
EARLY_SHARE_MAX = 0.95
HALO_LABELS = (1, 2, 5, 6, 7, 8, 9)
KIDNEY_FLUID = 2
HALO_MM = 10.0


def read_phantoms(nd, netdata, studies):
    """{study_id: stored arrays} for training or validation phantoms; refuses the test shards."""
    phantoms = pd.read_parquet(Path(netdata) / "phantoms.parquet")
    phantoms = phantoms[phantoms.study_id.isin(studies)]
    out = {}
    for shard in sorted(phantoms.shard.unique()):
        if shard.startswith("test_"):
            raise PermissionError(f"{shard} belongs to the test split")
        out.update(nd.read_shard(Path(netdata) / "shards" / shard, set(phantoms.study_id)))
    return out


class Phantom:
    """Labels, weights and structure masks of one phantom (compartments; lesions dilated by r_eval, inside the body)."""

    def __init__(self, nd, study, arrays, sources, kinds, grid_mm=4.0, r_eval=1):
        self.nd, self.study = nd, study
        self.code = arrays["compartment"].astype(np.int64)
        self.lesion = arrays["lesion"].astype(np.int64)
        self.weight = arrays["weight"].astype(np.float64)
        self.organs = arrays["organs"]
        self.pet = np.expm1(arrays["pet"].astype(np.float64)) * nd.PET_REF
        self.shape = self.code.shape
        self.voxel_ml = grid_mm ** 3 / 1000.0
        self.grid_mm = grid_mm
        self.n_lesions = max(int(self.lesion.max()), int(sources.lesion_id.max()))
        self.sums = nd.source_sum_tables(sources, self.n_lesions)
        norm = np.where(self.lesion > 0, self.sums[1][self.lesion], self.sums[0][self.code])
        self.inside = (self.code > 0) & (norm > 0)
        tia_codes = [c for s, c in nd.COMPARTMENT_CODES.items() if kinds[s] == "tia"]
        curve_codes = [c for s, c in nd.COMPARTMENT_CODES.items() if kinds[s] == "curve"]
        self.is_tia, self.is_curve = np.isin(self.code, tia_codes), np.isin(self.code, curve_codes)

        self.sources, self.structures, groups, own = [], [], [], []
        flat_code, flat_inside = self.code.ravel(), self.inside.ravel()
        for name, c in nd.COMPARTMENT_CODES.items():
            index = np.flatnonzero((flat_code == c) & flat_inside) if c != nd.LESION_CODE else np.array([], int)
            if index.size:
                self.sources.append(name)
                self.structures.append(name)
                groups.append(index)
                own.append(np.ones(index.size, bool))
        ball = nd._ball(r_eval)
        for j, box in enumerate(ndi.find_objects(self.lesion), start=1):
            if box is None:
                continue
            window = tuple(slice(max(b.start - r_eval, 0), min(b.stop + r_eval, n)) for b, n in zip(box, self.shape))
            mine = self.lesion[window] == j
            region = (ndi.binary_dilation(mine, ball) if r_eval else mine) & self.inside[window]
            if not region.any():
                continue
            local = np.nonzero(region)
            groups.append(np.ravel_multi_index(tuple(a + w.start for a, w in zip(local, window)), self.shape))
            own.append(mine[region])
            self.sources.append(f"lesion_{j - 1}")
            self.structures.append("lesion")
        self.entries = np.concatenate(groups)
        self.segment = np.concatenate([np.full(g.size, k) for k, g in enumerate(groups)])
        self.own = np.concatenate(own)
        self.n_voxels = [g.size for g in groups]
        stops = np.cumsum(self.n_voxels)
        self.bounds = [(int(b - m), int(b)) for b, m in zip(stops, self.n_voxels)]
        self.flatten = np.zeros(self.entries.size, bool)
        self.reference = np.zeros(0, int)
        if "rest_of_body" in self.sources:
            a, b = self.bounds[self.sources.index("rest_of_body")]
            self.flatten[a:b] = flattened_voxels(nd, self.code, self.lesion, grid_mm).ravel()[self.entries[a:b]]
            self.reference = a + np.flatnonzero(~self.flatten[a:b])

    def __len__(self):
        return len(self.sources)

    def targets(self, params_of_draw):
        nd = self.nd
        comp, les = nd.parameter_arrays(params_of_draw, self.n_lesions)
        return nd.rebuild(self.code, self.lesion, self.weight, self.sums[0], self.sums[1], comp, les)

    def shell(self):
        """Rest-of-body voxels set to the median by the source definition: within HALO_MM of kidneys, glands or bladder, and kidney fluid."""
        rest = self.code == self.nd.COMPARTMENT_CODES["rest_of_body"]
        distance = ndi.distance_transform_edt(~np.isin(self.organs, HALO_LABELS), sampling=self.grid_mm)
        return rest & ((distance <= HALO_MM) | (self.organs == KIDNEY_FLUID))

    def boundary(self):
        """Voxels with a 6-neighbour of another source (lesions as their own sources)."""
        label = np.where(self.lesion > 0, 100 + self.lesion, self.code)
        out = np.zeros(self.shape, bool)
        for axis in range(3):
            a, b = [slice(None)] * 3, [slice(None)] * 3
            a[axis], b[axis] = slice(1, None), slice(None, -1)
            different = label[tuple(a)] != label[tuple(b)]
            out[tuple(a)] |= different
            out[tuple(b)] |= different
        return out


def summarise(nd, a0, lambda_bio, lambda_u, lu_ok, late, early, tia, segment, n):
    """The sums of lib.structures._summarise on flat entries: {quantity: (structures,) array}."""
    total = lambda x: np.bincount(segment, weights=x, minlength=n)
    with np.errstate(divide="ignore", invalid="ignore"):
        a0_total, a0_lu, tia_total = total(a0), total(np.where(lu_ok, a0, 0.0)), total(tia)
        return {"log_a0": np.log(a0_total),
                "log_lambda_bio": np.log(np.maximum(total(a0 * lambda_bio) / a0_total, nd.LAMBDA_BIO_FLOOR)),
                "log_lambda_u": np.where(a0_lu > 0, np.log(total(np.where(lu_ok, a0 * lambda_u, 0.0)) / a0_lu), np.nan),
                "log_tiac": np.log(tia_total), "log_tiac_late": np.log(total(late)),
                "early_share": total(early) / tia_total}


def _weighted_median(values, weights):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return values[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])]


def _lower_median(values):
    return np.partition(values, (values.size - 1) // 2)[(values.size - 1) // 2]


def lambda_location(ph, log_lambda, a0):
    """As lib.structures._lambda_location: median in a compartment, A0-weighted median on a lesion mask."""
    out = np.empty(len(ph))
    for s, (a, b) in enumerate(ph.bounds):
        out[s] = (_weighted_median(log_lambda[a:b], a0[a:b]) if ph.structures[s] == "lesion"
                  else _lower_median(log_lambda[a:b]))
    return np.maximum(out, np.log(ph.nd.LAMBDA_BIO_FLOOR))


def truth_summary(ph, targets):
    """Structure quantities of the rebuilt targets with the default aggregation of lib.structures."""
    f = truth_fields(ph, targets)
    f["lambda_bio"] = np.maximum(f["lambda_bio"], ph.nd.LAMBDA_BIO_FLOOR)
    out = summarise(ph.nd, **f, segment=ph.segment, n=len(ph))
    out["log_lambda_bio"] = lambda_location(ph, np.log(f["lambda_bio"]), f["a0"])
    return out


def map_summary(ph, fields):
    """Structure quantities of the network mean maps with the default aggregation of lib.structures."""
    f = _at_entries(fields, ph.entries)
    if ph.reference.size:
        for key in ("a0", "late", "early", "tia"):
            f[key] = np.where(ph.flatten, _lower_median(f[key][ph.reference]), f[key])
    out = summarise(ph.nd, **f, segment=ph.segment, n=len(ph))
    out["log_lambda_bio"] = lambda_location(ph, np.log(fields["lambda_bio"].ravel()[ph.entries]), f["a0"])
    return out


def truth_fields(ph, targets):
    e = ph.entries
    lambda_u = targets["lambda_u"].ravel()[e]
    lu_ok = ph.is_curve.ravel()[e] & np.isfinite(lambda_u)
    return dict(a0=targets["a0"].ravel()[e], lambda_bio=targets["lambda_bio"].ravel()[e],
                lambda_u=np.where(lu_ok, lambda_u, 1.0), lu_ok=lu_ok, late=targets["tia_late"].ravel()[e],
                early=targets["tia_early"].ravel()[e], tia=targets["tia"].ravel()[e])


def map_fields(ph, mean):
    """Voxel quantities of the network mean maps (heads, X, Y, Z) on the whole phantom, as lib.structures.voxel_fields."""
    nd = ph.nd
    value = {h: mean[k].astype(np.float64) for k, h in enumerate(HEADS)}
    a0 = np.exp(value["log_a0"]) * ph.voxel_ml
    lambda_bio = np.exp(value["log_lambda_bio"])
    share = np.minimum(value["early_share"], EARLY_SHARE_MAX)
    late = a0 / (lambda_bio + nd.LAMBDA_PHYS_H)
    with np.errstate(over="ignore"):
        tia = np.where(ph.is_tia, np.exp(value["log_tiac"]) * ph.voxel_ml, late / (1.0 - share))
        lambda_u = np.exp(value["log_lambda_u"])
    early = np.where(ph.is_tia, share * tia, tia - late)
    return dict(a0=a0, lambda_bio=lambda_bio, lambda_u=lambda_u, lu_ok=ph.is_curve, late=late, early=early, tia=tia,
                share=share)


def _at_entries(fields, entries):
    return {k: v.ravel()[entries] for k, v in fields.items() if k != "share"}


def aggregation_variants(ph, fields):
    """
    Organ quantities of the network maps under alternative aggregations, all computable at inference (they use the
    prediction, the masks and the PET only). Rows (source, variant, quantity, value).
      default: as lib.structures (sums, rest of body flattened next to other sources, median lambda_bio).
      sum_and_weighted_mean: plain sums and A0-weighted mean lambda_bio (the first version of the evaluation).
      positive_weight: only voxels with a positive source weight (PET > 0 inside the mask).
      shell_median: rest of body with the voxels of the 10 mm shells around kidneys, glands and bladder (and kidney
        fluid) set to the median of the other rest-of-body voxels, the rule that defines the source.
      interior: only voxels with no 6-neighbour of another source (A0, TIA rescaled by the voxel count).
      lambda_log_mean, lambda_median, lambda_a0_median: log lambda_bio as the A0-weighted mean of the voxel logs,
        the voxel median, the A0-weighted median (the other quantities as current).
    """
    nd = ph.nd
    organ = np.array([s != "lesion" for s in ph.structures])[ph.segment]
    entries, segment, n = ph.entries[organ], ph.segment[organ], len(ph)
    names = [s for s in ph.sources if not s.startswith("lesion_")]
    rows = []

    def add(variant, values, keep=QUANTITIES):
        for q in keep:
            rows.extend(dict(source=s, variant=variant, quantity=q, value=float(values[q][k]))
                        for k, s in enumerate(names))

    current = summarise(nd, **_at_entries(fields, entries), segment=segment, n=n)
    add("sum_and_weighted_mean", current)
    default = map_summary(ph, fields)
    add("default", {q: v[:len(names)] for q, v in default.items()})
    positive = ph.weight.ravel()[entries] > 0
    add("positive_weight", summarise(nd, **_at_entries(fields, entries[positive]), segment=segment[positive], n=n))

    shell = ph.shell()
    rest = (ph.code == nd.COMPARTMENT_CODES["rest_of_body"]) & ph.inside & ~shell
    flattened = dict(fields)
    if shell.any() and rest.any():
        for key in ("a0", "late", "early", "tia", "lambda_bio"):
            flattened[key] = np.where(shell, np.median(fields[key][rest]), fields[key])
    add("shell_median", summarise(nd, **_at_entries(flattened, entries), segment=segment, n=n))

    inner = ~ph.boundary().ravel()[entries]
    interior = summarise(nd, **_at_entries(fields, entries[inner]), segment=segment[inner], n=n)
    with np.errstate(divide="ignore"):
        scale = np.log(np.bincount(segment, minlength=n) / np.maximum(np.bincount(segment[inner], minlength=n), 1))
    for q in ("log_a0", "log_tiac", "log_tiac_late"):
        interior[q] = interior[q] + scale
    empty = np.bincount(segment[inner], minlength=n) == 0
    interior = {q: np.where(empty, np.nan, v) for q, v in interior.items()}
    add("interior", interior)

    a0, log_lambda = fields["a0"].ravel()[entries], np.log(fields["lambda_bio"].ravel()[entries])
    a0_sum = np.bincount(segment, weights=a0, minlength=n)
    weighted = np.divide(np.bincount(segment, weights=a0 * log_lambda, minlength=n), a0_sum,
                         out=np.full(n, np.nan), where=a0_sum > 0)
    medians, a0_medians = np.full(n, np.nan), np.full(n, np.nan)
    for k in np.unique(segment):
        m = segment == k
        medians[k], a0_medians[k] = np.median(log_lambda[m]), _weighted_median(log_lambda[m], a0[m])
    for variant, values in (("lambda_log_mean", weighted), ("lambda_median", medians), ("lambda_a0_median", a0_medians)):
        add(variant, {**current, "log_lambda_bio": values})
    return pd.DataFrame(rows)


def rest_of_body_location(ph, fields, targets):
    """Share of the rest-of-body A0 (predicted and true) in the shells, next to another source, and in the 1% hottest voxels."""
    rest = (ph.code == ph.nd.COMPARTMENT_CODES["rest_of_body"]) & ph.inside
    shell, boundary = ph.shell() & rest, ph.boundary() & rest
    row = dict(voxels=int(rest.sum()), shell_voxels=float(shell.sum() / rest.sum()),
               boundary_voxels=float(boundary.sum() / rest.sum()))
    for name, a0 in (("predicted", fields["a0"]), ("true", targets["a0"])):
        total = a0[rest].sum()
        ranked = np.sort(a0[rest])[::-1]
        row.update({f"{name}_a0_in_shell": float(a0[shell].sum() / total),
                    f"{name}_a0_at_boundary": float(a0[boundary].sum() / total),
                    f"{name}_a0_in_top_1pct": float(ranked[:max(1, ranked.size // 100)].sum() / total)})
    row["log_predicted_over_true_a0"] = float(np.log(fields["a0"][rest].sum() / targets["a0"][rest].sum()))
    for name, mask in (("shell", shell), ("boundary_outside_shell", boundary & ~shell), ("elsewhere", rest & ~shell & ~boundary)):
        row[f"log_ratio_{name}"] = float(np.log(fields["a0"][mask].sum() / targets["a0"][mask].sum())) if mask.any() else np.nan
    return row


def lesion_decomposition(ph, fields, targets):
    """
    Per lesion (dilated mask), log predicted over true TIAC and the part of each head: the true voxel maps with one head
    replaced by the prediction (A0, lambda_bio, early share on the voxels where TIA = late / (1 - share); the TIAC head
    on the ring voxels of tia structures); the remainder is the interaction. Also the log ratio on the lesion's own
    voxels and on the ring, and the share of the true TIAC that sits in the ring.
    """
    nd = ph.nd
    lesion = np.array([s == "lesion" for s in ph.structures])[ph.segment]
    if not lesion.any():
        return pd.DataFrame()
    e, seg, own = ph.entries[lesion], ph.segment[lesion], ph.own[lesion]
    first = int(seg.min()) if seg.size else 0
    seg, n = seg - first, len(ph) - first
    is_tia = ph.is_tia.ravel()[e]
    pred = {k: v.ravel()[e] for k, v in fields.items()}
    a0_t, lam_t = targets["a0"].ravel()[e], targets["lambda_bio"].ravel()[e]
    tia_t = targets["tia"].ravel()[e]
    share_t = np.where(tia_t > 0, targets["tia_early"].ravel()[e] / np.where(tia_t > 0, tia_t, 1.0), 0.0)

    def tiac(a0, lam, share, head, mask=None):
        voxel = np.where(is_tia, head, a0 / (lam + nd.LAMBDA_PHYS_H) / (1.0 - share))
        return np.bincount(seg, weights=voxel if mask is None else voxel * mask, minlength=n)

    truth = tiac(a0_t, lam_t, share_t, tia_t)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = pd.DataFrame({
            "source": ph.sources[first:], "log_tiac_error": np.log(tiac(pred["a0"], pred["lambda_bio"], pred["share"], pred["tia"]) / truth),
            "part_a0": np.log(tiac(pred["a0"], lam_t, share_t, tia_t) / truth),
            "part_lambda_bio": np.log(tiac(a0_t, pred["lambda_bio"], share_t, tia_t) / truth),
            "part_early_share": np.log(tiac(a0_t, lam_t, pred["share"], tia_t) / truth),
            "part_ring_tiac_head": np.log(tiac(a0_t, lam_t, share_t, pred["tia"]) / truth),
            "log_ratio_own": np.log(tiac(pred["a0"], pred["lambda_bio"], pred["share"], pred["tia"], own) / tiac(a0_t, lam_t, share_t, tia_t, own)),
            "log_ratio_ring": np.log(tiac(pred["a0"], pred["lambda_bio"], pred["share"], pred["tia"], ~own) / tiac(a0_t, lam_t, share_t, tia_t, ~own)),
            "true_tiac_share_in_ring": tiac(a0_t, lam_t, share_t, tia_t, ~own) / truth})
    out["interaction"] = out.log_tiac_error - out[["part_a0", "part_lambda_bio", "part_early_share", "part_ring_tiac_head"]].sum(axis=1)
    return out


def run(nd, netdata, studies, scenario="A", arm="main", maps_dir=None, r_eval=1, log=print):
    """
    One pass over the phantoms. Always: truth (params rows), truth_like (lesion rows on the dilated mask) and the lesion
    properties (PET mean on the lesion labels, voxels). With maps_dir (float16 maps written by evaluate_cohort):
    aggregation variants, rest-of-body location of the activity, decomposition of the lesion TIAC error (every draw).
    """
    netdata = Path(netdata)
    params = pd.read_parquet(netdata / "params.parquet")
    kinds = params.drop_duplicates("structure").set_index("structure").target_kind
    params = params[(params.arm == arm) & (params.scenario == scenario) & params.has_source
                    & params.study_id.isin(studies)].reset_index(drop=True)
    sources = pd.read_parquet(netdata / "sources.parquet")
    arrays = read_phantoms(nd, netdata, studies)
    truths, lesions, variants, locations, parts, predictions = [], [], [], [], [], []
    for position, study in enumerate(studies):
        ph = Phantom(nd, study, arrays.pop(study), sources[sources.study_id == study], kinds, r_eval=r_eval)
        path = None if maps_dir is None else Path(maps_dir) / f"{study}.npz"
        fields = None
        if path is not None and path.exists():
            with np.load(path) as data:
                fields = map_fields(ph, data["mean"])
            variants.append(aggregation_variants(ph, fields).assign(study_id=study))
            predictions.append(pd.DataFrame({"study_id": study, "source": ph.sources, **map_summary(ph, fields)}))
        for j in np.unique(ph.lesion[ph.lesion > 0]):
            own = ph.lesion == j
            lesions.append(dict(study_id=study, source=f"lesion_{j - 1}", grid_voxels=int(own.sum()),
                                pet_mean_pct_ia_ml=float(ph.pet[own].mean()), pet_max_pct_ia_ml=float(ph.pet[own].max())))
        for draw, g in params[params.study_id == study].groupby("draw"):
            targets = ph.targets(g)
            values = truth_summary(ph, targets)
            truths.append(pd.DataFrame({"source": ph.sources, **values}).assign(virtual_id=g.virtual_id.iloc[0]))
            if fields is not None:
                parts.append(lesion_decomposition(ph, fields, targets).assign(study_id=study, draw=draw,
                                                                              virtual_id=g.virtual_id.iloc[0]))
                if draw == 0:
                    locations.append(dict(study_id=study, **rest_of_body_location(ph, fields, targets)))
        log(f"[{position + 1}/{len(studies)}] {study}: {len(ph)} structures" + ("" if fields is not None else " (no map)"))
    map_truth = pd.concat(truths, ignore_index=True)
    merged = params.merge(map_truth, on=["virtual_id", "source"], how="left", suffixes=("", "_map"))
    truth_like = params.copy()
    lesion = (params.structure == "lesion").to_numpy()
    for q in QUANTITIES:
        truth_like[q] = np.where(lesion, merged[f"{q}_map"].to_numpy(float), merged[q].to_numpy(float))
    table = lambda rows: pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return dict(truth=params, truth_like=truth_like, lesions=pd.DataFrame(lesions), variants=table(variants),
                rest_of_body=pd.DataFrame(locations), decomposition=table(parts), map_prediction=table(predictions))


def lesion_errors(prediction, truth_like, lesions, sources, phantoms):
    """
    One row per lesion: mean over the draws of the error (prediction - truth on the dilated mask) of log TIAC, log A0,
    log lambda_bio and early share, with the native volume, the share of the lesion weight retained by its labels, the
    scanner, the PET mean on the labels and the number of lesions of the phantom.
    """
    quantities = ["log_tiac", "log_a0", "log_lambda_bio", "early_share"]
    pred = prediction[prediction.structure == "lesion"]
    merged = truth_like[["virtual_id", "study_id", "source", *quantities]].merge(
        pred[["virtual_id", "source"] + [f"{q}_mean" for q in quantities]], on=["virtual_id", "source"])
    for q in quantities:
        merged[f"error_{q}"] = merged[f"{q}_mean"] - merged[q]
    out = merged.groupby(["study_id", "source"])[[f"error_{q}" for q in quantities]].mean().reset_index()
    out = out.merge(lesions, on=["study_id", "source"], how="left")
    keep = [c for c in ("native_volume_ml", "retained", "n_voxels", "rescued") if c in sources]
    out = out.merge(sources[["study_id", "source"] + keep], on=["study_id", "source"], how="left")
    return out.merge(phantoms[["study_id", "subject", "scanner", "n_lesions"]], on="study_id", how="left")


def binned(table, column, edges=None, labels=None, quantities=("log_tiac", "log_a0", "log_lambda_bio", "early_share")):
    """Lesion errors by bins of a property: lesions, mean error (bias) and median |error| per quantity."""
    key = table[column] if edges is None else pd.cut(table[column], edges, labels=labels, include_lowest=True)
    rows = []
    for name, g in table.groupby(key, observed=True):
        row = dict(property=column, bin=str(name), lesions=len(g))
        for q in quantities:
            row[f"bias_{q}"], row[f"mae_{q}"] = float(g[f"error_{q}"].mean()), float(g[f"error_{q}"].abs().median())
        rows.append(row)
    return pd.DataFrame(rows)


def score_variants(variants, truth):
    """Bias and median |error| of every aggregation variant against the cohort parameters, per structure and quantity."""
    keep = ["log_a0", "log_lambda_bio", "log_tiac", "log_tiac_late", "early_share"]
    long = truth.melt(id_vars=["study_id", "source", "virtual_id"], value_vars=keep, var_name="quantity", value_name="truth")
    merged = variants.merge(long, on=["study_id", "source", "quantity"])
    merged["error"] = merged.value - merged.truth
    out = merged.groupby(["source", "quantity", "variant"]).error.agg(bias="mean", median_abs_error=lambda e: e.abs().median())
    return out.reset_index().rename(columns={"source": "structure"})


def pet_residual(ph, targets, floors=(1e-5, 1e-4, 1e-3)):
    """
    Per source of one phantom, what a PET-residual head would have to predict: r = log(A0 target per mL) - log(PET),
    PET in %IA/mL, over the voxels with a positive target. Rows: voxels, SD of r and of the log target itself within
    the source, the median of r, r at the voxels next to another source minus its median over the others, the same
    for the rest-of-body shells, and the share of voxels with the PET below each floor.
    """
    a0, pet = targets["a0"] / ph.voxel_ml, ph.pet
    label = np.where(ph.lesion > 0, 100 + ph.lesion, ph.code)
    valid = ph.inside & (a0 > 0)
    boundary, shell = ph.boundary(), ph.shell()
    names = {c: s for s, c in ph.nd.COMPARTMENT_CODES.items()}
    with np.errstate(divide="ignore"):
        log_a0 = np.log(np.where(valid, a0, 1.0))
        r = log_a0 - np.log(np.maximum(pet, min(floors)))
    rows = []
    objects = ndi.find_objects(label.astype(np.int32))
    for value in np.unique(label[valid]):
        box = objects[value - 1]
        m = valid[box] & (label[box] == value)
        if m.sum() < 2:
            continue
        x, y, edge, p = r[box][m], log_a0[box][m], boundary[box][m], pet[box][m]
        row = dict(study_id=ph.study, source=names[value] if value < 100 else f"lesion_{value - 101}",
                   structure=names[value] if value < 100 else "lesion", voxels=int(m.sum()),
                   sd_residual=float(x.std()), sd_log_a0=float(y.std()), median_residual=float(np.median(x)),
                   boundary_share=float(edge.mean()),
                   boundary_minus_interior=float(x[edge].mean() - np.median(x[~edge])) if edge.any() and (~edge).any() else np.nan,
                   sd_residual_interior=float(x[~edge].std()) if (~edge).sum() > 1 else np.nan)
        if value == ph.nd.COMPARTMENT_CODES["rest_of_body"]:
            halo = shell[box][m]
            row.update(shell_share=float(halo.mean()),
                       shell_minus_elsewhere=float(x[halo].mean() - np.median(x[~halo])) if halo.any() else np.nan,
                       sd_residual_outside_shell=float(x[~halo].std()))
        for floor in floors:
            row[f"pet_below_{floor:g}"] = float((p < floor).mean())
        rows.append(row)
    return pd.DataFrame(rows)
