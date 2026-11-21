"""
Evaluation of a trained network: voxel-level NLL and calibration against the population prior in the metric of the
loss, structure-level predictions keyed like the baselines, variance scales (closed form, cross-validated by subject),
comparison with the baselines through net_data.evaluate, and the test cohort (built only on request).
"""
import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from . import data as data_module
from . import figures
from .config import HEADS
from .evaluate import prior_lookup, prior_output
from .losses import _structure_weights, instance_ids, kinetic_loss, voxel_targets
from .scoring import (KEYS, QUANTITIES, apply_scales, compare_methods, cross_validated, early_share_cap, fit_scales,
                      lesion_overlap, side_by_side, subject_folds, washout_check, with_sd)
from .structures import (mean_tia_map, phantom_structures, prediction_summary, separated_lesion_tiac,
                         structure_rows, truth_summary)
from .scenario import CEILING, scenario_rho
from .utils import save_table

ABS_BINS, ABS_MAX = 5000, 25.0  # bins of 0.005 in |z|
Z_BINS, Z_MAX = 120, 6.0
LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
NOMINAL_68 = float(2 * stats.norm.cdf(1.0) - 1)


class VoxelStats:
    """
    Sums over phantoms and draws of the loss weights (alpha x balanced by structure + (1 - alpha) x uniform, as in
    kinetic_loss) times: 1, the NLL, z and z^2 with z = (y - mean) / sd, and histograms of z and |z|, by head and
    structure class. The weights of one head sum to 1 on each phantom and draw.
    """

    def __init__(self, cohort, cfg):
        self.cohort, self.cfg = cohort, cfg
        self.n_classes = cohort.nd.LESION_CODE + 1
        shape = (len(HEADS), self.n_classes)
        zeros = lambda *s: torch.zeros(*s, dtype=torch.float64, device=cohort.device)
        self.sums = {k: zeros(shape) for k in ("weight", "nll", "z", "z2", "voxels")}
        self.abs_hist, self.z_hist = zeros(*shape, ABS_BINS + 1), zeros(*shape, Z_BINS)
        self.units = np.zeros(len(HEADS))
        self.loss_parts = {}
        self.n_updates = 0

    @torch.no_grad()
    def update(self, output, y, mask, compartment, lesion):
        cohort, cfg, n_classes = self.cohort, self.cfg, self.n_classes
        _, parts = kinetic_loss(output, y, mask, compartment, lesion, cohort, cfg)
        for key, value in parts.items():
            self.loss_parts[key] = self.loss_parts.get(key, 0.0) + float(value)
        self.n_updates += 1
        output = output.float()
        mean, logvar = output[:, 0::2], output[:, 1::2].clamp(*cfg.logvar_range)
        lesion_code = cohort.nd.LESION_CODE
        ids, stride = instance_ids(compartment, lesion, lesion_code)
        classes = torch.where(lesion > 0, torch.full_like(lesion, lesion_code), compartment)
        for k in range(len(HEADS)):
            m = mask[:, k]
            n = int(m.sum())
            if n == 0:
                continue
            selected = ids[m]
            weight, _ = _structure_weights(selected, stride, lesion_code, stride * compartment.shape[0])
            w = cfg.alpha * weight[selected] + (1.0 - cfg.alpha) / n
            lv = logvar[:, k][m].double()
            z = (y[:, k][m].double() - mean[:, k][m].double()) * torch.exp(-0.5 * lv)
            c = classes[m]
            for name, value in (("weight", w), ("nll", w * 0.5 * (lv + z * z)), ("z", w * z), ("z2", w * z * z),
                                ("voxels", torch.ones_like(w))):
                self.sums[name][k] += torch.bincount(c, weights=value, minlength=n_classes)
            bins = (z.abs() * (ABS_BINS / ABS_MAX)).long().clamp(max=ABS_BINS)
            self.abs_hist[k] += torch.bincount(c * (ABS_BINS + 1) + bins, weights=w,
                                               minlength=n_classes * (ABS_BINS + 1)).view(n_classes, -1)
            bins = ((z + Z_MAX) * (Z_BINS / (2 * Z_MAX))).long().clamp(0, Z_BINS - 1)
            self.z_hist[k] += torch.bincount(c * Z_BINS + bins, weights=w,
                                             minlength=n_classes * Z_BINS).view(n_classes, -1)
            self.units[k] += 1

    def _rows(self):
        names = {code: name for name, code in self.cohort.nd.COMPARTMENT_CODES.items()}
        sums = {k: v.cpu().numpy() for k, v in self.sums.items()}
        abs_hist, z_hist = self.abs_hist.cpu().numpy(), self.z_hist.cpu().numpy()
        for k, head in enumerate(HEADS):
            if self.units[k] == 0:
                continue
            groups = [("all", slice(None))] + [(names[c], slice(c, c + 1)) for c in range(1, self.n_classes)]
            for structure, sel in groups:
                weight = sums["weight"][k, sel].sum()
                if weight <= 0:
                    continue
                yield dict(head=head, structure=structure, voxels=sums["voxels"][k, sel].sum(), weight=weight,
                           units=self.units[k], nll_sum=sums["nll"][k, sel].sum(), z=sums["z"][k, sel].sum(),
                           z2=sums["z2"][k, sel].sum(), abs_hist=abs_hist[k, sel].sum(axis=0),
                           z_hist=z_hist[k, sel].sum(axis=0))

    @staticmethod
    def _coverage(abs_hist, level, scale):
        cumulative = np.concatenate([[0.0], np.cumsum(abs_hist[:ABS_BINS])])
        position = stats.norm.ppf(0.5 + level / 2) * scale * ABS_BINS / ABS_MAX
        return float(np.interp(position, np.arange(ABS_BINS + 1), cumulative) / abs_hist.sum())

    def tables(self, scales=None):
        """
        summary: per head and structure class ('all' = the loss of the head): voxels per phantom and draw, share of the
        loss weight, NLL ('all': as in validate; a class: weighted mean over its voxels), weighted mean of z and z^2,
        coverage of |z| < 1 and |z| < 1.96, and the same after multiplying every SD of the head by its scale.
        scales {head: SD factor} defaults to the NLL-optimal one, sqrt(weighted mean z^2) over all the voxels of the head.
        reliability: nominal against empirical central coverage at LEVELS. histogram: weighted density of z.
        """
        rows = list(self._rows())
        if scales is None:
            scales = {r["head"]: float(np.sqrt(r["z2"] / r["weight"])) for r in rows if r["structure"] == "all"}
        summary, reliability, histogram = [], [], []
        edges = np.linspace(-Z_MAX, Z_MAX, Z_BINS + 1)
        for r in rows:
            s = scales[r["head"]]
            whole = r["structure"] == "all"
            norm = r["units"] if whole else r["weight"]
            z2 = r["z2"] / r["weight"]
            nll = r["nll_sum"] / norm
            shift = 0.5 * np.log(s ** 2) + 0.5 * (r["z2"] / norm) * (1 / s ** 2 - 1)
            summary.append(dict(
                head=r["head"], structure=r["structure"], voxels=r["voxels"] / r["units"],
                weight_share=r["weight"] / r["units"], nll=nll, mean_z=r["z"] / r["weight"], mean_z2=z2,
                own_scale=float(np.sqrt(z2)), coverage_68=self._coverage(r["abs_hist"], NOMINAL_68, 1.0),
                coverage_95=self._coverage(r["abs_hist"], 0.95, 1.0), head_scale=s,
                nll_scaled=nll + shift,
                coverage_68_scaled=self._coverage(r["abs_hist"], NOMINAL_68, s),
                coverage_95_scaled=self._coverage(r["abs_hist"], 0.95, s)))
            for level in LEVELS:
                reliability.append(dict(head=r["head"], structure=r["structure"], nominal=level,
                                        empirical=self._coverage(r["abs_hist"], level, 1.0),
                                        empirical_scaled=self._coverage(r["abs_hist"], level, s)))
            density = r["z_hist"] / (r["z_hist"].sum() * (edges[1] - edges[0]))
            histogram.append(pd.DataFrame(dict(head=r["head"], structure=r["structure"], z_low=edges[:-1],
                                               z_high=edges[1:], density=density)))
        return pd.DataFrame(summary), pd.DataFrame(reliability), pd.concat(histogram, ignore_index=True)

    def loss_check(self):
        """Per head: the NLL of kinetic_loss averaged over phantoms and draws (what validate returns) and the one from the sums here."""
        sums = self.sums["nll"].sum(dim=1).cpu().numpy()
        return pd.DataFrame([dict(head=head, nll_kinetic_loss=self.loss_parts.get(head, np.nan) / max(1, self.n_updates),
                                  nll_from_sums=sums[k] / max(1, self.n_updates)) for k, head in enumerate(HEADS)])


def nll_against_prior(network_summary, prior_summary):
    """Network and prior NLL side by side per head and structure class; positive difference = network worse than the prior."""
    columns = ["head", "structure", "voxels", "weight_share", "nll", "nll_scaled", "mean_z2"]
    out = network_summary[columns].merge(prior_summary[["head", "structure", "nll", "mean_z2"]],
                                         on=["head", "structure"], suffixes=("_network", "_prior"))
    out["network_minus_prior"] = out.nll_network - out.nll_prior
    out["scaled_minus_prior"] = out.nll_scaled - out.nll_prior
    return out.rename(columns={"nll_scaled": "nll_network_scaled"})


def _read_params(cohort, studies):
    params = pd.read_parquet(cohort._params_path)
    return params[(params.arm == cohort.arm) & (params.scenario == cohort.scenario) & params.has_source
                  & params.study_id.isin(studies)].reset_index(drop=True)


@torch.no_grad()
def evaluate_cohort(predict, cohort, indices, loss_cfg, prior, k_samples=64, seed=20261114, r_eval=1, maps_dir=None,
                    log=print, voxel_statistics=True):
    """
    One pass over the phantoms `indices` of the cohort. predict(i) gives the network output of phantom i. Returns a dict:
      network, prior: VoxelStats of the network and of the prior predictor, on every draw of every phantom;
      prediction: structure predictions keyed (arm, virtual_id, source), <q>_mean, <q>_sd (correlated errors within
        a structure), <q>_sd_independent, log_tiac_separated (lesions: net_data.separate_lesion), repeated for the
        draws of the phantom (the network does not see the draw); mask_has_other_lesion (the dilated mask of the
        lesion holds voxels of another lesion) and early_share_capped (share of the voxels of the structure whose
        mean early share is above the cap);
      truth: the cohort parameters of these phantoms (params.parquet rows);
      truth_like: the same with the lesion rows replaced by the summary of the rebuilt targets on the dilated mask;
      organ_check: largest difference between the organ summaries of the rebuilt maps and params.parquet.
    maps_dir: if given, the mean and log-variance maps are saved there as float16 npz, one file per phantom.
    voxel_statistics False skips the voxel statistics (structure tables only, e.g. when recomputing from saved maps).
    """
    nd = cohort.nd
    studies = [cohort.phantoms.study_id[int(i)] for i in indices]
    params = _read_params(cohort, studies)
    lookup = prior_lookup(prior, cohort, cohort.arm)
    network, baseline = VoxelStats(cohort, loss_cfg), VoxelStats(cohort, loss_cfg)
    predictions, truths = [], []
    for position, i in enumerate(indices):
        i = int(i)
        study = cohort.phantoms.study_id[i]
        output = predict(i)
        constant = prior_output(cohort, i, lookup)
        st = phantom_structures(cohort, i, r_eval)
        compartment, lesion = cohort.compartment[i].long()[None], cohort.lesion[i].long()[None]
        virtual = params[params.study_id == study].drop_duplicates("virtual_id").set_index("draw").virtual_id
        for draw in range(cohort.n_draws[i]):
            targets = cohort.targets(i, draw)
            if voxel_statistics:
                y, mask = voxel_targets({k: v[None] for k, v in targets.items()}, compartment, cohort)
                network.update(output, y, mask, compartment, lesion)
                baseline.update(constant, y, mask, compartment, lesion)
            truths.append(structure_rows(st, truth_summary(targets, st, nd), virtual_id=virtual[draw]))
            del targets
        summary, capped = prediction_summary(output, cohort, i, st, loss_cfg.logvar_range, k_samples, seed)
        columns = {f"{q}_{name}": values for q, parts in summary.items() for name, values in parts.items()}
        columns["early_share_capped"] = capped
        table = structure_rows(st, columns, study_id=study, arm=cohort.arm)
        separated = separated_lesion_tiac(mean_tia_map(output, cohort, i), cohort, i, st, r_eval)
        table["log_tiac_separated"] = table.source.map(separated)
        predictions.append(table.merge(pd.DataFrame({"study_id": study, "virtual_id": virtual.to_numpy()}),
                                       on="study_id"))
        if maps_dir is not None:
            os.makedirs(maps_dir, exist_ok=True)
            maps = output[0].float().cpu().numpy()
            path = os.path.join(maps_dir, f"{study}.npz")
            np.savez(path + ".tmp.npz", mean=maps[0::2].astype(np.float16), logvar=maps[1::2].astype(np.float16),
                     heads=np.array(HEADS))
            os.replace(path + ".tmp.npz", path)
        log(f"[{position + 1}/{len(indices)}] {study}: {len(st)} structures, {cohort.n_draws[i]} draws")
        del output, constant
    prediction = pd.concat(predictions, ignore_index=True)
    map_truth = pd.concat(truths, ignore_index=True)
    truth_like, organ_check = _like_with_like(params, map_truth)
    return dict(network=network, prior=baseline, prediction=prediction, truth=params, truth_like=truth_like,
                organ_check=organ_check)


def _like_with_like(params, map_truth):
    """Lesion rows of params replaced by the dilated-mask summary; organ rows compared with it (largest difference)."""
    merged = params.merge(map_truth[["virtual_id", "source", *QUANTITIES]], on=["virtual_id", "source"], how="left",
                          suffixes=("", "_map"))
    lesion = (merged.structure == "lesion").to_numpy()
    rows = []
    for q in QUANTITIES:
        cohort_value, map_value = merged[q].to_numpy(float), merged[f"{q}_map"].to_numpy(float)
        ok = ~lesion & np.isfinite(cohort_value) & np.isfinite(map_value)
        if q == "log_lambda_u":
            ok &= merged.lu_target.to_numpy()
        if q == "log_lambda_bio":
            ok &= ~merged.lambda_bio_floored.to_numpy()
        difference = np.abs(map_value - cohort_value)[ok]
        rows.append(dict(quantity=q, n=int(ok.sum()), kind="absolute" if q == "early_share" else "relative",
                         largest_difference=float(np.max(difference if q == "early_share" else np.expm1(difference)))
                         if ok.any() else np.nan))
    truth_like = params.copy()
    for q in QUANTITIES:
        truth_like[q] = np.where(lesion, merged[f"{q}_map"].to_numpy(float), merged[q].to_numpy(float))
    return truth_like, pd.DataFrame(rows)


def file_md5(path):
    with open(path, "rb") as handle:
        return hashlib.md5(handle.read()).hexdigest()


def maps_size_gb(cohort, indices):
    """Size of the float16 maps (mean and log-variance of every head) of the given phantoms."""
    voxels = sum(int(np.prod(cohort.shape[int(i)])) for i in indices)
    return voxels * 2 * len(HEADS) * 2 / 1024**3


def test_file_names(netdata):
    """Shard files of the test split, from phantoms_test.parquet. Only for the frozen test evaluation."""
    return sorted(pd.read_parquet(Path(netdata) / "phantoms_test.parquet").shard.unique())


def build_test_cohort(cohort_class, nd, netdata, shard_dir, device, scenario="A", arm="main"):
    """
    The test phantoms as a cohort, for the frozen test evaluation only. lib.data refuses the test files by name, so
    the three *_test tables are linked under the names the cohort reads in a temporary folder and the name guard is
    lifted for the time of the construction only.
    """
    netdata = Path(netdata).resolve()
    view = Path(tempfile.mkdtemp(prefix="test_view_"))
    for name in ("phantoms", "sources", "params"):
        os.symlink(netdata / f"{name}_test.parquet", view / f"{name}.parquet")
    guard = data_module._guard
    data_module._guard = lambda path: path
    try:
        return cohort_class(nd, view, shard_dir, device, scenario=scenario, arm=arm, splits=("test",))
    finally:
        data_module._guard = guard


RAW, CALIBRATED, INDEPENDENT = "network", "network (calibrated)", "network (independent voxels)"


def voxel_report(results, out_dir, name, stamp, prefix="", scales=None):
    """
    Voxel-level tables and figures of one evaluate_cohort pass, saved in out_dir as <prefix>voxel_*_<name>: network
    against prior NLL, calibration summary, reliability, z histogram. scales {head: SD factor}: applied instead of
    being fitted (test). Returns (nll table, calibration summary, loss check).
    """
    summary, reliability, histogram = results["network"].tables(scales)
    prior_summary, _, _ = results["prior"].tables()
    nll = nll_against_prior(summary, prior_summary)
    check = results["network"].loss_check().merge(results["prior"].loss_check(), on="head", suffixes=("", "_prior"))
    tag = f"{out_dir}/{prefix}voxel"
    for table, kind in ((nll, "nll"), (summary, "calibration"), (reliability, "reliability"),
                        (histogram, "z_histogram"), (check, "loss_check")):
        save_table(table, f"{tag}_{kind}_{name}.csv", stamp)
    figures.reliability_figure(reliability, f"{tag}_reliability_{name}.png", stamp)
    figures.z_histogram_figure(histogram, summary, f"{tag}_z_histogram_{name}.png", stamp)
    return nll, summary, check


def structure_report(nd, results, baselines, phantoms, out_dir, name, stamp, prefix="", calibration=None, folds=5,
                     seed=20261114):
    """
    Structure-level tables and figures, saved in out_dir as <prefix>*_<name>. calibration None (validation): the scales
    are fitted here, the calibrated network is scaled out of fold (cross-validation by subject) and the scales fitted on
    all the phantoms are saved as calibration_<name>.csv. calibration given (test): applied unchanged, nothing fitted.
    baselines: {'population': table, 'pbpk_fit': table} and, in a B scenario, scenario.CEILING: the other draws of the
    same phantom in that scenario (the washout table then reports the network against that ceiling). Returns a dict of tables.
    """
    truth, truth_like = results["truth"], results["truth_like"]
    scenario = truth.scenario.iloc[0]
    raw = with_sd(results["prediction"], "sd")
    out = dict(organ_check=results["organ_check"])
    if calibration is None:
        fold_of_study = subject_folds(phantoms, raw.study_id.unique(), folds, seed)
        calibrated, fold_scales = cross_validated(raw, truth_like, fold_of_study)
        out["calibration"] = fit_scales(raw, truth_like)
        out["calibration_folds"] = fold_scales
        save_table(out["calibration"], f"{out_dir}/calibration_{name}.csv", stamp)
        save_table(fold_scales, f"{out_dir}/calibration_folds_{name}.csv", stamp)
    else:
        calibrated = apply_scales(raw, calibration)
    lesion = raw.structure == "lesion"
    separated = raw.loc[lesion, KEYS + ["log_tiac_sd"]].assign(
        log_tiac_mean=results["prediction"].loc[lesion, "log_tiac_separated"].to_numpy())
    methods = {RAW: (raw, truth_like), CALIBRATED: (calibrated, truth_like),
               INDEPENDENT: (with_sd(results["prediction"], "sd_independent"), truth_like),
               **{key: (table[table.arm == truth.arm.iloc[0]], truth) for key, table in baselines.items()},
               "network, lesions against the cohort value": (raw[lesion], truth),
               "network, separated lesion TIAC against the cohort value": (separated, truth)}
    out["comparison"] = compare_methods(nd, methods, scenario)
    out["washout"] = washout_check(out["comparison"], RAW, ceiling=CEILING if CEILING in baselines else None,
                                   rho=scenario_rho(scenario))
    out["prediction"] = results["prediction"]
    out["lesion_overlap"] = lesion_overlap(results["prediction"])
    out["early_share_cap"] = early_share_cap(results["prediction"])
    out["calibrated"] = calibrated
    for key in ("prediction", "organ_check", "comparison", "washout", "lesion_overlap", "early_share_cap"):
        save_table(out[key], f"{out_dir}/{prefix}structure_{key}_{name}.csv", stamp)
    shown = {RAW: methods[RAW], **{key: methods[key] for key in baselines if key != CEILING}}
    figures.tiac_scatter_figure(shown, f"{out_dir}/{prefix}tiac_scatter_{name}.png", stamp)
    both = out["comparison"][out["comparison"].method.isin([RAW, CALIBRATED])]
    figures.coverage_figure(both, RAW, CALIBRATED, f"{out_dir}/{prefix}structure_coverage_{name}.png", stamp)
    return out
