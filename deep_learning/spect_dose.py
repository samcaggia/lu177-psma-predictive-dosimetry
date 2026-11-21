"""
Absorbed dose from the TIAC of the single-SPECT experiment (no torch): reporting half of the validation subjects,
main condition and shape mismatch. Dose of a target = sum over the sources of TIAC_source x the Monte Carlo dose basis
of the phantom (src/cohort_doses.py, notebook 21: organ sources from the stored basis, every lesion through the split
native kernel with its own TIAC), cross-dose included; salivary glands linked to their Monte Carlo source as in the
cohort. Truth = the same operator on the true (perturbed) TIAC. Uncertainty by sampling the log TIAC of every source
(independent normals with the SD of the method). Writes runs/eval_<scenario>/dose/.
With SPECT_STAGE=test (frozen test run): the test virtual patients (cohort table tia_test), outputs test_*.
Usage: [SPECT_STAGE=test] python spect_dose.py
"""
import json
import sys
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
import cohort_doses as cd  # noqa: E402
import spect_report as report  # noqa: E402
from lib import frozen  # noqa: E402

SCENARIO = report.SCENARIO
OUT = report.EVAL / "dose"
PREFIX, LABEL = report.PREFIX, report.LABEL
CONDITIONS = {"mismatch SD 0.15 (main)": "mismatch", "shape mismatch": "shape"}
MAIN = report.MAIN
ORGAN_TARGETS = ["kidneys_functional", "liver", "spleen", "salivary_glands", "spine_L2_L4", "whole_body"]
TARGETS = ["kidneys_functional", "liver", "spleen", "salivary_glands", "lesion", "spine_L2_L4", "whole_body"]
OWN_SOURCE = {"kidneys_functional": "kidneys_functional", "liver": "liver", "spleen": "spleen", "salivary_glands": "salivary_glands",
              "spine_L2_L4": "spine"}
M0, M0P, M1, M2, M2P, M3, M4, M5 = (report.M0, report.M0P, report.M1, report.M2, report.M2P, report.M3, report.M4, report.M5)
METHODS = [M0, M0P, M1, M2, M2P, M3, M4, M5]
COLOURS = {**report.COLOURS, M0P: "#999999", M2P: "#56B4E9"}
MARKERS = {**report.MARKERS, M0P: None, M2P: "o"}
SAMPLES, SEED, CYCLE_GBQ = 400, 20261116, 7.4


def dose_matrix(study_id):
    """
    Gy/GBq per hour of TIAC: rows = ORGAN_TARGETS then every lesion label, columns = the organ sources of the basis
    then every placed lesion (cohort index = label - 1). Cached per phantom.
    """
    path = OUT / "matrix" / f"{study_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return z["matrix"], list(z["targets"]), list(z["sources"]), float(z["gland_factor"])
    basis = cd.StoredBasis(study_id)
    rho, body = basis.density, basis.body
    n_lesions = int(basis.lesion_index.max())

    def column(dose):
        targets, lesions = basis.summarise(dose)
        whole = float((dose[body] * rho[body]).sum() / rho[body].sum())
        return np.concatenate([[targets[t] for t in ORGAN_TARGETS[:-1]], [whole], lesions])

    sources, columns = [], []
    for name, b in basis.basis.items():
        sources.append(name)
        columns.append(column(b.astype(np.float64)))
    for label in sorted(basis.lesion_w):
        sources.append(f"lesion_{label - 1}")
        columns.append(column(basis.lesion_map({label - 1: 1.0})))
    matrix = np.stack(columns, axis=1) * cd.MAP_SCALE
    targets = ORGAN_TARGETS + [f"lesion_{j}" for j in range(n_lesions)]
    factor = float(cd.gland_link([study_id]).iloc[0])
    np.savez(path, matrix=matrix, targets=np.array(targets), sources=np.array(sources), gland_factor=factor)
    return matrix, targets, sources, factor


def linked(tiac, sources, factor):
    """TIAC per Monte Carlo source: glands scaled to their mask, the excess taken from the rest of body (cohort_doses.linked_tia)."""
    out = tiac.copy()
    if "salivary_glands" in sources and "rest_of_body" in sources:
        g, r = sources.index("salivary_glands"), sources.index("rest_of_body")
        excess = tiac[..., g] * (factor - 1)
        out[..., g] += excess
        out[..., r] = np.maximum(out[..., r] - excess, 0.0)
    return out


def case_rows(table, cohort_tiac):
    """Dose rows of one condition: per method, SPECT time, virtual patient and target."""
    rows = []
    prior_methods = set(table[table.t_h.isna()].method)
    times = sorted(table.t_h.dropna().unique())
    for virtual_id, patient in table.groupby("virtual_id", sort=True):
        study_id = virtual_id.rsplit("_", 1)[0]
        matrix, targets, sources, factor = dose_matrix(study_id)
        fallback = cohort_tiac.loc[virtual_id].reindex(sources).fillna(0.0).to_numpy()
        truth_rows = patient[patient.method == M0].set_index("source").truth.reindex(sources)
        truth = np.where(truth_rows.notna(), np.exp(truth_rows.to_numpy(float)), fallback)
        true_dose = matrix @ linked(truth, sources, factor)
        own = {t: (sources.index(OWN_SOURCE.get(t, t)) if OWN_SOURCE.get(t, t) in sources else -1) for t in targets}
        true_self = np.array([matrix[k, own[t]] * linked(truth, sources, factor)[own[t]] if own[t] >= 0 else np.nan
                              for k, t in enumerate(targets)])
        z = np.random.default_rng([SEED, zlib.crc32(virtual_id.encode())]).standard_normal((SAMPLES, len(sources)))
        for method, by_method in patient.groupby("method", sort=False):
            for t_h in times:
                g = by_method if method in prior_methods else by_method[by_method.t_h == t_h]
                if g.empty:
                    continue
                g = g.set_index("source").reindex(sources)
                known = g["mean"].notna().to_numpy()
                tiac = np.where(known, np.exp(g["mean"].to_numpy(float)), fallback)
                dose = matrix @ linked(tiac, sources, factor)
                sd, n = g.sd.to_numpy(float), g.n.to_numpy(float)
                sd = sd * np.where(np.isfinite(n) & (n > 1), np.sqrt(1 + 1 / np.where(np.isfinite(n) & (n > 1), n, 1.0)), 1.0)
                quantiles = np.full((4, len(targets)), np.nan)
                if np.isfinite(sd[known]).all():
                    samples = np.where(known, tiac * np.exp(np.where(known, sd, 0.0) * z), fallback)
                    quantiles = np.percentile(np.log(np.maximum(linked(samples, sources, factor) @ matrix.T, 1e-300)), [16, 84, 2.5, 97.5], axis=0)
                for k, target in enumerate(targets):
                    if not (true_dose[k] > 0 and dose[k] > 0):
                        continue
                    j = own[target]
                    rows.append(dict(virtual_id=virtual_id, study_id=study_id, method=method, t_h=t_h,
                                     target="lesion" if target.startswith("lesion_") else target, name=target,
                                     dose_true=true_dose[k], dose=dose[k], self_true=true_self[k],
                                     self_dose=matrix[k, j] * linked(tiac, sources, factor)[j] if j >= 0 else np.nan,
                                     tiac_true=truth[j] if j >= 0 else np.nan, tiac=tiac[j] if j >= 0 else np.nan,
                                     q16=quantiles[0, k], q84=quantiles[1, k], q025=quantiles[2, k], q975=quantiles[3, k]))
    return pd.DataFrame(rows)


def score(rows):
    out = []
    for key, g in rows.groupby(["condition", "method", "target", "t_h"], sort=False):
        error = np.log(g.dose / g.dose_true)
        tiac_error = np.log(g.tiac / g.tiac_true)
        row = dict(zip(["condition", "method", "target", "t_h"], key), n_cases=len(g), virtual_patients=g.virtual_id.nunique(),
                   bias=float(error.mean()), median_abs_error=float(error.abs().median()),
                   median_abs_tiac_error=float(tiac_error.abs().median()) if tiac_error.notna().any() else np.nan,
                   median_true_gy_per_gbq=float(g.dose_true.median()), median_true_gy_cycle=float(g.dose_true.median() * CYCLE_GBQ),
                   median_abs_error_gy_cycle=float((g.dose - g.dose_true).abs().median() * CYCLE_GBQ),
                   cross_dose_share=float((1 - g.self_true / g.dose_true).median()) if g.self_true.notna().any() else np.nan)
        if g.q16.notna().all():
            truth = np.log(g.dose_true)
            row["coverage_68"] = float(((truth > g.q16) & (truth < g.q84)).mean())
            row["coverage_95"] = float(((truth > g.q025) & (truth < g.q975)).mean())
        out.append(row)
    return pd.DataFrame(out)


def error_figure(summary, methods, path):
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6), sharex=True)
    for ax, target in zip(axes.flat, TARGETS):
        d = summary[summary.target == target]
        for method in methods:
            g = d[d.method == method].sort_values("t_h")
            if len(g):
                ax.plot(g.t_h, g.median_abs_error, color=COLOURS[method], marker=MARKERS[method], lw=2, ms=6,
                        ls="--" if method in (M0, M0P) else "-", label=method.removesuffix(", gated"))
        ax.set_title(target.replace("_", " "), fontsize=10)
        ax.set_xticks([24, 48, 96, 168])
        ax.set_ylim(0, max(0.3, d[d.method.isin(methods)].median_abs_error.max() * 1.08))
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[7].axis("off")
    axes.flat[7].legend(handles, labels, loc="center", frameon=False)
    for ax in list(axes[1, :3]) + [axes[0, 3]]:
        ax.set_xlabel("SPECT time (h)")
        ax.tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("median |log dose error|")
    fig.suptitle(f"Absorbed dose from one SPECT, {LABEL}, main condition: error against the SPECT time")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def scatter_figure(rows, path, t_h=96.0):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4))
    for row, target in zip(axes, ("kidneys_functional", "lesion")):
        d = rows[(rows.target == target) & (rows.t_h == t_h) & (rows.condition == MAIN)]
        low, high = d.dose_true.min() * CYCLE_GBQ / 2, d.dose_true.max() * CYCLE_GBQ * 2
        for ax, method in zip(row, (M0, M2, M5)):
            g = d[d.method == method]
            ax.plot([low, high], [low, high], color="#999999", lw=1)
            ax.scatter(g.dose_true * CYCLE_GBQ, g.dose * CYCLE_GBQ, s=14 if target == "lesion" else 28, color=COLOURS[method], alpha=0.7, lw=0)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            ax.set_aspect("equal")
            ax.set_title(f"{target.replace('_', ' ')}: {method.removesuffix(', gated')}" + ("" if method == M0 else f", SPECT at {t_h:g} h"), fontsize=10)
            ax.set_xlabel(f"true dose (Gy per {CYCLE_GBQ:g} GBq)")
            ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
        row[0].set_ylabel(f"predicted dose (Gy per {CYCLE_GBQ:g} GBq)")
    fig.suptitle(f"Predicted against true absorbed dose, {LABEL}, main condition (grey: identity)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ranking(summary):
    """Per condition, target and SPECT time: rank of every method by dose error and by TIAC error of the own source."""
    d = summary[summary.median_abs_tiac_error.notna() & ~summary.method.isin([M0, M0P])].copy()
    keys = ["condition", "target", "t_h"]
    d["rank_dose"] = d.groupby(keys).median_abs_error.rank()
    d["rank_tiac"] = d.groupby(keys).median_abs_tiac_error.rank()
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    choice = json.loads((report.EVAL / f"tto_choice_{SCENARIO}.json").read_text())
    stamp = frozen.check_manifest(report.EVAL, frozen.netdata_dir()) if report.STAGE == "test" else {}
    tia = pd.read_parquet(frozen.cohort_dir() / f"tia_{'test' if report.STAGE == 'test' else 'train'}.parquet")
    tia = tia[(tia.scenario == SCENARIO) & (tia.arm == "main")].set_index(["virtual_id", "compartment"]).tiac_h
    all_rows = []
    for name, tag in CONDITIONS.items():
        table = report.load(tag, choice)
        table = table[table.method.isin(METHODS)]
        all_rows.append(case_rows(table, tia).assign(condition=name))
        print(name, "done", flush=True)
    rows = pd.concat(all_rows, ignore_index=True)
    rows.assign(**stamp).to_csv(OUT / f"{PREFIX}dose_rows_{SCENARIO}.csv", index=False)
    summary = score(rows)
    summary.assign(**stamp).to_csv(OUT / f"{PREFIX}dose_table_{SCENARIO}.csv", index=False)
    ranks = ranking(summary)
    ranks.to_csv(OUT / f"{PREFIX}dose_vs_tiac_ranking_{SCENARIO}.csv", index=False)
    main_rows = summary[summary.condition == MAIN]
    methods = [m for m in METHODS if (main_rows.method == m).any()]
    error_figure(main_rows, methods, OUT / f"{PREFIX}dose_error_vs_time_{SCENARIO}.png")
    scatter_figure(rows, OUT / f"{PREFIX}dose_predicted_vs_true_{SCENARIO}.png")
    pd.set_option("display.width", 250, "display.max_rows", 400, "display.max_columns", 60)
    for name in CONDITIONS:
        d = summary[summary.condition == name]
        for column in ("median_abs_error", "median_abs_tiac_error", "coverage_68", "coverage_95", "median_abs_error_gy_cycle"):
            print(f"\n== {name}: {column} ==\n" + d.pivot_table(index=["method", "t_h"], columns="target", values=column)
                  .reindex(columns=TARGETS).reindex(methods, level=0).round(2).to_string())
    d = main_rows[(main_rows.method == M0) & (main_rows.t_h == main_rows.t_h.min())]
    print("\nTrue dose, median, and cross-dose share\n" + d.set_index("target")[["n_cases", "median_true_gy_per_gbq", "median_true_gy_cycle",
                                                                                 "cross_dose_share"]].reindex(TARGETS).round(3).to_string())
    changed = ranks[ranks.rank_dose != ranks.rank_tiac]
    print("\nRank changes between TIAC and dose:", len(changed), "of", len(ranks))
    print(ranks.groupby(["condition", "target"]).apply(lambda g: (g.rank_dose != g.rank_tiac).mean()).round(2).to_string())


if __name__ == "__main__":
    main()
