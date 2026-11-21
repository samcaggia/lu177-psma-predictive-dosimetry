"""
Single-SPECT update, final report on the reporting half of the validation subjects (no torch): gathers the rows of
M0, M1, M2 (spect_update.py), M3 (spect_update_maps.py), M4 (notebook section 11, if its table is there) and M5
(spect_update_pbpk.py) for every condition, applies the gates of tto_choice_<scenario>.json, and writes the final table
(median |error|, bias, coverage of the 68% and 95% intervals of log TIAC per structure, method and SPECT time), the
figures and a markdown summary. Nothing is chosen here. If the population-prior runs of M2 are there (spect_update.py
... population), it adds them and the paired difference gated prior minus population prior, with a bootstrap over subjects.
The scenario comes from the environment variable SCENARIO (default A); methods and conditions without a table are left
out (a B scenario has no M3 or M4). The same-phantom-draws and PBPK-fit prior runs of M2 are added when they are there.
With SPECT_STAGE=test (frozen test run, scenario A) it reads the test_* / *_test_* rows of every method and writes
test_final_* files; the frozen files must match test_freeze_A.json.
Usage: [SCENARIO=<scenario>] [SPECT_STAGE=test] python spect_report.py
"""
import json
import os
import sys
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
import net_data as nd  # noqa: E402
from lib import frozen  # noqa: E402

SCENARIO = os.environ.get("SCENARIO", "A")
STAGE = os.environ.get("SPECT_STAGE", "reporting")
PREFIX = "test_" if STAGE == "test" else ""
LABEL = "test split" if STAGE == "test" else "reporting half"
EVAL = frozen.eval_dir(SCENARIO)
OUT = EVAL / "spect"
CONDITIONS = {"mismatch SD 0.15 (main)": "mismatch", "mismatch SD 0.10": "mismatch_sd0.1", "mismatch SD 0.20": "mismatch_sd0.2",
              "shape mismatch": "shape"}
MAIN = "mismatch SD 0.15 (main)"
STRUCTURES = ["kidneys_functional", "liver", "spleen", "salivary_glands", "gi_tract", "lesion", "blood_pool", "spine",
              "rest_of_body", "bladder"]
UPTAKE = STRUCTURES[:6]
M0, M1, M2, M3, M4, M5 = ("M0 gated prior", "M1 population rate, gated", "M2 Bayes, gated", "M3 maps, gated", "M4 weights",
                          "M5 PBPK")
COLOURS = {M0: "#555555", M1: "#E69F00", M2: "#0072B2", M3: "#009E73", M4: "#CC79A7", M5: "#D55E00"}
MARKERS = {M0: None, M1: "s", M2: "o", M3: "^", M4: "D", M5: "v"}
M0P, M2P = "M0 population prior", "M2 Bayes, population prior, gated"
M0F, M2F = "M0 PBPK-fit prior", "M2 Bayes, PBPK-fit prior, gated"
M0C, M2C = "M0 same-phantom-draws prior", "M2 Bayes, same-phantom-draws prior, gated"
PAIRS = ((M2, M2P), (M0, M0P), (M2C, M2P), (M0C, M0P), (M2C, M2F), (M0C, M0F), (M2, M2C), (M0, M0C))
PRIOR_RUNS = ("popprior", "pbpkprior", "ceilingprior")
BOOTSTRAP, SEED = 2000, 20261115
KEYS = ["virtual_id", "source", "structure", "t_h", "quantity"]


def gate_at(gate, t_h):
    return bool(gate[min(gate, key=lambda k: (abs(float(k) - t_h), float(k)))]) if gate else False


def gated(rows, prior, gate, name):
    """Rows of a method where its gate is open, the M0 rows elsewhere."""
    keep = np.array([gate_at(gate.get(s, {}), t) for s, t in zip(rows.structure, rows.t_h)])
    fallback = rows[~keep][KEYS].merge(prior.drop(columns=["t_h", "structure"]), on=["virtual_id", "source", "quantity"])
    return pd.concat([rows[keep], fallback], ignore_index=True).assign(method=name)


def joined(path):
    """path, written first from its parallel pieces <stem>.<k>.csv if it is missing and they are there."""
    if not path.exists():
        pieces = [p for p in sorted(path.parent.glob(f"{path.stem}.*.csv")) if "partial" not in p.name]
        if pieces:
            pd.concat([pd.read_csv(p) for p in pieces], ignore_index=True).to_csv(path, index=False)
    return path


def load(tag, choice):
    """All the log TIAC rows of one condition, columns KEYS + method, mean, sd, n, truth."""
    columns = KEYS + ["method", "mean", "sd", "n", "truth"]
    table = pd.read_csv(OUT / f"update_rows_{STAGE}_{tag}_{SCENARIO}.csv")
    table = table[table.quantity == "log_tiac"]
    prior = table[table.method == M0]
    parts = [table[columns]]
    for name in PRIOR_RUNS:
        other = OUT / f"update_rows_{STAGE}_{tag}_{name}_{SCENARIO}.csv"
        if other.exists():
            rows = pd.read_csv(other)
            parts.append(rows[rows.quantity == "log_tiac"][columns])
    pbpk = joined(OUT / f"pbpk_rows_{STAGE}_{tag}_{SCENARIO}.csv")
    if pbpk.exists():
        m5 = pd.read_csv(pbpk)
        parts.append(m5[(m5.quantity == "log_tiac") & (m5.method == M5)][columns])
    maps = joined(OUT / (f"maps_rows_{STAGE}_{tag}_{SCENARIO}_lr{choice['M3']['lr']:g}_rs{choice['M3']['residual_scale']:g}.csv"
                         if "M3" in choice else "no_maps"))
    if maps.exists():
        m3 = pd.read_csv(maps)
        m3 = m3[(m3.quantity == "log_tiac") & (m3.step == choice["M3"]["steps"])].assign(sd=np.nan, n=np.nan)
        m3_prior = prior[prior.virtual_id.isin(set(m3.virtual_id))]
        parts += [m3.assign(method="M3 maps")[columns], gated(m3, m3_prior, choice["M3"]["gate"], M3)[columns]]
    weights = OUT / (f"test_weights_rows_{SCENARIO}.csv" if STAGE == "test" else f"weights_rows_reporting_{SCENARIO}.csv")
    if tag == "mismatch" and weights.exists() and "M4" in choice:
        m4 = pd.read_csv(weights)
        m4 = m4[(m4.quantity == "log_tiac") & (m4.step == choice["M4"]["steps"]) & m4.method.str.startswith("M4")]
        parts.append(m4.assign(method=M4, sd=np.nan, n=np.nan)[columns])
    return pd.concat(parts, ignore_index=True)


def score(table, by=("method", "structure", "t_h")):
    rows = []
    for key, g in table.groupby(list(by), sort=False, dropna=False):
        error = g["mean"] - g.truth
        row = dict(zip(by, key), n_cases=len(g), virtual_patients=g.virtual_id.nunique(), bias=float(error.mean()),
                   median_abs_error=float(error.abs().median()))
        if g.sd.notna().all():
            for level in (0.68, 0.95):
                half = nd.predictive_halfwidth(g.sd.to_numpy(float), g.n.to_numpy(float), level)
                row[f"coverage_{int(level * 100)}"] = float((error.abs().to_numpy() < half).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_difference(table, first, second, replicates=BOOTSTRAP, seed=SEED):
    """
    Median |log TIAC error| of `first` minus that of `second` on the same cases, per structure and SPECT time, and for
    the mean over the structures with uptake; 95% percentile interval from a bootstrap over subjects (all the cases of
    a drawn subject enter together, the same draws for every structure and time).
    """
    a = table[table.method == first].set_index(["virtual_id", "source", "t_h"] if table[table.method == first].t_h.notna().all()
                                               else ["virtual_id", "source"])
    b = table[table.method == second].set_index(a.index.names)
    both = a[["structure", "mean", "truth"]].join(b[["mean"]].rename(columns={"mean": "other"}), how="inner").reset_index()
    both["ea"], both["eb"] = (both["mean"] - both.truth).abs(), (both.other - both.truth).abs()
    both["subject"] = both.virtual_id.str.split("_").str[1]
    if "t_h" not in both:
        both["t_h"] = np.nan
    subjects = sorted(both.subject.unique())
    draws = np.random.default_rng(seed).integers(0, len(subjects), size=(replicates, len(subjects)))
    rows, uptake = [], {}
    for (structure, t_h), g in both.groupby(["structure", "t_h"], dropna=False):
        index = [np.flatnonzero(g.subject.to_numpy() == s) for s in subjects]
        ea, eb = g.ea.to_numpy(), g.eb.to_numpy()
        boot = np.full(replicates, np.nan)
        for r, draw in enumerate(draws):
            chosen = np.concatenate([index[k] for k in draw])
            if len(chosen):
                boot[r] = np.median(ea[chosen]) - np.median(eb[chosen])
        rows.append(dict(structure=structure, t_h=t_h, n_cases=len(g), subjects=g.subject.nunique(), first=np.median(ea),
                         second=np.median(eb), difference=np.median(ea) - np.median(eb), low=np.nanpercentile(boot, 2.5),
                         high=np.nanpercentile(boot, 97.5)))
        if structure in UPTAKE:
            uptake.setdefault(-1.0 if np.isnan(t_h) else t_h, []).append((np.median(ea), np.median(eb), boot))
    for t_h, parts in uptake.items():
        boot = np.mean([p[2] for p in parts], axis=0)
        first_mean, second_mean = np.mean([p[0] for p in parts]), np.mean([p[1] for p in parts])
        rows.append(dict(structure="mean of the six with uptake", t_h=np.nan if t_h < 0 else t_h, n_cases=0, subjects=len(subjects), first=first_mean,
                         second=second_mean, difference=first_mean - second_mean, low=np.nanpercentile(boot, 2.5),
                         high=np.nanpercentile(boot, 97.5)))
    return pd.DataFrame(rows).assign(first_method=first, second_method=second)


def difference_markdown(d):
    """One row per structure, one column per SPECT time: difference [low, high], in bold where the interval excludes 0."""
    times = sorted(d.t_h.dropna().unique()) or [np.nan]
    lines = ["| structure | " + " | ".join("no SPECT" if np.isnan(t) else f"{t:g} h" for t in times) + " |", "|" + "---|" * (1 + len(times))]
    for structure in STRUCTURES + ["mean of the six with uptake"]:
        cells = []
        for t in times:
            g = d[(d.structure == structure) & (d.t_h.isna() if np.isnan(t) else d.t_h == t)]
            if len(g):
                r = g.iloc[0]
                cell = f"{r.difference:+.2f} [{r.low:+.2f}, {r.high:+.2f}]"
                cells.append(f"**{cell}**" if r.low > 0 or r.high < 0 else cell)
            else:
                cells.append("")
        lines.append("| " + structure.replace("_", " ") + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def expand_prior(summary, times):
    """The M0 rows (no SPECT time) repeated at every time."""
    prior = summary[summary.t_h.isna()]
    return pd.concat([summary[summary.t_h.notna()]] + [prior.assign(t_h=t) for t in times], ignore_index=True)


def error_figure(summary, methods, path):
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.4), sharex=True)
    for ax, structure in zip(axes.flat, STRUCTURES):
        d = summary[summary.structure == structure]
        for method in methods:
            g = d[d.method == method].sort_values("t_h")
            if len(g):
                ax.plot(g.t_h, g.median_abs_error, color=COLOURS[method], marker=MARKERS[method], lw=2, ms=6,
                        ls="--" if method == M0 else "-", label=method)
        ax.set_title(structure.replace("_", " "), fontsize=10)
        ax.set_xticks([24, 48, 96, 168])
        ax.set_ylim(0, min(1.0, max(0.5, d[d.method.isin(methods)].median_abs_error.max() * 1.05)))
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("SPECT time (h)")
    for ax in axes[:, 0]:
        ax.set_ylabel("median |log TIAC error|")
    handles, labels = axes.flat[5].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle(f"Single-SPECT update, {LABEL}, main condition: error of the TIAC against the SPECT time (panels clipped at 1)")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def coverage_figure(summary, methods, path):
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.4), sharex=True, sharey=True)
    for ax, structure in zip(axes.flat, STRUCTURES):
        d = summary[summary.structure == structure]
        for level, style in ((68, "-"), (95, ":")):
            ax.axhline(level / 100, color="#999999", lw=1, ls=style)
            for method in methods:
                g = d[d.method == method].sort_values("t_h")
                if len(g) and f"coverage_{level}" in g:
                    ax.plot(g.t_h, g[f"coverage_{level}"], color=COLOURS[method], marker=MARKERS[method] or "x", lw=2, ms=6, ls=style,
                            label=method if level == 68 else None)
        ax.set_title(structure.replace("_", " "), fontsize=10)
        ax.set_xticks([24, 48, 96, 168])
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("SPECT time (h)")
    for ax in axes[:, 0]:
        ax.set_ylabel("coverage of the interval")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle(f"Coverage of the 68% (solid) and 95% (dotted) intervals of log TIAC, {LABEL}, main condition; grey: nominal")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pivot(summary, methods, column, structures=STRUCTURES):
    d = summary[summary.method.isin(methods) & summary.structure.isin(structures)]
    table = d.pivot_table(index="structure", columns=["method", "t_h"], values=column).reindex(structures)
    return table.reindex(columns=[m for m in methods if m in table.columns.get_level_values(0)], level=0)


def markdown(table, digits=2):
    header = ["structure"] + [f"{m.removesuffix(', gated')} {t:g} h" for m, t in table.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, row in table.iterrows():
        lines.append("| " + " | ".join([name.replace("_", " ")] + [("" if np.isnan(v) else f"{v:.{digits}f}") for v in row]) + " |")
    return "\n".join(lines)


def main():
    choice = json.loads((EVAL / f"tto_choice_{SCENARIO}.json").read_text())
    stamp = frozen.check_manifest(EVAL, frozen.netdata_dir()) if STAGE == "test" else {}
    summaries, tables = [], {}
    for name, tag in CONDITIONS.items():
        if not (OUT / f"update_rows_{STAGE}_{tag}_{SCENARIO}.csv").exists():
            continue
        tables[name] = load(tag, choice)
        times = sorted(tables[name].t_h.dropna().unique())
        summaries.append(expand_prior(score(tables[name]), times).assign(condition=name))
    summary = pd.concat(summaries, ignore_index=True)
    summary = summary[["condition", "method", "structure", "t_h", "n_cases", "virtual_patients", "bias", "median_abs_error",
                       "coverage_68", "coverage_95"]].sort_values(["condition", "method", "structure", "t_h"])
    summary.assign(**stamp).to_csv(OUT / f"{PREFIX}final_table_{SCENARIO}.csv", index=False)
    main_rows = summary[summary.condition == MAIN]
    methods = [m for m in (M0, M1, M2, M3, M4, M5) if (main_rows.method == m).any()]
    error_figure(main_rows, methods, OUT / f"{PREFIX}final_error_vs_time_{SCENARIO}.png")
    coverage_figure(main_rows, [m for m in (M0, M2, M5) if m in methods], OUT / f"{PREFIX}final_coverage_{SCENARIO}.png")

    text = [f"# Single-SPECT update, scenario {SCENARIO}: {LABEL}", "",
            f"Written by `spect_report.py`; every number comes from `{PREFIX}final_table_{SCENARIO}.csv`. Median |log TIAC error| "
            "unless stated; an error of 0.10 in log is about 10%.", ""]
    if stamp:
        text += ["Frozen files (md5): " + ", ".join(f"{k} {v}" for k, v in stamp.items()), ""]
    for name in tables:
        d = summary[summary.condition == name]
        shown = [m for m in (M0, M0P, M0F, M0C, M1, M2, M2P, M2F, M2C, M3, M4, M5) if (d.method == m).any()]
        text += [f"## {name}", "", "Median |log TIAC error|", "", markdown(pivot(d, shown, "median_abs_error")), ""]
        for level in (68, 95):
            text += [f"Coverage of the {level}% interval", "",
                     markdown(pivot(d, [m for m in (M0, M2, M2P, M2F, M2C, M5) if m in shown], f"coverage_{level}")), ""]
    if M4 in methods:
        cases = tables[MAIN][tables[MAIN].method == M4][["virtual_id", "source", "t_h"]]
        same = tables[MAIN].merge(cases, on=["virtual_id", "source", "t_h"])
        prior = tables[MAIN][tables[MAIN].method == M0].merge(cases[["virtual_id", "source"]].drop_duplicates(), on=["virtual_id", "source"])
        d = expand_prior(score(pd.concat([same, prior])), sorted(cases.t_h.unique()))
        d.assign(condition=MAIN + ", cases of M4").to_csv(OUT / f"{PREFIX}final_table_m4_cases_{SCENARIO}.csv", index=False)
        text += ["## Main condition, the cases of M4 only (one draw per phantom)", "",
                 markdown(pivot(d, [M0, M2, M3, M4, M5], "median_abs_error")), ""]
    differences = []
    for name in tables:
        for first, second in PAIRS:
            if (tables[name].method == second).any() and (tables[name].method == first).any():
                differences.append(paired_difference(tables[name], first, second).assign(condition=name))
    if differences:
        differences = pd.concat(differences, ignore_index=True)
        differences.to_csv(OUT / f"{PREFIX}final_prior_contribution_{SCENARIO}.csv", index=False)
        text += ["## Contribution of the pre-therapy prior: paired differences between priors", "",
                 f"Difference of the median |log TIAC error| on the same virtual patients (negative: the first prior is better), with "
                 f"the 95% interval of a bootstrap over subjects ({BOOTSTRAP} replicates); bold where the interval excludes 0. M2 with "
                 "the same gate, discrepancy and seeds; where the gate is closed each arm keeps its own prior. Gated prior: the "
                 "network where it was chosen; population prior: no PET; PBPK-fit prior: the PET-fit posterior of the phantom "
                 "without the PET-washout link; same-phantom-draws prior: everything the phantom can tell, link included.", ""]
        for name in differences.condition.unique():
            for first, second in PAIRS:
                d = differences[(differences.condition == name) & (differences.first_method == first) & (differences.second_method == second)]
                if len(d):
                    text += [f"{name}: {first} minus {second}", "", difference_markdown(d), ""]
    mean = summary[summary.structure.isin(UPTAKE)].groupby(["condition", "method", "t_h"]).median_abs_error.mean().unstack("t_h")
    text += ["## Mean over kidneys, liver, spleen, glands, GI and lesions of the median |log TIAC error|", "",
             "| condition | method | " + " | ".join(f"{t:g} h" for t in mean.columns) + " |", "|" + "---|" * (2 + mean.shape[1])]
    text += ["| " + " | ".join([c, m] + [("" if np.isnan(v) else f"{v:.3f}") for v in row]) + " |" for (c, m), row in mean.iterrows()] + [""]
    (OUT / f"{PREFIX}final_tables_{SCENARIO}.md").write_text("\n".join(text))
    pd.set_option("display.width", 250, "display.max_rows", 400, "display.max_columns", 60)
    for name in tables:
        d = summary[summary.condition == name]
        shown = [m for m in (M0, M0P, M0F, M0C, "M1 Hanscheid", M1, "M2 Bayes", M2, M2P, M2F, M2C, "M3 maps", M3, M4, M5) if (d.method == m).any()]
        print(f"\n===== {name}: median |log TIAC error| =====\n" + pivot(d, shown, "median_abs_error").round(2).T.to_string())
        for level in (68, 95):
            print(f"\n{name}: coverage {level}%\n" + pivot(d, [M0, "M2 Bayes", M2, M5], f"coverage_{level}").round(2).T.to_string())
        print(f"\n{name}: bias\n" + pivot(d, [m for m in (M0, M2, M5) if m in shown], "bias").round(2).T.to_string())
    if len(differences):
        print("\nGated prior minus population prior\n" + differences.round(3).to_string())
    print("\nMean over the six structures with uptake\n" + mean.round(3).to_string())
    print("\nCases:\n" + summary.groupby(["condition", "method"]).agg(cases=("n_cases", "max"), patients=("virtual_patients", "max")).to_string())


if __name__ == "__main__":
    main()
