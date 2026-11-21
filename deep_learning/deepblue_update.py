"""
Single-time-point TIA on real kinetics: the two Deep Blue patients (177Lu-DOTATATE, four SPECT/CT each), every scan
used alone, with the methods of the single-SPECT experiment: M1 (Hanscheid 2t/ln2; mono-exponential with a population
effective rate) and M2 (lib.update.bayes_update, the same code and VOI likelihood: 5% calibration plus the relative
model discrepancy of tto_choice_<scenario>.json per structure type). Priors from the DOTATATE literature as collected
in notebook 07 (kinetics/bayes_stp_priors.csv: Hanscheid 2018 for liver, spleen, lesions, Devasia 2021 for kidneys),
never the PSMA cohort; no prior on the amplitude (no pre-therapy PET used). Reference: the four-point uptake-washout
TIA of notebook 02 with its Monte Carlo 68% interval. Two patients: a demonstration on real kinetics, not a validation.
Usage: python deepblue_update.py
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from lib import spect, update  # noqa: E402
from lib.kinetics import LAMBDA_PHYS_H  # noqa: E402

KINETICS = ROOT / "data" / "processed" / "deepblue" / "kinetics"
CHOICE = HERE / "runs" / "eval_A" / "tto_choice_A.json"
DISCREPANCY_OF = {"kidney": "kidneys_functional", "liver": "liver", "spleen": "spleen", "lesion": "lesion"}
FLAT_SD = 10.0
METHODS = ("M1 Hanscheid", "M1 population rate", "M2 Bayes")
COLOURS = {"kidney": "#0072B2", "liver": "#D55E00", "spleen": "#009E73", "lesion": "#555555"}


def cases():
    """One row per patient, structure and scan: VOI activity and the four-point reference of notebook 02."""
    t = pd.read_csv(KINETICS / "stp_tia.csv")
    keys = ["patient", "snmmi", "structure", "type", "scan", "t_h", "A_MBq", "TIA_ref", "TIA_p16", "TIA_p84", "T_eff_h"]
    return t[keys].drop_duplicates().reset_index(drop=True)


def main():
    priors = pd.read_csv(KINETICS / "bayes_stp_priors.csv").set_index(["component", "type"])
    discrepancy = json.loads(CHOICE.read_text())["discrepancy"]
    cfg = spect.SpectConfig()
    one = torch.ones(1, dtype=torch.float64)
    rows = []
    for k, c in cases().iterrows():
        measured = torch.tensor([c.A_MBq], dtype=torch.float64)
        rate_population = math.exp(priors.loc[("washout lam_w", c.type), "log_location"])
        location, scale = priors.loc[("washout lam_bio", c.type), ["log_location", "log_scale"]]
        value = discrepancy[DISCREPANCY_OF[c.type]]
        prior = {"log_a0": (torch.tensor([math.log(c.A_MBq) + rate_population * c.t_h]), torch.tensor([FLAT_SD])),
                 "log_lambda_bio": (torch.tensor([location]), torch.tensor([scale])),
                 "early_share": (torch.zeros(1), torch.zeros(1))}
        posterior, ess, _ = update.bayes_update(prior, measured, torch.eye(1, dtype=torch.float64), one, 0 * one, c.t_h, cfg, value,
                                                n_samples=65536, seed=k)
        mean, sd = float(posterior["log_tiac"][0]), float(posterior["log_tiac"][1])
        estimates = {"M1 Hanscheid": (float(update.hanscheid(measured, c.t_h)), np.nan),
                     "M1 population rate": (float(update.madsen(measured, c.t_h, torch.tensor(rate_population))), np.nan),
                     "M2 Bayes": (math.exp(mean), sd)}
        for method, (tia, log_sd) in estimates.items():
            error = math.log(tia / c.TIA_ref)
            rows.append({**c.to_dict(), "method": method, "TIA_MBq_h": tia, "log_sd": log_sd, "log_error": error,
                         "error_pct": 100 * (tia / c.TIA_ref - 1), "in_ref_68": c.TIA_p16 <= tia <= c.TIA_p84,
                         "ref_in_68": abs(error) < log_sd if log_sd == log_sd else np.nan,
                         "ref_in_95": abs(error) < 1.96 * log_sd if log_sd == log_sd else np.nan,
                         "T_eff_post_h": math.log(2) / (math.exp(float(posterior["log_lambda_bio"][0])) + LAMBDA_PHYS_H) if method == "M2 Bayes" else np.nan,
                         "discrepancy": value if method == "M2 Bayes" else np.nan, "ess": float(ess[0]) if method == "M2 Bayes" else np.nan})
    table = pd.DataFrame(rows)
    table.to_csv(KINETICS / "stp_update_tia.csv", index=False)

    def summarise(keys):
        return table.groupby(keys).agg(
            n=("error_pct", "size"), median_abs_error_pct=("error_pct", lambda x: x.abs().median()),
            max_abs_error_pct=("error_pct", lambda x: x.abs().max()), median_error_pct=("error_pct", "median"),
            in_ref_68=("in_ref_68", "sum"), half_width_68_pct=("log_sd", lambda x: 100 * (np.exp(x.median()) - 1)),
            ref_in_68=("ref_in_68", lambda x: x.sum() if x.notna().all() else np.nan),
            ref_in_95=("ref_in_95", lambda x: x.sum() if x.notna().all() else np.nan)).reset_index()

    by_scan, by_type = summarise(["method", "scan"]), summarise(["method", "scan", "type"])
    by_scan.to_csv(KINETICS / "stp_update_summary_by_scan.csv", index=False)
    by_type.to_csv(KINETICS / "stp_update_summary_by_type.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, method in zip(axes, METHODS):
        d = table[table.method == method]
        for (patient, structure), g in d.groupby(["patient", "structure"]):
            g = g.sort_values("t_h")
            ax.plot(g.t_h, g.error_pct.clip(-100, 100), color=COLOURS[g.type.iloc[0]], marker="o" if patient == "patient_4" else "s",
                    ls="-" if patient == "patient_4" else "--", lw=1.5, ms=5, alpha=0.85)
        ax.axhline(0, color="#999999", lw=1)
        ax.set_title(method, fontsize=10)
        ax.set_xlabel("time of the single SPECT (h)")
        ax.set_ylim(-100, 100)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("TIA error against the four-point fit (%), clipped at 100")
    handles = [plt.Line2D([], [], color=c, lw=2, label=t) for t, c in COLOURS.items()]
    handles += [plt.Line2D([], [], color="#333333", marker="o", ls="-", label="patient 4 (A)"),
                plt.Line2D([], [], color="#333333", marker="s", ls="--", label="patient 6 (B)")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False)
    fig.suptitle("Deep Blue, 177Lu-DOTATATE, two patients: single-time-point TIA from each scan alone")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(KINETICS / "stp_update_error_vs_time.png", dpi=150)
    plt.close(fig)

    pd.set_option("display.width", 250, "display.max_rows", 200)
    print(by_scan.round(1).to_string())
    print(by_type.round(1).to_string())
    d = table[table.method == "M2 Bayes"]
    print(d.pivot_table(index=["patient", "structure"], columns="scan", values="error_pct").round(1).to_string())
    print(table[table.method == "M1 population rate"].pivot_table(index=["patient", "structure"], columns="scan", values="error_pct").round(1).to_string())
    print(d.pivot_table(index=["patient", "structure"], columns="scan", values="log_sd").round(3).to_string())
    print("ESS min", d.ess.min())


if __name__ == "__main__":
    main()
