import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description="Draw the README figures from the aggregate tables in docs/results.")
parser.add_argument("--results", type=Path, default=ROOT / "docs" / "results")
parser.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
args = parser.parse_args()
RES, OUT = args.results, args.out
OUT.mkdir(parents=True, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
RED = "#e34948"
NEUTRAL = "#f0efec"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.labelcolor": INK2,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def spect_error_vs_time():
    df = pd.read_csv(RES / "spect_update_test_mean_over_structures.csv")
    df = df[df.condition == "mismatch SD 0.15 (main)"]
    series = [
        ("M2 Bayes", "M2  Bayesian update on VOIs", BLUE),
        ("M5 PBPK", "M5  differentiable PBPK", ORANGE),
        ("M1 population rate", "M1  single time point, population rate", AQUA),
        ("M3 maps", "M3  test-time optimisation of maps", YELLOW),
        ("M4 weights", "M4  test-time optimisation of weights", MAGENTA),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    m0 = df[df.method == "M0 gated prior"].mean_median_abs_error.iloc[0]
    ax.axhline(m0, color=MUTED, lw=1.5, ls=(0, (4, 3)))
    ax.text(170, m0 + 0.008, "M0  prediction from the PET, no SPECT", color=INK2, ha="right", va="bottom", fontsize=9)
    for key, label, colour in series:
        d = df[df.method == key].sort_values("t_h")
        ax.plot(d.t_h, d.mean_median_abs_error, color=colour, lw=2, marker="o", ms=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
    ax.set_xticks([24, 48, 96, 168])
    ax.set_xticklabels(["24 h", "48 h", "96 h", "168 h"])
    ax.set_xlim(14, 178)
    ax.set_ylim(0, 0.45)
    ax.set_xlabel("Time of the single post-therapy SPECT")
    ax.set_ylabel("Median absolute log error of the TIA")
    ax.set_title("One SPECT at 48-96 h cuts the TIA error by about 2.5 times")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=9)
    fig.text(0.0, -0.47, "Frozen test split of the virtual cohort (34 subjects, 47 phantoms), mean over kidneys, liver, spleen, "
             "salivary glands, GI tract and lesions.", color=MUTED, fontsize=8, transform=ax.transAxes)
    save(fig, "spect_update_error_vs_time.png")


def prior_ablation():
    df = pd.read_csv(RES / "spect_update_test_prior_ablation.csv")
    df = df[(df.structure == "mean of the six with uptake") & (df.condition == "mismatch SD 0.15 (main)")].copy()
    df["x"] = df.t_h.fillna(0)
    df = df.sort_values("x")
    labels = ["no SPECT", "24 h", "48 h", "96 h", "168 h"]
    pos = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.axhline(0, color=AXIS, lw=1.2)
    ax.vlines(pos, df.low, df.high, color=BLUE, lw=2)
    ax.plot(pos, df.difference, "o", color=BLUE, ms=8, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    for p, v in zip(pos, df.difference):
        ax.text(p + 0.12, v, f"{v:+.3f}", va="center", fontsize=8.5, color=INK2)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(pos) - 0.3)
    ax.set_ylabel("Network prior minus population prior\n(difference in log error)")
    ax.set_title("Once a SPECT is available the network prior adds almost nothing")
    ax.text(0, -0.24, "Negative values: the network helps. Paired bootstrap over 34 test subjects, 95% intervals; "
            "mean over the six structures with uptake.", color=MUTED, fontsize=8, transform=ax.transAxes)
    save(fig, "prior_contribution.png")


def network_skill():
    df = pd.read_csv(RES / "network_test_by_structure.csv")
    df = df[(df.method == "network (calibrated)") & (df.quantity == "log_tiac")].copy()
    names = {"bladder": "Bladder", "blood_pool": "Blood pool", "gi_tract": "GI tract", "kidneys_functional": "Kidneys",
             "lesion": "Lesions", "liver": "Liver", "rest_of_body": "Rest of body", "salivary_glands": "Salivary glands",
             "spine": "Spine", "spleen": "Spleen"}
    df["name"] = df.structure.map(names)
    df = df.sort_values("skill")
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    colours = [BLUE if s > 0 else RED for s in df.skill]
    ax.barh(df.name, df.skill, color=colours, height=0.62, edgecolor=SURFACE, linewidth=2)
    ax.axvline(0, color=AXIS, lw=1.2)
    for y, s in enumerate(df.skill):
        ax.text(s + (0.015 if s >= 0 else -0.015), y, f"{s:+.2f}", va="center", ha="left" if s >= 0 else "right",
                fontsize=8.5, color=INK2)
    ax.set_xlim(-0.75, 0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Skill of the PET-based network vs the population prior (1 - error / population error)")
    ax.set_title("Before therapy, the PET informs some organs, not kidneys or lesions")
    ax.text(0, -0.2, "Time-integrated activity per structure, frozen test split. Positive: the network beats the population prior.",
            color=MUTED, fontsize=8, transform=ax.transAxes)
    save(fig, "network_skill_by_structure.png")


def deepblue():
    df = pd.read_csv(RES / "deepblue_single_time_point_by_scan.csv")
    series = [("M2 Bayes", "Bayesian update (M2)", BLUE), ("M1 population rate", "Population effective half-life", ORANGE),
              ("M1 Hanscheid", "Hänscheid formula", AQUA)]
    labels = ["~4 h", "~1 day", "~4 days", "5-8 days"]
    pos = np.arange(4)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for key, label, colour in series:
        d = df[df.method == key].sort_values("scan")
        ax.plot(pos, d.median_abs_error_pct, color=colour, lw=2, marker="o", ms=6, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=label, zorder=3)
    ax.set_yscale("log")
    ax.set_yticks([2, 5, 10, 20, 50, 100])
    ax.set_yticklabels(["2%", "5%", "10%", "20%", "50%", "100%"])
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.2, 3.2)
    ax.set_xlabel("Time of the single SPECT after injection")
    ax.set_ylabel("Median absolute error of the TIA")
    ax.set_title("Real kinetics: a scan at about 4 days gives the TIA within a few percent")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.text(0, -0.2, "Two 177Lu-DOTATATE patients (Deep Blue Data), 13 organs and lesions per scan; "
            "reference: four-time-point TIA.", color=MUTED, fontsize=8, transform=ax.transAxes)
    save(fig, "deepblue_single_time_point.png")


def stp_principle():
    x = np.linspace(0.05, 4, 400)
    ratio = 2 * x * 2 ** (-x)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.axhspan(0.9, 1.1, color=NEUTRAL, zorder=0)
    ax.axhline(1, color=AXIS, lw=1.2)
    ax.plot(x, ratio, color=BLUE, lw=2.2, zorder=3)
    for xx in (1, 2):
        ax.plot(xx, 1, "o", color=BLUE, ms=7, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
    ax.text(1.5, 1.16, "exact at t = T and t = 2T", ha="center", color=INK2, fontsize=9)
    ax.text(3.95, 1.11, "within 10%", ha="right", va="bottom", color=MUTED, fontsize=8)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("Scan time / effective half-life of the structure  (t / T)")
    ax.set_ylabel("Estimated / true TIA")
    ax.set_title("Why the SPECT time matters: the single-time-point estimate 2t A(t) / ln 2")
    ax.text(0, -0.22, "Mono-exponential washout. The estimate equals 2x 2^(-x) times the true TIA, x = t/T: "
            "too early or too late underestimates.", color=MUTED, fontsize=8, transform=ax.transAxes)
    save(fig, "single_time_point_principle.png")


def box(ax, x, y, w, h, title, body, face, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02", fc=face, ec=edge, lw=1.2))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.085, body, ha="center", va="top", fontsize=8, color=INK2, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4,
                                                                  shrinkA=2, shrinkB=2))


def pipeline():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(False)
    build, light_b = "#cde2fb", BLUE
    use, light_u = "#fde2d6", ORANGE
    w, h = 0.2, 0.25
    row1 = 0.64
    row2 = 0.14
    xs = [0.02, 0.27, 0.52, 0.77]
    box(ax, xs[0], row1, w, h, "PSMA PET/CT", "public 68Ga-PSMA studies\nTotalSegmentator organs\n588 voxel phantoms", build, light_b)
    box(ax, xs[1], row1, w, h, "PBPK model", "written from published\nequations, fitted to PET\npopulation calibration", build, light_b)
    box(ax, xs[2], row1, w, h, "Virtual cohort", "real anatomies\nsimulated 177Lu kinetics\n10 draws per phantom", build, light_b)
    box(ax, xs[3], row1, w, h, "Monte Carlo dose", "GATE 10 / Geant4\n226 phantoms, dose basis\nper source organ", build, light_b)
    box(ax, xs[0], row2, w, h, "Pre-therapy network", "SwinUNETR: voxel kinetic\nparameters + uncertainty\nfrom PET/CT", use, light_u)
    box(ax, xs[1], row2, w, h, "Physics layer", "differentiable kinetics\nA(t), time-integrated\nactivity per voxel", use, light_u)
    box(ax, xs[2], row2, w, h, "One SPECT", "post-therapy, 24-168 h\nBayesian update (M2)\nor PBPK update (M5)", use, light_u)
    box(ax, xs[3], row2, w, h, "Personalised dose", "TIA x dose basis\norgans, lesions, marrow\nwith uncertainty", use, light_u)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + w, row1 + h / 2, b, row1 + h / 2)
        arrow(ax, a + w, row2 + h / 2, b, row2 + h / 2)
    mid = (row1 + row2 + h) / 2
    ax.plot([xs[2] + w / 2, xs[2] + w / 2, xs[0] + w / 2], [row1, mid, mid], color=MUTED, lw=1.4)
    arrow(ax, xs[0] + w / 2, mid, xs[0] + w / 2, row2 + h)
    ax.text((xs[0] + xs[2] + w) / 2, mid + 0.015, "training targets", color=MUTED, fontsize=8.5, ha="center")
    arrow(ax, xs[3] + w / 2, row1, xs[3] + w / 2, row2 + h)
    ax.text(xs[3] + w / 2 + 0.012, (row1 + row2 + h) / 2, "dose basis", color=MUTED, fontsize=8.5, va="center")
    ax.text(0.02, 0.97, "Building the virtual cohort", color=BLUE, fontsize=10, fontweight="bold", va="top")
    ax.text(xs[1], row2 + h + 0.025, "Predicting and personalising the dose", color=ORANGE, fontsize=10, fontweight="bold", va="bottom")
    ax.text(0.02, 0.03, "Validation: Deep Blue 177Lu-DOTATATE patients (real kinetics), Lanconelli voxel S-values, "
            "frozen test split.", color=MUTED, fontsize=8.5)
    save(fig, "pipeline.png")


if __name__ == "__main__":
    pipeline()
    spect_error_vs_time()
    prior_ablation()
    network_skill()
    deepblue()
    stp_principle()
    print(sorted(p.name for p in OUT.glob("*.png")))
