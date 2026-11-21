"""Figures of the evaluation, saved as PNG: reliability curves, z histograms, predicted against true TIAC, coverage."""
import os

import numpy as np

from .config import HEADS

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d2"


def _pyplot():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    return plt


def _style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def _save(fig, path, stamp):
    fig.text(0.005, 0.005, " | ".join(f"{k} {v}" for k, v in stamp.items()), fontsize=5, color=MUTED)
    fig.savefig(f"{path}.tmp", format="png", dpi=130, metadata={k: str(v) for k, v in stamp.items()})
    _pyplot().close(fig)
    os.replace(f"{path}.tmp", path)


def reliability_figure(reliability, path, stamp, title="Voxel-level reliability"):
    """Nominal against empirical central coverage per head, all masked voxels, before and after the variance scale."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(HEADS), figsize=(3.1 * len(HEADS), 3.6), sharex=True, sharey=True)
    for ax, head in zip(axes, HEADS):
        d = reliability[(reliability["head"] == head) & (reliability.structure == "all")]
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="ideal")
        ax.plot(d.nominal, d.empirical, "o-", color=BLUE, lw=2, ms=4, label="network")
        ax.plot(d.nominal, d.empirical_scaled, "s-", color=ORANGE, lw=2, ms=4, label="after variance scale")
        ax.set_title(head, fontsize=10, color=INK)
        ax.set_xlabel("nominal coverage", fontsize=9)
        ax.set_aspect("equal")
        _style(ax)
    axes[0].set_ylabel("empirical coverage (loss weights)", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    _save(fig, path, stamp)


def z_histogram_figure(histogram, summary, path, stamp, title="Standardised voxel residuals"):
    """Weighted density of z = (target - mean) / SD per head against the standard normal."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(HEADS), figsize=(3.1 * len(HEADS), 3.3), sharex=True)
    grid = np.linspace(-6, 6, 241)
    for ax, head in zip(axes, HEADS):
        d = histogram[(histogram["head"] == head) & (histogram.structure == "all")]
        row = summary[(summary["head"] == head) & (summary.structure == "all")]
        ax.stairs(d.density, np.append(d.z_low.to_numpy(), d.z_high.iloc[-1]) if len(d) else [0, 1], color=BLUE,
                  fill=True, alpha=0.55, label="network")
        ax.plot(grid, np.exp(-0.5 * grid ** 2) / np.sqrt(2 * np.pi), color=INK, lw=1.2, label="N(0, 1)")
        if len(row):
            ax.set_title(f"{head}\nmean z {row.mean_z.iloc[0]:+.2f}, mean z² {row.mean_z2.iloc[0]:.2f}", fontsize=9,
                         color=INK)
        ax.set_xlabel("z", fontsize=9)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 2)
        _style(ax)
    axes[0].set_ylabel("density (loss weights)", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(title + " (the outermost bins collect |z| > 6)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    _save(fig, path, stamp)


def tiac_scatter_figure(methods, path, stamp, quantity="log_tiac", title="Predicted against true log TIAC"):
    """methods {name: (prediction, truth)}; one panel per structure, one colour per method (at most three)."""
    plt = _pyplot()
    colours = dict(zip(methods, (BLUE, ORANGE, AQUA)))
    merged = {name: truth[["arm", "virtual_id", "source", "structure", quantity]].merge(
        prediction[["arm", "virtual_id", "source", f"{quantity}_mean"]], on=["arm", "virtual_id", "source"])
        for name, (prediction, truth) in methods.items()}
    structures = sorted(set().union(*(m.structure.unique() for m in merged.values())))
    columns = 5
    rows = int(np.ceil(len(structures) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 3.1 * rows), squeeze=False)
    for ax, structure in zip(axes.flat, structures):
        low, high = np.inf, -np.inf
        for name, m in merged.items():
            d = m[m.structure == structure].dropna()
            if len(d) > 1500:
                d = d.sample(1500, random_state=0)
            ax.scatter(d[quantity], d[f"{quantity}_mean"], s=5, color=colours[name], alpha=0.35, lw=0, label=name)
            if len(d):
                low, high = min(low, d[quantity].min()), max(high, d[quantity].max())
        if np.isfinite(low):
            ax.plot([low, high], [low, high], color=MUTED, lw=1, ls="--")
        ax.set_title(structure, fontsize=9, color=INK)
        _style(ax)
    for ax in axes.flat[len(structures):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(f"true {quantity}", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"predicted {quantity}", fontsize=9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="upper right", fontsize=8, frameon=False, ncol=len(labels), markerscale=3)
    for handle in legend.legend_handles:
        handle.set_alpha(1)
    fig.suptitle(title, fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    _save(fig, path, stamp)


def coverage_figure(comparison, before, after, path, stamp, title="Structure-level coverage"):
    """Coverage of the 68% and 95% intervals per structure and quantity for two methods of the comparison table."""
    plt = _pyplot()
    quantities = list(comparison.quantity.unique())
    structures = sorted(comparison.structure.unique())
    fig, axes = plt.subplots(2, len(quantities), figsize=(2.6 * len(quantities), 6.2), sharey="row", squeeze=False)
    for row, (column, nominal) in enumerate((("coverage_68", 0.683), ("coverage_95", 0.95))):
        for ax, q in zip(axes[row], quantities):
            d = comparison[comparison.quantity == q]
            y = np.arange(len(structures))
            for name, colour, marker, offset in ((before, BLUE, "o", -0.15), (after, ORANGE, "s", 0.15)):
                values = d[d.method == name].set_index("structure")[column].reindex(structures)
                ax.scatter(values, y + offset, color=colour, marker=marker, s=28, label=name, zorder=3)
            ax.axvline(nominal, color=MUTED, lw=1, ls="--")
            ax.set_yticks(y)
            ax.set_yticklabels(structures, fontsize=8)
            ax.set_xlim(0, 1.02)
            ax.invert_yaxis()
            ax.set_title(q if row == 0 else "", fontsize=9, color=INK)
            ax.set_xlabel(f"coverage of the {nominal:.0%} interval", fontsize=8)
            _style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8, frameon=False, ncol=2)
    fig.suptitle(title, fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    _save(fig, path, stamp)
