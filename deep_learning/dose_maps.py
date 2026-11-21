"""
Voxel dose maps of the network prediction against the truth on a few validation phantoms of the reporting half (no
torch), to document the leakage of the predicted activity across organ boundaries. Both maps come from the voxel TIA
on the 4 mm grid (truth: net_data.rebuild, draw 0; prediction: the saved network maps, no gating and no aggregation)
through the same engine: the GATE water kernel of Lu-177 on 4 mm voxels over the CT density (display approximation,
not the Monte Carlo basis of the dose tables). Writes runs/eval_<scenario>/dose/.
Usage: python dose_maps.py [study_id ...]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.signal import fftconvolve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
import net_data as nd  # noqa: E402
from lib import diagnostics  # noqa: E402

SCENARIO = "A"
NETDATA = ROOT / "data" / "processed" / "psma" / "netdata"
EVAL = HERE / "runs" / f"eval_{SCENARIO}"
OUT = EVAL / "dose"
KERNEL = ROOT / "output" / "vsv" / "vsv_lu177_mGy_per_MBq_s.npy"
CYCLE_GBQ = 7.4


def dose(tia, density, kernel):
    """Gy/GBq from the voxel TIAC (h per voxel): mGy/(MBq s) kernel, 1000 MBq, 3600 s/h, over the density."""
    return np.maximum(fftconvolve(tia, kernel, mode="same"), 0.0) * 3600.0 / density


def main():
    choice = json.loads((EVAL / f"tto_choice_{SCENARIO}.json").read_text())
    params = pd.read_parquet(NETDATA / "params.parquet")
    kinds = params.drop_duplicates("structure").set_index("structure").target_kind
    params = params[(params.arm == "main") & (params.scenario == SCENARIO) & params.has_source & (params.draw == 0)]
    phantoms = pd.read_parquet(NETDATA / "phantoms.parquet")
    reporting = [s for s in phantoms.study_id if choice["subject_half"].get(str(phantoms.set_index("study_id").subject[s])) == "reporting"
                 and (EVAL / "maps" / f"{s}.npz").exists()]
    studies = sys.argv[1:] or reporting[:3]
    sources = pd.read_parquet(NETDATA / "sources.parquet")
    arrays = diagnostics.read_phantoms(nd, NETDATA, studies)
    kernel = np.load(KERNEL)
    rows = []
    fig, axes = plt.subplots(len(studies), 3, figsize=(13, 4.6 * len(studies)), squeeze=False)
    for row, study in zip(axes, studies):
        stored = arrays[study]
        ph = diagnostics.Phantom(nd, study, stored, sources[sources.study_id == study], kinds)
        density = np.clip(1.0 + stored["ct"].astype(np.float64), 0.3, 1.8)
        truth = np.nan_to_num(ph.targets(params[params.study_id == study])["tia"])
        with np.load(EVAL / "maps" / f"{study}.npz") as data:
            predicted = np.where(ph.inside, diagnostics.map_fields(ph, data["mean"])["tia"], 0.0)
        d_true, d_pred = dose(truth, density, kernel), dose(predicted, density, kernel)
        body = ph.code > 0
        ratio = np.where(body & (d_true > 0), np.log(np.maximum(d_pred, 1e-12) / np.maximum(d_true, 1e-12)), np.nan)
        rest = ph.code == nd.COMPARTMENT_CODES["rest_of_body"]
        shell = ph.shell()
        far = rest & ~shell
        for name, mask in (("rest of body, shells of kidneys, glands, bladder", shell), ("rest of body elsewhere", far),
                           *((s.replace("_", " "), ph.code == nd.COMPARTMENT_CODES[s]) for s in ("kidneys_functional", "liver", "spleen", "salivary_glands")),
                           ("lesions", ph.lesion > 0)):
            if mask.any():
                rows.append(dict(study_id=study, region=name, voxels=int(mask.sum()), tia_ratio=predicted[mask].sum() / max(truth[mask].sum(), 1e-30),
                                 mean_dose_true_gy_per_gbq=d_true[mask].mean(), mean_dose_predicted_gy_per_gbq=d_pred[mask].mean(),
                                 median_log_dose_ratio=float(np.nanmedian(ratio[mask]))))
        kidneys = ph.code == nd.COMPARTMENT_CODES["kidneys_functional"]
        y = int(np.argmax(kidneys.sum(axis=(0, 2)))) if kidneys.any() else ph.shape[1] // 2
        extent = [0, ph.shape[0] * ph.grid_mm / 10, 0, ph.shape[2] * ph.grid_mm / 10]
        top = np.nanpercentile(np.where(body, d_true, np.nan)[:, y, :], 99.5) * CYCLE_GBQ
        norm = LogNorm(vmin=top / 300, vmax=top)
        for ax, image, title in ((row[0], d_true, "truth"), (row[1], d_pred, "network voxel maps")):
            shown = np.where(body, image, np.nan)[:, y, :].T * CYCLE_GBQ
            im = ax.imshow(np.clip(shown, top / 300, None), origin="lower", cmap="magma", norm=norm, extent=extent, aspect="equal")
            ax.set_title(f"{title}: dose (Gy per {CYCLE_GBQ:g} GBq)", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        im = row[2].imshow(ratio[:, y, :].T, origin="lower", cmap="RdBu_r", vmin=-2, vmax=2, extent=extent, aspect="equal")
        row[2].contour(np.linspace(0, extent[1], ph.shape[0]), np.linspace(0, extent[3], ph.shape[2]), (ph.code[:, y, :].T != nd.COMPARTMENT_CODES["rest_of_body"]) & (ph.code[:, y, :].T > 0),
                       levels=[0.5], colors="#333333", linewidths=0.5)
        row[2].set_title("log(predicted / true), organ outlines in grey", fontsize=10)
        fig.colorbar(im, ax=row[2], fraction=0.035, pad=0.02)
        row[0].set_ylabel(f"{study[5:13]}…  z (cm)")
        for ax in row:
            ax.set_xlabel("x (cm)")
            ax.set_facecolor("#dddddd")
    fig.suptitle("Voxel dose from the pre-therapy network maps against the truth, coronal slice through the kidneys (4 mm grid, kernel over density)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / f"dose_maps_network_vs_truth_{SCENARIO}.png", dpi=130)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / f"dose_maps_regions_{SCENARIO}.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
