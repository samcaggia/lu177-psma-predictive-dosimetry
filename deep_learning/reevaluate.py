"""
Recomputes the structure tables of an evaluation from the saved maps, on the CPU (needs torch, not the network or a GPU):
the same pass as section 9 of the notebook with the maps in place of the sliding-window prediction. The voxel tables
are not touched. Usage: python reevaluate.py [scenario]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from lib.calibration import evaluate_cohort, structure_report  # noqa: E402
from lib.config import LossConfig  # noqa: E402
from lib.data import GpuCohort, import_net_data  # noqa: E402
from lib.scenario import CEILING, scenario_file  # noqa: E402
from lib.utils import provenance  # noqa: E402

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "A"
NETDATA = ROOT / "data" / "processed" / "psma" / "netdata"
EVAL = HERE / "runs" / f"eval_{SCENARIO}"
SAMPLES, SEED, FOLDS = 64, 20261114, 5


def main():
    nd = import_net_data(str(ROOT / "src"))
    cohort = GpuCohort(nd, NETDATA, NETDATA / "shards", torch.device("cpu"), scenario=SCENARIO, splits=("val",))
    indices = cohort.split_indices("val")

    def from_maps(i):
        with np.load(EVAL / "maps" / f"{cohort.phantoms.study_id[i]}.npz") as data:
            mean, logvar = data["mean"].astype(np.float32), data["logvar"].astype(np.float32)
        return torch.from_numpy(np.stack([mean, logvar], axis=1).reshape(-1, *mean.shape[1:]))[None]

    previous = pd.read_csv(EVAL / f"voxel_calibration_{SCENARIO}.csv")
    stamp = {**provenance(str(HERE), str(ROOT / "src")), "checkpoint_step": int(previous.checkpoint_step.iloc[0]),
             "checkpoint_val_nll": float(previous.checkpoint_val_nll.iloc[0]), "prediction": "float16 maps"}
    prior = pd.read_parquet(scenario_file(NETDATA, "population_prior", SCENARIO))
    results = evaluate_cohort(from_maps, cohort, indices, LossConfig(), prior, k_samples=SAMPLES, seed=SEED,
                              voxel_statistics=False)
    baselines = {"population": pd.read_parquet(scenario_file(NETDATA, "baseline_population", SCENARIO)),
                 "pbpk_fit": pd.read_parquet(NETDATA / "baseline_pbpk_fit.parquet")}
    if SCENARIO != "A":
        baselines[CEILING] = pd.read_parquet(scenario_file(NETDATA, f"baseline_{CEILING}", SCENARIO))
    report = structure_report(nd, results, baselines, cohort.phantoms, str(EVAL), SCENARIO, stamp, folds=FOLDS, seed=SEED)
    pd.set_option("display.width", 250)
    comparison = report["comparison"]
    for column in ("bias", "median_abs_error", "skill"):
        table = comparison[comparison.method.isin(["network", "population", "pbpk_fit"])]
        print(f"\n{column}\n" + table.pivot(index="structure", columns=["quantity", "method"], values=column)[
            ["log_a0", "log_lambda_bio", "log_tiac"]].round(2).to_string())
    print("\n" + report["washout"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
