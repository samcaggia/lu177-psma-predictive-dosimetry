"""
How much of the between-patient variability of A0 the PET explains on its own (local, no torch): per structure, over
the training phantoms (draw 0), the SD of log A0 per mL and of the source-mean residual log(A0 per mL / PET per mL).
Usage: python pet_explains.py [scenario]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import net_data as nd  # noqa: E402
from lib import diagnostics as D  # noqa: E402

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "A"
NETDATA = ROOT / "data" / "processed" / "psma" / "netdata"
OUT = HERE / "runs" / f"eval_{SCENARIO}"


def main():
    phantoms = pd.read_parquet(NETDATA / "phantoms.parquet")
    phantoms = phantoms[phantoms.net_split == "train"]
    params = pd.read_parquet(NETDATA / "params.parquet")
    params = params[(params.arm == "main") & (params.scenario == SCENARIO) & (params.draw == 0) & params.has_source]
    a0 = params.set_index(["study_id", "source"]).a0
    rows = []
    for shard, group in phantoms.groupby("shard"):
        arrays = D.read_phantoms(nd, NETDATA, list(group.study_id))
        for study in group.study_id:
            x = arrays.pop(study)
            code, lesion = x["compartment"].astype(np.int64), x["lesion"].astype(np.int64)
            pet = np.expm1(x["pet"].astype(np.float64)) * nd.PET_REF
            label = np.where(lesion > 0, 100 + lesion, code).ravel()
            count = np.bincount(label)
            total = np.bincount(label, weights=pet.ravel())
            names = {c: s for s, c in nd.COMPARTMENT_CODES.items()}
            for value in np.flatnonzero(count):
                if value == 0 or value == nd.LESION_CODE:
                    continue
                source = names[value] if value < 100 else f"lesion_{value - 101}"
                if (study, source) not in a0.index or total[value] <= 0:
                    continue
                volume = count[value] * 0.064
                rows.append(dict(study_id=study, source=source, structure=names[value] if value < 100 else "lesion",
                                 log_a0_per_ml=np.log(a0[(study, source)] / volume),
                                 log_pet_mean=np.log(total[value] / count[value])))
    table = pd.DataFrame(rows)
    table["residual"] = table.log_a0_per_ml - table.log_pet_mean
    table.to_csv(OUT / f"diag_pet_explains_{SCENARIO}.csv", index=False)
    summary = table.groupby("structure").agg(
        sources=("residual", "size"), sd_log_a0_per_ml=("log_a0_per_ml", "std"), sd_log_pet=("log_pet_mean", "std"),
        sd_residual=("residual", "std"))
    summary["correlation"] = table.groupby("structure")[["log_a0_per_ml", "log_pet_mean"]].corr().unstack().iloc[:, 1]
    slope = table.groupby("structure").apply(lambda g: np.polyfit(g.log_pet_mean, g.log_a0_per_ml, 1)[0])
    summary["slope"] = slope
    summary["sd_after_regression"] = summary.sd_log_a0_per_ml * np.sqrt(1 - summary.correlation ** 2)
    summary["variance_explained_ratio_1"] = 1 - (summary.sd_residual / summary.sd_log_a0_per_ml) ** 2
    summary["variance_explained_regression"] = summary.correlation ** 2
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
