"""
What a PET-residual parametrisation of the log A0 head would have to learn (local, no torch): within every source of
the training and validation phantoms, draw 0, the spread of log(A0 target per mL / PET per mL) over its voxels.
Usage: python pet_residual.py [scenario]
"""
import hashlib
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

pd.set_option("display.width", 250, "display.max_columns", 40)
SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "A"
NETDATA = ROOT / "data" / "processed" / "psma" / "netdata"
OUT = HERE / "runs" / f"eval_{SCENARIO}"


def main():
    phantoms = pd.read_parquet(NETDATA / "phantoms.parquet")
    params = pd.read_parquet(NETDATA / "params.parquet")
    kinds = params.drop_duplicates("structure").set_index("structure").target_kind
    params = params[(params.arm == "main") & (params.scenario == SCENARIO) & (params.draw == 0) & params.has_source]
    sources = pd.read_parquet(NETDATA / "sources.parquet")
    tables = []
    for shard, group in phantoms.groupby("shard"):
        arrays = D.read_phantoms(nd, NETDATA, list(group.study_id))
        for study in group.study_id:
            ph = D.Phantom(nd, study, arrays.pop(study), sources[sources.study_id == study], kinds, r_eval=0)
            tables.append(D.pet_residual(ph, ph.targets(params[params.study_id == study])))
        print(f"{shard}: {len(group)} phantoms", flush=True)
    table = pd.concat(tables, ignore_index=True).merge(phantoms[["study_id", "net_split", "scanner"]], on="study_id")
    commit, _ = (HERE / "VERSION").read_text().split()
    table.assign(code_commit=commit, net_data_md5=hashlib.md5((ROOT / "src" / "net_data.py").read_bytes()).hexdigest()
                 ).to_csv(OUT / f"diag_pet_residual_{SCENARIO}.csv", index=False)

    q90 = lambda x: x.quantile(0.9)
    for split, g in table.groupby("net_split"):
        summary = g.groupby("structure").agg(
            sources=("voxels", "size"), sd_log_a0_median=("sd_log_a0", "median"),
            sd_residual_median=("sd_residual", "median"), sd_residual_p90=("sd_residual", q90),
            sd_interior_median=("sd_residual_interior", "median"), boundary_share=("boundary_share", "median"),
            boundary_minus_interior=("boundary_minus_interior", "median"),
            between_phantom_sd_of_median=("median_residual", "std"), median_residual=("median_residual", "median"),
            pet_below_1e_5=("pet_below_1e-05", "mean"), pet_below_1e_4=("pet_below_0.0001", "mean"),
            pet_below_1e_3=("pet_below_0.001", "mean"))
        print(f"\n== {split} ==\n" + summary.round(3).to_string())
    rest = table[table.structure == "rest_of_body"]
    print("\nRest of body, shells (median over phantoms):\n" + rest.groupby("net_split")[
        ["shell_share", "shell_minus_elsewhere", "sd_residual", "sd_residual_outside_shell"]].median().round(3).to_string())
    lesion = table[table.structure == "lesion"]
    bins = pd.cut(lesion.voxels, [1, 8, 30, 100, np.inf], labels=["2-8", "9-30", "31-100", "> 100"])
    print("\nLesions by grid voxels: SD of the residual and of the log target (median)\n"
          + lesion.groupby(bins, observed=True)[["sd_residual", "sd_log_a0", "boundary_share"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
