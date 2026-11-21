"""
Local diagnostics of an evaluation (run with the base environment, no torch): lesion errors against lesion
properties, aggregation variants and lesion TIAC decomposition from the saved maps, train/validation gap, gated prior.
Usage: python diagnose.py [scenario]
"""
import hashlib
import os
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
from lib import gating as G  # noqa: E402
from lib.scenario import scenario_file  # noqa: E402
from lib.scoring import with_sd  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 300)
SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "A"
NETDATA = ROOT / "data" / "processed" / "psma" / "netdata"
EVAL = HERE / "runs" / f"eval_{SCENARIO}"
SEED = 20261114


def stamp():
    commit, digest = (HERE / "VERSION").read_text().split()
    files = sorted((HERE / "lib").glob("*.py"))
    current = hashlib.md5(b"".join(f.read_bytes() for f in files)).hexdigest()
    return dict(code_commit=commit if current == digest else commit + "+modified",
                net_data_md5=hashlib.md5((ROOT / "src" / "net_data.py").read_bytes()).hexdigest())


def save(table, name, extra):
    table.assign(**extra).to_csv(EVAL / f"{name}_{SCENARIO}.csv", index=False)


def main():
    prediction = pd.read_csv(EVAL / f"structure_prediction_{SCENARIO}.csv")
    extra = {**stamp(), "checkpoint_step": int(prediction.checkpoint_step.iloc[0]),
             "evaluation_commit": prediction.code_commit.iloc[0]}
    phantoms = pd.read_parquet(NETDATA / "phantoms.parquet")
    sources = pd.read_parquet(NETDATA / "sources.parquet")
    studies = sorted(prediction.study_id.unique())
    maps = EVAL / "maps"
    have_maps = maps.is_dir() and all((maps / f"{s}.npz").exists() for s in studies)
    print(f"Scenario {SCENARIO}: {len(studies)} validation phantoms, checkpoint step {extra['checkpoint_step']}, "
          f"maps {'found' if have_maps else 'NOT found: aggregation variants and decomposition skipped'}")
    out = D.run(nd, NETDATA, studies, SCENARIO, maps_dir=maps if have_maps else None, log=lambda s: None)
    truth, truth_like = out["truth"], out["truth_like"]

    print("\n== Lesions: error against lesion properties (prediction - truth on the dilated mask, mean over draws) ==")
    lesions = D.lesion_errors(prediction, truth_like, out["lesions"], sources, phantoms)
    bins = pd.concat([
        D.binned(lesions, "native_volume_ml", [0, 0.5, 2, 10, np.inf], ["< 0.5 mL", "0.5-2 mL", "2-10 mL", "> 10 mL"]),
        D.binned(lesions, "retained", [0, 0.6, 0.8, 0.9, 1.01], ["< 0.6", "0.6-0.8", "0.8-0.9", "> 0.9"]),
        D.binned(lesions, "pet_mean_pct_ia_ml", list(lesions.pet_mean_pct_ia_ml.quantile([0, .25, .5, .75, 1])),
                 ["PET Q1", "PET Q2", "PET Q3", "PET Q4"]),
        D.binned(lesions, "n_lesions", [0, 3, 20, np.inf], ["1-3", "4-20", "> 20"]),
        D.binned(lesions, "scanner")], ignore_index=True)
    print(bins.round(3).to_string(index=False))
    numeric = lesions[["error_log_tiac", "error_log_a0", "error_log_lambda_bio", "error_early_share"]]
    covariates = pd.DataFrame({"log_volume": np.log(lesions.native_volume_ml), "retained": lesions.retained,
                               "log_pet_mean": np.log(lesions.pet_mean_pct_ia_ml), "log_n_lesions": np.log(lesions.n_lesions)})
    correlation = pd.concat([numeric, covariates], axis=1).corr(method="spearman").loc[numeric.columns, covariates.columns]
    print("\nSpearman correlation of the lesion errors with the properties:\n" + correlation.round(2).to_string())
    save(lesions, "diag_lesion_errors", extra)
    save(bins, "diag_lesion_bins", extra)

    if have_maps:
        both = out["map_prediction"].merge(prediction.drop_duplicates(["study_id", "source"]), on=["study_id", "source"])
        worst = max(float(np.nanmax(np.abs(both[q] - both[f"{q}_mean"]))) for q in D.QUANTITIES)
        print(f"\nNumPy summary of the maps against structure_prediction: largest difference {worst:.4f}")
        scores = D.score_variants(out["variants"], truth)
        print("\n== Aggregation variants, organs: bias ==\n" + scores.pivot(index=["quantity", "structure"], columns="variant", values="bias").round(3).to_string())
        print("\n== Aggregation variants, organs: median |error| ==\n" + scores.pivot(index=["quantity", "structure"], columns="variant", values="median_abs_error").round(3).to_string())
        print("\n== Rest of body: where the predicted A0 sits (mean over phantoms) ==\n" + out["rest_of_body"].drop(columns="study_id").mean().round(3).to_string())
        parts = out["decomposition"]
        columns = ["log_tiac_error", "part_a0", "part_lambda_bio", "part_early_share", "part_ring_tiac_head", "interaction",
                   "log_ratio_own", "log_ratio_ring", "true_tiac_share_in_ring"]
        print("\n== Lesion TIAC error by head (dilated mask, all draws) ==\n" + parts[columns].agg(["mean", "median"]).round(3).to_string())
        save(scores, "diag_aggregation", extra)
        save(out["rest_of_body"], "diag_rest_of_body", extra)
        save(parts, "diag_lesion_decomposition", extra)

    train_file = EVAL / f"train_voxel_calibration_{SCENARIO}.csv"
    if train_file.exists():
        columns = ["head", "structure", "mean_z", "mean_z2", "coverage_68", "nll"]
        both = pd.read_csv(EVAL / f"voxel_calibration_{SCENARIO}.csv")[columns].merge(
            pd.read_csv(train_file)[columns], on=["head", "structure"], suffixes=("_val", "_train"))
        print("\n== Voxel calibration, validation against the training phantoms of the check ==\n"
              + both[both.structure.isin(["lesion", "all", "liver", "rest_of_body"])].round(3).to_string(index=False))
        train_comparison = pd.read_csv(EVAL / f"train_structure_comparison_{SCENARIO}.csv")
        val_comparison = pd.read_csv(EVAL / f"structure_comparison_{SCENARIO}.csv")
        pick = lambda t: t[(t.method == "network") & (t.structure == "lesion")].set_index("quantity")[["n", "bias", "median_abs_error", "skill"]]
        print("\nLesions, structure level:\n" + pick(val_comparison).join(pick(train_comparison), lsuffix="_val", rsuffix="_train").round(3).to_string())
    else:
        print("\nNo train_voxel_calibration file: the train/validation gap needs the Colab run of section 9.5")

    print("\n== Gated prior ==")
    population = pd.read_parquet(scenario_file(NETDATA, "baseline_population", SCENARIO))
    population = population[population.arm == "main"]
    for criterion in ("nlpd", "median_abs_error"):
        choice, table = G.gated_prior(nd, with_sd(prediction), truth_like, population, truth, phantoms, SEED, criterion)
        if criterion == "nlpd":
            G.save_choice(choice, EVAL / f"gated_prior_{SCENARIO}.json", extra)
            save(table, "gated_prior", extra)
            halves = pd.Series(choice["subject_half"]).value_counts().to_dict()
            print(f"subjects per half: {halves}; choice by {criterion} on the tuning half:")
            print(pd.DataFrame(choice["choice"]).T.replace({"network (calibrated)": "network", "population": "pop"}).to_string())
            report = table[table.half == "reporting"]
            for column in ("median_abs_error", "nlpd", "coverage_68", "coverage_95"):
                print(f"\nreporting half, {column}:\n" + report[report.quantity.isin(["log_a0", "log_lambda_bio", "log_tiac", "early_share"])]
                      .pivot(index=["quantity", "structure"], columns="method", values=column).round(3).to_string())
        else:
            other = pd.DataFrame(choice["choice"]).T.replace({"network (calibrated)": "network", "population": "pop"})
            print(f"\nchoice if the criterion were {criterion} (not saved):\n" + other.to_string())
    print(f"\nTables in {EVAL}")


if __name__ == "__main__":
    main()
