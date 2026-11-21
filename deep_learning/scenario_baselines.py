"""
Baselines of a scenario other than A (local, no torch), with the functions of src/net_data.py that built those of
scenario A in notebook 22, applied to the rows of the scenario. Written in netdata/:
  population_prior_<scenario>.parquet     population prior from the training phantoms of the scenario;
  baseline_population_<scenario>.parquet  that prior as a prediction for every virtual patient and source;
  baseline_same_phantom_draws_<scenario>.parquet  the other draws of the same phantom in the same scenario: what a
      predictor that knows the phantom perfectly (its PET-fit posterior and the PET-washout link of the scenario) can
      say about a new draw, i.e. the ceiling of any PET-based prediction (estimated from 9 draws).
The PBPK-fit baseline (baseline_pbpk_fit.parquet, the scenario A draws of the phantom) applies to every scenario: it
is the PBPK fit of the PET without the PET-washout link. With scenario A the three tables are rebuilt in memory and
compared with the files of notebook 22 (check only, nothing written). The test tables are not read.
Usage: python scenario_baselines.py <scenario>
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
from lib.scenario import CEILING, gaussian_skill, scenario_rho  # noqa: E402

NETDATA = ROOT / "data" / "processed" / "psma" / "netdata"


def build(scenario):
    params = pd.read_parquet(NETDATA / "params.parquet")
    phantoms = pd.read_parquet(NETDATA / "phantoms.parquet")
    volumes = nd.structure_volumes(pd.read_parquet(NETDATA / "sources.parquet"))
    assignment = phantoms.set_index("study_id").net_split
    rows = params[params.scenario == scenario].assign(scenario="A")
    prior = nd.population_prior(rows, volumes, list(assignment.index[assignment == "train"]))
    population = nd.population_prediction(rows, volumes, prior).assign(net_split=lambda x: x.study_id.map(assignment))
    draws = nd.pbpk_leave_one_out(rows).assign(baseline=CEILING, net_split=lambda x: x.study_id.map(assignment))
    return params[params.scenario == scenario], prior, population, draws


def washout_ceiling(params, draws):
    """Per network split and structure: SD of log lambda_bio, share of its variance between phantoms, skill of the same-phantom draws."""
    truth = params[(params.arm == "main") & params.has_source]
    merged = truth.merge(draws[["virtual_id", "source", "log_lambda_bio_mean"]], on=["virtual_id", "source"])
    rows = []
    for (split, structure), g in merged.groupby(["net_split", "structure"]):
        x = g.log_lambda_bio
        by_source = g.groupby(["study_id", "source"]).log_lambda_bio
        between = (by_source.mean().var() - by_source.var().mean() / by_source.size().mean()) / x.var()
        population = (x - truth.log_lambda_bio[(truth.net_split == "train") & (truth.structure == structure)].mean()).abs().median()
        rows.append(dict(net_split=split, structure=structure, n=len(g), sd=float(x.std()), between_phantom_share=float(between),
                         skill_same_phantom_draws=float(1 - (g.log_lambda_bio_mean - x).abs().median() / population)))
    return pd.DataFrame(rows)


def main():
    scenario = sys.argv[1]
    params, prior, population, draws = build(scenario)
    if scenario == "A":
        for name, table in (("population_prior", prior), ("baseline_population", population)):
            stored = pd.read_parquet(NETDATA / f"{name}.parquet")
            pd.testing.assert_frame_equal(table.reset_index(drop=True), stored.reset_index(drop=True), check_dtype=False)
        stored = pd.read_parquet(NETDATA / "baseline_pbpk_fit.parquet")
        pd.testing.assert_frame_equal(draws.assign(baseline="pbpk_fit").reset_index(drop=True), stored.reset_index(drop=True), check_dtype=False)
        print("Scenario A: the three tables rebuilt here equal those of notebook 22")
    else:
        prior.to_parquet(NETDATA / f"population_prior_{scenario}.parquet", index=False)
        population.to_parquet(NETDATA / f"baseline_population_{scenario}.parquet", index=False)
        draws.to_parquet(NETDATA / f"baseline_{CEILING}_{scenario}.parquet", index=False)
        print(f"Written for {scenario}: {len(prior)} prior rows, {len(population)} population rows, {len(draws)} same-phantom rows")
    pd.set_option("display.width", 200)
    rho = scenario_rho(scenario)
    print(f"\nWashout ceiling, rho {rho:+.1f}: Gaussian bound of the skill with the uptake of the draw known {gaussian_skill(rho):.3f} "
          f"(share of variance {rho ** 2:.2f})")
    print(washout_ceiling(params, draws).pivot(index="structure", columns="net_split",
                                               values=["sd", "between_phantom_share", "skill_same_phantom_draws"])
          .drop(columns="test", level=1, errors="ignore").round(3).to_string())
    if scenario != "A":
        reference = pd.read_parquet(NETDATA / "population_prior.parquet")
        both = prior[prior.arm == "main"].merge(reference[reference.arm == "main"], on="structure", suffixes=("", "_A")).set_index("structure")
        columns = ["log_lambda_bio_mean", "log_lambda_bio_sd", "log_tiac_per_ml_mean", "log_tiac_per_ml_sd"]
        print("\nPopulation prior of the scenario minus that of A (main arm):\n"
              + pd.DataFrame({c: both[c] - both[f"{c}_A"] for c in columns}).round(3).to_string())
        print("SD of log TIAC per mL in the scenario:\n" + both.log_tiac_per_ml_sd.round(3).to_string())


if __name__ == "__main__":
    main()
