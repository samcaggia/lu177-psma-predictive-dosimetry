"""
Tables that depend on the scenario (no torch). Scenario A keeps the file names of notebook 22; the tables of another
scenario are netdata/<stem>_<scenario>.parquet, written locally by scenario_baselines.py and src/spect_data.py.
"""
import math
import re
from pathlib import Path

CEILING = "same_phantom_draws"


def scenario_file(netdata, stem, scenario):
    """Path of a per-scenario table; for a scenario other than A the file must exist (no silent fall-back on A)."""
    if scenario == "A":
        return Path(netdata) / f"{stem}.parquet"
    path = Path(netdata) / f"{stem}_{scenario}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: build it locally (scenario_baselines.py, src/spect_data.py) and upload it")
    return path


def scenario_rho(scenario):
    """Correlation of log lambda_rel with the log PET uptake of the draw in the scenario name ('B+0.6' -> 0.6); 0 for A."""
    match = re.fullmatch(r"B([+-]\d+(?:\.\d+)?)", scenario)
    return float(match.group(1)) if match else 0.0


def gaussian_skill(rho):
    """Skill (1 - MAE / MAE of the population) of the best predictor of a Gaussian variable from one correlated with it at rho."""
    return 1.0 - math.sqrt(1.0 - rho ** 2)
