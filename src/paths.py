"""Repository and raw-data locations, shared by notebooks, scripts and modules.

LU177_DATASET: folder holding the raw datasets ("Deep Blue Dataset", "PSMA-PET-CT-Lesions_v3",
"Totalsegmentator_dataset_v201"); default data/raw/ inside the repository (see data/README.md).
Processed data and outputs are always written inside the repository (data/processed/, output/, deep_learning/runs/);
LU177_NETDATA, LU177_COHORT and LU177_RUNS redirect the network data, the cohort and the run outputs.
"""
import os
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
DATASET = Path(os.environ.get("LU177_DATASET") or WORKSPACE / "data" / "raw")
PROCESSED = WORKSPACE / "data" / "processed"
