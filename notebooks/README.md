# notebooks/

Numbered in order of dependency; the number is global (later notebooks and cache file names such as `nb18_*` refer
to them as nb18 etc.). Gaps in the numbering are earlier development notebooks that are not part of this repository;
some markdown cells still mention them (nb10, nb12, nb13, nb16, nb17). The model they describe is in `src/pbpk.py`
and `docs/methods.md`.

Outputs are cleared. The first cell of each notebook ends with its purpose, inputs, outputs, run time and
environment. Kernel: "Python (lu177)" (base environment). The notebooks find the repository root by looking for
`src/paths.py` above the current folder and read the raw data from `LU177_DATASET` (default `data/raw/`).

| Folder | Notebooks | Depends on |
|---|---|---|
| `01_deepblue/` | 01 loading, 02 kinetics, 03 voxel kinetics, 04 dose, 05 Monte Carlo, 06 single time point, 07 Bayesian single time point | Deep Blue raw data, `output/vsv/` kernel |
| `02_psma_data/` | 08 inventory and split, 09 TotalSegmentator (Colab), 11 phantoms | PSMA-PET-CT-Lesions raw data |
| `03_pbpk_and_cohort/` | 18 PBPK fit on the real PETs, 19 population calibration, 20 virtual cohort | 11 |
| `04_monte_carlo/` | 14 pilot, 15 re-test | 11 |
| `05_cohort_doses/` | 21 cohort doses | 15, 20, the full Monte Carlo run |
| `06_netdata/` | 22 network data | 11, 20, 21 |

The network notebook is `deep_learning/main.ipynb` (Colab).
