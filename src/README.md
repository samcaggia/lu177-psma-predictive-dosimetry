# src/: shared library

Flat modules, imported by name after `sys.path.insert(0, "<repository>/src")` (notebooks and scripts do this).

| Topic | Module | Content |
|---|---|---|
| paths | `paths.py` | `WORKSPACE`, `DATASET` (environment variable `LU177_DATASET`, default `data/raw/`), `PROCESSED` |
| PSMA data | `psma_exclusions.py` | studies whose SUV must not be used and slice crop; running it writes `data/processed/psma/psma_pet_exclusions.csv` |
| | `psma_data.py` | single loader of the PSMA-PET-CT-Lesions studies (exclusions and crop always applied) |
| phantoms | `psma_phantoms.py` | phantom per study: CT, PET in decay-corrected %IA/mL, compartment labels, lesions |
| PBPK | `pbpk.py` | PSMA PBPK model from the equations of Kletting et al. 2016 (S1 File) |
| | `pbpk_fit.py` | per-patient fit on the 68Ga PET, posterior sampling, empirical Bayes |
| | `pbpk_calibration.py` | population calibration against published PSMA-617 half-lives and doses |
| | `pbpk_cohort.py` | virtual cohort generation (scenarios A and B) |
| Monte Carlo | `psma_mc.py` | GATE sources, material maps, native kernels, voxel target |
| doses | `cohort_doses.py` | Monte Carlo dose basis and organ/lesion doses; `dose_map` rebuilds any voxel map |
| network data | `net_data.py` | 4 mm shards, kinetic targets, `rebuild` (NumPy or torch); the Colab notebook imports it from Drive |
| | `spect_data.py` | activity of each source at the SPECT times for the update experiment |

Module names are part of the data format: the cohort and PBPK caches are pickles that reference `pbpk_fit` by name.
