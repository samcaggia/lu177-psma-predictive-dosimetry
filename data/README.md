# data/

Empty in the repository; nothing here is ever committed (`.gitignore`).

```
data/
├── raw/                                  raw datasets (or set LU177_DATASET to another folder)
│   ├── Deep Blue Dataset/patient_{4,6}/  SPECT_Cts/, TIA/, VOI/ as distributed by Deep Blue
│   ├── PSMA-PET-CT-Lesions_v3/           imagesTr/, labelsTr/, psma_metadata.csv, splits_final.json (FDAT v3)
│   │   └── psma_segmentations/           organs/ and salivary/ masks from notebooks/02_psma_data/09_totalsegmentator.ipynb
│   └── Totalsegmentator_dataset_v201/    optional
└── processed/                            written by the notebooks and scripts
    ├── deepblue/                         NIfTI per scan, kinetics/, voxel and dose maps     (notebooks/01_deepblue)
    └── psma/
        ├── *.csv, *.json                 inventory, lesions, split, PET headers, exclusions (nb08, scripts/dataset, src/psma_exclusions.py)
        ├── phantoms/, phantoms_qc/       one phantom per study                              (nb11)
        ├── pbpk/                         PBPK fit, calibration, caches                      (nb18, nb19)
        ├── cohort/, cohort_manifest.csv  virtual cohort                                     (nb20)
        ├── mc_pilot/, mc_full/           Monte Carlo runs                                   (nb14-15, scripts/monte_carlo/run_mc_full.py)
        ├── cohort_doses/                 dose basis and cohort doses                        (scripts/monte_carlo/run_cohort_doses.py, nb21)
        └── netdata/                      network data, shards, baselines (its README is the format reference) (nb22)
```

`deep_learning/runs/` (network outputs) is created by `deep_learning/main.ipynb` and the update scripts;
`LU177_NETDATA`, `LU177_COHORT` and `LU177_RUNS` redirect the network data, the cohort and the runs.

Everything derived from PSMA-PET-CT-Lesions or Deep Blue falls under their licences (`DATA_LICENSES.md`).
