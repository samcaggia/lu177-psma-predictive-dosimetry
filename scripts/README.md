# scripts/: command-line entry points

Run them with `.venv/bin/python scripts/<group>/<script>.py` from the repository root (relative defaults such as
`--output output` are relative to the current folder). Each script finds the repository from its own location.
The GATE scripts score with DoseActor `hit_type = split` by default (`--hit-type random` reproduces the old outputs).

| Group | Script | What it does | Output | Cost |
|---|---|---|---|---|
| `gate/` | `lu177_water_validation.py` | Lu-177 voxel S-value kernel in water (`vsv`) and sphere check (`sphere`) | `output/vsv/`, `output/sphere/` | minutes to hours by decays |
| | `lu177_native_kernel.py` | kernel on an anisotropic PET voxel | `output/vsv_native*/` | about 13 min per grid at 2e7 decays, 8 threads |
| | `lu177_patient_mc.py` | voxelised Lu-177 source in a CT phantom (GATE 10) | the folder given by `--output` | about 3400 decays/s on 8 threads |
| `monte_carlo/` | `run_mc_pilot.py` | pilot runs of nb14 (resumable) | `data/processed/psma/mc_pilot/` | |
| | `run_mc_retest.py` | re-test runs and native kernels of nb15 | `mc_pilot/v2/`, `output/vsv_native/` | |
| | `run_mc_full.py` | 10 sources per training/test phantom, 5e5 decays each (resumable); calls `check_mc_full.py` | `data/processed/psma/mc_full/` | about 5 days CPU |
| | `check_mc_full.py` | independent check of the voxel target against direct runs | `mc_full/independent_check*.csv` | |
| | `run_mc_hit_check.py` | paired split/random DoseActor runs | `data/processed/psma/mc_hit_check/` | |
| | `run_cohort_doses.py` | dose basis per phantom and cohort doses (`--metal-basis`) | `data/processed/psma/cohort_doses/` | about 8 h on 3 processes |
| `dataset/` | `fetch_psma_pet_headers.py` | PT DICOM headers from the public IDC bucket (`--limit 0` rebuilds offline) | `data/processed/psma/pet_dicom_headers.csv` | about 597 requests |
| | `check_psma_suv_definition.py` | SUV definition of the NIfTI against full DICOM instances | `suv_definition_check.csv` | |
| | `audit_psma_slice_mismatch.py` | interpolated slices of one study | `psma_slice_audit_3959.csv` | |
| | `fetch_opendose_svalues.py` | OpenDose S-values (CC BY-SA) | `mc_pilot/opendose_lu177_icrp110am.csv` | |
| `analysis/` | `compare_vsv_lanconelli.py` | GATE kernel against Lanconelli 2012 (`--lanconelli` = your reference table, not included) | `output/vsv/lanconelli_*` | |
| | `vsv_hit_type_impact.py` | effect of the DoseActor hit type in water | `output/vsv/hit_type_*` | |
| | `hit_type_cohort_check.py` | effect of the hit type on the cohort doses | `output/vsv/hit_type_cohort*` | |
