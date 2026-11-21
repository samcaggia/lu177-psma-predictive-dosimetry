# Figures

Drawn by `scripts/analysis/make_readme_figures.py` (base environment) from the aggregate tables in `docs/results/`:

```bash
python scripts/analysis/make_readme_figures.py
```

| File | Content | Source |
|---|---|---|
| `pipeline.png` | overview of the pipeline | schematic, no data |
| `spect_update_error_vs_time.png` | median absolute log error of the TIA against the time of the single SPECT, methods M0-M5, frozen test split | `spect_update_test_mean_over_structures.csv` |
| `prior_contribution.png` | network prior minus population prior, without SPECT and after one SPECT, paired bootstrap 95% intervals | `spect_update_test_prior_ablation.csv` |
| `network_skill_by_structure.png` | skill of the pre-therapy network against the population prior, per structure, frozen test split | `network_test_by_structure.csv` |
| `deepblue_single_time_point.png` | single-time-point error on two real 177Lu-DOTATATE patients against the scan time | `deepblue_single_time_point_by_scan.csv` |
| `single_time_point_principle.png` | ratio of the estimate 2t A(t)/ln 2 to the true TIA for mono-exponential washout | analytical, no data |

No figure shows images or values of individual patients. Figures computed from the aggregate tables follow the terms
of those tables (see `DATA_LICENSES.md`); `pipeline.png` and `single_time_point_principle.png` contain no data.
