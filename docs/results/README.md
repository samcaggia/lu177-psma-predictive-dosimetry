# Aggregate results

Every number in this folder is an aggregate (median, mean, coverage) over many structures and virtual patients. No
image, map, per-patient or per-lesion value is included.

**Where the numbers come from.** All files except the two `deepblue_*` tables come from the **virtual cohort**:
real PSMA PET/CT anatomies (PSMA-PET-CT-Lesions, CC BY-NC 4.0) with therapy kinetics simulated by a PBPK model and
doses by Monte Carlo, so the "truth" is known by construction. They are results of the frozen test split (47 phantoms,
470 virtual patients), evaluated once after every choice had been fixed on the validation split. The `deepblue_*`
tables are the only results on real patient kinetics (two patients, 13 organs and lesions, 177Lu-DOTATATE).

Errors are absolute errors of the natural logarithm (0.10 is about 10%). Coverage is the share of cases whose truth
falls inside the 68% or 95% interval. "Six structures with uptake" = functional kidneys, liver, spleen, salivary
glands, GI tract, lesions.

| File | Content |
|---|---|
| `spect_update_test_mean_over_structures.csv` | single-SPECT update, methods M0-M5 and variants, mean over the six structures with uptake of the median error of the TIA, and mean coverage, per condition (main mismatch, shape mismatch) and SPECT time (24, 48, 96, 168 h) |
| `spect_update_test_by_structure.csv` | the same per structure: number of cases, bias, median error, 68% and 95% coverage |
| `spect_update_test_prior_ablation.csv` | network-derived prior against population prior: paired difference of the median error (gated minus population), bootstrap interval over subjects (`low`, `high`), with and without a SPECT (`t_h` empty = no SPECT) |
| `dose_test_summary.csv` | absorbed dose of the update experiment per target (kidneys, liver, spleen, glands, lesions, L2-L4, whole body): median error of the dose and of the TIA, median true dose (Gy/GBq and Gy per 7.4 GBq cycle), cross-dose share, coverage |
| `network_test_by_structure.csv` | network prediction from the pre-therapy PET/CT against the population prior and the PBPK fit, per structure and quantity (log A0, log lambda_bio, early share, log TIA): bias, median error, RMSE, coverage, skill |
| `deepblue_single_time_point_by_scan.csv` | Deep Blue, one SPECT at a time: Hänscheid formula, population rate and Bayesian update (M1, M2 with this code) against the 4-point reference, median error (%) and how often the reference falls in the interval |
| `deepblue_bayesian_stp_by_scan.csv` | Deep Blue, classical single-time-point methods and the Bayesian variants with literature priors, per scan, with interval half-widths and coverage (counts over 13 structures) |
| `vsv_kernel_lu177_water.png` | Lu-177 voxel S-value kernel (4 mm, water) computed with GATE 10, with the corrected and the original photon scoring. The published reference values used for validation are not redistributed |

Scan times of the Deep Blue tables: scan 1 about 4 h, scan 2 about 1 day, scan 3 about 4 days, scan 4 day 5 or day 8
after injection.

Method names: M0 prior only (network-gated or population), M1 classical single time point (Hänscheid, population
rate), M2 Bayesian update of organ kinetics, M3 map-level test-time optimisation, M4 weight-level test-time
optimisation, M5 PBPK fit. "gated" = per-structure choice between network and population prior fixed on validation;
"no discrepancy" = M2 without the model-discrepancy term. See `docs/methods.md`.
