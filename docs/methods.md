# Methods overview

## Question

Can the absorbed dose of [177Lu]Lu-PSMA radioligand therapy be predicted, voxel by voxel, from the pre-therapy
68Ga-PSMA PET/CT, and how much does a single post-therapy SPECT add when it updates that prediction for the
individual patient by Bayesian inference? The working hypothesis is that the limit of pre-therapy prediction is
physical and biological: a PET about 1 h after injection carries little information on the washout over several
days, so physics (kinetic and PBPK models) and one late measurement should help more than a different network.

## 1. Dosimetry pipeline on real kinetics (Deep Blue)

Two 177Lu-DOTATATE patients of the Deep Blue datasets (SPECT/CT at four time points, organ and lesion contours, a
reference TIA map), the cases of the SNMMI Lu-177 dosimetry challenge.

- Loading of the quantitative SPECT with the decay-correction convention read from each scan (`ADMIN`: corrected to
  injection, `START`: physical activity at the start of acquisition).
- Structure-wise alignment of the scans (rigid on CT or SPECT, or centroid translation, by structure), time-activity
  curves, uptake-washout fit A(t) = A0 (exp(-lw t) - exp(-(lw + lu) t)) per organ and per voxel, TIA uncertainty by
  Monte Carlo on the data points.
- Absorbed dose by voxel S-value convolution with a GATE water kernel, local deposition and full GATE Monte Carlo in
  the patient CT. Density-scaled VSV agrees with Monte Carlo within 2-3% on organ means; the TIA map, not the dose
  engine, dominates the dose uncertainty.
- Single-time-point dosimetry: literature effective half-lives, the Hänscheid formula, population shapes, and a
  Bayesian single-time-point estimate with literature priors on the kinetics. A SPECT at about 4 days recovers organ
  TIAs within a few percent; the Bayesian update reaches comparable accuracy and adds a calibrated interval. An early
  amplitude measurement does not help, even with an oracle uptake prior: what a pre-therapy prior must carry is the
  washout.

## 2. PSMA data and phantoms

PSMA-PET-CT-Lesions (autoPET): 597 PSMA PET/CT studies with lesion labels. Inventory, PET exclusions (implausible
header data, interpolated slices), patient-level split of the 68Ga studies (132 training/validation subjects, 34
test subjects). Organs from TotalSegmentator (`total` and `head_glands_cavities` for the salivary glands). One phantom
per study: CT, PET in decay-corrected %IA/mL, compartment labels (functional kidney, liver, spleen, salivary glands,
bladder, GI tract, blood pool, spine, rest of body), numbered lesions. Biodistribution checked against published
68Ga-PSMA-11 values.

## 3. PBPK model and virtual cohort

- **Model.** A whole-body PSMA PBPK model written from the published equations of Kletting et al. 2016 (plasma,
  interstitial, receptor-bound and internalised ligand; kidneys with filtration; liver, spleen, salivary glands,
  lesions; rest of body), verified against an independent SBML implementation (identical TIA after correcting four
  errors found in that SBML) and against the published figures of the original paper. Hot and cold ligand reduce
  exactly to total ligand, so 68Ga and 177Lu share the kinetics at equal amount.
- **Fit to the real PET.** Receptor densities per patient from the 68Ga PET at about 1 h, with nuisance parameters
  marginalised, local background, recovery and image error in the likelihood, posterior sampling, and empirical Bayes
  for the population on the training split. The 1 h PET constrains liver and spleen densities and lesion delivery,
  not the washout, the therapy ligand or the lesion density.
- **Population calibration.** Release rates, per-organ agent factors (from the diagnostic to the therapeutic agent)
  and between-patient spreads calibrated against published [177Lu]Lu-PSMA-617 half-lives and absorbed doses; therapy
  amount from a published measurement.
- **Virtual cohort.** Ten posterior draws per phantom: 1800 training/validation and 470 test virtual patients, each
  with simulated 68Ga values, therapy TIA per source and per lesion, and kinetic parameters (A0, lambda_bio, early TIA
  share, uptake rate). Five scenarios of the link between PET uptake and washout (A: none; B: rank correlation
  -0.6 to +0.6). The cohort is a low-burden staging population, not a therapy population.

## 4. Monte Carlo doses

GATE 10 (Geant4 11) with the full Lu-177 decay. Water kernels on the three native PET grids; one Monte Carlo run per
phantom and per source (ten sources, PET-weighted, 5e5 decays each) gives a dose basis from which any virtual
patient's organ, lesion and voxel dose is rebuilt exactly: a split native kernel (electrons within 10 mm scaled by
density, photon tail as dose) per draw, plus long-range corrections by material from the Monte Carlo. Self-dose
agrees with published S-values within about 1%. Metal implants are excluded from the long-range low-pass. A scoring
artefact of the GATE DoseActor on photon steps (the default `random` hit type) inflates the photon tail of water
kernels by up to 1.7 times at 6-10 mm; the scripts now use the `split` scoring, and its effect on the cohort doses
was measured below 1% on organs.

## 5. Network

A SwinUNETR (MONAI, self-supervised weights) takes the PET and CT on a 4 mm grid and predicts, per voxel, the mean and
log-variance of five kinetic heads: log A0 per mL, log lambda_bio, early TIA share, log uptake rate, log TIA per mL.
Loss: Gaussian negative log-likelihood, class-balanced over structures, plus a consistency term on the washout within
each structure. Training on Colab (one GPU). Predictions are aggregated per structure, calibrated by out-of-fold
scale factors on validation, and turned into a gated prior M0: for each structure group and quantity, network or
population prior, chosen on half of the validation subjects.

## 6. Single-SPECT update

A realistic SPECT is simulated at 24, 48, 96 or 168 h: patient-specific resolution (9-12 mm), misregistration, 5%
calibration error, correlated noise matched to the Deep Blue SPECTs. The truth includes biology outside the PBPK
model (log-normal factors on amplitude and late washout, SD 0.15), and in a second condition a washout rate that
drifts in time ("shape mismatch").

| Method | Idea |
|---|---|
| M0 | prior only (network-gated or population) |
| M1 | classical single time point (Hänscheid; population rate with recovery correction) |
| M2 | Bayesian update of the organ uptake-washout parameters with the VOI activity as likelihood, model-discrepancy term chosen by coverage on validation |
| M3 | test-time optimisation of the voxel maps against the SPECT image |
| M4 | test-time optimisation of the network weights (output head) |
| M5 | fit of the PBPK model to the SPECT, Laplace interval |

All settings (gates, discrepancy, learning rates, steps) were fixed on a tuning half of validation and reported on
the other half; the test split was then evaluated once with every choice frozen (checksums of every choice file
checked by the pipeline, `deep_learning/test_run.py`).

## 7. Main results (test split)

Mean over the six structures with uptake of the median absolute log error of the TIA (`docs/results/`):

| Method | no SPECT | 24 h | 48 h | 96 h | 168 h |
|---|---|---|---|---|---|
| M0 gated prior | 0.355 | | | | |
| M0 population prior | 0.380 | | | | |
| M2 Bayesian update (gated prior) | | 0.151 | 0.145 | 0.135 | 0.175 |
| M2 Bayesian update (population prior) | | 0.157 | 0.140 | 0.146 | 0.190 |
| M5 PBPK fit | | 0.156 | 0.145 | 0.162 | 0.199 |
| M3 map optimisation | | 0.321 | 0.242 | 0.220 | 0.263 |
| M4 weight optimisation | | 0.397 | 0.341 | 0.312 | 0.304 |

- One SPECT reduces the error from 0.36 to 0.14-0.18 (median absolute log error of the TIA, M2) on organs and lesions; absorbed-dose errors equal
  TIA errors because cross-dose is 0.2-6% of organ and lesion doses (L2-L4 marrow is the exception).
- Once a SPECT is available, the network-derived prior changes the error by -0.015 to +0.005 against the population
  prior (paired difference, bootstrap over subjects; the interval excludes zero only at 96 h, -0.010); without a
  SPECT it gains 0.025 (interval including zero). The PET at 1 h
  informs the uptake level of some organs, not the washout.
- M2 intervals are close to nominal (68% coverage 0.54-0.68, 95% coverage 0.89-0.94); the Laplace intervals of M5
  are too narrow. Test-time optimisation of maps or weights does not beat the Bayesian organ update (M4 is a
  negative result).
- The network beats the population prior on liver, spleen, bladder and rest of body, not on lesions (it memorises
  the training lesions); the PBPK fit to the PET is better than the network on every structure.

## 8. Limitations

- Virtual cohort: the truth is a PBPK model calibrated on published population data; agent factors are effective
  parameters, the cohort is a low-burden staging population, and the mismatch levels (SD 0.15, the drift) are
  assumptions with sensitivity analyses.
- Real kinetics come from two Deep Blue patients (177Lu-DOTATATE): a demonstration, not a validation.
- The network was trained on one scenario (no PET-washout link); scenarios with a link are prepared, not trained.
- Salivary glands: segmentation volumes below published values; lacrimal glands not segmented.
- Monte Carlo: the cohort doses use kernels with the original photon scoring (effect measured, below 1% on organs).
