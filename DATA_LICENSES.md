# Data licences

**No data is included in this repository.** The MIT licence in `LICENSE` covers the code only. The datasets below
must be obtained from their sources and are used under their own terms; anything you derive from them (phantoms,
virtual cohorts, Monte Carlo doses, network weights, tables, figures) inherits those terms.

| Dataset | Licence | Notes |
|---|---|---|
| PSMA-PET-CT-Lesions (autoPET), NIfTI version 3, FDAT (doi:10.57754/FDAT.g27kx-86t35) | CC BY-NC 4.0 | images, labels, metadata and splits. **Anything derived from these data must not be used commercially.** |
| PSMA-PET-CT-Lesions, DICOM version, TCIA / NCI Imaging Data Commons (doi:10.7937/r7ep-3x37) | CC BY 4.0 | used only for the PET DICOM headers (weight, activity, timing) |
| Lu-177 DOTATATE anonymized patient datasets, University of Michigan Deep Blue Data (doi:10.7302/0n8e-rz46 and related records) | CC BY-NC 4.0 (verified on the Deep Blue record) | any publication or presentation must carry the acknowledgement text given in `README.md` and in the `acknowledgement` field of the `dewaraja2021*` entries of `references.bib` |
| TotalSegmentator dataset v2.0.1 | CC BY 4.0 | TotalSegmentator software: Apache-2.0 |
| OpenDose S-values (opendose.polsl.pl) | CC BY-SA 4.0 | downloaded by `scripts/dataset/fetch_opendose_svalues.py`, not included |
| Lanconelli et al., Phys Med Biol 2012, voxel S-values | copyright of the authors and publisher | not included; supply your own reference table to `scripts/analysis/compare_vsv_lanconelli.py` |

The aggregate tables in `docs/results/` are summary statistics computed on a virtual cohort built from
PSMA-PET-CT-Lesions anatomies, and on two Deep Blue patients; they are provided for non-commercial use under the
same terms as those datasets (CC BY-NC 4.0).
