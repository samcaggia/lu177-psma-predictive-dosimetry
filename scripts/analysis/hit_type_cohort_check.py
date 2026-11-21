#!/usr/bin/env python3
"""Effect of the DoseActor hit type on the cohort doses (output/vsv/hit_type_cohort_check.md).

The patient Monte Carlo maps (mc_full) are not affected by the hit type (steps end on the CT voxel boundaries; paired rerun in run_mc_hit_check.py: split equals random to 6e-16), so only the native water kernels change.
Per phantom, three dose operators (rows: targets and lesions; columns: Monte Carlo sources and every lesion):
  stored       the basis on disk (cohort_doses, nb21),
  rebuilt_old  rebuilt here with the old native kernels and the mc_full maps (must equal stored),
  corrected    native kernels scored with hit type split (output/vsv_native_split), same mc_full maps.
Applied to the cohort TIAC of the virtual patients of these phantoms (true doses) and to the TIAC estimates of the single-SPECT experiment (spect_dose.case_rows).
Usage: python hit_type_cohort_check.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "deep_learning"))
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import cohort_doses as cd  # noqa: E402
import psma_mc as M  # noqa: E402

KERNELS = {"old": ROOT / "output" / "vsv_native", "split": ROOT / "output" / "vsv_native_split"}
OUT = ROOT / "output" / "vsv" / "hit_type_cohort"
VARIANTS = {"rebuilt_old": "old", "corrected": "split"}
STUDIES = ["psma_9b733a4106c0fe37_2016-10-14", "psma_24f62f21c04402ec_2015-04-26", "psma_d7c4cf294221ea45_2017-03-03"]
SPHERES_ML = (0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 300)
COHORT_TARGETS = ["kidneys_functional", "liver", "spleen", "salivary_glands", "bladder", "gi_tract", "blood_pool", "spine_L2_L4", "rest_of_body"]
SPECT_TARGETS = ["kidneys_functional", "liver", "spleen", "salivary_glands", "spine_L2_L4", "blood_pool"]


def read(path):
    return np.asanyarray(nib.load(path).dataobj).astype(np.float64)


class VariantBasis(cd.PhantomBasis):
    def __init__(self, study_id, kernel, workers=4):
        self.study_id = study_id
        organs = M.load(study_id, "organs")
        ct = M.load(study_id, "ct")
        self.spacing = M.spacing_of(study_id)
        self.density = M.density_map(ct, organs).astype(np.float64)
        self.body = organs > 0
        metal = cd.metal_mask(study_id, organs, ct)
        self.masks = M.target_masks(study_id)
        saved = M.NATIVE_KERNELS
        M.NATIVE_KERNELS = KERNELS[kernel]
        try:
            self.split = cd.FastSplit(organs.shape, self.spacing, workers)
        finally:
            M.NATIVE_KERNELS = saved
        self.lesion_index = M.load(study_id, "lesions").astype(np.int32)
        self.lesion_w = M.lesion_weights(study_id)
        self.basis = {}
        for c in cd.ORGAN_SOURCES:
            path = cd.MC_FULL / study_id / f"D_{c}_pet.nii.gz"
            w = M.source_weights_v2(study_id, c, "pet")
            if w is None or not path.exists():
                continue
            base = self.split(w, self.density)
            sigma = cd.REST_SIGMA_MM if c == "rest_of_body" else cd.SIGMA_MM
            self.basis[c] = (base + M.long_range_by_material(read(path), base, self.density, self.body, sigma, self.spacing, near=metal)).astype(np.float32)


def operator(basis):
    """Gy/GBq per hour of TIAC; rows COHORT_TARGETS, whole_body, lesion_j; columns organ sources then lesion_j (placed lesions)."""
    rho, body = basis.density, basis.body
    n_lesions = int(basis.lesion_index.max())

    def column(dose):
        targets, lesions = basis.summarise(dose)
        return np.concatenate([[targets[t] for t in COHORT_TARGETS], [float((dose[body] * rho[body]).sum() / rho[body].sum())], lesions])

    sources, columns = [], []
    for name, b in basis.basis.items():
        sources.append(name)
        columns.append(column(b.astype(np.float64)))
    for label in sorted(basis.lesion_w):
        sources.append(f"lesion_{label - 1}")
        columns.append(column(basis.lesion_map({label - 1: 1.0})))
    targets = COHORT_TARGETS + ["whole_body"] + [f"lesion_{j}" for j in range(n_lesions)]
    return np.stack(columns, axis=1) * cd.MAP_SCALE, targets, sources


def operators(study):
    path = OUT / f"operators_{study}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {v: z[v] for v in ["stored", *VARIANTS]}, list(z["targets"]), list(z["sources"])
    out = {}
    stored = cd.StoredBasis(study)
    out["stored"], targets, sources = operator(stored)
    for name, kernel in VARIANTS.items():
        basis = VariantBasis(study, kernel)
        basis.basis = {c: basis.basis[c] for c in stored.basis}
        matrix, t, s = operator(basis)
        assert t == targets and s == sources, (study, name)
        out[name] = matrix
    np.savez(path, targets=np.array(targets), sources=np.array(sources), **out)
    return out, targets, sources


def sphere_self_doses():
    """Self-dose of uniform spheres through the split kernel (short part over density 1.04, tail as dose), old and split kernels, each PET grid."""
    rows = []
    for path in sorted(KERNELS["old"].glob("kernel_*.json")):
        spacing = tuple(json.loads(path.read_text())["voxel_mm"])
        shape = tuple(int(np.ceil(160 / v)) | 1 for v in spacing)
        axes = [(np.arange(n) - n // 2) * v for n, v in zip(shape, spacing)]
        x, y, z = np.meshgrid(*axes, indexing="ij", sparse=True)
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        density = np.full(shape, 1.04)
        ops = {}
        for kernel, folder in KERNELS.items():
            saved = M.NATIVE_KERNELS
            M.NATIVE_KERNELS = folder
            try:
                ops[kernel] = cd.FastSplit(shape, spacing, 4)
            finally:
                M.NATIVE_KERNELS = saved
        for ml in SPHERES_ML:
            mask = r <= (3 * ml * 1000 / (4 * np.pi)) ** (1 / 3)
            if mask.sum() == 0:
                continue
            w = mask / mask.sum()
            d = {k: float(op(w, density)[mask].mean()) for k, op in ops.items()}
            rows.append(dict(grid="x".join(f"{v:.2f}" for v in spacing), volume_ml=ml, voxels=int(mask.sum()), self_old=d["old"], self_split=d["split"], split_over_old=d["split"] / d["old"]))
    return pd.DataFrame(rows)


def cohort_tiac(study, sources):
    """Linked TIAC per source (columns as the operator) of every virtual patient, arm and scenario of the phantom."""
    tia = cd.load_tia("train")
    tia = tia[tia.study_id == study]
    organ = cd.linked_tia(tia, cd.gland_link([study]))
    lesion = tia[tia.compartment.str.startswith("lesion_")].pivot_table(index=["virtual_id", "arm", "scenario", "split", "study_id"], columns="compartment", values="tiac_h")
    table = organ.join(lesion).fillna(0.0)
    return table.reindex(columns=sources).fillna(0.0)


def true_doses(studies):
    rows = []
    for study in studies:
        ops, targets, sources = operators(study)
        tiac = cohort_tiac(study, sources)
        for variant, matrix in ops.items():
            dose = tiac.to_numpy() @ matrix.T
            frame = pd.DataFrame(dose, index=tiac.index, columns=targets).reset_index().melt(id_vars=list(tiac.index.names), var_name="name", value_name="dose")
            frame["variant"] = variant
            rows.append(frame)
    out = pd.concat(rows, ignore_index=True)
    out["target"] = np.where(out.name.str.startswith("lesion_"), "lesion", out.name)
    wide = out.pivot_table(index=["virtual_id", "arm", "scenario", "study_id", "target", "name"], columns="variant", values="dose").reset_index()
    return wide


def spect_rows(studies):
    """spect_dose.case_rows on the reporting-half virtual patients of the studies, once per operator."""
    import spect_dose as sd
    import spect_report as report
    choice = json.loads((report.EVAL / f"tto_choice_{sd.SCENARIO}.json").read_text())
    tia = pd.read_parquet(cd.COHORT / "tia_train.parquet")
    tia = tia[(tia.scenario == sd.SCENARIO) & (tia.arm == "main")].set_index(["virtual_id", "compartment"]).tiac_h
    cache = {s: operators(s) for s in studies}
    keep = [COHORT_TARGETS.index(t) for t in SPECT_TARGETS]
    frames = []
    for variant in ["stored", *VARIANTS]:
        def matrix_of(study_id, variant=variant):
            ops, targets, sources = cache[study_id]
            factor = float(cd.gland_link([study_id]).iloc[0])
            rows_idx = keep + [len(COHORT_TARGETS)] + list(range(len(COHORT_TARGETS) + 1, len(targets)))
            return ops[variant][rows_idx], SPECT_TARGETS + ["whole_body"] + targets[len(COHORT_TARGETS) + 1:], sources, factor
        sd.dose_matrix = matrix_of
        for name, tag in sd.CONDITIONS.items():
            table = report.load(tag, choice)
            table = table[table.method.isin(sd.METHODS) & table.virtual_id.str.rsplit("_", n=1).str[0].isin(studies)]
            frames.append(sd.case_rows(table, tia).assign(condition=name, variant=variant))
    rows = pd.concat(frames, ignore_index=True)
    return rows, sd


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    studies = STUDIES
    pd.set_option("display.width", 250, "display.max_rows", 300)

    spheres = sphere_self_doses()
    spheres.to_csv(OUT / "lesion_sphere_self_dose.csv", index=False)
    print(spheres.round(4).to_string())

    for study in studies:
        ops, targets, sources = operators(study)
        spect_cache = ROOT / "deep_learning" / "runs" / "eval_A" / "dose" / "matrix" / f"{study}.npz"
        print(study, "rebuilt_old against stored: max relative difference",
              float(np.nanmax(np.abs(ops["rebuilt_old"] - ops["stored"]) / np.maximum(np.abs(ops["stored"]), 1e-30))))
        if spect_cache.exists():
            z = np.load(spect_cache)
            rows = [COHORT_TARGETS.index(t) for t in ["kidneys_functional", "liver", "spleen", "salivary_glands", "spine_L2_L4"]] + [len(COHORT_TARGETS)] + list(range(len(COHORT_TARGETS) + 1, len(targets)))
            print(study, "stored operator against the spect_dose cache: max relative difference",
                  float(np.nanmax(np.abs(ops["stored"][rows] - z["matrix"]) / np.maximum(np.abs(z["matrix"]), 1e-30))), flush=True)
        n_org = sum(not c.startswith("lesion_") for c in sources)
        lesion_rows = [i for i, t in enumerate(targets) if t.startswith("lesion_")]
        diag = [(i, sources.index(targets[i])) for i in lesion_rows if targets[i] in sources]
        if diag:
            ratio = np.array([ops["corrected"][i, j] / ops["stored"][i, j] for i, j in diag])
            print(study, f"lesion self-dose operator corrected/stored: n {len(ratio)}, median {np.median(ratio):.4f}, range {ratio.min():.4f}-{ratio.max():.4f}; organ columns of n_org={n_org}")

    wide = true_doses(studies)
    wide["corrected_over_stored"] = wide.corrected / wide.stored
    wide["rebuilt_over_stored"] = wide.rebuilt_old / wide.stored
    wide.to_csv(OUT / "true_doses.csv", index=False)
    main_arm = wide[wide.arm == "main"]
    summary = main_arm.groupby(["study_id", "target"]).agg(
        n=("corrected_over_stored", "size"),
        median=("corrected_over_stored", "median"),
        min=("corrected_over_stored", "min"),
        max=("corrected_over_stored", "max"),
        rebuilt_over_stored_max_dev=("rebuilt_over_stored", lambda x: float(np.abs(x - 1).max())),
        dose_stored_median=("stored", "median")).reset_index()
    summary.to_csv(OUT / "true_dose_ratio_by_phantom.csv", index=False)
    overall = main_arm.groupby("target").corrected_over_stored.describe(percentiles=[0.05, 0.5, 0.95])
    overall.to_csv(OUT / "true_dose_ratio_overall.csv")
    print(summary.round(4).to_string())
    print(overall.round(4).to_string())

    rows, sd = spect_rows(studies)
    rows.to_csv(OUT / "spect_dose_rows.csv", index=False)
    score = sd.score(rows.drop(columns="variant").assign(condition=rows.condition + " | " + rows.variant))
    score[["condition", "variant"]] = score.condition.str.split(" \\| ", expand=True)
    score.to_csv(OUT / "spect_dose_scores.csv", index=False)
    d = score[score.method.isin([sd.M0, sd.M2, sd.M5]) & score.t_h.isin([48.0, 96.0])]
    piv = d.pivot_table(index=["condition", "method", "t_h", "target"], columns="variant", values="median_abs_error")
    piv["corrected_minus_stored"] = piv.corrected - piv.stored
    piv["corrected_over_stored"] = piv.corrected / piv.stored
    piv.to_csv(OUT / "spect_error_comparison.csv")
    print(piv.round(4).to_string())
    ranks = pd.concat([sd.ranking(g.drop(columns="variant")).assign(variant=v) for v, g in score.groupby("variant")])
    ranks.to_csv(OUT / "spect_ranking.csv", index=False)
    key = ["condition", "target", "t_h", "method"]
    rk = ranks.pivot_table(index=key, columns="variant", values="rank_dose")
    rt = ranks.pivot_table(index=key, columns="variant", values="rank_tiac")
    result = dict(rank_by_dose_changed=int((rk.corrected != rk.stored).sum()), rank_cells=int(len(rk)),
                  dose_rank_equals_tiac_rank={v: float((rk[v] == rt[v]).mean()) for v in rk.columns})
    (OUT / "ranking_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
