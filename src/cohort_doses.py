"""Monte Carlo doses of the frozen virtual cohort: integrity of the full run, per-phantom dose basis, organ, lesion and voxel doses per virtual patient.

Decisions and results are in notebooks/21_cohort_doses.ipynb. Voxel target as chosen in nb15: split native kernel on the source with per-lesion TIA, plus the long-range term L_c of every non-lesion compartment.
"""
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.fft import irfftn, next_fast_len, rfftn

import psma_mc as M

MC_FULL = M.ph.psma_data.PROCESSED / "mc_full"
COHORT = M.ph.psma_data.PROCESSED / "cohort"
OUT = M.ph.psma_data.PROCESSED / "cohort_doses"
SOURCES = tuple(M.COMPARTMENTS_V2)
ORGAN_SOURCES = tuple(c for c in SOURCES if c != "lesions")
TARGETS = tuple(M.TARGETS)
REST_SIGMA_MM, SIGMA_MM = 40.0, 10.0
EANM_SELF = {"kidneys_functional": (0.204, 422.0), "liver": (0.0376, 2360.0), "spleen": (0.377, 228.4), "salivary_glands": (0.947, 88.98)}
DELTA_E_MJ = 0.0853
MAP_SCALE = 1000.0
METAL_HU = 3000


def read(path):
    return np.asanyarray(nib.load(path).dataobj).astype(np.float64)


def phantom_info(study_id):
    """Masses of the targets and of each lesion, voxel counts, PET share of the spine source inside L2-L4."""
    organs, ct = M.load(study_id, "organs"), M.load(study_id, "ct")
    density = M.density_map(ct, organs).astype(np.float64)
    voxel_ml = float(np.prod(M.spacing_of(study_id))) / 1000
    pet = M.load(study_id, "pet_pct_ia_ml").astype(np.float64)
    rows = []
    for name, labels in M.TARGETS.items():
        mask = np.isin(organs, labels)
        rows.append(dict(study_id=study_id, target=name, voxels=int(mask.sum()), volume_ml=mask.sum() * voxel_ml, mass_g=float(density[mask].sum() * voxel_ml),
                         mean_density=float(density[mask].mean()) if mask.any() else np.nan, gas_fraction=float((density[mask] < 0.5).mean()) if mask.any() else np.nan))
    spine = np.isin(organs, M.COMPARTMENTS_V2["spine"])
    lumbar = organs == 12
    extra = dict(study_id=study_id, voxel_ml=voxel_ml, spine_pet_in_l2l4=float(pet[lumbar].sum() / pet[spine].sum()) if pet[spine].sum() > 0 else np.nan,
                 spine_mass_g=float(density[spine].sum() * voxel_ml), shape=list(organs.shape), spacing=list(M.spacing_of(study_id)))
    lesions = M.load(study_id, "lesions")
    lesion_rows = []
    for i in range(1, int(lesions.max()) + 1):
        m = lesions == i
        lesion_rows.append(dict(study_id=study_id, lesion=i - 1, voxels=int(m.sum()), volume_ml=m.sum() * voxel_ml, mass_g=float(density[m].sum() * voxel_ml)))
    return rows, extra, lesion_rows


def source_table():
    """Mean dose per unit TIA of every source on every target, Gy/(MBq h), from targets.csv of the full run (source runs only)."""
    t = pd.read_csv(MC_FULL / "targets.csv")
    t = t[t.kind == "source"].copy()
    t["source"] = t.run.str.removeprefix("D_").str.removesuffix("_pet")
    return t.melt(id_vars=["study_id", "source", "n", "seed"], value_vars=list(TARGETS), var_name="target", value_name="d")


class FastSplit:
    """SplitVsv of psma_mc with one forward transform shared by the short and the tail kernel, in single precision."""

    def __init__(self, shape, spacing, workers=4):
        kernel, r = M.native_kernel(spacing)
        self.shape, self.workers = tuple(shape), workers
        self.half = tuple(n // 2 for n in kernel.shape)
        self.size = tuple(next_fast_len(a + b - 1, real=True) for a, b in zip(shape, kernel.shape))
        scaled = (kernel * M.MGY_MBQ_S_TO_GY_MBQ_H).astype(np.float32)
        self.short_ft = rfftn(np.where(r <= M.TAIL_START_MM, scaled, 0).astype(np.float32), s=self.size, workers=workers)
        self.tail_ft = rfftn(np.where(r > M.TAIL_START_MM, scaled, 0).astype(np.float32), s=self.size, workers=workers)

    def _crop(self, full):
        h = self.half
        return np.maximum(full[h[0]:h[0] + self.shape[0], h[1]:h[1] + self.shape[1], h[2]:h[2] + self.shape[2]], 0.0)

    def __call__(self, source, density):
        ft = rfftn(source.astype(np.float32), s=self.size, workers=self.workers)
        short = self._crop(irfftn(ft * self.short_ft, s=self.size, workers=self.workers))
        tail = self._crop(irfftn(ft * self.tail_ft, s=self.size, workers=self.workers))
        return short / np.maximum(density, 0.1) + tail


def metal_mask(study_id, organs=None, ct=None):
    """Body voxels at or above METAL_HU, left out of the long-range low-pass."""
    organs = M.load(study_id, "organs") if organs is None else organs
    ct = M.load(study_id, "ct") if ct is None else ct
    return (organs > 0) & (ct >= METAL_HU)


class PhantomBasis:
    """Voxel dose basis of one phantom on its own grid, Gy/(MBq h) per unit-sum source: B_c = SplitVsv(w_c) + L_c for the organ sources, metal voxels out of the low-pass of L_c; lesions through SplitVsv with their own TIA."""

    def __init__(self, study_id, workers=4):
        self.study_id = study_id
        organs = M.load(study_id, "organs")
        ct = M.load(study_id, "ct")
        self.spacing = M.spacing_of(study_id)
        self.density = M.density_map(ct, organs).astype(np.float64)
        self.body = organs > 0
        metal = metal_mask(study_id, organs, ct)
        self.masks = M.target_masks(study_id)
        self.split = FastSplit(organs.shape, self.spacing, workers)
        self.lesion_index = M.load(study_id, "lesions").astype(np.int32)
        self.lesion_w = M.lesion_weights(study_id)
        self.basis = {}
        for c in ORGAN_SOURCES:
            path = MC_FULL / study_id / f"D_{c}_pet.nii.gz"
            w = M.source_weights_v2(study_id, c, "pet")
            if w is None or not path.exists():
                continue
            base = self.split(w, self.density)
            sigma = REST_SIGMA_MM if c == "rest_of_body" else SIGMA_MM
            self.basis[c] = (base + M.long_range_by_material(read(path), base, self.density, self.body, sigma, self.spacing, near=metal)).astype(np.float32)

    def lesion_map(self, lesion_tia):
        """Split-kernel dose of the lesions, Gy per MBq injected, from TIAC per lesion (0-based index of the cohort = label - 1)."""
        source = np.zeros(self.density.shape)
        for label, (idx, w) in self.lesion_w.items():
            value = lesion_tia.get(label - 1, 0.0)
            if value > 0:
                source[idx] += value * w
        return self.split(source, self.density).astype(np.float64) if source.any() else np.zeros(self.density.shape)

    def dose_map(self, organ_tia, lesion_tia):
        out = self.lesion_map(lesion_tia)
        for c, b in self.basis.items():
            if organ_tia.get(c, 0.0):
                out += organ_tia[c] * b
        return out

    def summarise(self, dose):
        """Mass-weighted mean on every target and lesion, in the units of dose."""
        rho = self.density
        targets = {t: float((dose[m] * rho[m]).sum() / rho[m].sum()) if m.any() else np.nan for t, m in self.masks.items()}
        n = int(self.lesion_index.max())
        if n:
            energy = np.bincount(self.lesion_index.ravel(), weights=(dose * rho).ravel(), minlength=n + 1)
            mass = np.bincount(self.lesion_index.ravel(), weights=rho.ravel(), minlength=n + 1)
            lesions = energy[1:] / np.maximum(mass[1:], 1e-12)
        else:
            lesions = np.zeros(0)
        return targets, lesions


def save_map(path, dose, affine):
    """Float32 NIfTI, written to a temporary name and moved into place."""
    image = nib.Nifti1Image(dose.astype(np.float32), affine)
    image.header.set_data_dtype(np.float32)
    temporary = path.with_name(path.name.replace(".nii.gz", ".tmp.nii.gz"))
    nib.save(image, temporary)
    os.replace(temporary, path)


def load_tia(split):
    tia = pd.read_parquet(COHORT / f"tia_{split}.parquet")
    tia["study_id"] = tia.virtual_id.str.rsplit("_", n=1).str[0]
    return tia


def tia_vectors(rows):
    """Organ TIAC and per-lesion TIAC (0-based index) of one virtual patient and scenario, in hours."""
    organ = {r.compartment: r.tiac_h for r in rows.itertuples() if not r.compartment.startswith("lesion_")}
    lesion = {int(r.compartment.split("_")[1]): r.tiac_h for r in rows.itertuples() if r.compartment.startswith("lesion_")}
    return organ, lesion


def phantom_doses(study_id, tia, workers=3, save_basis=True):
    """Basis, its summaries and the lesion part of every virtual patient and scenario of one phantom. Returns two tables in Gy/(MBq h) per unit TIAC (basis) and Gy/GBq (lesion part); with tia None, the basis only."""
    folder = OUT / "basis" / study_id
    folder.mkdir(parents=True, exist_ok=True)
    basis = PhantomBasis(study_id, workers)
    affine = nib.load(M.ph.PHANTOMS / study_id / "ct.nii.gz").affine
    rows = []
    for c, b in basis.basis.items():
        if save_basis:
            save_map(folder / f"B_{c}.nii.gz", np.where(basis.body, b, 0), affine)
        targets, lesions = basis.summarise(b.astype(np.float64))
        rows += [dict(study_id=study_id, source=c, kind="target", name=t, d=v) for t, v in targets.items()]
        rows += [dict(study_id=study_id, source=c, kind="lesion", name=str(j), d=v) for j, v in enumerate(lesions)]
    if tia is None:
        return pd.DataFrame(rows), None, 0
    placed = {label - 1 for label in basis.lesion_w}
    out, done = [], {}
    for (virtual_id, scenario, arm, split), group in tia.groupby(["virtual_id", "scenario", "arm", "split"], sort=True):
        organ, lesion = tia_vectors(group)
        key = tuple(sorted((j, round(v, 12)) for j, v in lesion.items()))
        if key not in done:
            if lesion and placed:
                targets, lesions = basis.summarise(basis.lesion_map(lesion) * MAP_SCALE)
            else:
                targets, lesions = {t: 0.0 for t in TARGETS}, np.zeros(int(basis.lesion_index.max()))
            done[key] = (targets, lesions)
        targets, lesions = done[key]
        unplaced = sum(v for j, v in lesion.items() if j not in placed)
        out += [dict(virtual_id=virtual_id, scenario=scenario, arm=arm, split=split, kind="target", name=t, lesion_part=v, unplaced_lesion_tiac=unplaced) for t, v in targets.items()]
        out += [dict(virtual_id=virtual_id, scenario=scenario, arm=arm, split=split, kind="lesion", name=str(j), lesion_part=v, unplaced_lesion_tiac=unplaced) for j, v in enumerate(lesions)]
    return pd.DataFrame(rows), pd.DataFrame(out), len(done)


def dose_map(virtual_id, scenario="A", split="train", arm="main", basis=None):
    """Voxel dose map of one virtual patient, Gy/GBq, rebuilt exactly from the stored basis and one lesion transform."""
    study_id = virtual_id.rsplit("_", 1)[0]
    tia = load_tia(split)
    rows = tia[(tia.virtual_id == virtual_id) & (tia.scenario == scenario) & (tia.arm == arm)]
    if rows.empty:
        raise KeyError(f"{virtual_id} {arm} {scenario} not in tia_{split}")
    rows = linked_tia(rows, gland_link([study_id])).reset_index()
    rows = rows.melt(id_vars=["virtual_id", "arm", "scenario", "split", "study_id"], var_name="compartment", value_name="tiac_h").pipe(lambda f: pd.concat([f, tia[(tia.virtual_id == virtual_id) & (tia.scenario == scenario) & (tia.arm == arm) & tia.compartment.str.startswith("lesion_")]]))
    organ, lesion = tia_vectors(rows)
    basis = basis or StoredBasis(study_id)
    return basis.dose_map(organ, lesion) * MAP_SCALE


class StoredBasis(PhantomBasis):
    """PhantomBasis read back from disk instead of rebuilt from the Monte Carlo maps."""

    def __init__(self, study_id, workers=4):
        self.study_id = study_id
        organs = M.load(study_id, "organs")
        self.spacing = M.spacing_of(study_id)
        self.density = M.density_map(M.load(study_id, "ct"), organs).astype(np.float64)
        self.body = organs > 0
        self.masks = M.target_masks(study_id)
        self.split = FastSplit(organs.shape, self.spacing, workers)
        self.lesion_index = M.load(study_id, "lesions").astype(np.int32)
        self.lesion_w = M.lesion_weights(study_id)
        folder = OUT / "basis" / study_id
        self.basis = {p.name[2:-7]: np.asanyarray(nib.load(p).dataobj).astype(np.float32) for p in sorted(folder.glob("B_*.nii.gz"))}


def gland_link(studies):
    """Factor from the TIA of the PBPK glands to the TIA of the Monte Carlo gland source (labels 5-8): same concentration, volume of the mask over the PBPK gland volume."""
    targets = pd.read_csv(OUT / "phantom_targets.csv").set_index(["study_id", "target"]).volume_ml
    out = {}
    for s in studies:
        meta = json.loads((M.ph.PHANTOMS / s / "meta.json").read_text())
        used = any(v["use_for_pbpk"] for v in meta["salivary_glands"].values())
        pbpk_ml = meta["groups"]["salivary_glands_pbpk" if used else "salivary_glands_all"]["volume_ml"] or 1.0
        out[s] = float(targets[(s, "salivary_glands")] / pbpk_ml)
    return pd.Series(out, name="gland_factor")


def linked_tia(tia, gland_factor):
    """TIAC per Monte Carlo source: glands scaled to their mask, the excess taken from the rest of body so the total is kept."""
    organ = tia[~tia.compartment.str.startswith("lesion_")].pivot_table(index=["virtual_id", "arm", "scenario", "split", "study_id"], columns="compartment", values="tiac_h").fillna(0.0)
    factor = gland_factor.reindex(organ.index.get_level_values("study_id")).to_numpy()
    excess = organ["salivary_glands"].to_numpy() * (factor - 1)
    organ["salivary_glands"] += excess
    organ["rest_of_body"] -= excess
    return organ


def assemble(split, link_glands=True):
    """Organ and lesion doses, Gy/GBq, of every virtual patient and scenario of one split from the parts of run_cohort_doses.py. Organ part: linked TIAC times the basis means; lesion part: the split-kernel transform of the lesions."""
    tia = load_tia(split)
    parts = OUT / "parts"
    studies = [s for s in sorted(tia.study_id.unique()) if (parts / f"{s}_lesion_part.parquet").exists()]
    tia = tia[tia.study_id.isin(studies)]
    basis = pd.concat([pd.read_parquet(parts / f"{s}_basis.parquet") for s in studies])
    lesion_part = pd.concat([pd.read_parquet(parts / f"{s}_lesion_part.parquet") for s in studies])
    factor = gland_link(studies) if link_glands else pd.Series(1.0, index=studies)
    organ = linked_tia(tia, factor)
    keys = organ.index.to_frame(index=False)
    frames = []
    for kind in ("target", "lesion"):
        b = basis[basis.kind == kind].pivot_table(index=["study_id", "name"], columns="source", values="d").fillna(0.0)
        sources = [c for c in b.columns if c in organ.columns]
        rows = keys.merge(b.reset_index(), on="study_id")
        tiac = organ.reindex(pd.MultiIndex.from_frame(rows[["virtual_id", "arm", "scenario", "split", "study_id"]]))[sources].to_numpy()
        parts_by_source = tiac * rows[sources].to_numpy() * MAP_SCALE
        out = rows[["virtual_id", "arm", "scenario", "split", "study_id", "name"]].copy()
        out["organ_part"] = parts_by_source.sum(axis=1)
        for j, c in enumerate(sources):
            out[f"from_{c}"] = parts_by_source[:, j]
        lp = lesion_part[lesion_part.kind == kind].drop(columns="kind")
        out = out.merge(lp, on=["virtual_id", "arm", "scenario", "split", "name"], how="left")
        out["lesion_part"] = out.lesion_part.fillna(0.0)
        out["dose"] = out.organ_part + out.lesion_part
        frames.append(out)
    organs, lesions = frames
    lesions["lesion"] = lesions.pop("name").astype(int)
    return organs.rename(columns=dict(name="target")), lesions, factor
