"""Monte Carlo pilot on the PSMA phantoms: material map, per-compartment unit sources, VSV dose, target statistics.

GATE reads the images in index space (identity direction, origin 0), so inputs and outputs align voxel by voxel with the phantom.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.signal import fftconvolve
from scipy.fft import irfftn, next_fast_len, rfftn

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))
import psma_phantoms as ph  # noqa: E402

PILOT = ph.psma_data.PROCESSED / "mc_pilot"
MC_SCRIPT = WORKSPACE / "scripts" / "gate" / "lu177_patient_mc.py"
KERNEL = WORKSPACE / "output" / "vsv" / "vsv_lu177_mGy_per_MBq_s.npy"
KERNEL_MM = 4.0
MGY_MBQ_S_TO_GY_MBQ_H = 3.6
DECAYS_PER_MBQ_H = 3.6e9

COMPARTMENTS = {
    "kidneys_functional": [1], "liver": [3], "spleen": [4], "salivary_glands": [5, 6, 7, 8], "bladder": [9], "gi_tract": [10],
    "rest_of_body": [2, 11, 12, 13, 14], "lesions": [15],
}
TARGETS = {
    "kidneys_functional": [1], "liver": [3], "spleen": [4], "salivary_glands": [5, 6, 7, 8], "bladder": [9], "gi_tract": [10],
    "blood_pool": [11], "spine_L2_L4": [12], "rest_of_body": [14], "lesions": [15],
}
OVERRIDE_VALUE = 10000
OVERRIDE_LABELS = {
    1: "kidney_functional", 2: "kidney_nonfunctional", 3: "liver", 4: "spleen", 5: "parotid_gland_left", 6: "parotid_gland_right",
    7: "submandibular_gland_left", 8: "submandibular_gland_right", 9: "urinary_bladder", 11: "blood_pool",
}
SOFT_TISSUE = dict(elements=["H", "C", "N", "O", "Na", "P", "S", "Cl", "K"],
                   weights=[0.103, 0.134, 0.030, 0.723, 0.002, 0.002, 0.002, 0.002, 0.002])


def load(study_id, name):
    return np.asanyarray(nib.load(ph.PHANTOMS / study_id / f"{name}.nii.gz").dataobj)


def spacing_of(study_id):
    return tuple(float(z) for z in nib.load(ph.PHANTOMS / study_id / "ct.nii.gz").header.get_zooms()[:3])


def write_image(array, spacing, path):
    image = sitk.GetImageFromArray(np.asarray(array).T)
    image.SetSpacing(spacing)
    sitk.WriteImage(image, str(path))


def read_image(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).T


def override_density(organs):
    """g/mL of the overridden voxels, nan elsewhere."""
    density = np.full(organs.shape, np.nan, dtype=np.float32)
    for label, name in OVERRIDE_LABELS.items():
        density[organs == label] = ph.NOMINAL_DENSITY[name]
    return density


def density_map(ct_hu, organs):
    """Density of every voxel in the Monte Carlo geometry: Schneider from the HU, ICRP 110 in the overridden compartments."""
    density = ph.density_from_hu(ct_hu).astype(np.float32)
    override = override_density(organs)
    return np.where(np.isnan(override), density, override)


def write_material_inputs(study_id, folder):
    """CT with the overridden voxels set to labels outside the Schneider HU range, and the json of their materials."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    ct, organs = load(study_id, "ct").astype(np.int16), load(study_id, "organs")
    out, items = ct.copy(), []
    for k, (label, name) in enumerate(OVERRIDE_LABELS.items()):
        value = OVERRIDE_VALUE + k
        out[organs == label] = value
        items.append(dict(name=f"icrp110_{name}", value=value, density_g_cm3=ph.NOMINAL_DENSITY[name], **SOFT_TISSUE))
    write_image(out, spacing_of(study_id), folder / "ct_materials.mhd")
    (folder / "materials.json").write_text(json.dumps(items, indent=1))


def source_weights(study_id, compartment, variant):
    """Unit-sum spatial weights of a compartment: real PET inside it, or uniform per unit volume."""
    organs = load(study_id, "organs")
    mask = np.isin(organs, COMPARTMENTS[compartment])
    if not mask.any():
        return None
    if variant == "uniform":
        weights = mask.astype(np.float64)
    else:
        weights = np.where(mask, load(study_id, "pet_pct_ia_ml").astype(np.float64), 0.0)
        if weights.sum() <= 0:
            return None
    return weights / weights.sum()


def run_gate(ct, materials, source, output, n, seed, threads=4, extra=()):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, str(MC_SCRIPT), "--ct", str(ct), "--source", str(source), "--output", str(output), "--n", str(n),
               "--threads", str(threads), "--seed", str(seed), "--material-overrides", str(materials), *extra]
    subprocess.run(command, check=True, capture_output=True, env=env, cwd=str(output))


def target_masks(study_id):
    organs = load(study_id, "organs")
    return {name: np.isin(organs, labels) for name, labels in TARGETS.items()}


def mass_weighted_on_targets(dose, density, masks):
    """Organ mean dose: energy over mass, so low-density voxels (gas) do not dominate. dose is on the phantom grid."""
    return {name: float(np.sum(dose[mask] * density[mask], dtype=np.float64) / np.sum(density[mask], dtype=np.float64)) if mask.any() else np.nan
            for name, mask in masks.items()}


def deposit_on_kernel_grid(weights, spacing):
    """Activity moved to the 4 mm kernel grid by trilinear deposit (total conserved exactly)."""
    idx = np.nonzero(weights)
    values = weights[idx]
    position = [(i + 0.5) * s / KERNEL_MM - 0.5 for i, s in zip(idx, spacing)]
    shape = tuple(int(np.ceil(n * s / KERNEL_MM)) + 1 for n, s in zip(weights.shape, spacing))
    grid = np.zeros(shape, dtype=np.float64)
    base = [np.floor(p).astype(np.int64) for p in position]
    frac = [p - b for p, b in zip(position, base)]
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = values.copy()
                for axis, d in enumerate((dx, dy, dz)):
                    w *= frac[axis] if d else 1 - frac[axis]
                i = np.clip(base[0] + dx, 0, shape[0] - 1)
                j = np.clip(base[1] + dy, 0, shape[1] - 1)
                k = np.clip(base[2] + dz, 0, shape[2] - 1)
                np.add.at(grid, (i, j, k), w)
    return grid


def back_to_phantom_grid(array, shape, spacing):
    coordinates = np.meshgrid(*[((np.arange(n) + 0.5) * s / KERNEL_MM - 0.5) for n, s in zip(shape, spacing)], indexing="ij", sparse=True)
    return ndi.map_coordinates(array, np.broadcast_arrays(*coordinates), order=1, mode="nearest").astype(np.float32)


def vsv_dose(weights, spacing, density, kernel=None):
    """Gy per MBq h of VSV dose on the phantom grid: water kernel on 4 mm, linear back, then divided by the local density."""
    kernel = np.load(KERNEL) if kernel is None else kernel
    water = fftconvolve(deposit_on_kernel_grid(weights, spacing), kernel, mode="same") * MGY_MBQ_S_TO_GY_MBQ_H
    back = back_to_phantom_grid(water, weights.shape, spacing)
    return back / np.maximum(density, 0.1), back


def through_kernel_grid(dose, spacing):
    """The same field after the round trip to the 4 mm grid and back, with no kernel: the blur of the resampling alone."""
    voxel_ratio = np.prod(spacing) / KERNEL_MM ** 3
    return back_to_phantom_grid(deposit_on_kernel_grid(dose, spacing) * voxel_ratio, dose.shape, spacing)


# Re-test (notebook 15): 10 compartments, rest of body without halo, per-lesion sources, kernels on the native PET grids.
V2 = PILOT / "v2"
NATIVE_KERNELS = WORKSPACE / "output" / "vsv_native"
COMPARTMENTS_V2 = {
    "kidneys_functional": [1], "liver": [3], "spleen": [4], "salivary_glands": [5, 6, 7, 8], "bladder": [9], "gi_tract": [10],
    "blood_pool": [11], "spine": [12, 13], "rest_of_body": [2, 14], "lesions": [15],
}
HALO_ORGANS = {"kidneys_functional": [1, 2], "salivary_glands": [5, 6, 7, 8], "bladder": [9]}
HALO_MM = 10.0
TAIL_START_MM = 10.0
TAIL_BIN_MM = 4.0
TEST_T_EFF_H_V2 = {"kidneys_functional": 50, "liver": 70, "spleen": 70, "salivary_glands": 30, "bladder": 20, "gi_tract": 40, "blood_pool": 30, "spine": 40, "rest_of_body": 40}
TEST_LESION_T_EFF_RANGE_H = (30.0, 130.0)
TEST_ACTIVITY_MBQ = 7400.0
TEST_SEED = 20261104


def halo_shells(organs, spacing):
    """Per organ group, rest-of-body voxels within HALO_MM of it, kidney fluid included with the kidneys; a voxel goes to the nearest group."""
    rest = np.isin(organs, COMPARTMENTS_V2["rest_of_body"])
    distances = {name: ndi.distance_transform_edt(~np.isin(organs, labels), sampling=spacing) for name, labels in HALO_ORGANS.items()}
    names = list(distances)
    stack = np.stack([distances[n] for n in names])
    nearest = np.argmin(stack, axis=0)
    close = np.min(stack, axis=0) <= HALO_MM
    return {n: rest & close & (nearest == i) for i, n in enumerate(names)}, rest


def rest_without_halo(study_id):
    """PET of the rest of body with the shells and kidney fluid set to the median of the other rest-of-body voxels."""
    organs, pet = load(study_id, "organs"), load(study_id, "pet_pct_ia_ml").astype(np.float64)
    shells, rest = halo_shells(organs, spacing_of(study_id))
    any_shell = np.logical_or.reduce(list(shells.values()))
    median = float(np.median(pet[rest & ~any_shell]))
    weights = np.where(rest, pet, 0.0)
    weights[any_shell] = median
    return weights, median, shells, pet


def source_weights_v2(study_id, compartment, variant):
    if compartment != "rest_of_body" or variant == "uniform":
        organs = load(study_id, "organs")
        mask = np.isin(organs, COMPARTMENTS_V2[compartment])
        if not mask.any():
            return None
        weights = mask.astype(np.float64) if variant == "uniform" else np.where(mask, load(study_id, "pet_pct_ia_ml").astype(np.float64), 0.0)
    else:
        weights = rest_without_halo(study_id)[0]
    return weights / weights.sum() if weights.sum() > 0 else None


def lesion_weights(study_id):
    """Per lesion: (voxel indices, unit-sum weights), from the numbered lesions of the phantom and the PET."""
    lesions, pet = load(study_id, "lesions"), load(study_id, "pet_pct_ia_ml").astype(np.float64)
    out = {}
    for i in range(1, int(lesions.max()) + 1):
        idx = np.nonzero(lesions == i)
        w = pet[idx]
        if w.sum() > 0:
            out[i] = (idx, w / w.sum())
    return out


def test_tia_v2(study_id):
    """Test TIA vector, not a PBPK draw: activity x PET fraction x T_eff/ln 2. Excess of a shell over the median goes to its organ, so activity is conserved; lesions draw their own T_eff."""
    organs = load(study_id, "organs")
    weights, median, shells, pet = rest_without_halo(study_id)
    voxel_ml = float(np.prod(spacing_of(study_id))) / 1000
    fractions = {c: pet[np.isin(organs, labels)].sum() * voxel_ml / 100 for c, labels in COMPARTMENTS_V2.items() if c not in ("rest_of_body", "lesions")}
    for name, shell in shells.items():
        fractions[name] += (pet[shell] - median).sum() * voxel_ml / 100
    fractions["rest_of_body"] = weights.sum() * voxel_ml / 100
    tia = {c: TEST_ACTIVITY_MBQ * f * TEST_T_EFF_H_V2[c] / np.log(2) for c, f in fractions.items()}
    rng = np.random.default_rng(TEST_SEED)
    lesion_tia = {}
    lesions = load(study_id, "lesions")
    for i in range(1, int(lesions.max()) + 1):
        t_eff = float(np.exp(rng.uniform(*np.log(TEST_LESION_T_EFF_RANGE_H))))
        lesion_tia[i] = TEST_ACTIVITY_MBQ * pet[lesions == i].sum() * voxel_ml / 100 * t_eff / np.log(2)
    return tia, lesion_tia, fractions


def native_kernel(spacing, tail_start_mm=TAIL_START_MM):
    """Kernel on a PET voxel, mGy/(MBq s): symmetrised over the voxel symmetries; beyond tail_start_mm the radial profile of the photon tail."""
    name = "kernel_{:.2f}x{:.2f}x{:.2f}".format(*spacing).replace(".", "p")
    raw = np.load(NATIVE_KERNELS / f"{name}_raw.npy")
    sym = (raw + raw[::-1] + raw[:, ::-1] + raw[:, :, ::-1] + raw[::-1, ::-1] + raw[::-1, :, ::-1] + raw[:, ::-1, ::-1] + raw[::-1, ::-1, ::-1]) / 8
    if np.isclose(spacing[0], spacing[1]):
        sym = (sym + sym.transpose(1, 0, 2)) / 2
    axes = [(np.arange(n) - n // 2) * s for n, s in zip(raw.shape, spacing)]
    x, y, z = np.meshgrid(*axes, indexing="ij", sparse=True)
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    edges = np.arange(tail_start_mm, r.max() + TAIL_BIN_MM, TAIL_BIN_MM)
    index = np.digitize(r, edges)
    counts = np.bincount(index.ravel(), minlength=len(edges) + 1)
    totals = np.bincount(index.ravel(), weights=sym.ravel(), minlength=len(edges) + 1)
    valid = (counts[1:len(edges)] > 0) & (totals[1:len(edges)] > 0)
    centres = 0.5 * (edges[:-1] + edges[1:])[valid]
    profile = (totals[1:len(edges)] / np.maximum(counts[1:len(edges)], 1))[valid]
    slope = np.log(profile[1] / profile[0]) / (centres[1] - centres[0])
    start = np.log(profile[0]) + slope * (tail_start_mm - centres[0])
    tail = np.exp(np.interp(r, np.r_[tail_start_mm, centres], np.r_[start, np.log(profile)]))
    return np.where(r > tail_start_mm, tail, sym), r


class NativeVsv:
    """Convolution with a pre-transformed native kernel: Gy/(MBq h) in water for a unit-sum source on the phantom grid."""

    def __init__(self, shape, spacing, kernel=None):
        kernel = native_kernel(spacing)[0] if kernel is None else kernel
        self.shape = tuple(shape)
        self.half = tuple(n // 2 for n in kernel.shape)
        self.size = tuple(next_fast_len(a + b - 1, real=True) for a, b in zip(shape, kernel.shape))
        self.kernel_ft = rfftn(kernel * MGY_MBQ_S_TO_GY_MBQ_H, s=self.size)

    def __call__(self, source):
        full = irfftn(rfftn(source, s=self.size) * self.kernel_ft, s=self.size)
        h = self.half
        return np.maximum(full[h[0]:h[0] + self.shape[0], h[1]:h[1] + self.shape[1], h[2]:h[2] + self.shape[2]], 0.0)


def long_range_term(mc_dose, vsv_water, density, body, sigma_mm, spacing):
    """Smooth MC minus VSV: low-pass of the energy difference over low-pass of the mass, inside the body, in Gy/(MBq h)."""
    sigma = [sigma_mm / s for s in spacing]
    residual = np.where(body, (mc_dose * density - vsv_water), 0.0)
    mass = np.where(body, density, 0.0).astype(np.float64)
    low_mass = ndi.gaussian_filter(mass, sigma, mode="constant")
    return np.where(body, ndi.gaussian_filter(residual, sigma, mode="constant") / np.maximum(low_mass, 1e-3), 0.0)


MATERIAL_EDGES = (0.5, 1.15)
NEAR_SOURCE_MM = 10.0


class SplitVsv:
    """Native kernel in two parts: within TAIL_START_MM over the density (electrons), beyond it as dose (photons, mass absorption close to water)."""

    def __init__(self, shape, spacing):
        kernel, r = native_kernel(spacing)
        self.short = NativeVsv(shape, spacing, np.where(r <= TAIL_START_MM, kernel, 0.0))
        self.tail = NativeVsv(shape, spacing, np.where(r > TAIL_START_MM, kernel, 0.0))

    def __call__(self, source, density):
        return self.short(source) / np.maximum(density, 0.1) + self.tail(source)


def material_classes(density, body):
    low, high = MATERIAL_EDGES
    return [body & (density < low), body & (density >= low) & (density < high), body & (density >= high)]


def near_source(weights, spacing, mm=NEAR_SOURCE_MM):
    return ndi.maximum_filter(weights > 0, size=[2 * int(round(mm / s)) + 1 for s in spacing])


def _mass_lowpass(residual, density, mask, sigma):
    mass = ndi.gaussian_filter(np.where(mask, density, 0.0), sigma, mode="constant")
    energy = ndi.gaussian_filter(np.where(mask, residual * density, 0.0), sigma, mode="constant")
    return energy / np.maximum(mass, 1e-3), mass > 0.02


def long_range_by_material(mc_dose, base, density, body, sigma_mm, spacing, near=None):
    """Mass-weighted low-pass of MC minus base within lung, soft tissue and bone separately, in Gy/(MBq h). near: voxels left out of the filter unless they cover over half the body."""
    sigma = [sigma_mm / s for s in spacing]
    residual = mc_dose - base
    support = body if near is None or (body & near).sum() > 0.5 * body.sum() else body & ~near
    term, valid = _mass_lowpass(residual, density, support, sigma)
    out = np.where(valid, term, 0.0)
    for tissue in material_classes(density, body):
        term, valid = _mass_lowpass(residual, density, tissue & support, sigma)
        out = np.where(tissue & valid, term, out)
    return np.where(body, out, 0.0)
