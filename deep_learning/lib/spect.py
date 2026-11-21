"""
Virtual post-therapy SPECT on the 4 mm grid, in torch: the true voxel activity of a virtual patient at the SPECT time,
the simulated acquisition (resolution, calibration, misregistration, correlated noise, all drawn with a seed fixed by
the virtual patient and the time) and the forward model that the update methods assume. The two differ on purpose,
as a real acquisition differs from its model: the simulation draws the resolution of each patient and shifts the
image, the forward model has one nominal resolution and no shift.
"""
import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import _guard
from .scenario import scenario_file


@dataclass
class SpectConfig:
    """
    Forward model assumed by the update methods: Gaussian resolution fwhm_mm, global calibration factor with
    log-normal SD calibration_sd.
    Simulation only: the resolution of a virtual patient is uniform in fwhm_range_mm; the image is shifted by a vector
    of random direction and length uniform in [0, shift_max_voxels] voxels (misregistration of the VOIs).
    Noise (both): Gaussian, spatially correlated as white noise blurred with a Gaussian of noise_fwhm_mm; voxel SD =
    voxel_cv x A x (A / reference_activity) ** -cv_exponent, A the blurred activity of the voxel (fraction of the
    injected activity). Values from the Deep Blue SPECTs at day 4 and later (data/processed/deepblue/kinetics/
    spect_noise_day4.csv; 12 organ VOIs, 3.9 mm grid, slow component removed): local voxel CV 0.30 at 8.35e-7 per
    voxel with a log-log slope of -0.17 against the activity (not the -0.5 of counting noise); autocorrelation
    exp(-d^2 / (2 L^2)) with L = 6.0 mm, that of white noise blurred with a 10 mm FWHM Gaussian.
    """
    fwhm_mm: float = 10.0
    calibration_sd: float = 0.05
    fwhm_range_mm: Tuple[float, float] = (9.0, 12.0)
    shift_max_voxels: float = 2.0
    voxel_cv: float = 0.30
    reference_activity: float = 8.35e-7
    cv_exponent: float = 0.17
    noise_fwhm_mm: float = 10.0
    seed: int = 20261114

    def noise_sd(self, activity):
        """Voxel noise SD at a blurred activity."""
        a = activity.clamp_min(1e-4 * self.reference_activity)
        return self.voxel_cv * a * (a / self.reference_activity) ** (-self.cv_exponent)


def load_activity(netdata, studies, scenario="A", arm="main", test=False):
    """Source activities at the SPECT times (netdata/spect_activity.parquet, spect_activity_<scenario>.parquet for a scenario other than A) of the given phantoms; negative rounding values set to 0. test: spect_activity_test.parquet (scenario A only), for the frozen test run."""
    if test:
        if scenario != "A":
            raise NotImplementedError("No test table of a B scenario")
        table = pd.read_parquet(Path(netdata) / "spect_activity_test.parquet")
    else:
        table = pd.read_parquet(_guard(scenario_file(netdata, "spect_activity", scenario)))
    table = table[(table.arm == arm) & (table.scenario == scenario)]
    table = table[table.virtual_id.str.rsplit("_d", n=1).str[0].isin(set(studies))]
    return table.assign(activity=table.activity.clip(lower=0.0))


def gaussian_kernel(fwhm_mm, grid_mm, device, dtype):
    sigma = fwhm_mm / (2.0 * math.sqrt(2.0 * math.log(2.0))) / grid_mm
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _separable(volume, kernels):
    shape = volume.shape
    x = volume.reshape(-1, 1, *shape[-3:])
    for axis, kernel in enumerate(kernels):
        view = [1, 1, 1, 1, 1]
        view[2 + axis] = -1
        padding = [0, 0, 0]
        padding[axis] = (kernel.numel() - 1) // 2
        x = F.conv3d(x, kernel.view(view), padding=padding)
    return x.reshape(shape)


def blur(volume, fwhm_mm, grid_mm):
    """Separable Gaussian blur of (..., X, Y, Z) with zeros outside the volume; identity for fwhm_mm = 0."""
    if fwhm_mm <= 0:
        return volume
    return _separable(volume, [gaussian_kernel(fwhm_mm, grid_mm, volume.device, volume.dtype)] * 3)


def shift(volume, offset):
    """Translation of (..., X, Y, Z) by a real offset in voxels (per axis, linear interpolation, zeros outside)."""
    kernels = []
    for value in offset:
        whole = math.floor(value)
        part = value - whole
        radius = abs(whole) + 1
        kernel = torch.zeros(2 * radius + 1, device=volume.device, dtype=volume.dtype)
        kernel[radius - whole] = 1.0 - part
        kernel[radius - whole - 1] = part
        kernels.append(kernel)
    return _separable(volume, kernels)


def noise_gain_of_voxel(cfg, grid_mm):
    """Sum of the squares of the 3-D noise kernel: the variance of blurred unit white noise."""
    if cfg.noise_fwhm_mm <= 0:
        return 1.0
    kernel = gaussian_kernel(cfg.noise_fwhm_mm, grid_mm, "cpu", torch.float64)
    return float((kernel ** 2).sum()) ** 3


def correlation_volume(cfg, grid_mm):
    """Number of voxels over which the noise is correlated, 1 / (variance of blurred unit white noise)."""
    return 1.0 / noise_gain_of_voxel(cfg, grid_mm)


def source_tables(cohort, i, activity_of_source):
    """Lookup tables (compartment code, lesion id) of a {source: value} mapping, like those of net_data.parameter_arrays."""
    nd = cohort.nd
    comp = torch.zeros(nd.LESION_CODE + 1, dtype=torch.float64)
    lesion = torch.zeros(cohort.n_lesions[i] + 1, dtype=torch.float64)
    for source, value in activity_of_source.items():
        if source.startswith("lesion_"):
            j = int(source.split("_")[1]) + 1
            if j < lesion.numel():
                lesion[j] = value
        else:
            comp[nd.COMPARTMENT_CODES[source]] = value
    return comp.to(cohort.device), lesion.to(cohort.device)


def activity_map(cohort, i, activity_of_source, dtype=torch.float32):
    """True voxel activity: source activity x unit weight w(x) of the voxel within its source (net_data.rebuild)."""
    comp, lesion = source_tables(cohort, i, activity_of_source)
    ids = cohort.lesion[i].long()
    per_source = torch.where(ids > 0, lesion[ids], comp[cohort.compartment[i].long()])
    return (per_source * cohort.targets(i, 0)["w"].double()).to(dtype)


def forward(volume, cfg, grid_mm, calibration=1.0):
    """Expected SPECT of an activity volume under the assumed model: nominal resolution and calibration factor. Differentiable."""
    return calibration * blur(volume, cfg.fwhm_mm, grid_mm)


def _generator(cfg, virtual_id, t_h, device):
    seed = (zlib.crc32(f"{virtual_id}|{float(t_h):.3f}".encode()) + cfg.seed) % (2 ** 63 - 1)
    return torch.Generator(device=device).manual_seed(seed)


def acquisition(cfg, virtual_id):
    """Resolution (mm) and misregistration (voxels, 3 components) of a virtual patient, the same at every SPECT time."""
    rng = np.random.default_rng([cfg.seed, zlib.crc32(str(virtual_id).encode())])
    fwhm = float(rng.uniform(*cfg.fwhm_range_mm))
    direction = rng.normal(size=3)
    offset = direction / np.linalg.norm(direction) * rng.uniform(0.0, cfg.shift_max_voxels)
    return fwhm, tuple(float(v) for v in offset)


def simulate(volume, cfg, grid_mm, virtual_id, t_h):
    """
    Measured SPECT of a true activity volume: resolution and shift of the virtual patient, a calibration factor drawn
    per image, correlated Gaussian noise with the voxel SD of cfg.noise_sd at the blurred activity. Returns
    (image, {'calibration', 'fwhm_mm', 'shift'}); the random numbers depend only on (seed, virtual_id, t_h).
    """
    fwhm, offset = acquisition(cfg, virtual_id)
    generator = _generator(cfg, virtual_id, t_h, volume.device)
    calibration = float(torch.exp(cfg.calibration_sd * torch.randn((), generator=generator, device=volume.device)))
    expected = blur(volume, fwhm, grid_mm)
    if cfg.shift_max_voxels > 0:
        expected = shift(expected, offset)
    noise = torch.randn(volume.shape, generator=generator, device=volume.device, dtype=volume.dtype)
    if cfg.noise_fwhm_mm > 0:
        noise = blur(noise, cfg.noise_fwhm_mm, grid_mm) / math.sqrt(noise_gain_of_voxel(cfg, grid_mm))
    image = calibration * expected + cfg.noise_sd(expected.clamp_min(0.0)) * noise
    return image, dict(calibration=calibration, fwhm_mm=fwhm, shift=offset)


def voi_sums(volume, st):
    """Sum of (..., X, Y, Z) over the mask of every structure of a PhantomStructures: (..., structures)."""
    flat = volume.flatten(-3)[..., st.entries]
    return flat.new_zeros(*flat.shape[:-1], len(st)).index_add_(flat.dim() - 1, st.segment, flat)


@torch.no_grad()
def voi_noise_gain(cohort, i, st, cfg, chunk=6):
    """
    Per structure, the variance of the mean over its VOI of the unit-variance correlated noise field: between
    1 / voxels (independent voxels) and 1 (one fully correlated VOI). The noise SD of a VOI sum is then
    sqrt(gain) x voxels x the voxel noise SD at the mean voxel activity.
    """
    grid = float(cohort.phantoms.grid_mm.iloc[0])
    shape = cohort.shape[i]
    counts = torch.tensor(st.n_voxels, dtype=torch.float32, device=cohort.device)
    if cfg.noise_fwhm_mm <= 0:
        return (1.0 / counts).double()
    gains = []
    for start in range(0, len(st), chunk):
        stop = min(start + chunk, len(st))
        masks = torch.zeros(stop - start, int(np.prod(shape)), device=cohort.device)
        for k in range(start, stop):
            a, b = st.bounds[k]
            masks[k - start, st.entries[a:b]] = 1.0 / st.n_voxels[k]
        blurred = blur(masks.view(-1, *shape), cfg.noise_fwhm_mm, grid)
        gains.append((blurred ** 2).flatten(1).sum(dim=1) / noise_gain_of_voxel(cfg, grid))
    return torch.cat(gains).double()
