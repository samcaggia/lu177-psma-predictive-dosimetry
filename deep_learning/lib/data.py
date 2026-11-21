"""Network data: parquet and rebuild checks, the cohort held in GPU memory, patch sampling with augmentation and target rebuild on the GPU."""
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TARGET_FIELDS = ("w", "a0", "lambda_bio", "lambda_u", "tia_late", "tia_early", "tia", "degenerate", "inside")
AIR = -1.0


def import_net_data(src_dir):
    """Imports src/net_data.py, the single source of the target rebuild."""
    if not os.path.isfile(os.path.join(src_dir, "net_data.py")):
        raise FileNotFoundError(f"Cannot find {src_dir}/net_data.py: check SRC_DIR")
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    sys.modules.pop("net_data", None)
    return importlib.import_module("net_data")


def is_test_file(path):
    name = os.path.basename(str(path))
    return "_test" in name or name.startswith("test_")


def _guard(path):
    if is_test_file(path):
        raise PermissionError(f"{path} belongs to the test split: not read here")
    return path


def check_parquet(netdata):
    """Reads every parquet file of netdata except the test ones; one row per file with its size or the error."""
    import pyarrow
    print(f"pandas {pd.__version__} | pyarrow {pyarrow.__version__}")
    rows = []
    for path in sorted(Path(netdata).rglob("*.parquet")):
        if is_test_file(path) or "_cache" in path.parts:
            continue
        row = dict(file=str(path.relative_to(netdata)))
        try:
            table = pd.read_parquet(path)
            row.update(rows=len(table), columns=table.shape[1], ok=True, error="")
        except Exception as e:
            row.update(rows=0, columns=0, ok=False, error=f"{type(e).__name__}: {e}")
        rows.append(row)
    return pd.DataFrame(rows)


def shard_names(netdata, splits=("train", "val")):
    """Shard files of the given network splits, from phantoms.parquet."""
    phantoms = pd.read_parquet(_guard(Path(netdata) / "phantoms.parquet"))
    names = sorted(phantoms.shard[phantoms.net_split.isin(splits)].unique())
    for name in names:
        _guard(name)
    return names


class GpuCohort:
    """
    Phantoms of the training and validation splits held on the device as stored (float16, uint8, int16), with the
    lookup tables of every draw of one scenario and the voxel index lists used by the foreground sampling.
    """

    def __init__(self, nd, netdata, shard_dir, device, scenario="A", arm="main", splits=("train", "val")):
        self.nd, self.device, self.arm = nd, device, arm
        netdata = Path(netdata)
        phantoms = pd.read_parquet(_guard(netdata / "phantoms.parquet"))
        self.phantoms = phantoms[phantoms.net_split.isin(splits)].reset_index(drop=True)
        self.index = {s: i for i, s in enumerate(self.phantoms.study_id)}
        self.voxel_ml = float(self.phantoms.grid_mm.iloc[0]) ** 3 / 1000.0
        if self.phantoms.grid_mm.nunique() != 1:
            raise ValueError("Phantoms on different grids")
        sources = pd.read_parquet(_guard(netdata / "sources.parquet"))
        self._params_path = _guard(netdata / "params.parquet")

        n = len(self.phantoms)
        self.pet, self.ct, self.compartment, self.lesion, self.weight = ([None] * n for _ in range(5))
        self.foreground, self.shape = [None] * n, [None] * n
        self.weight_sum_comp, self.weight_sum_lesion, self.n_lesions = [None] * n, [None] * n, [0] * n
        by_study = dict(tuple(sources.groupby("study_id")))
        for name in sorted(self.phantoms.shard.unique()):
            shard = nd.read_shard(_guard(Path(shard_dir) / name), studies=set(self.index))
            for study, arrays in shard.items():
                i = self.index[study]
                lesion = arrays["lesion"].astype(np.int16)
                if lesion.min() < 0:
                    raise ValueError(f"{study}: lesion ids do not fit int16")
                self.pet[i] = torch.from_numpy(arrays["pet"]).to(device)
                self.ct[i] = torch.from_numpy(arrays["ct"]).to(device)
                self.weight[i] = torch.from_numpy(arrays["weight"]).to(device)
                self.compartment[i] = torch.from_numpy(arrays["compartment"]).to(device)
                self.lesion[i] = torch.from_numpy(lesion).to(device)
                self.shape[i] = tuple(lesion.shape)
                self.n_lesions[i] = max(int(lesion.max()), int(by_study[study].lesion_id.max()))
                comp_sum, lesion_sum = nd.source_sum_tables(by_study[study], self.n_lesions[i])
                self.weight_sum_comp[i] = torch.from_numpy(comp_sum).float().to(device)
                self.weight_sum_lesion[i] = torch.from_numpy(lesion_sum).float().to(device)
                flat = self.compartment[i].flatten()
                codes = [c for c in range(1, nd.LESION_CODE + 1) if c != nd.COMPARTMENT_CODES["rest_of_body"]]
                self.foreground[i] = [idx for idx in (torch.nonzero(flat == c).flatten().int() for c in codes)
                                      if idx.numel()]
            del shard
        missing = [s for s, i in self.index.items() if self.pet[i] is None]
        if missing:
            raise FileNotFoundError(f"{len(missing)} phantoms not found in the shards: {missing[:3]}")
        self.load_scenario(scenario)

    def load_scenario(self, scenario):
        """Lookup tables (draw, code or lesion id, 4) of every phantom for one scenario, and the target kind of each code."""
        nd = self.nd
        params = pd.read_parquet(self._params_path)
        kinds = params.drop_duplicates("structure").set_index("structure").target_kind
        params = params[(params.arm == self.arm) & (params.scenario == scenario) & params.has_source
                        & params.study_id.isin(self.index)]
        if params.empty:
            raise ValueError(f"No parameters for arm {self.arm}, scenario {scenario}")
        self.scenario = scenario
        self.kind_of_code = torch.zeros(nd.LESION_CODE + 1, dtype=torch.long)
        for structure, code in nd.COMPARTMENT_CODES.items():
            self.kind_of_code[code] = nd.TARGET_KINDS.index(kinds[structure]) + 1
        self.is_curve = (self.kind_of_code == nd.TARGET_KINDS.index("curve") + 1).to(self.device)
        self.is_tia = (self.kind_of_code == nd.TARGET_KINDS.index("tia") + 1).to(self.device)

        columns = ["draw", "source", "a0", "lw", "lu", "tiac_early_h"]
        self.comp_table, self.lesion_table = [None] * len(self.index), [None] * len(self.index)
        for study, group in params[["study_id"] + columns].groupby("study_id"):
            i = self.index[study]
            draws = sorted(group.draw.unique())
            if draws != list(range(len(draws))):
                raise ValueError(f"{study}: draws {draws}")
            tables = [nd.parameter_arrays(group[group.draw == d], self.n_lesions[i]) for d in draws]
            self.comp_table[i] = torch.from_numpy(np.stack([t[0] for t in tables])).float().to(self.device)
            self.lesion_table[i] = torch.from_numpy(np.stack([t[1] for t in tables])).float().to(self.device)
        self.n_draws = [t.shape[0] for t in self.comp_table]

    def split_indices(self, split):
        return np.flatnonzero(self.phantoms.net_split.to_numpy() == split)

    def inputs(self, i):
        """Network input of a whole phantom, (1, 2, X, Y, Z) float32: PET and CT as stored."""
        return torch.stack([self.pet[i], self.ct[i]]).float()[None]

    def targets(self, i, draw, compartment=None, lesion=None, weight=None):
        """Voxel targets of one draw by net_data.rebuild, on the whole phantom or on the given crops (long, long, float)."""
        if compartment is None:
            compartment, lesion, weight = self.compartment[i].long(), self.lesion[i].long(), self.weight[i].float()
        return self.nd.rebuild(compartment, lesion, weight, self.weight_sum_comp[i], self.weight_sum_lesion[i],
                               self.comp_table[i][draw], self.lesion_table[i][draw], xp=torch)

    def memory_gb(self):
        tensors = self.pet + self.ct + self.weight + self.compartment + self.lesion
        tensors += [t for group in self.foreground for t in group]
        return sum(t.numel() * t.element_size() for t in tensors) / 1024**3


def _window(centre, shape, patch, rng):
    """Source start and destination offset of a patch around centre; an axis shorter than the patch is placed at random inside it."""
    start, offset = [], []
    for c, s, p in zip(centre, shape, patch):
        if s >= p:
            start.append(min(max(c - p // 2, 0), s - p))
            offset.append(0)
        else:
            start.append(0)
            offset.append(int(rng.integers(0, p - s + 1)))
    return start, offset


def _crop(volume, start, offset, patch, fill):
    out = volume.new_full(patch, fill)
    size = [min(p, s) for p, s in zip(patch, volume.shape)]
    src = tuple(slice(a, a + n) for a, n in zip(start, size))
    dst = tuple(slice(o, o + n) for o, n in zip(offset, size))
    out[dst] = volume[src]
    return out


def _centre(cohort, i, patch, cfg, rng):
    shape = cohort.shape[i]
    groups = cohort.foreground[i]
    if groups and rng.random() < cfg.p_foreground:
        group = groups[int(rng.integers(len(groups)))]
        flat = int(group[int(rng.integers(group.numel()))])
        centre = list(np.unravel_index(flat, shape))
    else:
        centre = [int(rng.integers(s)) for s in shape]
    return [int(c + rng.integers(-int(cfg.jitter * p), int(cfg.jitter * p) + 1)) for c, p in zip(centre, patch)]


def sample_batch(cohort, indices, batch, patch, cfg, rng, generator=None):
    """
    Random training batch built on the device: phantom, draw, patch centre (foreground-weighted), flips, intensity
    augmentation and the voxel targets of the draw. rng (NumPy) drives the discrete choices, generator (torch) the noise.
    Returns x (B, 2, *patch), compartment and lesion (B, *patch, long) and the stacked target fields.
    """
    patch = tuple(int(p) for p in patch)
    aug = cfg.augment
    pets, cts, comps, lesions, targets = [], [], [], [], []
    for _ in range(batch):
        i = int(indices[int(rng.integers(len(indices)))])
        draw = int(rng.integers(cohort.n_draws[i]))
        start, offset = _window(_centre(cohort, i, patch, cfg, rng), cohort.shape[i], patch, rng)
        fields = [_crop(cohort.pet[i], start, offset, patch, 0.0), _crop(cohort.ct[i], start, offset, patch, AIR),
                  _crop(cohort.compartment[i], start, offset, patch, 0), _crop(cohort.lesion[i], start, offset, patch, 0),
                  _crop(cohort.weight[i], start, offset, patch, 0.0)]
        axes = [a for a in (aug.flip_axes if aug is not None else ()) if rng.random() < 0.5]
        if axes:
            fields = [torch.flip(f, axes) for f in fields]
        pet, ct, comp, lesion, weight = fields
        comp, lesion = comp.long(), lesion.long()
        pets.append(pet)
        cts.append(ct)
        comps.append(comp)
        lesions.append(lesion)
        targets.append(cohort.targets(i, draw, comp, lesion, weight.float()))
    pet, ct = torch.stack(pets).float(), torch.stack(cts).float()
    if aug is not None:
        shape = (batch, 1, 1, 1)
        device = pet.device
        if aug.pet_scale:
            scale = torch.exp((torch.rand(shape, device=device, generator=generator) * 2 - 1) * np.log1p(aug.pet_scale))
            pet = torch.log1p(scale * torch.expm1(pet))
        pet = (pet + aug.pet_noise * torch.randn(pet.shape, device=device, generator=generator)).clamp_min(0.0)
        ct = ct + (torch.rand(shape, device=device, generator=generator) * 2 - 1) * aug.ct_shift_hu / cohort.nd.CT_SCALE
    return dict(x=torch.stack([pet, ct], dim=1), compartment=torch.stack(comps), lesion=torch.stack(lesions),
                targets={k: torch.stack([t[k] for t in targets]) for k in targets[0]})


def _max_relative(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & (a != 0)
    relative = float(np.max(np.abs(b[ok] - a[ok]) / np.abs(a[ok]))) if ok.any() else 0.0
    scale = float(np.max(np.abs(a[np.isfinite(a)]))) if np.isfinite(a).any() else 1.0
    absolute = float(np.max(np.abs(b[np.isfinite(a)] - a[np.isfinite(a)]))) if np.isfinite(a).any() else 0.0
    return relative, absolute / scale if scale > 0 else absolute


def check_rebuild(cohort, netdata, shard_dir, studies=None, draw=0):
    """
    net_data.rebuild with torch tensors on the device (float64 and float32, the training precision) against NumPy
    float64, on two training phantoms: largest relative difference per field (voxel by voxel, and of the largest
    absolute difference over the largest value), and whether the masks and the pattern of infinite lambda_u agree.
    """
    nd = cohort.nd
    train = cohort.phantoms[cohort.phantoms.net_split == "train"]
    if studies is None:
        first = train.sort_values("n_lesions").iloc[-1]
        other = train[(train.scanner != first.scanner) & (train.n_lesions > 0)].sort_values("n_lesions")
        studies = [first.study_id, other.iloc[len(other) // 2].study_id]
    sources = pd.read_parquet(_guard(Path(netdata) / "sources.parquet"))
    params = pd.read_parquet(_guard(Path(netdata) / "params.parquet"))
    params = params[(params.arm == cohort.arm) & (params.scenario == cohort.scenario) & (params.draw == draw)
                    & params.has_source]
    rows = []
    for study in studies:
        i = cohort.index[study]
        arrays = nd.read_shard(_guard(Path(shard_dir) / train.shard[train.study_id == study].iloc[0]), {study})[study]
        comp, lesion = arrays["compartment"].astype(np.int64), arrays["lesion"].astype(np.int64)
        weight = arrays["weight"].astype(np.float64)
        comp_sum, lesion_sum = nd.source_sum_tables(sources[sources.study_id == study], cohort.n_lesions[i])
        comp_table, lesion_table = nd.parameter_arrays(params[params.study_id == study], cohort.n_lesions[i])
        reference = nd.rebuild(comp, lesion, weight, comp_sum, lesion_sum, comp_table, lesion_table, xp=np)

        device = cohort.device
        as64 = lambda a: torch.from_numpy(np.asarray(a)).to(device)
        candidates = {"torch float64": nd.rebuild(as64(comp), as64(lesion), as64(weight), as64(comp_sum),
                                                  as64(lesion_sum), as64(comp_table), as64(lesion_table), xp=torch),
                      "torch float32 (training)": cohort.targets(i, draw)}
        for name, result in candidates.items():
            for field in TARGET_FIELDS:
                ref, value = reference[field], result[field].cpu().numpy()
                row = dict(study_id=study, shape=str(cohort.shape[i]), n_lesions=cohort.n_lesions[i], path=name,
                           field=field)
                if ref.dtype == bool:
                    row.update(max_relative=0.0, max_abs_over_max=0.0, pattern_equal=bool(np.array_equal(ref, value)))
                else:
                    relative, scaled = _max_relative(ref, value)
                    row.update(max_relative=relative, max_abs_over_max=scaled,
                               pattern_equal=bool(np.array_equal(np.isfinite(ref), np.isfinite(value))))
                rows.append(row)
    return pd.DataFrame(rows)
