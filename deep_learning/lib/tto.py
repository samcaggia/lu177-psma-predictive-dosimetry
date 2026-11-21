"""
Test-time optimisation against one SPECT through the kinetics layer and the SPECT forward model.
M3 (output space): the voxel maps of log A0 and log lambda_bio are the variables. M4 (weight space): a small subset
of the weights of a copy of the network is; the whole-phantom output is assembled from sliding windows and the
gradient is sent back window by window, so the memory is that of one window. With the 'head' subset (output
convolution and last decoder block) the features that enter that block do not change, so they are computed once
per window and every step runs the last block only.
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
from monai.data.utils import compute_importance_map, dense_patch_slices
from monai.inferers.utils import _get_scan_interval

from . import kinetics, spect
from .config import HEADS
from .data import AIR
from .utils import autocast_bf16

A0, LAMBDA = HEADS.index("log_a0"), HEADS.index("log_lambda_bio")


def spect_nll(log_a0, log_lambda_bio, measured, t_h, cfg, grid_mm, voxel_ml, log_calibration, inside):
    """
    Gaussian NLL of a measured SPECT given voxel maps of log A0 per mL and log lambda_bio: activity at t_h by the
    kinetics layer (late exponential), forward model, voxel noise variance at the expected image (not differentiated),
    plus the log-normal prior of the calibration factor. Sum over the voxels, divided by the number of voxels over
    which the noise is correlated (the voxels are not independent measurements).
    """
    activity = kinetics.activity(torch.exp(log_a0) * voxel_ml, torch.exp(log_lambda_bio), t_h)
    activity = torch.where(inside, activity, torch.zeros_like(activity))
    expected = spect.forward(activity, cfg, grid_mm, torch.exp(log_calibration))
    variance = cfg.noise_sd(expected.detach()) ** 2 * spect.correlation_volume(cfg, grid_mm)
    return 0.5 * ((measured - expected) ** 2 / variance).sum() + 0.5 * (log_calibration / cfg.calibration_sd) ** 2


def structure_labels(compartment, lesion, lesion_code):
    """Dense index of the structure of every voxel (0 outside): compartments, then each lesion on its own labels."""
    label = torch.where(lesion > 0, lesion_code + lesion, compartment)
    values, dense = torch.unique(label, return_inverse=True)
    return dense, values


def optimise_maps(mean, sd, measured, t_h, cfg, grid_mm, voxel_ml, inside, label, offset_sd, steps=60, lr=0.05,
                  residual_scale=1.0, record=None):
    """
    M3. mean, sd: (2, X, Y, Z) prior voxel maps of log A0 per mL and log lambda_bio (the M0 prior painted on the
    voxels: see prior_maps) and their voxel SD. The maps are mean + offset[structure] + residual: one offset per
    structure and map, with prior SD offset_sd (2, structures) (the M0 SD of the structure: errors shared by its
    voxels), and a voxel residual with prior SD residual_scale x sd. Adam for `steps` steps on the SPECT NLL plus the
    two priors and the calibration prior. record(step, maps) is called at every step. Returns the final maps (2, X, Y, Z).
    """
    offset = torch.zeros_like(offset_sd, requires_grad=True)
    residual = torch.zeros_like(mean, requires_grad=True)
    log_calibration = torch.zeros((), dtype=mean.dtype, requires_grad=True)
    optimiser = torch.optim.Adam([offset, residual, log_calibration], lr=lr)
    current = lambda: mean + offset[:, label] + residual
    for step in range(steps + 1):
        maps = current()
        if record is not None:
            record(step, maps.detach())
        if step == steps:
            break
        optimiser.zero_grad()
        loss = spect_nll(maps[0], maps[1], measured, t_h, cfg, grid_mm, voxel_ml, log_calibration, inside)
        loss = loss + 0.5 * ((offset / offset_sd) ** 2).sum() \
            + 0.5 * (((residual / (residual_scale * sd)) ** 2) * inside).sum()
        loss.backward()
        optimiser.step()
    return current().detach()


def prior_maps(output, label, values, structure_of_value, m0_mean, m0_sd, network_chosen, head_scale, logvar_range,
               aggregate):
    """
    Voxel prior of M3 from the network output (1, outputs, X, Y, Z) and the gated prior M0. For each of log A0 and
    log lambda_bio: the network mean map, shifted in every structure by a constant so that its structure value
    (aggregate(map, head) -> (structures,)) equals the M0 mean; voxel SD = network SD x the voxel scale of the head;
    offset SD = M0 SD of the structure. Returns (mean (2, X, Y, Z), sd (2, X, Y, Z), offset_sd (2, structures)).
    """
    means, sds, offsets = [], [], []
    for head in (A0, LAMBDA):
        name = HEADS[head]
        voxel = output[0, 2 * head].float()
        shift = torch.zeros(values.numel(), dtype=voxel.dtype, device=voxel.device)
        spread = torch.ones(values.numel(), dtype=voxel.dtype, device=voxel.device)
        current = aggregate(voxel, name)
        for k, structure in structure_of_value.items():
            if structure in m0_mean[name]:
                shift[k] = m0_mean[name][structure] - float(current[structure])
                spread[k] = m0_sd[name][structure]
        means.append(voxel + shift[label])
        sds.append(torch.exp(0.5 * output[0, 2 * head + 1].float().clamp(*logvar_range)) * head_scale[name])
        offsets.append(spread)
    return torch.stack(means), torch.stack(sds), torch.stack(offsets)


def trainable_subset(model, subset):
    """
    Freezes the model and returns the parameters of the subset: 'norm' = every normalisation layer with weights
    (the LayerNorm layers of the Swin encoder), 'head' = the output convolution and the last decoder block.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    if subset == "norm":
        chosen = [p for m in model.modules() if isinstance(m, (torch.nn.LayerNorm, torch.nn.GroupNorm,
                                                                torch.nn.InstanceNorm3d, torch.nn.BatchNorm3d))
                  for p in m.parameters()]
    elif subset == "head":
        chosen = list(model.backbone.out.parameters()) + list(model.backbone.decoder1.parameters())
    else:
        raise ValueError(subset)
    for p in chosen:
        p.requires_grad_(True)
    return chosen


def head_inputs(backbone, x):
    """The two tensors that enter the last decoder block of a MONAI SwinUNETR (its forward up to decoder1)."""
    hidden = backbone.swinViT(x, backbone.normalize)
    enc0 = backbone.encoder1(x)
    enc1 = backbone.encoder2(hidden[0])
    enc2 = backbone.encoder3(hidden[1])
    enc3 = backbone.encoder4(hidden[2])
    dec4 = backbone.encoder10(hidden[4])
    dec3 = backbone.decoder5(dec4, hidden[3])
    dec2 = backbone.decoder4(dec3, enc3)
    dec1 = backbone.decoder3(dec2, enc2)
    return backbone.decoder2(dec1, enc1), enc0


def head_output(backbone, dec0, enc0):
    return backbone.out(backbone.decoder1(dec0, enc0))


class WindowedModel:
    """
    Whole-phantom prediction of a model as the Gaussian-weighted mean of sliding windows (as predict_phantom), with a
    backward pass that sends a gradient on the assembled output back to the weights one window at a time.
    cache_head: the inputs of the last decoder block are computed once per window (valid while only that block and
    the output convolution are trained).
    """

    def __init__(self, model, x, patch, overlap=0.25, bf16=True, cache_head=False):
        self.model, self.bf16 = model, bf16
        self.shape = tuple(x.shape[2:])
        pad = [max(0, p - s) for p, s in zip(patch, self.shape)]
        if any(pad):
            widths = [w for p in reversed(pad) for w in (0, p)]
            x = torch.stack([F.pad(x[:, 0], widths, value=0.0), F.pad(x[:, 1], widths, value=AIR)], dim=1)
        self.x = x
        size = tuple(x.shape[2:])
        interval = _get_scan_interval(size, patch, 3, (overlap,) * 3)
        self.slices = dense_patch_slices(size, patch, interval)
        self.weight = compute_importance_map(patch, mode="gaussian", sigma_scale=0.125, device=x.device, dtype=torch.float32)
        self.weight = self.weight.clamp_min(float(self.weight[self.weight > 0].min()))
        total = torch.zeros(size, device=x.device)
        for s in self.slices:
            total[s] += self.weight
        self.total = total
        self.cache = None
        if cache_head:
            with torch.no_grad():
                self.cache = [self._run(lambda s=s: head_inputs(self.model.backbone, self.x[(slice(None), slice(None), *s)]))
                              for s in self.slices]

    def _run(self, function):
        if self.bf16:
            with autocast_bf16(self.x.device):
                return function()
        return function()

    def _window(self, k):
        if self.cache is not None:
            return self._run(lambda: head_output(self.model.backbone, *self.cache[k])).float()
        s = self.slices[k]
        return self._run(lambda: self.model(self.x[(slice(None), slice(None), *s)])).float()

    @torch.no_grad()
    def predict(self):
        out = None
        for k, s in enumerate(self.slices):
            window = self._window(k)
            if out is None:
                out = torch.zeros(1, window.shape[1], *self.total.shape, device=window.device)
            out[(slice(None), slice(None), *s)] += window * self.weight
        out = out / self.total
        return out[..., :self.shape[0], :self.shape[1], :self.shape[2]]

    def backward(self, gradient):
        """Accumulates in the trainable weights the gradient of a loss whose gradient on predict() is `gradient`."""
        full = torch.zeros(1, gradient.shape[1], *self.total.shape, device=gradient.device)
        full[..., :self.shape[0], :self.shape[1], :self.shape[2]] = gradient
        full = full / self.total
        for k, s in enumerate(self.slices):
            (self._window(k) * (full[(slice(None), slice(None), *s)] * self.weight)).sum().backward()


def optimise_weights(model, x, measured, t_h, cfg, grid_mm, voxel_ml, inside, patch, subset="head", steps=30, lr=1e-3,
                     anchor_weight=1.0, l2_weight=1e-3, head_scale=None, logvar_range=(-10.0, 6.0), overlap=0.25,
                     bf16=True, record=None, shift=None, likelihood="voxel", voi=None, label=None):
    """
    M4. A copy of the model is adapted to one SPECT. Its maps of log A0 per mL and log lambda_bio, plus the fixed
    `shift` (2, X, Y, Z) if given (the per-structure constants that bring the network to the gated prior M0, so that
    step 0 is M0), go through the kinetics layer and the forward model. likelihood 'voxel': Gaussian NLL of the SPECT
    image, divided by the noise correlation volume, with the calibration factor as a parameter. likelihood 'voi': the
    VOI likelihood of M2 (voi = dict(st, measured, n_voxels, gain, discrepancy, sd): expected VOI sums of the forward
    image, SD from sd(expected)), which does not see where inside a VOI the activity sits. Anchor: Gaussian NLL of
    the maps under their initial values and voxel SDs (x head_scale), divided by the correlation volume ('voxel') or
    averaged within each structure of `label` and summed over the structures ('voi'), times anchor_weight; plus
    l2_weight x the squared distance of the trainable weights from their start, relative to their squared norm.
    Adam, `steps` steps. record(step, maps) gets the two maps at every step. Returns the final maps (2, X, Y, Z).
    """
    model = copy.deepcopy(model).eval()
    parameters = trainable_subset(model, subset)
    initial = [p.detach().clone() for p in parameters]
    scale = sum(float((p ** 2).sum()) for p in initial) or 1.0
    log_calibration = torch.zeros((), device=x.device, requires_grad=likelihood == "voxel")
    groups = [{"params": parameters, "lr": lr}] + ([{"params": [log_calibration], "lr": 0.01}] if likelihood == "voxel" else [])
    optimiser = torch.optim.Adam(groups)
    windows = WindowedModel(model, x, patch, overlap, bf16, cache_head=subset == "head")
    start = windows.predict()
    index = [2 * A0, 2 * LAMBDA]
    shift = torch.zeros_like(start[0, index]) if shift is None else shift.to(start.dtype)
    anchor_mean = start[0, index] + shift
    head_scale = head_scale or {h: 1.0 for h in HEADS}
    anchor_sd = torch.stack([torch.exp(0.5 * start[0, 2 * h + 1].clamp(*logvar_range)) * head_scale[HEADS[h]]
                             for h in (A0, LAMBDA)])
    volume = spect.correlation_volume(cfg, grid_mm)
    if likelihood == "voi":
        n_label = int(label.max()) + 1
        count = torch.bincount(label.flatten(), minlength=n_label).clamp_min(1).to(start.dtype)
    output = start
    for step in range(steps + 1):
        if record is not None:
            record(step, output[0, index] + shift)
        if step == steps:
            break
        optimiser.zero_grad()
        leaf = output.detach().clone().requires_grad_(True)
        maps = leaf[0, index] + shift
        penalty = (((maps - anchor_mean) / anchor_sd) ** 2) * inside
        if likelihood == "voxel":
            loss = spect_nll(maps[0], maps[1], measured, t_h, cfg, grid_mm, voxel_ml, log_calibration, inside)
            loss = loss + anchor_weight * 0.5 * penalty.sum() / volume
        else:
            activity = kinetics.activity(torch.exp(maps[0]) * voxel_ml, torch.exp(maps[1]), t_h)
            image = spect.forward(torch.where(inside, activity, torch.zeros_like(activity)), cfg, grid_mm)
            expected = spect.voi_sums(image, voi["st"]).double()
            sd = voi["sd"](expected)
            loss = (0.5 * ((voi["measured"] - expected) / sd) ** 2 + torch.log(sd)).sum().to(maps.dtype)
            per_structure = torch.zeros(2, n_label, dtype=maps.dtype, device=maps.device).index_add_(
                1, label.flatten(), penalty.flatten(1)) / count
            loss = loss + anchor_weight * 0.5 * per_structure[:, 1:].sum()
        loss.backward()
        windows.backward(leaf.grad)
        (l2_weight * sum(((p - p0) ** 2).sum() for p, p0 in zip(parameters, initial)) / scale).backward()
        optimiser.step()
        output = windows.predict()
    return output[0, index] + shift


def m0_shift(output, cohort, i, st, prior_rows, label, values):
    """
    Per-voxel constants (2, X, Y, Z) that move the network maps of log A0 and log lambda_bio to the gated prior M0:
    in every structure (own labels, st built with r_eval = 0) the M0 mean minus the structure value of the network
    under the default aggregation. prior_rows: the M0 table rows of one virtual patient. Also returns the M0 early
    share per source (capped), used for the TIAC.
    """
    from .structures import EARLY_SHARE_MAX, _from_heads
    nd = cohort.nd
    prior = prior_rows.set_index("source")
    heads = output[0, 0::2].flatten(1)[:, st.entries].double()
    current = {q: dict(zip(st.sources, v[0].tolist())) for q, v in _from_heads(heads[None], st, cohort).items()}
    names = {c: s for s, c in nd.COMPARTMENT_CODES.items()}
    shifts = []
    for q in ("log_a0", "log_lambda_bio"):
        table = torch.zeros(values.numel(), dtype=torch.float32, device=output.device)
        for k, v in enumerate(values.tolist()):
            source = names.get(v) if 0 < v <= nd.LESION_CODE else (f"lesion_{v - nd.LESION_CODE - 1}" if v > nd.LESION_CODE else None)
            if source in current[q] and source in prior.index and np.isfinite(prior.loc[source, f"{q}_mean"]):
                table[k] = float(prior.loc[source, f"{q}_mean"]) - current[q][source]
        shifts.append(table[label])
    return torch.stack(shifts), prior.early_share_mean.clip(upper=EARLY_SHARE_MAX).to_dict()


def weight_update_rows(model, cohort, cases, activity, truth, m0, cfg, patch, variants, steps, record_steps, discrepancy,
                       head_scale=None, logvar_range=(-10.0, 6.0), bf16=True, log=print):
    """
    Runs M4 on a list of cases [(phantom index, virtual_id, t_h)] for every variant (dicts of optimise_weights
    arguments: subset, lr, anchor_weight, likelihood, ...). Every run starts from the gated prior M0 (m0_shift).
    Returns one row per case, variant, recorded step, structure and quantity. The estimate of log_a0, log_lambda_bio
    and log_tiac is the M0 mean plus the change of the structure value of the maps since step 0 (default aggregation
    on the structure's own labels; for log_tiac the change of the log late TIA), so step 0 is M0 exactly, also where
    M0 is the population and its TIAC does not come from A0 and lambda_bio; `map_value` keeps the structure value of
    the maps themselves (for log_tiac: log late TIA - log(1 - M0 early share)). Plus the rows of M0 for the same cases,
    and `seconds`, the wall time of the run. activity: {(virtual_id, t_h): {source: activity}}; truth: indexed
    by (virtual_id, source); m0: update.gated_table; discrepancy: {structure: relative SD} for the VOI likelihood.
    """
    import time

    from . import update
    from .structures import _from_heads, phantom_structures
    grid = float(cohort.phantoms.grid_mm.iloc[0])
    nd = cohort.nd
    rows = []
    for i, vid, t_h in cases:
        st = phantom_structures(cohort, i, 0)
        st_voi = phantom_structures(cohort, i, 1)
        inside = cohort.targets(i, 0)["inside"]
        label, values = structure_labels(cohort.compartment[i].long(), cohort.lesion[i].long(), nd.LESION_CODE)
        image, _ = spect.simulate(spect.activity_map(cohort, i, activity[(vid, t_h)]), cfg, grid, vid, t_h)
        x = cohort.inputs(i)
        with torch.no_grad():
            first = WindowedModel(model, x, patch, bf16=bf16).predict()
        prior_rows = m0[m0.virtual_id == vid]
        shift, share = m0_shift(first, cohort, i, st, prior_rows, label, values)
        fixed = first[0, 0::2].flatten(1)[:, st.entries].double()
        gain = spect.voi_noise_gain(cohort, i, st_voi, cfg)
        n_voxels = torch.tensor(st_voi.n_voxels, dtype=torch.float64, device=cohort.device)
        relative = torch.tensor([discrepancy[s] for s in st_voi.structures], dtype=torch.float64, device=cohort.device)
        voi = dict(st=st_voi, measured=spect.voi_sums(image, st_voi).double(),
                   sd=lambda expected: update.voi_sd(expected, n_voxels, gain, cfg, relative))
        key = dict(virtual_id=vid, t_h=t_h)
        prior = prior_rows.set_index("source")
        for k, source in enumerate(st.sources):
            if (vid, source) in truth.index and source in prior.index:
                for q in ("log_a0", "log_lambda_bio", "log_tiac"):
                    rows.append({**key, "source": source, "structure": st.structures[k], "step": 0, "method": "M0 gated prior",
                                 "quantity": q, "mean": float(prior.loc[source, f"{q}_mean"]), "truth": float(truth.loc[(vid, source), q])})
        for variant in variants:
            name = "M4 " + ", ".join(f"{k} {v}" for k, v in variant.items())
            start = time.perf_counter()

            baseline, first_row = {}, len(rows)

            def record(step, maps):
                if step not in record_steps and step != 0:
                    return
                heads = fixed.clone()
                heads[A0], heads[LAMBDA] = maps[0].flatten()[st.entries].double(), maps[1].flatten()[st.entries].double()
                values_ = {q: v[0].tolist() for q, v in _from_heads(heads[None], st, cohort).items()}
                for k, source in enumerate(st.sources):
                    if (vid, source) not in truth.index or source not in share or source not in prior.index:
                        continue
                    estimates = {"log_a0": values_["log_a0"][k], "log_lambda_bio": values_["log_lambda_bio"][k],
                                 "log_tiac": values_["log_tiac_late"][k] - float(np.log1p(-share[source]))}
                    if step == 0:
                        baseline[source] = estimates
                    if step not in record_steps:
                        continue
                    for q, value in estimates.items():
                        rows.append({**key, "source": source, "structure": st.structures[k], "step": step, "method": name,
                                     "quantity": q, "mean": float(prior.loc[source, f"{q}_mean"]) + value - baseline[source][q],
                                     "map_value": value, "truth": float(truth.loc[(vid, source), q])})

            optimise_weights(model, x, image, t_h, cfg, grid, cohort.voxel_ml, inside, patch, steps=steps, head_scale=head_scale,
                             logvar_range=logvar_range, bf16=bf16, record=record, shift=shift, voi=voi, label=label, **variant)
            seconds = time.perf_counter() - start
            for row in rows[first_row:]:
                row["seconds"] = seconds
            log(f"{vid} @{t_h:g} h, {name}: {seconds:.0f} s")
    return rows
