"""
Structure-level kinetic quantities from voxel maps: organs on their compartment, lesions on their voxels dilated by
r_eval (the lesion rule of the netdata README), for the rebuilt targets and for the network output, with the
predictive SD by sampling the voxel Gaussians.

Aggregation (chosen on the tuning half of the validation subjects, scenario A): A0, late, early TIA and TIA are sums
over the voxels; lambda_bio is the median of the voxel values in a compartment and the A0-weighted median on a
dilated lesion mask (a sum-weighted mean follows the few voxels that take the kinetics of a hot neighbour); in the
rest of body the predicted voxel quantities next to another source, or within HALO_MM of kidneys, glands and bladder
(the shells that the source definition sets to its median), are replaced by their median over the other voxels.
"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi

from .config import HEADS
from .scoring import QUANTITIES, flattened_voxels

EARLY_SHARE_MAX = 0.95


@dataclass
class PhantomStructures:
    """Structures of one phantom: entries are flat voxel indices, segment the structure of each entry."""
    sources: List[str]
    structures: List[str]
    n_voxels: List[int]
    other_lesion: List[bool]
    bounds: List[Tuple[int, int]]
    flatten: torch.Tensor
    reference: torch.Tensor
    entries: torch.Tensor
    segment: torch.Tensor
    is_tia: torch.Tensor
    is_curve: torch.Tensor

    def __len__(self):
        return len(self.sources)


def phantom_structures(cohort, i, r_eval=1):
    """
    Compartments on their own voxels; every lesion on its voxels dilated by r_eval voxels with the ball of
    net_data._ball (the mask of net_data.lesion_dilation), kept inside the body. A voxel can belong to several masks;
    other_lesion marks the lesions whose mask holds voxels of another lesion.
    """
    nd = cohort.nd
    compartment, lesion = cohort.compartment[i].long(), cohort.lesion[i].long()
    inside = cohort.targets(i, 0)["inside"]
    shape = tuple(compartment.shape)
    flat_code, flat_inside = compartment.flatten(), inside.flatten()
    sources, structures, groups, other_lesion = [], [], [], []
    for name, code in nd.COMPARTMENT_CODES.items():
        if code == nd.LESION_CODE:
            continue
        index = torch.nonzero((flat_code == code) & flat_inside).flatten()
        if index.numel():
            sources.append(name)
            structures.append(name)
            groups.append(index)
            other_lesion.append(False)
    lesion_np, inside_np = lesion.cpu().numpy(), inside.cpu().numpy()
    ball = nd._ball(r_eval)
    for j, box in enumerate(ndi.find_objects(lesion_np), start=1):
        if box is None:
            continue
        window = tuple(slice(max(b.start - r_eval, 0), min(b.stop + r_eval, n)) for b, n in zip(box, shape))
        own = lesion_np[window] == j
        region = (ndi.binary_dilation(own, ball) if r_eval else own) & inside_np[window]
        if not region.any():
            continue
        local = np.nonzero(region)
        index = np.ravel_multi_index(tuple(axis + w.start for axis, w in zip(local, window)), shape)
        sources.append(f"lesion_{j - 1}")
        structures.append("lesion")
        groups.append(torch.from_numpy(index).to(cohort.device))
        other_lesion.append(bool((region & (lesion_np[window] > 0) & ~own).any()))
    entries = torch.cat(groups)
    segment = torch.cat([torch.full((g.numel(),), k, dtype=torch.long, device=cohort.device)
                         for k, g in enumerate(groups)])
    code = flat_code[entries]
    stops = np.cumsum([g.numel() for g in groups])
    bounds = [(int(b - g.numel()), int(b)) for b, g in zip(stops, groups)]
    flatten = torch.zeros(entries.numel(), dtype=torch.bool, device=cohort.device)
    reference = torch.zeros(0, dtype=torch.long, device=cohort.device)
    if "rest_of_body" in sources:
        a, b = bounds[sources.index("rest_of_body")]
        marked = torch.from_numpy(flattened_voxels(nd, compartment.cpu().numpy(), lesion_np,
                                                   float(cohort.phantoms.grid_mm.iloc[0])).ravel()).to(cohort.device)
        flatten[a:b] = marked[entries[a:b]]
        reference = a + torch.nonzero(~flatten[a:b]).flatten()
    return PhantomStructures(sources, structures, [int(g.numel()) for g in groups], other_lesion, bounds, flatten,
                             reference, entries, segment, cohort.is_tia[code], cohort.is_curve[code])


def _weighted_median(values, weights):
    """Per row of (k, n): the smallest value whose cumulative weight reaches half of the total."""
    order = values.argsort(dim=1)
    ranked, cumulative = values.gather(1, order), weights.gather(1, order).cumsum(dim=1)
    index = (cumulative >= 0.5 * cumulative[:, -1:]).to(torch.int8).argmax(dim=1, keepdim=True)
    return ranked.gather(1, index)[:, 0]


def _lambda_location(log_lambda, a0, st, floor):
    """log lambda_bio of every structure from (k, entries) voxel values: median in a compartment, A0-weighted median on a lesion mask."""
    out = log_lambda.new_empty(log_lambda.shape[0], len(st))
    for s, (a, b) in enumerate(st.bounds):
        if st.structures[s] == "lesion":
            out[:, s] = _weighted_median(log_lambda[:, a:b], a0[:, a:b])
        else:
            out[:, s] = log_lambda[:, a:b].median(dim=1).values
    return out.clamp_min(float(np.log(floor)))


def _summarise(a0, lambda_bio, lambda_u, lu_ok, late, early, tia, segment, n, nd):
    """
    Sums over the entries of each structure (last axis): A0, late and early TIA and TIA summed, lambda_u as the
    A0-weighted mean over the entries where it is defined; log_lambda_bio here is the A0-weighted mean, replaced by
    _lambda_location in the callers. Logs except the early share.
    """
    def total(x):
        return x.new_zeros(*x.shape[:-1], n).index_add_(x.dim() - 1, segment, x)

    a0_total, a0_lu = total(a0), total(torch.where(lu_ok, a0, torch.zeros_like(a0)))
    lu_sum = total(torch.where(lu_ok, a0 * lambda_u, torch.zeros_like(a0)))
    nan = torch.full_like(a0_total, float("nan"))
    tia_total = total(tia)
    return {"log_a0": torch.log(a0_total),
            "log_lambda_bio": torch.log((total(a0 * lambda_bio) / a0_total).clamp_min(nd.LAMBDA_BIO_FLOOR)),
            "log_lambda_u": torch.where(a0_lu > 0, torch.log(lu_sum / a0_lu.clamp_min(1e-300)), nan),
            "log_tiac": torch.log(tia_total),
            "log_tiac_late": torch.log(total(late)),
            "early_share": total(early) / tia_total}


def truth_summary(targets, st, nd):
    """Structure quantities of the rebuilt targets of one draw, on the masks of st; {quantity: (structures,) array}."""
    def field(name):
        return targets[name].flatten()[st.entries].double()

    lambda_u = field("lambda_u")
    lu_ok = st.is_curve & torch.isfinite(lambda_u)
    a0, lambda_bio = field("a0"), field("lambda_bio").clamp_min(nd.LAMBDA_BIO_FLOOR)
    out = _summarise(a0, lambda_bio, torch.where(lu_ok, lambda_u, torch.ones_like(lambda_u)), lu_ok,
                     field("tia_late"), field("tia_early"), field("tia"), st.segment, len(st), nd)
    out["log_lambda_bio"] = _lambda_location(torch.log(lambda_bio)[None], a0[None], st, nd.LAMBDA_BIO_FLOOR)[0]
    return {q: v.cpu().numpy() for q, v in out.items()}


def voxel_fields(heads, is_tia, voxel_ml, nd):
    """
    Voxel quantities from head values (..., heads, voxels): A0 from the per-mL head, late TIA = A0 / lambda_w, TIA from
    the TIAC head on tia structures and late / (1 - early share) elsewhere, early TIA = share x TIA on tia structures
    and TIA - late elsewhere. The early share is capped at EARLY_SHARE_MAX.
    """
    value = {head: heads[..., k, :] for k, head in enumerate(HEADS)}
    a0 = torch.exp(value["log_a0"]) * voxel_ml
    lambda_bio = torch.exp(value["log_lambda_bio"])
    share = value["early_share"].clamp(max=EARLY_SHARE_MAX)
    late = a0 / (lambda_bio + nd.LAMBDA_PHYS_H)
    tia = torch.where(is_tia, torch.exp(value["log_tiac"]) * voxel_ml, late / (1.0 - share))
    early = torch.where(is_tia, share * tia, tia - late)
    return dict(a0=a0, lambda_bio=lambda_bio, lambda_u=torch.exp(value["log_lambda_u"]), late=late, early=early, tia=tia)


def _from_heads(heads, st, cohort):
    f = voxel_fields(heads, st.is_tia, cohort.voxel_ml, cohort.nd)
    if st.reference.numel():
        for key in ("a0", "late", "early", "tia"):
            f[key] = torch.where(st.flatten, f[key][:, st.reference].median(dim=1, keepdim=True).values, f[key])
    lu_ok = st.is_curve.expand(f["a0"].shape)
    out = _summarise(f["a0"], f["lambda_bio"], f["lambda_u"], lu_ok, f["late"], f["early"], f["tia"], st.segment,
                     len(st), cohort.nd)
    out["log_lambda_bio"] = _lambda_location(heads[:, HEADS.index("log_lambda_bio")], f["a0"], st,
                                             cohort.nd.LAMBDA_BIO_FLOOR)
    return out


def prediction_summary(output, cohort, i, st, logvar_range, k_samples=64, seed=0, chunk_elements=8e6):
    """
    Structure quantities of a network output (1, outputs, X, Y, Z). Mean: from the voxel means. SD: over k_samples
    samples of the voxel Gaussians, with one standard normal per (structure, head, sample) shared by the voxels of
    the structure ('sd', fully correlated errors) or one per voxel ('sd_independent', comparison only); heads independent.
    Returns {quantity: {'mean' | 'sd' | 'sd_independent': (structures,) array}} and, per structure, the share of its
    voxels whose mean early share is above EARLY_SHARE_MAX (where the cap acts on the mean prediction).
    """
    flat = output[0].flatten(1)[:, st.entries].double()
    mean, sd = flat[0::2], torch.exp(0.5 * flat[1::2].clamp(*logvar_range))
    result = {q: {"mean": v[0].cpu().numpy()} for q, v in _from_heads(mean[None], st, cohort).items()}
    n, m = len(st), st.entries.numel()
    share_head = mean[HEADS.index("early_share")]
    capped = share_head.new_zeros(n).index_add_(0, st.segment, (share_head > EARLY_SHARE_MAX).double())
    capped = (capped / torch.bincount(st.segment, minlength=n).clamp_min(1)).cpu().numpy()
    shared = torch.from_numpy(np.random.default_rng([seed, int(i)]).standard_normal((k_samples, len(HEADS), n)))
    shared = shared.to(cohort.device)
    generator = torch.Generator(device=cohort.device).manual_seed(int(seed) + int(i))
    chunk = max(1, int(chunk_elements // max(1, m)))
    samples = {"sd": {q: [] for q in QUANTITIES}, "sd_independent": {q: [] for q in QUANTITIES}}
    for start in range(0, k_samples, chunk):
        k = min(chunk, k_samples - start)
        noise = {"sd": shared[start:start + k][:, :, st.segment],
                 "sd_independent": torch.randn((k, len(HEADS), m), generator=generator, device=cohort.device).double()}
        for name, eps in noise.items():
            for q, v in _from_heads(mean[None] + eps * sd[None], st, cohort).items():
                samples[name][q].append(v)
    for name, by_quantity in samples.items():
        for q, parts in by_quantity.items():
            result[q][name] = torch.cat(parts).std(dim=0).cpu().numpy()
    return result, capped


def mean_tia_map(output, cohort, i):
    """Voxel TIA (h, fraction of the injected activity) of the network mean on the whole phantom, 0 outside the body."""
    lesion = cohort.lesion[i].long()
    code = torch.where(lesion > 0, torch.full_like(lesion, cohort.nd.LESION_CODE), cohort.compartment[i].long())
    f = voxel_fields(output[0, 0::2].flatten(1).double(), cohort.is_tia[code].flatten(), cohort.voxel_ml, cohort.nd)
    inside = cohort.targets(i, 0)["inside"]
    return torch.where(inside, f["tia"].view(inside.shape), torch.zeros((), dtype=torch.float64, device=inside.device))


def separated_lesion_tiac(tia_map, cohort, i, st, r_eval=1):
    """
    net_data.separate_lesion on a voxel TIA map for every lesion of st (TIA of the lesion alone: dilated sum minus the
    local background of the ring); returns {source: log separated TIA, NaN if not positive}. Window as in
    net_data.lesion_separation.
    """
    nd = cohort.nd
    lesion = cohort.lesion[i].long().cpu().numpy()
    code = cohort.compartment[i].long().cpu().numpy()
    values = tia_map.cpu().numpy()
    pad = r_eval + nd.SHELL_VOXELS + 1
    boxes = ndi.find_objects(lesion)
    out = {}
    for source in st.sources:
        if not source.startswith("lesion_"):
            continue
        j = int(source.split("_")[1]) + 1
        window = tuple(slice(max(b.start - pad, 0), min(b.stop + pad, n)) for b, n in zip(boxes[j - 1], lesion.shape))
        separated, _ = nd.separate_lesion(values, code, lesion, j, r_eval, window)
        out[source] = float(np.log(separated)) if separated > 0 else float("nan")
    return out


def structure_rows(st, values, **keys):
    """One row per structure with the given quantity columns ({column: (structures,) array}) and constant keys."""
    table = pd.DataFrame({"source": st.sources, "structure": st.structures, "n_voxels": st.n_voxels,
                          "mask_has_other_lesion": st.other_lesion, **values})
    return table.assign(**keys)
