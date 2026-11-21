"""Voxel targets of the heads, masked heteroscedastic Gaussian NLL balanced by structure, and the consistency term."""
import torch

from .config import HEADS


def voxel_targets(targets, compartment, cohort):
    """
    Target and mask of every head, (B, heads, X, Y, Z), from the rebuilt fields: log A0 per mL, log lambda_bio (floored),
    early TIA share, log lambda_u on curve structures where finite, log TIAC per mL on tia structures.
    """
    nd = cohort.nd
    inside, a0, tia, lambda_u = targets["inside"], targets["a0"], targets["tia"], targets["lambda_u"]
    positive = inside & (tia > 0)
    masks = [inside & (a0 > 0), inside, positive,
             inside & cohort.is_curve[compartment] & torch.isfinite(lambda_u) & (lambda_u > 0),
             positive & cohort.is_tia[compartment]]
    one = torch.ones_like(a0)
    values = [torch.log(torch.where(masks[0], a0, one) / cohort.voxel_ml),
              torch.log(targets["lambda_bio"].clamp_min(nd.LAMBDA_BIO_FLOOR)),
              targets["tia_early"] / torch.where(positive, tia, one),
              torch.log(torch.where(masks[3], lambda_u, one)),
              torch.log(torch.where(masks[4], tia, one) / cohort.voxel_ml)]
    y = torch.stack([torch.where(m, v, torch.zeros_like(v)) for m, v in zip(masks, values)], dim=1)
    return y.float(), torch.stack(masks, dim=1)


def instance_ids(compartment, lesion, lesion_code):
    """
    Structure instance of every voxel, unique over the batch: one per compartment and one per lesion of each sample.
    Returns (ids, stride): ids % stride is the compartment code, or lesion_code + lesion id for lesions.
    """
    local = torch.where(lesion > 0, lesion_code + lesion, compartment)
    stride = int(local.max()) + 1
    sample = torch.arange(local.shape[0], device=local.device).view(-1, *([1] * (local.dim() - 1)))
    return local + sample * stride, stride


def _structure_weights(ids, stride, lesion_code, n_total):
    """Per-instance weight 1 / (voxels of the instance x instances of its class x classes present); lesions are one class."""
    count = torch.bincount(ids, minlength=n_total).double()
    present = count > 0
    classes = (torch.arange(n_total, device=ids.device) % stride).clamp_max(lesion_code)
    per_class = torch.bincount(classes[present], minlength=lesion_code + 1).double()
    weight = 1.0 / (count.clamp_min(1) * per_class[classes].clamp_min(1) * (per_class > 0).sum().clamp_min(1))
    return torch.where(present, weight, torch.zeros_like(weight)), count


def kinetic_loss(output, y, mask, compartment, lesion, cohort, cfg):
    """
    Loss of a batch (or of a whole phantom): for each head, Gaussian NLL of the masked voxels with weight
    alpha x (balanced by structure) + (1 - alpha) x (uniform over the voxels), plus cfg.consistency x the variance of the
    predicted log lambda_bio within each structure. Returns (loss, parts); parts['nll'] is the sum over the heads.
    """
    output = output.float()
    mean, logvar = output[:, 0::2], output[:, 1::2].clamp(*cfg.logvar_range)
    nll = 0.5 * (logvar + (y - mean) ** 2 * torch.exp(-logvar))
    lesion_code = cohort.nd.LESION_CODE
    ids, stride = instance_ids(compartment, lesion, lesion_code)
    n_total = stride * compartment.shape[0]
    parts, total = {}, output.new_zeros(())
    for k, head in enumerate(HEADS):
        m = mask[:, k]
        n = int(m.sum())
        if n == 0:
            continue
        selected = ids[m]
        weight, _ = _structure_weights(selected, stride, lesion_code, n_total)
        voxel_weight = cfg.alpha * weight[selected] + (1.0 - cfg.alpha) / n
        value = (voxel_weight * nll[:, k][m].double()).sum().float()
        parts[head] = value.detach()
        total = total + cfg.head_weights[head] * value
    parts["nll"] = total.detach()

    inside = mask[:, 1]
    selected = ids[inside]
    weight, count = _structure_weights(selected, stride, lesion_code, n_total)
    prediction = mean[:, 1][inside].double()
    zeros = prediction.new_zeros(n_total)
    first = zeros.index_add(0, selected, prediction) / count.clamp_min(1)
    second = zeros.index_add(0, selected, prediction ** 2) / count.clamp_min(1)
    variance = (second - first ** 2).clamp_min(0.0)
    consistency = (variance * weight * count).sum().float()
    parts["consistency"] = consistency.detach()
    return total + cfg.consistency * consistency, parts
