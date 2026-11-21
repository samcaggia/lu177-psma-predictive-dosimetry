"""
Differentiable kinetics of the voxel parameters, in torch, with the equations that build the targets
(net_data.rebuild and the uptake-washout curve of the cohort): A(t) = A0 (exp(-lambda_w t) - exp(-lambda_u t)),
lambda_w = lambda_bio + lambda_phys of 177Lu, late TIA = A0 / lambda_w, TIA = late + early with the early share
= early / TIA. Times in hours, rates in 1/h, activities as fractions of the injected activity.
"""
import math

import torch

from .config import HEADS

LAMBDA_PHYS_H = math.log(2) / (6.647 * 24)


def lambda_w(lambda_bio):
    return lambda_bio + LAMBDA_PHYS_H


def activity(a0, lambda_bio, t, lambda_u=None):
    """
    Activity at time t (broadcast against the parameters). lambda_u None, or infinite in a voxel, means no uptake
    phase there: the late exponential alone.
    """
    late = a0 * torch.exp(-lambda_w(lambda_bio) * t)
    if lambda_u is None:
        return late
    finite = torch.isfinite(lambda_u)
    uptake = torch.exp(-torch.where(finite, lambda_u, torch.zeros_like(lambda_u)) * t)
    return late - a0 * torch.where(finite, uptake, torch.zeros_like(uptake))


def tia_late(a0, lambda_bio):
    """Integral of the late exponential from 0 to infinity."""
    return a0 / lambda_w(lambda_bio)


def tia(a0, lambda_bio, early_share):
    """Time-integrated activity from the late part and the early share (early / TIA, signed)."""
    return tia_late(a0, lambda_bio) / (1.0 - early_share)


def tia_curve(a0, lambda_bio, lambda_u):
    """Integral of the uptake-washout curve: A0 (1 / lambda_w - 1 / lambda_u), the second term 0 where lambda_u is infinite."""
    finite = torch.isfinite(lambda_u)
    return a0 * (1.0 / lambda_w(lambda_bio) - torch.where(finite, 1.0 / torch.where(finite, lambda_u, torch.ones_like(lambda_u)),
                                                           torch.zeros_like(lambda_u)))


def early_share_of_curve(lambda_bio, lambda_u):
    """Early share implied by an uptake phase: -lambda_w / (lambda_u - lambda_w), 0 where lambda_u is infinite."""
    finite = torch.isfinite(lambda_u)
    safe = torch.where(finite, lambda_u, torch.ones_like(lambda_u))
    return torch.where(finite, -lambda_w(lambda_bio) / (safe - lambda_w(lambda_bio)), torch.zeros_like(lambda_u))


def parameters_from_heads(mean, voxel_ml, head_axis=0):
    """
    Voxel parameters from the head values of the network (config.HEADS along head_axis): A0 per voxel (fraction of the
    injected activity), lambda_bio, lambda_u (1/h), early share, and the TIA per voxel of the TIAC head (h).
    """
    value = {head: mean.select(head_axis, k) for k, head in enumerate(HEADS)}
    return dict(a0=torch.exp(value["log_a0"]) * voxel_ml, lambda_bio=torch.exp(value["log_lambda_bio"]),
                lambda_u=torch.exp(value["log_lambda_u"]), early_share=value["early_share"],
                tia_head=torch.exp(value["log_tiac"]) * voxel_ml)
