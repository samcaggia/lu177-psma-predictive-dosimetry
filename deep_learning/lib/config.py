"""Configuration dataclasses and the layout of the network outputs."""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

HEADS = ("log_a0", "log_lambda_bio", "early_share", "log_lambda_u", "log_tiac")
N_OUTPUTS = 2 * len(HEADS)
PRIOR_COLUMNS = {"log_a0": "log_a0_per_ml", "log_lambda_bio": "log_lambda_bio", "early_share": "early_share",
                 "log_lambda_u": "log_lambda_u", "log_tiac": "log_tiac_per_ml"}


@dataclass
class AugmentConfig:
    """
    GPU augmentation: flips along flip_axes, PET noise (on the stored log1p encoding), CT shift. pet_scale stays 0:
    scaling the PET without scaling the A0 target would teach that the PET amplitude matters less than it does.
    """
    flip_axes: Tuple[int, ...] = (0,)
    pet_scale: float = 0.0
    pet_noise: float = 0.02
    ct_shift_hu: float = 20.0


@dataclass
class SamplingConfig:
    """Patch sampling: share of patches centred on an organ or lesion voxel, jitter of the centre as a share of the patch."""
    p_foreground: float = 0.7
    jitter: float = 0.25
    augment: Optional[AugmentConfig] = field(default_factory=AugmentConfig)


@dataclass
class LossConfig:
    """
    alpha: share of the loss balanced by structure (the rest by voxel); consistency: weight of the variance of the
    predicted log lambda_bio within each structure; logvar_range clamps the predicted log-variances.
    """
    alpha: float = 0.5
    consistency: float = 0.1
    logvar_range: Tuple[float, float] = (-10.0, 6.0)
    head_weights: Dict[str, float] = field(default_factory=lambda: {h: 1.0 for h in HEADS})


@dataclass
class TrainConfig:
    """
    One training run of a scenario. An optimiser step accumulates the gradients of accum_steps micro-batches of `batch`
    patches; warmup_steps, total_steps, val_every and log_every count optimiser steps. grad_clip only cuts spikes (the
    median gradient norm of the benchmark is 25-47). init_weights starts from another run (fine-tuning of the B scenarios).
    """
    scenario: str = "A"
    patch: Tuple[int, int, int] = (128, 96, 160)
    batch: int = 1
    accum_steps: int = 3
    checkpointing: bool = False
    bf16: bool = True
    lr: float = 1e-4
    transformer_lr_ratio: float = 0.2
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    total_steps: int = 21300
    min_lr_ratio: float = 0.01
    grad_clip: float = 100.0
    val_every: int = 500
    patience: int = 10
    checkpoint_minutes: float = 15.0
    log_every: int = 50
    sw_batch: int = 2
    sw_overlap: float = 0.25
    init_weights: Optional[str] = None
    seed: int = 20261113
