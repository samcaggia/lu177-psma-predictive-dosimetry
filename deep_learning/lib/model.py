"""Network: MONAI SwinUNETR on (PET, CT) with one mean and one log-variance output per head."""
import numpy as np
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from monai.networks.nets.swin_unetr import filter_swinunetr
from monai.networks.utils import copy_model_state

from .config import HEADS, N_OUTPUTS, PRIOR_COLUMNS

SSL_WEIGHTS_URL = ("https://github.com/Project-MONAI/MONAI-extra-test-data/releases/"
                   "download/0.8.1/ssl_pretrained_weights.pth")


class KineticSwinUNETR(nn.Module):
    """SwinUNETR backbone; output channels 2k and 2k + 1 are the mean and the log-variance of head k (config.HEADS)."""

    def __init__(self, in_channels=2, feature_size=48, use_checkpoint=True):
        super().__init__()
        self.backbone = SwinUNETR(spatial_dims=3, in_channels=in_channels, out_channels=N_OUTPUTS,
                                  feature_size=feature_size, use_checkpoint=use_checkpoint)

    def forward(self, x):
        return self.backbone(x)


def _load_ssl_weights(model, in_channels, path=None):
    if path is None:
        weights = torch.hub.load_state_dict_from_url(SSL_WEIGHTS_URL, map_location="cpu", progress=False)["model"]
    else:
        weights = torch.load(path, map_location="cpu")["model"]
    patch_weights = {k: weights.pop(k) for k in list(weights) if "patch_embed.proj" in k}
    copy_model_state(model.backbone, weights, filter_func=filter_swinunetr)
    proj = model.backbone.swinViT.patch_embed.proj
    with torch.no_grad():
        for k, v in patch_weights.items():
            if k.endswith("weight"):
                proj.weight.copy_(v.repeat(1, in_channels, *([1] * (v.ndim - 2))) / in_channels)
            elif k.endswith("bias") and proj.bias is not None:
                proj.bias.copy_(v)


def output_layer(model):
    return [m for m in model.modules() if isinstance(m, nn.Conv3d)][-1]


def init_output_bias(model, prior, arm="main"):
    """Sets the output biases to the population prior: mean over the structures, and the log of their pooled variance."""
    prior = prior[prior.arm == arm]
    layer = output_layer(model)
    with torch.no_grad():
        for k, head in enumerate(HEADS):
            mean = prior[f"{PRIOR_COLUMNS[head]}_mean"].dropna()
            sd = prior[f"{PRIOR_COLUMNS[head]}_sd"].dropna()
            variance = float(np.mean(sd ** 2) + (np.var(mean) if len(mean) > 1 else 0.0))
            layer.bias[2 * k] = float(mean.mean())
            layer.bias[2 * k + 1] = float(np.log(variance))
    return model


def set_checkpointing(model, enabled=True):
    """Activation checkpointing of the Swin blocks (recomputed in the backward): same result, less memory, slower."""
    for module in model.modules():
        if hasattr(module, "use_checkpoint"):
            module.use_checkpoint = enabled
    return model


def build_model(device, prior=None, in_channels=2, feature_size=48, load_ssl=True, ssl_path=None, use_checkpoint=True):
    """SwinUNETR with the MONAI self-supervised weights adapted to the input channels and the prior in the output biases."""
    model = KineticSwinUNETR(in_channels=in_channels, feature_size=feature_size, use_checkpoint=use_checkpoint)
    if load_ssl:
        try:
            _load_ssl_weights(model, in_channels, ssl_path)
            print(f"MONAI SSL weights loaded and adapted to {in_channels} channels.")
        except Exception as e:
            print(f"[WARNING] SSL weights not loaded ({e}): the backbone starts from scratch.")
    if prior is not None:
        init_output_bias(model, prior)
    return model.to(device)
