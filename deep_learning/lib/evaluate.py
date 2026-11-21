"""
Whole-phantom inference with MONAI sliding windows, the validation NLL over the draws of each phantom, and the
population prior as a voxel predictor in the layout of the network output.
"""
import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference

from .config import HEADS, N_OUTPUTS, PRIOR_COLUMNS
from .data import AIR
from .losses import kinetic_loss, voxel_targets
from .utils import autocast_bf16


@torch.no_grad()
def predict_phantom(model, cohort, i, patch, sw_batch=2, overlap=0.25):
    """Network output on a whole phantom, (1, outputs, X, Y, Z) float32; axes shorter than the patch are padded with air."""
    x = cohort.inputs(i)
    shape = x.shape[2:]
    pad = [max(0, p - s) for p, s in zip(patch, shape)]
    if any(pad):
        widths = [w for p in reversed(pad) for w in (0, p)]
        x = torch.stack([F.pad(x[:, 0], widths, value=0.0), F.pad(x[:, 1], widths, value=AIR)], dim=1)

    def predictor(window):
        with autocast_bf16(window.device):
            return model(window).float()

    out = sliding_window_inference(x, tuple(patch), sw_batch, predictor, overlap=overlap, mode="gaussian")
    return out[..., :shape[0], :shape[1], :shape[2]]


@torch.no_grad()
def validate(model, cohort, indices, patch, loss_cfg, sw_batch=2, overlap=0.25):
    """Mean over the phantoms and their draws of the loss parts on the whole phantom; 'nll' drives the early stopping."""
    model.eval()
    sums, n = {}, 0
    for i in indices:
        i = int(i)
        output = predict_phantom(model, cohort, i, patch, sw_batch, overlap)
        compartment, lesion = cohort.compartment[i].long()[None], cohort.lesion[i].long()[None]
        for draw in range(cohort.n_draws[i]):
            targets = {k: v[None] for k, v in cohort.targets(i, draw).items()}
            y, mask = voxel_targets(targets, compartment, cohort)
            _, parts = kinetic_loss(output, y, mask, compartment, lesion, cohort, loss_cfg)
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + float(value)
            n += 1
        del output
    return {key: sums.get(key, 0.0) / max(1, n) for key in ("nll", *HEADS, "consistency")}


def prior_lookup(prior, cohort, arm="main"):
    """
    Output of a predictor that knows only the structure of a voxel: (codes, outputs) table with the population mean and
    log(SD^2) of every head for each compartment code and for lesions. Heads without a prior in a structure (masked
    there) get mean 0 and log-variance 0.
    """
    nd = cohort.nd
    prior = prior[prior.arm == arm].set_index("structure")
    table = torch.zeros(nd.LESION_CODE + 1, N_OUTPUTS, dtype=torch.float32)
    for structure, code in nd.COMPARTMENT_CODES.items():
        if structure not in prior.index:
            continue
        for k, head in enumerate(HEADS):
            mean, sd = (float(prior.loc[structure, f"{PRIOR_COLUMNS[head]}_{m}"]) for m in ("mean", "sd"))
            if np.isfinite(mean) and np.isfinite(sd) and sd > 0:
                table[code, 2 * k], table[code, 2 * k + 1] = mean, 2.0 * np.log(sd)
    return table.to(cohort.device)


def prior_output(cohort, i, lookup):
    """The prior predictor on a whole phantom, in the layout of the network output: (1, outputs, X, Y, Z) float32."""
    lesion = cohort.lesion[i].long()
    code = torch.where(lesion > 0, torch.full_like(lesion, cohort.nd.LESION_CODE), cohort.compartment[i].long())
    return lookup[code].permute(3, 0, 1, 2)[None].contiguous()


def load_best(model, path, device):
    """Loads the best weights of a run into the model; returns the optimiser step and the validation NLL stored with them."""
    chk = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(chk["model_state"])
    model.eval()
    return int(chk["step"]), float(chk["val_nll"]), chk.get("provenance")
