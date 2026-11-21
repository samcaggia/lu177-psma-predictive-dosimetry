"""Training of one scenario: GPU sampling, bf16 steps with gradient accumulation, sliding-window validation, early stopping, checkpoints with resume."""
import os
import time

import numpy as np
import pandas as pd
import torch

from .data import sample_batch
from .evaluate import validate
from .losses import kinetic_loss, voxel_targets
from .model import set_checkpointing
from .utils import (BackgroundSaver, GpuMonitor, autocast_bf16, get_warmup_cosine_scheduler, load_checkpoint,
                    load_state_dict_checked, save_curves, save_table)


def build_optimizer(model, cfg):
    """AdamW with the pretrained transformer at a lower learning rate than the rest, and the schedule per optimiser step."""
    transformer = list(model.backbone.swinViT.parameters())
    transformer_ids = {id(p) for p in transformer}
    rest = [p for p in model.parameters() if id(p) not in transformer_ids]
    opt = torch.optim.AdamW([{"params": transformer, "lr": cfg.lr * cfg.transformer_lr_ratio},
                             {"params": rest, "lr": cfg.lr}], weight_decay=cfg.weight_decay)
    sched = get_warmup_cosine_scheduler(opt, cfg.warmup_steps, cfg.total_steps, cfg.min_lr_ratio)
    return opt, sched


def make_rng(seed):
    """NumPy generator driving every random choice of the sampling; its state is all a resume needs."""
    return np.random.default_rng(seed)


def micro_step(model, cohort, indices, batch, patch, sampling_cfg, loss_cfg, rng, scale=1.0, bf16=True, timed=False):
    """
    Forward and backward of one freshly sampled micro-batch; the gradients of scale x loss are added to those already
    stored. The noise of the augmentation is seeded from rng. Returns (loss, parts, seconds spent on sampling and targets).
    """
    device = cohort.device
    t0 = time.perf_counter()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(rng.integers(2**62)))
    data = sample_batch(cohort, indices, batch, patch, sampling_cfg, rng, generator)
    y, mask = voxel_targets(data["targets"], data["compartment"], cohort)
    if timed and device.type == "cuda":
        torch.cuda.synchronize()
    data_seconds = time.perf_counter() - t0
    if bf16:
        with autocast_bf16(device):
            output = model(data["x"])
    else:
        output = model(data["x"])
    loss, parts = kinetic_loss(output, y, mask, data["compartment"], data["lesion"], cohort, loss_cfg)
    (loss * scale).backward()
    return loss.detach(), parts, data_seconds


def optimizer_step(model, opt, grad_clip):
    """Clips the accumulated gradients and updates; the update is skipped when their norm is not finite. Returns the norm."""
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip or float("inf"))
    if torch.isfinite(norm):
        opt.step()
    return norm


def train_step(model, opt, cohort, indices, batch, patch, sampling_cfg, loss_cfg, rng, grad_clip=100.0, timed=False):
    """One optimisation step on one micro-batch (no accumulation), as used by the benchmark; parts['grad_norm'] is added."""
    opt.zero_grad(set_to_none=True)
    loss, parts, data_seconds = micro_step(model, cohort, indices, batch, patch, sampling_cfg, loss_cfg, rng, timed=timed)
    parts["grad_norm"] = optimizer_step(model, opt, grad_clip)
    return loss, parts, data_seconds


def _model_state(obj):
    return obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj


def train(model, cohort, cfg, sampling_cfg, loss_cfg, run_dir, stamp=None, stop_after_micro=None, max_hours=None):
    """
    Trains cfg.scenario until cfg.total_steps optimiser steps or early stopping on the validation NLL. In run_dir:
    best_<scenario>.pth (weights, step, validation NLL), resume_<scenario>.pth (every cfg.checkpoint_minutes, at each
    validation and at the end; the file it replaces is kept as .prev and used if the last one is unreadable),
    train_log_<scenario>.csv (one row every cfg.log_every steps), val_log_<scenario>.csv (one row per validation) and
    curves_<scenario>.png, all three updated at every log line and validation. The logs are part of the checkpoint and
    the files are rewritten from them, so after a resume they hold no duplicated or orphan rows. The checkpoint holds model, optimiser, scheduler, optimiser step, position inside the
    accumulation cycle with the gradients accumulated so far, sampling RNG, running sums, best NLL and early-stopping
    counter, so a resumed run continues the same sequence. stamp (code commit, md5 of net_data.py) is written into
    every file. stop_after_micro (resume test) writes a checkpoint after that many micro-batches and returns.
    max_hours: budget of the run over all its sessions (training and validations, counted in state['hours'] and kept
    in the checkpoint); when it is reached a checkpoint is written and the function returns, the run stays resumable.
    Returns the best validation NLL.
    """
    os.makedirs(run_dir, exist_ok=True)
    name, stamp = cfg.scenario, dict(stamp or {})
    best_path, resume_path = os.path.join(run_dir, f"best_{name}.pth"), os.path.join(run_dir, f"resume_{name}.pth")
    train_log_path = os.path.join(run_dir, f"train_log_{name}.csv")
    val_log_path = os.path.join(run_dir, f"val_log_{name}.csv")
    curves_path = os.path.join(run_dir, f"curves_{name}.png")
    device = cohort.device
    if cohort.scenario != cfg.scenario:
        cohort.load_scenario(cfg.scenario)
    set_checkpointing(model, cfg.checkpointing)
    train_idx, val_idx = cohort.split_indices("train"), cohort.split_indices("val")
    params = [p for p in model.parameters() if p.requires_grad]
    opt, sched = build_optimizer(model, cfg)
    rng = make_rng(cfg.seed)
    state = dict(step=0, micro=0, best=float("inf"), bad=0, running=0.0, n_running=0, recent=0.0, n_recent=0, cycle=0.0,
                 skipped=0, train_log=[], log=[])

    def write_logs():
        if state["train_log"]:
            save_table(pd.DataFrame(state["train_log"]), train_log_path, stamp)
        if state["log"]:
            save_table(pd.DataFrame(state["log"]), val_log_path, stamp)
        save_curves(state["train_log"], state["log"], curves_path, f"Scenario {name}", stamp)

    chk, source = load_checkpoint(resume_path, device)
    if chk is not None:
        load_state_dict_checked(model, chk["model_state"], source)
        opt.load_state_dict(chk["opt_state"])
        sched.load_state_dict(chk["sched_state"])
        rng.bit_generator.state = chk["rng_state"]
        state = chk["state"]
        if chk["grads"] is not None:
            for p, g in zip(params, chk["grads"]):
                p.grad = None if g is None else g.to(device)
        if chk.get("provenance") != stamp:
            print(f"[WARNING] {source} was written by {chk.get('provenance')}, now running {stamp}")
        print(f"[RESUME] {name} from {os.path.basename(source)}: optimiser step {state['step']}, micro-batch "
              f"{state['micro']}/{cfg.accum_steps}, best validation NLL {state['best']:.4f}, "
              f"no improvement {state['bad']}/{cfg.patience}")
    elif cfg.init_weights:
        load_state_dict_checked(model, _model_state(torch.load(cfg.init_weights, map_location=device,
                                                               weights_only=False)), cfg.init_weights)
        print(f"[{name}] starting from {cfg.init_weights}")

    def checkpoint():
        grads = [p.grad for p in params] if state["micro"] else None
        saver.save({"model_state": model.state_dict(), "opt_state": opt.state_dict(), "sched_state": sched.state_dict(),
                    "rng_state": rng.bit_generator.state, "state": state, "grads": grads, "provenance": stamp,
                    "config": vars(cfg)}, resume_path, keep_previous=True)

    saver = BackgroundSaver()
    last_save = last_log = clock = time.perf_counter()
    done_micro = 0
    state.setdefault("hours", 0.0)
    if max_hours is not None and state["hours"] >= max_hours:
        print(f"[{name}] budget of {max_hours:.2f} h already used ({state['hours']:.2f} h): nothing run")
        saver.close()
        return state["best"]
    try:
        with GpuMonitor(interval=5.0) as monitor:
            model.train()
            while state["step"] < cfg.total_steps and state["bad"] < cfg.patience:
                if state["micro"] == 0:
                    opt.zero_grad(set_to_none=True)
                _, parts, _ = micro_step(model, cohort, train_idx, cfg.batch, cfg.patch, sampling_cfg, loss_cfg, rng,
                                         scale=1.0 / cfg.accum_steps, bf16=cfg.bf16)
                state["micro"] += 1
                state["cycle"] += float(parts["nll"]) / cfg.accum_steps
                done_micro += 1

                if state["micro"] == cfg.accum_steps:
                    norm = optimizer_step(model, opt, cfg.grad_clip)
                    sched.step()
                    state["step"] += 1
                    state["micro"] = 0
                    if torch.isfinite(norm) and np.isfinite(state["cycle"]):
                        state["running"], state["n_running"] = state["running"] + state["cycle"], state["n_running"] + 1
                        state["recent"], state["n_recent"] = state["recent"] + state["cycle"], state["n_recent"] + 1
                    else:
                        state["skipped"] += 1
                    state["cycle"] = 0.0
                    step = state["step"]

                    if step % cfg.log_every == 0:
                        now = time.perf_counter()
                        rate = cfg.log_every / (now - last_log)
                        last_log = now
                        memory = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
                        used = monitor.utilisation[-12:]
                        row = dict(step=step, train_nll=state["recent"] / max(1, state["n_recent"]),
                                   grad_norm=float(norm), skipped=state["skipped"], lr=opt.param_groups[1]["lr"],
                                   steps_per_s=rate, it_per_s=rate * cfg.accum_steps, peak_gb=memory,
                                   gpu_util=float(np.mean(used)) if used else float("nan"))
                        state["train_log"].append(row)
                        state["recent"], state["n_recent"] = 0.0, 0
                        print(f"[{name}] step {step}/{cfg.total_steps} | train NLL {row['train_nll']:.4f} | "
                              f"grad norm {row['grad_norm']:.1f} | skipped {row['skipped']} | "
                              f"{rate:.2f} steps/s ({row['it_per_s']:.2f} it/s) | peak {memory:.1f} GB | "
                              f"GPU {row['gpu_util']:.0f}% | lr {row['lr']:.2e}")
                        if step % cfg.val_every and step != cfg.total_steps:
                            write_logs()

                    if step % cfg.val_every == 0 or step == cfg.total_steps:
                        val = validate(model, cohort, val_idx, cfg.patch, loss_cfg, cfg.sw_batch, cfg.sw_overlap)
                        model.train()
                        improved = val["nll"] < state["best"]
                        state["best"], state["bad"] = (val["nll"], 0) if improved else (state["best"], state["bad"] + 1)
                        state["log"].append({"step": step, "train_nll": state["running"] / max(1, state["n_running"]),
                                             "skipped": state["skipped"], **{f"val_{k}": v for k, v in val.items()}})
                        state["running"], state["n_running"] = 0.0, 0
                        write_logs()
                        print(f"[{name}] step {step} | val NLL {val['nll']:.4f} "
                              f"({', '.join(f'{k} {v:.3f}' for k, v in val.items() if k != 'nll')})"
                              + (" -> best model saved" if improved
                                 else f" | no improvement {state['bad']}/{cfg.patience}"))
                        if improved:
                            saver.save({"model_state": model.state_dict(), "step": step, "val_nll": val["nll"],
                                        "provenance": stamp}, best_path)
                        checkpoint()
                        last_save = last_log = time.perf_counter()

                now = time.perf_counter()
                state["hours"] += (now - clock) / 3600
                clock = now
                if max_hours is not None and state["hours"] >= max_hours:
                    write_logs()
                    checkpoint()
                    print(f"[{name}] HARD STOP: {state['hours']:.2f} h used, budget {max_hours:.2f} h, at optimiser step "
                          f"{state['step']}/{cfg.total_steps}; checkpoint written, the best weights so far are in "
                          f"{os.path.basename(best_path)}")
                    return state["best"]
                if time.perf_counter() - last_save > cfg.checkpoint_minutes * 60:
                    checkpoint()
                    last_save = time.perf_counter()
                if stop_after_micro is not None and done_micro >= stop_after_micro:
                    checkpoint()
                    print(f"[{name}] stopped on request after {done_micro} micro-batches (step {state['step']}, "
                          f"micro-batch {state['micro']}/{cfg.accum_steps})")
                    return state["best"]
            if state["bad"] >= cfg.patience:
                print(f"[{name}] Early stopping at optimiser step {state['step']}")
            checkpoint()
            print(f"[{name}] finished at optimiser step {state['step']}: {state['hours']:.2f} h over all sessions")
    finally:
        saver.close()
    return state["best"]


def resume_test(make_model, cohort, cfg, sampling_cfg, loss_cfg, run_dir, stamp=None, stop_after_micro=None):
    """
    Runs cfg once without interruption and once stopped inside an accumulation cycle and resumed, from the same initial
    model (make_model()), then truncates the last checkpoint of the second run to check the fallback to the previous one.
    Returns a table: largest weight difference between the two final models, largest difference of the validation logs, whether
    the corrupted checkpoint fell back to .prev. The two runs agree to rounding, not bit by bit. The checkpoints of the
    test are deleted at the end, the logs are kept.
    """
    stop_after_micro = stop_after_micro or cfg.accum_steps + max(1, cfg.accum_steps // 2)
    name = cfg.scenario
    finals, logs = [], []
    for label, stops in (("reference", [None]), ("resumed", [stop_after_micro, None])):
        folder = os.path.join(run_dir, f"resume_test_{label}")
        for leftover in (os.listdir(folder) if os.path.isdir(folder) else []):
            os.remove(os.path.join(folder, leftover))
        model = make_model()
        for stop in stops:
            train(model, cohort, cfg, sampling_cfg, loss_cfg, folder, stamp, stop_after_micro=stop)
        finals.append({k: v.detach().float().cpu() for k, v in model.state_dict().items()})
        logs.append(pd.read_csv(os.path.join(folder, f"val_log_{name}.csv")).select_dtypes("number"))
        del model
    weight_difference = max(float((finals[0][k] - finals[1][k]).abs().max()) for k in finals[0])
    weight_scale = max(float(v.abs().max()) for v in finals[0].values())
    log_difference = float((logs[0] - logs[1]).abs().max().max()) if logs[0].shape == logs[1].shape else float("nan")

    resume_path = os.path.join(run_dir, "resume_test_resumed", f"resume_{name}.pth")
    with open(resume_path, "r+b") as handle:
        handle.truncate(os.path.getsize(resume_path) // 2)
    chk, source = load_checkpoint(resume_path, "cpu")
    fallback = chk is not None and source.endswith(".prev")
    for label in ("reference", "resumed"):
        folder = os.path.join(run_dir, f"resume_test_{label}")
        for leftover in [f for f in os.listdir(folder) if ".pth" in f]:
            os.remove(os.path.join(folder, leftover))
    return pd.DataFrame([dict(optimiser_steps=cfg.total_steps, accum_steps=cfg.accum_steps,
                              stopped_after_micro=stop_after_micro, max_weight_difference=weight_difference,
                              max_weight=weight_scale, max_log_difference=log_difference,
                              fallback_to_previous=fallback,
                              fallback_step=chk["state"]["step"] if fallback else -1)])
