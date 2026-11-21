"""
Single-SPECT update, method M3 (local, CPU torch, from the saved network maps): output-space test-time optimisation
of the voxel maps of log A0 and log lambda_bio against the simulated SPECT image, with the gated prior M0.
Usage: python spect_update_maps.py [tuning|reporting|test] [draws] [times] [steps] [lr] [residual scale] [max phantoms]
       [mismatch|shape|exact] [mismatch SD] [k/n]
(k/n: only the phantoms k, k + n, ... of the list, for parallel runs; the output name ends in .k)
`test` (frozen test run): every test phantom, maps of notebook section 10 (maps_test/), test structure predictions;
steps, learning rate and residual scale must equal those stored in tto_choice_A.json["M3"] (checked, not chosen);
the frozen files must match test_freeze_A.json; refuses to run if its output exists.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from lib import frozen, spect, tto, update  # noqa: E402
from lib.calibration import build_test_cohort  # noqa: E402
from lib.config import HEADS, LossConfig  # noqa: E402
from lib.data import GpuCohort, import_net_data  # noqa: E402
from lib.scoring import with_sd  # noqa: E402
from lib.structures import EARLY_SHARE_MAX, _from_heads, phantom_structures  # noqa: E402
from lib.utils import provenance  # noqa: E402

SCENARIO = "A"
NETDATA = frozen.netdata_dir()
EVAL = frozen.eval_dir(SCENARIO)
OUT = EVAL / "spect"
RECORD = (0, 5, 10, 20, 40, 60, 100, 150)
ROW_COLUMNS = ["virtual_id", "study_id", "half", "source", "structure", "t_h", "step", "method", "quantity", "mean", "truth"]


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "tuning"
    n_draws = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    times = [float(t) for t in sys.argv[3].split(",")] if len(sys.argv) > 3 else [48.0, 96.0]
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    lr = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
    residual_scale = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
    limit = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    condition = sys.argv[8] if len(sys.argv) > 8 else "mismatch"
    sd = float(sys.argv[9]) if len(sys.argv) > 9 else update.MISMATCH_SD
    part, parts = (int(v) for v in sys.argv[10].split("/")) if len(sys.argv) > 10 else (0, 1)
    test = which == "test"
    tag = (f"{which}_{condition}{'' if sd == update.MISMATCH_SD else f'_sd{sd:g}'}_{SCENARIO}_lr{lr:g}_rs{residual_scale:g}"
           f"{'' if parts == 1 else f'.{part}'}")
    freeze_stamp = {}
    if test:
        frozen_m3 = json.loads((EVAL / f"tto_choice_{SCENARIO}.json").read_text())["M3"]
        if (steps, lr, residual_scale) != (int(frozen_m3["steps"]), float(frozen_m3["lr"]), float(frozen_m3["residual_scale"])):
            raise ValueError(f"M3 on the test split runs the frozen setting {frozen_m3}, not {(steps, lr, residual_scale)}")
        frozen.refuse_if_done([OUT / f"maps_rows_{tag}.csv"])
        freeze_stamp = {**frozen.check_manifest(EVAL, NETDATA), **frozen.check_inputs(EVAL, NETDATA)}
    OUT.mkdir(parents=True, exist_ok=True)
    nd = import_net_data(str(ROOT / "src"))
    cfg, loss_cfg = spect.SpectConfig(), LossConfig()
    choice = json.loads((EVAL / f"gated_prior_{SCENARIO}.json").read_text())
    if test:
        cohort = build_test_cohort(GpuCohort, nd, NETDATA, NETDATA / "shards", torch.device("cpu"), scenario=SCENARIO)
        studies = list(cohort.phantoms.study_id)
    else:
        cohort = GpuCohort(nd, NETDATA, NETDATA / "shards", torch.device("cpu"), scenario=SCENARIO, splits=("val",))
        half_of_study = cohort.phantoms.set_index("study_id").subject.astype(str).map(choice["subject_half"])
        studies = [s for s in cohort.phantoms.study_id if half_of_study[s] == which]
    grid = float(cohort.phantoms.grid_mm.iloc[0])
    studies = sorted(studies, key=lambda s: cohort.n_lesions[cohort.index[s]])[:limit or None][part::parts]
    stamp = {**provenance(str(HERE), str(ROOT / "src")), "steps": steps, "lr": lr, "residual_scale": residual_scale, "condition": condition,
             "mismatch_sd": 0.0 if condition == "exact" else sd, **freeze_stamp}
    if not studies:
        pd.DataFrame(columns=ROW_COLUMNS).assign(**stamp).to_csv(OUT / f"maps_rows_{tag}.csv", index=False)
        print(f"part {part}/{parts}: no phantom, empty maps_rows_{tag}.csv written", flush=True)
        return
    network = with_sd(pd.read_csv(EVAL / f"{'test_' if test else ''}structure_prediction_{SCENARIO}.csv"))
    population = pd.read_parquet(NETDATA / f"baseline_population{'_test' if test else ''}.parquet")
    m0 = update.gated_table(choice, network, population[population.arm == "main"])
    scales = pd.read_csv(EVAL / f"voxel_calibration_{SCENARIO}.csv")
    head_scale = scales[scales.structure == "all"].set_index("head").head_scale.to_dict()
    truth, activity = update.case_data(NETDATA, studies, SCENARIO, condition, sd=sd, test=test)
    rows = []
    for study in studies:
        i = cohort.index[study]
        with np.load(EVAL / ("maps_test" if test else "maps") / f"{study}.npz") as data:
            mean, logvar = data["mean"].astype(np.float32), data["logvar"].astype(np.float32)
        output = torch.from_numpy(np.stack([mean, logvar], axis=1).reshape(-1, *mean.shape[1:]))[None]
        st = phantom_structures(cohort, i, 0)
        targets = cohort.targets(i, 0)
        inside = targets["inside"]
        label, values = tto.structure_labels(cohort.compartment[i].long(), cohort.lesion[i].long(), nd.LESION_CODE)
        names = {c: s for s, c in nd.COMPARTMENT_CODES.items()}
        structure_of_value = {k: (names[int(v)] if int(v) <= nd.LESION_CODE else f"lesion_{int(v) - nd.LESION_CODE - 1}")
                              for k, v in enumerate(values.tolist()) if v > 0}
        fixed = output[0, 0::2].flatten(1)[:, st.entries].double()

        def structure_values(maps):
            heads = fixed.clone()
            heads[tto.A0], heads[tto.LAMBDA] = maps[0].flatten()[st.entries].double(), maps[1].flatten()[st.entries].double()
            out = _from_heads(heads[None], st, cohort)
            return {q: dict(zip(st.sources, v[0].tolist())) for q, v in out.items()}

        aggregate = lambda voxel, name: structure_values(
            [voxel if name == "log_a0" else output[0, 2 * tto.A0], voxel if name == "log_lambda_bio" else output[0, 2 * tto.LAMBDA]])[name]
        for draw in range(n_draws):
            vid = f"{study}_d{draw:02d}"
            prior = m0[m0.virtual_id == vid].set_index("source")
            widen = lambda q: np.where(prior[f"{q}_n"].notna(), np.sqrt(1 + 1 / prior[f"{q}_n"].fillna(1)), 1.0)
            m0_mean = {q: prior[f"{q}_mean"].dropna().to_dict() for q in ("log_a0", "log_lambda_bio")}
            m0_sd = {q: (prior[f"{q}_sd"] * widen(q)).dropna().to_dict() for q in ("log_a0", "log_lambda_bio")}
            share = prior.early_share_mean.clip(upper=EARLY_SHARE_MAX).to_dict()
            prior_mean, prior_sd, offset_sd = tto.prior_maps(output, label, values, structure_of_value, m0_mean, m0_sd, None,
                                                             head_scale, loss_cfg.logvar_range, aggregate)
            for t_h in times:
                image, _ = spect.simulate(spect.activity_map(cohort, i, activity[(vid, t_h)]), cfg, grid, vid, t_h)
                start = time.perf_counter()

                def record(step, maps):
                    if step not in RECORD and step != steps:
                        return
                    v = structure_values(maps)
                    for k, source in enumerate(st.sources):
                        if (vid, source) not in truth.index or source not in share:
                            continue
                        estimates = {"log_a0": v["log_a0"][source], "log_lambda_bio": v["log_lambda_bio"][source],
                                     "log_tiac": v["log_tiac_late"][source] - np.log1p(-share[source])}
                        for q, value in estimates.items():
                            rows.append(dict(virtual_id=vid, study_id=study, half=which, source=source, structure=st.structures[k],
                                             t_h=t_h, step=step, method="M3 output-space TTO", quantity=q, mean=value,
                                             truth=float(truth.loc[(vid, source), q])))

                tto.optimise_maps(prior_mean, prior_sd, image, t_h, cfg, grid, cohort.voxel_ml, inside, label, offset_sd,
                                  steps, lr, residual_scale, record)
                print(f"{vid} @{t_h:g} h: {len(st)} structures, {time.perf_counter() - start:.0f} s", flush=True)
    table = pd.DataFrame(rows, columns=ROW_COLUMNS)
    table.assign(**stamp).to_csv(OUT / f"maps_rows_{tag}.csv", index=False)
    table["error"] = (table["mean"] - table.truth).abs()
    pd.set_option("display.width", 250)
    for q in ("log_tiac", "log_a0", "log_lambda_bio"):
        print(f"\n{q}: median |error| against the step\n" + table[table.quantity == q].pivot_table(
            index="structure", columns=["t_h", "step"], values="error", aggfunc="median").round(2).to_string())


if __name__ == "__main__":
    main()
