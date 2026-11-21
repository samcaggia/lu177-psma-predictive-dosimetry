"""
Frozen test run of the network and of every single-SPECT method, scenario A. Run once, in this order:

  1. python test_run.py freeze             local: md5 of every frozen choice file -> runs/eval_A/test_freeze_A.json
  2. python test_run.py spect-data [n]     local: netdata/spect_activity_test.parquet (spect_data.build_test, n processes)
                                           and the md5 of every test input -> runs/eval_A/test_inputs_A.json
  3. python test_run.py checklist          what to upload, which flags to set, cost of the Colab part
  4. Colab, main.ipynb sections 10 and 12  network test evaluation (test_*, maps_test/) and M4 (spect/test_weights_rows_A.csv)
  5. python test_run.py local [parts]      local: M0, M1, M2 (gated and population prior), M3, M5, main condition and
                                           shape mismatch, four SPECT times; then the report and the absorbed doses

Steps 1, 2, 5 run with .venv/bin/python (base environment); step 5 starts the torch scripts with TORCH_PYTHON (default
.venv-torch/bin/python, CPU torch). Every choice is read from the frozen files, whose md5 every step checks
against test_freeze_A.json; no step overwrites an output (step 5 skips the steps whose output exists, so an interrupted
run is continued, and M5 resumes from its .partial.csv).

  python test_run.py dry [folder] [phantoms]   dry run of steps 1-5 on a stand-in test split: `phantoms` validation
      phantoms (default 2) relabelled as test, in `folder` (default the folder given by DRY_RUN_DIR or
      /tmp/lu177_test_dry_run). The notebook sections run on CPU with a stand-in best_A.pth (weights of
      runs/resume_A.pth.prev). No test file is opened; the real runs/ and netdata/ are checked to be unchanged.
"""
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(WORKSPACE / "src"))
from lib import frozen  # noqa: E402

TORCH_PYTHON = os.environ.get("TORCH_PYTHON") or str(WORKSPACE / ".venv-torch" / "bin" / "python")
PLAIN_PYTHON = str(WORKSPACE / ".venv" / "bin" / "python")
TIMES = "24,48,96,168"
DRAWS = 2
M5_STEPS = 30
CONDITIONS = ("mismatch", "shape")
CU = dict(setup=0.2, evaluation=0.6, m4=2.0)
REAL_NETDATA = WORKSPACE / "data" / "processed" / "psma" / "netdata"
REAL_COHORT = WORKSPACE / "data" / "processed" / "psma" / "cohort"
REAL_RUNS = HERE / "runs"


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def git_dirty():
    out = subprocess.run(["git", "-C", str(WORKSPACE), "status", "--porcelain", "--", "deep_learning", "src"],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    return [line for line in out if "deep_learning/runs/" not in line and "__pycache__" not in line]


def freeze(allow_dirty=False):
    eval_path, netdata = frozen.eval_dir(), frozen.netdata_dir()
    dirty = git_dirty()
    if dirty and not allow_dirty:
        raise RuntimeError("Uncommitted code in deep_learning/ or src/: commit, run lib.utils.write_version, commit VERSION, "
                           f"then freeze\n{dirty}")
    stamp = frozen.code_stamp()
    if stamp["code_commit"].endswith("+modified") and not allow_dirty:
        raise RuntimeError("VERSION does not match lib/: run lib.utils.write_version after the last commit")
    calibration = pd.read_csv(eval_path / "calibration_A.csv", usecols=["checkpoint_step", "checkpoint_val_nll"])
    head = subprocess.run(["git", "-C", str(WORKSPACE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest = frozen.write_manifest(eval_path, netdata, dict(
        git_head=head, code_commit=stamp["code_commit"], net_data_md5=stamp["net_data_md5"],
        checkpoint_step=int(calibration.checkpoint_step.iloc[0]), checkpoint_val_nll=float(calibration.checkpoint_val_nll.iloc[0]),
        written=time.strftime("%Y-%m-%d %H:%M:%S")))
    log(f"freeze written: {eval_path / frozen.MANIFEST}\n{json.dumps(manifest, indent=1)}")


def spect_data(processes=3):
    eval_path, netdata = frozen.eval_dir(), frozen.netdata_dir()
    frozen.check_manifest(eval_path, netdata)
    sys.path.insert(0, str(WORKSPACE / "src"))
    import spect_data as S
    log("building spect_activity_test.parquet (spect_data.build_test)")
    n_rows, failed = S.build_test(processes=processes)
    log(f"{n_rows} rows, {len(failed)} failures" + (f": {failed[:3]}" if failed else ""))
    if failed:
        raise RuntimeError("Virtual patients failed in the re-simulation: inspect before going on")
    table = frozen.write_inputs(eval_path, netdata)
    log(f"md5 of {len(table)} test inputs written to {eval_path / frozen.INPUTS}")
    checklist()


def checklist():
    eval_path = frozen.eval_dir()
    inputs = eval_path / frozen.INPUTS
    shards = [n for n in json.loads(inputs.read_text())] if inputs.exists() else ["shards/test_*.npz (listed in test_inputs_A.json after step 2)"]
    shards = [n for n in shards if n.startswith("shards/")] or shards
    text = f"""
UPLOAD to Google Drive (<DRIVE_ROOT>/, the DRIVE_ROOT of main.ipynb), overwriting older copies:
  deep_learning/main.ipynb, deep_learning/VERSION, deep_learning/lib/*.py   -> <DRIVE_ROOT>/deep_learning/
  src/net_data.py                                                            -> <DRIVE_ROOT>/src/
  deep_learning/runs/eval_A/ calibration_A.csv, voxel_calibration_A.csv, gated_prior_A.json, tto_choice_A.json,
                             {frozen.MANIFEST}, {frozen.INPUTS}           -> <DRIVE_ROOT>/deep_learning/runs/eval_A/
  deep_learning/runs/eval_A/spect/weights_rows_reporting_A.csv (cost estimate; already there if never deleted)
  data/processed/psma/netdata/ {', '.join(frozen.TEST_TABLES)}
                                                                             -> <DRIVE_ROOT>/netdata/
  data/processed/psma/netdata/{', '.join(shards)}   -> <DRIVE_ROOT>/netdata/shards/
  Must already be on Drive, unchanged: runs/best_A.pth (step and NLL checked against calibration_A.csv),
  netdata/population_prior.parquet (md5 checked).

FLAGS in section 2 of main.ipynb (every other flag stays False, SCENARIO = "A"):
  RUN_TEST_EVALUATION = True, RUN_SPECT_UPDATE_TEST = True; CU_PER_HOUR = the rate shown by Colab; TEST_CU_LIMIT = 3.0
  Runtime: L4 (any GPU works; the estimate is for L4). Then Run all.

COST (L4, estimates, not measurements): set-up {CU['setup']} CU, section 10 {CU['evaluation']} CU (validation pass of 27
  phantoms about 0.35 CU, 47 test phantoms), section 12 {CU['m4']} CU (188 runs x about 20 s measured on the reporting
  half; the cell prints its own estimate and stops above TEST_CU_LIMIT): about {sum(CU.values()):.1f} CU in all.

DOWNLOAD after the run into deep_learning/runs/eval_A/: every test_* file, the folder maps_test/ (about 2 GB),
  spect/test_weights_rows_A.csv. Then: python test_run.py local
"""
    print(text)
    return text


def run(command, env=None, out=None):
    """Runs one step; its output goes to `out` (a log file) and the last lines are printed if it fails."""
    log(" ".join(map(str, command)))
    environment = {**os.environ, **(env or {})}
    with open(out, "w") if out else open(os.devnull, "w") as handle:
        result = subprocess.run(list(map(str, command)), cwd=HERE, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = Path(out).read_text().splitlines()[-30:] if out else []
        raise RuntimeError(f"failed ({result.returncode}): {' '.join(map(str, command))}\n" + "\n".join(tail))


def parallel(commands, env, logs):
    """Starts the commands together and waits for all; raises if any failed."""
    procs = []
    for command, path in zip(commands, logs):
        log(" ".join(map(str, command)))
        handle = open(path, "w")
        procs.append((subprocess.Popen(list(map(str, command)), cwd=HERE, env={**os.environ, **env}, stdout=handle,
                                       stderr=subprocess.STDOUT), handle, command, path))
    failed = []
    for proc, handle, command, path in procs:
        proc.wait()
        handle.close()
        if proc.returncode:
            failed.append(f"{' '.join(map(str, command))}\n" + "\n".join(Path(path).read_text().splitlines()[-30:]))
    if failed:
        raise RuntimeError("\n\n".join(failed))


def local(parts=4):
    eval_path, netdata = frozen.eval_dir(), frozen.netdata_dir()
    frozen.check_manifest(eval_path, netdata)
    frozen.check_inputs(eval_path, netdata)
    needed = [eval_path / "test_structure_prediction_A.csv", eval_path / "maps_test"]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Outputs of notebook section 10 missing (download them first): {missing}")
    if not (eval_path / "spect" / "test_weights_rows_A.csv").exists():
        log("WARNING: spect/test_weights_rows_A.csv (M4, section 12) not found: the report will leave M4 out")
    out = eval_path / "spect"
    logs = out / "test_logs"
    logs.mkdir(parents=True, exist_ok=True)
    m3 = json.loads((eval_path / "tto_choice_A.json").read_text())["M3"]
    sd = f"{json.loads((eval_path / 'tto_choice_A.json').read_text())['mismatch_sd']:g}"
    threads = str(max(1, (os.cpu_count() or 4) // parts))
    env = {"OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads, "SCENARIO": "A"}
    start = time.perf_counter()
    for condition in CONDITIONS:
        for prior, suffix in (("gated", ""), ("population", "_popprior")):
            target = out / f"update_rows_test_{condition}{suffix}_A.csv"
            if target.exists():
                log(f"done: {target.name}")
                continue
            run([TORCH_PYTHON, "spect_update.py", "test", DRAWS, TIMES, condition, sd, prior], env,
                logs / f"m012_{condition}_{prior}.log")
    m3_name = f"_A_lr{float(m3['lr']):g}_rs{float(m3['residual_scale']):g}"
    for condition in CONDITIONS:
        todo = [k for k in range(parts) if not (out / f"maps_rows_test_{condition}{m3_name}.{k}.csv").exists()
                and not (out / f"maps_rows_test_{condition}{m3_name}.csv").exists()]
        if todo:
            parallel([[TORCH_PYTHON, "spect_update_maps.py", "test", DRAWS, TIMES, int(m3["steps"]), float(m3["lr"]),
                       float(m3["residual_scale"]), 0, condition, sd, f"{k}/{parts}"] for k in todo], env,
                     [logs / f"m3_{condition}.{k}.log" for k in todo])
        else:
            log(f"done: M3 {condition}")
    for condition in CONDITIONS:
        todo = [k for k in range(parts) if not (out / f"pbpk_rows_test_{condition}_A.{k}.csv").exists()
                and not (out / f"pbpk_rows_test_{condition}_A.csv").exists()]
        if todo:
            parallel([[TORCH_PYTHON, "spect_update_pbpk.py", "test", DRAWS, TIMES, M5_STEPS, condition, 0, sd, f"{k}/{parts}"]
                      for k in todo], env, [logs / f"m5_{condition}.{k}.log" for k in todo])
        else:
            log(f"done: M5 {condition}")
    run([PLAIN_PYTHON, "spect_report.py"], {"SPECT_STAGE": "test"}, logs / "report.log")
    run([PLAIN_PYTHON, "spect_dose.py"], {"SPECT_STAGE": "test"}, logs / "dose.log")
    log(f"local test run finished in {(time.perf_counter() - start) / 3600:.2f} h: {out / 'test_final_tables_A.md'}, "
        f"{eval_path / 'dose' / 'test_dose_table_A.csv'}, logs in {logs}")


def real_state():
    """Names (not contents) of the real test outputs and inputs that a dry run must not create or change."""
    paths = [REAL_NETDATA / "spect_activity_test.parquet", REAL_RUNS / "eval_A" / frozen.MANIFEST, REAL_RUNS / "eval_A" / frozen.INPUTS,
             REAL_RUNS / "eval_A" / "maps_test"]
    found = {str(p): p.stat().st_mtime for p in paths if p.exists()}
    for pattern in ("eval_A/test_*", "eval_A/spect/*test*", "eval_A/dose/test_*"):
        found.update({str(p): p.stat().st_mtime for p in REAL_RUNS.glob(pattern)})
    return found


def build_standin(folder, n_phantoms):
    """A stand-in test split made of validation phantoms, with the file names and layout of the real test split."""
    netdata, cohort, runs = folder / "netdata", folder / "cohort", folder / "runs"
    for d in (netdata / "shards", cohort, runs / "eval_A"):
        d.mkdir(parents=True, exist_ok=True)
    phantoms = pd.read_parquet(REAL_NETDATA / "phantoms.parquet")
    val = phantoms[(phantoms.net_split == "val") & phantoms.n_lesions.between(1, 6)].copy()
    val["voxels"] = val["shape"].map(lambda s: int(pd.Series(json.loads(s)).prod()))
    chosen = val.sort_values(["voxels", "study_id"]).head(n_phantoms)
    studies = set(chosen.study_id)
    renamed = {s: f"test_standin_{s}" for s in chosen.shard.unique()}
    for old, new in renamed.items():
        os.symlink(REAL_NETDATA / "shards" / old, netdata / "shards" / new)
    chosen.drop(columns="voxels").assign(net_split="test", shard=chosen.shard.map(renamed)).to_parquet(netdata / "phantoms_test.parquet")
    sources = pd.read_parquet(REAL_NETDATA / "sources.parquet")
    sources[sources.study_id.isin(studies)].to_parquet(netdata / "sources_test.parquet")
    params = pd.read_parquet(REAL_NETDATA / "params.parquet")
    params[params.study_id.isin(studies)].assign(net_split="test", split="test").to_parquet(netdata / "params_test.parquet")
    for stem in ("baseline_population", "baseline_pbpk_fit"):
        table = pd.read_parquet(REAL_NETDATA / f"{stem}.parquet")
        table[table.virtual_id.str.rsplit("_", n=1).str[0].isin(studies)].to_parquet(netdata / f"{stem}_test.parquet")
    os.symlink(REAL_NETDATA / "population_prior.parquet", netdata / "population_prior.parquet")
    for stem in ("bundles_train", "virtual_train_main"):
        items = [x for x in pickle.loads((REAL_COHORT / f"{stem}.pkl").read_bytes()) if x.get("study_id") in studies]
        (cohort / f"{stem.replace('train', 'test')}.pkl").write_bytes(pickle.dumps(items))
    for stem in ("lambda_rel", "lesion_lambda_rel", "tia"):
        table = pd.read_parquet(REAL_COHORT / f"{stem}_train.parquet")
        key = table.virtual_id.str.rsplit("_", n=1).str[0]
        table = table[key.isin(studies)]
        if "split" in table:
            table = table.assign(split="test")
        table.to_parquet(cohort / f"{stem}_test.parquet")
    for name in frozen.FROZEN_CHOICES:
        shutil.copy2(REAL_RUNS / "eval_A" / name, runs / "eval_A" / name)
    log(f"stand-in test split: {sorted(studies)} (validation phantoms), in {folder}")
    return dict(LU177_NETDATA=str(netdata), LU177_COHORT=str(cohort), LU177_RUNS=str(runs))


def standin_checkpoint(runs):
    """Stand-in best_A.pth: the model weights of runs/resume_A.pth.prev with the step and NLL of calibration_A.csv (torch)."""
    import torch
    calibration = pd.read_csv(REAL_RUNS / "eval_A" / "calibration_A.csv", usecols=["checkpoint_step", "checkpoint_val_nll"])
    state = torch.load(REAL_RUNS / "resume_A.pth.prev", map_location="cpu", weights_only=False)["model_state"]
    torch.save({"model_state": state, "step": int(calibration.checkpoint_step.iloc[0]),
                "val_nll": float(calibration.checkpoint_val_nll.iloc[0]), "provenance": {"stand_in": "resume_A.pth.prev"}},
               Path(runs) / "best_A.pth")


def notebook_cells():
    """Executes the code cells of main.ipynb in order with RUN_TEST_EVALUATION and RUN_SPECT_UPDATE_TEST set (torch)."""
    import matplotlib
    matplotlib.use("Agg")
    nb = json.loads((HERE / "main.ipynb").read_text())
    namespace = {"__name__": "__main__"}
    os.chdir(HERE)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        source = "\n".join((line[:len(line) - len(line.lstrip())] + "pass") if line.lstrip().startswith("!") else line
                           for line in source.split("\n"))
        exec(compile(source, "main.ipynb", "exec"), namespace)
        if "from lib.tto import" in source and namespace["device"].type == "cpu":
            import contextlib
            sys.modules["lib.tto"].autocast_bf16 = lambda device: contextlib.nullcontext()
            log("dry run on CPU: M4 in float32 (no bf16 hardware here; on Colab it runs in bf16 on CUDA)")
        if "RUN_TEST_EVALUATION = False" in source:
            namespace.update(RUN_TEST_EVALUATION=True, RUN_SPECT_UPDATE_TEST=True)
            flags = {k: v for k, v in namespace.items() if k.startswith("RUN_") and v}
            log(f"flags switched on for the dry run: {sorted(flags)}")


def dry(folder, n_phantoms=2):
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(f"{folder} exists: choose a new folder for the dry run")
    before = real_state()
    env = build_standin(folder, n_phantoms)
    os.environ.update(env)
    run([TORCH_PYTHON, "test_run.py", "standin-checkpoint", env["LU177_RUNS"]], env, folder / "standin_checkpoint.log")
    freeze(allow_dirty=True)
    spect_data(processes=2)
    run([TORCH_PYTHON, "test_run.py", "notebook-cells"], {**env, "OMP_NUM_THREADS": str(os.cpu_count() or 4)}, folder / "notebook.log")
    again = subprocess.run([TORCH_PYTHON, "test_run.py", "notebook-cells"], cwd=HERE, env={**os.environ, **env},
                           capture_output=True, text=True)
    refused = again.returncode != 0 and "already been run" in again.stdout + again.stderr
    log(f"second notebook run refused: {refused}")
    if not refused:
        raise RuntimeError("The notebook test sections did not refuse a second run")
    local(parts=2)
    after = real_state()
    if before != after:
        raise RuntimeError(f"The dry run changed real files: {set(before.items()) ^ set(after.items())}")
    log(f"dry run passed; real runs/ and netdata/ unchanged ({len(before)} real test paths, unchanged). Outputs in {folder}")


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "checklist"
    if command == "freeze":
        freeze()
    elif command == "spect-data":
        spect_data(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    elif command == "checklist":
        checklist()
    elif command == "local":
        local(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    elif command == "dry":
        default = Path(os.environ.get("DRY_RUN_DIR") or "/tmp/lu177_test_dry_run")
        dry(sys.argv[2] if len(sys.argv) > 2 else default, int(sys.argv[3]) if len(sys.argv) > 3 else 2)
    elif command == "standin-checkpoint":
        standin_checkpoint(sys.argv[2])
    elif command == "notebook-cells":
        notebook_cells()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
