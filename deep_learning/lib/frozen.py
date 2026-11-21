"""
Frozen test run (no torch): paths that a dry run can redirect, the md5 manifest of every frozen choice file and of the
test inputs, and the guards that refuse a second run. Used by test_run.py, the local SPECT scripts and the notebook.
"""
import glob
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
WORKSPACE = HERE.parent
SCENARIO = "A"
FROZEN_CHOICES = ("calibration_A.csv", "voxel_calibration_A.csv", "gated_prior_A.json", "tto_choice_A.json")
FROZEN_NETDATA = ("population_prior.parquet",)
TEST_TABLES = ("phantoms_test.parquet", "sources_test.parquet", "params_test.parquet", "baseline_population_test.parquet",
               "baseline_pbpk_fit_test.parquet", "spect_activity_test.parquet")
MANIFEST = "test_freeze_A.json"
INPUTS = "test_inputs_A.json"


def netdata_dir():
    return Path(os.environ.get("LU177_NETDATA") or WORKSPACE / "data" / "processed" / "psma" / "netdata")


def runs_dir():
    return Path(os.environ.get("LU177_RUNS") or HERE / "runs")


def cohort_dir():
    return Path(os.environ.get("LU177_COHORT") or WORKSPACE / "data" / "processed" / "psma" / "cohort")


def eval_dir(scenario=SCENARIO):
    return runs_dir() / f"eval_{scenario}"


def md5(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            h.update(block)
    return h.hexdigest()


def code_stamp():
    """Same values as lib.utils.provenance (VERSION commit, '+modified' if lib/ no longer matches it; net_data.py md5), without torch."""
    lib = HERE / "lib"
    digest = hashlib.md5()
    for name in sorted(os.listdir(lib)):
        if name.endswith(".py"):
            digest.update((lib / name).read_bytes())
    commit = "unknown"
    if (HERE / "VERSION").is_file():
        commit, stored = (HERE / "VERSION").read_text().split()
        if stored != digest.hexdigest():
            commit += "+modified"
    return dict(code_commit=commit, net_data_md5=md5(WORKSPACE / "src" / "net_data.py"))


def frozen_files(eval_path, netdata):
    """{name: path} of the files whose content fixes every choice applied to the test split."""
    out = {name: Path(eval_path) / name for name in FROZEN_CHOICES}
    out.update({name: Path(netdata) / name for name in FROZEN_NETDATA})
    out["net_data.py"] = WORKSPACE / "src" / "net_data.py"
    return out


def write_manifest(eval_path, netdata, extra):
    """Writes eval_path/test_freeze_A.json once: md5 of every frozen file plus `extra` (commit, lib digest). Refuses to overwrite."""
    path = Path(eval_path) / MANIFEST
    if path.exists():
        raise FileExistsError(f"{path} exists: the freeze is written once")
    files = frozen_files(eval_path, netdata)
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Frozen files missing: {missing}")
    manifest = dict(files={name: md5(p) for name, p in files.items()}, **extra)
    path.write_text(json.dumps(manifest, indent=1))
    return manifest


def check_manifest(eval_path, netdata, src_dir=None):
    """
    md5 of the frozen files now against test_freeze_A.json; raises on any difference. src_dir: folder of net_data.py
    (Drive on Colab). Returns a flat stamp {freeze_<name>_md5: md5} for the output tables.
    """
    manifest = json.loads((Path(eval_path) / MANIFEST).read_text())
    files = frozen_files(eval_path, netdata)
    if src_dir is not None:
        files["net_data.py"] = Path(src_dir) / "net_data.py"
    changed = {name: (manifest["files"][name], md5(p) if p.exists() else "missing")
               for name, p in files.items() if not p.exists() or md5(p) != manifest["files"][name]}
    if changed:
        raise RuntimeError(f"Frozen files differ from {MANIFEST} (expected, found): {changed}")
    return {f"freeze_{name.split('.')[0]}_md5": value for name, value in manifest["files"].items()}


def test_input_files(netdata):
    """{name: path} of the test tables and shards. Reads only the shard column of phantoms_test.parquet."""
    import pandas as pd
    netdata = Path(netdata)
    out = {name: netdata / name for name in TEST_TABLES}
    shards = sorted(pd.read_parquet(netdata / "phantoms_test.parquet").shard.unique())
    out.update({f"shards/{s}": netdata / "shards" / s for s in shards})
    return out


def write_inputs(eval_path, netdata):
    path = Path(eval_path) / INPUTS
    if path.exists():
        raise FileExistsError(f"{path} exists")
    files = test_input_files(netdata)
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Test inputs missing: {missing}")
    table = {name: md5(p) for name, p in files.items()}
    path.write_text(json.dumps(table, indent=1))
    return table


def check_inputs(eval_path, netdata, shard_dir=None):
    """md5 of the test inputs against test_inputs_A.json (shards in shard_dir if given); raises on any difference."""
    expected = json.loads((Path(eval_path) / INPUTS).read_text())
    netdata = Path(netdata)
    bad = {}
    for name, value in expected.items():
        p = (Path(shard_dir) / name.split("/", 1)[1]) if (shard_dir and name.startswith("shards/")) else netdata / name
        found = md5(p) if p.exists() else "missing"
        if found != value:
            bad[name] = (value, found)
    if bad:
        raise RuntimeError(f"Test inputs differ from {INPUTS} (expected, found): {bad}")
    return {"test_inputs_md5": hashlib.md5(json.dumps(expected, sort_keys=True).encode()).hexdigest()}


def existing_test_outputs(eval_path, pattern="test_*"):
    """Test outputs already written under eval_path matching `pattern` (the manifests excluded)."""
    found = [p for p in glob.glob(str(Path(eval_path) / pattern)) if Path(p).name not in (MANIFEST, INPUTS)]
    return sorted(found)


def refuse_if_done(paths):
    done = [str(p) for p in paths if Path(p).exists()]
    if done:
        raise RuntimeError(f"Already run, outputs exist (the test run is done once): {done[:5]}")
