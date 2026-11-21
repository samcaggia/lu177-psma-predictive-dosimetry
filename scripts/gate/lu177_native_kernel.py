#!/usr/bin/env python3
"""Lu-177 voxel S-value kernel in water on an anisotropic PET voxel, as in lu177_water_validation.py vsv but extended (GATE 10).

The source fills the central voxel; dose is scored on the same voxel grid. Output: <output>/<name>_raw.npy in mGy/(MBq s), axes x, y, z.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import opengate as gate
import SimpleITK as sitk
from opengate.utility import g4_units

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lu177_water_validation import DEFAULT_HIT_TYPE, HIT_TYPE_HELP, HIT_TYPES, add_dose_actors, add_lu177, add_stats, base_simulation, merge_split_outputs  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--voxel-mm", type=float, nargs=3, required=True)
    p.add_argument("--half-extent-mm", type=float, default=200.0)
    p.add_argument("--phantom-mm", type=float, default=800.0)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261104)
    p.add_argument("--cut-mm", type=float, default=0.1)
    p.add_argument("--name", required=True)
    p.add_argument("--hit-type", choices=HIT_TYPES, default=DEFAULT_HIT_TYPE, help=HIT_TYPE_HELP)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    mm = g4_units.mm
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    work = out / f"{args.name}_work"
    work.mkdir(exist_ok=True)
    sim = base_simulation(args, work)
    water = sim.add_volume("Box", "water")
    water.size = [args.phantom_mm * mm] * 3
    water.material = "G4_WATER"
    sim.physics_manager.set_production_cut(water.name, "all", args.cut_mm * mm)
    source = add_lu177(sim, water.name, args.n, args.threads)
    source.position.type = "box"
    source.position.size = [v * mm for v in args.voxel_mm]
    size = [2 * int(np.ceil(args.half_extent_mm / v)) + 1 for v in args.voxel_mm]
    def configure(dose):
        dose.attached_to = water.name
        dose.size = size
        dose.spacing = [v * mm for v in args.voxel_mm]
        dose.dose.active = True

    add_dose_actors(sim, "vsv", args.hit_type, configure)
    stats = add_stats(sim)
    start = time.time()
    sim.run()
    elapsed = time.time() - start
    if args.hit_type == "split":
        merge_split_outputs(work, "vsv")
    events = int(stats.counts.events)
    kernel = sitk.GetArrayFromImage(sitk.ReadImage(str(work / "vsv_dose.mhd"))).T / events * 1e6 * 1e3
    np.save(out / f"{args.name}_raw.npy", kernel.astype(np.float64))
    (out / f"{args.name}.json").write_text(json.dumps(dict(voxel_mm=args.voxel_mm, half_extent_mm=args.half_extent_mm, phantom_mm=args.phantom_mm, n=events, seconds=elapsed, threads=args.threads, seed=args.seed, cut_mm=args.cut_mm, shape=size, hit_type=args.hit_type), indent=1))
    print(events, elapsed, size, flush=True)


if __name__ == "__main__":
    main()
