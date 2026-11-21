#!/usr/bin/env python3
import argparse
import json
import pathlib
import sys
import time

import opengate as gate
from opengate.geometry.materials import HounsfieldUnit_to_material
from opengate.image import get_translation_between_images_center, read_image_info
from opengate.utility import g4_units

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lu177_water_validation import DEFAULT_HIT_TYPE, HIT_TYPE_HELP, HIT_TYPES, add_dose_actors, merge_split_outputs  # noqa: E402

GATE_DATA = pathlib.Path(gate.__file__).parent / "data"


def build(args):
    m, mm, keV, gcm3 = g4_units.m, g4_units.mm, g4_units.keV, g4_units.g_cm3
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    sim = gate.Simulation()
    sim.g4_verbose = False
    sim.progress_bar = False
    sim.number_of_threads = args.threads
    sim.random_seed = args.seed
    sim.output_dir = out
    sim.world.size = [2 * m, 2 * m, 2 * m]
    sim.world.material = "G4_AIR"

    ct = sim.add_volume("Image", "ct")
    ct.image = args.ct
    ct.material = "G4_AIR"
    if args.phantom == "ct":
        ct.voxel_materials, _ = HounsfieldUnit_to_material(
            sim, args.density_tolerance * gcm3,
            GATE_DATA / "Schneider2000MaterialsTable.txt",
            GATE_DATA / "Schneider2000DensitiesTable.txt",
        )
        if args.material_overrides:
            db = sim.volume_manager.material_database
            for item in json.loads(pathlib.Path(args.material_overrides).read_text()):
                db.add_material_weights(item["name"], item["elements"], item["weights"], item["density_g_cm3"] * gcm3)
                ct.voxel_materials.append([item["value"], item["value"] + 1, item["name"]])
        else:
            ct.dump_label_image = out / "labels.mhd"
    else:
        ct.voxel_materials = [[-100000, 100000, "G4_WATER"]]

    source = sim.add_source("VoxelSource", "lu177")
    source.attached_to = ct.name
    source.particle = "ion"
    source.ion.Z = 71
    source.ion.A = 177
    source.energy.mono = 0 * keV
    source.direction.type = "iso"
    source.image = args.source
    source.n = args.n // args.threads
    source.position.translation = get_translation_between_images_center(args.ct, args.source)

    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option4"
    sim.physics_manager.enable_decay = True
    sim.physics_manager.set_production_cut("world", "all", 1 * m)
    sim.physics_manager.set_production_cut(ct.name, "all", args.cut_mm * mm)

    info = read_image_info(args.source)
    def configure(dose):
        dose.attached_to = ct.name
        dose.size = info.size
        dose.spacing = info.spacing
        dose.translation = source.position.translation
        dose.output_coordinate_system = "attached_to_image"
        dose.dose.active = True
        dose.dose_uncertainty.active = True

    add_dose_actors(sim, "mc", args.hit_type, configure)
    if args.random_reference:
        add_dose_actors(sim, "mcref", "random", configure)

    stats = sim.add_actor("SimulationStatisticsActor", "stats")
    stats.output_filename = "stats.txt"
    stats.stats.write_to_disk = True
    return sim, stats


def main():
    p = argparse.ArgumentParser(description="Lu-177 voxelised-source Monte Carlo in a patient CT or in water (GATE 10).")
    p.add_argument("--ct", required=True, help="CT image (HU), same grid as the source")
    p.add_argument("--source", required=True, help="cumulated-activity image used as decay probability map")
    p.add_argument("--output", required=True)
    p.add_argument("--phantom", choices=["ct", "water"], default="ct")
    p.add_argument("--n", type=int, required=True, help="total number of Lu-177 decays (all threads)")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=123456)
    p.add_argument("--cut-mm", type=float, default=1.0)
    p.add_argument("--density-tolerance", type=float, default=0.05, help="g/cm3, HU-to-material binning")
    p.add_argument("--hit-type", choices=HIT_TYPES, default=DEFAULT_HIT_TYPE, help=HIT_TYPE_HELP)
    p.add_argument("--random-reference", action="store_true", help="also score with hit type random in the same run (mcref_dose.mhd): paired comparison")
    p.add_argument("--material-overrides", help="json list of {name, value, density_g_cm3, elements, weights}: voxels of the CT image equal to value get this material")
    args = p.parse_args()

    sim, stats = build(args)
    t0 = time.time()
    sim.run()
    elapsed = time.time() - t0
    if args.hit_type == "split":
        merge_split_outputs(args.output, "mc")
    s = stats.counts
    summary = {"n_requested": args.n, "threads": args.threads, "events": int(s.events), "tracks": int(s.tracks),
               "seconds": elapsed, "phantom": args.phantom, "cut_mm": args.cut_mm,
               "density_tolerance_gcm3": args.density_tolerance, "seed": args.seed, "material_overrides": args.material_overrides,
               "physics": "G4EmStandardPhysics_option4", "hit_type": args.hit_type, "random_reference": args.random_reference, "ct": str(args.ct), "source": str(args.source)}
    (pathlib.Path(args.output) / "run.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
