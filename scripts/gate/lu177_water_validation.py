#!/usr/bin/env python3
import argparse
import json
import pathlib
import time

import numpy as np
import opengate as gate
import opengate.sources.utility as rad
from opengate.utility import g4_units

MEV_TO_J = 1.602176634e-13
WATER_KG_M3 = 1000.0


def emitted_energy_mev(nuclide="Lu177"):
    def total(kind):
        s = rad.get_spectrum(nuclide, kind)
        return float(np.sum(np.asarray(s.energies) * np.asarray(s.weights)))

    electrons = total("beta-") + total("auger") + total("IE")
    photons = total("gamma") + total("X")
    return electrons, photons


def photon_yield(nuclide="Lu177"):
    total = 0.0
    for kind in ("gamma", "X"):
        s = rad.get_spectrum(nuclide, kind)
        e, w = np.asarray(s.energies), np.asarray(s.weights)
        total += float(w[e >= 1e-3].sum())
    return total


def base_simulation(args, out):
    m = g4_units.m
    sim = gate.Simulation()
    sim.g4_verbose = False
    sim.progress_bar = False
    sim.number_of_threads = args.threads
    sim.random_seed = args.seed
    sim.output_dir = out
    sim.world.size = [2 * m, 2 * m, 2 * m]
    sim.world.material = "G4_AIR"
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option4"
    sim.physics_manager.enable_decay = True
    sim.physics_manager.set_production_cut("world", "all", 1 * m)
    return sim


def add_lu177(sim, volume, n, threads, kind="ion"):
    src = sim.add_source("GenericSource", "lu177")
    src.attached_to = volume
    if kind == "photon":
        e, w = [], []
        for k in ("gamma", "X"):
            sp = rad.get_spectrum("Lu177", k)
            e += list(sp.energies)
            w += list(sp.weights)
        e, w = np.asarray(e), np.asarray(w)
        keep = e >= 1e-3
        src.particle = "gamma"
        src.energy.type = "spectrum_discrete"
        src.energy.spectrum_energies = list(e[keep])
        src.energy.spectrum_weights = list(w[keep])
    elif kind == "beta":
        src.particle = "e-"
        b = rad.get_spectrum("Lu177", "b-spectra")
        e, w = np.asarray(b.energies), np.asarray(b.weights)
        edges = np.linspace(0, e[-1], 2001)
        centres = 0.5 * (edges[1:] + edges[:-1])
        src.energy.type = "spectrum_histogram"
        src.energy.spectrum_energy_bin_edges = list(edges)
        src.energy.spectrum_weights = list(np.interp(centres, e, w) * np.diff(edges))
    else:
        src.particle = "ion"
        src.ion.Z = 71
        src.ion.A = 177
        src.energy.mono = 0
    src.direction.type = "iso"
    src.n = n // threads
    return src


def add_stats(sim):
    stats = sim.add_actor("SimulationStatisticsActor", "stats")
    stats.output_filename = "stats.txt"
    stats.stats.write_to_disk = True
    return stats


HIT_TYPES = ["split", "random", "post", "pre", "middle"]
DEFAULT_HIT_TYPE = "split"
HIT_TYPE_HELP = "split (default) = steps of gammas scored at the post-step point (the interaction point), all other steps at a random point of the step; random = GATE default, used for every run before 2026-11-17 (inflates the photon tail in a homogeneous volume, identical in a voxelised CT)"


def add_dose_actors(sim, prefix, hit_type, configure):
    """DoseActor(s) named after prefix. split: steps of gammas scored at the post-step point (the interaction point), all other steps at a random point of the step; merge_split_outputs then writes <prefix>_dose.mhd etc. as for a single actor."""
    def actor(name, hit, filt=None):
        dose = sim.add_actor("DoseActor", name)
        configure(dose)
        dose.output_filename = f"{name}.mhd"
        dose.hit_type = hit
        if filt is not None:
            dose.filter = filt
        return dose

    if hit_type != "split":
        return [actor(prefix, hit_type)]
    from opengate.actors.filters import GateFilterBuilder
    F = GateFilterBuilder()
    return [actor(f"{prefix}_gamma", "post", F("ParticleName") == "gamma"), actor(f"{prefix}_other", "random", F("ParticleName") != "gamma")]


def merge_split_outputs(out, prefix="vsv"):
    import SimpleITK as sitk
    out = pathlib.Path(out)

    def read(name):
        path = out / name
        if not path.exists():
            return None, None
        img = sitk.ReadImage(str(path))
        return img, sitk.GetArrayFromImage(img)

    ref, dg = read(f"{prefix}_gamma_dose.mhd")
    _, do = read(f"{prefix}_other_dose.mhd")
    arrays = {}
    if dg is not None and do is not None:
        arrays[f"{prefix}_dose.mhd"] = dg + do
        _, ug = read(f"{prefix}_gamma_dose_uncertainty.mhd")
        _, uo = read(f"{prefix}_other_dose_uncertainty.mhd")
        if ug is not None and uo is not None:
            total = dg + do
            arrays[f"{prefix}_dose_uncertainty.mhd"] = np.divide(np.hypot(ug * dg, uo * do), total, out=np.zeros_like(total), where=total > 0)
    edep_ref, eg = read(f"{prefix}_gamma_edep.mhd")
    _, eo = read(f"{prefix}_other_edep.mhd")
    if eg is not None and eo is not None:
        arrays[f"{prefix}_edep.mhd"] = eg + eo
    ref = ref if ref is not None else edep_ref
    for name, arr in arrays.items():
        img = sitk.GetImageFromArray(arr)
        img.CopyInformation(ref)
        sitk.WriteImage(img, str(out / name))


def run_vsv(args):
    mm = g4_units.mm
    out = pathlib.Path(args.output) / args.subdir
    out.mkdir(parents=True, exist_ok=True)
    sim = base_simulation(args, out)

    water = sim.add_volume("Box", "water")
    water.size = [args.phantom_mm * mm] * 3
    water.material = args.material
    sim.physics_manager.set_production_cut(water.name, "all", args.cut_mm * mm)

    src = add_lu177(sim, water.name, args.n, args.threads, args.source)
    src.position.type = "box"
    src.position.size = [args.voxel_mm * mm] * 3

    def configure(dose):
        dose.attached_to = water.name
        dose.size = [args.kernel_size] * 3
        dose.spacing = [args.voxel_mm * mm] * 3
        dose.dose.active = True
        dose.dose_uncertainty.active = True
        dose.dose_squared.active = True

    add_dose_actors(sim, "vsv", args.hit_type, configure)
    stats = add_stats(sim)

    t0 = time.time()
    sim.run()
    elapsed = time.time() - t0

    import SimpleITK as sitk
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.hit_type == "split":
        merge_split_outputs(out, "vsv")

    events = int(stats.counts.events)
    dose_gy = sitk.GetArrayFromImage(sitk.ReadImage(str(out / "vsv_dose.mhd")))
    edep = sitk.GetArrayFromImage(sitk.ReadImage(str(out / "vsv_edep.mhd")))
    per_decay = photon_yield() if args.source == "photon" else 1.0
    kernel = dose_gy / events * 1e6 * 1e3 * per_decay
    np.save(out / "vsv_lu177_mGy_per_MBq_s.npy", kernel)

    c = args.kernel_size // 2
    electrons, photons = emitted_energy_mev()
    voxel_kg = (args.voxel_mm * 1e-3) ** 3 * WATER_KG_M3
    limit = electrons * MEV_TO_J / voxel_kg * 1e9
    idx = np.indices(kernel.shape) - c
    r = np.sqrt((idx ** 2).sum(0)) * args.voxel_mm

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(r.ravel(), kernel.ravel(), ".", ms=3)
    ax.set_xlabel("distance from source voxel centre (mm)")
    ax.set_ylabel("S (mGy / (MBq s))")
    ax.set_title(f"Lu-177 voxel S-values in water, {args.voxel_mm:g} mm voxels")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "vsv_radial_profile.png", dpi=150)

    results = {"voxel_mm": args.voxel_mm, "kernel_size": args.kernel_size, "phantom_mm": args.phantom_mm,
               "material": args.material, "source": args.source, "hit_type": args.hit_type,
               "decays": events, "seconds": elapsed, "cut_mm": args.cut_mm, "seed": args.seed,
               "S_centre_mGy_per_MBq_s": float(kernel[c, c, c]),
               "S_first_neighbour_mGy_per_MBq_s": float(kernel[c, c, c + 1]),
               "electron_self_absorption_limit_mGy_per_MBq_s": limit,
               "centre_over_limit": float(kernel[c, c, c] / limit),
               "kernel_deposited_MeV_per_decay": float(edep.sum() / events * per_decay),
               "source_voxel_deposited_MeV_per_decay": float(edep[c, c, c] / events * per_decay),
               "kernel_scale_per_decay": per_decay,
               "emitted_electrons_MeV": electrons, "emitted_photons_MeV": photons}
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def run_sphere(args):
    mm = g4_units.mm
    out = pathlib.Path(args.output) / "sphere"
    out.mkdir(parents=True, exist_ok=True)
    sim = base_simulation(args, out)

    sphere = sim.add_volume("Sphere", "sphere")
    sphere.rmax = args.radius_mm * mm
    sphere.material = "G4_WATER"
    sim.physics_manager.set_production_cut(sphere.name, "all", args.cut_mm * mm)

    src = add_lu177(sim, sphere.name, args.n, args.threads)
    src.position.type = "sphere"
    src.position.radius = args.radius_mm * mm

    def configure(edep):
        edep.attached_to = sphere.name
        edep.size = [1, 1, 1]
        edep.spacing = [2 * args.radius_mm * mm] * 3

    add_dose_actors(sim, "sphere", args.hit_type, configure)
    stats = add_stats(sim)

    t0 = time.time()
    sim.run()
    elapsed = time.time() - t0
    if args.hit_type == "split":
        merge_split_outputs(out, "sphere")

    import SimpleITK as sitk
    events = int(stats.counts.events)
    deposited = float(sitk.GetArrayFromImage(sitk.ReadImage(str(out / "sphere_edep.mhd"))).sum()) / events
    electrons, photons = emitted_energy_mev()
    mass = 4 / 3 * np.pi * (args.radius_mm * 1e-3) ** 3 * WATER_KG_M3
    to_s = MEV_TO_J / mass * 1e9
    results = {"radius_mm": args.radius_mm, "mass_kg": mass, "hit_type": args.hit_type, "decays": events, "seconds": elapsed,
               "cut_mm": args.cut_mm, "seed": args.seed,
               "emitted_electrons_MeV": electrons, "emitted_photons_MeV": photons,
               "deposited_MeV": deposited, "deposited_over_electrons": deposited / electrons,
               "S_mc_mGy_per_MBq_s": deposited * to_s, "S_electrons_only_mGy_per_MBq_s": electrons * to_s}
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def main():
    p = argparse.ArgumentParser(description="Lu-177 validation in water with GATE 10: voxel S-value kernel and uniform sphere.")
    sub = p.add_subparsers(dest="test", required=True)
    for name in ("vsv", "sphere"):
        s = sub.add_parser(name)
        s.add_argument("--output", default="output")
        s.add_argument("--threads", type=int, default=8)
        s.add_argument("--seed", type=int, default=123456)
        s.add_argument("--cut-mm", type=float, default=0.1)
        s.add_argument("--hit-type", default=DEFAULT_HIT_TYPE, choices=HIT_TYPES, help=HIT_TYPE_HELP)
    v = sub.choices["vsv"]
    v.add_argument("--n", type=int, default=10_000_000, help="total number of decays (all threads)")
    v.add_argument("--voxel-mm", type=float, default=4.0)
    v.add_argument("--kernel-size", type=int, default=21)
    v.add_argument("--phantom-mm", type=float, default=300.0)
    v.add_argument("--material", default="G4_WATER")
    v.add_argument("--source", choices=["ion", "beta", "photon"], default="ion",
                   help="ion = full Lu-177 decay; beta = ICRP-107 beta- spectrum only; photon = ICRP-107 gamma and X lines >= 1 keV, kernel scaled to per decay")
    v.add_argument("--subdir", default="vsv")

    s = sub.choices["sphere"]
    s.add_argument("--n", type=int, default=1_000_000, help="total number of decays (all threads)")
    s.add_argument("--radius-mm", type=float, default=20.0)
    args = p.parse_args()
    {"vsv": run_vsv, "sphere": run_sphere}[args.test](args)


if __name__ == "__main__":
    main()
