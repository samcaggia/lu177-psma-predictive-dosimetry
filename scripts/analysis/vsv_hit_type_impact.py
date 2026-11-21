#!/usr/bin/env python3
import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve


def load(folder):
    return np.load(pathlib.Path(folder) / "vsv_lu177_mGy_per_MBq_s.npy")


def split_component(folder, name):
    import SimpleITK as sitk
    full = load(folder)
    part = sitk.GetArrayFromImage(sitk.ReadImage(str(folder / f"vsv_{name}_dose.mhd")))
    total = sitk.GetArrayFromImage(sitk.ReadImage(str(folder / "vsv_dose.mhd")))
    return full * np.divide(part, total, out=np.zeros_like(total), where=total > 0)


def sphere(volume_ml, voxel_mm, size):
    c = size // 2
    r = (3 * volume_ml * 1000 / (4 * np.pi)) ** (1 / 3)
    idx = (np.indices((size,) * 3) - c) * voxel_mm
    return (np.sqrt((idx ** 2).sum(0)) <= r).astype(float)


def mean_dose(mask, kernel):
    a = mask / mask.sum()
    return float((fftconvolve(a, kernel, mode="same") * mask).sum() / mask.sum())


def main():
    p = argparse.ArgumentParser(description="Effect of the GATE DoseActor hit type (random vs post on photon steps) on Lu-177 doses in water.")
    p.add_argument("--controls", default="output/vsv_lanconelli")
    p.add_argument("--out", default="output/vsv")
    args = p.parse_args()
    ctrl, out = pathlib.Path(args.controls), pathlib.Path(args.out)
    voxel = 4.0

    pr, pp = load(ctrl / "photon_big_random"), load(ctrl / "photon_big_post")
    other = split_component(ctrl / "ion_water_split", "other")
    size = pr.shape[0]
    c = size // 2
    o = np.zeros_like(pr)
    h = other.shape[0] // 2
    o[c - h:c + h + 1, c - h:c + h + 1, c - h:c + h + 1] = other

    n = ((np.indices(pr.shape) - c) ** 2).sum(0)
    rows = []
    for m in np.unique(n):
        sel = n == m
        rows.append({"r_mm": voxel * np.sqrt(m), "photon_random": pr[sel].mean(), "photon_post": pp[sel].mean()})
    radial = pd.DataFrame(rows)
    radial["random_over_post"] = radial["photon_random"] / radial["photon_post"]
    bins = [0, 0.1, 6, 10, 20, 40, 60, 80, 101]
    radial["band"] = pd.cut(radial["r_mm"], bins, right=False)
    band = radial.groupby("band", observed=True)["random_over_post"].median()

    srows = []
    for v in (0.5, 1, 5, 10, 50, 100, 300):
        mask = sphere(v, voxel, size)
        d_other = mean_dose(mask, o)
        d_r, d_p = mean_dose(mask, pr), mean_dose(mask, pp)
        srows.append({"volume_ml": v, "voxels": int(mask.sum()), "electron_part": d_other,
                      "photon_random": d_r, "photon_post": d_p,
                      "photon_share_post": d_p / (d_other + d_p),
                      "total_random_over_post": (d_other + d_r) / (d_other + d_p)})
    spheres = pd.DataFrame(srows)

    in_r = pr.sum() * (voxel * 1e-3) ** 3 * 1000 / (1.602176634e-13 * 1e9)
    in_p = pp.sum() * (voxel * 1e-3) ** 3 * 1000 / (1.602176634e-13 * 1e9)
    radial.drop(columns="band").to_csv(out / "hit_type_photon_radial.csv", index=False)
    spheres.to_csv(out / "hit_type_spheres.csv", index=False)
    summary = {"photon_in_grid_MeV_per_decay_random": float(in_r), "photon_in_grid_MeV_per_decay_post": float(in_p),
               "grid_half_width_mm": c * voxel,
               "photon_random_over_post_by_band": {str(k): float(v) for k, v in band.items()}}
    (out / "hit_type_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(spheres.to_string(float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
