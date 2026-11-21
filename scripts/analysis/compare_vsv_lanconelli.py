#!/usr/bin/env python3
import argparse
import json
import pathlib

import numpy as np
import pandas as pd

MEV_TO_J = 1.602176634e-13
CONTROLS = ("ion_water_split", "ion_water_post", "ion_soft", "beta_water", "beta_soft", "photon_water", "photon_water_post")
RE_CM = 2.8179403262e-13
ME_MEV = 0.51099895
ELECTRONS_PER_G_WATER = 10 / 18.01528 * 6.02214076e23

NIST_WATER = np.array([
    [1.0e-3, 4.078e3, 4.065e3], [1.5e-3, 1.376e3, 1.372e3], [2.0e-3, 6.173e2, 6.152e2],
    [3.0e-3, 1.929e2, 1.917e2], [4.0e-3, 8.278e1, 8.191e1], [5.0e-3, 4.258e1, 4.188e1],
    [6.0e-3, 2.464e1, 2.405e1], [8.0e-3, 1.037e1, 9.915e0], [1.0e-2, 5.329e0, 4.944e0],
    [1.5e-2, 1.673e0, 1.374e0], [2.0e-2, 8.096e-1, 5.503e-1], [3.0e-2, 3.756e-1, 1.557e-1],
    [4.0e-2, 2.683e-1, 6.947e-2], [5.0e-2, 2.269e-1, 4.223e-2], [6.0e-2, 2.059e-1, 3.190e-2],
    [8.0e-2, 1.837e-1, 2.597e-2], [1.0e-1, 1.707e-1, 2.546e-2], [1.5e-1, 1.505e-1, 2.764e-2],
    [2.0e-1, 1.370e-1, 2.967e-2], [3.0e-1, 1.186e-1, 3.192e-2], [4.0e-1, 1.061e-1, 3.279e-2],
])


def kn_total(e):
    k = e / ME_MEV
    a = (1 + k) / k ** 2 * (2 * (1 + k) / (1 + 2 * k) - np.log(1 + 2 * k) / k)
    b = np.log(1 + 2 * k) / (2 * k) - (1 + 3 * k) / (1 + 2 * k) ** 2
    return 2 * np.pi * RE_CM ** 2 * (a + b) * ELECTRONS_PER_G_WATER


def kn_energy_transfer(e, n=4001):
    k = e[:, None] / ME_MEV
    c = np.linspace(-1, 1, n)[None, :]
    p = 1 / (1 + k * (1 - c))
    d = np.pi * RE_CM ** 2 * p ** 2 * (p + 1 / p - (1 - c ** 2))
    return np.trapezoid(d * (1 - p), c, axis=1) * ELECTRONS_PER_G_WATER


def photo_table():
    e, mu, mu_en = NIST_WATER.T
    tau = mu_en - kn_energy_transfer(e)
    fit = (e >= 1e-2) & (e <= 4e-2)
    slope, icpt = np.polyfit(np.log(e[fit]), np.log(tau[fit]), 1)
    tau = np.where(e <= 4e-2, tau, np.exp(icpt + slope * np.log(e)))
    return e, tau, mu, slope


def photon_lines():
    import opengate.sources.utility as rad
    es, ys = [], []
    for kind in ("gamma", "X"):
        s = rad.get_spectrum("Lu177", kind)
        es.append(np.asarray(s.energies))
        ys.append(np.asarray(s.weights))
    e, y = np.concatenate(es), np.concatenate(ys)
    keep = e >= 1e-3
    return e[keep], y[keep]


def photon_vsv(n_photons, voxel_mm, size, half_box_mm, seed, cut_mev=1e-3, batch=500_000):
    rng = np.random.default_rng(seed)
    e_tab, tau_tab, _, _ = photo_table()
    lines, yields = photon_lines()
    p_line = yields / yields.sum()

    def tau(e):
        return np.exp(np.interp(np.log(e), np.log(e_tab), np.log(tau_tab)))

    v = voxel_mm / 10
    half = size * v / 2
    edges = np.linspace(-half, half, size + 1)
    edep = np.zeros((size, size, size))
    box = half_box_mm / 10
    escaped = 0.0

    def score(pos, de):
        inside = np.all(np.abs(pos) < half, axis=1)
        if inside.any():
            h, _ = np.histogramdd(pos[inside], bins=(edges,) * 3, weights=de[inside])
            edep[...] += h

    done = 0
    while done < n_photons:
        m = min(batch, n_photons - done)
        done += m
        e = lines[rng.choice(len(lines), m, p=p_line)]
        pos = (rng.random((m, 3)) - 0.5) * v
        cz = 2 * rng.random(m) - 1
        ph = 2 * np.pi * rng.random(m)
        sz = np.sqrt(1 - cz ** 2)
        d = np.stack([sz * np.cos(ph), sz * np.sin(ph), cz], 1)
        while len(e):
            t, s = tau(e), kn_total(e)
            mu = t + s
            pos = pos + d * (-np.log(rng.random(len(e))) / mu)[:, None]
            out = np.any(np.abs(pos) > box, axis=1)
            escaped += e[out].sum()
            e, pos, d, t, mu = e[~out], pos[~out], d[~out], t[~out], mu[~out]
            photo = rng.random(len(e)) < t / mu
            score(pos[photo], e[photo])
            e, pos, d = e[~photo], pos[~photo], d[~photo]
            k = e / ME_MEV
            c = np.empty(len(e))
            todo = np.arange(len(e))
            while len(todo):
                cc = 2 * rng.random(len(todo)) - 1
                p = 1 / (1 + k[todo] * (1 - cc))
                acc = rng.random(len(todo)) * 2 < p ** 2 * (p + 1 / p - (1 - cc ** 2))
                c[todo[acc]] = cc[acc]
                todo = todo[~acc]
            e_new = e / (1 + k * (1 - c))
            score(pos, e - e_new)
            phi = 2 * np.pi * rng.random(len(e))
            d = rotate(d, c, phi)
            e = e_new
            low = e < cut_mev
            score(pos[low], e[low])
            e, pos, d = e[~low], pos[~low], d[~low]
    yield_total = yields.sum()
    edep_per_decay = edep / n_photons * yield_total
    mass_kg = (voxel_mm * 1e-3) ** 3 * 1000.0
    s_vsv = edep_per_decay * MEV_TO_J / mass_kg * 1e9
    return s_vsv, edep_per_decay, escaped / n_photons * yield_total


def rotate(d, c, phi):
    s = np.sqrt(np.clip(1 - c ** 2, 0, None))
    ux, uy, uz = d.T
    perp = np.sqrt(np.clip(1 - uz ** 2, 1e-12, None))
    nx = ux * c + s * (ux * uz * np.cos(phi) - uy * np.sin(phi)) / perp
    ny = uy * c + s * (uy * uz * np.cos(phi) + ux * np.sin(phi)) / perp
    nz = uz * c - s * np.cos(phi) * perp
    polar = perp < 1e-6
    nx = np.where(polar, s * np.cos(phi), nx)
    ny = np.where(polar, s * np.sin(phi), ny)
    nz = np.where(polar, np.sign(uz) * c, nz)
    out = np.stack([nx, ny, nz], 1)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def shell_table(kernel, rel_unc, voxel_mm):
    c = kernel.shape[0] // 2
    idx = np.indices(kernel.shape) - c
    n = (idx ** 2).sum(0)
    rows = []
    for m in np.unique(n):
        sel = n == m
        vals = kernel[sel]
        abs_unc = np.sqrt(np.sum((vals * rel_unc[sel]) ** 2)) / sel.sum() if rel_unc is not None else np.nan
        rows.append({"n": int(m), "r_mm": voxel_mm * np.sqrt(m), "nvox": int(sel.sum()),
                     "S": vals.mean(), "S_se": abs_unc})
    return pd.DataFrame(rows)


def load_gate(folder):
    import SimpleITK as sitk
    folder = pathlib.Path(folder)
    k = np.load(folder / "vsv_lu177_mGy_per_MBq_s.npy")
    u = sitk.GetArrayFromImage(sitk.ReadImage(str(folder / "vsv_dose_uncertainty.mhd")))
    return k, u


def main():
    p = argparse.ArgumentParser(description="Compare the GATE Lu-177 4 mm VSV with reference values of Lanconelli et al. 2012 "
                                "(Phys Med Biol 57:517). The reference table is not distributed: supply your own CSV with columns "
                                "distance_mm, S_mGy_per_MBq_s (soft tissue, 4 mm voxels).")
    p.add_argument("--kernel", default="output/vsv")
    p.add_argument("--controls", default="output/vsv_lanconelli")
    p.add_argument("--lanconelli", required=True, help="CSV with columns distance_mm, S_mGy_per_MBq_s")
    p.add_argument("--out", default="output/vsv")
    p.add_argument("--photons", type=int, default=4_000_000)
    p.add_argument("--seed", type=int, default=20261117)
    args = p.parse_args()
    out = pathlib.Path(args.out)
    ctrl = pathlib.Path(args.controls)
    voxel = 4.0

    k, u = load_gate(args.kernel)
    main_tab = shell_table(k, u, voxel)

    cache = ctrl / "photon_mc.npz"
    if cache.exists():
        z = np.load(cache)
        s_ph, escaped = z["s"], float(z["escaped"])
    else:
        s_ph, _, escaped = photon_vsv(args.photons, voxel, k.shape[0], 150.0, args.seed)
        ctrl.mkdir(parents=True, exist_ok=True)
        np.savez(cache, s=s_ph, escaped=escaped, photons=args.photons, seed=args.seed)
    ph_tab = shell_table(s_ph, None, voxel)
    main_tab["S_photon_mc"] = ph_tab["S"].values

    controls = {}
    for name in CONTROLS:
        f = ctrl / name
        if (f / "vsv_lu177_mGy_per_MBq_s.npy").exists():
            kc, uc = load_gate(f)
            t = shell_table(kc, uc, voxel)
            main_tab[f"S_{name}"] = t["S"].values
            main_tab[f"S_{name}_se"] = t["S_se"].values
            controls[name] = json.loads((f / "results.json").read_text())

    lan = pd.read_csv(args.lanconelli)
    lan["n"] = np.rint((lan["distance_mm"] / voxel) ** 2).astype(int)
    lan["lattice_mm"] = voxel * np.sqrt(lan["n"])
    m = lan.merge(main_tab, on="n", how="left")
    m["ratio_gate_lanconelli"] = m["S"] / m["S_mGy_per_MBq_s"]
    m["ratio_se_stat"] = m["S_se"] / m["S_mGy_per_MBq_s"]
    m["ratio_photon_mc_lanconelli"] = m["S_photon_mc"] / m["S_mGy_per_MBq_s"]
    m["gate_over_photon_mc"] = m["S"] / m["S_photon_mc"]
    if "S_photon_water_post" in m:
        m["gate_photon_post_over_photon_mc"] = m["S_photon_water_post"] / m["S_photon_mc"]
        m["gate_photon_random_over_photon_mc"] = m["S_photon_water"] / m["S_photon_mc"]
    if "S_beta_soft" in m and "S_photon_water_post" in m:
        m["ratio_beta_soft_plus_photon_lanconelli"] = (m["S_beta_soft"] + m["S_photon_water_post"]) / m["S_mGy_per_MBq_s"]
    for name in controls:
        m[f"ratio_{name}_lanconelli"] = m[f"S_{name}"] / m["S_mGy_per_MBq_s"]
    m.to_csv(out / "lanconelli_comparison.csv", index=False)
    main_tab.to_csv(out / "vsv_shells.csv", index=False)

    import opengate.sources.utility as rad
    y = {kind: float(np.sum(np.asarray(rad.get_spectrum("Lu177", kind).energies) *
                            np.asarray(rad.get_spectrum("Lu177", kind).weights)))
         for kind in ("beta-", "auger", "IE", "gamma", "X")}
    mass_water = (voxel * 1e-3) ** 3 * 1000.0
    to_s = MEV_TO_J * 1e9
    electrons = y["beta-"] + y["auger"] + y["IE"]
    limits = {
        "full_electrons_water": electrons * to_s / mass_water,
        "beta_only_water": y["beta-"] * to_s / mass_water,
        "beta_only_rho_1.03": y["beta-"] * to_s / (mass_water * 1.03),
        "beta_only_rho_1.04": y["beta-"] * to_s / (mass_water * 1.04),
        "full_electrons_rho_1.03": electrons * to_s / (mass_water * 1.03),
    }
    c = k.shape[0] // 2
    n_all = ((np.indices(k.shape) - c) ** 2).sum(0)
    total = k.sum()
    tail = k[n_all >= 4].sum()
    lan_by_n = dict(zip(lan["n"], lan["S_mGy_per_MBq_s"]))
    ratio_tail = m.loc[m["n"] >= 4, "ratio_gate_lanconelli"]
    e_tab, tau_tab, mu_tab, slope = photo_table()
    summary = {
        "emitted_keV_per_decay": {kk: v * 1e3 for kk, v in y.items()},
        "electrons_keV": electrons * 1e3,
        "electron_ratio_full_over_beta": electrons / y["beta-"],
        "limits_mGy_per_MBq_s": limits,
        "gate_centre": float(k[c, c, c]),
        "lanconelli_centre": float(lan_by_n[0]),
        "gate_centre_over_full_limit_water": float(k[c, c, c] / limits["full_electrons_water"]),
        "lanconelli_centre_over_beta_limit_1.04": float(lan_by_n[0] / limits["beta_only_rho_1.04"]),
        "lanconelli_centre_over_beta_limit_1.03": float(lan_by_n[0] / limits["beta_only_rho_1.03"]),
        "kernel_sum_share_n_ge_4": float(tail / total),
        "kernel_sum_share_n_ge_4_if_tail_scaled_by_median_ratio": float(
            (tail / np.median(ratio_tail)) / (total - tail + tail / np.median(ratio_tail))),
        "median_tail_ratio_n_ge_4": float(np.median(ratio_tail)),
        "photon_mc_photons": args.photons,
        "photon_mc_escaped_MeV_per_decay": escaped,
        "photon_mc_in_kernel_MeV_per_decay": float(s_ph.sum() / to_s * mass_water),
        "photon_emitted_MeV_per_decay": y["gamma"] + y["X"],
        "photo_fit_slope": slope,
        "mu_model_over_nist": dict(zip([f"{e*1e3:g} keV" for e in e_tab[8:20]],
                                       ((np.exp(np.interp(np.log(e_tab[8:20]), np.log(e_tab), np.log(tau_tab)))
                                         + kn_total(e_tab[8:20])) / mu_tab[8:20]).round(4).tolist())),
        "controls": {kk: {"S_centre": v["S_centre_mGy_per_MBq_s"], "S_face": v["S_first_neighbour_mGy_per_MBq_s"],
                          "deposited_MeV": v["kernel_deposited_MeV_per_decay"], "decays": v["decays"]}
                     for kk, v in controls.items()},
    }
    (out / "lanconelli_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(m[["distance_mm", "n", "lattice_mm", "S_mGy_per_MBq_s", "S", "ratio_gate_lanconelli", "ratio_se_stat",
             "ratio_photon_mc_lanconelli", "gate_over_photon_mc"] +
            [c for c in ("gate_photon_post_over_photon_mc", "gate_photon_random_over_photon_mc") if c in m] +
            [f"ratio_{nm}_lanconelli" for nm in controls]].to_string(float_format=lambda v: f"{v:.4g}"))

    figure(main_tab, m, controls, out / "lanconelli_comparison.png")


def figure(tab, m, controls, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(2, 1, figsize=(7, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a.semilogy(tab["r_mm"], tab["S"], "o", ms=4, color="#9aa5b1", label="GATE water, output/vsv (hit type random), 1e7 decays")
    if "S_ion_water_split" in tab:
        a.semilogy(tab["r_mm"], tab["S_ion_water_split"], "o", ms=3, color="#1f5fa8", label="GATE water, photon steps at post-step point, 1e7 decays")
    pos = tab["S_photon_mc"] > 0
    a.semilogy(tab["r_mm"][pos & (tab["n"] >= 1)], tab["S_photon_mc"][pos & (tab["n"] >= 1)], "s", ms=3, mfc="none",
               color="#2a9d5c", label="photons only, independent NumPy MC (water)")
    a.semilogy(m["distance_mm"], m["S_mGy_per_MBq_s"], "^", ms=6, color="#c0392b",
               label="Lanconelli 2012, soft tissue (user-supplied reference)")
    a.set_ylabel("S (mGy / (MBq s))")
    a.set_title("Lu-177 voxel S values, 4 mm voxels")
    a.grid(alpha=0.3, which="both")
    a.legend(fontsize=8)
    b.errorbar(m["lattice_mm"], m["ratio_gate_lanconelli"], yerr=np.hypot(m["ratio_se_stat"], 0.04 * m["ratio_gate_lanconelli"]),
               fmt="o", color="#9aa5b1", ms=4, capsize=2, label="output/vsv (random) / Lanconelli (bars: stat + 4% digitisation)")
    if "ratio_ion_water_split_lanconelli" in m:
        b.errorbar(m["lattice_mm"] + 0.3, m["ratio_ion_water_split_lanconelli"],
                   yerr=np.hypot(m["S_ion_water_split_se"] / m["S_mGy_per_MBq_s"], 0.04 * m["ratio_ion_water_split_lanconelli"]),
                   fmt="o", color="#1f5fa8", ms=4, capsize=2, label="corrected water kernel / Lanconelli")
    if "ratio_beta_soft_plus_photon_lanconelli" in m:
        b.plot(m["lattice_mm"] - 0.3, m["ratio_beta_soft_plus_photon_lanconelli"], "x", color="#e08a1e", ms=5,
               label="GATE beta only (ICRP soft tissue) + photons (post) / Lanconelli")
    b.plot(m["lattice_mm"], m["ratio_photon_mc_lanconelli"].where(m["n"] >= 4), "s", mfc="none", color="#2a9d5c", ms=4,
           label="independent photon MC / Lanconelli (n >= 4 only)")
    b.set_ylim(0.6, 2.0)
    b.axhline(1, color="k", lw=0.8)
    b.set_ylabel("ratio to Lanconelli")
    b.set_xlabel("distance between voxel centres (mm)")
    b.grid(alpha=0.3)
    b.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
