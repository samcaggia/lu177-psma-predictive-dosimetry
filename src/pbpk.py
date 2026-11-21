"""Whole-body PBPK model of PSMA ligands, written from Kletting et al., PLoS One 2016, S1 File (Eqs. 1-13, Table A).

Equations, sources and deviations are documented in notebooks/12_pbpk_model.ipynb. Units: nmol, L, min.
"""
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline
from scipy.sparse import diags

HALF_LIFE_MIN = {"Lu177": 6.647 * 24 * 60, "Ga68": 67.71}
AVOGADRO_NMOL = 6.022e14
DEFAULT_GFR_ML_MIN = 193 / 3 * 20 / 15
COUPLED_TLV_ML = 1000.0

NEGATIVE_TISSUES = {
    "muscle": dict(v71=30.078, vv=0.14, alpha=5.9, flow=0.17, k=0.02),
    "skin": dict(v71=3.408, vv=0.03, alpha=8.9, flow=0.05, k=0.02),
    "adipose": dict(v71=13.465, vv=0.05, alpha=15.5, flow=0.05, k=0.02),
    "red_marrow": dict(v71=1.1, vv=0.04, alpha=3.7, flow=0.03, k=2.0),
    "heart": dict(v71=0.341, vv=0.01, alpha=3.7, flow=0.04, k=0.02),
}
K_MUSCLE = 0.02
K_ORGAN = K_MUSCLE * 100
V_BONE71 = 10.165
V_LUNG71, V_BRAIN71, V_PROSTATE71 = 1.0, 1.45, 0.016
V_GI71 = 0.385 + 0.548 + 0.104 + 0.15
DEFAULT_RDEN = {"kidneys": 24.0, "salivary": 42.0, "liver": 1.33, "spleen": 0.532, "gi": 1.596, "prostate": 26.6}
DEFAULT_LAMBDA_REL = {"kidneys": 2.9e-4, "salivary": 3.7e-4, "liver": 2.9e-4, "spleen": 2.9e-4, "gi": 2.9e-4, "prostate": 2.9e-4}
ORGANS = ("kidneys", "liver", "spleen", "salivary")


@dataclass
class Lesion:
    volume_ml: float
    r_den: float = 110.0
    lambda_rel: float = 2.0e-4
    f_ml_g_min: float = 0.5
    name: str | None = None


@dataclass
class Ligand:
    k_on: float = 0.046
    k_off: float = 0.046
    lambda_int: float = 1.0e-3
    k_pr: float = 4.7e-4
    phi: float = 0.66
    f_ex: float = 0.96


@dataclass
class Patient:
    patient_id: str
    weight_kg: float
    kidneys_ml: float
    liver_ml: float
    spleen_ml: float
    salivary_ml: float
    height_cm: float | None = None
    bsa_m2: float | None = None
    hematocrit: float = 0.4
    gfr_ml_min: float | None = None
    n_kidneys: int = 2
    single_kidney_compensation: float = 1.0
    f_sal: float = 0.16
    r_den: dict = field(default_factory=dict)
    lambda_rel: dict = field(default_factory=dict)
    lesions: list = field(default_factory=list)

    @property
    def bsa(self):
        if self.bsa_m2 is not None:
            return self.bsa_m2
        if self.height_cm is None:
            raise ValueError("bsa_m2 or height_cm is required")
        return 0.007184 * self.height_cm**0.725 * self.weight_kg**0.425

    @property
    def gfr_l_min(self):
        if self.gfr_ml_min is not None:
            return self.gfr_ml_min / 1000
        scale = 1.0 if self.n_kidneys == 2 else 0.5 * self.single_kidney_compensation
        return DEFAULT_GFR_ML_MIN * scale / 1000


@dataclass
class Injection:
    radionuclide: str
    activity_mbq: float
    total_nmol: float
    infusion_min: float = 0.0
    half_life_min: float | None = None

    @property
    def lambda_phys(self):
        return math.log(2) / (self.half_life_min or HALF_LIFE_MIN[self.radionuclide])

    @property
    def mbq_per_nmol(self):
        return self.lambda_phys / 60 * AVOGADRO_NMOL / 1e6

    @property
    def hot_nmol(self):
        return self.activity_mbq / self.mbq_per_nmol


def lesion_parameters(lesion, hematocrit, k_tumor=0.6):
    volume = lesion.volume_ml / 1000
    return dict(F=lesion.f_ml_g_min * volume, Vv=0.05 * (1 - hematocrit) * volume, Vint=0.38 * volume, PS=k_tumor * volume,
                R0=lesion.r_den * volume, lrel=lesion.lambda_rel, V=volume)


def aggregate_lesions(lesions, hematocrit, k_on, preserve_clearance=True):
    """Mean-field tumour: conserves total volume and binding sites, release rate weighted by binding sites.

    Flow and PS in series with binding give each lesion a low-occupancy clearance. Summing flows and PS overestimates
    the sum of these clearances for heterogeneous lesions (concavity), so the aggregate flow is reduced to preserve it.
    """
    parts = [lesion_parameters(x, hematocrit) for x in lesions]
    total = {key: sum(p[key] for p in parts) for key in ("F", "Vv", "Vint", "PS", "R0", "V")}
    total["lrel"] = sum(p["R0"] * p["lrel"] for p in parts) / total["R0"]
    if preserve_clearance:
        clearance = sum(1 / (1 / p["F"] + 1 / p["PS"] + 1 / (k_on * p["R0"])) for p in parts)
        remainder = 1 / clearance - 1 / total["PS"] - 1 / (k_on * total["R0"])
        total["F"] = 1 / remainder if remainder > 0 else total["F"]
    return total


def tissue_table(patient):
    """Normal tissues of Table A; liver, spleen, kidneys and salivary glands take the measured volumes."""
    h, bw = patient.hematocrit, patient.weight_kg
    vp = 2.8 * (1 - h) * patient.bsa
    f = vp * 1.23
    scale = bw / 71
    rden = {**DEFAULT_RDEN, **patient.r_den}
    lrel = {**DEFAULT_LAMBDA_REL, **patient.lambda_rel}
    t = {}

    def add(name, V, Vv, Vint, F, PS, positive=False):
        t[name] = dict(V=V, Vv=Vv, Vint=Vint, F=F, PS=PS, R0=rden[name] * V if positive else 0.0, lrel=lrel[name] if positive else 0.0)

    vk, vl, vs, vsal = (x / 1000 for x in (patient.kidneys_ml, patient.liver_ml, patient.spleen_ml, patient.salivary_ml))
    kidney_flow = 0.19 * f * (1.0 if patient.n_kidneys == 2 else 0.5 * patient.single_kidney_compensation)
    add("kidneys", vk, 0.055 * vk, 0.15 * vk, kidney_flow, 0.0, True)
    add("liver", vl, 0.085 * vl, 0.2 * vl, 0.065 * f, K_ORGAN * vl, True)
    add("spleen", vs, 0.12 * vs, 0.2 * vs, 0.03 * f, K_ORGAN * vs, True)
    add("salivary", vsal, 0.03 * (1 - h) * vsal, 0.23 * vsal, patient.f_sal * vsal, K_ORGAN * vsal, True)
    vgi = V_GI71 * scale
    add("gi", vgi, 0.076 * vp, 8.8 * 0.076 * vp, 0.16 * f, K_MUSCLE * vgi, True)
    vpro = V_PROSTATE71 * scale
    add("prostate", vpro, 0.004 * (1 - h) * vpro, 0.25 * vpro, 0.18 * (1 - h) * vpro, 0.1 * vpro, True)
    for name, p in NEGATIVE_TISSUES.items():
        v = p["v71"] * scale
        vv = p["vv"] * vp
        add(name, v, vv, p["alpha"] * vv, p["flow"] * f, p["k"] * v)
    vbone = V_BONE71 * scale - t["red_marrow"]["V"]
    vvb = 0.07 * vp - t["red_marrow"]["Vv"]
    add("bone", vbone, vvb, 8.4 * vvb, 0.05 * f, K_MUSCLE * vbone)
    vbr = V_BRAIN71 * scale
    add("brain", vbr, 0.012 * vp, 0.0, 0.12 * f, 0.0)
    vlu = V_LUNG71 * scale
    vvl = 0.105 * vp
    v_rest = bw - sum(x["V"] for x in t.values()) - vlu
    vv_rest = vp - sum(x["Vv"] for x in t.values()) - vvl
    f_rest = f - sum(x["F"] for x in t.values())
    if v_rest <= 0 or vv_rest <= 0 or f_rest <= 0:
        raise ValueError(f"{patient.patient_id}: rest of body not positive (V {v_rest:.3f}, Vv {vv_rest:.3f}, F {f_rest:.3f})")
    add("rest", v_rest, vv_rest, 4.1 * vv_rest, f_rest, K_MUSCLE * v_rest)
    arterial = (0.06 + 0.045) * vp
    venous = (0.18 + 0.045) * vp
    lung = dict(V=vlu, Vv=vvl, Vint=5.5 * vvl, F=f, PS=K_ORGAN * vlu, R0=0.0, lrel=0.0)
    return t, lung, dict(F=f, VP=vp, Vart=arterial, Vven=venous, Hct=h)


def _filling_time(x):
    """Interstitial filling time with the well-stirred clearance PS F / (PS + F)."""
    return x["Vint"] * (x["PS"] + x["F"]) / (x["PS"] * x["F"]) if x["Vint"] > 0 and x["PS"] > 0 else 0.0


def lump_rest(tissues, lung, n_pools=2, split_min=15.0):
    """PSMA-negative tissues, GI, prostate and lung interstitium in n_pools pools, split by interstitial filling time.

    Each pool sums volumes and sites and keeps the first moment of interstitial filling. Lung flow is in series and GI
    flow enters the liver directly (GI extraction is small), so neither crosses a pool; their volumes and sites stay.
    """
    members = {k: v for k, v in tissues.items() if k not in ORGANS} | {"lung": lung}
    groups = [list(members)] if n_pools == 1 else [[k for k in members if _filling_time(members[k]) < split_min], [k for k in members if _filling_time(members[k]) >= split_min]]
    pools = []
    for names in groups:
        parts = [members[k] for k in names]
        exchanging = [x for x in parts if x["Vint"] > 0 and x["PS"] > 0]
        vint = sum(x["Vint"] for x in exchanging)
        flow = sum(members[k]["F"] for k in names if k not in ("lung", "gi"))
        clearance = vint**2 / sum(x["Vint"] * _filling_time(x) for x in exchanging)
        r0 = sum(x["R0"] for x in parts)
        pools.append(dict(V=sum(x["V"] for x in parts), Vv=sum(x["Vv"] for x in parts), Vint=vint, F=flow, R0=r0, lrel=sum(x["R0"] * x["lrel"] for x in parts) / r0 if r0 > 0 else 0.0,
                          PS=1 / (1 / clearance - 1 / flow) if clearance < flow else 100 * flow))
    return pools


class Network:
    """Linear transfer matrix plus receptor binding terms, for total ligand."""

    def __init__(self, ligand):
        self.ligand = ligand
        self.names = []
        self.index = {}
        self.transfers = []
        self.receptors = []
        self.groups = {}

    def state(self, name, group=None, component=None):
        self.index[name] = len(self.names)
        self.names.append(name)
        if group is not None:
            self.groups.setdefault(group, {})[component] = self.index[name]
        return self.index[name]

    def transfer(self, source, target, rate):
        self.transfers.append((source, target, rate))

    def compile(self):
        n = len(self.names)
        A = np.zeros((n, n))
        for source, target, rate in self.transfers:
            A[source, source] -= rate
            if target is not None:
                A[target, source] += rate
        self.A = A
        self.n = n
        r = np.array(self.receptors) if self.receptors else np.zeros((0, 4))
        self.i_int = r[:, 0].astype(int)
        self.i_bound = r[:, 1].astype(int)
        self.r0 = r[:, 2]
        self.kon_over_v = self.ligand.k_on / r[:, 3] if len(r) else r[:, 3]
        return self

    def add_tissue(self, label, p, art, ven, drain=None, extra_inflow=0.0):
        """Eqs. 2-4 and 10-11. drain: venous target, liver for the portal tissues."""
        v = self.state(f"{label}.vascular", label, "vascular")
        self.transfer(art, v, (p["F"] + extra_inflow) / self.volumes["art"])
        self.transfer(v, ven if drain is None else drain, (p["F"] + extra_inflow) / p["Vv"])
        if p["Vint"] <= 0 or p["PS"] <= 0:
            return v, None
        i = self.state(f"{label}.interstitial", label, "interstitial")
        self.transfer(v, i, p["PS"] / p["Vv"])
        self.transfer(i, v, p["PS"] / p["Vint"])
        if p["R0"] > 0:
            self.add_binding(label, p, i)
        return v, i

    def add_binding(self, label, p, interstitial):
        lig = self.ligand
        b = self.state(f"{label}.bound", label, "bound")
        c = self.state(f"{label}.internalized", label, "internalized")
        self.receptors.append((interstitial, b, p["R0"], p["Vint"]))
        self.transfer(b, interstitial, lig.k_off)
        self.transfer(b, c, lig.lambda_int)
        self.transfer(c, self.index["released"], p["lrel"])


def build_network(patient, ligand=None, topology="reduced", lesion_mode="satellite", balance_flows=True, intra_return_volume_l=None,
                  portal_outflow=True, n_rest_pools=2, preserve_clearance=True):
    """topology: reduced (kidneys, liver, spleen, salivary glands, tumour, aggregated rest) or full (all tissues of Table A).

    lesion_mode: satellite (one aggregate tumour in the network) or coupled (every lesion in the network).
    balance_flows: lung flow equals the sum of tissue flows including tumours. False is Table A (F_LU = F).
    intra_return_volume_l: volume dividing the kidney cell outflow; None = V_intra (Eq. 6), 1.0 reproduces the shipped SBML.
    portal_outflow: liver outflow includes spleen and GI flow (Eq. 7). False reproduces the shipped SBML (full topology only).
    """
    ligand = ligand or Ligand()
    tissues, lung, body = tissue_table(patient)
    net = Network(ligand)
    net.volumes = {"art": body["Vart"], "ven": body["Vven"]}
    art = net.state("arteries", "blood", "arteries")
    ven = net.state("veins", "blood", "veins")
    prp = net.state("protein_bound", "blood", "protein_bound")
    urine = net.state("urine", "urine", "urine")
    net.state("released", "released", "released")
    net.transfer(ven, prp, ligand.k_pr)

    v_liver = None
    if topology == "full":
        p = tissues["liver"]
        v_liver, _ = net.add_tissue("liver", p, art, ven)
        p_spl, p_gi = tissues["spleen"], tissues["gi"]
        if portal_outflow:
            net.transfer(v_liver, ven, (p_spl["F"] + p_gi["F"]) / p["Vv"])
        net.add_tissue("spleen", p_spl, art, ven, drain=v_liver)
        net.add_tissue("gi", p_gi, art, ven, drain=v_liver)
        for name in ("salivary", "prostate", *NEGATIVE_TISSUES, "bone", "brain", "rest"):
            net.add_tissue(name, tissues[name], art, ven)
        v_lung = net.state("lung.vascular", "lung", "vascular")
        i_lung = net.state("lung.interstitial", "lung", "interstitial")
        net.transfer(v_lung, i_lung, lung["PS"] / lung["Vv"])
        net.transfer(i_lung, v_lung, lung["PS"] / lung["Vint"])
    elif topology == "reduced":
        pools = lump_rest(tissues, lung, n_rest_pools)
        p = tissues["liver"]
        p_spl, p_gi = tissues["spleen"], tissues["gi"]
        v_liver, _ = net.add_tissue("liver", p, art, ven, extra_inflow=p_gi["F"])
        net.transfer(v_liver, ven, p_spl["F"] / p["Vv"])
        net.add_tissue("spleen", p_spl, art, ven, drain=v_liver)
        net.add_tissue("salivary", tissues["salivary"], art, ven)
        for j, pool in enumerate(pools):
            net.add_tissue("rest" if len(pools) == 1 else f"rest_{j}", pool, art, ven)
    else:
        raise ValueError(topology)

    p = tissues["kidneys"]
    gfr_phi = patient.gfr_l_min * ligand.phi
    f_ex = gfr_phi * ligand.f_ex
    f_cell = gfr_phi - f_ex
    v_intra = (p["V"] - p["Vint"] - p["Vv"]) * 2 / 3
    k_v = net.state("kidneys.vascular", "kidneys", "vascular")
    k_i = net.state("kidneys.interstitial", "kidneys", "interstitial")
    k_c = net.state("kidneys.intracellular", "kidneys", "intracellular")
    net.transfer(art, k_v, p["F"] / body["Vart"])
    net.transfer(k_v, ven, p["F"] / p["Vv"])
    net.transfer(k_v, k_i, gfr_phi / p["Vv"])
    net.transfer(k_i, urine, f_ex / p["Vint"])
    net.transfer(k_i, k_c, f_cell / p["Vint"])
    net.transfer(k_c, k_v, f_cell / (v_intra if intra_return_volume_l is None else intra_return_volume_l))
    net.add_binding("kidneys", p, k_i)

    lesions = list(patient.lesions)
    total_flow = sum(x["F"] for x in tissues.values())
    net.satellites = []
    if lesions:
        hct = body["Hct"]
        if lesion_mode == "coupled":
            for j, lesion in enumerate(lesions):
                pl = lesion_parameters(lesion, hct)
                net.add_tissue(lesion.name or f"lesion_{j}", pl, art, ven)
                total_flow += pl["F"]
        elif lesion_mode == "satellite":
            agg = aggregate_lesions(lesions, hct, ligand.k_on, preserve_clearance)
            net.add_tissue("tumour", agg, art, ven)
            total_flow += agg["F"]
            net.satellites = lesions
        else:
            raise ValueError(lesion_mode)

    lung_flow = total_flow if balance_flows else body["F"]
    if topology == "full":
        net.transfer(ven, v_lung, lung_flow / body["Vven"])
        net.transfer(v_lung, art, lung_flow / lung["Vv"])
    else:
        net.transfer(ven, art, lung_flow / body["Vven"])
    net.body = body
    net.compile()
    return net


class _Trajectory:
    def __init__(self):
        self.parts = []

    def add(self, t, y, dy):
        self.parts.append((t[0], t[-1], CubicHermiteSpline(t, y, dy, axis=0)))

    def __call__(self, t):
        t = np.atleast_1d(t)
        first = self.parts[0][2](t)
        out = np.array(first)
        for start, _, spline in self.parts[1:]:
            mask = t >= start
            if mask.any():
                out[mask] = spline(t[mask])
        return out


def _make_rhs(net, input_rate, explicit, lam, f_hot):
    A0 = net.A
    i_int, i_b, r0, kv = net.i_int, net.i_bound, net.r0, net.kon_over_v
    ven = net.index["veins"]
    n = net.n
    if not explicit:
        def rhs(t, y):
            dy = A0 @ y
            rate = kv * y[i_int] * (r0 - y[i_b])
            dy[i_int] -= rate
            dy[i_b] += rate
            dy[ven] += input_rate
            return dy

        def jac(t, y):
            J = A0.copy()
            dint = kv * (r0 - y[i_b])
            dbound = kv * y[i_int]
            J[i_int, i_int] -= dint
            J[i_int, i_b] += dbound
            J[i_b, i_int] += dint
            J[i_b, i_b] -= dbound
            return J
        return rhs, jac

    block = np.zeros((2 * n, 2 * n))
    block[:n, :n] = A0
    block[n:, n:] = A0 - lam * np.eye(n)
    block[:n, n:] = lam * np.eye(n)
    ih, bh = i_int + n, i_b + n

    def rhs(t, y):
        dy = block @ y
        free = r0 - y[i_b] - y[bh]
        rc = kv * y[i_int] * free
        rh = kv * y[ih] * free
        dy[i_int] -= rc
        dy[i_b] += rc
        dy[ih] -= rh
        dy[bh] += rh
        hot_rate = input_rate * f_hot * math.exp(-lam * t)
        dy[ven + n] += hot_rate
        dy[ven] += input_rate - hot_rate
        return dy

    return rhs, None


def _output_grid(stop, times, knots):
    grid = np.concatenate([[0.0], np.geomspace(0.05, stop, 300), np.asarray(times, float), np.asarray(knots, float)])
    return np.unique(grid[grid <= stop])


def _exp_trapezoid(y, t, lam=0.0):
    """Integral of y(t) exp(-lam t), exact for piecewise exponentials, with the trapezoid where y is not positive."""
    dt = np.diff(t)
    y = y * np.exp(-lam * t)
    a, b = y[:-1], y[1:]
    out = 0.5 * (a + b) * dt
    ok = (a > 0) & (b > 0) & (np.abs(a - b) > 1e-9 * np.maximum(a, b))
    out[ok] = (b[ok] - a[ok]) / np.log(b[ok] / a[ok]) * dt[ok]
    return out.sum()


class Result:
    def __init__(self, patient, injection, net, time_min, y, hot=None, satellites=None):
        self.patient = patient
        self.injection = injection
        self.net = net
        self.time_min = time_min
        self.y = y
        self._hot = hot
        self.satellites = satellites or {}

    @property
    def time_h(self):
        return self.time_min / 60

    @property
    def labels(self):
        return [*self.net.groups, *self.satellites]

    def components(self, label):
        if label in self.satellites:
            return self.satellites[label]
        return {c: self.y[:, i] for c, i in self.net.groups[label].items()}

    def amount_nmol(self, label):
        """Total ligand (hot plus cold) in a structure."""
        return sum(self.components(label).values())

    def fraction(self, label):
        """Fraction of the injected activity, physical decay included."""
        return self.amount_nmol(label) / self.injection.total_nmol * np.exp(-self.injection.lambda_phys * self.time_min)

    def activity_mbq(self, label):
        return self.injection.activity_mbq * self.fraction(label)

    def hot_nmol(self, label):
        if label in self.satellites or self._hot is None:
            return self.injection.hot_nmol * self.fraction(label)
        return sum(self._hot[:, i] for i in self.net.groups[label].values())

    def percent_ia(self, label, t_min, decay_corrected=False):
        values = self.amount_nmol(label) / self.injection.total_nmol
        out = 100 * np.interp(t_min, self.time_min, values)
        return out if decay_corrected else out * np.exp(-self.injection.lambda_phys * np.asarray(t_min))

    def tiac_h(self, label, include_tail=False):
        amount = self.amount_nmol(label) / self.injection.total_nmol
        lam = self.injection.lambda_phys
        area = _exp_trapezoid(amount, self.time_min, lam)
        if include_tail and amount[-1] > 0:
            window = self.time_min >= self.time_min[-1] / 2
            rate = -np.polyfit(self.time_min[window], np.log(np.clip(amount[window], 1e-30, None)), 1)[0]
            area += amount[-1] * math.exp(-lam * self.time_min[-1]) / (rate + lam)
        return area / 60

    def t_eff_h(self, label, start_h=48.0, stop_h=168.0):
        fraction = self.fraction(label)
        mask = (self.time_h >= start_h) & (self.time_h <= stop_h) & (fraction > 0)
        slope = np.polyfit(self.time_h[mask], np.log(fraction[mask]), 1)[0]
        return math.log(2) / -slope

    def mass_balance_error(self):
        """Largest absolute deviation of total ligand from the amount injected so far, relative to the total injected."""
        inj = self.injection
        total = self.y.sum(axis=1)
        if inj.infusion_min > 0:
            expected = inj.total_nmol * np.clip(self.time_min / inj.infusion_min, 0, 1)
        else:
            expected = np.full_like(total, inj.total_nmol)
        return float(np.abs(total - expected).max() / inj.total_nmol)


def _segments(injection, stop):
    if injection.infusion_min > 0:
        return [(0.0, injection.infusion_min, injection.total_nmol / injection.infusion_min), (injection.infusion_min, stop, 0.0)]
    return [(0.0, stop, 0.0)]


def solve_network(net, injection, stop_min, explicit=False, method="LSODA", rtol=1e-5, atol_factor=1e-8):
    """Integrate. Returns the Hermite interpolant on the solver steps, exact within the tolerance at any output time."""
    lam = injection.lambda_phys
    n = net.n * (2 if explicit else 1)
    y = np.zeros(n)
    f_hot = injection.hot_nmol / injection.total_nmol
    if explicit and f_hot > 1:
        raise ValueError("hot amount above the total amount")
    if injection.infusion_min <= 0:
        ven = net.index["veins"]
        if explicit:
            y[ven] = injection.total_nmol - injection.hot_nmol
            y[ven + net.n] = injection.hot_nmol
        else:
            y[ven] = injection.total_nmol
    trajectory = _Trajectory()
    n_steps = 0
    for start, stop, rate in _segments(injection, stop_min):
        rhs, jac = _make_rhs(net, rate, explicit, lam, f_hot)
        kwargs = {"jac": jac} if jac is not None else {}
        try:
            sol = solve_ivp(rhs, (start, stop), y, method=method, rtol=rtol, atol=atol_factor * injection.total_nmol, **kwargs)
            failed = not sol.success
        except RuntimeError:
            failed = True
        if failed:
            warnings.warn(f"{method} failed on the network: retried with BDF")
            sol = solve_ivp(rhs, (start, stop), y, method="BDF", rtol=rtol, atol=atol_factor * injection.total_nmol, **kwargs)
        if not sol.success:
            raise RuntimeError(sol.message)
        dy = np.array([rhs(t, s) for t, s in zip(sol.t, sol.y.T)])
        trajectory.add(sol.t, sol.y.T, dy)
        y = sol.y[:, -1]
        n_steps += len(sol.t)
    return trajectory, n_steps


def solve_satellites(net, trajectory, injection, lesions, grid, method="LSODA", rtol=1e-5, atol_factor=1e-8):
    """Each lesion driven by the arterial concentration. No feedback on plasma. One tridiagonal system for all lesions."""
    hct = net.body["Hct"]
    parts = [lesion_parameters(x, hct) for x in lesions]
    F, Vv, Vint, PS, R0, lrel = (np.array([p[k] for p in parts]) for k in ("F", "Vv", "Vint", "PS", "R0", "lrel"))
    lig = net.ligand
    n = len(parts)
    kv = lig.k_on / Vint
    art = net.index["arteries"]
    knots = [t0 for t0, _, _ in trajectory.parts]
    splines = [s for _, _, s in trajectory.parts]
    v_art = net.volumes["art"]
    cf = [(s.c[:, :, art], s.x) for s in splines]

    def c_art(t):
        k = 0 if len(knots) == 1 or t < knots[1] else 1
        c, x = cf[k]
        i = min(max(np.searchsorted(x, t, side="right") - 1, 0), len(x) - 2)
        d = t - x[i]
        return ((c[0, i] * d + c[1, i]) * d + c[2, i]) * d + c[3, i]

    a_vv = (F + PS) / Vv
    a_iv = PS / Vint
    a_vi = PS / Vv
    koff, lint = lig.k_off, lig.lambda_int

    def rhs(t, y):
        v, i, b, c = y[0::4], y[1::4], y[2::4], y[3::4]
        rate = kv * i * (R0 - b)
        dy = np.empty_like(y)
        dy[0::4] = F * c_art(t) / v_art - a_vv * v + a_iv * i
        dy[1::4] = a_vi * v - a_iv * i - rate + koff * b
        dy[2::4] = rate - (koff + lint) * b
        dy[3::4] = lint * b - lrel * c
        return dy

    def jac(t, y):
        i, b = y[1::4], y[2::4]
        d_i = kv * (R0 - b)
        d_b = kv * i
        J = np.zeros((3, 4 * n))
        J[1, 0::4] = -a_vv
        J[1, 1::4] = -a_iv - d_i
        J[1, 2::4] = -d_b - koff - lint
        J[1, 3::4] = -lrel
        J[0, 1::4] = a_iv
        J[0, 2::4] = d_b + koff
        J[2, 0::4] = a_vi
        J[2, 1::4] = d_i
        J[2, 2::4] = lint
        return J

    y0 = np.zeros(4 * n)
    if method == "LSODA":
        options = dict(jac=jac, lband=1, uband=1)
    else:
        options = dict(jac_sparsity=diags([np.ones(4 * n - 1), np.ones(4 * n), np.ones(4 * n - 1)], [-1, 0, 1]))
    try:
        sol = solve_ivp(rhs, (grid[0], grid[-1]), y0, method=method, rtol=rtol, atol=atol_factor * injection.total_nmol, t_eval=grid, **options)
        failed = not sol.success
    except RuntimeError:
        failed = True
    if failed:
        warnings.warn(f"{method} failed on the satellites of {len(lesions)} lesions: retried with BDF")
        sparsity = diags([np.ones(4 * n - 1), np.ones(4 * n), np.ones(4 * n - 1)], [-1, 0, 1])
        sol = solve_ivp(rhs, (grid[0], grid[-1]), y0, method="BDF", jac_sparsity=sparsity, rtol=rtol, atol=atol_factor * injection.total_nmol, t_eval=grid)
    if not sol.success:
        raise RuntimeError(sol.message)
    out = {}
    for j, lesion in enumerate(lesions):
        block = sol.y[4 * j: 4 * j + 4]
        out[lesion.name or f"lesion_{j}"] = dict(zip(("vascular", "interstitial", "bound", "internalized"), block))
    return out


def simulate(patient, injection, stop_min=30000.0, ligand=None, topology="reduced", lesion_mode="auto", balance_flows=True,
             intra_return_volume_l=None, portal_outflow=True, n_rest_pools=2, preserve_clearance=True, explicit_hot_cold=False, times_min=(), rtol=1e-5, atol_factor=1e-8, method="LSODA"):
    """Hot and cold have the same kinetics, so by default only total ligand is integrated and hot = f0 T exp(-lambda t).

    explicit_hot_cold integrates the two species, for verification.
    lesion_mode auto: satellites up to COUPLED_TLV_ML of total lesion volume, coupled above it.
    """
    if lesion_mode == "auto":
        lesion_mode = "coupled" if sum(x.volume_ml for x in patient.lesions) > COUPLED_TLV_ML else "satellite"
    net = build_network(patient, ligand, topology, lesion_mode, balance_flows, intra_return_volume_l, portal_outflow, n_rest_pools, preserve_clearance)
    trajectory, _ = solve_network(net, injection, stop_min, explicit_hot_cold, method=method, rtol=rtol, atol_factor=atol_factor)
    grid = _output_grid(stop_min, times_min, [injection.infusion_min])
    y = trajectory(grid)
    hot = None
    if explicit_hot_cold:
        hot = y[:, net.n:]
        y = y[:, :net.n] + hot
    satellites = solve_satellites(net, trajectory, injection, net.satellites, grid, method=method, rtol=rtol, atol_factor=atol_factor) if net.satellites else {}
    return Result(patient, injection, net, grid, y, hot, satellites)


def self_dose_gy(result, label, s_gy_per_mbq_min):
    """D = A0 x time-integrated activity coefficient x S (Eqs. 14-15). Kletting's T = 30000 min with stop_min at that value."""
    return result.injection.activity_mbq * result.tiac_h(label) * 60 * s_gy_per_mbq_min


S_VALUES_ML = np.array([0.5, 1, 1.5, 2, 3, 4, 13, 17, 21, 29, 34, 52, 54])
S_VALUES_GY = np.array([2.77e-3, 1.40e-3, 9.00e-4, 7.02e-4, 4.50e-4, 3.52e-4, 1.10e-4, 8.30e-5, 6.90e-5, 4.80e-5, 4.30e-5, 2.67e-5, 2.65e-5])
S_KIDNEY = 4.82e-6


def s_sphere(volume_ml):
    """Table C, logarithmic interpolation (Kletting 2016, S-values)."""
    return float(np.exp(np.interp(np.log(volume_ml), np.log(S_VALUES_ML), np.log(S_VALUES_GY))))
