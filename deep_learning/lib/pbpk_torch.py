"""
Differentiable PSMA PBPK in torch (method M5): the model of src/pbpk.py (our reproduction of the Kletting 2016 S1 File
equations; the network of compartments is assembled by pbpk.build_network) integrated with a fixed-step, linearly
implicit second-order scheme (ROS2), so that the source activities and TIACs of a therapy are differentiable in the
receptor densities and release rates of the patient. Source curves and TIACs are built as in
pbpk_cohort.simulate_therapy. Local only: it needs src/ on the path. Times in minutes inside, hours in the outputs.
"""
import math

import numpy as np
import torch

GAMMA = 1.0 + 1.0 / math.sqrt(2.0)
VOID_INTERVAL_H = 4.0
SOURCE_TISSUES = dict(kidneys_functional="kidneys", liver="liver", spleen="spleen", salivary_glands="salivary",
                      gi_tract="gi")
SKELETON = ("red_marrow", "bone")


def _exp_trapezoid(y, t, lam):
    """Integral of y(t) exp(-lam t): exact for piecewise exponentials, trapezoid where y is not positive (as pbpk._exp_trapezoid)."""
    y = y * torch.exp(-lam * t)
    dt = t[1:] - t[:-1]
    a, b = y[:-1], y[1:]
    ok = (a > 0) & (b > 0) & ((a - b).abs() > 1e-9 * torch.maximum(a, b))
    safe_a, safe_b = torch.where(ok, a, torch.ones_like(a)), torch.where(ok, b, 2 * torch.ones_like(b))
    exact = (safe_b - safe_a) / torch.log(safe_b / safe_a) * dt
    return torch.where(ok, exact, 0.5 * (a + b) * dt).sum()


def _slope(x, y):
    x = x - x.mean()
    return (x * (y - y.mean())).sum() / (x * x).sum()


class TorchPBPK:
    """
    One patient and one injection. solve(density, release, lesion_density, lesion_release) takes log multipliers of
    the receptor amount and of the release rate, by tissue label (dict) and by lesion (tensor), all 0 at the patient
    given to the constructor, and returns the source curves on self.time_h and the TIACs. With satellite lesions
    (total lesion volume up to pbpk.COUPLED_TLV_ML) the aggregate tumour in the network, the plasma sink, stays at
    the constructor's lesions; above it every lesion is in the network.
    """

    def __init__(self, pbpk, patient, injection, spine_ml, blood_ml, substeps=2, stop_min=30000.0, topology="full"):
        self.P, self.patient, self.injection = pbpk, patient, injection
        tlv = sum(x.volume_ml for x in patient.lesions)
        self.coupled = tlv > pbpk.COUPLED_TLV_ML
        net = pbpk.build_network(patient, None, topology, "coupled" if self.coupled else "satellite")
        self.net, self.n, self.substeps = net, net.n, substeps
        self.released = net.index["released"]
        t = lambda x, dtype=torch.float64: torch.as_tensor(np.asarray(x), dtype=dtype)
        self.i_int, self.i_bound = t(net.i_int, torch.long), t(net.i_bound, torch.long)
        self.r0, self.kv = t(net.r0), t(net.kon_over_v)
        bound_label = {groups["bound"]: label for label, groups in net.groups.items() if "bound" in groups}
        self.receptor_labels = [bound_label[int(i)] for i in net.i_bound]
        self.i_cell = t([net.groups[label]["internalized"] for label in self.receptor_labels], torch.long)
        base = {source: rate for source, target, rate in net.transfers if target == self.released}
        self.release = t([base[int(i)] for i in self.i_cell])
        A = net.A.copy()
        for c, rate in zip(self.i_cell.tolist(), self.release.tolist()):
            A[c, c] += rate
            A[self.released, c] -= rate
        gfr = np.zeros_like(A)
        pairs = (("kidneys.vascular", "kidneys.interstitial"), ("kidneys.interstitial", "urine"),
                 ("kidneys.interstitial", "kidneys.intracellular"), ("kidneys.intracellular", "kidneys.vascular"))
        wanted = {(net.index[a], net.index[b]) for a, b in pairs}
        for source, target, rate in net.transfers:
            if (source, target) in wanted:
                gfr[source, source] -= rate
                gfr[target, source] += rate
        self.A_fixed, self.A_gfr = t(A - gfr), t(gfr)
        self.grid = t(np.unique(np.concatenate([[0.0], np.geomspace(0.05, stop_min, 300)])))
        self.time_h = self.grid / 60.0
        self.lam = injection.lambda_phys
        self.decay = torch.exp(-self.lam * self.grid)
        self.total = injection.total_nmol
        body = net.body
        self.hct, self.blood_ml, self.v_art = body["Hct"], blood_ml, net.volumes["art"]
        self.vessels_ml = (body["Vart"] + body["Vven"]) * 1000
        tissues, _, _ = pbpk.tissue_table(patient)
        skeleton_ml = sum(tissues[k]["V"] for k in SKELETON if k in net.groups) * 1000
        self.spine_share = min(spine_ml / skeleton_ml, 1.0) if skeleton_ml else 0.0
        self.tissue_groups = [k for k in net.groups if k not in ("urine", "released", "tumour") and not k.startswith("lesion_")]
        self.n_lesions = len(patient.lesions)
        if self.n_lesions and not self.coupled:
            parts = [pbpk.lesion_parameters(x, self.hct) for x in patient.lesions]
            self.sat = {k: t([p[k] for p in parts]) for k in ("F", "Vv", "Vint", "PS", "R0", "lrel")}

    def _matrix(self, release, gfr=None):
        A = self.A_fixed + self.A_gfr * (torch.exp(gfr) if gfr is not None else 1.0)
        A[self.i_cell, self.i_cell] = A[self.i_cell, self.i_cell] - release
        A[self.released, self.i_cell] = A[self.released, self.i_cell] + release
        return A

    def _network(self, A, r0):
        eye = torch.eye(self.n, dtype=torch.float64)
        i, b, kv = self.i_int, self.i_bound, self.kv

        def f(y):
            rate = kv * y[i] * (r0 - y[b])
            return (A @ y).index_add(0, i, -rate).index_add(0, b, rate)

        y = torch.zeros(self.n, dtype=torch.float64)
        y = y.index_add(0, torch.tensor([self.net.index["veins"]]), torch.tensor([float(self.total)], dtype=torch.float64))
        out = [y]
        for start, stop in zip(self.grid[:-1].tolist(), self.grid[1:].tolist()):
            h = (stop - start) / self.substeps
            for _ in range(self.substeps):
                free, at_site = kv * (r0 - y[b]), kv * y[i]
                J = A.clone()
                J[i, i] = J[i, i] - free
                J[i, b] = J[i, b] + at_site
                J[b, i] = J[b, i] + free
                J[b, b] = J[b, b] - at_site
                lu, pivots = torch.linalg.lu_factor(eye - GAMMA * h * J)
                k1 = torch.linalg.lu_solve(lu, pivots, f(y)[:, None])[:, 0]
                k2 = torch.linalg.lu_solve(lu, pivots, (f(y + h * k1) - 2.0 * k1)[:, None])[:, 0]
                y = y + 1.5 * h * k1 + 0.5 * h * k2
            out.append(y)
        return torch.stack(out)

    def _satellites(self, arterial, r0, release, flow):
        """Every lesion driven by the arterial concentration (no feedback on plasma): (times, lesions, 4) amounts."""
        s, lig = self.sat, self.net.ligand
        n = self.n_lesions
        kv = lig.k_on / s["Vint"]
        a_vv, a_iv, a_vi = (flow + s["PS"]) / s["Vv"], s["PS"] / s["Vint"], s["PS"] / s["Vv"]
        eye = torch.eye(4, dtype=torch.float64).expand(n, 4, 4)
        unit = lambda r, c: torch.zeros(4, 4, dtype=torch.float64).index_put((torch.tensor(r), torch.tensor(c)),
                                                                           torch.tensor(1.0, dtype=torch.float64))
        constant = torch.zeros(n, 4, 4, dtype=torch.float64)
        constant[:, 0, 1], constant[:, 1, 0], constant[:, 1, 1], constant[:, 1, 2] = a_iv, a_vi, -a_iv, lig.k_off
        constant[:, 2, 2] = -(lig.k_off + lig.lambda_int)
        constant[:, 3, 2] = lig.lambda_int
        base = constant - a_vv[:, None, None] * unit(0, 0) - release[:, None, None] * unit(3, 3)

        def f(y, c_art):
            rate = kv * y[:, 1] * (r0 - y[:, 2])
            dy = (base @ y[:, :, None])[:, :, 0]
            return dy + torch.stack([flow * c_art / self.v_art, -rate, rate,
                                     torch.zeros(n, dtype=torch.float64)], dim=1)

        y = torch.zeros(n, 4, dtype=torch.float64)
        out = [y]
        for k in range(self.grid.numel() - 1):
            start, stop = float(self.grid[k]), float(self.grid[k + 1])
            h = (stop - start) / self.substeps
            for m in range(self.substeps):
                c0 = arterial[k] + (arterial[k + 1] - arterial[k]) * m / self.substeps
                c1 = arterial[k] + (arterial[k + 1] - arterial[k]) * (m + 1) / self.substeps
                J = base.clone()
                free, at_site = kv * (r0 - y[:, 2]), kv * y[:, 1]
                J[:, 1, 1] = J[:, 1, 1] - free
                J[:, 1, 2] = J[:, 1, 2] + at_site
                J[:, 2, 1] = J[:, 2, 1] + free
                J[:, 2, 2] = J[:, 2, 2] - at_site
                lu, pivots = torch.linalg.lu_factor(eye - GAMMA * h * J)
                k1 = torch.linalg.lu_solve(lu, pivots, f(y, c0)[:, :, None])[:, :, 0]
                k2 = torch.linalg.lu_solve(lu, pivots, (f(y + h * k1, c1) - 2.0 * k1)[:, :, None])[:, :, 0]
                y = y + 1.5 * h * k1 + 0.5 * h * k2
            out.append(y)
        return torch.stack(out)

    def _tiac(self, amount):
        """TIAC in h of an amount curve (nmol), with the exponential tail fitted on the last half of the time span."""
        fraction = amount / self.total
        area = _exp_trapezoid(fraction, self.grid, self.lam)
        window = self.grid >= self.grid[-1] / 2
        rate = -_slope(self.grid[window], torch.log(fraction[window].clamp_min(1e-30)))
        tail = fraction[-1] * math.exp(-self.lam * float(self.grid[-1])) / (rate + self.lam)
        return (area + torch.where(fraction[-1] > 0, tail, torch.zeros_like(tail))) / 60.0

    def solve(self, density=None, release=None, lesion_density=None, lesion_release=None, gfr=None, lesion_flow=None):
        zero = torch.zeros((), dtype=torch.float64)
        scale = lambda table, label: torch.exp((table or {}).get(label, zero))
        r0 = self.r0 * torch.stack([scale(density, label) for label in self.receptor_labels])
        rel = self.release * torch.stack([scale(release, label) for label in self.receptor_labels])
        if self.coupled and self.n_lesions:
            where = {label: k for k, label in enumerate(self.receptor_labels)}
            index = torch.tensor([where[f"lesion_{j}"] for j in range(self.n_lesions)])
            if lesion_density is not None:
                r0 = r0.index_put((index,), r0[index] * torch.exp(lesion_density))
            if lesion_release is not None:
                rel = rel.index_put((index,), rel[index] * torch.exp(lesion_release))
        y = self._network(self._matrix(rel, gfr), r0)
        group = lambda label: sum(y[:, i] for i in self.net.groups[label].values())
        curve = {k: group(k) / self.total * self.decay for k in self.tissue_groups}
        tia = {k: self._tiac(group(k)) for k in self.tissue_groups}
        index = self.net.groups["blood"]
        serum = (y[:, index["arteries"]] + y[:, index["veins"]]) / self.total * self.decay / self.vessels_ml
        blood = serum * (1 - self.hct) * self.blood_ml
        skeleton = [k for k in SKELETON if k in tia]
        urine = group("urine") / self.total
        urine_decays = ((urine[1:] - urine[:-1]) * 0.5 * (self.decay[1:] + self.decay[:-1])).sum()
        sources = {name: curve[k] for name, k in SOURCE_TISSUES.items()}
        tiac = {name: tia[k] for name, k in SOURCE_TISSUES.items()}
        sources.update(bladder=torch.gradient(urine, spacing=(self.time_h,))[0] * self.decay * VOID_INTERVAL_H / 2,
                       blood_pool=blood, spine=sum(curve[k] for k in skeleton) * self.spine_share)
        tiac.update(bladder=VOID_INTERVAL_H / 2 * urine_decays,
                    blood_pool=(0.5 * (blood[1:] + blood[:-1]) * (self.time_h[1:] - self.time_h[:-1])).sum(),
                    spine=sum(tia[k] for k in skeleton) * self.spine_share)
        named = list(sources)
        sources["rest_of_body"] = sum(curve.values()) - sum(sources[k] for k in named if k != "bladder")
        tiac["rest_of_body"] = sum(tia.values()) - sum(tiac[k] for k in named if k != "bladder")
        if self.n_lesions:
            if self.coupled:
                amounts = [group(f"lesion_{j}") for j in range(self.n_lesions)]
            else:
                r0_les = self.sat["R0"] * (torch.exp(lesion_density) if lesion_density is not None else 1.0)
                rel_les = self.sat["lrel"] * (torch.exp(lesion_release) if lesion_release is not None else 1.0)
                flow = self.sat["F"] * (torch.exp(lesion_flow) if lesion_flow is not None else 1.0)
                sat = self._satellites(y[:, self.net.index["arteries"]], r0_les, rel_les, flow).sum(dim=2)
                amounts = [sat[:, j] for j in range(self.n_lesions)]
            for j, amount in enumerate(amounts):
                sources[f"lesion_{j}"] = amount / self.total * self.decay
                tiac[f"lesion_{j}"] = self._tiac(amount)
        return sources, tiac

    def at(self, curve, t_h):
        """A source curve at the times t_h (hours), interpolated linearly in the log where positive."""
        t_h = torch.as_tensor(t_h, dtype=torch.float64)
        k = torch.searchsorted(self.time_h, t_h).clamp(1, self.time_h.numel() - 1)
        w = (t_h - self.time_h[k - 1]) / (self.time_h[k] - self.time_h[k - 1])
        a, b = curve[k - 1], curve[k]
        positive = (a > 0) & (b > 0)
        log = torch.exp(torch.log(torch.where(positive, a, torch.ones_like(a))) * (1 - w)
                        + torch.log(torch.where(positive, b, torch.ones_like(b))) * w)
        return torch.where(positive, log, a * (1 - w) + b * w)


RELEASE_GROUPS = dict(kidneys=("kidneys", "gi", "prostate"), liver=("liver", "spleen"), salivary=("salivary",))
DENSITY_ORGANS = ("kidneys", "liver", "spleen", "salivary")


class Parameters:
    """
    Patient-level parameters of M5 as log multipliers around the patient of the model (its prior mean), in a flat
    vector: receptor amount of kidneys, liver, spleen, glands (density x agent factor: only their product enters);
    release rate of the three groups of the cohort (kidneys with GI and prostate; liver with spleen; glands); GFR;
    then receptor amount, release rate and flow of every lesion (the flow only for satellite lesions).
    """

    def __init__(self, n_lesions):
        self.n_lesions = n_lesions
        self.n_organ = len(DENSITY_ORGANS) + len(RELEASE_GROUPS) + 1
        self.size = self.n_organ + 3 * n_lesions

    def unpack(self, vector):
        """Keyword arguments of TorchPBPK.solve."""
        d, r, n, o = len(DENSITY_ORGANS), len(RELEASE_GROUPS), self.n_lesions, self.n_organ
        out = dict(density={k: vector[j] for j, k in enumerate(DENSITY_ORGANS)},
                   release={label: vector[d + j] for j, labels in enumerate(RELEASE_GROUPS.values()) for label in labels},
                   gfr=vector[d + r])
        if n:
            out.update(lesion_density=vector[o:o + n], lesion_release=vector[o + n:o + 2 * n],
                       lesion_flow=vector[o + 2 * n:])
        return out


def expected_voi(model, parameters, vector, sources, R, t_h):
    """Expected VOI activities of the SPECT at t_h for a parameter vector, and the TIACs: (expected, {source: TIAC})."""
    curves, tiac = model.solve(**parameters.unpack(vector))
    activity = torch.stack([model.at(curves[s], t_h) for s in sources])
    return R @ activity, tiac


def fit(model, measured, R, sources, t_h, prior_sd, voi_sd, steps=40, lr=0.1, method="lbfgs"):
    """
    Maximum a posteriori parameters: Gaussian likelihood of the measured VOI activities (SD = voi_sd(expected)) and Gaussian
    prior on the log multipliers (prior_sd, same layout as the parameter vector), minimised with the gradient of the
    differentiable model. method 'lbfgs': L-BFGS with a strong-Wolfe line search, at most `steps` evaluations (the
    likelihood is much sharper than the prior, and fixed-size Adam steps overshoot); 'adam': `steps` steps of size lr.
    Returns (vector, history): per evaluation the loss and the log TIAC of every source; the vector is the best one seen.
    """
    parameters = Parameters(model.n_lesions)
    vector = torch.zeros(parameters.size, dtype=torch.float64, requires_grad=True)
    history, best = [], dict(loss=float("inf"), vector=vector.detach().clone())

    def evaluate():
        expected, tiac = expected_voi(model, parameters, vector, sources, R, t_h)
        sd = voi_sd(expected)
        loss = (0.5 * ((measured - expected) / sd) ** 2 + torch.log(sd)).sum() + 0.5 * ((vector / prior_sd) ** 2).sum()
        history.append(dict(step=len(history), loss=float(loss), **{s: float(torch.log(tiac[s])) for s in sources}))
        if float(loss) < best["loss"]:
            best.update(loss=float(loss), vector=vector.detach().clone())
        return loss

    if method == "lbfgs":
        optimiser = torch.optim.LBFGS([vector], lr=1.0, max_iter=steps, max_eval=steps, history_size=20,
                                      tolerance_grad=1e-6, tolerance_change=1e-10, line_search_fn="strong_wolfe")

        def closure():
            optimiser.zero_grad()
            loss = evaluate()
            loss.backward()
            return loss

        optimiser.step(closure)
    else:
        optimiser = torch.optim.Adam([vector], lr=lr)
        for _ in range(steps):
            optimiser.zero_grad()
            evaluate().backward()
            optimiser.step()
        with torch.no_grad():
            evaluate()
    return best["vector"], history


@torch.no_grad()
def laplace(model, vector, measured, R, sources, t_h, prior_sd, voi_sd, eps=0.05):
    """
    SD of the log TIAC of every source from a Gauss-Newton Laplace approximation at the optimum, by finite
    differences: one solve per organ parameter, and three for the lesions (all densities, all release rates, all
    flows), each lesion parameter being taken to act on its own lesion only. Returns ({source: log TIAC}, {source: SD}).
    """
    parameters = Parameters(model.n_lesions)
    expected, tiac = expected_voi(model, parameters, vector, sources, R, t_h)
    sd = voi_sd(expected)
    log_tiac = torch.stack([torch.log(tiac[s]) for s in sources])
    n_organ, n = parameters.n_organ, model.n_lesions
    organ_index = [k for k, s in enumerate(sources) if not s.startswith("lesion_")]
    lesion_index = [k for k, s in enumerate(sources) if s.startswith("lesion_")]
    lesion_number = [int(sources[k].split("_")[1]) for k in lesion_index]

    def shifted(columns):
        other = vector.clone()
        other[columns] += eps
        e, t = expected_voi(model, parameters, other, sources, R, t_h)
        return (e - expected) / eps, (torch.stack([torch.log(t[s]) for s in sources]) - log_tiac) / eps

    jac_e, jac_t = zip(*[shifted([j]) for j in range(n_organ)])
    J, G = torch.stack(jac_e, dim=1), torch.stack(jac_t, dim=1)
    weight = 1.0 / sd ** 2
    H = (J * weight[:, None]).T @ J + torch.diag(1.0 / prior_sd[:n_organ] ** 2)
    covariance = torch.linalg.inv(H)
    variance = torch.einsum("sj,jk,sk->s", G, covariance, G)
    if n:
        parts = [shifted(list(range(n_organ + block * n, n_organ + (block + 1) * n))) for block in (0, 1, 2)]
        for k, j in zip(lesion_index, lesion_number):
            Jl = torch.stack([part[0][k] for part in parts])
            Gl = torch.stack([part[1][k] for part in parts])
            prior = torch.stack([prior_sd[n_organ + block * n + j] for block in (0, 1, 2)])
            Hl = weight[k] * torch.outer(Jl, Jl) + torch.diag(1.0 / prior ** 2)
            variance[k] = variance[k] + Gl @ torch.linalg.inv(Hl) @ Gl
    return ({s: float(log_tiac[k]) for k, s in enumerate(sources)},
            {s: float(torch.sqrt(variance[k].clamp_min(0.0))) for k, s in enumerate(sources)})
