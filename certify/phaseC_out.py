"""Phase C outputs: invariant checks, certs.csv (brief §10 schema plus err_clamped), graph.csv, timings,
curves, and the Gate C summary tables. Works on the dict returned by certify.phaseC.run_dataset.

Tightness ratios follow item C-iii of the Phase A review: a ratio is reported only where the true error of
the block exceeds 10x the dataset's pose floor (and is >= 1e-12, brief §5); elsewhere it is NaN and the
absolute E_i and e_i are reported instead.
"""

import csv
import gzip
import json

import numpy as np

from certify import radii
from certify.phaseC import OUT, mu_pair, verify_cov_block

TOL = 1e-9


def viol(u, E):
    u, E = np.broadcast_arrays(np.asarray(u, float), np.asarray(E, float))
    den = np.maximum(np.maximum(np.abs(E), np.abs(u)), 1e-300)
    with np.errstate(invalid="ignore"):
        v = (u - E) / den
    v = np.array(v, float)
    v[~np.isfinite(E) & (E > 0)] = -1.0
    v[np.isnan(E)] = np.nan
    return v


def psd_viol(C, X, Y):
    """Relative violation of X <= Y in the metric C C^T."""
    A, B = C.T @ X @ C, C.T @ Y @ C
    d = np.linalg.eigvalsh(0.5 * ((B - A) + (B - A).T))[0]
    sc = max(np.linalg.eigvalsh(0.5 * (A + A.T))[-1], np.linalg.eigvalsh(0.5 * (B + B.T))[-1], 1e-300)
    return float(-d / sc)


class Inv:
    def __init__(self, R):
        self.R, self.rows = R, []

    def add(self, name, v, where=None, tol=TOL):
        v = np.asarray(v, float)
        fin = np.isfinite(v)
        if not fin.any():
            self.rows.append(dict(check=name, n=0, worst=np.nan, tol=tol, ok=True, where=""))
            return
        k = int(np.nanargmax(np.where(fin, v, -np.inf)))
        w = ""
        if where is not None:
            w = where(np.unravel_index(k, v.shape))
        self.rows.append(dict(check=name, n=int(fin.sum()), worst=float(v.ravel()[k]), tol=tol,
                              ok=bool(v.ravel()[k] <= tol), where=w))


def invariant_checks(R):
    m, err, met = R["model"], R["err"], R["met"]
    iters = R["iters"]
    inv = Inv(R)
    bw = lambda idx: "block %s (%s) iter %s" % (m.block_ids[idx[0]], m.kinds[idx[0]], iters[idx[1]] if len(idx) > 1 else "")
    # certificates vs the oracle error
    for nm in ("E_C1", "E_C2"):
        if np.isfinite(R["cols"][nm]).any():
            inv.add("%s >= err" % nm, viol(err, R["cols"][nm]), bw)
    for nm, (M, nu) in R["onestep"].items():
        inv.add("%s one-step u <= nu + M u" % nm, viol(err, nu + M @ err), bw)
    for K, c in R["c3"].items():
        inv.add("C3 one-step u <= s + A u K=%s" % K, viol(err, c["s"] + c["A"] @ err), bw)
        if c["E"] is not None:
            inv.add("E_C3 >= err K=%s" % K, viol(err, c["E"]), bw)
            inv.add("C3 monotone-iteration check K=%s" % K, np.array([-1.0 if c["iterations"] is not None else np.nan]))
    for w, W in R["c4"].items():
        for K, KR in W["K"].items():
            inv.add("C4 one-step (supernode) w=%d K=%d" % (w, K), viol(W["u_sn"], KR["s"] + KR["A"] @ W["u_sn"]))
            inv.add("Prop3.6 coefficient <= bound w=%d K=%d" % (w, K), viol(KR["coeff"], KR["coeff_bound"]))
            if KR["c"] is not None:
                inv.add("C4 c* >= ||e~_I|| (supernode) w=%d K=%d" % (w, K), viol(W["u_sn"], KR["c"]))
                inv.add("E_C4 >= ||e~_i|| (per block) w=%d K=%d" % (w, K), viol(W["err_clamped"], KR["E_per"]), bw)
            if "E_hub" in KR:
                inv.add("E_hub >= err w=%d K=%d" % (w, K), viol(err, KR["E_hub"]), bw)
            if "E_S" in KR:
                inv.add("Lemma4.4 ||e_S|| <= E_S w=%d K=%d" % (w, K), viol(KR["eS_norm"], KR["E_S"]))
                inv.add("Lemma4.5 phi <= phibar (supernode) w=%d K=%d" % (w, K), viol(KR["phi_sn_oracle"], KR["phibar_sn"]))
                inv.add("Lemma4.5 phi <= phibar (per block) w=%d K=%d" % (w, K), viol(KR["phi_blk"], KR["phibar_blk"]))
                ok = np.isfinite(KR["mu_hi_hubaware"])
                inv.add("Prop4.7 lam_max <= mu_hi_hubaware w=%d K=%d" % (w, K),
                        viol(R["lam"][ok, 1], KR["mu_hi_hubaware"][ok]))
    # covariance bounds vs the oracle marginals; flagged blocks are re-verified in extended precision
    n = m.n_blocks
    Sig = R["Sig"]
    recheck = {}

    def cov_rows(name, pairs, lower=True):
        v = np.full(n, np.nan)
        for i, B in pairs:
            if B is None:
                continue
            v[i] = psd_viol(met.C[i], B, Sig[i]) if lower else psd_viol(met.C[i], Sig[i], B)
        bad = np.flatnonzero(v > TOL)
        for i in bad:  # re-verify with an extended-precision refined marginal
            if i not in recheck:
                recheck[i] = verify_cov_block(m, R["sysm"], int(i))
            B = dict(pairs)[i]
            v[i] = psd_viol(met.C[i], B, recheck[i]) if lower else psd_viol(met.C[i], recheck[i], B)
        inv.add(name, v, lambda idx: "block %s (%s)" % (m.block_ids[idx[0]], m.kinds[idx[0]]))
        return len(bad)

    D = m.diag_blocks()
    cov_rows("trivial lower <= Sigma", [(i, np.linalg.inv(D[i])) for i in range(n)])
    prevL, prevU = None, None
    for K, c in R["c3"].items():
        if not isinstance(K, int):
            continue
        cov_rows("patch lower <= Sigma K=%d" % K, list(enumerate(c["inv_ii"])))
        cov_rows("cut upper >= Sigma K=%d" % K, list(enumerate(c["cut_ii"])), lower=False)
        if prevL is not None:
            inv.add("patch lower monotone K=%d" % K, [psd_viol(met.C[i], prevL[i], c["inv_ii"][i]) for i in range(n)])
            inv.add("cut upper monotone K=%d" % K, [psd_viol(met.C[i], c["cut_ii"][i], prevU[i]) for i in range(n)
                                                   if c["cut_ii"][i] is not None and prevU[i] is not None])
        prevL, prevU = c["inv_ii"], c["cut_ii"]
    if "3mix" in R["c3"]:
        c = R["c3"]["3mix"]
        cov_rows("patch lower <= Sigma K=3 (sample)", list(c["inv_ii"].items()))
        cov_rows("cut upper >= Sigma K=3 (sample)", list(c["cut_ii"].items()), lower=False)
    for ell in (3, 5):
        cov_rows("krylov lower <= Sigma l=%d" % ell, list(R["kry"][ell].items()))
    inv.add("krylov monotone l=3->5", [psd_viol(met.C[i], R["kry"][3][i], R["kry"][5][i]) for i in R["kry"][5]])
    # oracle consistency: H^-1 g against x - xbar
    R["res"]["oracle_err_consistency"] = float(np.nanmax(np.abs(R["err"] - R["err_xdiff"]) / np.maximum(R["err"], 1e-300)))
    R["res"]["cov_rechecked_blocks"] = len(recheck)
    # global energy bounds
    g = R["glob"]
    inv.add("global lb1 <= g^T H^-1 g", viol(g[:, 0], g[:, 2]))
    inv.add("global lb5 <= g^T H^-1 g", viol(g[:, 1], g[:, 2]))
    return inv.rows


def per_block_bounds(R, K):
    """mu_lo (trivial, patch K, krylov 3/5), mu_hi (cut K, hub-aware K), oracle lambda_min/max, certifiable, radii."""
    m, met = R["model"], R["met"]
    n = m.n_blocks
    out = {k: np.full(n, np.nan) for k in ("mu_lo_patch", "mu_lo_krylov3", "mu_lo_krylov5", "mu_hi_cut", "mu_hi_hubaware")}
    out["mu_lo_trivial"] = np.ones(n)
    out["locally_certifiable"] = np.zeros(n, bool)
    c = R["c3"].get(K) if K in R["c3"] else None
    if K == "3s":
        c3m = R["c3"]["3mix"]
        for i, B in c3m["inv_ii"].items():
            out["mu_lo_patch"][i] = mu_pair(met.C[i], B)[0]
            out["mu_hi_cut"][i] = mu_pair(met.C[i], c3m["cut_ii"][i])[1] if c3m["cut_ii"][i] is not None else np.inf
            out["locally_certifiable"][i] = bool(c3m["certifiable"][i])
    elif c is not None:
        for i in range(n):
            out["mu_lo_patch"][i] = mu_pair(met.C[i], c["inv_ii"][i])[0]
            out["mu_hi_cut"][i] = mu_pair(met.C[i], c["cut_ii"][i])[1] if c["cut_ii"][i] is not None else np.inf
        out["locally_certifiable"] = c["certifiable"].copy()
    for ell in (3, 5):
        for i, B in R["kry"][ell].items():
            out["mu_lo_krylov%d" % ell][i] = mu_pair(met.C[i], B)[0]
    for w, W in R["c4"].items():
        if w != min(R["c4"]):
            continue
        KR = W["K"].get(K if isinstance(K, int) else -1)
        if KR is not None and "mu_hi_hubaware" in KR:
            out["mu_hi_hubaware"] = KR["mu_hi_hubaware"].copy()
    d = m.dims.astype(float)
    q = np.array([radii.q(int(k)) for k in m.dims])
    mu_lo = np.nanmax(np.stack([out["mu_lo_trivial"], out["mu_lo_patch"], out["mu_lo_krylov3"], out["mu_lo_krylov5"]]), axis=0)
    mu_hi = np.nanmin(np.stack([out["mu_hi_cut"], out["mu_hi_hubaware"]]), axis=0)
    out["S_lo"] = np.sqrt(mu_lo) * q
    out["S_hi"] = np.sqrt(mu_hi) * q
    out["lam_min_oracle"], out["lam_max_oracle"] = R["lam"][:, 0], R["lam"][:, 1]
    return out


def write_certs(R, dataset, snapshot, fn):
    m, err = R["model"], R["err"]
    floor = R["pose_floor"]
    iters = R["iters"]
    w0 = min(R["c4"]) if R["c4"] else None
    Ks = [K for K in R["c3"] if isinstance(K, int)] + (["3s"] if "3mix" in R["c3"] else [])
    cols = ["dataset", "snapshot", "frozen", "iter", "block_id", "block_type", "dim", "K", "err_true", "err_clamped",
            "E_C1", "E_C2", "E_C3", "E_C4", "E_hub", "C3_local", "C3_leak", "C4_local", "C4_leak", "hub_term",
            "hub_term_oracle", "ratio_C1", "ratio_C2", "ratio_C3", "ratio_C4", "ratio_hub", "mu_lo_trivial",
            "mu_lo_patch", "mu_lo_krylov3", "mu_lo_krylov5", "lam_min_oracle", "mu_hi_cut", "mu_hi_hubaware",
            "lam_max_oracle", "locally_certifiable", "S_lo", "S_hi", "stopped"]
    gate = (err > 10 * floor) & (err >= 1e-12)
    rows_written = 0
    with gzip.open(fn, "wt", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(cols)
        for K in Ks:
            b = per_block_bounds(R, K)
            if K == "3s":
                c = R["c3"]["3mix"]
                blocks = c["sample"]
            else:
                c = R["c3"][K]
                blocks = np.arange(m.n_blocks)
            E3 = c["E"]
            KR = R["c4"][w0]["K"].get(K) if (w0 is not None and isinstance(K, int)) else None
            W = R["c4"][w0] if w0 is not None else None
            has_hubs = W is not None and W["n_hubs"] > 0
            for bi in blocks:
                for j, t in enumerate(iters):
                    e = err[bi, j]
                    E1, E2 = R["cols"]["E_C1"][bi, j], R["cols"]["E_C2"][bi, j]
                    e3 = E3[bi, j] if E3 is not None else np.nan
                    loc3 = c["s"][bi, j]
                    ecl = W["err_clamped"][bi, j] if W is not None else np.nan
                    e4 = KR["E_per"][bi, j] if KR is not None else np.nan
                    ehub = KR["E_hub"][bi, j] if (KR is not None and "E_hub" in KR) else (e4 if (KR is not None and not has_hubs) else np.nan)
                    loc4 = np.nan
                    leak4 = np.nan
                    if KR is not None and KR["c"] is not None and np.isfinite(W["kappa"][bi]):
                        I = _supernode_of(R, w0, bi)
                        k = np.sqrt(W["kappa"][bi])
                        loc4 = k * KR["s"][I, j]
                        leak4 = k * (KR["c"][I, j] - KR["s"][I, j])
                    ht = KR["hub_term"][bi, j] if (KR is not None and "hub_term" in KR) else np.nan
                    hto = KR["hub_term_oracle"][bi, j] if (KR is not None and "hub_term_oracle" in KR) else np.nan
                    g = gate[bi, j]
                    rat = lambda Ev, den=e: (Ev / den) if (g and np.isfinite(Ev) and den >= 1e-12) else np.nan
                    cand = [v for v in (E1, E2, e3, ehub) if np.isfinite(v)]
                    Ebest = min(cand) if cand else np.inf
                    stopped = bool(Ebest <= radii.ETA * b["S_lo"][bi])
                    wr.writerow([dataset, snapshot, True, t, m.block_ids[bi], m.kinds[bi], int(m.dims[bi]), K,
                                 "%.6e" % e, "%.6e" % ecl, "%.6e" % E1, "%.6e" % E2, "%.6e" % e3, "%.6e" % e4, "%.6e" % ehub,
                                 "%.6e" % loc3, "%.6e" % (e3 - loc3), "%.6e" % loc4, "%.6e" % leak4, "%.6e" % ht, "%.6e" % hto,
                                 "%.6e" % rat(E1), "%.6e" % rat(E2), "%.6e" % rat(e3), "%.6e" % rat(e4, ecl), "%.6e" % rat(ehub),
                                 "%.6e" % b["mu_lo_trivial"][bi], "%.6e" % b["mu_lo_patch"][bi], "%.6e" % b["mu_lo_krylov3"][bi],
                                 "%.6e" % b["mu_lo_krylov5"][bi], "%.6e" % b["lam_min_oracle"][bi], "%.6e" % b["mu_hi_cut"][bi],
                                 "%.6e" % b["mu_hi_hubaware"][bi], "%.6e" % b["lam_max_oracle"][bi],
                                 bool(b["locally_certifiable"][bi]), "%.6e" % b["S_lo"][bi], "%.6e" % b["S_hi"][bi], stopped])
                    rows_written += 1
    return rows_written


_SN = {}


def _supernode_of(R, w, b):
    key = (id(R), w)
    if key not in _SN:
        W = R["c4"][w]
        sn = {}
        for I, g in enumerate(W["groups"]):
            for x in g:
                sn[int(x)] = I
        _SN[key] = sn
    return _SN[key][int(b)]


def ratio_stats(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return dict(n=0, median=np.nan, p95=np.nan, max=np.nan)
    return dict(n=int(len(v)), median=float(np.median(v)), p95=float(np.percentile(v, 95)), max=float(v.max()))


def gate_tables(R):
    """Per certificate: ratios (gated), shares, spectral radii, psibar, theta, certifiable fraction, radii targets."""
    m, err = R["model"], R["err"]
    floor = R["pose_floor"]
    gate = (err > 10 * floor) & (err >= 1e-12)
    poses = m.kinds == "pose"
    T = dict(pose_floor=floor, gated_entries=int(gate.sum()), gated_pose_entries=int(gate[poses].sum()),
             ratios={}, shares={}, rho={}, radii={}, psibar={}, hubs={})
    T["rho"]["M"] = R["res"]["rho_M"]
    T["rho"]["M_upper"] = R["res"]["rho_M_upper"]
    if "rho_Mhat" in R["res"]:
        T["rho"]["Mhat"] = R["res"]["rho_Mhat"]
        T["rho"]["Mhat_upper"] = R["res"]["rho_Mhat_upper"]
    for nm in ("E_C1", "E_C2"):
        Ev = R["cols"][nm]
        T["ratios"][nm] = dict(all=ratio_stats((Ev / err)[gate]), poses=ratio_stats((Ev / err)[gate & poses[:, None]]))
    for K, c in R["c3"].items():
        T["rho"]["A_C3_K%s" % K] = c["rho"]
        T["rho"]["A_C3_K%s_upper" % K] = c["rho_upper"]
        if isinstance(K, int):
            T["certifiable_fraction_K%d" % K] = dict(all=float(np.mean(c["certifiable"])), poses=float(np.mean(c["certifiable"][poses])))
        if c["E"] is None:
            continue
        Ev = c["E"]
        if K == "3mix":
            sm = np.zeros(m.n_blocks, bool)
            sm[c["sample"]] = True
            T["ratios"]["E_C3_K3s"] = dict(poses=ratio_stats((Ev / err)[gate & sm[:, None]]))
            continue
        T["ratios"]["E_C3_K%d" % K] = dict(all=ratio_stats((Ev / err)[gate]), poses=ratio_stats((Ev / err)[gate & poses[:, None]]))
        sh = c["s"] / Ev
        T["shares"]["C3_K%d_local" % K] = dict(all=ratio_stats(sh[gate]), poses=ratio_stats(sh[gate & poses[:, None]]))
    for w, W in R["c4"].items():
        for K, KR in W["K"].items():
            key = "w%d_K%d" % (w, K)
            T["rho"]["A_C4_" + key] = KR["rho"]
            T["rho"]["A_C4_" + key + "_upper"] = KR["rho_upper"]
            ps = np.concatenate([KR["psi"]["right"][np.isfinite(KR["psi"]["right"])], KR["psi"]["left"][np.isfinite(KR["psi"]["left"])]])
            T["psibar"][key] = [float(q) for q in np.quantile(ps, [0, 0.25, 0.5, 0.75, 1])] if len(ps) else []
            if KR["c"] is not None:
                ecl = W["err_clamped"]
                gcl = gate & np.isfinite(ecl) & (ecl >= 1e-12)
                T["ratios"]["E_C4_" + key] = dict(all=ratio_stats((KR["E_per"] / ecl)[gcl]),
                                                  poses=ratio_stats((KR["E_per"] / ecl)[gcl & poses[:, None]]))
                sh = KR["s"] / KR["c"]
                T["shares"]["C4_" + key + "_local_supernode"] = ratio_stats(sh.ravel())
            if "E_hub" in KR:
                Eh = KR["E_hub"]
                T["ratios"]["E_hub_" + key] = dict(all=ratio_stats((Eh / err)[gate]), poses=ratio_stats((Eh / err)[gate & poses[:, None]]))
                if "hub_term" in KR:
                    fin = np.isfinite(KR["hub_term"]) & gate
                    T["shares"]["hub_" + key] = dict(
                        chat_share=ratio_stats((KR["chat_per"] / Eh)[fin]), hub_share=ratio_stats((KR["hub_term"] / Eh)[fin]),
                        oracle_hub_over_bound=ratio_stats((KR["hub_term_oracle"] / KR["hub_term"])[fin & (KR["hub_term"] > 0)]))
            for k in ("theta_S", "theta_S_oracle", "hub_window", "HSS_dim", "HSS_block_diagonal", "hub_note"):
                if k in KR:
                    T["hubs"].setdefault(key, {})[k] = KR[k]
            if "E_S" in KR:
                T["hubs"][key]["E_S_over_eS"] = ratio_stats(KR["E_S"] / np.maximum(KR["eS_norm"], 1e-300))
                T["hubs"][key]["energy_over_E_S"] = ratio_stats(KR["energy_aggregate"] / KR["E_S"])
                T["hubs"][key]["sum_term_over_gS"] = ratio_stats(KR["sum_term"] / np.maximum(KR["gS_dual"], 1e-300))
        if "hub_diag" in W:
            hd = W["hub_diag"]
            T["hubs"]["w%d_diag" % w] = dict(n_hubs=hd["n_hubs"], hub_dof=hd["hub_dof"],
                                             theta_local=[float(q) for q in np.quantile(hd["theta_local"][np.isfinite(hd["theta_local"])], [0, 0.5, 0.95, 1])] if np.isfinite(hd["theta_local"]).any() else [],
                                             theta_local_inf=int(np.sum(~np.isfinite(hd["theta_local"]))),
                                             theta_oracle=[float(q) for q in np.quantile(hd["theta_oracle"], [0, 0.5, 0.95, 1])],
                                             theta_S_joint_oracle=hd["theta_S_joint_oracle"],
                                             energy_joint=ratio_stats(hd["energy_aggregate_joint_oracle"]),
                                             eS_joint=ratio_stats(hd["eS_norm_joint"]))
    # radii targets: mu_lo >= 0.5 lambda_min at K = 3 or l = 5; mu_hi <= 4 lambda_max at K = 3 for >= 90% of poses
    Kmax = max([K for K in R["c3"] if isinstance(K, int)])
    b = per_block_bounds(R, Kmax)
    lo = np.nanmax(np.stack([b["mu_lo_patch"], b["mu_lo_krylov5"]]), axis=0)
    ok_lo = lo >= 0.5 * b["lam_min_oracle"]
    ok_hi = b["mu_hi_cut"] <= 4 * b["lam_max_oracle"]
    hub_hi = np.nanmin(np.stack([b["mu_hi_cut"], b["mu_hi_hubaware"]]), axis=0) <= 4 * b["lam_max_oracle"]
    T["radii"] = dict(K=Kmax, frac_mu_lo_ge_half_lam_min_poses=float(np.mean(ok_lo[poses])),
                      frac_mu_lo_ge_half_lam_min_all=float(np.mean(ok_lo)),
                      frac_mu_hi_cut_le_4_lam_max_poses=float(np.mean(ok_hi[poses])),
                      frac_mu_hi_best_le_4_lam_max_poses=float(np.mean(hub_hi[poses])),
                      mu_lo_over_lam_min_poses=ratio_stats((lo / b["lam_min_oracle"])[poses]),
                      mu_hi_cut_over_lam_max_poses=ratio_stats((b["mu_hi_cut"] / b["lam_max_oracle"])[poses]),
                      lam_min_oracle_poses=ratio_stats(b["lam_min_oracle"][poses]),
                      lam_max_oracle_poses=ratio_stats(b["lam_max_oracle"][poses]),
                      krylov_blocks=int(len(R["kry"][5])))
    return T


def worst_cases(R):
    """The single worst block for every ratio (gated) and the largest true error, with its numbers."""
    m, err = R["model"], R["err"]
    floor = R["pose_floor"]
    gate = (err > 10 * floor) & (err >= 1e-12)
    iters = R["iters"]
    out = {}

    def rec(name, V, den=None):
        den = err if den is None else den
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(gate & (den >= 1e-12), V / den, np.nan)
        if not np.isfinite(r).any():
            return
        i, j = np.unravel_index(np.nanargmax(r), r.shape)
        out[name] = dict(block=str(m.block_ids[i]), kind=str(m.kinds[i]), pose_index=int(m.pose_index[i]), iter=int(iters[j]),
                         ratio=float(r[i, j]), E=float(V[i, j]), err=float(den[i, j]), degree=int(m.adj[i].nnz))

    rec("C1", R["cols"]["E_C1"])
    rec("C2", R["cols"]["E_C2"])
    for K, c in R["c3"].items():
        if c["E"] is not None:
            rec("C3_K%s" % K, c["E"])
    for w, W in R["c4"].items():
        for K, KR in W["K"].items():
            if KR["c"] is not None:
                rec("C4_w%d_K%d" % (w, K), KR["E_per"], W["err_clamped"])
            if "E_hub" in KR:
                rec("hub_w%d_K%d" % (w, K), KR["E_hub"])
    i, j = np.unravel_index(np.argmax(err[:, -1:]), err[:, -1:].shape)
    out["largest_err_last_iter"] = dict(block=str(m.block_ids[i]), kind=str(m.kinds[i]), err=float(err[i, -1]), iter=int(iters[-1]))
    return out


def curves(R):
    """E and true error against iteration for the median and the worst block (ranked by the true error at
    the first evaluated iteration), for every certificate."""
    m, err = R["model"], R["err"]
    order = np.argsort(err[:, 0])
    pick = dict(worst=int(order[-1]), median=int(order[len(order) // 2]))
    out = {}
    for lab, i in pick.items():
        d = dict(block=str(m.block_ids[i]), kind=str(m.kinds[i]), iters=R["iters"], err=err[i].tolist(),
                 E_C1=R["cols"]["E_C1"][i].tolist(), E_C2=R["cols"]["E_C2"][i].tolist())
        for K, c in R["c3"].items():
            if c["E"] is not None and isinstance(K, int):
                d["E_C3_K%d" % K] = c["E"][i].tolist()
        for w, W in R["c4"].items():
            for K, KR in W["K"].items():
                if "E_hub" in KR:
                    d["E_hub_w%d_K%d" % (w, K)] = KR["E_hub"][i].tolist()
                if KR["c"] is not None:
                    d["E_C4_w%d_K%d" % (w, K)] = KR["E_per"][i].tolist()
                    d["err_clamped_w%d" % w] = W["err_clamped"][i].tolist()
        out[lab] = d
    return out


def timings(R):
    t = dict(R["res"]["timing"])
    for K, c in R["c3"].items():
        if isinstance(K, int):
            tb = c["t_block"]
            t["C3_K%d_per_block_ms" % K] = dict(local_median=float(np.median(tb[:, 0]) * 1e3), local_max=float(tb[:, 0].max() * 1e3),
                                                coeff_median=float(np.median(tb[:, 1]) * 1e3), cut_median=float(np.median(tb[:, 2]) * 1e3),
                                                total_median=float(np.median(tb.sum(1)) * 1e3), total_p95=float(np.percentile(tb.sum(1), 95) * 1e3),
                                                patch_blocks_median=float(np.median(c["sizes"][:, 0])),
                                                patch_dofs_median=float(np.median(c["sizes"][:, 1])),
                                                patch_dofs_max=float(c["sizes"][:, 1].max()),
                                                global_solve_s=c["t_global"])
        elif K == "3mix":
            tb = np.array(list(c["t_block"].values()))
            t["C3_K3_sample_per_block_ms"] = dict(total_median=float(np.median(tb.sum(1)) * 1e3), total_max=float(tb.sum(1).max() * 1e3))
    n = R["model"].n_blocks
    t["C1_per_block_ms"] = t.get("C1_s", np.nan) / n * 1e3
    t["C2_per_block_ms"] = t.get("C2_s", np.nan) / n * 1e3 if "C2_s" in t else None
    for w, W in R["c4"].items():
        for K, KR in W["K"].items():
            t["C4_w%d_K%d_per_supernode_s" % (w, K)] = KR["time_s"] / W["n_super"]
    return t


def write_all(R, dataset, snapshot):
    d = OUT / dataset
    d.mkdir(parents=True, exist_ok=True)
    inv = invariant_checks(R)
    rows = write_certs(R, dataset, snapshot, d / "certs.csv.gz")
    T = gate_tables(R)
    wc = worst_cases(R)
    cv = curves(R)
    tm = timings(R)
    g = R["glob"]
    graph = dict(dataset=dataset, snapshot=snapshot, n_blocks=R["model"].n_blocks, n_factors=R["model"].n_factors,
                 w=[int(w) for w in R["c4"]], n_hubs={int(w): W["n_hubs"] for w, W in R["c4"].items()},
                 rho_M=R["res"]["rho_M"], rho_Mhat=R["res"].get("rho_Mhat"),
                 rho_A={str(K): c["rho"] for K, c in R["c3"].items()},
                 theta_S={k: v.get("theta_S") for k, v in T["hubs"].items() if isinstance(v, dict)},
                 theta_S_oracle={k: v.get("theta_S_oracle") for k, v in T["hubs"].items() if isinstance(v, dict)},
                 psibar_quantiles=T["psibar"], global_lb_1step=g[:, 0].tolist(), global_lb_5step=g[:, 1].tolist(),
                 global_energy_oracle=g[:, 2].tolist(), iters=R["iters"])
    summary = dict(dataset=dataset, snapshot=snapshot, res=R["res"], gate=T, worst=wc, timing=tm, invariants=inv,
                   certs_rows=rows)
    (d / "summary.json").write_text(json.dumps(summary, indent=1, default=_js))
    (d / "graph.json").write_text(json.dumps(graph, indent=1, default=_js))
    (d / "curves.json").write_text(json.dumps(cv, indent=1, default=_js))
    return summary


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)
