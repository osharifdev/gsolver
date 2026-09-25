"""Certificates on the reduced pose model (exact landmark/marker marginalization; certify.model.reduce_blocks).

The reduced model has the same pose solution as the certified model of the export (decisions D2(a)-(c)); its
residual g_red = S x_p - b_red depends only on GBP's pose estimates. Everything is measured in the reduced
model's own metric D_i = S_ii (S = the reduced H). Computes, for the evaluated iterations:
  oracle     x_bar and e = S^-1 g_red (double-double refined), exact Sigma = S^-1, the frozen-GBP pose
             floor in this metric (slope rule of Phase A over every dumped iteration);
  C1, C3     (K = 1, 2, 3; Theorem 3.5 on the reduced model), with local/leak shares and per-block times;
  bounds     trivial / patch / Krylov (l = 3, 5) lower bounds, cut upper bounds, radii (delta = 0.05) with
             S_lo = sqrt(max mu_lo) q_12 from the certified lower bounds only;
  stopping   fraction of poses with E_i <= 0.1 S_lo_i per iteration, against the oracle fraction;
  invariants every certificate >= the true error, one-step inequalities, covariance bounds against Sigma.
Writes results/phaseC/reduced/<dataset>.json and a per-(pose, iteration, K) CSV.

Usage: python3 scripts/phaseC_reduced.py dataset [workers] [elim kinds, default landmark]
"""

import csv
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from certify import certs, covbounds, phaseC, radii  # noqa: E402
from certify import model as mdl  # noqa: E402
from certify.linalg import SPDSystem  # noqa: E402
from certify.phaseA import floor_and_rate  # noqa: E402

TOL = 1e-9


def viol(u, E):
    u, E = np.broadcast_arrays(np.asarray(u, float), np.asarray(E, float))
    den = np.maximum(np.maximum(np.abs(E), np.abs(u)), 1e-300)
    with np.errstate(invalid="ignore"):
        v = (u - E) / den
    v = np.array(v, float)
    v[~np.isfinite(E) & (E > 0)] = -1.0
    return v


def psd_viol(C, X, Y):
    A, B = C.T @ X @ C, C.T @ Y @ C
    d = np.linalg.eigvalsh(0.5 * ((B - A) + (B - A).T))[0]
    sc = max(np.linalg.eigvalsh(0.5 * (A + A.T))[-1], np.linalg.eigvalsh(0.5 * (B + B.T))[-1], 1e-300)
    return float(-d / sc)


def stats(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return dict(n=int(len(v)), median=float(np.median(v)) if len(v) else None, p95=float(np.percentile(v, 95)) if len(v) else None,
                max=float(v.max()) if len(v) else None)


def main(name, workers=12, elim_kinds=("landmark",)):
    t_all = time.perf_counter()
    path, cfg = phaseC.DATASETS[name]
    full, info = mdl.from_export(path, p_L=cfg["p_L"])
    t0 = time.perf_counter()
    red, kept = mdl.reduce_blocks(full, elim_kinds=elim_kinds)
    t_reduce = time.perf_counter() - t0
    n = red.n_blocks
    print("%s: full %d blocks -> reduced %d blocks (%d factors) in %.0fs" % (name, full.n_blocks, n, red.n_factors, t_reduce), flush=True)
    kd = full.dof_index(kept)
    # frozen states restricted to the kept blocks
    ex = info["export"]
    Z = ex.states()
    avail = np.load(Path(path) / "states_iters.npy")
    xcols = np.concatenate([np.arange(ex.offsets[b], ex.offsets[b] + ex.dims[b]) for b in info["kept_export_blocks"][kept]])
    # oracle
    sysm = SPDSystem(red.H, red.dims, None, refine=3, K_lo=red.H_lo)
    assert sysm.ok
    xbar = sysm.solve(red.b)
    met = certs.Metric(red)
    # floor curve over every dumped iteration (in chunks)
    curve = []
    Hd = red.H.tocsr()
    for s in range(0, len(avail), 200):
        Xc = np.stack([np.asarray(Z[k])[xcols] for k in range(s, min(len(avail), s + 200))], axis=1)
        # floor estimate only (not an invariant check): double-precision residual and solve
        Ec = sysm.solve(Hd @ Xc - red.b[:, None], extended=False, refine=1)
        curve.append(met.norms(Ec).max(axis=0))
    curve = np.concatenate(curve)
    fr = floor_and_rate(avail, curve)
    floor = float(fr["floor"])
    want = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000]
    iters = [t for t in want if t in set(avail.tolist())]
    row = {int(t): k for k, t in enumerate(avail)}
    X = np.stack([np.asarray(Z[row[t]])[xcols] for t in iters], axis=1)
    G = red.residual(X)
    E = sysm.solve(G)
    err = met.norms(E)
    # consistency with the full model's solution (D2 run, p_L of the dataset)
    tag = "none" if cfg["p_L"] is None else "%g" % cfg["p_L"]
    xf_path = phaseC.OUT / "d2" / ("%s_xbar_pL%s.npy" % (name, tag))
    xfull_diff = None
    if xf_path.exists():
        xf = np.load(xf_path)[kd]
        xfull_diff = float(met.norms(xf - xbar).max())
    # exact marginals: backward-stable Jacobi-scaled Cholesky with two double-precision refinement steps; any
    # block whose covariance check is not passed is re-verified below with double-double refinement of its columns
    Sig = sysm.solve(np.eye(red.N), extended=False, refine=2)
    Sig = 0.5 * (Sig + Sig.T)
    SigB = [Sig[red.block_slice(i), red.block_slice(i)] for i in range(n)]
    del Sig
    rechecked = {}

    def sig_exact(i):
        if i not in rechecked:
            d = int(red.dims[i])
            Eb = np.zeros((red.N, d))
            Eb[red.block_slice(i)] = np.eye(d)
            B = sysm.solve(Eb)[red.block_slice(i)]
            rechecked[i] = 0.5 * (B + B.T)
        return rechecked[i]
    lam = np.array([np.linalg.eigvalsh(met.C[i].T @ SigB[i] @ met.C[i])[[0, -1]] for i in range(n)])
    q = np.array([radii.q(int(d)) for d in red.dims])
    S_oracle = np.sqrt(lam[:, 0]) * q
    gate = (err > 10 * floor) & (err >= 1e-12)
    res = dict(dataset=name, eliminated=list(elim_kinds), n_blocks_full=int(full.n_blocks), n_blocks=int(n), n_factors=int(red.n_factors),
               reduce_s=t_reduce, iters=iters, floor=fr, full_model_solution_diff_max=xfull_diff,
               oracle_stop_fraction=[float(np.mean(err[:, j] <= radii.ETA * S_oracle)) for j in range(len(iters))],
               err_median=[float(np.median(err[:, j])) for j in range(len(iters))],
               err_max=[float(np.max(err[:, j])) for j in range(len(iters))],
               lam_min_oracle=stats(lam[:, 0]), lam_max_oracle=stats(lam[:, 1]),
               coupled_blocks_median=float(np.median(np.diff(red.adj.indptr))), checks=[], K={})
    chk = res["checks"]

    def add(nm, v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        w = float(v.max()) if len(v) else -np.inf
        chk.append(dict(check=nm, n=int(len(v)), worst=w, ok=bool(w <= TOL)))

    def cov_check(nm, pairs, lower=True):
        """pairs: (block, bound); flagged blocks re-verified against a double-double refined Sigma_ii."""
        v = []
        for i, B in pairs:
            if B is None:
                continue
            x = psd_viol(met.C[i], B, SigB[i]) if lower else psd_viol(met.C[i], SigB[i], B)
            if x > TOL:
                Se = sig_exact(i)
                x = psd_viol(met.C[i], B, Se) if lower else psd_viol(met.C[i], Se, B)
            v.append(x)
        add(nm, v)

    # C1
    t0 = time.perf_counter()
    r1 = certs.c1(red, met, G)
    res["C1"] = dict(rho=r1["rho"], rho_upper=r1["rho_upper"], seconds=time.perf_counter() - t0)
    add("C1 one-step", viol(err, r1["nu"] + r1["M"] @ err))
    if r1["E"] is not None:
        add("C1 E>=err", viol(err, r1["E"]))
        res["C1"]["ratio"] = stats((r1["E"] / err)[gate])
    print("C1 rho(M) = %.4f" % r1["rho"], flush=True)
    # Krylov lower bounds (l = 3, 5 from one nested basis)
    t0 = time.perf_counter()
    KM = phaseC._parallel_krylov(red, list(range(n)), (3, 5), workers, chunk=max(1, n // (4 * workers)))
    res["krylov_s"] = time.perf_counter() - t0
    mu_k = {ell: np.array([covbounds.mu_extremes(met.C[i], KM[i][ell][0])[0] for i in range(n)]) for ell in (3, 5)}
    for ell in (3, 5):
        cov_check("krylov lower<=Sigma l=%d" % ell, [(i, KM[i][ell][0]) for i in range(n)])
    add("krylov monotone 3->5", [psd_viol(met.C[i], KM[i][3][0], KM[i][5][0]) for i in range(n)])
    D = red.diag_blocks()
    cov_check("trivial lower<=Sigma", [(i, np.linalg.inv(D[i])) for i in range(n)])
    prevL, prevU = None, None
    rows_out = []
    for K in (1, 2, 3):
        t0 = time.perf_counter()
        P = certs.patch_pass(red, met, K, G, workers=workers, chunk=4)
        tp = time.perf_counter() - t0
        r3 = certs.c3_from_patches(red, P)
        mu_p = np.array([covbounds.mu_extremes(met.C[p["i"]], p["inv_ii"])[0] for p in P])
        mu_c = np.array([covbounds.mu_extremes(met.C[p["i"]], p["cut_ii"])[1] if p["cut_ii"] is not None else np.inf for p in P])
        mu_lo = np.maximum.reduce([np.ones(n), mu_p, mu_k[3], mu_k[5]])
        S_lo = np.sqrt(mu_lo) * q
        S_hi = np.sqrt(mu_c) * q
        add("C3 one-step K=%d" % K, viol(err, r3["s"] + r3["A"] @ err))
        cov_check("patch lower<=Sigma K=%d" % K, [(p["i"], p["inv_ii"]) for p in P])
        cov_check("cut upper>=Sigma K=%d" % K, [(p["i"], p["cut_ii"]) for p in P], lower=False)
        if prevL is not None:
            add("patch lower monotone K=%d" % K, [psd_viol(met.C[i], prevL[i], P[i]["inv_ii"]) for i in range(n)])
            add("cut upper monotone K=%d" % K, [psd_viol(met.C[i], P[i]["cut_ii"], prevU[i]) for i in range(n)
                                               if P[i]["cut_ii"] is not None and prevU[i] is not None])
        prevL, prevU = [p["inv_ii"] for p in P], [p["cut_ii"] for p in P]
        rec = dict(rho=r3["rho"], rho_upper=r3["rho_upper"], iterations=r3["iterations"], row_sum_max=r3["row_sum_max"],
                   patch_blocks=stats([p["n_patch"] for p in P]), patch_fraction_median=float(np.median([p["n_patch"] for p in P]) / n),
                   certifiable_fraction=float(np.mean([bool(p["certifiable"]) for p in P])),
                   seconds_patches=tp, per_block_ms=stats([1e3 * (p["t_local"] + p["t_coeff"] + p["t_cut"]) for p in P]),
                   mu_lo_over_lam_min=stats(mu_lo / lam[:, 0]), mu_hi_over_lam_max=stats(mu_c / lam[:, 1]),
                   frac_mu_lo_ge_half_lam_min=float(np.mean(mu_lo >= 0.5 * lam[:, 0])),
                   frac_mu_hi_le_4_lam_max=float(np.mean(mu_c <= 4 * lam[:, 1])),
                   S_lo=stats(S_lo), S_hi=stats(S_hi), S_oracle=stats(S_oracle),
                   oracle_stop_fraction_with_S_lo=[float(np.mean(err[:, j] <= radii.ETA * S_lo)) for j in range(len(iters))])
        if r3["E"] is not None:
            Ec = r3["E"]
            add("C3 E>=err K=%d" % K, viol(err, Ec))
            rec.update(ratio=stats((Ec / err)[gate]), local_share=stats((r3["s"] / np.maximum(Ec, 1e-300))[gate]),
                       abs_near_floor=dict(E=stats(Ec[~gate]), err=stats(err[~gate])),
                       stop_fraction=[float(np.mean(Ec[:, j] <= radii.ETA * S_lo)) for j in range(len(iters))])
            for i in range(n):
                for j, t in enumerate(iters):
                    rows_out.append([name, "reduced", t, str(red.block_ids[i]), K, "%.6e" % err[i, j], "%.6e" % Ec[i, j],
                                     "%.6e" % r3["s"][i, j], "%.6e" % (Ec[i, j] - r3["s"][i, j]),
                                     "%.6e" % (Ec[i, j] / err[i, j] if gate[i, j] else np.nan), "%.6e" % mu_p[i],
                                     "%.6e" % mu_k[3][i], "%.6e" % mu_k[5][i], "%.6e" % lam[i, 0], "%.6e" % mu_c[i], "%.6e" % lam[i, 1],
                                     bool(P[i]["certifiable"]), "%.6e" % S_lo[i], "%.6e" % S_hi[i], bool(Ec[i, j] <= radii.ETA * S_lo[i])])
        res["K"][K] = rec
        print("C3 K=%d: rho(A) = %.4f, patch median %d of %d blocks%s" % (
            K, r3["rho"], rec["patch_blocks"]["median"], n,
            "" if r3["E"] is None else ", ratio median %.3f p95 %.3f max %.3f, stop %s (oracle %s)" % (
                rec["ratio"]["median"] or np.nan, rec["ratio"]["p95"] or np.nan, rec["ratio"]["max"] or np.nan,
                np.round(rec["stop_fraction"][:12], 2), np.round(rec["oracle_stop_fraction_with_S_lo"][:12], 2))), flush=True)
    res["cov_blocks_rechecked"] = len(rechecked)
    res["seconds"] = time.perf_counter() - t_all
    res["failed_checks"] = [c for c in chk if not c["ok"]]
    out = phaseC.OUT / "reduced"
    out.mkdir(parents=True, exist_ok=True)
    (out / ("%s.json" % name)).write_text(json.dumps(res, indent=1, default=float))
    with gzip.open(out / ("%s_certs.csv.gz" % name), "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "iter", "block_id", "K", "err_true", "E_C3", "C3_local", "C3_leak", "ratio_C3", "mu_lo_patch",
                    "mu_lo_krylov3", "mu_lo_krylov5", "lam_min_oracle", "mu_hi_cut", "lam_max_oracle", "locally_certifiable",
                    "S_lo", "S_hi", "stopped"])
        w.writerows(rows_out)
    print("%s: done in %.0fs; %d checks, %d failed; floor %.3e (reached %s)" % (
        name, res["seconds"], len(chk), len(res["failed_checks"]), floor, fr["floor_reached"]), flush=True)
    for c in res["failed_checks"]:
        print("   FAILED", c, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]) if len(a) > 1 else 12, tuple(a[2].split(",")) if len(a) > 2 else ("landmark",))
