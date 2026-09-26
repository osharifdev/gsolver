"""I2 at scale: C3 in the D-metric and in the statistical metric W_i = L_i^-1 from one factorization per patch.

For each patch family (hop radius K, or certificate-driven grown patches) every pose's patch is solved once and its
metric-free pieces are kept: Z_i = [H_PP^-1 g_P]_i (d x T), L_i = [H_PP^-1]_ii (a certified lower bound on
Sigma_ii, Lemma 2.1) and the raw leak blocks X_ij = sum_{b in dP} [H_PP^-1]_ib H_bj. Any per-block metric
M_j = C_j C_j^T then gives s_i = ||C_i^T Z_i|| and a_ij = ||C_i^T X_ij C_j^-T|| (Definition 3.4), and Theorem 3.5
applies in that metric. Two metrics per family:
  D  (D_i = H_ii): the brief's rule E_i^D <= eta sqrt(mu_lo) q_d(delta), with mu_lo from L_i (and >= 1, trivial bound);
  W  (W_i = L_i^-1): ||e_i||_{Sigma_ii^-1} <= ||e_i||_{W_i} <= E_i^W; rule E_i^W <= eta q_d(delta).
The W-rule guarantees ||e_i||_{Sigma_ii^-1} <= eta q_d(delta); the D-rule implies that but not conversely, so each
rule is compared with its own oracle (the rule applied to the exact error and exact Sigma_ii).

Usage: python3 scripts/explore_metric.py dataset [workers] [families, default K=1,grown0.3,K=2]
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from certify import certs, phaseC, radii  # noqa: E402
from certify import model as mdl  # noqa: E402
from certify.linalg import SPDSystem  # noqa: E402

spec = importlib.util.spec_from_file_location("greedy", REPO / "scripts" / "explore_greedy.py")
greedy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(greedy)

_W = {}


def patch_raw(i, P):
    """Metric-free pieces of pose i's patch P (mirrors certs.patch_one)."""
    red, G = _W["red"], _W["G"]
    P = np.asarray(P, int)
    n = red.n_blocks
    inP = np.zeros(n, bool)
    inP[P] = True
    dP = red.dims[P]
    loc_off = np.zeros(n, int)
    loc_off[P] = np.concatenate([[0], np.cumsum(dP)[:-1]])
    dofs = red.dof_index(P)
    Kpp, Kpp_lo = red.sub(dofs)
    sysm = SPDSystem(Kpp, dP, red.elim[P].copy(), K_lo=Kpp_lo)
    assert sysm.ok
    d = int(red.dims[i])
    Ei = np.zeros((len(dofs), d))
    Ei[loc_off[i]:loc_off[i] + d] = np.eye(d)
    Y = sysm.solve(Ei, extended=True)
    L = 0.5 * (Y[loc_off[i]:loc_off[i] + d] + Y[loc_off[i]:loc_off[i] + d].T)
    Z = Y.T @ G[dofs]
    Adj = red.adj
    rowsP = Adj[P]
    outside = ~inP[rowsP.indices]
    row_of = np.repeat(np.arange(len(P)), np.diff(rowsP.indptr))
    cnt = np.bincount(row_of, weights=outside.astype(float), minlength=len(P))
    bP = P[cnt > 0]
    ext = np.zeros(0, int)
    X = np.zeros((d, 0))
    if len(bP):
        ext = np.unique(Adj[bP].indices)
        ext = ext[~inP[ext]]
        if len(ext):
            bdofs = red.dof_index(bP)
            Yb = Y[np.concatenate([np.arange(loc_off[b], loc_off[b] + red.dims[b]) for b in bP])]
            X = (red.H[bdofs][:, red.dof_index(ext)].T @ Yb).T
    return dict(i=int(i), n_patch=len(P), Z=Z, L=L, ext=ext, X=X)


def coeffs(raw, C, Cinv, dims):
    """s_i and a_ij of Definition 3.4 in the per-block metric with factors C (lower, M_j = C_j C_j^T)."""
    i = raw["i"]
    s = np.linalg.norm(C[i].T @ raw["Z"], axis=0)
    ext = raw["ext"]
    a = np.zeros(len(ext))
    if len(ext):
        W = C[i].T @ raw["X"]
        off = np.concatenate([[0], np.cumsum(dims[ext])[:-1]])
        for k, j in enumerate(ext):
            a[k] = np.linalg.norm(W[:, off[k]:off[k] + dims[j]] @ Cinv[j].T, 2)
    return dict(i=i, s=s, a_cols=ext, a_vals=a)


def _raw_worker(tasks):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        return [patch_raw(int(i), P) for i, P in tasks]


def _grow_worker(blocks):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        return [greedy.grow_set(int(i), _W["r"])[0] for i in blocks]


def main(name, workers=12, families=("K=1", "grown0.3", "K=2")):
    import multiprocessing as mp
    t_start = time.perf_counter()
    path, cfg = phaseC.DATASETS[name]
    full, info = mdl.from_export(path, p_L=cfg["p_L"])
    red, kept = mdl.reduce_blocks(full)
    n = red.n_blocks
    ex = info["export"]
    Zs = ex.states()
    avail = np.load(Path(path) / "states_iters.npy")
    xcols = np.concatenate([np.arange(ex.offsets[b], ex.offsets[b] + ex.dims[b]) for b in info["kept_export_blocks"][kept]])
    want = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 1000]
    iters = [t for t in want if t in set(avail.tolist())]
    row = {int(t): k for k, t in enumerate(avail)}
    X = np.stack([np.asarray(Zs[row[t]])[xcols] for t in iters], axis=1)
    sysm = SPDSystem(red.H, red.dims, None, refine=3, K_lo=red.H_lo)
    G = red.residual(X)
    E = sysm.solve(G)
    met = certs.Metric(red)
    # exact Sigma_ii, block by block columns in chunks (backward-stable solve, as in the pose-model run)
    Sig = []
    for b0 in range(0, n, 40):
        blk = np.arange(b0, min(n, b0 + 40))
        dofs = red.dof_index(blk)
        R = np.zeros((red.N, len(dofs)))
        R[dofs, np.arange(len(dofs))] = 1.0
        S = sysm.solve(R, extended=False, refine=2)
        off = 0
        for b in blk:
            d = int(red.dims[b])
            Sb = S[red.block_slice(b), off:off + d]
            Sig.append(0.5 * (Sb + Sb.T))
            off += d
    q = np.array([radii.q(int(d)) for d in red.dims])
    errD = met.norms(E)
    lam_min = np.array([np.linalg.eigvalsh(met.C[i].T @ Sig[i] @ met.C[i])[0] for i in range(n)])
    CS = [np.linalg.cholesky(np.linalg.inv(S)) for S in Sig]
    err_stat = np.array([np.linalg.norm(CS[i].T @ E[red.block_slice(i)], axis=0) for i in range(n)])
    T = len(iters)
    oracle_D = [float(np.mean(errD[:, j] <= radii.ETA * np.sqrt(lam_min) * q)) for j in range(T)]
    oracle_W = [float(np.mean(err_stat[:, j] <= radii.ETA * q)) for j in range(T)]
    out = dict(dataset=name, n=n, iters=iters, oracle_D=oracle_D, oracle_W=oracle_W, families={})
    print("%s: %d poses; oracle D-rule %s; oracle W-rule %s at %s (%.0fs)" % (
        name, n, np.round(oracle_D, 2), np.round(oracle_W, 2), iters, time.perf_counter() - t_start), flush=True)
    _W.update(red=red, met=met, G=G)
    greedy._W.update(red=red, met=met, G=G)
    for fam in families:
        t0 = time.perf_counter()
        if fam.startswith("K="):
            sets = [red.ball(i, int(fam[2:])) for i in range(n)]
        else:
            _W["r"] = float(fam[len("grown"):])
            with mp.get_context("fork").Pool(workers) as pool:
                sets = [s for part in pool.map(_grow_worker, [np.arange(k, min(n, k + 4)) for k in range(0, n, 4)]) for s in part]
        tasks = [(i, sets[i]) for i in range(n)]
        order = np.argsort([-len(s) for s in sets])   # largest patches first for load balance
        chunks = [[tasks[k] for k in order[c:c + 2]] for c in range(0, n, 2)]
        with mp.get_context("fork").Pool(workers) as pool:
            raws = sorted([r for part in pool.map(_raw_worker, chunks) for r in part], key=lambda r: r["i"])
        rec = dict(patch_median=float(np.median([r["n_patch"] for r in raws])), seconds=0.0)
        # D metric
        PD = [coeffs(r, met.C, met.Cinv, red.dims) for r in raws]
        rD = certs.c3_from_patches(red, PD)
        mu = np.array([max(1.0, np.linalg.eigvalsh(met.C[r["i"]].T @ r["L"] @ met.C[r["i"]])[0]) for r in raws])
        rec["rho_D"] = rD["rho"]
        if rD["E"] is not None:
            Ed = rD["E"]
            rec.update(valid_D=bool(np.all(Ed >= errD * (1 - 1e-9))),
                       stop_D=[float(np.mean(Ed[:, j] <= radii.ETA * np.sqrt(mu) * q)) for j in range(T)])
        # W metric: W_i = L_i^-1
        CW = [np.linalg.cholesky(np.linalg.inv(r["L"])) for r in raws]
        CWinv = [sla.solve_triangular(c, np.eye(len(c)), lower=True) for c in CW]
        PW = [coeffs(r, CW, CWinv, red.dims) for r in raws]
        rW = certs.c3_from_patches(red, PW)
        rec["rho_W"] = rW["rho"]
        if rW["E"] is not None:
            Ew = rW["E"]
            errW = np.array([np.linalg.norm(CW[i].T @ E[red.block_slice(i)], axis=0) for i in range(n)])
            ok = errW >= 1e-12
            rec.update(valid_W=bool(np.all(Ew >= errW * (1 - 1e-9))), valid_stat=bool(np.all(Ew >= err_stat * (1 - 1e-9))),
                       ratio_W_median=float(np.median((Ew / np.where(ok, errW, 1))[ok])),
                       stop_W=[float(np.mean(Ew[:, j] <= radii.ETA * q)) for j in range(T)])
        rec["seconds"] = time.perf_counter() - t0
        out["families"][fam] = rec
        print("%-9s patch median %5.1f | rho_D %.3f rho_W %.3f | valid D %s W %s stat %s | ratio_W median %s | %.0fs" % (
            fam, rec["patch_median"], rec["rho_D"], rec["rho_W"], rec.get("valid_D"), rec.get("valid_W"), rec.get("valid_stat"),
            None if "ratio_W_median" not in rec else round(rec["ratio_W_median"], 3), rec["seconds"]), flush=True)
        print("          stop D-rule %s (oracle %s)" % (np.round(rec.get("stop_D", [np.nan]), 2), np.round(oracle_D, 2)), flush=True)
        print("          stop W-rule %s (oracle %s)" % (np.round(rec.get("stop_W", [np.nan]), 2), np.round(oracle_W, 2)), flush=True)
        del raws
    out["seconds"] = time.perf_counter() - t_start
    d = phaseC.OUT / "explore"
    d.mkdir(parents=True, exist_ok=True)
    (d / ("metric_%s.json" % name)).write_text(json.dumps(out, indent=1, default=float))
    print("done in %.0fs" % out["seconds"], flush=True)


if __name__ == "__main__":
    fams = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ("K=1", "grown0.3", "K=2")
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12, fams)
