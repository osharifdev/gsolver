"""I2 (progress log 17:56): certificates in the statistical (Mahalanobis) metric on the reduced pose model.

Theorem 3.5 holds for any SPD metric per block. With W_i = L_i^-1, L_i = [H_PP^-1]_ii the certified patch lower
bound on Sigma_ii (Lemma 2.1), we have ||e_i||_{Sigma_ii^-1} <= ||e_i||_{W_i}; and ||xbar_i - x*_i||_{Sigma_ii^-1}
~ chi_d exactly, so its (1-delta) quantile is q_d(delta) with no lower-bound slack. Stopping rule in this metric:
E_i^W <= eta q_d(delta). Compared with the brief's D-metric rule E_i^D <= eta sqrt(mu_lo) q_d(delta), and with
the oracle ||e_i||_{Sigma_ii^-1} <= eta q_d(delta), for fixed-radius and grown patches.

Usage: python3 scripts/explore_mahalanobis.py dataset [workers]
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as sla

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from certify import certs, covbounds, phaseC, radii  # noqa: E402
from certify import model as mdl  # noqa: E402
from certify.linalg import SPDSystem  # noqa: E402

spec = importlib.util.spec_from_file_location("greedy", REPO / "scripts" / "explore_greedy.py")
greedy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(greedy)


class MetricW:
    """Metric-like object with per-block factors W_i = C_i C_i^T (C_i lower), for certs.patch_one."""

    def __init__(self, Ws):
        self.C = [np.linalg.cholesky(0.5 * (W + W.T)) for W in Ws]
        self.Cinv = [sla.solve_triangular(C, np.eye(len(C)), lower=True) for C in self.C]


def norms(Cs, V, red):
    return np.array([np.linalg.norm(Cs[i].T @ V[red.block_slice(i)], axis=0) for i in range(red.n_blocks)])


def main(name, workers=12):
    import multiprocessing as mp
    path, cfg = phaseC.DATASETS[name]
    full, info = mdl.from_export(path, p_L=cfg["p_L"])
    red, kept = mdl.reduce_blocks(full)
    n = red.n_blocks
    ex = info["export"]
    Z = ex.states()
    avail = np.load(Path(path) / "states_iters.npy")
    xcols = np.concatenate([np.arange(ex.offsets[b], ex.offsets[b] + ex.dims[b]) for b in info["kept_export_blocks"][kept]])
    want = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 1000]
    iters = [t for t in want if t in set(avail.tolist())]
    row = {int(t): k for k, t in enumerate(avail)}
    X = np.stack([np.asarray(Z[row[t]])[xcols] for t in iters], axis=1)
    sysm = SPDSystem(red.H, red.dims, None, refine=3, K_lo=red.H_lo)
    G = red.residual(X)
    E = sysm.solve(G)
    metD = certs.Metric(red)
    Sig = sysm.solve(np.eye(red.N), extended=False, refine=2)
    SigB = [0.5 * (Sig[red.block_slice(i), red.block_slice(i)] + Sig[red.block_slice(i), red.block_slice(i)].T) for i in range(n)]
    del Sig
    CSinv = [np.linalg.cholesky(np.linalg.inv(S)) for S in SigB]  # Sigma_ii^-1 = C C^T
    err_stat = norms(CSinv, E, red)                              # ||e_i||_{Sigma_ii^-1}
    q = radii.q(12)
    oracle = [float(np.mean(err_stat[:, j] <= radii.ETA * q)) for j in range(len(iters))]
    errD = metD.norms(E)
    out = dict(dataset=name, iters=iters, oracle_stat=oracle, runs={})
    print("%s: oracle stop in the statistical metric: %s at %s" % (name, np.round(oracle, 2), iters), flush=True)
    greedy._W.update(red=red, met=metD, G=G)

    def patches_for(tag):
        if tag.startswith("K="):
            K = int(tag[2:])
            return certs.patch_pass(red, metD, K, G, workers=workers, chunk=4, want_cut=False), None
        r = float(tag.split("=")[1])
        blocks = np.arange(n)
        with mp.get_context("fork").Pool(workers) as pool:
            sets = pool.map(_grow, [(int(i), r) for i in blocks])
        return None, sets

    for tag in ("K=1", "K=2", "greedy r=0.3"):
        PD, sets = patches_for(tag)
        if PD is None:
            PD = [certs.patch_one(red, metD, i, 0, G, want_cut=False, P=sets[i], extended=True) for i in range(n)]
        # D-metric certificate and the brief's rule
        rD = certs.c3_from_patches(red, PD)
        mu = np.array([max(1.0, covbounds.mu_extremes(metD.C[p["i"]], p["inv_ii"])[0]) for p in PD])
        rec = dict(rho_D=rD["rho"])
        if rD["E"] is not None:
            rec["stop_D_rule"] = [float(np.mean(rD["E"][:, j] <= radii.ETA * np.sqrt(mu) * q)) for j in range(len(iters))]
            rec["stop_D_converted"] = [float(np.mean(rD["E"][:, j] / np.sqrt(mu) <= radii.ETA * q)) for j in range(len(iters))]
        # W-metric certificate on the same patches
        metW = MetricW([np.linalg.inv(p["inv_ii"]) for p in PD])
        Pset = [p for p in PD]
        PW = []
        for p in Pset:
            i = p["i"]
            Pblocks = red.ball(i, int(tag[2:])) if tag.startswith("K=") else sets[i]
            PW.append(certs.patch_one(red, metW, i, 0, G, want_cut=False, P=Pblocks, extended=True))
        rW = certs.c3_from_patches(red, PW)
        errW = norms(metW.C, E, red)
        rec["rho_W"] = rW["rho"]
        if rW["E"] is not None:
            Ew = rW["E"]
            ok = errW >= 1e-12
            rec.update(valid_W=bool(np.all(Ew >= errW * (1 - 1e-9))),
                       valid_stat=bool(np.all(Ew >= err_stat * (1 - 1e-9))),
                       ratio_W_median=float(np.median((Ew / errW)[ok])),
                       stop_W_rule=[float(np.mean(Ew[:, j] <= radii.ETA * q)) for j in range(len(iters))])
        out["runs"][tag] = rec
        print("%-13s rho_D %.3f rho_W %.3f | stop D-rule %s | stop D->W converted %s | stop W-rule %s (valid %s, ratio_W median %s)" % (
            tag, rec["rho_D"], rec["rho_W"], np.round(rec.get("stop_D_rule", [np.nan])[:10], 2),
            np.round(rec.get("stop_D_converted", [np.nan])[:10], 2), np.round(rec.get("stop_W_rule", [np.nan])[:10], 2),
            rec.get("valid_W"), None if "ratio_W_median" not in rec else round(rec["ratio_W_median"], 3)), flush=True)
    d = phaseC.OUT / "explore"
    d.mkdir(parents=True, exist_ok=True)
    (d / ("mahalanobis_%s.json" % name)).write_text(json.dumps(out, indent=1, default=float))


def _grow(args):
    i, r = args
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        return greedy.grow_set(i, r)[0]


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12)
