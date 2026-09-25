"""I1 (progress log 18:05): certificate-driven adaptive patches on the reduced (landmark-marginalized) pose
model. Each pose grows its own patch until its C3 leak row sum sum_j a_ij <= r. Theorem 3.5 allows any patch
per node, and the row-sum condition certifies rho(A) <= r locally (Collatz-Wielandt with w = 1), so no global
eigenvalue computation is needed. Growth: start from {i} and its strongest neighbours; at each step add the
exterior blocks whose coefficient is at least half of the largest (ties included), until the row sum is <= r
or the patch is the whole graph.

Compares, for r in {0.3, 0.5, 0.8}: patch sizes, rho(A), max row sum, certificate/true-error ratios (entries
with error > 10x floor), certified stop fractions (S_lo from the trivial and patch lower bounds only) against
the oracle, with the fixed-radius K = 1, 2 covisibility patches as reference.

Usage: python3 scripts/explore_greedy.py dataset [workers]
"""

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

_W = {}


def grow(i, r, max_steps=200):
    red, met, G = _W["red"], _W["met"], _W["G"]
    n = red.n_blocks
    P = np.array([i])
    steps = 0
    while True:
        p = certs.patch_one(red, met, i, 0, G, want_cut=False, P=P, extended=False)
        rs = float(p["a_vals"].sum())
        if rs <= r or len(P) >= n or steps >= max_steps:
            p["row_sum"] = rs
            p["steps"] = steps
            return p
        a = p["a_vals"]
        add = p["a_cols"][a >= 0.5 * a.max()]
        P = np.union1d(P, add)
        steps += 1


def _worker(args):
    blocks, r = args
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        out = []
        for i in blocks:
            p = grow(int(i), r)
            # final patch: recompute with extended-precision refinement (the certificate's s_i and inv_ii)
            q = certs.patch_one(_W["red"], _W["met"], int(i), 0, _W["G"], want_cut=False, P=model_ball_set(p), extended=True)
            q["row_sum"], q["steps"] = p["row_sum"], p["steps"]
            out.append(q)
        return out


def model_ball_set(p):
    # the patch of a patch_one result = {i} + interior; reconstruct from the stored n_patch is not possible,
    # so grow() results carry the block set explicitly
    return p["P"]


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
    sysm = SPDSystem(red.H, red.dims, None, refine=3, K_lo=red.H_lo)
    met = certs.Metric(red)
    Hd = red.H.tocsr()
    curve = []
    for s0 in range(0, len(avail), 200):
        Xc = np.stack([np.asarray(Z[k])[xcols] for k in range(s0, min(len(avail), s0 + 200))], axis=1)
        curve.append(met.norms(sysm.solve(Hd @ Xc - red.b[:, None], extended=False, refine=1)).max(axis=0))
    floor = float(floor_and_rate(avail, np.concatenate(curve))["floor"])
    want = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 1000]
    iters = [t for t in want if t in set(avail.tolist())]
    row = {int(t): k for k, t in enumerate(avail)}
    X = np.stack([np.asarray(Z[row[t]])[xcols] for t in iters], axis=1)
    G = red.residual(X)
    err = met.norms(sysm.solve(G))
    gate = (err > 10 * floor) & (err >= 1e-12)
    q = np.array([radii.q(int(d)) for d in red.dims])
    Sig = sysm.solve(np.eye(red.N), extended=False, refine=2)
    lam_min = np.array([np.linalg.eigvalsh(met.C[i].T @ Sig[red.block_slice(i), red.block_slice(i)] @ met.C[i])[0] for i in range(n)])
    del Sig
    oracle_stop = [float(np.mean(err[:, j] <= radii.ETA * np.sqrt(lam_min) * q)) for j in range(len(iters))]
    _W.update(red=red, met=met, G=G)
    out = dict(dataset=name, n=n, floor=floor, iters=iters, oracle_stop=oracle_stop, runs={})
    print("%s: %d poses, floor %.3e; oracle stop %s at %s" % (name, n, floor, np.round(oracle_stop, 2), iters), flush=True)

    def evaluate(tag, P):
        r3 = certs.c3_from_patches(red, P)
        sizes = np.array([p["n_patch"] for p in P])
        rs = np.array([p["a_vals"].sum() for p in P])
        mu = np.array([max(1.0, covbounds.mu_extremes(met.C[p["i"]], p["inv_ii"])[0]) for p in P])
        S_lo = np.sqrt(mu) * q
        rec = dict(patch_median=float(np.median(sizes)), patch_p90=float(np.percentile(sizes, 90)), patch_max=int(sizes.max()),
                   rho=r3["rho"], row_sum_max=float(rs.max()))
        if r3["E"] is not None:
            Ec = r3["E"]
            rat = (Ec / err)[gate]
            rec.update(valid=bool(np.all(Ec >= err * (1 - 1e-9))), ratio_median=float(np.median(rat)), ratio_p95=float(np.percentile(rat, 95)),
                       ratio_max=float(rat.max()), stop=[float(np.mean(Ec[:, j] <= radii.ETA * S_lo)) for j in range(len(iters))])
        out["runs"][tag] = rec
        print("%-10s patch median %5.1f (p90 %5.1f, max %3d) of %d | rho(A) %.4f max row sum %.3f%s" % (
            tag, rec["patch_median"], rec["patch_p90"], rec["patch_max"], n, rec["rho"], rec["row_sum_max"],
            "" if r3["E"] is None else " | valid %s ratio median %.3f p95 %.3f max %.2f | stop %s" % (
                rec["valid"], rec["ratio_median"], rec["ratio_p95"], rec["ratio_max"], np.round(rec["stop"], 2))), flush=True)

    for K in (1, 2):
        P = certs.patch_pass(red, met, K, G, workers=workers, chunk=4, want_cut=False)
        evaluate("K=%d" % K, P)
    for r in (0.3, 0.5, 0.8):
        t0 = time.perf_counter()
        blocks = np.arange(n)
        parts = [(blocks[s0:s0 + 4], r) for s0 in range(0, n, 4)]
        with mp.get_context("fork").Pool(workers) as pool:
            res = pool.map(_worker_greedy, parts)
        P = [p for part in res for p in part]
        evaluate("greedy r=%.1f" % r, P)
        out["runs"]["greedy r=%.1f" % r]["seconds"] = time.perf_counter() - t0
        out["runs"]["greedy r=%.1f" % r]["growth_steps_median"] = float(np.median([p["steps"] for p in P]))
    d = phaseC.OUT / "explore"
    d.mkdir(parents=True, exist_ok=True)
    (d / ("greedy_%s.json" % name)).write_text(json.dumps(out, indent=1, default=float))


def grow_set(i, r, max_steps=200):
    red, met, G = _W["red"], _W["met"], _W["G"]
    n = red.n_blocks
    P = np.array([i])
    steps = 0
    while True:
        p = certs.patch_one(red, met, i, 0, G, want_cut=False, P=P, extended=False)
        rs = float(p["a_vals"].sum())
        if rs <= r or len(P) >= n or steps >= max_steps:
            return P, steps
        a = p["a_vals"]
        P = np.union1d(P, p["a_cols"][a >= 0.5 * a.max()])
        steps += 1


def _worker_greedy(args):
    blocks, r = args
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        out = []
        for i in blocks:
            P, steps = grow_set(int(i), r)
            q = certs.patch_one(_W["red"], _W["met"], int(i), 0, _W["G"], want_cut=False, P=P, extended=True)
            q["steps"] = steps
            out.append(q)
        return out


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12)
