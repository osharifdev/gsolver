"""I1b (progress log 18:26): decision-driven patch growth on the reduced (landmark-marginalized) pose model.

Theorem 3.5 holds for any patches fixed at the time the certificate is evaluated, so a patch may depend on the
current residual. Pose i starts from its certificate-driven patch (grown until its leak row sum is <= r, I1) and,
at the evaluated iteration t, keeps growing only while its stopping decision is open:

    base certificate c0 = C3 with the base patches (valid);
    repeat on the current patch P:  s_i, a_ij (Definition 3.4), thr_i = eta sqrt(max(1, mu_lo(P))) q_d(delta),
        predicted E_i ~ s_i + sum_j a_ij c0_j;
        stop growing if the row sum is <= r and (predicted <= thr_i [decision: stop]
                                                  or s_i >= thr_i [decision: continue]
                                                  or leak <= theta s_i [nothing left to gain]),
        or at the size cap; else add the exterior blocks with a_ij c0_j >= half of the largest.

The final certificate is C3 with the grown patches (Theorem 3.5, rho(A) <= max row sum certified per node), so
validity never depends on the prediction. Parameters fixed before any run: r = 0.3, theta = 0.02, cap = 250
poses, at most 40 growth steps.

Reports per iteration: certified stop fraction against the oracle and against the base, K = 1, K = 2 patches;
patch size (median, mean, p90, max) and mean |P|^3 relative to K = 2 (factorization cost); validity; ratios.

Usage: python3 scripts/explore_decision.py dataset [workers]
"""

import csv
import gzip
import importlib.util
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

spec = importlib.util.spec_from_file_location("greedy", REPO / "scripts" / "explore_greedy.py")
greedy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(greedy)

R_BASE, THETA, CAP, MAX_STEPS = 0.3, 0.02, 250, 40
ITERS = dict(kitti06_full_conv=[1, 10, 20, 30, 50, 70, 100, 150, 200],
             kitti06_prefix_conv=[1, 2, 3, 5, 10, 20, 50],
             kitti06_prefix_init=[3, 5, 7, 10, 20, 50, 100])
_W = {}


def mu_lo(i, inv_ii):
    return max(1.0, covbounds.mu_extremes(_W["met"].C[i], inv_ii)[0])


def grow_decision(i, t):
    red, met, G, c0, q = _W["red"], _W["met"], _W["G"], _W["c0"], _W["q"]
    g = G[:, [t]]
    P = _W["base_sets"][i]
    steps, why = 0, "cap"
    while True:
        p = certs.patch_one(red, met, i, 0, g, want_cut=False, P=P, extended=False)
        a, cols = p["a_vals"], p["a_cols"]
        rs = float(a.sum())
        w = a * c0[cols, t]
        leak = float(w.sum())
        s = float(p["s"][0])
        thr = radii.ETA * np.sqrt(mu_lo(i, p["inv_ii"])) * q[i]
        if rs <= R_BASE:
            if s + leak <= thr:
                why = "stop"
                break
            if s >= thr:
                why = "continue"
                break
            if leak <= THETA * s:
                why = "tight"
                break
        if len(P) >= min(CAP, red.n_blocks) or steps >= MAX_STEPS or not len(cols):
            break
        score = w if rs <= R_BASE else a
        P = np.union1d(P, cols[score >= 0.5 * score.max()])
        steps += 1
    return P, steps, why


def _worker(tasks):
    from threadpoolctl import threadpool_limits
    out = []
    with threadpool_limits(1):
        for i, t in tasks:
            P, steps, why = grow_decision(int(i), int(t))
            p = certs.patch_one(_W["red"], _W["met"], int(i), 0, _W["G"][:, [t]], want_cut=False, P=P, extended=True)
            p.update(steps=steps, why=why, t=int(t))
            out.append(p)
    return out


def _base_worker(blocks):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        return [greedy.grow_set(int(i), R_BASE)[0] for i in blocks]


def main(name, workers=12):
    import multiprocessing as mp
    t_start = time.perf_counter()
    path, cfg = phaseC.DATASETS[name]
    full, info = mdl.from_export(path, p_L=cfg["p_L"])
    red, kept = mdl.reduce_blocks(full)
    n = red.n_blocks
    ex = info["export"]
    Z = ex.states()
    avail = np.load(Path(path) / "states_iters.npy")
    xcols = np.concatenate([np.arange(ex.offsets[b], ex.offsets[b] + ex.dims[b]) for b in info["kept_export_blocks"][kept]])
    iters = [t for t in ITERS[name] if t in set(avail.tolist())]
    row = {int(t): k for k, t in enumerate(avail)}
    X = np.stack([np.asarray(Z[row[t]])[xcols] for t in iters], axis=1)
    sysm = SPDSystem(red.H, red.dims, None, refine=3, K_lo=red.H_lo)
    met = certs.Metric(red)
    G = red.residual(X)
    err = met.norms(sysm.solve(G))
    # exact lambda_min and the floor, from the pose-model run (scripts/phaseC_reduced.py)
    summ = json.loads((phaseC.OUT / "reduced" / ("%s.json" % name)).read_text())
    floor = summ["floor"]["floor"]
    bid = {str(b): k for k, b in enumerate(red.block_ids)}
    lam_min = np.full(n, np.nan)
    for r in csv.DictReader(gzip.open(phaseC.OUT / "reduced" / ("%s_certs.csv.gz" % name), "rt")):
        lam_min[bid[r["block_id"]]] = float(r["lam_min_oracle"])
    assert np.all(np.isfinite(lam_min))
    q = np.array([radii.q(int(d)) for d in red.dims])
    thr_oracle = radii.ETA * np.sqrt(lam_min) * q
    oracle = [float(np.mean(err[:, j] <= thr_oracle)) for j in range(len(iters))]
    gate = (err > 10 * floor) & (err >= 1e-12)
    print("%s: %d poses, iterations %s, oracle stop %s (%.0fs)" % (name, n, iters, np.round(oracle, 2), time.perf_counter() - t_start), flush=True)

    # base: certificate-driven patches (row sum <= R_BASE), data-independent
    greedy._W.update(red=red, met=met, G=G)
    blocks = np.arange(n)
    with mp.get_context("fork").Pool(workers) as pool:
        sets = [s for part in pool.map(_base_worker, [blocks[k:k + 4] for k in range(0, n, 4)]) for s in part]
    Pb = [certs.patch_one(red, met, i, 0, G, want_cut=False, P=sets[i], extended=True) for i in range(n)]
    rb = certs.c3_from_patches(red, Pb)
    assert rb["E"] is not None, "base patches not contractive"
    _W.update(red=red, met=met, G=G, c0=rb["E"], q=q, base_sets=sets)
    mu_b = np.array([mu_lo(p["i"], p["inv_ii"]) for p in Pb])
    stop_base = [float(np.mean(rb["E"][:, j] <= radii.ETA * np.sqrt(mu_b) * q)) for j in range(len(iters))]
    size_b = np.array([len(s) for s in sets])
    print("base (grown r=%.1f): rho %.3f, patch median %d mean %.1f; stop %s" % (
        R_BASE, rb["rho"], np.median(size_b), size_b.mean(), np.round(stop_base, 2)), flush=True)

    # decision-driven growth at every evaluated iteration
    tasks = [(i, j) for j in range(len(iters)) for i in range(n)]
    rng = np.random.default_rng(0)
    rng.shuffle(tasks)
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(workers) as pool:
        res = [p for part in pool.map(_worker, [tasks[k:k + 8] for k in range(0, len(tasks), 8)]) for p in part]
    t_grow = time.perf_counter() - t0
    out = dict(dataset=name, n=n, iters=iters, floor=floor, params=dict(r=R_BASE, theta=THETA, cap=CAP, max_steps=MAX_STEPS),
               oracle_stop=oracle, base=dict(rho=rb["rho"], stop=stop_base, patch_median=float(np.median(size_b)),
                                             patch_mean=float(size_b.mean())), per_iter=[], seconds_grow=t_grow)
    E = np.full((n, len(iters)), np.nan)
    size = np.zeros((n, len(iters)), int)
    stop = []
    for j, t in enumerate(iters):
        Pj = sorted([p for p in res if p["t"] == j], key=lambda p: p["i"])
        rj = certs.c3_from_patches(red, Pj)
        sz = np.array([p["n_patch"] for p in Pj])
        why = [p["why"] for p in Pj]
        mu = np.array([mu_lo(p["i"], p["inv_ii"]) for p in Pj])
        rec = dict(iter=t, rho=rj["rho"], row_sum_max=rj["row_sum_max"], patch_median=float(np.median(sz)), patch_mean=float(sz.mean()),
                   patch_p90=float(np.percentile(sz, 90)), patch_max=int(sz.max()),
                   reasons={k: why.count(k) for k in ("stop", "continue", "tight", "cap")},
                   steps_mean=float(np.mean([p["steps"] for p in Pj])))
        if rj["E"] is not None:
            e = rj["E"][:, 0]
            E[:, j] = e
            size[:, j] = sz
            rec.update(valid=bool(np.all(e >= err[:, j] * (1 - 1e-9))), worst_margin=float(np.max((err[:, j] - e) / np.maximum(e, 1e-300))),
                       stop=float(np.mean(e <= radii.ETA * np.sqrt(mu) * q)))
            # two-sided: ||e_i|| >= s_i - sum_j a_ij u_j >= s_i - (A c*)_i  (triangle inequality, u <= c*)
            lo = np.maximum(0.0, rj["s"][:, 0] - rj["A"] @ e)
            rec.update(lower_valid=bool(np.all(lo <= err[:, j] * (1 + 1e-9))),
                       lower_worst_margin=float(np.max((lo - err[:, j]) / np.maximum(err[:, j], 1e-300))))
            g = gate[:, j]
            if g.any():
                rat = e[g] / err[g, j]
                rec.update(ratio_median=float(np.median(rat)), ratio_p95=float(np.percentile(rat, 95)), ratio_max=float(rat.max()))
                gl = g & (lo > 0)
                if gl.any():
                    rec.update(self_ratio_median=float(np.median(e[gl] / lo[gl])), self_ratio_p95=float(np.percentile(e[gl] / lo[gl], 95)),
                               lower_positive=float(np.mean(lo[g] > 0)))
        stop.append(rec.get("stop", np.nan))
        out["per_iter"].append(rec)
        print("it %4d: rho %.3f max row %.3f | patch median %5.1f mean %5.1f p90 %5.1f max %3d | reasons %s | valid %s ratio median %s "
              "(self-assessed E/L median %s, lower valid %s) | stop %.2f (base %.2f, oracle %.2f)" % (
                  t, rec["rho"], rec["row_sum_max"], rec["patch_median"], rec["patch_mean"], rec["patch_p90"], rec["patch_max"], rec["reasons"],
                  rec.get("valid"), None if "ratio_median" not in rec else round(rec["ratio_median"], 3),
                  None if "self_ratio_median" not in rec else round(rec["self_ratio_median"], 3), rec.get("lower_valid"),
                  rec.get("stop", np.nan), stop_base[j], oracle[j]), flush=True)
    # cost proxy: mean |P|^3 relative to the K = 2 hop patches of the pose-model run
    kp = summ["K"]
    out["cost_note"] = "K=1 / K=2 patch medians from the pose-model run: %s / %s" % (kp["1"]["patch_blocks"]["median"], kp["2"]["patch_blocks"]["median"])
    out["seconds"] = time.perf_counter() - t_start
    d = phaseC.OUT / "explore"
    d.mkdir(parents=True, exist_ok=True)
    (d / ("decision_%s.json" % name)).write_text(json.dumps(out, indent=1, default=float))
    print("done in %.0fs (growth %.0fs)" % (out["seconds"], t_grow), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12)
