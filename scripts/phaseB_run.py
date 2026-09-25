"""Phase B sweep (brief §9 Phase B, Gate B): every synthetic family, every iterate, every invariant.

Families (certify/synth.py; GP, hub and point factors built and linearized by G-solver's bindings):
  pure WNOA chain       n in {20, 50, 100, 200, 500} x m/p in {0, 0.01, 0.1, 1, 10}
  chain + one hub       n in {20, 50, 100, 200}      x m/p in {0.01, 0.1, 1},    hub info h/p = 1
  chain + short tracks  n in {20, 50, 100, 200}      x m/p in {0.01, 0.1, 1, 10}, 2 points/pose, info/p = 1
  blind segment         n in {50, 100, 200}          x m/p in {0.1, 1, 10},       no measurement on 20 poses
  random SPD            30, 60, 120 blocks, 3 seeds each
Iterates: block Gauss-Seidel and block Jacobi from xbar + a random perturbation of unit size per block in
its D-metric, kept at iterations 0, 1, 2, 5, 10, 20, 50. Supernode width w = 3 (brief default).

Outputs (results/phaseB/): invariants.csv (every check, worst relative violation), problems.csv (spectral
radii and ratio statistics per problem), rhoM_curve.csv (pure chain: rho(M) vs cos(pi/(n+1))), gateB.json.
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from certify import check, synth  # noqa: E402

OUT = REPO / "results" / "phaseB"
KEEP = (0, 1, 2, 5, 10, 20, 50)


def problems():
    P = []
    for n in (20, 50, 100, 200, 500):
        for mp in (0.0, 0.01, 0.1, 1.0, 10.0):
            P.append(("chain", dict(n=n, m_over_p=mp)))
    for n in (20, 50, 100, 200):
        for mp in (0.01, 0.1, 1.0):
            P.append(("chain_hub", dict(n=n, m_over_p=mp, hub=1.0)))
    for n in (20, 50, 100, 200):
        for mp in (0.01, 0.1, 1.0, 10.0):
            P.append(("chain_tracks", dict(n=n, m_over_p=mp, short_tracks=(2, 1.0))))
    for n in (50, 100, 200):
        for mp in (0.1, 1.0, 10.0):
            P.append(("chain_blind", dict(n=n, m_over_p=mp, blind=(n // 2 - 10, 20))))
    for nb in (30, 60, 120):
        for seed in range(3):
            P.append(("random_spd", dict(n_blocks=nb, seed=seed)))
    return P


def build(fam, kw, seed):
    if fam == "random_spd":
        return synth.random_spd(kw["n_blocks"], seed=kw["seed"])
    return synth.wnoa_chain(kw["n"], kw["m_over_p"], seed=seed, hub=kw.get("hub"), short_tracks=kw.get("short_tracks"),
                            blind=kw.get("blind"))


def run_one(args):
    idx, fam, kw = args
    t0 = time.perf_counter()
    model = build(fam, kw, seed=1000 + idx)
    _, _, _, xbar = check.dense_oracle(model)
    rng = np.random.default_rng(idx)
    xs, keys = {}, []
    for kind in ("gs", "jacobi"):
        for t, x in synth.iterates(model, xbar, rng, kind=kind, keep=KEEP).items():
            keys.append((kind, t))
            xs[len(keys) - 1] = x
    label = "%s_%d" % (fam, idx)
    chk, info = check.evaluate(model, xs, label=label)
    u = info["err"]
    ok = u > 1e-12
    stats = dict(label=label, family=fam, **{"param_" + k: (str(v) if isinstance(v, tuple) else v) for k, v in kw.items()},
                 n=kw.get("n", kw.get("n_blocks")), m_over_p=kw.get("m_over_p", np.nan),
                 n_blocks=int(model.n_blocks), n_dof=int(model.N), p=model.meta.get("p"),
                 rho_M=info["rho_M"], rho_M_upper=info["rho_M_upper"], rho_Mhat=info["rho_Mhat"],
                 rho_Mhat_upper=info["rho_Mhat_upper"], e_consistency=info["e_consistency"])
    for name in ("E_C1", "E_C2"):
        if name in info:
            r = info[name][ok] / u[ok]
            stats["ratio_%s_median" % name[2:]] = float(np.median(r))
            stats["ratio_%s_p95" % name[2:]] = float(np.percentile(r, 95))
    for K, r3 in info["C3"].items():
        stats["rho_A_K%d" % K] = r3["rho"]
        if r3["E"] is not None:
            r = r3["E"][ok] / u[ok]
            stats["ratio_C3_K%d_median" % K] = float(np.median(r))
            stats["ratio_C3_K%d_p95" % K] = float(np.percentile(r, 95))
    for K, r4 in info.get("C4", {}).items():
        stats["rho_A4_K%d" % K] = r4["rho"]
    if "partition" in info:
        stats["w"] = info["partition"].w
        stats["n_hubs"] = int(len(info["partition"].S))
    if "hubs" in info:
        stats["theta_S_oracle"] = info["hubs"]["theta_oracle"]
    stats["seconds"] = time.perf_counter() - t0
    rows = [dict(r, family=fam, **{k: str(v) for k, v in kw.items()}) for r in chk.rows]
    return stats, rows


def run_safe(args):
    import traceback
    try:
        st, rows = run_one(args)
        return dict(stats=st, rows=rows)
    except Exception:
        return dict(error=traceback.format_exc(), problem=[args[0], args[1], {k: str(v) for k, v in args[2].items()}])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P = [(i, fam, kw) for i, (fam, kw) in enumerate(problems())]
    import multiprocessing as mp
    workers = int(os.environ.get("CERTIFY_WORKERS", "12"))
    res, errors = [], []
    part = OUT / "partial.jsonl"
    with open(part, "w") as fh, mp.get_context("fork").Pool(workers) as pool:
        for r in pool.imap_unordered(run_safe, P, chunksize=1):
            if "error" in r:
                errors.append(r)
            else:
                res.append((r["stats"], r["rows"]))
            fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
    if errors:
        print("ERRORS:", json.dumps(errors, indent=1, default=float))
    stats = sorted([r[0] for r in res], key=lambda s: s["label"])
    rows = [row for r in res for row in r[1]]
    import csv
    keys = sorted({k for s in stats for k in s})
    with open(OUT / "problems.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for s in stats:
            w.writerow(s)
    keys = ["label", "family", "check", "n", "worst", "tol", "ok"]
    with open(OUT / "invariants.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # rho(M) on the pure chain against cos(pi/(n+1))
    with open(OUT / "rhoM_curve.csv", "w") as f:
        f.write("n,m_over_p,rho_M,cos_pi_over_n_plus_1\n")
        for s in stats:
            if s["family"] == "chain":
                f.write("%d,%g,%.10f,%.10f\n" % (s["n"], s["m_over_p"], s["rho_M"], np.cos(np.pi / (s["n"] + 1))))
    failed = [r for r in rows if not r["ok"]]
    # aggregate: worst violation per check name (prefix without K)
    agg = {}
    for r in rows:
        agg.setdefault(r["check"], []).append(r["worst"])
    summary = dict(problems=len(stats), problem_errors=len(errors), checks=len(rows), failed=len(failed),
                   worst_by_check={k: float(np.max(v)) for k, v in sorted(agg.items())},
                   failed_rows=failed[:50])
    (OUT / "gateB.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k != "worst_by_check"}, indent=1, default=float))


if __name__ == "__main__":
    main()
