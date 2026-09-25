"""Exploration (brief §12: "If psibar ~ 1 on long segments ... rerun with K = 4, 5"): the recursive patch
certificate C3 on one export for K = 1..Kmax. For each K: rho(A) (and its Collatz-Wielandt upper bound), the
patch size (median and max blocks, and the fraction of the graph covered), the certificate/true-error ratio
where valid (entries with true error > 10x the pose floor), the local share s/E, and the fraction of poses
that satisfy the stopping rule E_i <= 0.1 S_lo_i at each evaluated iteration (S_lo from the best of the
trivial and patch lower bounds at that K).

Usage: python3 scripts/explore_K.py dataset Kmax [workers]
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from certify import certs, radii  # noqa: E402
from certify import phaseC  # noqa: E402
from certify import model as mdl  # noqa: E402
from certify.linalg import SPDSystem  # noqa: E402


def main(name, Kmax, workers):
    path, cfg = phaseC.DATASETS[name]
    model, info = mdl.from_export(path, p_L=cfg["p_L"])
    d2 = json.loads((phaseC.OUT / "d2" / ("%s.json" % name)).read_text())
    mrec = [m for m in d2["models"] if m["p_L"] == cfg["p_L"]][0]
    floor = float(mrec["pose_floor"]["floor"])
    iters = phaseC.eval_iterations(np.load(Path(path) / "states_iters.npy"), mrec["pose_floor"]["t_floor"])
    sysm = SPDSystem(model.H, model.dims, model.elim, refine=3, K_lo=model.H_lo)
    X = phaseC.load_states(model, info, iters)
    G = model.residual(X)
    E = sysm.solve(G)
    met = certs.Metric(model)
    err = met.norms(E)
    poses = model.kinds == "pose"
    gate = (err > 10 * floor) & (err >= 1e-12)
    q = np.array([radii.q(int(d)) for d in model.dims])
    out = dict(dataset=name, n_blocks=int(model.n_blocks), n_poses=int(poses.sum()), floor=floor, iters=iters, K={})
    for K in range(1, Kmax + 1):
        P = certs.patch_pass(model, met, K, G, workers=workers, chunk=8, want_cut=False)
        r = certs.c3_from_patches(model, P)
        sizes = np.array([p["n_patch"] for p in P])
        mu_lo = np.array([max(1.0, float(np.linalg.eigvalsh(met.C[p["i"]].T @ p["inv_ii"] @ met.C[p["i"]])[0])) for p in P])
        S_lo = np.sqrt(mu_lo) * q
        rec = dict(rho=r["rho"], rho_upper=r["rho_upper"], row_sum_max=r["row_sum_max"],
                   patch_blocks_median=float(np.median(sizes)), patch_blocks_max=int(sizes.max()),
                   patch_fraction_median=float(np.median(sizes) / model.n_blocks),
                   S_lo_poses_median=float(np.median(S_lo[poses])))
        if r["E"] is not None:
            Ec = r["E"]
            rat = (Ec / err)[gate]
            rp = (Ec / err)[gate & poses[:, None]]
            rec.update(ratio_median=float(np.median(rat)) if len(rat) else None,
                       ratio_p95=float(np.percentile(rat, 95)) if len(rat) else None,
                       ratio_poses_median=float(np.median(rp)) if len(rp) else None,
                       ratio_poses_p95=float(np.percentile(rp, 95)) if len(rp) else None,
                       local_share_median=float(np.median((r["s"] / np.maximum(Ec, 1e-300))[gate])) if gate.any() else None,
                       stopped_poses=[float(np.mean(Ec[poses, j] <= radii.ETA * S_lo[poses])) for j in range(len(iters))],
                       valid=bool(np.all(Ec >= err * (1 - 1e-9))))
        out["K"][K] = rec
        print(name, "K=%d rho(A)=%.4f patch median %.0f blocks (%.0f%% of graph), valid cert: %s%s" % (
            K, r["rho"], np.median(sizes), 100 * np.median(sizes) / model.n_blocks, r["E"] is not None,
            "" if r["E"] is None else ", ratio poses median %.3f p95 %.3f, stopped at iters %s: %s" % (
                rec["ratio_poses_median"] or np.nan, rec["ratio_poses_p95"] or np.nan, iters[:12],
                [round(x, 2) for x in rec["stopped_poses"][:12]])), flush=True)
        if r["E"] is not None and rec["patch_fraction_median"] >= 0.999:
            break
    d = phaseC.OUT / "explore"
    d.mkdir(parents=True, exist_ok=True)
    (d / ("K_%s.json" % name)).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 8)
