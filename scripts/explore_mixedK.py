"""Exploration: the recursive patch certificate (Theorem 3.5) with different patch radii per block type
(the theorem allows any patch P_i containing i). Poses get K_p hops, landmarks K_l hops. For each
combination: rho(A) (Collatz-Wielandt certified), and where rho(A) < 1 the certificate against the oracle
error (all rows valid? ratio statistics where the error exceeds 10x the pose floor), the patch sizes, and the
fraction of poses that satisfy E_i <= 0.1 S_lo_i at each iteration (S_lo from the patch lower bound).

Usage: python3 scripts/explore_mixedK.py dataset "Kp1,Kp2,..." "Kl1,Kl2,..." [workers]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from certify import certs, phaseC, radii  # noqa: E402
from certify import model as mdl  # noqa: E402
from certify.linalg import SPDSystem  # noqa: E402


def main(name, Kps, Kls, workers):
    path, cfg = phaseC.DATASETS[name]
    model, info = mdl.from_export(path, p_L=cfg["p_L"])
    d2 = json.loads((phaseC.OUT / "d2" / ("%s.json" % name)).read_text())
    mrec = [m for m in d2["models"] if m["p_L"] == cfg["p_L"]][0]
    floor = float(mrec["pose_floor"]["floor"])
    iters = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 3000]
    avail = set(np.load(Path(path) / "states_iters.npy").tolist())
    iters = [t for t in iters if t in avail]
    sysm = SPDSystem(model.H, model.dims, model.elim, refine=3, K_lo=model.H_lo)
    X = phaseC.load_states(model, info, iters)
    G = model.residual(X)
    E = sysm.solve(G)
    met = certs.Metric(model)
    err = met.norms(E)
    poses = np.flatnonzero(model.kinds == "pose")
    others = np.flatnonzero(model.kinds != "pose")
    gate = (err > 10 * floor) & (err >= 1e-12)
    q = np.array([radii.q(int(d)) for d in model.dims])
    cache = {}

    def patches(blocks, K, tag):
        key = (tag, K)
        if key not in cache:
            t0 = time.perf_counter()
            cache[key] = certs.patch_pass(model, met, K, G, blocks=blocks, workers=workers, chunk=4 if tag == "p" else 32,
                                          want_cut=False)
            print("  patches %s K=%d: %d blocks in %.0fs, median patch %d blocks" % (
                tag, K, len(blocks), time.perf_counter() - t0, np.median([p["n_patch"] for p in cache[key]])), flush=True)
        return cache[key]

    out = dict(dataset=name, floor=floor, iters=iters, runs=[])
    for Kl in Kls:
        Pl = patches(others, Kl, "l")
        for Kp in Kps:
            Pp = patches(poses, Kp, "p")
            r = certs.c3_from_patches(model, Pp + Pl)
            rec = dict(Kp=Kp, Kl=Kl, rho=r["rho"], rho_upper=r["rho_upper"],
                       pose_patch_blocks_median=float(np.median([p["n_patch"] for p in Pp])),
                       pose_patch_poses_median=float(np.median([np.sum(model.kinds[model.ball(p["i"], Kp)] == "pose") for p in Pp[:20]])))
            if r["E"] is not None:
                Ec = r["E"]
                allP = {p["i"]: p for p in Pp + Pl}
                mu = np.array([max(1.0, float(np.linalg.eigvalsh(met.C[i].T @ allP[i]["inv_ii"] @ met.C[i])[0])) for i in range(model.n_blocks)])
                S_lo = np.sqrt(mu) * q
                rat = (Ec / err)[gate[:, :]]
                rp = (Ec / err)[gate & (model.kinds == "pose")[:, None]]
                rec.update(valid=bool(np.all(Ec >= err * (1 - 1e-9))),
                           ratio_all_median=float(np.median(rat)) if len(rat) else None,
                           ratio_all_p95=float(np.percentile(rat, 95)) if len(rat) else None,
                           ratio_poses_median=float(np.median(rp)) if len(rp) else None,
                           ratio_poses_p95=float(np.percentile(rp, 95)) if len(rp) else None,
                           local_share_poses_median=float(np.median((r["s"] / Ec)[gate & (model.kinds == "pose")[:, None]])) if len(rp) else None,
                           S_lo_poses_median=float(np.median(S_lo[poses])),
                           stop_fraction=[float(np.mean(Ec[poses, j] <= radii.ETA * S_lo[poses])) for j in range(len(iters))],
                           oracle_stop_fraction=[float(np.mean(err[poses, j] <= radii.ETA * S_lo[poses])) for j in range(len(iters))])
            out["runs"].append(rec)
            print("Kp=%d Kl=%d: rho(A) = %.4f (upper %.4f) | pose patch median %.0f blocks, ~%.0f poses%s" % (
                Kp, Kl, r["rho"], r["rho_upper"], rec["pose_patch_blocks_median"], rec["pose_patch_poses_median"],
                "" if r["E"] is None else " | VALID: ratio poses median %.2f p95 %.2f, all median %.2f p95 %.2f | stop %s (oracle %s) at %s" % (
                    rec["ratio_poses_median"] or np.nan, rec["ratio_poses_p95"] or np.nan, rec["ratio_all_median"] or np.nan,
                    rec["ratio_all_p95"] or np.nan, np.round(rec["stop_fraction"], 2), np.round(rec["oracle_stop_fraction"], 2), iters)), flush=True)
    d = phaseC.OUT / "explore"
    d.mkdir(parents=True, exist_ok=True)
    (d / ("mixedK_%s.json" % name)).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1], [int(k) for k in sys.argv[2].split(",")], [int(k) for k in sys.argv[3].split(",")],
         int(sys.argv[4]) if len(sys.argv) > 4 else 8)
