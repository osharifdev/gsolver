"""Stage 2c: descriptive statistics of the clean-model runs (diagnosis only; from the Step 2 logs and the kept round-100 state).

Per corrected (and shipped) run: the oracle error ||e_i||_{D_i} / (eta S_lo,i) at round 300 (median, min, max, and its growth
along the window: first vs last ten poses), the largest belief change per round (change criterion) at rounds 100, 200, 300
and its geometric decay rate fitted over rounds 100-300, the variables relinearized per round at round 300, and at the
round-100 state (corrected) the share of the oracle error, in the S-norm, that is a rigid motion of the whole window
(the 6-dimensional gauge directions of the poses).
Writes results/stage2c/dynamics.json.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from certify.io import Export  # noqa: E402
from certify.sidecar_model import PoseModelBuilder  # noqa: E402

DS = ["kitti06", "kitti03", "kitti04", "kitti10", "prefix"]


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rigid_share(path):
    ex = Export(path)
    m = PoseModelBuilder(p_L=1e-6, g2o=str(REPO / ex.meta["g2o"])).build(None, ex=ex)
    S = m["S"]
    e = np.linalg.solve(S, S @ m["x"] - m["b"])
    pb = np.flatnonzero(ex.var_type == "SE3PoseVel")
    row = {int(b): k for k, b in enumerate(pb)}
    n = len(m["pose_ids"])
    G = np.zeros((12 * n, 6))
    for i, b in enumerate(m["pose_blocks"]):
        T = ex.pose_T_lin[row[int(b)]]
        R, t = T[:3, :3], T[:3, 3]
        G[12 * i:12 * i + 3, :3] = R.T
        G[12 * i:12 * i + 3, 3:] = -R.T @ skew(t)
        G[12 * i + 3:12 * i + 6, 3:] = R.T
    xi = np.linalg.solve(G.T @ S @ G, G.T @ S @ e)
    r = e - G @ xi
    return float(1 - r @ S @ r / (e @ S @ e))


def main():
    out = {}
    for ds in DS:
        for var in ("corrected", "shipped"):
            d = REPO / "results/stage2c" / ds / var
            rows = [r for r in csv.DictReader(open(d / "eval_rounds.csv")) if r.get("decision") not in ("A0_FAILS", "NONFINITE")]
            last = max(int(r["round"]) for r in rows)
            X = [r for r in rows if int(r["round"]) == last]
            ratio = np.array([float(r["err"]) / (0.1 * float(r["S_lo"])) for r in X])
            he = {int(r["round"]): r for r in csv.DictReader(open(d / "heuristics.csv"))}
            mc = lambda t: float(he[t]["max_change"]) if t in he and he[t]["max_change"] not in ("null", "") else np.nan
            ts = np.arange(100, min(300, max(he)) + 1)
            y = np.array([mc(t) for t in ts])
            ok = np.isfinite(y) & (y > 0)
            rate = float(np.exp(np.polyfit(ts[ok], np.log(y[ok]), 1)[0])) if ok.sum() > 10 and last == 300 else None  # none for a diverged run
            rl = {int(r["round"]): r for r in csv.DictReader(open(REPO / "exports/stage2c" / ds / var / "relin_rounds.csv"))}
            r300 = rl.get(300, {})
            num = lambda k: float(r300[k]) if r300.get(k) not in (None, "", "null") else np.nan
            o = dict(last_certifiable_round=last, err_over_eta_S_lo=dict(median=float(np.median(ratio)), min=float(ratio.min()), max=float(ratio.max()),
                                                                          first_ten_median=float(np.median(ratio[:10])), last_ten_median=float(np.median(ratio[-10:]))),
                     max_belief_change={str(t): mc(t) for t in (100, 200, 300)}, max_change_decay_per_round_100_300=rate,
                     relinearized_at_300=dict(poses=num("relin_poses"), other=num("relin_other"), d_pose_median=num("d_pose_median")))
            if var == "corrected":
                o["rigid_motion_share_of_error_round100"] = rigid_share(REPO / "exports/stage2c" / ds / var / "eval/model_r00100")
            out["%s/%s" % (ds, var)] = o
            print(ds, var, json.dumps(o))
    (REPO / "results/stage2c/dynamics.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
