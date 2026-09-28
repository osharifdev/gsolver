"""Part A (BRIEF_stage2.md §1): the unfinished fraction at fixed budgets, from the Stage 1 logs.

At each budget B (the state after B rounds, the Stage 1 evaluation after round B), for every pose:
  sense 1, the current problem:      ||e_i||_{D_i} > eta S_lo,i   (e: the evaluation's oracle error)
  sense 2, the relinearized problem: the remaining Gauss-Newton move at the beliefs (solver's Jacobian, D_i = S_ii; the
                                     Stage 1 §3.4 quantity, column move_S) > eta S_lo,i
with eta = 0.1 and S_lo,i the evaluation's yardstick; plus the median and p95 remaining Gauss-Newton move (WP1's
definition, column move) and the aligned keyframe ATE. Budgets: 10, 50, 100, 200, 300 on the KITTI segments; 103 (the
shipped islam budget), 200, 500, 1030 on ChArUco (103 from the rerun exports/stage1/charuco/<variant>_partA, which adds an
evaluation after round 103 and reproduces Stage 1's evaluations 10-100 exactly).

Writes results/partA/<dataset>/<variant>/budgets.csv and results/partA/summary.json.
"""

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ETA = 0.1
BUDGETS = {"kitti06": (10, 50, 100, 200, 300), "kitti03": (10, 50, 100, 200, 300), "kitti04": (10, 50, 100, 200, 300),
           "kitti10": (10, 50, 100, 200, 300), "charuco": (103, 200, 500, 1030)}


def rows_of(ds, var, rnd):
    runs = [REPO / "exports/stage1" / ds / var]
    if ds == "charuco" and rnd == 103:
        runs = [REPO / "exports/stage1" / ds / (var + "_partA")]
    for run in runs:
        R = [r for r in csv.DictReader(open(run / "eval/eval_rounds.csv")) if int(r["round"]) == rnd]
        S = [r for r in csv.DictReader(open(run / "eval/eval_summary.csv")) if int(r["round"]) == rnd]
        if R:
            return R, S[0], run
    raise KeyError((ds, var, rnd))


def main():
    out = {}
    for ds, budgets in BUDGETS.items():
        for var in ("corrected", "shipped"):
            stops = {r["criterion"]: r for r in csv.DictReader(open(REPO / f"results/stage1/{ds}/{var}/stops.csv"))}
            cert = stops["certified"]["round"] or None
            recs = []
            for b in budgets:
                R, S, run = rows_of(ds, var, b)
                f = lambda k: np.array([float(r[k]) for r in R])
                err, slo, mS, mv = f("err"), f("S_lo"), f("move_S"), f("move")
                recs.append(dict(budget=b, poses=len(R), unfinished_current=float(np.mean(err > ETA * slo)),
                                 unfinished_relinearized=float(np.mean(mS > ETA * slo)), certified_pass=float(np.mean(f("E") <= ETA * slo)),
                                 move_median=float(np.median(mv)), move_p95=float(np.percentile(mv, 95)),
                                 move_true_median=float(np.median(f("move_true"))), ate_aligned=float(S["ate_aligned"] or "nan"),
                                 source=str(run.relative_to(REPO))))
            d = REPO / "results/partA" / ds / var
            d.mkdir(parents=True, exist_ok=True)
            with open(d / "budgets.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
                w.writeheader()
                w.writerows(recs)
            out["%s/%s" % (ds, var)] = dict(certified_stop=cert, budgets=recs)
            print(ds, var, "certified", cert, " ".join("B%d: %.3f/%.3f med %.3g p95 %.3g ATE %.4f" % (
                r["budget"], r["unfinished_current"], r["unfinished_relinearized"], r["move_median"], r["move_p95"], r["ate_aligned"])
                for r in recs))
    (REPO / "results/partA/summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
