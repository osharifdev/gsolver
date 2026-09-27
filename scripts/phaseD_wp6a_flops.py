"""Phase D, WP6a: GBP flops per round against the certificate's (certify/FLOPS.md).

Inputs: the threshold-mode runs with --count-flops (exports/phaseD/wp6/flops/<run>: no sleeping, the WP1b setting
delta = 0.003, R = 1), the WP2 sidecar logs (results/phaseD/wp2/<run>/summary.csv: certificate flops per round, growth
rule as run in WP2), and the WP6b re-certification of the persisted rounds with the corrected growth rule
(results/phaseD/wp6/probe/<run>/probe_checks_r*.json).

Writes results/phaseD/wp6/flops/<run>/flops.csv (per round: GBP flops by category and message counts, the WP2
certificate flops of the same round and the ratio) and results/phaseD/wp6/flops/summary.json, which also holds:
  - the arithmetic for a variable-to-factor message formed as belief minus the one incoming message (the usual GBP
    implementation, O(degree) per variable instead of G-solver's O(degree^2) re-summation), from the degrees;
  - relinearization: operations per round, and an upper bound on their flops from the hand count of the stereo factor's
    updateErrorAndJacobian (665 flops; certify/FLOPS.md) and an order-of-magnitude allowance of 1e4 per GP factor.
"""

import csv
import glob
import json
import shutil
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUNS = {"charuco": "charuco", "kf40": "kf40", "prefix": "prefix"}
CERT_KEYS = ("factor_flops", "solve_flops", "matvec_flops", "sweep_flops", "yardstick_flops")
STEREO_LIN = 665
GP_LIN_ALLOWANCE = 1.0e4


def inv(n):
    return 2 * n ** 3


def v2f_belief_minus_incoming(d):
    # Lambda' = Lambda_b - Lambda_in (d^2), eta' = eta_b - Lambda_in (tau_in - z) (d + 2d^2 + d), + eps I (d),
    # delta = Lambda'^-1 eta' (2d^3 + 2d^2), tau = z + delta (d)
    return d * d + (d + 2 * d * d + d) + d + inv(d) + 2 * d * d + d


def v2f_shipped(d, deg):
    return (deg - 1) * (d + 2 * d * d + d + d * d) + d + inv(d) + 2 * d * d + d


def main():
    out = REPO / "results/phaseD/wp6/flops"
    summary = {}
    for run, wp2 in RUNS.items():
        src = REPO / "exports/phaseD/wp6/flops" / run
        dst = out / run
        dst.mkdir(parents=True, exist_ok=True)
        F = list(csv.DictReader(open(src / "flops.csv")))
        cert = {int(r["round"]): sum(float(r[k] or 0) for k in CERT_KEYS) for r in csv.DictReader(open(REPO / f"results/phaseD/wp2/{wp2}/summary.csv"))}
        rows = []
        for r in F:
            k = int(r["round"])
            x = dict(r)
            x["wp2_certificate_flops"] = cert.get(k, "")
            x["certificate_over_gbp"] = cert[k] / float(r["gbp_flops"]) if k in cert and float(r["gbp_flops"]) > 0 else ""
            rows.append(x)
        with open(dst / "flops.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        for f in ("flops_by_class.csv", "flops_trace.csv", "work.csv", "relin_rounds.csv", "run.json"):
            if (src / f).exists():
                shutil.copy(src / f, dst / f)
        g = np.array([float(r["gbp_flops"]) for r in F])
        c = np.array([cert[int(r["round"])] for r in F if int(r["round"]) in cert])
        ratio = np.array([float(x["certificate_over_gbp"]) for x in rows if x["certificate_over_gbp"] != ""])
        cats = {k: float(np.sum([float(r[k]) for r in F]) / g.sum()) for k in ("factor_update", "f2v", "belief", "v2f")}
        lin = np.array([int(r["linearizations"]) for r in F])
        # the degrees of the final graph (the graph of the last rounds; ChArUco grows)
        ex = REPO / f"exports/phaseD/wp2/{wp2}/final"
        deg = np.load(ex / "degree.npy")
        types = [l.split("\t")[2] for l in (ex / "variables.tsv").read_text().splitlines()[1:]]
        dims = {"SE3PoseVel": 12, "SE3Pose": 6, "SE3Point": 3}
        shipped = sum(dg * v2f_shipped(dims[t], dg) for dg, t in zip(deg, types) if dg >= 2)
        alt = sum(dg * v2f_belief_minus_incoming(dims[t]) for dg, t in zip(deg, types) if dg >= 2)
        last = F[-1]
        v2f_last = float(last["v2f"])
        stereo_or_gp = "stereo" if run != "charuco" else "gp"
        s = dict(
            rounds=len(F),
            gbp_flops_per_round=dict(median=float(np.median(g)), max=float(g.max()), first=float(g[0]), last=float(g[-1])),
            shares=cats,
            wp2_certificate_flops_per_round=dict(median=float(np.median(c)), max=float(c.max())),
            certificate_over_gbp=dict(median=float(np.median(ratio)), max=float(ratio.max()), min=float(ratio.min())),
            totals=dict(gbp=float(g.sum()), wp2_certificate=float(c.sum()), ratio=float(c.sum() / g.sum())),
            linearizations_per_round=dict(median=float(np.median(lin)), max=int(lin.max())),
            relinearization_flops_bound_per_round_max=float(lin.max()) * max(STEREO_LIN, GP_LIN_ALLOWANCE if run == "charuco" else STEREO_LIN)
            + (GP_LIN_ALLOWANCE * 160 if run == "prefix" else GP_LIN_ALLOWANCE * 39 if run == "kf40" else 0),
            v2f_final_graph=dict(shipped_formula=float(shipped), logged_last_round=v2f_last,
                                 belief_minus_incoming=float(alt), gbp_last_round_with_alternative=float(g[-1] - v2f_last + alt)),
        )
        # the corrected growth rule on the persisted rounds (WP6b)
        pc = sorted(glob.glob(str(REPO / f"results/phaseD/wp6/probe/{run}/probe_checks_r*.json")))
        if pc:
            s["certificate_flops_corrected_rule_persisted_rounds"] = {
                str(json.load(open(p))["round"]): float(json.load(open(p))["certificate_flops"]) for p in pc}
        summary[run] = s
        print(run, json.dumps(s, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
