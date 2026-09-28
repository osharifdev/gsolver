"""Stage 1 §2.5: the never-finishes result with the corrected yardstick.

At a belief-chart export (the shipped schedule's KITTI 06 40-keyframe state after 500 rounds; a certified-stop state):
the remaining Gauss-Newton move ||xbar_i||_{D_i} (xbar = S^-1 b with the solver's own Jacobian, D_i = S_ii: the sidecar's
conventions, as in the WP2 side computation) against eta S_lo,i, with S_lo,i on the base patch of the true-Jacobian Fisher
information (stereo and GP, Theorem T0; the export's f_*_Jtrue.npy) for every eta of §2.4, and, for comparison, the WP2
yardstick (stereo rotation columns halved, GP Jacobian as the solver's) and the exact yardstick. For the shipped state also
the change-based criterion's verdict at round 500 for every tau (heuristics.csv of the rerun).

Usage: python3 scripts/stage1_never_finishes.py <out.json> shipped500=<run dir> [certified_<variant>=<run dir> ...]
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as sla

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from certify.io import Export  # noqa: E402
from certify.sidecar import DOF, Q12, Engine, Ops  # noqa: E402
from certify.sidecar_model import PoseModelBuilder  # noqa: E402

ETAS = (0.05, 0.1, 0.2, 0.5)


def fractions(export_dir):
    ex = Export(export_dir)
    B = PoseModelBuilder(p_L=1e-6, g2o=str(REPO / ex.meta["g2o"]))
    m = B.build(None, ex=ex)
    S = m["S"]
    n = len(m["pose_ids"])
    assert np.abs(m["x"]).max() == 0.0
    xbar = sla.cho_solve(sla.cho_factor(S, lower=True), m["b"])
    C = [np.linalg.cholesky(S[DOF * i:DOF * i + DOF, DOF * i:DOF * i + DOF]) for i in range(n)]
    Cinv = [sla.solve_triangular(c, np.eye(DOF), lower=True) for c in C]
    move = np.array([np.linalg.norm(C[i].T @ xbar[DOF * i:DOF * i + DOF]) for i in range(n)])
    blocks = S.reshape(n, DOF, n, DOF)
    adj = np.abs(blocks).max(axis=(1, 3)) > 0
    np.fill_diagonal(adj, False)
    eng = Engine(0.003, 0.1)
    eng._fcache, eng._fcacheF = {}, {}
    ops = Ops()
    P = [eng._grow_base(S, -m["b"], C, Cinv, adj, i, np.array([i]), ops)[0].P for i in range(n)]
    out = dict(export=str(export_dir), poses=n, true_jacobians_solver=bool(ex.meta.get("true_jacobians", False)),
               move_median=float(np.median(move)), move_p95=float(np.percentile(move, 95)), move_max=float(move.max()))
    yard = {}
    variants = dict(corrected=dict(true_jacobian=True),
                    wp2=dict(true_jacobian=True, J_override={"gp": ex.factors["gp"]["J"]}) if "gp" in ex.factors else None)
    for name, kw in variants.items():
        if kw is None:
            continue
        SF = B.build(None, ex=ex, **kw)["S"]
        eng._fcacheF = {}  # the yardstick factorizations are cached by patch: never reuse them across Fisher informations
        S_lo = np.array([eng._yardstick(SF, C, i, P[i], ops)[0] for i in range(n)])
        SFi = np.linalg.inv(SF)
        S_ex = np.array([np.sqrt(max(0.0, np.linalg.eigvalsh(C[i].T @ SFi[DOF * i:DOF * i + DOF, DOF * i:DOF * i + DOF] @ C[i])[0])) * Q12
                         for i in range(n)])
        yard[name] = dict(S_lo_median=float(np.median(S_lo)), S_exact_median=float(np.median(S_ex)),
                          frac_move_gt_eta_S_lo={str(e): float(np.mean(move > e * S_lo)) for e in ETAS},
                          frac_move_gt_eta_S_exact={str(e): float(np.mean(move > e * S_ex)) for e in ETAS},
                          ratio_move_over_eta01_S_lo=dict(median=float(np.median(move / (0.1 * S_lo))), max=float(np.max(move / (0.1 * S_lo)))))
    out["yardstick"] = yard
    return out


def change_verdict(run_dir, at_round):
    rows = list(csv.DictReader(open(Path(run_dir) / "heuristics.csv")))
    r = next(x for x in rows if int(x["round"]) == at_round)
    first = {}
    for tau, col in (("1e-2", "below_0.01"), ("1e-3", "below_0.001"), ("1e-4", "below_0.0001")):
        first[tau] = next((int(x["round"]) for x in rows if x[col] == "1"), None)
    return dict(round=at_round, max_change=float(r["max_change"]), argmax=r["argmax_id"],
                stops_at_this_round={tau: float(r["max_change"]) < float(tau) for tau in first}, first_round_below=first)


def main(out, specs):
    res = {}
    for spec in specs:
        name, d = spec.split("=", 1)
        d = Path(d)
        if name == "shipped500":
            res[name] = fractions(d / "final")
            res[name]["change_criterion"] = change_verdict(d, 500)
            old = REPO / "exports/phaseD/wp1b/diag/default/final"
            if old.exists():
                a, b = Export(old), Export(d / "final")
                res[name]["identical_to_wp1b_state"] = bool(np.array_equal(a.pose_T_lin, b.pose_T_lin)
                                                            and np.array_equal(a.point_p_lin, b.point_p_lin))
        else:
            stop = d / "eval" / "certified_stop_beliefs"
            if not stop.exists():
                res[name] = dict(note="no certified stop in the run")
                continue
            res[name] = fractions(stop)
            res[name]["round"] = int((stop / "ROUND").read_text())
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
