"""Phase D, WP2 side computation (the WP2 amendment): on the shipped schedule's KITTI-40 state at round 500 (WP1b
diagnostic, belief-chart export), the fraction of poses whose remaining Gauss-Newton move ||xbar_i||_{D_i} exceeds
eta S_lo,i, with the sidecar's conventions (reduced pose model, D_i = S_ii, S_lo,i from the true-Jacobian Fisher
information on the base patch, q_12(0.05), eta = 0.1), and with the exact yardstick for comparison.
A self-consistency statement about the shipped fixed point, not an accuracy claim.

Usage: python3 scripts/phaseD_wp2_selfconsistency.py <export_dir> <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as sla

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from certify.io import Export  # noqa: E402
from certify.sidecar import Engine, Ops, Q12, DOF  # noqa: E402
from certify.sidecar_model import PoseModelBuilder  # noqa: E402

path, out = sys.argv[1], sys.argv[2]
ex = Export(path)
B = PoseModelBuilder(p_L=1e-6, g2o=str(REPO / ex.meta["g2o"]))
m, mF = B.build(None, ex=ex), B.build(None, true_jacobian=True, ex=ex)
S, SF = m["S"], mF["S"]
n = len(m["pose_ids"])
assert np.abs(m["x"]).max() == 0.0  # belief-chart export: the state is the linearization point
xbar = sla.cho_solve(sla.cho_factor(S, lower=True), m["b"])
C = [np.linalg.cholesky(S[DOF * i:DOF * i + DOF, DOF * i:DOF * i + DOF]) for i in range(n)]
Cinv = [sla.solve_triangular(c, np.eye(DOF), lower=True) for c in C]
move = np.array([np.linalg.norm(C[i].T @ xbar[DOF * i:DOF * i + DOF]) for i in range(n)])
eng = Engine(0.003, 0.1)
eng._fcache, eng._fcacheF = {}, {}
blocks = S.reshape(n, DOF, n, DOF)
adj = np.abs(blocks).max(axis=(1, 3)) > 0
np.fill_diagonal(adj, False)
g = -m["b"]
ops = Ops()
S_lo = np.zeros(n)
for i in range(n):
    row, _ = eng._grow_base(S, g, C, Cinv, adj, i, np.array([i]), ops)
    S_lo[i] = eng._yardstick(SF, C, i, row.P, ops)[0]
SFi = np.linalg.inv(SF)
S_ex = np.array([np.sqrt(max(0.0, np.linalg.eigvalsh(C[i].T @ SFi[DOF * i:DOF * i + DOF, DOF * i:DOF * i + DOF] @ C[i])[0])) * Q12 for i in range(n)])
res = dict(export=path, poses=n, move_median=float(np.median(move)), move_p95=float(np.percentile(move, 95)), move_max=float(move.max()),
           S_lo_median=float(np.median(S_lo)), S_exact_median=float(np.median(S_ex)),
           frac_move_gt_eta_S_lo=float(np.mean(move > 0.1 * S_lo)), frac_move_gt_eta_S_exact=float(np.mean(move > 0.1 * S_ex)),
           ratio_move_to_eta_S_lo_median=float(np.median(move / (0.1 * S_lo))), ratio_move_to_eta_S_lo_max=float(np.max(move / (0.1 * S_lo))),
           worst_pose_time=float(m["times"][int(np.argmax(move / (0.1 * S_lo)))]))
Path(out).write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
