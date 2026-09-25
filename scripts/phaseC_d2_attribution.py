"""Attribution of the stereo pose floors of D2 (diagnostic, as in Phase A): with p_L = 1e-6, re-solve the
certified model with every landmark whose observation-information smallest eigenvalue is below tau held at
GBP's last frozen state, and report the pose floor against that conditional solution. If the floor
collapses, it comes from those weakly observed landmarks (GBP's fixed point differs from the model's
solution in their weak directions). Output: results/phaseC/d2/attribution_<dataset>.json."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from certify import model as mdl  # noqa: E402
from certify.linalg import SPDSystem  # noqa: E402

spec = importlib.util.spec_from_file_location("d2", REPO / "scripts" / "phaseC_d2.py")
d2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d2)

TAUS = (1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


def run(name):
    path = d2.DATASETS[name]
    m, info = mdl.from_export(path, p_L=1e-6)
    xbar = np.load(d2.OUT / ("%s_xbar_pL1e-06.npy" % name))
    ex = info["export"]
    zT = mdl.state_restrict(info, np.asarray(ex.states()[-1]))
    lm = np.flatnonzero(m.kinds == "landmark")
    obs = d2.obs_info_min_eig(m)
    poses = np.flatnonzero(m.kinds == "pose")
    CT = [np.linalg.cholesky(m.diag_blocks()[b]).T for b in poses]

    def pose_err(x):
        return np.array([np.linalg.norm(CT[k] @ (zT[m.block_slice(b)] - x[m.block_slice(b)])) for k, b in enumerate(poses)])

    e0 = pose_err(xbar)
    out = dict(dataset=name, last_iteration=int(np.load(ex.path / "states_iters.npy")[-1]),
               no_holding=dict(pose_floor=float(e0.max()), worst_pose=int(m.pose_index[poses[e0.argmax()]]),
                               median=float(np.median(e0)), n_poses_above_0_05=int(np.sum(e0 > 0.05))), held=[])
    H, b = m.H.tocsr(), m.b
    for tau in TAUS:
        held = lm[obs < tau]
        hd = m.dof_index(held) if len(held) else np.zeros(0, int)
        keep = np.setdiff1d(np.arange(m.n_blocks), held)
        kd = m.dof_index(keep)
        s = SPDSystem(H[kd][:, kd], m.dims[keep], m.elim[keep], refine=2, K_lo=m.H_lo[kd][:, kd])
        x = zT.copy()
        x[kd] = s.solve(b[kd] - H[kd][:, hd] @ zT[hd])
        e = pose_err(x)
        # the same landmarks clamped at their frozen linearization point (z = 0): a fixed model
        x0 = np.zeros_like(zT)
        x0[kd] = s.solve(b[kd])
        e0c = pose_err(x0)
        out["held"].append(dict(tau=tau, n_held=int(len(held)), pose_floor=float(e.max()),
                                worst_pose=int(m.pose_index[poses[e.argmax()]]), median=float(np.median(e)),
                                n_poses_above_0_05=int(np.sum(e > 0.05)),
                                clamped_at_linpoint_pose_floor=float(e0c.max()),
                                clamped_at_linpoint_worst_pose=int(m.pose_index[poses[e0c.argmax()]])))
        print(name, "tau", tau, out["held"][-1], flush=True)
    (d2.OUT / ("attribution_%s.json" % name)).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    for n in sys.argv[1:]:
        run(n)
