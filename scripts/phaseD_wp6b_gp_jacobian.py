"""Phase D, WP6b side check: the yardstick's sensitivity to the GP factor's Jacobian.

The central-difference check (scripts/phaseD_wp6b_checks.py) finds the solver's GP Jacobian differs from the
derivative of the GP error in the first knot's angular-velocity columns, while Theorem T0 builds I_F with every
non-stereo Jacobian as the solver's. Here I_F is rebuilt with the central-difference GP Jacobians (stereo rotation
columns halved as in the sidecar) and the exact yardstick S_i = sqrt(lambda_min(C_i' [I_F^-1]_ii C_i)) q12(0.05) is
compared with the sidecar's, for every pose of one persisted export.

Usage: python3 scripts/phaseD_wp6b_gp_jacobian.py <export dir> <out json>
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from certify import radii  # noqa: E402
from certify.io import Export  # noqa: E402
from certify.nonlinear_probe import ProbeClient  # noqa: E402
from certify.sidecar_model import PoseModelBuilder  # noqa: E402

BIN = REPO / "build-certify/certify/cpp/certify_stereo"
GT = REPO / "examples/data/datasets/stereo_visual/gt/06.tum"
Q12 = radii.q(12)


def yardsticks(S, SF):
    n = S.shape[0] // 12
    Sinv = np.linalg.inv(SF)
    out = np.zeros(n)
    for i in range(n):
        C = np.linalg.cholesky(S[12 * i:12 * i + 12, 12 * i:12 * i + 12])
        Sig = Sinv[12 * i:12 * i + 12, 12 * i:12 * i + 12]
        out[i] = np.sqrt(max(0.0, np.linalg.eigvalsh(C.T @ (0.5 * (Sig + Sig.T)) @ C)[0])) * Q12
    return out


def main(ed, out):
    ex = Export(ed)
    g2o = str(REPO / ex.meta["g2o"])
    pdir = "/dev/shm/gbp_os717_probe_gpj"
    cmd = [str(BIN), "--g2o", g2o, "--gt", str(GT), "--keyframes", str(ex.meta["keyframes"]), "--config", ex.meta["config"],
           "--out", pdir + "/out", "--probe-export", str(ed), "--probe-dir", pdir]
    cl = ProbeClient(cmd, pdir)
    cl.fd_check(len(ex.factors["gp"]["J"]), 1e-6)  # every GP factor (and as many stereo ones)
    Jfd = np.load(Path(pdir) / "fd_J_gp.npy")
    cl.close()
    Jgp = ex.factors["gp"]["J"]
    assert Jfd.shape == Jgp.shape, (Jfd.shape, Jgp.shape)
    B = PoseModelBuilder(p_L=1e-6, g2o=g2o)
    model = B.build(None, ex=ex)
    F0 = B.build(None, true_jacobian=True, ex=ex)
    F1 = B.build(None, true_jacobian=True, ex=ex, J_override={"gp": Jfd})
    y0, y1 = yardsticks(model["S"], F0["S"]), yardsticks(model["S"], F1["S"])
    rel = y1 / y0 - 1.0
    dJ = np.abs(Jfd - Jgp)
    res = dict(export=str(ed), gp_factors=int(len(Jgp)), max_abs_dJ_gp=float(dJ.max()),
               max_abs_dJ_gp_outside_first_omega=float(np.delete(dJ, np.s_[9:12], axis=2).max()),
               yardstick_rel_change=dict(median=float(np.median(rel)), min=float(rel.min()), max=float(rel.max()),
                                         max_abs=float(np.abs(rel).max())),
               poses=len(y0), worst_pose=str(model["pose_ids"][int(np.argmax(np.abs(rel)))]),
               per_pose={str(p): dict(S_exact_sidecar=float(a), S_exact_gp_fd=float(b)) for p, a, b in zip(model["pose_ids"], y0, y1)})
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_pose"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
