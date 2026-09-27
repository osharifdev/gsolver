"""Phase D, WP6b: checks of the nonlinear probe's evaluation (certify/NL_PROBE.md), on one persisted export.

  1. Identity. Every variable a target, z1 = 0: each factor is re-evaluated at its linearization point, so
     F_nl = F_jac = F_lin (the C++ re-evaluation, the chart and the point setting agree with the frozen model).
  2. Order. Every variable a target, z1 = s z (the export's state scaled by s = 1, 1/2, 1/4, 1/8): the Jacobian-change
     term F_jac - F_lin and the model mismatch F_true - F_jac are first order in s, the curvature F_nl - F_true second
     order; the fitted log-log slopes are reported.
  3. Jacobians. For the first 200 factors of every class, the solver's J_lin against central differences of the
     factor's own error along the solver's charts (h = 1e-6), per variable and block of three tangent columns.
Norms are over the pose dofs, weighted by D_i^-1/2 (the D-metric of a gradient), and over all dofs unweighted.

Usage: python3 scripts/phaseD_wp6b_checks.py <export dir> <out json>
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from certify.io import Export  # noqa: E402
from certify.nonlinear_probe import ProbeClient  # noqa: E402

BIN = REPO / "build-certify/certify/cpp/certify_stereo"
GT = REPO / "examples/data/datasets/stereo_visual/gt/06.tum"


def main(ed, out):
    ex = Export(ed)
    pdir = "/dev/shm/gbp_os717_probe_check"
    cmd = [str(BIN), "--g2o", str(REPO / ex.meta["g2o"]), "--gt", str(GT), "--keyframes", str(ex.meta["keyframes"]), "--config",
           ex.meta["config"], "--out", pdir + "/out", "--probe-export", str(ed), "--probe-dir", pdir]
    cl = ProbeClient(cmd, pdir)
    targets = np.arange(ex.n_blocks, dtype=np.int64)
    poses = np.flatnonzero(ex.var_type == "SE3PoseVel")
    H, _ = ex.assemble(("meas",))
    Dm = []
    for i in poses:
        s_ = ex.block_slice(i)
        Dm.append(np.linalg.cholesky(H[s_, s_].toarray() + 1e-300 * np.eye(12)))

    def pose_norm(F):
        return float(np.sqrt(sum(np.sum(np.linalg.solve(Dm[k], F[ex.block_slice(i)]) ** 2) for k, i in enumerate(poses))))

    res = dict(export=str(ed), classes=cl.classes, load_check=cl.load_check)
    (Fnl, Flin, Fjac, Ftrue, st), = cl.evaluate([(targets, np.zeros(ex.n_dof))])
    scale = np.abs(Flin).max()
    res["identity"] = dict(max_abs_nl_minus_lin=float(np.abs(Fnl - Flin).max()), max_abs_jac_minus_lin=float(np.abs(Fjac - Flin).max()),
                           max_abs_true_minus_lin=float(np.abs(Ftrue - Flin).max()),
                           max_abs_F_lin=float(scale), inlier_flips=int(st[1]))
    rows = []
    for s in (1.0, 0.5, 0.25, 0.125):
        (Fnl, Flin, Fjac, Ftrue, st), = cl.evaluate([(targets, s * ex.z)])
        J, M, C = (Fjac - Flin).sum(axis=0), (Ftrue - Fjac).sum(axis=0), (Fnl - Ftrue).sum(axis=0)
        rows.append(dict(s=s, jacobian_D=pose_norm(J), mismatch_D=pose_norm(M), curvature_D=pose_norm(C), total_D=pose_norm(J + M + C),
                         jacobian_2=float(np.linalg.norm(J)), mismatch_2=float(np.linalg.norm(M)), curvature_2=float(np.linalg.norm(C)),
                         inlier_flips=int(st[1]),
                         per_class_D={c: pose_norm(Fnl[k] - Flin[k]) for k, c in enumerate(cl.classes)},
                         per_class_parts_D={c: dict(jacobian=pose_norm(Fjac[k] - Flin[k]), mismatch=pose_norm(Ftrue[k] - Fjac[k]),
                                                    curvature=pose_norm(Fnl[k] - Ftrue[k])) for k, c in enumerate(cl.classes)}))
    fd = cl.fd_check(200, 1e-6)
    cl.close()
    summ = {}
    for r in fd:
        k = "%s slot %s cols %s" % (r["class"], r["slot"], r["cols"])
        a = summ.setdefault(k, dict(n=0, ratios=[], rel=[]))
        a["n"] += 1
        if r["ratio"] != "nan":
            a["ratios"].append(float(r["ratio"]))
        m = float(r["max_abs_fd"])
        a["rel"].append(float(r["max_abs_diff"]) / m if m > 0 else float(r["max_abs_diff"]))
    res["fd_check"] = {k: dict(factors=v["n"], ratio_median=float(np.median(v["ratios"])) if v["ratios"] else None,
                               ratio_min=float(np.min(v["ratios"])) if v["ratios"] else None,
                               ratio_max=float(np.max(v["ratios"])) if v["ratios"] else None,
                               rel_diff_median=float(np.median(v["rel"])), rel_diff_max=float(np.max(v["rel"])))
                       for k, v in summ.items()}
    ls = np.log([r["s"] for r in rows])
    res["order"] = rows
    res["slope_jacobian_D"] = float(np.polyfit(ls, np.log([r["jacobian_D"] for r in rows]), 1)[0])
    res["slope_mismatch_D"] = float(np.polyfit(ls, np.log([r["mismatch_D"] for r in rows]), 1)[0])
    res["slope_curvature_D"] = float(np.polyfit(ls, np.log([r["curvature_D"] for r in rows]), 1)[0])
    for cls_i, c in enumerate(cl.classes):
        res["slope_class_" + c] = float(np.polyfit(ls, np.log([r["per_class_D"][c] for r in rows]), 1)[0])
        for part in ("jacobian", "mismatch", "curvature"):
            v = [r["per_class_parts_D"][c][part] for r in rows]
            if min(v) > 0:
                res["slope_class_%s_%s" % (c, part)] = float(np.polyfit(ls, np.log(v), 1)[0])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("load_check",)}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
