"""Stage 2b: checks of the estimating-equation evaluator (certify/cpp/t1_ad.h, t1_server.h) at a stop state.

  1. F at z = 0 against J' Lambda e from the export (the corrected solver's own Jacobian and error at x_c), per factor.
  2. F' against central differences of the assembled F (h = 1e-6) on 12 random coordinates.
  3. Interval enclosures at the point z1 contain the double F and F'; on a box z1 +- r (r = 1e-3 in every coordinate),
     the double F' at 200 random points of the box lie inside the box enclosure (per factor, all entries).
Writes results/stage2b/<dataset>/evaluator_checks.json.

Usage: python3 scripts/stage2b_checks.py <dataset>
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from certify.t1 import T1Client, Problem  # noqa: E402


def main(ds):
    ed = REPO / "exports/stage1" / ds / "corrected_stage2a" / "eval" / "certified_stop_beliefs"
    pb = Problem(ed)
    ex = pb.ex
    cl = T1Client(ed, f"/dev/shm/gbp_os717_t1chk_{ds}", true_jacobians=True)
    res = dict(dataset=ds, load_check=cl.load_check)
    out0 = cl.eval(np.zeros(ex.n_dof))
    for t in ("stereo", "gp"):
        g = ex.factors[t]
        k = pb.fac[t]["keep"]
        Fx = np.einsum("nma,nmb,nb->na", g["J"], np.linalg.inv(g["Sigma"]), g["e"])[k]
        d = out0[t]["F"][k]
        per = np.abs(d - Fx).max(axis=1) / np.maximum(np.abs(Fx).max(axis=1), 1e-300)
        res["F0_vs_solver_" + t] = dict(median=float(np.median(per)), p99=float(np.percentile(per, 99)), max=float(per.max()))
    F0, dF0, _ = pb.full(np.zeros(pb.n), cl)
    rng = np.random.default_rng(1)
    worst = 0.0
    for c in rng.choice(pb.n, 12, replace=False):
        e = np.zeros(pb.n)
        e[c] = 1e-6
        Fp, _, _ = pb.full(e, cl)
        Fm, _, _ = pb.full(-e, cl)
        col = dF0[:, c].toarray().ravel()
        worst = max(worst, float(np.abs((Fp - Fm) / 2e-6 - col).max() / max(np.abs(col).max(), 1e-300)))
    res["dF_vs_central_differences_max_rel"] = worst
    z1 = rng.normal(scale=1e-4, size=ex.n_dof)
    r = np.full(ex.n_dof, 1e-3)
    pt = cl.eval(z1)
    enc_pt = cl.enclose(z1, np.zeros(ex.n_dof))
    enc = cl.enclose(z1, r)
    contain = {}
    for t in ("stereo", "gp"):
        k = pb.fac[t]["keep"]
        ok_pt = bool(np.all((enc_pt[t]["Flo"][k] <= pt[t]["F"][k]) & (pt[t]["F"][k] <= enc_pt[t]["Fhi"][k]))
                     and np.all((enc_pt[t]["dFlo"][k] <= pt[t]["dF"][k]) & (pt[t]["dF"][k] <= enc_pt[t]["dFhi"][k])))
        width = np.max(enc_pt[t]["Fhi"][k] - enc_pt[t]["Flo"][k])
        contain[t] = dict(point_contains_double=ok_pt, point_F_width_max=float(width), box_flags=int(np.sum(enc[t]["flags"][k] > 0)))
    outside = {"stereo": 0, "gp": 0}
    for _ in range(200):
        zr = z1 + rng.uniform(-1, 1, size=ex.n_dof) * r * (1 - 1e-9)
        o = cl.eval(zr)
        for t in ("stereo", "gp"):
            k = pb.fac[t]["keep"] & (enc[t]["flags"] == 0)
            outside[t] += int(np.sum((o[t]["dF"][k] < enc[t]["dFlo"][k]) | (o[t]["dF"][k] > enc[t]["dFhi"][k])))
    for t in contain:
        contain[t]["box_random_points_outside"] = outside[t]
    res["enclosures"] = contain
    cl.close()
    out = REPO / "results/stage2b" / ds
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluator_checks.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "load_check"}, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
