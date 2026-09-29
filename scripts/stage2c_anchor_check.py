"""Stage 2c, Step 1 (BRIEF_stage2c.md §1): the clean model's counts and the anchor check.

Counts: the driver's cleaning.json of every clean-model run (landmarks and stereo factors removed and remaining).

Anchor check (§1.4): at the corrected variant's round-100 state (the evaluation's linearization, frozen charts, kept by the
evaluator with --keep-rounds 100) the linear system is solved twice, with the anchor's precision 1e4 (as run) and 1e6:
  full model  H e = g, H = sum_f J' Lambda J + the landmark priors p_L, g the gradient at the state (FullModel, the D2
              model before landmark elimination, with the anchor);
  pose model  S e_P = g_P (the certificate's reduced model, PoseModelBuilder).
e is the distance from the state to the solution of the linear model, so the solution is z - e. Reported: the largest
relative change of any pose's solution, ||x_i(1e6) - x_i(1e4)|| / ||x_i(1e4)|| (Euclidean in the tangent coordinates, and in
the D_i-metric of the certificate), the largest absolute change in the D_i-metric (against the pass threshold eta S_lo,i),
the same over landmarks, and lambda_min (and lambda_max) of H and of S for both.
Attribution (diagnostic): the same pair of solves without the landmark priors (p_L = 0). The priors p_L are the model's only
absolute terms besides the anchor; if the change vanishes without them, it is their pull along the gauge directions, balanced by
the anchor, that the anchor's precision scales. Also the gap between the certificate's model (with p_L) and the solver's own
graph (no p_L) at the run's anchor precision 1e4: ||x_i(p_L) - x_i(0)||_{D_i} / (eta S_lo,i).

Writes results/stage2c/cleaning.json.
Usage: python3 scripts/stage2c_anchor_check.py
"""

import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from certify.io import Export  # noqa: E402
from certify.nonlinear_probe import FullModel  # noqa: E402
from certify.sidecar_model import PoseModelBuilder, _landmark_init  # noqa: E402

DS = ["kitti06", "kitti03", "kitti04", "kitti10"]
DOF = 12


def with_anchor(ex, precision):
    """A copy of the export whose anchor prior (PRIOR_ANCHOR_*) has precision `precision` * I."""
    e2 = copy.copy(ex)
    e2.factors = {t: dict(g) for t, g in ex.factors.items()}
    g = e2.factors["prior"]
    sel = np.char.startswith(g["ids"].astype(str), "PRIOR_ANCHOR")
    assert sel.sum() == 1
    S = g["Sigma"].copy()
    S[sel] = np.eye(6) / precision
    g["Sigma"] = S
    return e2


def solve(ex, g2o, p_L=1e-6):
    fm = FullModel(ex, p_L=p_L, g2o=g2o, init=_landmark_init(g2o))
    idx = np.concatenate([fm.pdofs, fm.edofs])
    H = fm.H[idx][:, idx].tocsc()
    g = fm.g[idx]
    lu = spla.splu(H)
    e = lu.solve(g)
    r = H @ e - g
    x = ex.z[idx] - e
    lmin = float(spla.eigsh(H, k=1, sigma=0.0, which="LM", return_eigenvectors=False)[0])
    lmax = float(spla.eigsh(H, k=1, which="LA", return_eigenvectors=False)[0])
    B = PoseModelBuilder(p_L=p_L, g2o=g2o)
    m = B.build(None, ex=ex)
    S, gP = m["S"], m["S"] @ m["x"] - m["b"]
    eP = sla.cho_solve(sla.cho_factor(S, lower=True), gP)
    ev = np.linalg.eigvalsh(S)
    return dict(fm=fm, x=x, npd=len(fm.pdofs), resid=float(np.abs(r).max() / max(np.abs(g).max(), 1e-300)), lmin_H=lmin, lmax_H=lmax,
                xP=m["x"] - eP, S=S, lmin_S=float(ev[0]), lmax_S=float(ev[-1]), n_H=int(H.shape[0]))


def compare(res, ex, slo):
    a, b = res[1e4], res[1e6]
    S = a["S"]
    n = S.shape[0] // DOF
    C = [np.linalg.cholesky(0.5 * (S[DOF * i:DOF * i + DOF, DOF * i:DOF * i + DOF] + S[DOF * i:DOF * i + DOF, DOF * i:DOF * i + DOF].T))
         for i in range(n)]
    npd = a["npd"]
    rel_full, rel_D, abs_D, rel_red = [], [], [], []
    for i in range(n):
        s_ = slice(DOF * i, DOF * i + DOF)
        dx = b["x"][s_] - a["x"][s_]
        rel_full.append(float(np.linalg.norm(dx) / np.linalg.norm(a["x"][s_])))
        abs_D.append(float(np.linalg.norm(C[i].T @ dx)))
        rel_D.append(abs_D[-1] / float(np.linalg.norm(C[i].T @ a["x"][s_])))
        rel_red.append(float(np.linalg.norm(b["xP"][s_] - a["xP"][s_]) / np.linalg.norm(a["xP"][s_])))
    lm = np.abs(b["x"][npd:] - a["x"][npd:]).reshape(-1, 3)
    lmn = np.abs(a["x"][npd:]).reshape(-1, 3)
    rel_lm = np.linalg.norm(lm, axis=1) / np.maximum(np.linalg.norm(lmn, axis=1), 1e-300)
    worst = int(np.argmax(rel_full))
    return dict(state="corrected variant, evaluation after round 100 (frozen charts)", dofs_full=a["n_H"], dofs_pose=npd,
                pose_solution_rel_change_max=max(rel_full), pose_solution_rel_change_worst_pose=str(ex.var_id[a["fm"].poses[worst]]),
                pose_solution_rel_change_median=float(np.median(rel_full)), pose_solution_rel_change_D_max=max(rel_D),
                pose_solution_change_D_max=max(abs_D), pose_solution_change_D_over_eta_S_lo_max=float(np.max(np.array(abs_D) / (0.1 * slo))),
                pose_solution_rel_change_reduced_model_max=max(rel_red), landmark_solution_rel_change_max=float(rel_lm.max()),
                landmark_solution_change_m_max=float(np.linalg.norm(lm, axis=1).max()),
                solve_rel_residual={"1e4": a["resid"], "1e6": b["resid"]},
                lambda_min_H={"1e4": a["lmin_H"], "1e6": b["lmin_H"]}, lambda_max_H={"1e4": a["lmax_H"], "1e6": b["lmax_H"]},
                lambda_min_S={"1e4": a["lmin_S"], "1e6": b["lmin_S"]}, lambda_max_S={"1e4": a["lmax_S"], "1e6": b["lmax_S"]})


def main():
    out = dict(protocol=__doc__.strip().splitlines()[0], datasets={})
    for ds in DS:
        d = {}
        for var in ("corrected", "shipped"):
            p = REPO / "exports/stage2c" / ds / var / "cleaning.json"
            if p.exists():
                d.setdefault("cleaning", json.load(open(p)))
                d.setdefault("cleaning_identical_in", []).append(var)
        run = REPO / "exports/stage2c" / ds / "corrected" / "eval" / "model_r00100"
        ex = Export(run)
        g2o = str(REPO / ex.meta["g2o"])
        d["single_edge_variables_in_export"] = len(ex.single_edge)
        d["anchor_ids"] = [str(s) for s in ex.factors["prior"]["ids"]]
        slo = np.array([float(r["S_lo"]) for r in csv.DictReader(open(run.parent / "eval_rounds.csv")) if int(r["round"]) == 100])
        sols = {}
        for tag, pL in (("anchor_check", 1e-6), ("anchor_check_without_p_L", None)):
            res = {P: solve(with_anchor(ex, P), g2o, pL) for P in (1e4, 1e6)}
            sols[pL] = res
            d[tag] = compare(res, ex, slo)
            print(ds, tag, json.dumps(d[tag], indent=1), flush=True)
        gap = compare({1e4: sols[None][1e4], 1e6: sols[1e-6][1e4]}, ex, slo)
        d["p_L_gap_at_anchor_1e4"] = {k: gap[k] for k in ("pose_solution_change_D_max", "pose_solution_change_D_over_eta_S_lo_max",
                                                          "pose_solution_rel_change_max", "landmark_solution_change_m_max")}
        C = sols[1e-6][1e4]
        r = []
        S = C["S"]
        for i in range(S.shape[0] // DOF):
            Ci = np.linalg.cholesky(S[DOF * i:DOF * i + DOF, DOF * i:DOF * i + DOF])
            r.append(float(np.linalg.norm(Ci.T @ (C["x"][DOF * i:DOF * i + DOF] - sols[None][1e4]["x"][DOF * i:DOF * i + DOF]))) / (0.1 * slo[i]))
        d["p_L_gap_at_anchor_1e4"].update(over_eta_S_lo_median=float(np.median(r)), poses_over_eta_S_lo=int(np.sum(np.array(r) > 1)))
        print(ds, "p_L gap", d["p_L_gap_at_anchor_1e4"], flush=True)
        out["datasets"][ds] = d
    for ds in ("prefix",):
        for var in ("corrected", "shipped"):
            p = REPO / "exports/stage2c" / ds / var / "cleaning.json"
            if p.exists():
                out["datasets"].setdefault(ds, {}).setdefault("cleaning", json.load(open(p)))
    ch = json.load(open(REPO / "results/stage2c/charuco_markers.json")) if (REPO / "results/stage2c/charuco_markers.json").exists() else None
    if ch:
        out["charuco"] = ch
    (REPO / "results/stage2c").mkdir(parents=True, exist_ok=True)
    (REPO / "results/stage2c/cleaning.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
