"""Stage 2a (BRIEF_stage2.md §2): the final-answer check at the certified-stop states (certify/STAGE2A.md).

For one certified-stop state (exports/stage1/<dataset>/<variant>_stage2a: the Stage 1 run repeated to its stop round,
which persisted the linearization and every pose's final and base patch at the stop):
  0. The patch factorizations are rebuilt from the persisted patches (the same rows: S_PP Cholesky, [S_PP^-1]_{:,i}, leaks),
     and the certificate is recomputed from them (base rows -> c0, final rows -> c*, E = min(c*, s + a.c0), L); it must
     reproduce the logged E and L.
  1. One global candidate z1: every pose takes the step of its own patch, z1_p = z_p + [-S_PP^-1 g_P]_p; every landmark
     takes its back-substituted step in the full D2 model from the patch of the observing pose whose patch is largest
     (ties: the earlier pose), dl = -H_ll^-1 (g_l + H_lP dP).
  2. Landmark cap: each landmark step dl is clipped to the Euclidean ball of radius r_L in {0.1 m, 1 m, inf}.
  3. F(z1) = sum_f J_f' Lambda_f e_f at x_lin (+) z1, every factor re-evaluated in C++ by the solver's own code with the
     variant's Jacobians (certify_stereo --probe-export, plus --true-jacobians for the corrected variant), plus the D2
     pseudo-factors (landmark priors, frozen single-edge messages).
  4. s_i^NL = || [S_PiPi^-1 (F_Pi - H_Pi,L H_LL^-1 F_L)]_i ||_{D_i} with the same patch factorization (L: the landmarks
     observed by P_i).
  5. r_i = s_i^NL / (eta S_lo,i).
Attribution (norm shares of s^NL, not additive): stereo factors' nonlinear part F_nl - F_lin, GP factors' nonlinear part,
and the linear residue F_lin(z1) (nonzero because the global candidate mixes patches and because of the clipping).

Usage: python3 scripts/stage2a_probe.py <dataset> <variant>
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from certify.io import Export  # noqa: E402
from certify.nonlinear_probe import FullModel, ProbeClient  # noqa: E402
from certify.sidecar import DOF, Engine, Ops  # noqa: E402
from certify.sidecar_model import PoseModelBuilder, _landmark_init  # noqa: E402

ETA, P_L = 0.1, 1e-6
CAPS = (0.1, 1.0, np.inf)
BIN = REPO / "build-certify/certify/cpp/certify_stereo"
GT = {"kitti06": "06", "kitti03": "03", "kitti04": "04", "kitti10": "10"}


def main(ds, var):
    t0 = time.perf_counter()
    run = REPO / "exports/stage1" / ds / (var + "_stage2a") / "eval"
    ex = Export(run / "certified_stop_model")
    stop = json.load(open(run / "certified_stop_patches.json"))
    g2o = str(REPO / ex.meta["g2o"])
    B = PoseModelBuilder(p_L=P_L, g2o=g2o)
    model = B.build(None, ex=ex)
    S, x, b = model["S"], model["x"], model["b"]
    g = S @ x - b
    ids = [str(p) for p in model["pose_ids"]]
    n = len(ids)
    index = {p: k for k, p in enumerate(ids)}
    C = [np.linalg.cholesky(0.5 * (S[DOF * i:DOF * i + DOF, DOF * i:DOF * i + DOF] + S[DOF * i:DOF * i + DOF, DOF * i:DOF * i + DOF].T))
         for i in range(n)]
    Cinv = [sla.solve_triangular(c, np.eye(DOF), lower=True) for c in C]
    blocks = S.reshape(n, DOF, n, DOF)
    adj = np.abs(blocks).max(axis=(1, 3)) > 0
    np.fill_diagonal(adj, False)
    eng = Engine(np.inf, ETA)
    eng._fcache, eng._fcacheF = {}, {}
    ops = Ops()
    P = {i: np.array(sorted(index[q] for q in stop["patches"][ids[i]]["P"])) for i in range(n)}
    P0 = {i: np.array(sorted(index[q] for q in stop["patches"][ids[i]]["base_P"])) for i in range(n)}
    rows = {i: eng._row(S, g, C, Cinv, adj, i, P[i], ops) for i in range(n)}
    base = {i: eng._row(S, g, C, Cinv, adj, i, P0[i], ops) for i in range(n)}

    # 0. reproduce the logged certificate from the rebuilt rows
    c0, _, _, _ = eng._recursion(base, np.array([base[i].s for i in range(n)]), n)
    s = np.array([rows[i].s for i in range(n)])
    cs, A, _, _ = eng._recursion(rows, s, n)
    E = np.minimum(cs, np.array([rows[i].s + (float(rows[i].a @ c0[rows[i].ext]) if len(rows[i].ext) else 0.0) for i in range(n)]))
    L = np.maximum(0.0, s - A @ E)
    E_log = np.array([stop["patches"][p]["E"] for p in ids])
    L_log = np.array([stop["patches"][p]["L"] for p in ids])
    S_lo = np.array([stop["patches"][p]["S_lo"] for p in ids])
    checks = dict(round=stop["round"], poses=n, rel_E_rebuilt_vs_logged=float(np.max(np.abs(E - E_log) / E_log)),
                  abs_L_rebuilt_vs_logged=float(np.max(np.abs(L - L_log))), all_pass_logged=bool(np.all(E_log <= ETA * S_lo)))

    # the full D2 model
    fm = FullModel(ex, p_L=P_L, g2o=g2o, init=_landmark_init(g2o))
    S_full, g_full = fm.reduced()
    checks.update(rel_S_full_vs_sidecar=float(np.abs(S_full - S).max() / np.abs(S).max()),
                  rel_g_full_vs_sidecar=float(np.abs(g_full - g).max() / np.abs(g).max()))
    assert [str(p) for p in ex.var_id[fm.poses]] == ids

    # 1. the global candidate
    dP = {}
    for i in range(n):
        cf = eng._fcache[rows[i].P.tobytes()]
        dP[i] = -sla.cho_solve(cf, g[rows[i].dofP], check_finite=False)
    step = np.zeros(ex.n_dof)
    for i in range(n):
        k = int(np.flatnonzero(rows[i].P == i)[0])
        blk = fm.poses[i]
        step[ex.offsets[blk]:ex.offsets[blk] + DOF] = dP[i][DOF * k:DOF * k + DOF]
    observers = {}
    for i in range(n):
        for l in fm.adj[i]:
            observers.setdefault(int(l), []).append(i)
    owner = {l: max(obs, key=lambda q: (len(rows[q].P), -q)) for l, obs in observers.items()}
    by_owner = {}
    for l, q in owner.items():
        by_owner.setdefault(q, []).append(l)
    lstep = {}
    for q, ls in by_owner.items():
        pd = (ex.offsets[fm.poses[rows[q].P]][:, None] + np.arange(DOF)[None, :]).ravel()
        for l in ls:
            sl = np.arange(ex.offsets[l], ex.offsets[l] + ex.dims[l])
            rhs = fm.g[sl] + fm.H[sl][:, pd] @ dP[q]
            lstep[l] = -fm.einv[l] @ rhs
    lnorm = {l: float(np.linalg.norm(v[:3])) for l, v in lstep.items()}
    targets = np.concatenate([fm.poses, fm.elim]).astype(np.int64)
    tdofs = np.concatenate([np.arange(ex.offsets[t], ex.offsets[t] + ex.dims[t]) for t in targets])

    def candidate(cap):
        st = step.copy()
        for l, v in lstep.items():
            nv = lnorm[l]
            w = v if (not np.isfinite(cap) or nv <= cap) else v * (cap / nv)
            st[ex.offsets[l]:ex.offsets[l] + ex.dims[l]] = w
        return ex.z + st

    # 3. F at the stepped points (and the identity at z = 0) in C++
    pdir = f"/dev/shm/gbp_os717_s2a_{ds}_{var}"
    cmd = [str(BIN), "--g2o", g2o, "--gt", str(REPO / f"examples/data/datasets/stereo_visual/gt/{GT[ds]}.tum"), "--keyframes",
           str(ex.meta["keyframes"]), "--config", ex.meta["config"], "--out", pdir + "/out", "--probe-export", str(run / "certified_stop_model"),
           "--probe-dir", pdir] + (["--true-jacobians"] if var == "corrected" else [])
    cl = ProbeClient(cmd, pdir)
    checks["load_check"] = cl.load_check
    zs = [candidate(c) for c in CAPS] + [np.zeros(ex.n_dof)]
    outs = cl.evaluate([(targets, z[tdofs]) for z in zs])
    cl.close()
    Fnl0, Flin0, _, _, st0 = outs[-1]
    checks["identity_max_abs_Fnl_minus_Flin_at_x_lin"] = float(np.abs(Fnl0 - Flin0).max())
    classes = cl.classes

    # pseudo-factors (linear) on the targets
    def pseudo(z):
        F = np.zeros(ex.n_dof)
        for blk, (Lm, a) in fm.pseudo_of.items():
            sl = slice(ex.offsets[blk], ex.offsets[blk] + ex.dims[blk])
            F[sl] = Lm @ (z[sl] - a)
        return F

    # 4. per pose: reduce over the patch's landmarks, solve with the same patch factorization
    def s_of(i, Fvec):
        pd = (ex.offsets[fm.poses[rows[i].P]][:, None] + np.arange(DOF)[None, :]).ravel()
        ls = np.unique(np.concatenate([fm.adj[k] for k in rows[i].P]))
        y = np.zeros(ex.n_dof)
        for l in ls:
            sl = slice(ex.offsets[l], ex.offsets[l] + ex.dims[l])
            y[sl] = fm.einv[int(l)] @ Fvec[sl]
        ed = np.concatenate([np.arange(ex.offsets[l], ex.offsets[l] + ex.dims[l]) for l in ls])
        red = Fvec[pd] - fm.H[pd][:, ed] @ y[ed]
        return float(np.linalg.norm(C[i].T @ (rows[i].Z.T @ red)))

    per, summ = [], {}
    for ci, cap in enumerate(CAPS):
        Fnl, Flin, _, _, stat = outs[ci]
        full = lambda arr: _scatter(arr, tdofs, ex.n_dof)
        Fps = pseudo(zs[ci])
        F_tot = full(Fnl.sum(axis=0)) + Fps
        F_lin = full(Flin.sum(axis=0)) + Fps
        parts = {c: full(Fnl[k] - Flin[k]) for k, c in enumerate(classes)}
        capname = "inf" if not np.isfinite(cap) else "%g" % cap
        for i in range(n):
            sN = s_of(i, F_tot)
            ls = np.unique(np.concatenate([fm.adj[k] for k in rows[i].P]))
            rec = dict(dataset=ds, variant=var, round=stop["round"], cap=capname, pose=ids[i], t=float(model["times"][i]),
                       E=float(E_log[i]), S_lo=float(S_lo[i]), s_nl=sN, r=sN / (ETA * S_lo[i]),
                       s_linear_residue=s_of(i, F_lin), patch=len(rows[i].P),
                       max_landmark_step_before_clip=max(lnorm[int(l)] for l in ls) if len(ls) else 0.0,
                       landmarks_clipped=int(sum(lnorm[int(l)] > cap for l in ls)), landmarks=len(ls),
                       inlier_flips=int(stat[1]))
            for c in classes:
                rec["s_nl_" + c] = s_of(i, parts[c])
            per.append(rec)
        R = [x for x in per if x["cap"] == capname]
        r = np.array([x["r"] for x in R])
        worst = sorted(R, key=lambda x: -x["r"])[:max(1, int(np.ceil(0.1 * len(R))))]
        summ[capname] = dict(poses=len(R), r_median=float(np.median(r)), r_p90=float(np.percentile(r, 90)), r_max=float(r.max()),
                             frac_below_0_1=float(np.mean(r < 0.1)), frac_below_1=float(np.mean(r < 1.0)),
                             worst_pose=max(R, key=lambda x: x["r"])["pose"], inlier_flips=int(stat[1]),
                             worst_decile=dict(n=len(worst), r_min=float(worst[-1]["r"]),
                                               **{"share_" + c: float(np.median([x["s_nl_" + c] / x["s_nl"] for x in worst])) for c in classes},
                                               share_linear_residue=float(np.median([x["s_linear_residue"] / x["s_nl"] for x in worst])),
                                               largest_landmark_step_before_clip=float(max(x["max_landmark_step_before_clip"] for x in worst))))
    summ["landmark_steps"] = dict(landmarks=len(lnorm), median=float(np.median(list(lnorm.values()))),
                                  p99=float(np.percentile(list(lnorm.values()), 99)), max=float(max(lnorm.values())),
                                  over_0_1m=int(sum(v > 0.1 for v in lnorm.values())), over_1m=int(sum(v > 1.0 for v in lnorm.values())))
    summ["checks"] = checks
    summ["seconds"] = time.perf_counter() - t0
    out = REPO / "results/stage2a" / ds / var
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "probe.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per[0].keys()))
        w.writeheader()
        w.writerows(per)
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(ds, var, json.dumps({k: v for k, v in summ.items() if k not in ("checks",)}, default=float))
    print("   checks", json.dumps({k: v for k, v in checks.items() if k != "load_check"}, default=float))


def _scatter(arr, idx, n):
    out = np.zeros(n)
    out[idx] = arr
    return out


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
