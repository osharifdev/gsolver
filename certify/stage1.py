"""Stage 1: periodic certified termination with the statistical rule only (BRIEF_stage1_2.md, certify/STAGE1.md).

StatEngine.evaluate(): Theorem A on the reduced pose model for every pose, with
  - base patches (leak row sum <= 0.3, kept across evaluations and regrown from the previous patch),
  - the yardstick S_lo,i from the true-Jacobian Fisher information on the base patch (Theorem T0),
  - decision-driven growth whose only target is the statistical decision, threshold eta S_lo,i: it exits on CONTINUE only
    when the certified lower bound s_i - sum_j a_ij c0_j exceeds the threshold, on STOP only when the certified upper bound
    s_i + sum_j a_ij c0_j is at most the threshold, and otherwise grows (cap 250 poses, 40 steps);
  - E_i = min(c*_i, s_i + sum_j a_ij c0_j) (both are certified upper bounds: c* from the final recursion, the other from
    the pose's final row and the base certificates c0), L_i = max(0, s_i - sum_j a_ij E_j);
  - pass_i: E_i <= eta S_lo,i.
Oracle every evaluation: e = S^-1 g (dense Cholesky), the exact yardstick from [I_F^-1]_ii, and the validity invariants
L_i <= ||e_i||_{D_i} <= E_i (1e-9 relative); on sampled evaluations the patch lower bound against the exact marginal.

accuracy(): at a belief-chart export, the remaining Gauss-Newton move per pose, ||xbar_i||_{D_i} with xbar = H^-1 b
(WP1's definition: D_i = H_ii of the D2 model before the landmark elimination), with the solver's own Jacobian and with
the true Jacobian, and the aligned keyframe ATE against the ground truth.
"""

import numpy as np
import scipy.linalg as sla

from certify.sidecar import CAP, DOF, MAX_STEPS, Q12, R_BASE, Engine, Ops


class StatEngine(Engine):
    def __init__(self, eta):
        super().__init__(delta=np.inf, eta=eta, continue_rule="L", yard_all=True)

    def _grow_stat(self, S, g, C, Cinv, adj, i, base_row, c0, thr, ops):
        row, steps, why = base_row, 0, "cap"
        while True:
            rs, leak = row.a.sum(), float(row.a @ c0[row.ext]) if len(row.ext) else 0.0
            if row.s - leak > thr:
                why = "continue"
                break
            if row.s + leak <= thr:
                why = "stop"
                break
            if len(row.P) >= min(CAP, adj.shape[0]) or not len(row.ext):
                why = "cap"
                break
            if steps >= MAX_STEPS:
                why = "steps"
                break
            score = row.a * c0[row.ext] if rs <= R_BASE else row.a
            row = self._row(S, g, C, Cinv, adj, i, np.union1d(row.P, row.ext[score >= 0.5 * score.max()]), ops)
            steps += 1
        return row, steps, why

    def evaluate(self, rnd, model, modelF, sampled=False):
        ops = Ops()
        self._fcache, self._fcacheF = {}, {}
        S, b, x = model["S"], model["b"], model["x"]
        SF = modelF["S"]
        ids = [str(p) for p in model["pose_ids"]]
        n = len(ids)
        g = S @ x - b
        try:
            cfS = sla.cho_factor(S, lower=True, check_finite=True)
            sla.cho_factor(SF, lower=True, check_finite=True)
        except (np.linalg.LinAlgError, ValueError):
            self.prev = None
            rows = [dict(round=rnd, pose=p, t=float(model["times"][k]), decision="A0_FAILS", passed=0) for k, p in enumerate(ids)]
            return rows, dict(round=rnd, poses=n, a0_fails=1, all_pass=0, frac_pass=0.0, **ops.c)
        blocks = S.reshape(n, DOF, n, DOF)
        adj = np.abs(blocks).max(axis=(1, 3)) > 0
        np.fill_diagonal(adj, False)
        C = [np.linalg.cholesky(0.5 * (S[DOF * i:DOF * i + DOF, DOF * i:DOF * i + DOF] + S[DOF * i:DOF * i + DOF, DOF * i:DOF * i + DOF].T))
             for i in range(n)]
        Cinv = [sla.solve_triangular(Ci, np.eye(DOF), lower=True) for Ci in C]
        kappa = np.array([float(np.linalg.norm(ci, 2)) for ci in Cinv])
        index = {p: k for k, p in enumerate(ids)}

        # base rows, grown from the previous evaluation's patch
        rows, base_size = {}, {}
        for p in ids:
            i = index[p]
            cache = self.base.get(p)
            P0 = np.array([index[q] for q in cache["P"] if q in index]) if cache is not None else np.array([i])
            if i not in P0:
                P0 = np.union1d(P0, [i])
            row, _ = self._grow_base(S, g, C, Cinv, adj, i, P0, ops)
            self.base[p] = dict(P=[ids[k] for k in row.P], row=row)
            rows[i] = row
            base_size[i] = len(row.P)
        s0 = np.array([rows[i].s for i in range(n)])
        c0, _, up0, _ = self._recursion(rows, s0, n)
        c0 = c0 if c0 is not None else np.full(n, np.inf)

        # yardsticks on the base patches
        S_lo = np.zeros(n)
        yardL = {}
        for i in range(n):
            S_lo[i], yardL[i] = self._yardstick(SF, C, i, rows[i].P, ops)

        # decision-driven growth, target: the statistical decision only
        why, steps = {}, {}
        for i in range(n):
            if np.isfinite(c0).all():
                rows[i], steps[i], why[i] = self._grow_stat(S, g, C, Cinv, adj, i, rows[i], c0, self.eta * S_lo[i], ops)
            else:
                steps[i], why[i] = 0, "no_c0"

        s = np.array([rows[i].s for i in range(n)])
        E, A, up, sweeps = self._recursion(rows, s, n)
        if E is None:
            E = np.full(n, np.inf)
        else:
            ops.c["sweeps"] += int(sweeps or 0)
            ops.c["sweep_flops"] += 2.0 * A.nnz * int(sweeps or 0)
        if np.isfinite(c0).all():
            growth_bound = np.array([rows[i].s + (float(rows[i].a @ c0[rows[i].ext]) if len(rows[i].ext) else 0.0) for i in range(n)])
            E = np.minimum(E, growth_bound)
        if A is None:
            A = self._recursion(rows, s, n)[1]
        Lo = np.maximum(0.0, s - A @ np.where(np.isfinite(E), E, 0.0)) if np.isfinite(E).all() else np.zeros(n)
        passed = E <= self.eta * S_lo

        # oracle and invariants
        e = sla.cho_solve(cfS, g, check_finite=False)
        err = np.array([np.linalg.norm(C[i].T @ e[DOF * i:DOF * i + DOF]) for i in range(n)])
        SFinv = np.linalg.inv(SF)
        S_ex = np.zeros(n)
        loewner = np.full(n, np.nan)
        for i in range(n):
            Sig = SFinv[DOF * i:DOF * i + DOF, DOF * i:DOF * i + DOF]
            Sig = 0.5 * (Sig + Sig.T)
            M = C[i].T @ Sig @ C[i]
            S_ex[i] = np.sqrt(max(0.0, float(np.linalg.eigvalsh(M)[0]))) * Q12
            if sampled:
                Dm = C[i].T @ (Sig - yardL[i]) @ C[i]
                loewner[i] = float(np.linalg.eigvalsh(0.5 * (Dm + Dm.T))[0]) / float(np.linalg.eigvalsh(M)[-1])
        tol = 1e-9
        fin = np.isfinite(E)
        viol_up = fin & (err > E * (1 + tol) + 1e-12)
        viol_lo = Lo > err * (1 + tol) + 1e-12
        rel_up = np.where(fin & (E > 0), (err - E) / np.where(E > 0, E, 1.0), np.nan)
        rel_lo = np.where(err > 0, (Lo - err) / np.where(err > 0, err, 1.0), np.nan)
        self.prev = dict(S=S, SF=SF, ids=ids)
        self.last = dict(rows=rows, g=g, C=C, E=E, L=Lo, err=err, S_lo=S_lo, S_exact=S_ex, fcache=self._fcache, S=S)

        per_pose = [dict(round=rnd, pose=p, t=float(model["times"][index[p]]), E=float(E[index[p]]), L=float(Lo[index[p]]),
                         err=float(err[index[p]]), S_lo=float(S_lo[index[p]]), S_exact=float(S_ex[index[p]]),
                         kappa=float(kappa[index[p]]), passed=int(passed[index[p]]),
                         passed_oracle_exact=int(err[index[p]] <= self.eta * S_ex[index[p]]),
                         passed_oracle_lo=int(err[index[p]] <= self.eta * S_lo[index[p]]),
                         patch=len(rows[index[p]].P), base_patch=base_size[index[p]], grown_steps=steps[index[p]],
                         grow_why=why[index[p]], valid_upper=int(not viol_up[index[p]]), valid_lower=int(not viol_lo[index[p]]),
                         rel_upper_margin=float(rel_up[index[p]]), rel_lower_margin=float(rel_lo[index[p]]),
                         loewner_margin=float(loewner[index[p]]))
                    for p in ids]
        summary = dict(round=rnd, poses=n, a0_fails=0, all_pass=int(passed.all()), frac_pass=float(passed.mean()),
                       frac_pass_oracle_exact=float(np.mean(err <= self.eta * S_ex)),
                       rho_upper=float(up) if up is not None and np.isfinite(up) else np.nan,
                       violations_upper=int(viol_up.sum()), violations_lower=int(viol_lo.sum()),
                       worst_rel_upper=float(np.nanmax(rel_up)) if np.isfinite(rel_up).any() else np.nan,
                       worst_rel_lower=float(np.nanmax(rel_lo)) if np.isfinite(rel_lo).any() else np.nan,
                       loewner_min=float(np.nanmin(loewner)) if sampled else np.nan,
                       E_over_err_median=float(np.median(E / np.maximum(err, 1e-300))),
                       grow_exits={k: int(sum(1 for v in why.values() if v == k)) for k in ("stop", "continue", "cap", "steps", "no_c0")},
                       **ops.c)
        summary["certificate_flops"] = float(sum(ops.c[k] for k in ("factor_flops", "solve_flops", "matvec_flops", "sweep_flops",
                                                                   "yardstick_flops")))
        return per_pose, summary


def remaining_gn_move(model):
    """||xbar_i||_{D_i} per pose, xbar = S^-1 b at a belief-chart model (state 0), D_i = H_ii before landmark elimination
    (WP1's definition) and, second, D_i = S_ii (the reduced model's, the certificate's metric)."""
    S, b = model["S"], model["b"]
    assert np.abs(model["x"]).max() == 0.0, "accuracy needs a belief-chart export (state 0)"
    xbar = sla.cho_solve(sla.cho_factor(S, lower=True), b)
    n = len(model["pose_ids"])
    mH = np.array([np.sqrt(max(0.0, xbar[DOF * i:DOF * i + DOF] @ model["H_pose_diag"][i] @ xbar[DOF * i:DOF * i + DOF])) for i in range(n)])
    mS = np.array([np.sqrt(max(0.0, xbar[DOF * i:DOF * i + DOF] @ S[DOF * i:DOF * i + DOF, DOF * i:DOF * i + DOF] @ xbar[DOF * i:DOF * i + DOF]))
                   for i in range(n)])
    return mH, mS


def read_gt(path):
    rows = [l.split() for l in open(path) if l.strip() and not l.startswith("#")]
    a = np.array(rows, float)
    return a[:, 0], a[:, 1:4], a[:, 4:8]


def keyframe_ate(ex, gt_path, tol=1e-6):
    """Aligned (SE(3) Umeyama) keyframe ATE of the export's pose linearization points (the beliefs, for a belief-chart
    export) against the ground truth at the keyframe timestamps."""
    from certify.tools.traj_eval import umeyama_se3
    poses = np.flatnonzero(ex.var_type == "SE3PoseVel")
    order = np.argsort(ex.timestamp[poses], kind="stable")
    t = ex.timestamp[poses][order]
    P = ex.pose_T_lin[order][:, :3, 3]
    tg, pg, _ = read_gt(gt_path)
    idx = np.clip(np.searchsorted(tg, t - tol), 0, len(tg) - 1)
    ok = np.abs(tg[idx] - t) < tol
    Ra, ta = umeyama_se3(P[ok], pg[idx[ok]])
    pa = P[ok] @ Ra.T + ta
    return dict(ate_aligned=float(np.sqrt(np.mean(np.sum((pa - pg[idx[ok]]) ** 2, axis=1)))), matched=int(ok.sum()), keyframes=len(t))


def accuracy(ex, builder, gt_path=None):
    m = builder.build(None, ex=ex)
    mT = builder.build(None, true_jacobian=True, ex=ex)

    def moves(model):
        # Stage 2c: a diverged run can leave the reduced matrix at the beliefs numerically indefinite; the remaining
        # Gauss-Newton move is then undefined and recorded as NaN (flagged in `move_undefined`) instead of stopping the run
        try:
            return remaining_gn_move(model) + (False,)
        except np.linalg.LinAlgError:
            nan = np.full(len(model["pose_ids"]), np.nan)
            return nan, nan.copy(), True

    mH, mS, bad = moves(m)
    tH, tS, badT = moves(mT)
    out = dict(pose_ids=[str(p) for p in m["pose_ids"]], move=mH, move_S=mS, move_true=tH, move_true_S=tS, move_undefined=int(bad or badT))
    if gt_path:
        out.update(keyframe_ate(ex, gt_path))
    return out
