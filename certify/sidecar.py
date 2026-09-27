"""Phase D, WP2: Theorem C on the live solver, the certificate engine of the sidecar (certify/SIDECAR.md).

Per round the engine receives the reduced pose model of the current single-chart linearization (S, b, and the
reduced Fisher information SF with the true stereo Jacobian; certify/sidecar_model.py), the pose states x (tangent
coordinates at the linearization points), the solver's displacement d_i (physical metric W = I) and which poses the
solver has just relinearized, and returns the poses that sleep during this round's message passing.

Certificate (Theorem A on the reduced pose model, D_i = S_ii = C_i C_i'):
  every pose has a row (s_i, a_i.) from its patch: the base patch (grown until the leak row sum is <= r = 0.3; kept
  across rounds and refactorized only when a block row of S inside it changed), the decision-driven patch (Phase C
  I1b: r = 0.3, theta = 0.02, cap 250 poses, 40 steps) for the poses that can stop this round, or, for a sleeping
  pose whose patch did not change, the running bound of Proof Note Theorem 5.4:
      s_i <= s_i(t0) + sum_p l_ip ||C_p^-1 (g_p - g_p(t0))||,  l_ip = ||C_i' [S_PP^-1]_ip C_p||.
  E = c* = (I - A)^-1 s (rho(A) < 1 certified by Collatz-Wielandt), L_i = max(0, s_i^lo - (A E)_i).
Yardstick (Theorem T0): S_lo,i = sqrt(lambda_min(C_i' [SF_PP^-1]_ii C_i)) q_12(0.05) on the base patch.
Growth exits (continue_rule "L", decided after WP2): CONTINUE only when the certified lower bound exceeds the threshold.
Decision (Theorem C at R = 1; RELINEARIZE-NOW cannot fire, the solver relinearizes as soon as d_i > delta):
  STOP if d_i + kappa_i E_i <= delta and E_i <= eta S_lo,i, kappa_i = ||C_i^-T|| = lambda_min(D_i)^-1/2; else CONTINUE.
  Certified only for poses the solver did not relinearize this round (d_i <= delta), and for sleeping poses.
Oracle: e = S^-1 g (dense Cholesky), the exact S_i from [SF^-1]_ii, oracle STOP with ||e_i|| in place of E_i.
"""

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from certify import certs, radii

R_BASE, THETA, CAP, MAX_STEPS = 0.3, 0.02, 250, 40
Q12 = radii.q(12)
DOF = 12


class Ops:
    """Operation counts of the certificate work (flops by the textbook counts)."""

    def __init__(self):
        self.c = dict(factorizations=0, factorizations_shared=0, factor_flops=0.0, solve_rhs=0, solve_flops=0.0, matvec_flops=0.0,
                      sweeps=0, sweep_flops=0.0, yardstick_factorizations=0, yardstick_flops=0.0)

    def factor(self, n, yard=False):
        k = "yardstick_" if yard else ""
        self.c[k + "factorizations"] += 1
        self.c[k + "flops" if yard else "factor_flops"] += n ** 3 / 3.0

    def solve(self, n, rhs, yard=False):
        if yard:
            self.c["yardstick_flops"] += 2.0 * n * n * rhs
        else:
            self.c["solve_rhs"] += rhs
            self.c["solve_flops"] += 2.0 * n * n * rhs

    def matvec(self, flops):
        self.c["matvec_flops"] += flops


class Row:
    """One pose's certificate row from a factorized patch."""

    def __init__(self, i, P, ext, a, Z, dofP, s=None):
        self.i, self.P, self.ext, self.a, self.Z, self.dofP, self.s = i, P, ext, a, Z, dofP, s


class Engine:
    def __init__(self, delta, eta, continue_rule="L", yard_all=False):
        # continue_rule: "L" (the decision after WP2: the decision-driven growth decides CONTINUE only when the certified
        # lower bound L_i = s_i - sum_j a_ij c0_j exceeds the growth threshold) or "s" (WP2 as run: CONTINUE when s_i does)
        assert continue_rule in ("L", "s")
        self.delta, self.eta, self.continue_rule = float(delta), float(eta), continue_rule
        self.yard_all = yard_all  # also give the poses the round does not decide their yardstick (termination, WP6)
        self.last = None    # the last round's rows, residual and factorizations (for the nonlinear probe, WP6b)
        self.base = {}      # pose id -> dict(P ids, row cache, valid flag)
        self.yard = {}      # pose id -> dict(P ids, S_lo, L)
        self.sleep = {}     # pose id -> sleep record
        self.prev = None    # previous round's S, SF and pose ids (change detection)

    # ------------------------------------------------------------------ structure
    def _changed_rows(self, M, prevM, ids, prev_ids):
        """Block rows of M that differ from the previous round's (aligned by pose id; new poses count as changed)."""
        n = len(ids)
        if prevM is None:
            return np.ones(n, bool)
        pos = {p: k for k, p in enumerate(prev_ids)}
        common = np.array([pos.get(p, -1) for p in ids])
        P = np.zeros_like(M)
        have = np.flatnonzero(common >= 0)
        if len(have):
            di = (DOF * have[:, None] + np.arange(DOF)).ravel()
            dj = (DOF * common[have][:, None] + np.arange(DOF)).ravel()
            P[np.ix_(di, di)] = prevM[np.ix_(dj, dj)]
        diff = (M != P).reshape(n, DOF, n * DOF).any(axis=(1, 2))
        diff[common < 0] = True
        return diff

    def _dofs(self, P):
        return (DOF * np.asarray(P)[:, None] + np.arange(DOF)).ravel()

    def _row(self, S, g, C, Cinv, adj, i, P, ops):
        P = np.array(sorted(set(int(p) for p in P)))
        dofP = self._dofs(P)
        n = len(dofP)
        key = P.tobytes()
        cf = self._fcache.get(key)  # poses whose patches coincide share one factorization within a round
        if cf is None:
            cf = sla.cho_factor(S[np.ix_(dofP, dofP)], lower=True, check_finite=False)
            self._fcache[key] = cf
            ops.factor(n)
        else:
            ops.c["factorizations_shared"] = ops.c.get("factorizations_shared", 0) + 1
        k = int(np.flatnonzero(P == i)[0])
        Ei = np.zeros((n, DOF))
        Ei[DOF * k:DOF * k + DOF] = np.eye(DOF)
        Z = sla.cho_solve(cf, Ei, check_finite=False)  # [S_PP^-1]_{:, i}
        ops.solve(n, DOF)
        inP = np.zeros(adj.shape[0], bool)
        inP[P] = True
        ext = np.flatnonzero(adj[P].any(axis=0) & ~inP)
        a = np.zeros(len(ext))
        if len(ext):
            bnd = P[adj[np.ix_(P, ext)].any(axis=1)]
            db = self._dofs(bnd)
            rows_b = np.concatenate([np.flatnonzero(P == b) for b in bnd])
            zb = (DOF * rows_b[:, None] + np.arange(DOF)).ravel()
            M = Z[zb].T @ S[np.ix_(db, self._dofs(ext))]  # sum_b [S_PP^-1]_ib S_bj, all j at once
            ops.matvec(2.0 * DOF * len(db) * DOF * len(ext))
            for m, j in enumerate(ext):
                a[m] = np.linalg.norm(C[i].T @ M[:, DOF * m:DOF * m + DOF] @ Cinv[j].T, 2)
        row = Row(i, P, ext, a, Z, dofP)
        row.s = float(np.linalg.norm(C[i].T @ (Z.T @ g[dofP])))
        ops.matvec(2.0 * DOF * n)
        return row

    def _grow_base(self, S, g, C, Cinv, adj, i, P0, ops):
        """Grow a patch until its leak row sum is <= R_BASE (or the cap)."""
        row = self._row(S, g, C, Cinv, adj, i, P0, ops)
        steps = 0
        while row.a.sum() > R_BASE and len(row.P) < min(CAP, adj.shape[0]) and steps < MAX_STEPS and len(row.ext):
            row = self._row(S, g, C, Cinv, adj, i, np.union1d(row.P, row.ext[row.a >= 0.5 * row.a.max()]), ops)
            steps += 1
        return row, steps

    def _grow_decision(self, S, g, C, Cinv, adj, i, base_row, c0, thr, ops):
        """Phase C I1b: grow while the decision is open. With continue_rule "L" the growth exits on CONTINUE only when
        the certified lower bound s_i - sum_j a_ij c0_j exceeds the threshold (c0 are valid upper bounds, so STOP is then
        impossible), at any leak row sum; with "s" (WP2) it exited when s_i alone reached the threshold."""
        row, steps, why = base_row, 0, "cap"
        while True:
            rs, leak = row.a.sum(), float(row.a @ c0[row.ext]) if len(row.ext) else 0.0
            if self.continue_rule == "L" and row.s - leak > thr:
                why = "continue"
                break
            if rs <= R_BASE:
                if row.s + leak <= thr:
                    why = "stop"
                    break
                if self.continue_rule == "s" and row.s >= thr:
                    why = "continue"
                    break
                if leak <= THETA * row.s:
                    why = "tight"
                    break
            if len(row.P) >= min(CAP, adj.shape[0]) or steps >= MAX_STEPS or not len(row.ext):
                break
            score = row.a * c0[row.ext] if rs <= R_BASE else row.a
            row = self._row(S, g, C, Cinv, adj, i, np.union1d(row.P, row.ext[score >= 0.5 * score.max()]), ops)
            steps += 1
        return row, steps, why

    def _yardstick(self, SF, C, i, P, ops):
        dofP = self._dofs(P)
        n = len(dofP)
        key = np.asarray(P).tobytes()
        cf = self._fcacheF.get(key)
        if cf is None:
            cf = sla.cho_factor(SF[np.ix_(dofP, dofP)], lower=True, check_finite=False)
            self._fcacheF[key] = cf
            ops.factor(n, yard=True)
        k = int(np.flatnonzero(np.asarray(P) == i)[0])
        Ei = np.zeros((n, DOF))
        Ei[DOF * k:DOF * k + DOF] = np.eye(DOF)
        L = sla.cho_solve(cf, Ei, check_finite=False)[DOF * k:DOF * k + DOF]
        ops.solve(n, DOF, yard=True)
        L = 0.5 * (L + L.T)
        mu = max(0.0, float(np.linalg.eigvalsh(C[i].T @ L @ C[i])[0]))
        return float(np.sqrt(mu) * Q12), L

    @staticmethod
    def _recursion(rows, s, n):
        """c* = (I - A)^-1 s with a Collatz-Wielandt certificate of rho(A) < 1; returns (E, A, rho_upper, sweeps)."""
        r_, c_, v_ = [], [], []
        for i, row in rows.items():
            r_.extend([i] * len(row.ext))
            c_.extend(row.ext.tolist())
            v_.extend(row.a.tolist())
        A = sp.csr_matrix((v_, (r_, c_)), shape=(n, n))
        rowmax = float(np.max(np.asarray(A.sum(axis=1)).ravel())) if A.nnz else 0.0
        up = rowmax if rowmax < 1.0 else certs.perron(A)[1]
        E, it = certs.solve_comparison(A, s, up)
        return E, A, up, it

    # ------------------------------------------------------------------ one round
    def round(self, rnd, model, modelF, d, relin, asleep_in, sampled=False):
        ops = Ops()
        self._fcache, self._fcacheF = {}, {}
        S, b, x = model["S"], model["b"], model["x"]
        SF = modelF["S"]
        ids = [str(p) for p in model["pose_ids"]]
        n = len(ids)
        g = S @ x - b
        # assumption A0: the current linearization must be positive definite (it is not, e.g., while a pose's
        # velocity has no information yet); then nothing is certified and every pose stays awake this round
        try:
            cfS = sla.cho_factor(S, lower=True, check_finite=True)
            sla.cho_factor(SF, lower=True, check_finite=True)
        except (np.linalg.LinAlgError, ValueError):
            self.sleep.clear()
            self.prev = None
            per_pose = [dict(round=rnd, pose=p, t=float(model["times"][k]), d=float(d[p]), relinearized=int(bool(relin.get(p, False))),
                             decision="A0_FAILS", asleep_before=int(bool(asleep_in.get(p, False))), asleep_after=0)
                        for k, p in enumerate(ids)]
            return [], per_pose, dict(round=rnd, poses=n, a0_fails=1, certified_stop=0.0, asleep_before=sum(map(bool, asleep_in.values())),
                                      asleep_after=0, **ops.c)
        blocks = S.reshape(n, DOF, n, DOF)
        adj = np.abs(blocks).max(axis=(1, 3)) > 0
        np.fill_diagonal(adj, False)
        D = [S[DOF * i:DOF * i + DOF, DOF * i:DOF * i + DOF] for i in range(n)]
        C = [np.linalg.cholesky(0.5 * (Di + Di.T)) for Di in D]
        Cinv = [sla.solve_triangular(Ci, np.eye(DOF), lower=True) for Ci in C]
        kappa = np.array([float(np.linalg.norm(ci, 2)) for ci in Cinv])  # ||C_i^-T|| = lambda_min(D_i)^-1/2
        prev_ids = None if self.prev is None else self.prev["ids"]
        chS = self._changed_rows(S, None if self.prev is None else self.prev["S"], ids, prev_ids)
        chF = self._changed_rows(SF, None if self.prev is None else self.prev["SF"], ids, prev_ids)
        index = {p: k for k, p in enumerate(ids)}
        new_poses = set(ids) - set(prev_ids or [])

        rows, s_lo_of = {}, {}
        rec = {p: dict(refactorized=False, grown_steps=0, why="", patch=0, base_patch=0) for p in ids}
        asleep = {p for p in ids if asleep_in.get(p, False)}

        # base rows (every pose: needed to close the recursion), cached and refactorized only on change
        for p in ids:
            i = index[p]
            if p in self.sleep:
                continue
            cache = self.base.get(p)
            if cache is not None and all(q in index for q in cache["P"]):
                P = np.array([index[q] for q in cache["P"]])
                if not chS[P].any():
                    row = cache["row"]
                    row.s = float(np.linalg.norm(C[i].T @ (row.Z.T @ g[row.dofP])))
                    ops.matvec(2.0 * DOF * len(row.dofP))
                    rows[i] = row
                    rec[p]["base_patch"] = len(row.P)
                    continue
                P0 = P
            else:
                P0 = np.array([i])
            row, steps = self._grow_base(S, g, C, Cinv, adj, i, P0, ops)
            rec[p]["refactorized"] = True
            rec[p]["base_patch"] = len(row.P)
            self.base[p] = dict(P=[ids[k] for k in row.P], row=row)
            rows[i] = row

        # sleeping poses: Theorem 5.4 running bound, or refactorization when their patch changed
        sleep_changed = []
        s_lower = {}
        for p in list(self.sleep):
            if p not in index:
                del self.sleep[p]
                continue
            sl = self.sleep[p]
            i = index[p]
            P = np.array([index[q] for q in sl["P"]])  # poses are never removed, and new ones are appended
            if chS[P].any() or chF[P].any():
                sleep_changed.append(p)
                sl["reason"] = "new_factor" if near_new(new_poses, ids, P, adj) else "relinearization"
                row, steps = self._grow_base(S, g, C, Cinv, adj, i, P, ops)
                rec[p]["refactorized"] = True
                rec[p]["base_patch"] = len(row.P)
                self.base[p] = dict(P=[ids[k] for k in row.P], row=row)
                rows[i] = row
                continue
            dg = [np.linalg.norm(Cinv[k] @ (g[DOF * k:DOF * k + DOF] - sl["g0"][m])) for m, k in enumerate(P)]
            bump = float(np.dot(sl["ell"], dg))
            ops.matvec(2.0 * DOF * DOF * len(P))
            row = sl["row"]
            row.s = sl["s0"] + bump
            s_lower[i] = max(0.0, sl["s0"] - bump)
            rows[i] = row
            rec[p]["base_patch"] = len(row.P)

        # base certificates c0 (used to steer the decision-driven growth)
        s0 = np.array([rows[i].s for i in range(n)])
        c0, _, up0, _ = self._recursion(rows, s0, n)
        if c0 is None:
            c0 = np.full(n, np.inf)

        # poses that can stop this round: not relinearized by the solver (d_i <= delta) and awake, plus sleeping poses
        # whose patch changed; each gets its yardstick and the decision-driven patch
        eligible = [p for p in ids if (not relin.get(p, False) and p not in self.sleep) or p in sleep_changed]
        thr = {}
        yard_current = set()
        for p in eligible:
            i = index[p]
            P = rows[i].P
            yc = self.yard.get(p)
            Pids = [ids[k] for k in P]
            if yc is None or yc["P"] != Pids or chF[P].any():
                S_lo, L = self._yardstick(SF, C, i, P, ops)
                self.yard[p] = yc = dict(P=Pids, S_lo=S_lo, L=L)
            s_lo_of[p] = yc["S_lo"]
            yard_current.add(p)
            thr[p] = min(self.eta * yc["S_lo"], (self.delta - d[p]) / kappa[i]) if d[p] < self.delta else 0.0
            if np.isfinite(c0).all():
                row, steps, why = self._grow_decision(S, g, C, Cinv, adj, i, rows[i], c0, thr[p], ops)
                rec[p]["grown_steps"], rec[p]["why"] = steps, why
                if steps:
                    rec[p]["refactorized"] = True
                rows[i] = row
        for p in self.sleep:
            if p in index and p not in s_lo_of:
                s_lo_of[p] = self.sleep[p]["S_lo"]  # its patch rows of S and SF are unchanged since it fell asleep
                yard_current.add(p)
        if self.yard_all:
            for p in ids:
                if p not in s_lo_of:
                    i = index[p]
                    S_lo, L = self._yardstick(SF, C, i, rows[i].P, ops)
                    self.yard[p] = dict(P=[ids[k] for k in rows[i].P], S_lo=S_lo, L=L)
                    s_lo_of[p] = S_lo
                    yard_current.add(p)

        # final certificates
        s = np.array([rows[i].s for i in range(n)])
        E, A, up, sweeps = self._recursion(rows, s, n)
        if E is None:
            E = np.full(n, np.inf)
            Lo = np.zeros(n)
        else:
            s_lo_vec = np.array([s_lower.get(i, s[i]) for i in range(n)])
            Lo = np.maximum(0.0, s_lo_vec - A @ E)
            ops.c["sweeps"] += int(sweeps or 0)
            ops.c["sweep_flops"] += 2.0 * A.nnz * int(sweeps or 0)

        # decisions
        out_sleep, woke = set(), {}
        decision = {}
        for p in ids:
            i = index[p]
            if p in self.sleep and p not in sleep_changed:
                ok = d[p] + kappa[i] * E[i] <= self.delta and E[i] <= self.eta * s_lo_of[p]
                if ok:
                    out_sleep.add(p)
                    decision[p] = "SLEEP_KEPT"
                else:
                    woke[p] = "recheck"
                    decision[p] = "WOKE"
                    del self.sleep[p]
                continue
            if p not in eligible:
                decision[p] = "NONE"
                continue
            ok = d[p] + kappa[i] * E[i] <= self.delta and E[i] <= self.eta * s_lo_of[p]
            if p in sleep_changed:
                if ok:
                    decision[p] = "SLEEP_KEPT_REFACTORIZED"
                else:
                    woke[p] = self.sleep[p]["reason"]
                    decision[p] = "WOKE"
                    del self.sleep[p]
                    continue
            else:
                decision[p] = "STOP" if ok else "CONTINUE"
                if not ok:
                    continue
            # (re)enter sleep: record the Theorem 5.4 data at t0
            row = rows[i]
            ell = []
            for m, k in enumerate(row.P):
                blk = row.Z[DOF * m:DOF * m + DOF]  # [S_PP^-1]_{k,i}
                ell.append(float(np.linalg.norm(C[i].T @ blk.T @ C[k], 2)))
            self.sleep[p] = dict(P=[ids[k] for k in row.P], row=row, s0=row.s, ell=np.array(ell),
                                 g0=[g[DOF * k:DOF * k + DOF].copy() for k in row.P], S_lo=s_lo_of[p], reason="")
            out_sleep.add(p)
        for p in list(self.sleep):
            if p not in out_sleep:
                del self.sleep[p]

        # oracle and invariants
        e = sla.cho_solve(cfS, g, check_finite=False)
        err = np.array([np.linalg.norm(C[i].T @ e[DOF * i:DOF * i + DOF]) for i in range(n)])
        SFinv = np.linalg.inv(SF)
        S_ex = np.zeros(n)
        loewner = {}
        for i in range(n):
            Sig = SFinv[DOF * i:DOF * i + DOF, DOF * i:DOF * i + DOF]
            Sig = 0.5 * (Sig + Sig.T)
            S_ex[i] = np.sqrt(max(0.0, float(np.linalg.eigvalsh(C[i].T @ Sig @ C[i])[0]))) * Q12
            p = ids[i]
            if sampled and p in yard_current and p in self.yard:
                Dm = C[i].T @ (Sig - self.yard[p]["L"]) @ C[i]
                scale = float(np.linalg.eigvalsh(C[i].T @ Sig @ C[i])[-1])
                loewner[p] = float(np.linalg.eigvalsh(0.5 * (Dm + Dm.T))[0]) / scale
        tol = 1e-9
        fin = np.isfinite(E)
        viol_up = fin & (err > E * (1 + tol) + 1e-12)
        viol_lo = Lo > err * (1 + tol) + 1e-12
        oracle_stop = np.array([(not relin.get(p, False) or p in asleep) and d[p] + kappa[index[p]] * err[index[p]] <= self.delta
                                and err[index[p]] <= self.eta * S_ex[index[p]] for p in ids])

        self.prev = dict(S=S, SF=SF, ids=ids)
        self.last = dict(rows=rows, g=g, C=C, Cinv=Cinv, kappa=kappa, ids=ids, index=index, eligible=eligible, E=E, L=Lo,
                         err=err, S_lo=s_lo_of, S_exact=S_ex, fcache=self._fcache, S=S, ops=ops)
        per_pose = []
        for p in ids:
            i = index[p]
            per_pose.append(dict(round=rnd, pose=p, t=float(model["times"][i]), d=float(d[p]), relinearized=int(bool(relin.get(p, False))),
                                 eligible=int(p in eligible), asleep_before=int(p in asleep), E=float(E[i]), L=float(Lo[i]),
                                 err=float(err[i]), S_lo=float(s_lo_of.get(p, np.nan)), S_exact=float(S_ex[i]), kappa=float(kappa[i]),
                                 margin_delta=float(self.delta - d[p] - kappa[i] * E[i]),
                                 margin_eta=float(self.eta * s_lo_of[p] - E[i]) if p in s_lo_of else np.nan,
                                 decision=decision.get(p, "NONE"), asleep_after=int(p in out_sleep), woke=int(p in woke),
                                 woke_reason=woke.get(p, ""), refactorized=int(rec[p]["refactorized"]),
                                 patch=len(rows[i].P), base_patch=rec[p]["base_patch"], grown_steps=rec[p]["grown_steps"],
                                 grow_why=rec[p]["why"], oracle_stop=int(oracle_stop[i]),
                                 valid_upper=int(not viol_up[i]), valid_lower=int(not viol_lo[i]),
                                 loewner_margin=loewner.get(p, np.nan)))
        summary = dict(round=rnd, poses=n, a0_fails=0, eligible=len(eligible), asleep_before=len(asleep), asleep_after=len(out_sleep),
                       certified_stop=len(out_sleep) / n, oracle_stop=float(oracle_stop.mean()),
                       woke_recheck=sum(1 for v in woke.values() if v == "recheck"),
                       woke_relinearization=sum(1 for v in woke.values() if v == "relinearization"),
                       woke_new_factor=sum(1 for v in woke.values() if v == "new_factor"),
                       refactorized=sum(rec[p]["refactorized"] for p in ids), rho_upper=float(up) if np.isfinite(up) else np.nan,
                       violations_upper=int(viol_up.sum()), violations_lower=int(viol_lo.sum()),
                       loewner_min=float(min(loewner.values())) if loewner else np.nan, **ops.c)
        return sorted(out_sleep), per_pose, summary


def near_new(new_poses, ids, P, adj):
    """A sleeping pose's patch change is attributed to new factors when a pose added this round is in or next to its
    patch (on the incremental ChArUco sequence the new pose's marker observations change the reduced model there)."""
    near = set(int(k) for k in P) | set(int(j) for j in np.flatnonzero(adj[P].any(axis=0)))
    return any(ids[k] in new_poses for k in near)
