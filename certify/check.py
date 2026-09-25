"""Evaluate every certificate, bound and invariant on a model and a set of iterates, with dense oracles.

Used by the unit tests (certify/tests) and by the Phase B sweep (scripts/phaseB_run.py). Invariants are
recorded as relative violations; a check passes when its violation is <= TOL = 1e-9:
    scalar bound E >= u :   (u - E) / max(E, u, 1e-300)
    X <= Y (matrices)   :   covbounds.psd_violation in the block's D-metric
"""

import numpy as np
import scipy.linalg as sla

from certify import certs, covbounds, hubs, radii
from certify import chain as ch

TOL = 1e-9


def viol_scalar(u, E):
    """Relative violation of u <= E (elementwise); NaN/inf-safe (E = inf never violates)."""
    u, E = np.broadcast_arrays(np.atleast_1d(np.asarray(u, float)), np.atleast_1d(np.asarray(E, float)))
    den = np.maximum(np.maximum(np.abs(E), np.abs(u)), 1e-300)
    with np.errstate(invalid="ignore"):
        v = (u - E) / den
    v = np.array(v)
    v[~np.isfinite(E) & (E > 0)] = -1.0
    return v


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, name, v, tol=TOL, **meta):
        v = np.atleast_1d(np.asarray(v, float))
        v = v[np.isfinite(v)]
        worst = float(v.max()) if len(v) else -np.inf
        self.rows.append(dict(check=name, n=int(len(v)), worst=worst, tol=tol, ok=bool(worst <= tol), **meta))

    def failed(self):
        return [r for r in self.rows if not r["ok"]]


def dense_oracle(model):
    """(H dense, solver, Sigma, xbar): solves by Jacobi-scaled Cholesky with extended-precision iterative
    refinement (certify/precise.py), so the oracle is accurate far below the 1e-9 check tolerance."""
    from certify.linalg import SPDSystem
    Hd = model.H.toarray()
    Hd = 0.5 * (Hd + Hd.T)
    sysm = SPDSystem(model.H, model.dims, None, refine=3, K_lo=model.H_lo)
    assert sysm.ok
    Sigma = sysm.solve(np.eye(model.N))
    Sigma = 0.5 * (Sigma + Sigma.T)
    xbar = sysm.solve(model.b)
    return Hd, sysm, Sigma, xbar


def spd_solve(Hsub, R, Hsub_lo=None, R_lo=None):
    """Refined SPD solve of a dense or sparse sub-matrix (double-double residuals with the exact lo parts)."""
    import scipy.sparse as sp
    from certify.linalg import SPDSystem
    sysm = SPDSystem(sp.csr_matrix(Hsub), [Hsub.shape[0]], None, refine=3,
                     K_lo=None if Hsub_lo is None else sp.csr_matrix(Hsub_lo))
    assert sysm.ok
    return sysm.solve(R, R_lo=R_lo)


def sub_matrix(model, blocks, kinds_excluded=()):
    """H' of Lemma 4.3: the factors whose variables all lie in `blocks` (minus excluded kinds), dense on
    the dofs of `blocks` (in the given block order)."""
    blocks = np.asarray(blocks, int)
    pos = -np.ones(model.n_blocks, int)
    pos[blocks] = np.arange(len(blocks))
    off = np.concatenate([[0], np.cumsum(model.dims[blocks])[:-1]])
    N = int(model.dims[blocks].sum())
    Hs = np.zeros((N, N))
    for f in model.factors:
        if f["kind"] in kinds_excluded:
            continue
        inside = np.all(pos[f["vars"]] >= 0, axis=1)
        m = f["vars"].shape[1]
        dims = [int(model.dims[f["vars"][0, a]]) for a in range(m)]
        cs = np.concatenate([[0], np.cumsum(dims)])
        for r in np.flatnonzero(inside):
            for a in range(m):
                for c in range(m):
                    oa, oc = off[pos[f["vars"][r, a]]], off[pos[f["vars"][r, c]]]
                    Hs[oa:oa + dims[a], oc:oc + dims[c]] += f["Hf"][r][cs[a]:cs[a + 1], cs[c]:cs[c + 1]]
    return 0.5 * (Hs + Hs.T)


def evaluate(model, xs, Ks=(1, 2, 3), ells=(3, 5), w=3, forced_hubs=None, hub_window=None, label="",
             one_at_a_time=True):
    """Run everything; return (Checks, info dict)."""
    chk = Checks()
    info = dict(label=label)
    Hd, osys, Sigma, xbar = dense_oracle(model)
    n = model.n_blocks
    met = certs.Metric(model)
    D = model.diag_blocks()
    iters = sorted(xs)
    X = np.stack([xs[t] for t in iters], axis=1)
    G = model.residual(X)  # extended precision
    e_from_g = osys.solve(G)  # the oracle error e = H^-1 g (He = g exactly)
    info["e_consistency"] = float(np.abs(e_from_g - (X - xbar[:, None])).max() / max(np.abs(e_from_g).max(), 1e-300))
    E_true = e_from_g
    u = met.norms(E_true)  # n x t
    info["err"] = u

    # ---------------------------------------------------------------- C1
    r1 = certs.c1(model, met, G)
    info["rho_M"], info["rho_M_upper"] = r1["rho"], r1["rho_upper"]
    chk.add("C1 one-step u<=nu+Mu", viol_scalar(u, r1["nu"] + r1["M"] @ u), label=label)
    if r1["E"] is not None:
        chk.add("C1 E>=err", viol_scalar(u, r1["E"]), label=label)
        info["E_C1"] = r1["E"]
    # ---------------------------------------------------------------- C2
    r2 = certs.c2(model, met, G)
    info["rho_Mhat"], info["rho_Mhat_upper"] = r2["rho"], r2["rho_upper"]
    chk.add("C2 one-step u<=nuhat+Mhat u", viol_scalar(u, r2["nu"] + r2["M"] @ u), label=label)
    if r2["E"] is not None:
        chk.add("C2 E>=err", viol_scalar(u, r2["E"]), label=label)
        info["E_C2"] = r2["E"]
    # ---------------------------------------------------------------- C3 and patch/cut bounds
    prevL, prevU = None, None
    info["C3"] = {}
    for K in Ks:
        P = certs.patch_pass(model, met, K, G, workers=1)
        r3 = certs.c3_from_patches(model, P)
        info["C3"][K] = r3
        chk.add("C3 one-step u<=s+Au K=%d" % K, viol_scalar(u, r3["s"] + r3["A"] @ u), label=label)
        if r3["E"] is not None:
            chk.add("C3 E>=err K=%d" % K, viol_scalar(u, r3["E"]), label=label)
            chk.add("C3 local<=E K=%d" % K, viol_scalar(r3["s"], r3["E"]), label=label)
        Lp = [p["inv_ii"] for p in P]
        chk.add("patch lower<=Sigma K=%d" % K,
                [covbounds.psd_violation(met.C[i], Lp[i], Sigma[model.block_slice(i), model.block_slice(i)]) for i in range(n)],
                label=label)
        Ucut = [p["cut_ii"] for p in P]
        chk.add("cut upper>=Sigma K=%d" % K,
                [covbounds.psd_violation(met.C[i], Sigma[model.block_slice(i), model.block_slice(i)], Ucut[i])
                 for i in range(n) if Ucut[i] is not None], label=label)
        if prevL is not None:
            chk.add("patch lower monotone K=%d" % K, [covbounds.psd_violation(met.C[i], prevL[i], Lp[i]) for i in range(n)], label=label)
            chk.add("cut upper monotone K=%d" % K, [covbounds.psd_violation(met.C[i], Ucut[i], prevU[i]) for i in range(n)
                                                   if Ucut[i] is not None and prevU[i] is not None], label=label)
        prevL, prevU = Lp, Ucut
        info["C3"][K]["certifiable"] = np.array([p["certifiable"] for p in P])
        info["C3"][K]["patch_lower"] = Lp
        info["C3"][K]["cut_upper"] = Ucut
    # trivial and Krylov
    Lt = [np.linalg.inv(D[i]) for i in range(n)]
    chk.add("trivial lower<=Sigma", [covbounds.psd_violation(met.C[i], Lt[i], Sigma[model.block_slice(i), model.block_slice(i)])
                                     for i in range(n)], label=label)
    prevK = Lt
    kl = sorted(set(list(ells) + [2, 3, 4, 5, 6]))
    KM = [covbounds.krylov_lower_multi(model, i, kl) for i in range(n)]
    for ell in kl:
        Lk = [KM[i][ell][0] for i in range(n)]
        chk.add("krylov lower<=Sigma l=%d" % ell,
                [covbounds.psd_violation(met.C[i], Lk[i], Sigma[model.block_slice(i), model.block_slice(i)]) for i in range(n)],
                label=label)
        chk.add("krylov monotone l=%d" % ell, [covbounds.psd_violation(met.C[i], prevK[i], Lk[i]) for i in range(n)], label=label)
        prevK = Lk
        if ell in ells:
            info["krylov_%d" % ell] = Lk
    # radii sandwich
    mu_or = np.array([covbounds.mu_extremes(met.C[i], Sigma[model.block_slice(i), model.block_slice(i)]) for i in range(n)])
    info["mu_oracle"] = mu_or
    # global lower bounds (Proposition 5.7)
    gl = []
    for j in range(G.shape[1]):
        lb = radii.global_lower_bounds(model.H, G[:, j], steps=(1, 5))
        orc = float(G[:, j] @ e_from_g[:, j])
        gl.append((lb[1], lb[5], orc))
        chk.add("global lb1<=energy", viol_scalar(lb[1], orc), label=label)
        chk.add("global lb5<=energy", viol_scalar(lb[5], orc), label=label)
        chk.add("global lb1<=lb5", viol_scalar(lb[1], lb[5]), label=label)
    info["global_lb"] = gl

    # ---------------------------------------------------------------- chain, C4, hubs
    if np.sum(model.kinds == "pose") >= 3 and len(model.gp_pairs()):
        part = hubs.Partition(model, w, forced_hubs=forced_hubs)
        sc = hubs.SupernodeChain(model, part)
        info["partition"] = part
        T, S = part.T, part.S
        Td, Sd = model.dof_index(T), model.dof_index(S)
        HTT = Hd[np.ix_(Td, Td)]
        HTT_lo = model.H_lo.toarray()[np.ix_(Td, Td)]
        et = spd_solve(HTT, G[Td], HTT_lo)  # e~_T (hub-clamped error), in T-dof order
        # map to supernode blocks
        pos_T = -np.ones(model.N, int)
        pos_T[Td] = np.arange(len(Td))
        GT = [G[d] for d in sc.dofs]
        et_sn = [et[pos_T[d]] for d in sc.dofs]
        u_sn = np.array([np.linalg.norm(sc.CD[k].T @ et_sn[k], axis=0) for k in range(sc.n)])
        kap = sc.kappa()
        info["kappa"] = kap
        u_clamped = np.zeros((n, G.shape[1]))
        for b in T:
            u_clamped[b] = np.linalg.norm(met.C[b].T @ et[pos_T[model.dof_index([b])]], axis=0)
        info["err_clamped"] = u_clamped
        # chain lemmas on H_TT itself
        Gb = ch.backward_schur(sc.Pd, sc.Po)
        Fb = ch.forward_schur(sc.Pd, sc.Po)
        for m in range(sc.n):
            Lb = ch.truncated_backward(sc.Pd, sc.Po, m, sc.seed_back[m])
            chk.add("Lemma1.4 seed/1.3 L<=G (H_TT)", [covbounds.psd_violation(sc.CD[k], Lb[k], Gb[k]) for k in Lb], label=label)
            Lf = ch.truncated_forward(sc.Pd, sc.Po, m, sc.seed_fwd[m])
            chk.add("Lemma1.4 mirror L<=F (H_TT)", [covbounds.psd_violation(sc.CD[k], Lf[k], Fb[k]) for k in Lf], label=label)
        info["C4"] = {}
        SigT = spd_solve(HTT, np.eye(len(HTT)), HTT_lo)
        for K in Ks:
            r4 = hubs.c4(sc, K, GT)
            info["C4"][K] = r4
            chk.add("C4 one-step u~<=s+Au~ K=%d" % K, viol_scalar(u_sn, r4["s"] + r4["A"] @ u_sn), label=label)
            if r4["E"] is not None:
                chk.add("C4 c*>=err~ (supernode) K=%d" % K, viol_scalar(u_sn, r4["E"]), label=label)
                per = np.zeros((n, G.shape[1]))
                for k, g in enumerate(part.groups):
                    per[g] = np.sqrt(kap[g])[:, None] * r4["E"][k][None, :]
                chk.add("C4 per-block >= err~ K=%d" % K, viol_scalar(u_clamped[T], per[T]), label=label)
                if len(S) == 0:
                    chk.add("C4 per-block >= err (S empty) K=%d" % K, viol_scalar(u[T], per[T]), label=label)
            # Proposition 3.6 coefficient bounds
            chk.add("Prop3.6 a<=bound K=%d" % K, viol_scalar(r4["coeff"], r4["coeff_bound"]), label=label)
            # chain cut upper bound >= Sigma^T_II
            chk.add("chain cut >= Sigma^T K=%d" % K,
                    [covbounds.psd_violation(sc.CD[I], SigT[np.ix_(pos_T[sc.dofs[I]], pos_T[sc.dofs[I]])], r4["cut_II"][I])
                     for I in range(sc.n) if r4["cut_II"][I] is not None], label=label)
        # Lemma 1.6 on H_TT with the certified links
        psi = {K: sc.psibar_links(K) for K in Ks}
        info["psibar"] = psi
        SigT_sn = lambda I, J: SigT[np.ix_(pos_T[sc.dofs[I]], pos_T[sc.dofs[J]])]
        for K in Ks:
            v = []
            for I in range(sc.n):
                lam = ch.lam_max_scaled(sc.Pd[I], SigT_sn(I, I))
                for J in range(sc.n):
                    if J == I:
                        continue
                    lhs = ch.scaled_norm(sc.Pd[I], SigT_sn(I, J), sc.Pd[J])
                    prod = np.prod(psi[K]["right"][I:J]) if J > I else np.prod(psi[K]["left"][J + 1:I + 1])
                    v.append(float(viol_scalar(lhs, lam * prod)))
            chk.add("Lemma1.6 decay (H_TT) K=%d" % K, v, label=label)

        if len(S):
            info["hubs"] = hub_checks(model, chk, label, Hd, Sigma, G, E_true, u, met, part, sc, psi, kap, info, Ks,
                                      hub_window, one_at_a_time)
    return chk, info


def hub_checks(model, chk, label, Hd, Sigma, G, E_true, u, met, part, sc, psi, kap, info, Ks, hub_window, one_at_a_time):
    """Theorem 4.2, Lemmas 4.3-4.5, Theorem 4.6, Proposition 4.7 (joint S) and the one-at-a-time variant."""
    n = model.n_blocks
    T, S = part.T, part.S
    Td, Sd = model.dof_index(T), model.dof_index(S)
    HTT, HTS, HSS = Hd[np.ix_(Td, Td)], Hd[np.ix_(Td, Sd)], Hd[np.ix_(Sd, Sd)]
    Hl = model.H_lo.toarray()
    HTT_lo, HTS_lo = Hl[np.ix_(Td, Td)], Hl[np.ix_(Td, Sd)]
    X = spd_solve(HTT, HTS, HTT_lo, R_lo=HTS_lo)
    Shat = HSS - HTS.T @ X
    gT, gS = G[Td], G[Sd]
    et = spd_solve(HTT, gT, HTT_lo)
    rS = gS - HTS.T @ et
    eS = spd_solve(Shat, rS)
    eT = et - X @ eS
    # Theorem 4.2
    rec = np.zeros_like(E_true)
    rec[Td], rec[Sd] = eT, eS
    chk.add("Thm4.2 decomposition (rel err)", np.abs(rec - E_true).max() / max(np.abs(E_true).max(), 1e-300), tol=1e-10,
            label=label)
    out = dict(theta_oracle=None)
    CS = np.linalg.cholesky(HSS)
    theta_or = hubs.theta(HSS, Shat)
    out["theta_oracle"] = theta_or
    # Lemma 4.3 with a local window (anchor + first observers), if requested
    if hub_window is not None:
        blocks = np.unique(np.concatenate([S, hub_window]))
        Hp = sub_matrix(model, blocks)
        pos = np.searchsorted(blocks, S)
        sd = np.concatenate([np.arange(o, o + d) for o, d in zip(np.concatenate([[0], np.cumsum(model.dims[blocks])[:-1]])[pos], model.dims[S])])
        rest = np.setdiff1d(np.arange(len(Hp)), sd)
        Sloc = Hp[np.ix_(sd, sd)] - Hp[np.ix_(sd, rest)] @ spd_solve(Hp[np.ix_(rest, rest)], Hp[np.ix_(rest, sd)])
        chk.add("Lemma4.3 Shat_loc<=Shat", covbounds.psd_violation(np.linalg.cholesky(HSS), Sloc, Shat), label=label)
        theta_loc = hubs.theta(HSS, Sloc)
        out["theta_local"] = theta_loc
    else:
        theta_loc = theta_or  # window = whole graph: Shat_loc = Shat
        out["theta_local"] = theta_loc
    hs_block_diag = True
    for a in range(len(S)):
        for b in range(len(S)):
            if a != b and np.abs(Hd[np.ix_(model.dof_index([S[a]]), model.dof_index([S[b]]))]).max() > 0:
                hs_block_diag = False
    out["HSS_block_diagonal"] = hs_block_diag
    mSJ, _, _, _ = hubs.hub_coupling_norms(model, sc, part, S)
    chk.add("m_SJ<=1", viol_scalar(mSJ, np.ones_like(mSJ)), label=label)
    eS_norm = np.linalg.norm(CS.T @ eS, axis=0)
    gS_dual = np.linalg.norm(sla.solve_triangular(CS, gS, lower=True), axis=0)
    rS_dual = np.linalg.norm(sla.solve_triangular(CS, rS, lower=True), axis=0)
    # per-supernode oracle phi_IS
    pos_T = -np.ones(model.N, int)
    pos_T[Td] = np.arange(len(Td))
    phi_or = np.array([np.linalg.norm(sc.CD[I].T @ X[pos_T[sc.dofs[I]]] @ np.linalg.inv(CS).T, 2) for I in range(sc.n)])
    out["phi_oracle"] = phi_or
    out["E_S"], out["E_hub"] = {}, {}
    for K in Ks:
        r4 = info["C4"][K]
        if r4["E"] is None:
            continue
        chat = r4["E"]  # n_super x t
        # Lemma 4.4 with the certificate and with the exact chain errors (tightest)
        E_S = theta_loc * (gS_dual + (mSJ[:, None] * chat).sum(axis=0))
        chk.add("Lemma4.4 ||e_S||<=E_S K=%d" % K, viol_scalar(eS_norm, E_S), label=label)
        chk.add("sec4.7 energy <= E_S K=%d" % K, viol_scalar(theta_or * rS_dual, E_S), label=label)
        chk.add("||e_S|| <= theta*||r_S|| (oracle aggregate)", viol_scalar(eS_norm, theta_or * rS_dual), label=label)
        # Lemma 4.5
        mu_hi_T = np.array([ch.lam_max_scaled(sc.Pd[I], r4["cut_II"][I]) if r4["cut_II"][I] is not None else np.inf
                            for I in range(sc.n)])
        pb, pbt = hubs.phibar(sc, psi[K], mu_hi_T, mSJ, K=K)
        chk.add("Lemma4.5 phi<=phibar K=%d" % K, viol_scalar(phi_or, pb), label=label)
        chk.add("Lemma4.5 phi<=phibar(trunc+tail) K=%d" % K, viol_scalar(phi_or, pbt), label=label)
        # Theorem 4.6 (supernode level and per block)
        E_sn = chat + pb[:, None] * E_S[None, :]
        u_sn = np.array([np.linalg.norm(sc.CD[I].T @ (E_true[sc.dofs[I]]), axis=0) for I in range(sc.n)])
        chk.add("Thm4.6 supernode K=%d" % K, viol_scalar(u_sn, E_sn), label=label)
        per = np.zeros_like(u)
        for I, g in enumerate(part.groups):
            per[g] = np.sqrt(kap[g])[:, None] * E_sn[I][None, :]
        per[S] = E_S[None, :] if hs_block_diag else np.inf
        chk.add("Thm4.6 per-block E_hub>=err K=%d" % K, viol_scalar(u, per), label=label)
        out["E_S"][K], out["E_hub"][K] = E_S, per
        # Proposition 4.7 (per block in T)
        v = []
        for I, g in enumerate(part.groups):
            for b in g:
                o = sc.loc[b]
                d = int(model.dims[b])
                cut = r4["cut_II"][I]
                if cut is None:
                    continue
                muT = covbounds.mu_extremes(met.C[b], cut[o:o + d, o:o + d])[1]
                bound = muT + theta_loc * kap[b] * pb[I] ** 2
                lam = covbounds.mu_extremes(met.C[b], Sigma[model.block_slice(b), model.block_slice(b)])[1]
                v.append(float(viol_scalar(lam, bound)))
        chk.add("Prop4.7 hub-aware upper K=%d" % K, v, label=label)
        # one hub at a time: E_i = sqrt(kappa)(chat + sum_h phibar_ih E_h), E_h = C3 certificate of hub h
        if one_at_a_time and info["C3"].get(K) is not None and info["C3"][K]["E"] is not None:
            E3 = info["C3"][K]["E"]
            tot = np.zeros_like(chat)
            for h in S:
                mh, _, _, _ = hubs.hub_coupling_norms(model, sc, part, [h])
                pbh, _ = hubs.phibar(sc, psi[K], mu_hi_T, mh, K=None)
                # oracle check of Lemma 4.5 for the single hub
                hd = model.dof_index([h])
                Xh = spd_solve(HTT, Hd[np.ix_(Td, hd)], HTT_lo, R_lo=Hl[np.ix_(Td, hd)])
                Ch = np.linalg.cholesky(Hd[np.ix_(hd, hd)])
                phih = np.array([np.linalg.norm(sc.CD[I].T @ Xh[pos_T[sc.dofs[I]]] @ np.linalg.inv(Ch).T, 2) for I in range(sc.n)])
                chk.add("Lemma4.5 single hub K=%d" % K, viol_scalar(phih, pbh), label=label)
                tot += pbh[:, None] * E3[h][None, :]
            E1 = chat + tot
            chk.add("one-at-a-time supernode K=%d" % K, viol_scalar(u_sn, E1), label=label)
            per1 = np.zeros_like(u)
            for I, g in enumerate(part.groups):
                per1[g] = np.sqrt(kap[g])[:, None] * E1[I][None, :]
            per1[S] = E3[S]
            chk.add("one-at-a-time per-block K=%d" % K, viol_scalar(u, per1), label=label)
            out.setdefault("E_hub_1at", {})[K] = per1
    return out
