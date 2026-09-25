"""Supernodes, hubs, the chain certificate C4 and the hub correction (PROOF_NOTE §3.5, §4; brief §2.4, §5 C4, §6).

Partition (§2.4): consecutive poses are grouped into supernodes of w poses; a non-pose block whose
observing poses span at most w consecutive poses is assigned to the earlier supernode it touches; every
other non-pose block (and one endpoint of any pose-pose factor spanning more than w poses) is a hub (S).
Blocks can also be forced into S (ChArUco: all markers, decision of the Phase A review). H_TT is then
block-tridiagonal in supernode order (Lemma 4.1); this is asserted.

C4 is the recursive certificate (Theorem 3.5) with supernodes as blocks, metric D_I = (H_TT)_II and
patches [I-K, I+K] (Proposition 3.6). It bounds the error e~_T = H_TT^-1 g_T of the hub-clamped problem;
with S empty that is e_T. A per-variable bound follows from the supernode bound by
    ||v_i||_{H_ii} <= sqrt(kappa_i) ||v_I||_{D_I},   kappa_i = lambda_max(H_ii^{1/2} [D_I^-1]_ii H_ii^{1/2}),
(Cauchy-Schwarz on v_i = E_i^T v_I); kappa_i = 1 for single-block supernodes.

Hubs jointly (Theorem 4.6): E_i = c^_i + phibar_iS E_S with E_S from Lemma 4.4 and phibar from Lemma 4.5.
Hubs one at a time (PROOF_NOTE §4.5 "treat hubs one at a time (valid, loosens with their number)"): the
note gives no formula. The construction used here, valid by Theorem 4.2 and the triangle inequality: with
S the union of hubs h (hubs are never directly coupled, H_hh' = 0),
    e_T = e~_T - sum_h H_TT^-1 H_Th e_h,   so   ||e_i||_{D_i} <= c^_i + sum_h phibar_ih E_h,
where phibar_ih is Lemma 4.5 for the single hub h and E_h >= ||e_h||_{H_hh} is the C3 certificate of hub
block h (Theorem 3.5 on the full model, the hub treated as an ordinary block). Lemma 4.4 is not used for
the individual hubs: applied to one hub it needs the error of the problem with only that hub clamped,
not the chain error, and replacing one by the other is not valid.
"""

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from certify import chain as ch
from certify import precise
from certify.certs import perron, solve_comparison


def _sym(M):
    return 0.5 * (M + M.T)


# ------------------------------------------------------------------------------------------ partition
class Partition:
    def __init__(self, model, w, forced_hubs=None):
        self.w = int(w)
        n = model.n_blocks
        poses = np.flatnonzero(model.kinds == "pose")
        poses = poses[np.argsort(model.pose_index[poses])]
        rank = -np.ones(n, int)
        rank[poses] = np.arange(len(poses))
        self.rank = rank
        sn = -np.ones(n, int)
        sn[poses] = rank[poses] // self.w
        hub = np.zeros(n, bool)
        if forced_hubs is not None:
            hub[np.asarray(forced_hubs, int)] = True
        span = np.zeros(n, int)
        for j in np.flatnonzero(model.kinds != "pose"):
            nb = model.neighbours(j)
            obs = nb[rank[nb] >= 0]
            if len(obs):
                r = rank[obs]
                span[j] = int(r.max() - r.min() + 1)
            if hub[j]:
                continue
            if len(obs) == 0:
                raise ValueError("non-pose block %d has no observing pose" % j)
            if span[j] <= self.w:
                sn[j] = int(rank[obs].min() // self.w)
            else:
                hub[j] = True
        # pose-pose factors spanning more than w poses (loop closures): the later endpoint becomes a hub
        A = sp.triu(model.adj, k=1).tocoo()
        lc = (rank[A.row] >= 0) & (rank[A.col] >= 0) & (np.abs(rank[A.row] - rank[A.col]) > self.w)
        for a, b in zip(A.row[lc], A.col[lc]):
            hub[a if rank[a] > rank[b] else b] = True
        sn[hub] = -1
        self.sn, self.hub, self.span = sn, hub, span
        self.S = np.flatnonzero(hub)
        self.n_super = int(sn.max() + 1) if (sn >= 0).any() else 0
        groups = []
        for k in range(self.n_super):
            members = np.flatnonzero(sn == k)
            ps = members[rank[members] >= 0]
            ps = ps[np.argsort(rank[ps])]
            others = np.sort(members[rank[members] < 0])
            groups.append(np.concatenate([ps, others]))
        self.groups = groups
        self.T = np.concatenate(groups) if groups else np.zeros(0, int)
        # Lemma 4.1: H_TT block-tridiagonal in supernode order
        inT = sn >= 0
        B = A
        both = inT[B.row] & inT[B.col]
        gap = np.abs(sn[B.row[both]] - sn[B.col[both]])
        assert np.all(gap <= 1), "H_TT is not block-tridiagonal in supernode order"
        self.max_gap = int(gap.max()) if len(gap) else 0


# ------------------------------------------------------------------------------------------ chain on H_TT
class SupernodeChain:
    """H_TT in supernode blocks: Pd[k], Po[k] = (H_TT)_{k,k+1}, the Lemma 1.4 seeds and the metric."""

    def __init__(self, model, part):
        self.model, self.part = model, part
        H = model.H.tocsr()
        G = part.groups
        n = len(G)
        self.n = n
        self.dofs = [model.dof_index(g) for g in G]
        self.Pd = [_sym(H[d][:, d].toarray()) for d in self.dofs]
        self.Po = [H[self.dofs[k]][:, self.dofs[k + 1]].toarray() for k in range(n - 1)]
        Hl = model.H_lo.tocsr()
        self.Pd_lo = [Hl[d][:, d].toarray() for d in self.dofs]
        self.Po_lo = [Hl[self.dofs[k]][:, self.dofs[k + 1]].toarray() for k in range(n - 1)]
        # local offsets of each T block inside its supernode
        self.loc = np.full(model.n_blocks, -1)
        for g in G:
            self.loc[g] = np.concatenate([[0], np.cumsum(model.dims[g])[:-1]])
        # factors linking supernodes k and k+1: their contributions to Pd[k] (Bf) and Pd[k+1] (Bb)
        LD = precise.LD
        self.Bf = [np.zeros(P.shape, dtype=LD) for P in self.Pd]
        self.Bb = [np.zeros(P.shape, dtype=LD) for P in self.Pd]
        sn = part.sn
        for f in model.factors:
            V = f["vars"]
            m = V.shape[1]
            if m < 2:
                continue
            S = sn[V]
            valid = S >= 0
            smin = np.where(valid, S, 10 ** 9).min(axis=1)
            smax = np.where(valid, S, -1).max(axis=1)
            cross = np.flatnonzero((smax > smin) & (smax >= 0))
            if not len(cross):
                continue
            dims = [int(model.dims[V[0, a]]) for a in range(m)]
            cs = np.concatenate([[0], np.cumsum(dims)])
            for r in cross:
                k = smin[r]
                assert smax[r] == k + 1
                Hf = f["Hf"][r].astype(LD) + f["Hf_lo"][r].astype(LD)
                for a in range(m):
                    for c in range(m):
                        if S[r, a] < 0 or S[r, c] < 0 or S[r, a] != S[r, c]:
                            continue
                        blk = Hf[cs[a]:cs[a + 1], cs[c]:cs[c + 1]]
                        oa, oc = self.loc[V[r, a]], self.loc[V[r, c]]
                        tgt = self.Bf[k] if S[r, a] == k else self.Bb[k + 1]
                        tgt[oa:oa + dims[a], oc:oc + dims[c]] += blk
        # Lemma 1.4 seeds, formed exactly (extended precision), kept as hi (+ lo for refined solves)
        Pd_ld = [self.Pd[k].astype(LD) + self.Pd_lo[k].astype(LD) for k in range(n)]
        sb = [precise.split_ld(Pd_ld[k] - self.Bf[k]) for k in range(n)]
        sf = [precise.split_ld(Pd_ld[k] - self.Bb[k]) for k in range(n)]
        self.seed_back = [_sym(h) for h, _ in sb]       # P_kk^- (future removed)
        self.seed_back_lo = [l for _, l in sb]
        self.seed_fwd = [_sym(h) for h, _ in sf]        # mirror (past removed)
        self.seed_fwd_lo = [l for _, l in sf]
        self.Pd_ld = Pd_ld
        self.CD = [np.linalg.cholesky(P) for P in self.Pd]

    def kappa(self):
        """kappa_i = lambda_max(H_ii^{1/2} [D_I^-1]_ii H_ii^{1/2}) for every T block (per-variable conversion)."""
        m = self.model
        D = m.diag_blocks()
        out = np.full(m.n_blocks, np.nan)
        for k, g in enumerate(self.part.groups):
            Dinv = sla.cho_solve((self.CD[k], True), np.eye(len(self.Pd[k])))
            for b in g:
                o, d = self.loc[b], int(m.dims[b])
                Cb = np.linalg.cholesky(D[b])
                out[b] = float(np.linalg.eigvalsh(_sym(Cb.T @ Dinv[o:o + d, o:o + d] @ Cb))[-1])
        return out

    def to_blocks(self, v):
        """Split a T-dof vector (in the model's global numbering) into supernode parts."""
        return [v[d] for d in self.dofs]

    # ---------------------------------------------------------------- decay factors (Definition 1.5)
    def psibar_links(self, K):
        """Certified decay factors on every link: right[k] for (k -> k+1) with L_{k+1} <= G_{k+1} seeded
        at m = min(k+1+K, n-1) by Lemma 1.4; left[k] for (k -> k-1) with forward lower bounds seeded at
        max(k-1-K, 0). Also the survival and coupling factors."""
        n = self.n
        right = np.full(n, np.nan)
        left = np.full(n, np.nan)
        surv_r, coup = np.full(n, np.nan), np.full(n, np.nan)
        surv_l = np.full(n, np.nan)
        for k in range(n - 1):
            m = min(k + 1 + K, n - 1)
            L = ch.truncated_backward(self.Pd, self.Po, m, self.seed_back[m], k0=k + 1)
            psi, s, c = ch.psibar(self.Pd[k + 1], L[k + 1], self.Po[k].T, self.Pd[k])
            right[k], surv_r[k], coup[k] = psi, s, c
        for k in range(1, n):
            m = max(k - 1 - K, 0)
            L = ch.truncated_forward(self.Pd, self.Po, m, self.seed_fwd[m], k1=k - 1)
            psi, s, c = ch.psibar(self.Pd[k - 1], L[k - 1], self.Po[k - 1], self.Pd[k])
            left[k], surv_l[k] = psi, s
        return dict(right=right, left=left, survival_right=surv_r, survival_left=surv_l, coupling=coup)


# ------------------------------------------------------------------------------------------ C4
def c4(sc, K, GT, refine=None):
    """Proposition 3.6 on the supernode chain. GT: list over supernodes of residual blocks (d_k x t).
    Returns s, A, rho, c*, the coefficient bounds of Lemma 1.6, and the chain cut upper bounds.
    refine: extended-precision refinement steps of the block-tridiagonal solves (default 2 for supernode
    blocks of at most 200 dof, else 0: large dense blocks are solved in double precision)."""
    if refine is None:
        refine = 2 if max(len(P) for P in sc.Pd) <= 200 else 0
    n = sc.n
    t = GT[0].shape[1]
    s = np.zeros((n, t))
    rows, cols, vals, bounds = [], [], [], []
    inv_II, cut_II = [None] * n, [None] * n
    certifiable = np.zeros(n, bool)
    lam_patch = np.zeros(n)
    for I in range(n):
        a, b = max(0, I - K), min(n - 1, I + K)
        bt = ch.BlockTridiag(sc.Pd[a:b + 1], sc.Po[a:b], refine=refine, Pd_lo=sc.Pd_lo[a:b + 1], Po_lo=sc.Po_lo[a:b])
        col = bt.inv_block_column(I - a)  # [H_PP^-1]_{k, I}
        inv_II[I] = _sym(col[I - a])
        z = sum(col[k - a].T @ GT[k] for k in range(a, b + 1))
        s[I] = np.linalg.norm(sc.CD[I].T @ z, axis=0)
        lam_patch[I] = ch.lam_max_scaled(sc.Pd[I], inv_II[I])
        # right boundary coefficient a_{I,b+1}
        if b < n - 1:
            X = col[b - a].T @ sc.Po[b]  # [H_PP^-1]_{I,b} H_{b,b+1}
            val = float(np.linalg.norm(sc.CD[I].T @ sla.solve_triangular(sc.CD[b + 1], X.T, lower=True).T, 2))
            # certified bound: lambda_max * prod psibar (inside the patch, seeded at b by Lemma 1.4) * m_b
            L = ch.truncated_backward(sc.Pd, sc.Po, b, sc.seed_back[b], k0=I + 1) if b > I else {}
            prod = 1.0
            for k in range(I, b):
                prod *= ch.psibar(sc.Pd[k + 1], L[k + 1], sc.Po[k].T, sc.Pd[k])[0]
            mb = ch.normalized_coupling(sc.Pd[b], sc.Po[b], sc.Pd[b + 1])
            rows.append(I), cols.append(b + 1), vals.append(val), bounds.append(lam_patch[I] * prod * mb)
        if a > 0:
            X = col[0].T @ sc.Po[a - 1].T  # [H_PP^-1]_{I,a} H_{a,a-1}
            val = float(np.linalg.norm(sc.CD[I].T @ sla.solve_triangular(sc.CD[a - 1], X.T, lower=True).T, 2))
            L = ch.truncated_forward(sc.Pd, sc.Po, a, sc.seed_fwd[a], k1=I - 1) if a < I else {}
            prod = 1.0
            for k in range(I, a, -1):
                prod *= ch.psibar(sc.Pd[k - 1], L[k - 1], sc.Po[k - 1], sc.Pd[k])[0]
            ma = ch.normalized_coupling(sc.Pd[a], sc.Po[a - 1].T, sc.Pd[a - 1])
            rows.append(I), cols.append(a - 1), vals.append(val), bounds.append(lam_patch[I] * prod * ma)
        # chain cut upper bound (Theorem 2.4 on H_TT): boundary factor contributions removed at a and b
        Pl = [P.copy() for P in sc.Pd_ld[a:b + 1]]
        if a > 0:
            Pl[0] = Pl[0] - sc.Bb[a]      # factors linking a-1 and a
        if b < n - 1:
            Pl[-1] = Pl[-1] - sc.Bf[b]    # factors linking b and b+1 (same block as Pl[0] when a == b)
        spl = [precise.split_ld(P) for P in Pl]
        Pd = [_sym(h) for h, _ in spl]
        try:
            btc = ch.BlockTridiag(Pd, sc.Po[a:b], refine=max(refine, 1), Pd_lo=[l for _, l in spl], Po_lo=sc.Po_lo[a:b])
            colc = btc.inv_block_column(I - a)
            cut_II[I] = _sym(colc[I - a])
            certifiable[I] = bool(np.all(np.linalg.eigvalsh(cut_II[I]) > 0))
            if not certifiable[I]:
                cut_II[I] = None
        except (np.linalg.LinAlgError, precise.NotConverged):
            certifiable[I] = False
            cut_II[I] = None
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    rho, up = perron(A)
    c, it = solve_comparison(A, s, up)
    return dict(s=s, A=A, rho=rho, rho_upper=up, E=c, iterations=it, coeff=np.array(vals), coeff_bound=np.array(bounds),
                coeff_rows=np.array(rows, int), coeff_cols=np.array(cols, int), inv_II=inv_II, cut_II=cut_II,
                certifiable=certifiable)


# ------------------------------------------------------------------------------------------ hub terms
def hub_coupling_norms(model, sc, part, S_blocks):
    """m_SJ = ||H_SS^{-1/2} H_SJ D_J^{-1/2}||_2 for every supernode J (0 where not coupled), with S the
    given hub blocks jointly; also H_SS and its Cholesky factor."""
    H = model.H.tocsr()
    sd = model.dof_index(S_blocks)
    HSS = _sym(H[sd][:, sd].toarray())
    CS = np.linalg.cholesky(HSS)
    m = np.zeros(sc.n)
    for J in range(sc.n):
        X = H[sd][:, sc.dofs[J]]
        if X.nnz == 0:
            continue
        Y = sla.solve_triangular(CS, X.toarray(), lower=True)
        Y = sla.solve_triangular(sc.CD[J], Y.T, lower=True).T
        m[J] = float(np.linalg.norm(Y, 2))
    return m, HSS, CS, sd


def phibar(sc, psi, mu_hi_T, mJS, K=None):
    """Lemma 4.5 at supernode level: phibar_IS = mu_hi_T[I] * sum_J (prod of psibar along I->J) m'_JS.
    Returns (full-sum bound, truncated-at-K sum plus the tail bound 2 mu psibar^{K+1}/(1-psibar) or inf)."""
    n = sc.n
    full = np.zeros(n)
    trunc = np.zeros(n)
    for I in range(n):
        tot, tk = 0.0, 0.0
        for J in np.flatnonzero(mJS > 0):
            if J == I:
                p = 1.0
            elif J > I:
                p = float(np.prod(psi["right"][I:J]))
            else:
                p = float(np.prod(psi["left"][J + 1:I + 1]))
            tot += p * mJS[J]
            if K is not None and abs(J - I) <= K:
                tk += p * mJS[J]
        full[I] = mu_hi_T[I] * tot
        if K is not None:
            # tail beyond K links on each side: products bounded by psibar_side^d with psibar_side the
            # largest decay factor on that side from I outward (m'_JS <= 1, one supernode per distance/side)
            tail = 0.0
            if I + K + 1 <= n - 1:
                pr = float(np.max(psi["right"][I:n - 1]))
                tail += pr ** (K + 1) / (1 - pr) if pr < 1 else np.inf
            if I - K - 1 >= 0:
                pl = float(np.max(psi["left"][1:I + 1]))
                tail += pl ** (K + 1) / (1 - pl) if pl < 1 else np.inf
            trunc[I] = mu_hi_T[I] * (tk + tail)
    return full, trunc


def theta(HSS, Shat_loc):
    """theta_S = lambda_max(H_SS^{1/2} Shat_loc^-1 H_SS^{1/2}) = largest generalized eigenvalue of (H_SS, Shat_loc)."""
    return float(sla.eigh(_sym(HSS), _sym(Shat_loc), eigvals_only=True)[-1])
