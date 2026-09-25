"""Computational certificates (PROOF_NOTE §3; brief §5). Each is a per-block scalar E_i with the invariant
E_i >= ||e_i||_{D_i}, e = x - xbar, from the residual g = H x - b alone.

Block norms are taken in the D-metric throughout: for a block matrix X_ij,
    ||X_ij||_{D} := ||D_i^{1/2} X_ij D_j^{-1/2}||_2,
which is how the comparison matrices must be read for Theorem 3.2 to reduce to Theorem 3.1 at P = D
(unit test test_c2_reduces_to_c1). ρ(A) < 1 is certified by a Collatz-Wielandt bound: a positive vector x
with max_i (A x)_i / x_i < 1 proves ρ(A) < 1; certificates are computed only then.

C1  classical comparison bound (Theorem 3.1)
C2  backbone-preconditioned comparison bound (Theorem 3.2)
C3  recursive patch certificate (Theorem 3.5, Definition 3.4)
C4  chain certificate on the supernode chain (Proposition 3.6): certify/hubs.py, c4()
"""

import os
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from certify import precise
from certify.linalg import SPDSystem

TOL = 1e-9
MONOTONE_CAP = 5000


# ------------------------------------------------------------------------------------------ metric
class Metric:
    """Cholesky factors C_i of D_i = H_ii (lower) and C_i^-1, grouped by block dimension for batching."""

    def __init__(self, model):
        self.model = model
        D = model.diag_blocks()
        self.C = [np.linalg.cholesky(0.5 * (Di + Di.T)) for Di in D]
        self.Cinv = [sla.solve_triangular(C, np.eye(len(C)), lower=True) for C in self.C]
        self.groups = {}
        for d in np.unique(model.dims):
            blk = np.flatnonzero(model.dims == d)
            idx = model.offsets[blk][:, None] + np.arange(d)[None, :]
            self.groups[int(d)] = (blk, idx, np.array([self.C[i] for i in blk]), np.array([self.Cinv[i] for i in blk]))

    def norms(self, V):
        """||v_i||_{D_i} = ||C_i^T v_i|| for all blocks; V is N or N x n_t."""
        vec = V.ndim == 1
        V2 = V[:, None] if vec else V
        out = np.zeros((self.model.n_blocks, V2.shape[1]))
        for d, (blk, idx, C, _) in self.groups.items():
            X = V2[idx]  # n x d x t
            out[blk] = np.linalg.norm(np.einsum("nki,nkt->nit", C, X), axis=1)
        return out[:, 0] if vec else out

    def dual_norms(self, G):
        """||g_i||_{D_i^-1} = ||C_i^-1 g_i|| for all blocks."""
        vec = G.ndim == 1
        G2 = G[:, None] if vec else G
        out = np.zeros((self.model.n_blocks, G2.shape[1]))
        for d, (blk, idx, _, Ci) in self.groups.items():
            X = G2[idx]
            out[blk] = np.linalg.norm(np.einsum("nik,nkt->nit", Ci, X), axis=1)
        return out[:, 0] if vec else out


def batched_opnorm(X):
    """Spectral norms of a stack of small matrices (n, a, b)."""
    if X.shape[0] == 0:
        return np.zeros(0)
    return np.linalg.svd(X, compute_uv=False)[:, 0]


# ------------------------------------------------------------------------------------------ Perron
def perron(A, symmetric=False, power_steps=200):
    """Spectral radius of a nonnegative matrix and a rigorous upper bound on it (Collatz-Wielandt:
    rho(A) <= max_i (A x)_i / x_i for any x > 0). Returns (rho_estimate, rho_upper)."""
    A = sp.csr_matrix(A)
    n = A.shape[0]
    if n == 0 or A.nnz == 0:
        return 0.0, 0.0
    if n <= 3000:
        Ad = A.toarray()
        if symmetric:
            w, V = np.linalg.eigh(Ad)
            k = int(np.argmax(w))
            rho, v = float(w[k]), V[:, k]
        else:
            w, V = np.linalg.eig(Ad)
            k = int(np.argmax(np.abs(w)))
            rho, v = float(abs(w[k])), np.real(V[:, k])
    else:
        try:
            if symmetric:
                w, V = spla.eigsh(A, k=1, which="LA", tol=1e-10, maxiter=20000)
            else:
                w, V = spla.eigs(A, k=1, which="LM", tol=1e-10, maxiter=20000)
            rho, v = float(abs(w[0])), np.real(V[:, 0])
        except spla.ArpackNoConvergence:
            rho, v = np.nan, np.ones(n)
    x = np.abs(v)
    x = np.maximum(x, 1e-12 * max(x.max(), 1e-300))

    def cw(x):
        y = A @ x
        return float(np.max(y / x))

    up = cw(x)
    if not np.isfinite(rho) or up >= 1.0 > rho or up > 1.001 * rho:
        # smooth the vector by shifted power steps (keeps positivity); each step's bound is valid
        s = max(rho if np.isfinite(rho) else 1.0, 1e-3)
        for _ in range(power_steps):
            x = A @ x + s * x
            x = x / x.max()
            x = np.maximum(x, 1e-300)
            up = min(up, cw(x))
            if np.isfinite(rho) and up <= 1.0001 * rho:
                break
        if not np.isfinite(rho):
            rho = up
    return rho, up


def solve_comparison(A, s, cw_upper):
    """c* = (I - A)^-1 s for rho(A) < 1 (certified by cw_upper < 1), plus the monotone iteration count
    to relative accuracy 1e-10 (Theorem 3.5). s: n or n x t. Returns (c, iterations) or (None, None)."""
    if not (cw_upper < 1.0):
        return None, None
    n = A.shape[0]
    M = (sp.eye(n, format="csc") - sp.csc_matrix(A))
    lu = spla.splu(M)
    c = lu.solve(np.asarray(s, float))
    if c.ndim == 1:
        c = c
    # iterative refinement (M is an M-matrix; its solve is backward stable)
    for _ in range(2):
        c = c + lu.solve(s - M @ c)
    # monotone iteration c <- s + A c from c = 0 (Theorem 3.5): count the steps to relative accuracy
    # 1e-10 (capped), and check that the iterates are nondecreasing and stay below c*
    cm = np.zeros_like(np.asarray(s, float))
    it = 0
    target = np.maximum(np.abs(c), 1e-300)
    monotone = True
    while it < MONOTONE_CAP:
        new = s + A @ cm
        monotone &= bool(np.all(new >= cm - 1e-12 * target)) and bool(np.all(new <= c + 1e-9 * target))
        cm = new
        it += 1
        if np.all(np.abs(c - cm) <= 1e-10 * target):
            break
    if not monotone:
        raise RuntimeError("monotone iteration of Theorem 3.5 is not monotone / exceeds c*")
    return c, (it if it < MONOTONE_CAP else -1)


# ------------------------------------------------------------------------------------------ C1
def c1_matrix(model, metric):
    """M_ij = ||D_i^-1/2 H_ij D_j^-1/2||_2 (i != j), M_ii = 0 (Theorem 3.1). Symmetric."""
    A = sp.triu(model.adj, k=1).tocoo()
    I, J = A.row, A.col
    H = model.H.tocsr()
    vals = np.zeros(len(I))
    for di in np.unique(model.dims[I]):
        for dj in np.unique(model.dims[J]):
            sel = np.flatnonzero((model.dims[I] == di) & (model.dims[J] == dj))
            if not len(sel):
                continue
            r = model.offsets[I[sel]][:, None, None] + np.arange(di)[None, :, None]
            c = model.offsets[J[sel]][:, None, None] + np.arange(dj)[None, None, :]
            r, c = np.broadcast_arrays(r, c)
            blocks = np.asarray(H[r.ravel(), c.ravel()]).reshape(len(sel), di, dj)
            Ci = np.array([metric.Cinv[i] for i in I[sel]])
            Cj = np.array([metric.Cinv[j] for j in J[sel]])
            X = np.einsum("nik,nkl,njl->nij", Ci, blocks, Cj)
            vals[sel] = batched_opnorm(X)
    n = model.n_blocks
    M = sp.csr_matrix((vals, (I, J)), shape=(n, n))
    return (M + M.T).tocsr()


def c1(model, metric, G):
    """Theorem 3.1 for residual columns G (N x t). Returns dict with M, rho, rho_upper, nu, E (or None)."""
    t0 = time.perf_counter()
    M = c1_matrix(model, metric)
    rho, up = perron(M, symmetric=True)
    nu = metric.dual_norms(G)
    E, it = solve_comparison(M, nu, up)
    return dict(M=M, rho=rho, rho_upper=up, nu=nu, E=E, iterations=it, time=time.perf_counter() - t0)


# ------------------------------------------------------------------------------------------ C2
class Backbone:
    """Pbb = H with every off-diagonal block that is not a GP-chain coupling set to zero (brief §5 C2):
    block-tridiagonal on the chain poses, block diagonal elsewhere. R_H = H - Pbb."""

    def __init__(self, model, use_chain=True):
        self.model = model
        n = model.n_blocks
        D = model.diag_blocks()
        pairs = model.gp_pairs() if use_chain else np.zeros((0, 2), int)
        chain = []
        if len(pairs):
            # chain order: follow the GP pairs
            nxt = {int(a): int(b) for a, b in pairs}
            prv = {int(b): int(a) for a, b in pairs}
            starts = [a for a in nxt if a not in prv]
            assert len(starts) == 1, "backbone must be one chain"
            k = starts[0]
            chain = [k]
            while k in nxt:
                k = nxt[k]
                chain.append(k)
        self.chain = np.array(chain, int)
        self.in_chain = np.zeros(n, bool)
        self.in_chain[self.chain] = True
        H = model.H.tocsr()
        # Pbb as a sparse matrix
        rows, cols, vals = [], [], []
        for i in range(n):
            s = model.block_slice(i)
            r = np.arange(s.start, s.stop)
            rows.append(np.repeat(r, len(r)))
            cols.append(np.tile(r, len(r)))
            vals.append(D[i].ravel())
        self.Po = []
        for k in range(len(chain) - 1):
            a, b = chain[k], chain[k + 1]
            blk = H[model.block_slice(a), model.block_slice(b)].toarray()
            self.Po.append(blk)
            ra = np.arange(model.offsets[a], model.offsets[a] + model.dims[a])
            rb = np.arange(model.offsets[b], model.offsets[b] + model.dims[b])
            rows += [np.repeat(ra, len(rb)), np.repeat(rb, len(ra))]
            cols += [np.tile(rb, len(ra)), np.tile(ra, len(rb))]
            vals += [blk.ravel(), blk.T.ravel()]
        self.P = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=H.shape)
        # the exact lo part of Pbb: H_lo restricted to Pbb's pattern (diagonal blocks and chain couplings)
        mask = self.P.copy()
        mask.data[:] = 1.0
        self.P_lo = model.H_lo.multiply(mask).tocsr()
        self.R = (H - self.P).tocsr()
        self.R.eliminate_zeros()
        if len(chain):
            from certify.chain import BlockTridiag
            self.bt = BlockTridiag([D[k] for k in chain], self.Po)
            self.chain_dofs = model.dof_index(self.chain)
        self.Dinv = [np.linalg.inv(Di) for Di in D]

    def solve(self, R, refine=2):
        """Pbb^-1 R for R (N x m) dense, with extended-precision iterative refinement."""
        from certify import precise
        Y = self._solve_once(R)
        if refine:
            if not hasattr(self, "P_ld"):
                self.P_ld = precise.to_ld(self.P, self.P_lo)
            Y = precise.refine(self._solve_once, self.P_ld, R, Y, strict=True)
        return Y

    def _solve_once(self, R):
        m = self.model
        out = np.zeros_like(R)
        for i in np.flatnonzero(~self.in_chain):
            s = m.block_slice(i)
            out[s] = self.Dinv[i] @ R[s]
        if len(self.chain):
            parts = [R[m.block_slice(k)] for k in self.chain]
            ys = self.bt.solve(parts)
            for k, y in zip(self.chain, ys):
                out[m.block_slice(k)] = y
        return out


def c2_matrix_bipartite(model, metric, bb, chunk=2000):
    """Mhat when R_H couples chain blocks only with non-chain blocks and non-chain blocks only with chain blocks
    (stereo BA and ChArUco: measurement factors join a pose and a landmark or marker). Then
      - non-chain row l (Pbb is D_l there): [Pbb^-1 R_H]_lj = D_l^-1 H_lj, so Mhat_lj = ||D_l^{-1/2} H_lj D_j^{-1/2}||;
      - chain row i: [Pbb^-1 R_H]_il = [Pchain^-1 H_{chain,l}]_i, one block-tridiagonal solve per column chunk.
    Returns None when the structure does not hold."""
    n = model.n_blocks
    ic = bb.in_chain
    R = bb.R.tocoo()
    blk_of = np.repeat(np.arange(n), model.dims)
    br, bc = blk_of[R.row], blk_of[R.col]
    if R.nnz == 0:
        return sp.csr_matrix((n, n))
    if np.any(ic[br] == ic[bc]) or not len(bb.chain):
        return None
    rows_out, cols_out, vals_out = [], [], []
    # non-chain rows: pairs (l, j) with l non-chain, j chain
    pairs = np.unique(np.stack([br, bc], axis=1)[~ic[br]], axis=0)
    H = model.H.tocsr()
    for dl in np.unique(model.dims[pairs[:, 0]]):
        for dj in np.unique(model.dims[pairs[:, 1]]):
            sel = pairs[(model.dims[pairs[:, 0]] == dl) & (model.dims[pairs[:, 1]] == dj)]
            if not len(sel):
                continue
            r = model.offsets[sel[:, 0]][:, None, None] + np.arange(dl)[None, :, None]
            c = model.offsets[sel[:, 1]][:, None, None] + np.arange(dj)[None, None, :]
            r, c = np.broadcast_arrays(r, c)
            Hb = np.asarray(H[r.ravel(), c.ravel()]).reshape(len(sel), dl, dj)
            Cl = np.array([metric.Cinv[i] for i in sel[:, 0]])
            Cj = np.array([metric.Cinv[j] for j in sel[:, 1]])
            rows_out.append(sel[:, 0])
            cols_out.append(sel[:, 1])
            vals_out.append(batched_opnorm(np.einsum("nik,nkl,njl->nij", Cl, Hb, Cj)))
    # chain rows: Y = Pchain^-1 R[chain dofs, non-chain dofs], per chunk of non-chain column blocks
    chain = bb.chain
    cd = bb.chain_dofs
    nonchain = np.flatnonzero(~ic)
    Rcsc = bb.R.tocsc()
    csz = [int(model.dims[k]) for k in chain]
    coff = np.concatenate([[0], np.cumsum(csz)])
    bt_plain = bb.bt
    for start in range(0, len(nonchain), chunk):
        lb = nonchain[start:start + chunk]
        ld = model.dof_index(lb)
        Rc = Rcsc[:, ld][cd].toarray()  # chain dofs x chunk dofs
        if not np.any(Rc):
            continue
        parts = [Rc[coff[k]:coff[k + 1]] for k in range(len(chain))]
        Y = np.concatenate(bt_plain._solve_list(parts), axis=0)  # double precision (comparison matrix entries)
        loff = np.concatenate([[0], np.cumsum(model.dims[lb])])
        for dl in np.unique(model.dims[lb]):
            ls = np.flatnonzero(model.dims[lb] == dl)
            Cl = np.array([metric.Cinv[b] for b in lb[ls]])  # (nl, dl, dl)
            for dc in np.unique(csz):
                ks = np.flatnonzero(np.array(csz) == dc)
                Ci = np.array([metric.C[chain[k]] for k in ks])  # (nk, dc, dc)
                # blocks Y[i, l]: (nk, nl, dc, dl)
                ridx = coff[ks][:, None] + np.arange(dc)[None, :]
                cidx = loff[ls][:, None] + np.arange(dl)[None, :]
                Yb = Y[ridx][:, :, cidx]  # (nk, dc, nl, dl)
                Yb = np.transpose(Yb, (0, 2, 1, 3))  # (nk, nl, dc, dl)
                Z = np.einsum("kai,klaj->klij", Ci, Yb)          # C_i^T Y
                Z = np.einsum("klij,lmj->klim", Z, Cl)          # Y C_l^-T
                nrm = np.linalg.svd(Z.reshape(-1, dc, dl), compute_uv=False)[:, 0].reshape(len(ks), len(ls))
                nz = nrm > 0
                kk, ll = np.nonzero(nz)
                rows_out.append(chain[ks][kk])
                cols_out.append(lb[ls][ll])
                vals_out.append(nrm[kk, ll])
    if not rows_out:
        return sp.csr_matrix((n, n))
    Mh = sp.csr_matrix((np.concatenate(vals_out), (np.concatenate(rows_out), np.concatenate(cols_out))), shape=(n, n))
    return Mh


def c2_matrix(model, metric, bb, chunk=512):
    """Mhat_ij = ||D_i^{1/2} [Pbb^-1 R_H]_ij D_j^{-1/2}||_2 (i != j), column block by column block (general
    structure; c2() uses c2_matrix_bipartite when it applies)."""
    n = model.n_blocks
    Rcsc = bb.R.tocsc()
    rows_out, cols_out, vals_out = [], [], []
    blocks = np.arange(n)
    for start in range(0, n, chunk):
        jb = blocks[start:start + chunk]
        dofs = model.dof_index(jb)
        Rc = Rcsc[:, dofs]
        if Rc.nnz == 0:
            continue
        X = bb.solve(Rc.toarray())  # N x |dofs|
        nzr = np.flatnonzero(np.abs(X).max(axis=1) > 0)
        if not len(nzr):
            continue
        rblk = np.unique(np.searchsorted(model.offsets, nzr, side="right") - 1)
        col_off = np.concatenate([[0], np.cumsum(model.dims[jb])[:-1]])
        for jj, j in enumerate(jb):
            dj = int(model.dims[j])
            Xj = X[:, col_off[jj]:col_off[jj] + dj]
            for di in np.unique(model.dims[rblk]):
                ib = rblk[model.dims[rblk] == di]
                ib = ib[ib != j]
                if not len(ib):
                    continue
                idx = model.offsets[ib][:, None] + np.arange(di)[None, :]
                Xi = Xj[idx]  # (n_i, di, dj)
                nz = np.abs(Xi).reshape(len(ib), -1).max(axis=1) > 0
                if not nz.any():
                    continue
                ib, Xi = ib[nz], Xi[nz]
                Ci = np.array([metric.C[i] for i in ib])
                Y = np.einsum("nki,nkl->nil", Ci, Xi) @ metric.Cinv[j].T
                rows_out.append(ib)
                cols_out.append(np.full(len(ib), j))
                vals_out.append(batched_opnorm(Y))
    if rows_out:
        Mh = sp.csr_matrix((np.concatenate(vals_out), (np.concatenate(rows_out), np.concatenate(cols_out))), shape=(n, n))
    else:
        Mh = sp.csr_matrix((n, n))
    return Mh


def c2(model, metric, G, bb=None):
    """Theorem 3.2 for residual columns G (N x t)."""
    t0 = time.perf_counter()
    bb = Backbone(model) if bb is None else bb
    Mh = c2_matrix_bipartite(model, metric, bb)
    if Mh is None:
        Mh = c2_matrix(model, metric, bb)
    rho, up = perron(Mh)
    nuh = metric.norms(bb.solve(G if G.ndim == 2 else G[:, None]))
    E, it = solve_comparison(Mh, nuh, up)
    return dict(M=Mh, rho=rho, rho_upper=up, nu=nuh, E=E, iterations=it, time=time.perf_counter() - t0, backbone=bb)


# ------------------------------------------------------------------------------------------ C3
def boundary_matrix(model, P, inP, loc_off):
    """B_P (brief §2.3, Definition 2.3) in patch-local dof coordinates: sum over factors touching P and its
    exterior of J_{f,P}^T Lambda_f J_{f,P}. Returns (rows, cols, vals) and the number of boundary factors."""
    fids = model.factors_touching(P)
    rows, cols, vals = [], [], []
    nb = 0
    if not len(fids):
        return rows, cols, vals, 0
    grp = model.factor_group[fids]
    for gi in np.unique(grp):
        f = model.factors[gi]
        loc = model.factor_local[fids[grp == gi]]
        V = f["vars"][loc]
        m = V.shape[1]
        ins = inP[V]
        bnd = ins.any(axis=1) & ~ins.all(axis=1)
        if not bnd.any():
            continue
        nb += int(bnd.sum())
        dims = [int(model.dims[V[0, a]]) for a in range(m)]
        cs = np.concatenate([[0], np.cumsum(dims)])
        Hf = f["Hf"][loc[bnd]].astype(precise.LD) + f["Hf_lo"][loc[bnd]].astype(precise.LD)
        Vb, insb = V[bnd], ins[bnd]
        for a in range(m):
            for c in range(m):
                sel = insb[:, a] & insb[:, c]
                if not sel.any():
                    continue
                blk = Hf[sel][:, cs[a]:cs[a + 1], cs[c]:cs[c + 1]]
                ra = loc_off[Vb[sel, a]][:, None, None] + np.arange(dims[a])[None, :, None]
                rc = loc_off[Vb[sel, c]][:, None, None] + np.arange(dims[c])[None, None, :]
                ra, rc = np.broadcast_arrays(ra, rc)
                rows.append(ra.ravel())
                cols.append(rc.ravel())
                vals.append(blk.ravel())
    return rows, cols, vals, nb


def patch_one(model, metric, i, K, G, want_cut=True, P=None, extended=True):
    """Everything C3 and the patch/cut covariance bounds need for block i and hop count K (or for the explicit
    patch P, a sorted block array containing i; Theorem 3.5 allows any patch per node)."""
    t0 = time.perf_counter()
    P = model.ball(i, K) if P is None else np.asarray(P, int)
    n = model.n_blocks
    inP = np.zeros(n, bool)
    inP[P] = True
    dP = model.dims[P]
    loc_off = np.zeros(n, int)
    loc_off[P] = np.concatenate([[0], np.cumsum(dP)[:-1]])
    dofs = model.dof_index(P)
    H = model.H
    Kpp, Kpp_lo = model.sub(dofs)
    elim = model.elim[P].copy()
    sysm = SPDSystem(Kpp, dP, elim, K_lo=Kpp_lo)
    if not sysm.ok:
        raise RuntimeError("H_PP not positive definite at block %d (K=%d)" % (i, K))
    d = int(model.dims[i])
    Ei = np.zeros((len(dofs), d))
    Ei[loc_off[i]:loc_off[i] + d] = np.eye(d)
    Y = sysm.solve(Ei, extended=extended)  # [H_PP^-1]_{:, i}
    inv_ii = 0.5 * (Y[loc_off[i]:loc_off[i] + d] + Y[loc_off[i]:loc_off[i] + d].T)
    Z = Y.T @ G[dofs]  # d x t   = [H_PP^-1 g_P]_i
    s = np.linalg.norm(metric.C[i].T @ Z, axis=0)
    t1 = time.perf_counter()

    # boundary and exterior neighbours
    Adj = model.adj
    rowsP = Adj[P]
    outside = ~inP[rowsP.indices]
    row_of = np.repeat(np.arange(len(P)), np.diff(rowsP.indptr))
    cnt = np.bincount(row_of, weights=outside.astype(float), minlength=len(P))
    bP = P[cnt > 0]
    if len(bP):
        ext = np.unique(Adj[bP].indices)
        ext = ext[~inP[ext]]
    else:
        ext = np.zeros(0, int)
    a_cols, a_vals = np.zeros(0, int), np.zeros(0)
    if len(ext):
        bdofs = model.dof_index(bP)
        edofs = model.dof_index(ext)
        Yb = Y[np.concatenate([np.arange(loc_off[b], loc_off[b] + model.dims[b]) for b in bP])]
        Hbe = H[bdofs][:, edofs]
        W = (Hbe.T @ Yb).T  # d x |edofs|: sum_b [H_PP^-1]_{ib} H_{bj}
        W = metric.C[i].T @ W
        e_off = np.concatenate([[0], np.cumsum(model.dims[ext])[:-1]])
        a_vals = np.zeros(len(ext))
        for dj in np.unique(model.dims[ext]):
            sel = np.flatnonzero(model.dims[ext] == dj)
            idx = e_off[sel][:, None] + np.arange(dj)[None, :]
            Wj = np.transpose(W[:, idx], (1, 0, 2))  # n x d x dj
            Cj = np.array([metric.Cinv[j] for j in ext[sel]])
            a_vals[sel] = batched_opnorm(np.einsum("nik,njk->nij", Wj, Cj))
        a_cols = ext
    t2 = time.perf_counter()

    cut_ii, certifiable, n_bf = None, None, 0
    if want_cut:
        r, c, v, n_bf = boundary_matrix(model, P, inP, loc_off)
        if r:
            # H_PP - B_P formed exactly: extended precision from the hi + lo parts, then split again
            LD = precise.LD
            B = sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=Kpp.shape, dtype=LD)
            B.sum_duplicates()
            Kc, Kc_lo = precise.split_ld(precise.join_ld(Kpp, Kpp_lo) - B)
        else:
            Kc, Kc_lo = Kpp, Kpp_lo
        sc = SPDSystem(Kc, dP, elim, K_lo=Kc_lo)
        certifiable = bool(sc.ok)
        if sc.ok:
            # Cholesky of a (numerically) singular cut matrix can succeed through rounding; its refined solve
            # then does not converge, and the patch is not locally certifiable (upper bound +inf).
            try:
                Yc = sc.solve(Ei)
                cut_ii = 0.5 * (Yc[loc_off[i]:loc_off[i] + d] + Yc[loc_off[i]:loc_off[i] + d].T)
            except precise.NotConverged:
                certifiable, cut_ii = False, None
    t3 = time.perf_counter()
    return dict(i=int(i), s=s, inv_ii=inv_ii, a_cols=a_cols, a_vals=a_vals, cut_ii=cut_ii, certifiable=certifiable,
                n_patch=len(P), n_patch_dofs=len(dofs), n_boundary=len(bP), n_exterior=len(ext), n_boundary_factors=n_bf,
                t_local=t1 - t0, t_coeff=t2 - t1, t_cut=t3 - t2)


_WORK = {}


def _worker(args):
    blocks, K, want_cut = args
    from threadpoolctl import threadpool_limits
    m, met, G = _WORK["model"], _WORK["metric"], _WORK["G"]
    with threadpool_limits(1):
        return [patch_one(m, met, int(i), K, G, want_cut) for i in blocks]


def patch_pass(model, metric, K, G, blocks=None, want_cut=True, workers=None, chunk=64):
    """patch_one for every block (or the given blocks), in parallel (fork)."""
    import multiprocessing as mp
    blocks = np.arange(model.n_blocks) if blocks is None else np.asarray(blocks, int)
    workers = workers or int(os.environ.get("CERTIFY_WORKERS", "1"))
    G = G if G.ndim == 2 else G[:, None]
    if workers <= 1 or len(blocks) < 2 * chunk:
        return [patch_one(model, metric, int(i), K, G, want_cut) for i in blocks]
    _WORK.update(model=model, metric=metric, G=G)
    parts = [(blocks[s:s + chunk], K, want_cut) for s in range(0, len(blocks), chunk)]
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        res = pool.map(_worker, parts)
    _WORK.clear()
    return [r for part in res for r in part]


def c3_from_patches(model, patches):
    """Assemble A (Definition 3.4) and s from patch_one results; return rho, bound, c* (Theorem 3.5)."""
    n = model.n_blocks
    rows = np.concatenate([np.full(len(p["a_cols"]), p["i"]) for p in patches]) if patches else np.zeros(0, int)
    cols = np.concatenate([p["a_cols"] for p in patches]) if patches else np.zeros(0, int)
    vals = np.concatenate([p["a_vals"] for p in patches]) if patches else np.zeros(0)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    s = np.zeros((n, patches[0]["s"].shape[0]))
    for p in patches:
        s[p["i"]] = p["s"]
    t0 = time.perf_counter()
    rho, up = perron(A)
    c, it = solve_comparison(A, s, up)
    return dict(A=A, s=s, rho=rho, rho_upper=up, E=c, iterations=it, t_global=time.perf_counter() - t0,
                row_sum_max=float(np.asarray(A.sum(axis=1)).max()) if A.nnz else 0.0)
