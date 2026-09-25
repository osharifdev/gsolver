"""SPD solves for global and local (patch) systems.

SPDSystem factorizes a symmetric matrix K given in local dof coordinates. An optional independent set of
small blocks (landmarks: no factor joins two of them, so K restricted to them is block diagonal) is
eliminated first; the remaining (pose) part is a dense Jacobi-scaled Cholesky factorization of the Schur
complement. `ok` is False when K is not positive definite (a block or the Schur complement fails
Cholesky); that is the "Cholesky succeeds" test of brief §2.3 and Definition 2.5.
"""

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from certify import precise


class SPDSystem:
    def __init__(self, K, blk_sizes, elim_mask=None, refine=2, K_lo=None):
        """K (hi part, factorized) and optionally K_lo (exact lo part, used in the refinement residuals)."""
        K = sp.csr_matrix(K)
        self.K = K
        self.K_lo = K_lo
        self.n = K.shape[0]
        self.refine = refine
        blk_sizes = np.asarray(blk_sizes, int)
        offs = np.concatenate([[0], np.cumsum(blk_sizes)[:-1]])
        self.blk_sizes, self.blk_offs = blk_sizes, offs
        elim_mask = np.zeros(len(blk_sizes), bool) if elim_mask is None else np.asarray(elim_mask, bool)
        self.elim_mask = elim_mask
        self.ok = True
        dof_elim = np.repeat(elim_mask, blk_sizes)
        self.L = np.flatnonzero(dof_elim)
        self.A = np.flatnonzero(~dof_elim)
        # position of each dof in L / A
        self.nl_blocks = int(elim_mask.sum())
        try:
            if len(self.L):
                self._eliminate(K, blk_sizes, elim_mask)
            else:
                self._dense(K.toarray())
        except np.linalg.LinAlgError:
            self.ok = False

    # ------------------------------------------------------------------ factorization
    def _dense(self, S):
        S = 0.5 * (S + S.T)
        d = np.diag(S).copy()
        if np.any(d <= 0):
            raise np.linalg.LinAlgError("nonpositive diagonal")
        s = np.sqrt(d)
        self.s = s
        self.cf = sla.cho_factor(S / s[:, None] / s[None, :], lower=True, check_finite=False)
        if not np.all(np.isfinite(self.cf[0])) or np.any(np.diag(self.cf[0]) <= 0):
            raise np.linalg.LinAlgError("cholesky failed")
        self.S = S

    def _eliminate(self, K, blk_sizes, elim_mask):
        sizes = blk_sizes[elim_mask]
        if not np.all(sizes == sizes[0]):
            raise ValueError("eliminated blocks must have equal size")
        d = int(sizes[0])
        nl = len(sizes)
        L, A = self.L, self.A
        KLL = K[L][:, L].tocoo()
        r, c = KLL.row, KLL.col
        if np.any(r // d != c // d):
            raise ValueError("eliminated blocks are coupled")
        blocks = np.zeros((nl, d, d))
        np.add.at(blocks, (r // d, r % d, c % d), KLL.data)
        blocks = 0.5 * (blocks + np.transpose(blocks, (0, 2, 1)))
        Lc = np.linalg.cholesky(blocks)  # raises LinAlgError if any block is not PD
        Linv = np.linalg.inv(Lc)
        self.Binv = np.einsum("nki,nkj->nij", Linv, Linv)  # blocks of K_LL^-1
        self.KLLinv = sp.block_diag(list(self.Binv), format="csr") if nl else sp.csr_matrix((0, 0))
        self.KAL = K[A][:, L].tocsr()
        KAA = K[A][:, A].toarray()
        if len(A):
            S = KAA - (self.KAL @ self.KLLinv @ self.KAL.T).toarray()
            self._dense(S)
        else:
            self.S = np.zeros((0, 0))
            self.s = np.zeros(0)
            self.cf = None

    # ------------------------------------------------------------------ solves
    def _solve_S(self, R):
        if R.shape[0] == 0:
            return R
        return sla.cho_solve(self.cf, R / self.s[:, None], check_finite=False) / self.s[:, None]

    def _solve_once(self, R):
        if len(self.L) == 0:
            return self._solve_S(R)
        RA, RL = R[self.A], R[self.L]
        yA = self._solve_S(RA - self.KAL @ (self.KLLinv @ RL))
        yL = self.KLLinv @ (RL - self.KAL.T @ yA)
        Y = np.empty_like(R)
        Y[self.A] = yA
        Y[self.L] = yL
        return Y

    def solve(self, R, refine=None, extended=True, strict=True, R_lo=None):
        """K^-1 R (R: vector or n x k). With extended=True (default) the solution is refined with
        extended-precision residuals until it has converged (certify/precise.py; NotConverged is raised if
        it does not, i.e. K is numerically singular). extended=False: `refine` double-precision steps."""
        assert self.ok
        vec = R.ndim == 1
        R2 = R[:, None] if vec else R
        Y = self._solve_once(R2)
        if extended:
            if not hasattr(self, "K_ld"):
                self.K_ld = precise.to_ld(self.K, self.K_lo)
            Rl = None if R_lo is None else (R_lo[:, None] if R_lo.ndim == 1 else R_lo)
            Y = precise.refine(self._solve_once, self.K_ld, R2, Y, strict=strict, R_lo=Rl)
        else:
            for _ in range(self.refine if refine is None else refine):
                Y = Y + self._solve_once(R2 - self.K @ Y)
        return Y[:, 0] if vec else Y

    def inv_diag_blocks(self, blocks):
        """[K^-1]_ii (dense d_i x d_i) for the given local blocks, by solving with the unit columns."""
        out = []
        for i in blocks:
            o, d = int(self.blk_offs[i]), int(self.blk_sizes[i])
            E = np.zeros((self.n, d))
            E[o:o + d] = np.eye(d)
            out.append(self.solve(E)[o:o + d])
        return out


class GlobalMarginals:
    """Exact marginal covariance blocks Sigma_ii of an SPDSystem with landmarks eliminated:
    kept blocks from the dense inverse of the Schur complement, eliminated blocks by
    Sigma_ll = K_ll^-1 + W S^-1 W^T, W = K_ll^-1 K_la (sparse in the observing poses)."""

    def __init__(self, sysm):
        self.sys = sysm
        if sysm.cf is not None:
            Sinv = sla.cho_solve(sysm.cf, np.eye(len(sysm.A)), check_finite=False)
            self.Sinv = Sinv / sysm.s[:, None] / sysm.s[None, :]
        else:
            self.Sinv = np.zeros((0, 0))
        self.posA = -np.ones(sysm.n, int)
        self.posA[sysm.A] = np.arange(len(sysm.A))
        self.posL = -np.ones(sysm.n, int)
        self.posL[sysm.L] = np.arange(len(sysm.L))
        if len(sysm.L):
            self.KLA = sysm.KAL.T.tocsr()

    def block(self, o, d):
        s = self.sys
        if len(s.L) == 0:
            return self.Sinv[o:o + d, o:o + d]
        if self.posA[o] >= 0:
            a = self.posA[o:o + d]
            return self.Sinv[np.ix_(a, a)]
        l = self.posL[o:o + d]
        bi = l[0] // d
        Binv = s.Binv[bi]
        rows = self.KLA[l]  # d x |A| sparse
        cols = np.unique(rows.indices)
        if len(cols) == 0:
            return Binv
        W = Binv @ rows[:, cols].toarray()
        return Binv + W @ self.Sinv[np.ix_(cols, cols)] @ W.T


def chol_lower(M):
    return np.linalg.cholesky(0.5 * (M + M.T))


def op_norm_scaled(Ci, X, Cj):
    """||D_i^{1/2} X D_j^{-1/2}||_2 with D = C C^T (Cholesky): = ||C_i^T X C_j^{-T}||_2."""
    Y = Ci.T @ X
    Y = sla.solve_triangular(Cj, Y.T, lower=True).T  # Y C_j^{-T}
    return float(np.linalg.norm(Y, 2))


def op_norm_invscaled(Ci, X, Cj):
    """||D_i^{-1/2} X D_j^{-1/2}||_2 = ||C_i^-1 X C_j^-T||_2."""
    Y = sla.solve_triangular(Ci, X, lower=True)
    Y = sla.solve_triangular(Cj, Y.T, lower=True).T
    return float(np.linalg.norm(Y, 2))


def eig_scaled(C, M):
    """Eigenvalues of D^{1/2} M D^{1/2} (= those of C^T M C), ascending."""
    return np.linalg.eigvalsh(C.T @ (0.5 * (M + M.T)) @ C)


def gen_eig_max(D, L):
    """lambda_max(D^{1/2} L^-1 D^{1/2}) = largest generalized eigenvalue of D v = lambda L v."""
    return float(sla.eigh(0.5 * (D + D.T), 0.5 * (L + L.T), eigvals_only=True)[-1])
