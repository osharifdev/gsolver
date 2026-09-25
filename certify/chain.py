"""Chain lemmas (PROOF_NOTE §1; brief §3) for a block-tridiagonal SPD matrix P given by its diagonal blocks
Pd[k] = P_kk and superdiagonal blocks Po[k] = P_{k,k+1} (so P_{k+1,k} = Po[k]^T). Blocks may have
different sizes (supernodes).

Errata 2026-09-25 (i), fact F4: on a GBP chain G_k = (H_kk - Lambda^f_kk) + message_{k+1->k}, where
Lambda^f_kk is the forward GP factor's diagonal contribution; Lemma 1.4's seed P_mm^- is H_mm - Lambda^f_mm.
"""

import numpy as np
import scipy.linalg as sla


def _sym(M):
    return 0.5 * (M + M.T)


def _solve_spd(A, B):
    return sla.cho_solve(sla.cho_factor(_sym(A), lower=True, check_finite=False), B, check_finite=False)


# ------------------------------------------------------------------------------ Definition 1.1
def backward_schur(Pd, Po):
    """G_n = P_nn, G_k = P_kk - P_{k,k+1} G_{k+1}^-1 P_{k+1,k}  (Definition 1.1)."""
    n = len(Pd)
    G = [None] * n
    G[n - 1] = _sym(Pd[n - 1])
    for k in range(n - 2, -1, -1):
        G[k] = _sym(Pd[k] - Po[k] @ _solve_spd(G[k + 1], Po[k].T))
    return G


def forward_schur(Pd, Po):
    """F_1 = P_11, F_k = P_kk - P_{k,k-1} F_{k-1}^-1 P_{k-1,k}  (Definition 1.1, mirror)."""
    n = len(Pd)
    F = [None] * n
    F[0] = _sym(Pd[0])
    for k in range(1, n):
        F[k] = _sym(Pd[k] - Po[k - 1].T @ _solve_spd(F[k - 1], Po[k - 1]))
    return F


# ------------------------------------------------------------------------------ Lemma 1.2
def transfer_backward(G, Po):
    """Psi_k = -G_{k+1}^-1 P_{k+1,k}, k = 0..n-2 (Lemma 1.2)."""
    return [-_solve_spd(G[k + 1], Po[k].T) for k in range(len(Po))]


def transfer_forward(F, Po):
    """Phi_k = -F_{k-1}^-1 P_{k-1,k}, k = 1..n-1 (Lemma 1.2, i > j); Phi[0] is None."""
    return [None] + [-_solve_spd(F[k - 1], Po[k - 1]) for k in range(1, len(Po) + 1)]


def diag_marginals(Pd, Po, G=None, F=None):
    """Sigma_kk = (F_k + G_k - P_kk)^-1 for all k."""
    G = backward_schur(Pd, Po) if G is None else G
    F = forward_schur(Pd, Po) if F is None else F
    return [np.linalg.inv(_sym(F[k] + G[k] - Pd[k])) for k in range(len(Pd))]


def cov_product(Sigma_ii, Psi, i, j):
    """Sigma_ij for i < j by the product formula Sigma_ii Psi_i^T ... Psi_{j-1}^T (Lemma 1.2)."""
    X = Sigma_ii
    for k in range(i, j):
        X = X @ Psi[k].T
    return X


def cov_product_left(Sigma_ii, Phi, i, j):
    """Sigma_ij for i > j: Sigma_ii Phi_i^T Phi_{i-1}^T ... Phi_{j+1}^T (mirror of Lemma 1.2)."""
    X = Sigma_ii
    for k in range(i, j, -1):
        X = X @ Phi[k].T
    return X


# ------------------------------------------------------------------------------ Lemmas 1.3-1.4
def truncated_backward(Pd, Po, m, seed, k0=0):
    """Lower bounds L_k <= G_k for k0 <= k <= m from the seed L_m = seed <= G_m (Lemma 1.4) by the
    recursion L_k = P_kk - P_{k,k+1} L_{k+1}^-1 P_{k+1,k} (Lemma 1.3). Returns {k: L_k}."""
    L = {m: _sym(seed)}
    for k in range(m - 1, k0 - 1, -1):
        L[k] = _sym(Pd[k] - Po[k] @ _solve_spd(L[k + 1], Po[k].T))
    return L


def truncated_forward(Pd, Po, m, seed, k1=None):
    """Mirror of truncated_backward: lower bounds on the forward complements F_k for m <= k <= k1."""
    k1 = len(Pd) - 1 if k1 is None else k1
    L = {m: _sym(seed)}
    for k in range(m + 1, k1 + 1):
        L[k] = _sym(Pd[k] - Po[k - 1].T @ _solve_spd(L[k - 1], Po[k - 1]))
    return L


# ------------------------------------------------------------------------------ Definition 1.5
def normalized_coupling(Dk1, Pk1k, Dk):
    """||D_{k+1}^{-1/2} P_{k+1,k} D_k^{-1/2}||_2."""
    C1, C0 = np.linalg.cholesky(_sym(Dk1)), np.linalg.cholesky(_sym(Dk))
    Y = sla.solve_triangular(C1, Pk1k, lower=True)
    Y = sla.solve_triangular(C0, Y.T, lower=True).T
    return float(np.linalg.norm(Y, 2))


def survival(Dk1, Lk1):
    """lambda_max(D_{k+1}^{1/2} L_{k+1}^-1 D_{k+1}^{1/2}) (>= 1 when L <= D)."""
    return float(sla.eigh(_sym(Dk1), _sym(Lk1), eigvals_only=True)[-1])


def psibar(Dk1, Lk1, Pk1k, Dk):
    """Certified decay factor of link (k, k+1) (Definition 1.5); returns (psibar, survival, coupling)."""
    s = survival(Dk1, Lk1)
    c = normalized_coupling(Dk1, Pk1k, Dk)
    return s * c, s, c


# ------------------------------------------------------------------------------ Lemma 1.6
def scaled_norm(Di, X, Dj):
    """||D_i^{1/2} X D_j^{1/2}||_2."""
    Ci, Cj = np.linalg.cholesky(_sym(Di)), np.linalg.cholesky(_sym(Dj))
    return float(np.linalg.norm(Ci.T @ X @ Cj, 2))


def lam_max_scaled(Di, S):
    """lambda_max(D_i^{1/2} S D_i^{1/2})."""
    Ci = np.linalg.cholesky(_sym(Di))
    return float(np.linalg.eigvalsh(_sym(Ci.T @ S @ Ci))[-1])


# ------------------------------------------------------------------------------ block-tridiagonal solves
class BlockTridiag:
    """Factorization of a block-tridiagonal SPD matrix (dense blocks) by backward Schur complements, with
    solves and selected blocks of the inverse (Lemma 1.2). solve() refines with extended-precision
    residuals (certify/precise.py) unless refine=0."""

    def __init__(self, Pd, Po, refine=2, Pd_lo=None, Po_lo=None):
        self.Pd, self.Po = [_sym(P) for P in Pd], Po
        self.Pd_lo, self.Po_lo = Pd_lo, Po_lo  # exact lo parts (used only in the refinement residuals)
        self.n = len(Pd)
        self.refine = refine
        self.G = backward_schur(self.Pd, Po)
        self.cf = [sla.cho_factor(G, lower=True, check_finite=False) for G in self.G]
        for c in self.cf:
            if not np.all(np.isfinite(c[0])) or np.any(np.diag(c[0]) <= 0):
                raise np.linalg.LinAlgError("block-tridiagonal matrix is not positive definite")
        self.sizes = [len(P) for P in self.Pd]
        self.offs = np.concatenate([[0], np.cumsum(self.sizes)[:-1]])

    def _matrix_ld(self):
        if not hasattr(self, "_M_ld"):
            import scipy.sparse as sp
            from certify import precise
            n = self.n
            rows = [[None] * n for _ in range(n)]
            for k in range(n):
                rows[k][k] = sp.csr_matrix(self.Pd[k])
                if k + 1 < n:
                    rows[k][k + 1] = sp.csr_matrix(self.Po[k])
                    rows[k + 1][k] = sp.csr_matrix(self.Po[k].T)
            lo = None
            if self.Pd_lo is not None:
                lrows = [[None] * n for _ in range(n)]
                for k in range(n):
                    lrows[k][k] = sp.csr_matrix(self.Pd_lo[k])
                    if k + 1 < n:
                        lrows[k][k + 1] = sp.csr_matrix(self.Po_lo[k])
                        lrows[k + 1][k] = sp.csr_matrix(self.Po_lo[k].T)
                lo = sp.bmat(lrows, format="csr")
            self._M_ld = precise.to_ld(sp.bmat(rows, format="csr"), lo)
        return self._M_ld

    def solve(self, R):
        """R: list of per-block right-hand sides (d_k x m). Returns the list of solution blocks."""
        Y = self._solve_list(R)
        if not self.refine:
            return Y
        from certify import precise
        Rf = np.concatenate(R, axis=0)
        vec = Rf.ndim == 1
        Rf2 = Rf[:, None] if vec else Rf
        Yf = np.concatenate(Y, axis=0)
        Yf = Yf[:, None] if vec else Yf

        def once(r):
            parts = [r[o:o + d] for o, d in zip(self.offs, self.sizes)]
            return np.concatenate(self._solve_list(parts), axis=0)

        Yf = precise.refine(once, self._matrix_ld(), Rf2, Yf, strict=True)
        Yf = Yf[:, 0] if vec else Yf
        return [Yf[o:o + d] for o, d in zip(self.offs, self.sizes)]

    def _solve_list(self, R):
        """R: list of per-block right-hand sides (d_k or d_k x m). Block LDL^T with the backward
        complements: forward sweep from the end, then back substitution."""
        n = self.n
        y = [None] * n
        # eliminate from the last block: G_n = P_nn; w_k = r_k - P_{k,k+1} G_{k+1}^-1 w_{k+1}
        w = [None] * n
        w[n - 1] = R[n - 1]
        for k in range(n - 2, -1, -1):
            w[k] = R[k] - self.Po[k] @ sla.cho_solve(self.cf[k + 1], w[k + 1], check_finite=False)
        y[0] = sla.cho_solve(self.cf[0], w[0], check_finite=False)
        for k in range(1, n):
            y[k] = sla.cho_solve(self.cf[k], w[k] - self.Po[k - 1].T @ y[k - 1], check_finite=False)
        return y

    def inv_block_column(self, j):
        """All blocks [P^-1]_{kj}, k = 0..n-1 (solve with the unit block column j)."""
        R = [np.zeros((self.Pd[k].shape[0], self.Pd[j].shape[0])) for k in range(self.n)]
        R[j] = np.eye(self.Pd[j].shape[0])
        return self.solve(R)
