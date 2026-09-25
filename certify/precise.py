"""Extended-precision residuals for the invariant checks and the certificate solves.

The invariants are checked to 1e-9 relative, and some certificates are exact (C2 on a pure chain, C3 with
a patch equal to the graph), so both sides of a check must be computed to well below 1e-9 even when H is
ill-conditioned (the pure WNOA chain has cond(H) ~ 5e10). Every solve is therefore followed by iterative
refinement whose residual is computed in 80-bit extended precision (numpy.longdouble, eps ~ 1.1e-19):
with a backward-stable solver this gives forward errors of order eps_double * (small) whenever
cond(H) * eps_double << 1 (mixed-precision iterative refinement).
"""

import numpy as np
import scipy.sparse as sp

LD = np.longdouble


def to_ld(A, A_lo=None):
    """Matrix object for accurate residuals of A (+ A_lo) (double-double rows; see residual())."""
    A = A if sp.issparse(A) else sp.csr_matrix(np.asarray(A))
    if A_lo is not None and not sp.issparse(A_lo):
        A_lo = sp.csr_matrix(np.asarray(A_lo))
    return DDMatrix(A, A_lo)


def residual(A_ld, x, b, b_lo=None):
    """(b + b_lo - (A_hi + A_lo) x) with every row computed as an accurately rounded double-double dot
    product (Ogita-Rump-Oishi Dot2 with error-free TwoProduct/TwoSum), returned as float64. A_ld is a
    DDMatrix (to_ld() builds one); the result is as accurate as if computed in twice the working precision
    and then rounded, so iterative refinement converges to full double accuracy whenever cond * eps < 1."""
    if isinstance(A_ld, DDMatrix):
        return A_ld.residual(np.asarray(x, float), np.asarray(b, float), b_lo)
    r = b.astype(LD) - A_ld @ np.asarray(x).astype(LD)
    return np.asarray(r, dtype=float)


def split_ld(A_ld):
    """Exact split of an extended-precision array or CSR matrix into double hi + lo parts (hi + lo == A)."""
    if sp.issparse(A_ld):
        A_ld = sp.csr_matrix(A_ld)
        hi = A_ld.data.astype(float)
        lo = (A_ld.data - hi.astype(LD)).astype(float)
        return (sp.csr_matrix((hi, A_ld.indices.copy(), A_ld.indptr.copy()), shape=A_ld.shape),
                sp.csr_matrix((lo, A_ld.indices.copy(), A_ld.indptr.copy()), shape=A_ld.shape))
    A_ld = np.asarray(A_ld)
    hi = A_ld.astype(float)
    return hi, (A_ld - hi.astype(LD)).astype(float)


def join_ld(hi, lo):
    """hi + lo in extended precision (sparse or dense)."""
    if sp.issparse(hi):
        return sp.csr_matrix(hi).astype(LD) + (sp.csr_matrix(lo).astype(LD) if lo is not None else 0)
    return np.asarray(hi).astype(LD) + (np.asarray(lo).astype(LD) if lo is not None else 0)


_SPLIT = 134217729.0  # 2^27 + 1 (Dekker)


def _split(a):
    c = _SPLIT * a
    hi = c - (c - a)
    return hi, a - hi


def two_prod(a, b):
    x = a * b
    a1, a2 = _split(a)
    b1, b2 = _split(b)
    y = a2 * b2 - (((x - a1 * b1) - a2 * b1) - a1 * b2)
    return x, y


def two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


class DDMatrix:
    """Sparse matrix A = A_hi + A_lo for double-double residuals: rows of A_hi bucketed by length into padded
    (ELL) arrays so the sequential Dot2 recurrence is vectorized over the rows of a bucket; A_lo (|A_lo| <=
    eps |A_hi|) enters through one ordinary product, whose rounding error is O(eps^2 |A| |x|)."""

    def __init__(self, A, A_lo=None):
        self.A_lo = sp.csr_matrix(A_lo) if A_lo is not None else None
        A = sp.csr_matrix(A)
        A.sort_indices()
        self.shape = A.shape
        n = A.shape[0]
        lens = np.diff(A.indptr)
        self.buckets = []
        edges = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 1 << 62]
        for lo, hi in zip(edges[:-1], edges[1:]):
            rows = np.flatnonzero((lens > lo) & (lens <= hi)) if lo > 0 else np.flatnonzero(lens == 0)
            if not len(rows):
                continue
            L = max(int(lens[rows].max()), 1)
            data = np.zeros((len(rows), L))
            cols = np.zeros((len(rows), L), dtype=np.int64)
            for k in range(L):
                has = lens[rows] > k
                idx = A.indptr[rows[has]] + k
                data[has, k] = A.data[idx]
                cols[has, k] = A.indices[idx]
            self.buckets.append((rows, data, cols))

    def residual(self, X, B, B_lo=None):
        vec = X.ndim == 1
        X2 = X[:, None] if vec else X
        B2 = B[:, None] if B.ndim == 1 else B
        low = np.zeros((self.shape[0], X2.shape[1]))
        if B_lo is not None:
            low = low + (B_lo[:, None] if B_lo.ndim == 1 else B_lo)
        if self.A_lo is not None:
            low = low - self.A_lo @ X2
        out = np.zeros((self.shape[0], X2.shape[1]))
        for rows, data, cols in self.buckets:
            # terms: b_k (times 1) and -A_kj x_j, summed by Dot2; the small terms start the error accumulator
            p = B2[rows].copy()
            s = low[rows].copy()
            for k in range(data.shape[1]):
                a = -data[:, k][:, None]
                h, r = two_prod(a, X2[cols[:, k]])
                p, q = two_sum(p, h)
                s = s + (q + r)
            out[rows] = p + s
        return out[:, 0] if vec else out


class NotConverged(ArithmeticError):
    pass


REFINE_TOL = 2.0 ** -48   # ~3.6e-15: corrections below this fraction of the solution (column-wise) = converged
REFINE_MAX = 40


def refine(solve, A_ld, R, X, steps=None, strict=True, R_lo=None):
    """Mixed-precision iterative refinement X <- X + solve(R - A X), residuals in extended precision, until
    every column's correction is below REFINE_TOL of the column (at most REFINE_MAX steps). It converges
    whenever cond(A) * eps_double < 1 for the solver's backward error; if it does not converge the system is
    numerically singular (or the solver unusable) and NotConverged is raised when strict."""
    Xv = X if X.ndim == 2 else X[:, None]
    Rv = R if R.ndim == 2 else R[:, None]
    Rl = None if R_lo is None else (R_lo if R_lo.ndim == 2 else R_lo[:, None])
    for k in range(REFINE_MAX):
        dX = solve(residual(A_ld, Xv, Rv, Rl))
        Xv = Xv + dX
        num = np.abs(dX).max(axis=0)
        den = np.maximum(np.abs(Xv).max(axis=0), 1e-300)
        if np.all(num <= REFINE_TOL * den) and np.all(np.isfinite(Xv)):
            break
    else:
        if strict:
            raise NotConverged("iterative refinement did not converge (max correction %.2e of the solution)"
                               % float(np.max(num / den)))
    return Xv if X.ndim == 2 else Xv[:, 0]


def chol_ld(T):
    """Cholesky factor (lower) of a small SPD matrix in extended precision."""
    T = np.asarray(T, dtype=LD)
    n = len(T)
    L = np.zeros_like(T)
    for j in range(n):
        s = T[j, j] - np.dot(L[j, :j], L[j, :j])
        if not s > 0:
            raise np.linalg.LinAlgError("not positive definite (extended precision)")
        L[j, j] = np.sqrt(s)
        L[j + 1:, j] = (T[j + 1:, j] - L[j + 1:, :j] @ L[j, :j]) / L[j, j]
    return L


def chol_solve_ld(L, B):
    """Solve (L L^T) X = B in extended precision (forward and back substitution)."""
    B = np.asarray(B, dtype=LD)
    vec = B.ndim == 1
    B = B[:, None] if vec else B
    n = len(L)
    Y = np.zeros_like(B)
    for i in range(n):
        Y[i] = (B[i] - L[i, :i] @ Y[:i]) / L[i, i]
    X = np.zeros_like(B)
    for i in range(n - 1, -1, -1):
        X[i] = (Y[i] - L[i + 1:, i] @ X[i + 1:]) / L[i, i]
    return X[:, 0] if vec else X
