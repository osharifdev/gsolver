"""Marginal covariance bounds (PROOF_NOTE §2; brief §4).

Lower bounds, Lemma 2.1: Sigma_ii >= E_i^T V (V^T H V)^-1 V^T E_i for any V whose span contains E_i.
    trivial  V = E_i                            -> H_ii^-1
    patch    V = coordinates of P_i(K)           -> [(H_PP)^-1]_ii       (computed in certs.patch_pass)
    krylov   V = span{E_i, H E_i, ..., H^{l-1} E_i}  (block Lanczos with full reorthogonalization; the
             subspace lies in the coordinates of P_i(l-1), so H is restricted to that ball exactly)
Upper bound, Theorem 2.4: Sigma_ii <= [(H_PP - B_P)^-1]_ii when H_PP - B_P > 0 (certs.patch_pass).
Eigenvalues are reported in the D-metric: mu = eig(D_i^{1/2} X D_i^{1/2}) = eig(C_i^T X C_i).

Invariant tolerance: a matrix inequality X <= Y is accepted when
    lambda_min(D^{1/2} (Y - X) D^{1/2}) >= -1e-9 * max(lambda_max(D^{1/2} X D^{1/2}), lambda_max(D^{1/2} Y D^{1/2})).
"""

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

TOL = 1e-9


def psd_violation(C, X, Y):
    """Relative violation of X <= Y in the metric D = C C^T (<= 0 means valid, compare with TOL)."""
    A = C.T @ X @ C
    B = C.T @ Y @ C
    d = np.linalg.eigvalsh(0.5 * ((B - A) + (B - A).T))[0]
    scale = max(np.linalg.eigvalsh(0.5 * (A + A.T))[-1], np.linalg.eigvalsh(0.5 * (B + B.T))[-1], 1e-300)
    return float(-d / scale)


def mu_extremes(C, X):
    w = np.linalg.eigvalsh(0.5 * (C.T @ X @ C + (C.T @ X @ C).T))
    return float(w[0]), float(w[-1])


def krylov_lower_multi(model, i, ells):
    """Block-Krylov lower bounds (Lemma 2.1) on Sigma_ii for V spanning K_l(H, E_i) =
    span{E_i, H E_i, ..., H^{l-1} E_i}, for every l in `ells`; returns {l: (d x d bound, subspace dim)}.

    One basis is built for l_max on the ball of radius l_max - 1 around i (the subspace lies in those
    coordinates, so H is restricted to that ball exactly), by block Lanczos in the D-inner product
    (D = blkdiag(H_jj) = C C^T), working with hat coordinates w^ = C^T w in which D-orthonormal means
    orthonormal: W_1 = E_i C_i^-T (spans E_i exactly), W_{k+1} = the new directions of H W_k, found by two
    passes of reorthogonalization against the previous hat basis and an SVD (directions with singular value
    below 1e-13 of the norm of C^T H W_k are rounding noise and are dropped: the space is then invariant up
    to rounding). The bound for l uses the leading blocks W_{1..l} of this one basis, so the subspaces are
    nested exactly. T = W^T H W is formed and factorized once in extended precision (T = L L^T); since the
    leading c_l x c_l block of L factorizes T_l, the bound E_i^T W_l T_l^-1 W_l^T E_i = Y_l^T Y_l with Y_l the
    first c_l rows of Y = L^-1 W^T E_i, and bound(l') - bound(l) = sum of the extra rows' outer products is
    positive semidefinite by construction (Lemma 2.2 holds exactly as computed). Any basis of the same span
    gives the same bound; this one makes T as well conditioned as the block-Jacobi-scaled H."""
    from certify import precise
    ells = sorted(set(int(l) for l in ells))
    lmax = ells[-1]
    blocks = model.ball(i, lmax - 1)
    dofs = model.dof_index(blocks)
    Hs = model.H[dofs][:, dofs].tocsr()
    pos = np.searchsorted(dofs, model.dof_index([i]))
    d = int(model.dims[i])
    Dm = model.diag_blocks()
    Cs = [np.linalg.cholesky(Dm[b]) for b in blocks]
    C = sp.block_diag(Cs, format="csr")                 # D = C C^T on the ball
    CinvT = sp.block_diag([sla.solve_triangular(c, np.eye(len(c)), lower=True).T for c in Cs], format="csr")
    What = np.zeros((len(dofs), d))
    What[pos] = np.eye(d)                               # hat basis of W_1 = E_i C_i^-T
    hats = [What]
    ncols = [d]
    Wk = CinvT @ What
    for _ in range(lmax - 1):
        Zh = C.T @ (Hs @ Wk)
        ref = np.linalg.norm(Zh, 2)
        Hall = np.concatenate(hats, axis=1)
        for _ in range(2):
            Zh = Zh - Hall @ (Hall.T @ Zh)
        U, sv, _ = np.linalg.svd(Zh, full_matrices=False)
        keep = sv > 1e-13 * ref
        if not keep.any():
            break
        Uk = U[:, keep]
        for _ in range(2):
            Uk = Uk - Hall @ (Hall.T @ Uk)
        Uk, _ = np.linalg.qr(Uk)
        hats.append(Uk)
        ncols.append(ncols[-1] + Uk.shape[1])
        Wk = CinvT @ Uk
    Wh = np.concatenate(hats, axis=1)
    W = CinvT @ Wh
    Hl = sp.csr_matrix(Hs).astype(precise.LD) + model.H_lo[dofs][:, dofs].tocsr().astype(precise.LD)
    Wl = W.astype(precise.LD)
    T = Wl.T @ (Hl @ Wl)
    T = (T + T.T) / 2
    L = precise.chol_ld(T)
    Ei = W[pos].astype(precise.LD)                      # d x m: rows of W at block i's coordinates
    # Y = L^-1 Ei^T by forward substitution in extended precision
    B = Ei.T
    m = len(L)
    Y = np.zeros_like(B)
    for r in range(m):
        Y[r] = (B[r] - L[r, :r] @ Y[:r]) / L[r, r]
    out = {}
    for l in ells:
        c = ncols[min(l, len(ncols)) - 1]
        Bl = np.asarray(Y[:c].T @ Y[:c], dtype=float)
        out[l] = (0.5 * (Bl + Bl.T), int(c))
    return out


def krylov_lower(model, i, ell):
    """Single-l version of krylov_lower_multi: (bound, subspace dimension, ball dofs)."""
    r = krylov_lower_multi(model, i, [ell])[ell]
    return r[0], r[1], int(len(model.dof_index(model.ball(i, ell - 1))))


def trivial_lower(model, i):
    return np.linalg.inv(model.diag_blocks()[i])
