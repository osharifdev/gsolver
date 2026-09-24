"""Gate A validation of frozen exports (amendment A4).

For one frozen export directory:
  (i)  direct solve of H xbar = b (Jacobi-scaled factorization + iterative refinement), relative residual;
  (ii) the frozen GBP error curve err(t) = max_i ||z_i^(t) - xbar_i||_{D_i} over the dumped iterations,
       its floor, the iteration where the floor is reached, and the geometric rate;
plus symmetry, the extreme eigenvalues and condition numbers of H and of D^-1/2 H D^-1/2, and the
near-null dimension (eigenvalues below 1e-8 * lambda_max) required by A3.

Floor rule. Main rate rho: least-squares slope of log err(t) over the dumped t in the decreasing phase
(from the first t >= 10 with err(t) <= 0.1 err(0), or t = 10, to the first t with err(t) <= 10 F).
Tail rate rho_tail: the same fit over the last 20% of the run (t >= 0.8 T). The floor is reached iff
the curve has flattened, |log rho_tail| <= 0.1 |log rho|; then F = median of err over the tail and
t_floor = first dumped t after which err stays <= 2 F. If it has not flattened, the run ended before
the floor: F is reported as the last value (an upper bound on the floor) with floor_reached = false.
(A first version declared a floor whenever the tail stayed within 2x of its median; on the first
test export it labelled a still-decreasing curve a floor, so it was replaced before any Gate A result
was recorded.)

Single-edge variables (degree 1): G-solver never updates their outgoing message, so it acts as a prior
in the rest of the graph but is absent from the variable's own belief. GBP's belief of such a variable
is its factor's message computed from the neighbour's GBP cavity, which on a loopy graph depends on
GBP's approximate precisions: it is not x_bar_v and not any exact model quantity. The floor is therefore
computed over the other blocks ("non_single_edge"); single-edge blocks are reported separately as an
offset from x_bar ("single_edge_vs_xbar").
"""

import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from certify.io import Export, diag_blocks, informed_blocks

DENSE_MAX = 25000


# --------------------------------------------------------------------------------------- SO(3) x R^3
def so3_exp(r):
    """Rodrigues, batched (n,3) -> (n,3,3), as gsolver/maths/geometry3d.h expMapSO3."""
    r = np.atleast_2d(r)
    th = np.linalg.norm(r, axis=1)
    K = np.zeros((len(r), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -r[:, 2], r[:, 1], -r[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = r[:, 2], -r[:, 1], r[:, 0]
    small = th ** 2 < 1e-8
    R = np.repeat(np.eye(3)[None], len(r), 0) + K
    big = ~small
    if big.any():
        Kn = K[big] / th[big, None, None]
        s, c1 = np.sin(th[big]), 2 * np.sin(th[big] / 2) ** 2
        R[big] = np.eye(3)[None] + s[:, None, None] * Kn + c1[:, None, None] * Kn @ Kn
    return R


def so3_log(R):
    tr = np.trace(R, axis1=1, axis2=2)
    th = np.arccos(np.clip((tr - 1) / 2, -1, 1))
    v = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], axis=1)
    mag = np.where(th > 1e-7, th / (2 * np.sin(np.maximum(th, 1e-300))), 0.5 - (tr - 3) ** 2 / 12)
    return mag[:, None] * v


def poses_from_state(ex, z):
    """Pose-velocity blocks: (R, t, v) = T_lin [+] z (right perturbation, geometry3d.h conventions)."""
    blk = ex.blocks_of_type("SE3PoseVel")
    Z = z[ex.offsets[blk][:, None] + np.arange(12)]
    T = ex.pose_T_lin
    R = T[:, :3, :3] @ so3_exp(Z[:, 3:6])
    t = T[:, :3, 3] + np.einsum("nij,nj->ni", T[:, :3, :3], Z[:, 0:3])
    return blk, R, t, ex.pose_v_lin + Z[:, 6:12]


def state_from_poses(ex, blk, R, t, v, z_template):
    z = z_template.copy()
    T = ex.pose_T_lin
    Rl = T[:, :3, :3]
    z[ex.offsets[blk][:, None] + np.arange(0, 3)] = np.einsum("nji,nj->ni", Rl, t - T[:, :3, 3])
    z[ex.offsets[blk][:, None] + np.arange(3, 6)] = so3_log(np.einsum("nji,njk->nik", Rl, R))
    z[ex.offsets[blk][:, None] + np.arange(6, 12)] = v - ex.pose_v_lin
    return z


def align_to(ex, z, zref):
    """SE(3) (no scale) that best maps the pose positions of z onto those of zref, applied to poses and
    SE3Point landmarks of z (velocities are body-frame, unchanged). Returns the aligned state."""
    blk, R, t, v = poses_from_state(ex, z)
    _, _, tr, _ = poses_from_state(ex, zref)
    mu_s, mu_d = t.mean(0), tr.mean(0)
    U, _, Vt = np.linalg.svd((tr - mu_d).T @ (t - mu_s))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    Ra = U @ S @ Vt
    ta = mu_d - Ra @ mu_s
    za = state_from_poses(ex, blk, np.einsum("ij,njk->nik", Ra, R), t @ Ra.T + ta, v, z)
    pts = ex.blocks_of_type("SE3Point")
    if len(pts):
        idx = ex.offsets[pts][:, None] + np.arange(3)
        p = ex.point_p_lin + z[idx]
        za[idx] = p @ Ra.T + ta - ex.point_p_lin
    return za, Ra, ta


# ---------------------------------------------------------------------------------------- solvers
class Solver:
    """Factorization of H with Jacobi (block-diagonal square-root) scaling; dense below DENSE_MAX dof,
    otherwise Schur elimination of the (block-diagonal) SE3Point landmarks."""

    def __init__(self, ex, H):
        self.ex, self.H = ex, H.tocsr()
        self.N = H.shape[0]
        self.dg = np.sqrt(H.diagonal())
        if self.N <= DENSE_MAX:
            self.mode = "dense"
            B = (self.H.toarray() / self.dg[:, None]) / self.dg[None, :]
            self.B = B
            self.cf = sla.cho_factor(B, lower=True)
        else:
            self.mode = "schur"
            pts = ex.blocks_of_type("SE3Point")
            npd = int(ex.offsets[pts[0]]) if len(pts) else self.N
            assert np.all(ex.var_type[: len(ex.var_type) - len(pts)] != "SE3Point")
            self.npd = npd
            Hpp = self.H[:npd, :npd].toarray()
            self.Hpl = self.H[:npd, npd:].tocsr()
            Hll = self.H[npd:, npd:].tocsr()
            nl = (self.N - npd) // 3
            r = (np.arange(nl)[:, None, None] * 3 + np.arange(3)[None, :, None]).repeat(3, 2)
            c = (np.arange(nl)[:, None, None] * 3 + np.arange(3)[None, None, :]).repeat(3, 1)
            self.Lb = np.asarray(Hll[r.ravel(), c.ravel()]).reshape(nl, 3, 3)
            self.Lb_inv = np.linalg.inv(self.Lb)
            self.Hll_inv = sp.block_diag(list(self.Lb_inv), format="csr")
            S = Hpp - (self.Hpl @ self.Hll_inv @ self.Hpl.T).toarray()
            self.sdg = np.sqrt(np.diag(S))
            self.S = S
            self.cf = sla.cho_factor((S / self.sdg[:, None]) / self.sdg[None, :], lower=True)

    def _solve_once(self, rhs):
        if self.mode == "dense":
            return sla.cho_solve(self.cf, rhs / self.dg) / self.dg
        npd = self.npd
        bp, bl = rhs[:npd], rhs[npd:]
        y = bp - self.Hpl @ (self.Hll_inv @ bl)
        xp = sla.cho_solve(self.cf, y / self.sdg) / self.sdg
        xl = self.Hll_inv @ (bl - self.Hpl.T @ xp)
        return np.concatenate([xp, xl])

    def solve(self, rhs, refine=4):
        x = self._solve_once(rhs)
        hist = [np.linalg.norm(self.H @ x - rhs) / np.linalg.norm(rhs)]
        for _ in range(refine):
            x = x + self._solve_once(rhs - self.H @ x)
            hist.append(np.linalg.norm(self.H @ x - rhs) / np.linalg.norm(rhs))
        return x, hist

    def marginal_blocks(self, blocks):
        """Exact Sigma_ii for the given blocks (dense: via the scaled inverse; schur: landmark formula)."""
        ex = self.ex
        out = {}
        if self.mode == "dense":
            Binv = sla.cho_solve(self.cf, np.eye(self.N))
            for i in blocks:
                s = ex.block_slice(i)
                out[int(i)] = Binv[s, s] / np.outer(self.dg[s], self.dg[s])
            return out
        Sinv = sla.cho_solve(self.cf, np.eye(self.npd)) / np.outer(self.sdg, self.sdg)
        for i in blocks:
            s = ex.block_slice(i)
            if s.start < self.npd:
                out[int(i)] = Sinv[s, s]
            else:
                l = (s.start - self.npd) // 3
                w = (self.Lb_inv[l] @ self.Hpl[:, s.start - self.npd:s.stop - self.npd].T.toarray())  # 3 x npd
                out[int(i)] = self.Lb_inv[l] + w @ Sinv @ w.T
        return out


def lambda_max(H):
    return float(spla.eigsh(H, k=1, which="LA", return_eigenvectors=False, tol=1e-8)[0])


def smallest_eigs(solver, k=6):
    """Smallest eigenvalues of H via Lanczos on H^-1, applied with the solver's scaled factorization and
    one step of refinement (accurate far below eps * lambda_max, unlike a dense eigensolver on H)."""
    N = solver.N
    op = spla.LinearOperator((N, N), matvec=lambda v: solver.solve(np.asarray(v).ravel(), refine=1)[0], dtype=float)
    mu = spla.eigsh(op, k=k, which="LA", return_eigenvectors=False, tol=1e-10)
    return np.sort(1.0 / mu)


def count_below(H, ex, tau):
    """Exact number of eigenvalues of H below tau (Sylvester inertia of H - tau I). SE3Point blocks
    are eliminated first (Haynsworth additivity), so only the pose part is factorized densely."""
    H = H.tocsr()
    pts = ex.blocks_of_type("SE3Point")
    npd = int(ex.offsets[pts[0]]) if len(pts) else H.shape[0]
    if npd == H.shape[0] or npd <= 0:
        return int(np.sum(np.linalg.eigvalsh(H.toarray() - tau * np.eye(H.shape[0])) < 0))
    nl = (H.shape[0] - npd) // 3
    Hll = H[npd:, npd:]
    r = (np.arange(nl)[:, None, None] * 3 + np.arange(3)[None, :, None]).repeat(3, 2)
    c = (np.arange(nl)[:, None, None] * 3 + np.arange(3)[None, None, :]).repeat(3, 1)
    L = np.asarray(Hll[r.ravel(), c.ravel()]).reshape(nl, 3, 3) - tau * np.eye(3)[None]
    wl = np.linalg.eigvalsh(L)
    n_ll = int(np.sum(wl < 0))
    Linv = sp.block_diag(list(np.linalg.inv(L)), format="csr")
    Hpl = H[:npd, npd:]
    S = H[:npd, :npd].toarray() - tau * np.eye(npd) - (Hpl @ Linv @ Hpl.T).toarray()
    return n_ll + int(np.sum(np.linalg.eigvalsh(S) < 0))


def spectrum(H, dense_max=DENSE_MAX, k=12):
    """(lambda_min, lambda_max, smallest eigenvalues) of H; dense for small systems, ARPACK otherwise.
    Dense values are only accurate to ~1e-16 lambda_max in absolute terms (used for scaled matrices)."""
    N = H.shape[0]
    if N <= dense_max:
        w = np.linalg.eigvalsh(H.toarray())
        return w[0], w[-1], w[:k]
    lmax = lambda_max(H)
    sig = -1e-12 * lmax
    lu = spla.splu((H - sig * sp.eye(N)).tocsc())
    op = spla.LinearOperator((N, N), matvec=lu.solve, dtype=float)
    mu = spla.eigsh(op, k=k, which="LM", return_eigenvectors=False, tol=1e-8)
    w = np.sort(1.0 / mu + sig)
    return w[0], lmax, w


def _restrict_solve(solver, ex, inf_dofs, v):
    """H^-1 v on the informed dofs (uninformed placeholders are decoupled, so this is exact)."""
    full = np.zeros(ex.n_dof)
    full[inf_dofs] = v
    return solver.solve(full, refine=1)[0][inf_dofs]


class _SubExport:
    """Block layout of H restricted to the kept dofs (for count_below): a block keeps the number of its
    dofs that survive; blocks with none are dropped."""

    def __init__(self, ex, kept_dofs):
        kept = np.zeros(ex.n_dof, bool)
        kept[kept_dofs] = True
        n = np.array([kept[ex.block_slice(i)].sum() for i in range(ex.n_blocks)])
        keep = n > 0
        self.var_type = ex.var_type[keep]
        self.dims = n[keep]
        self.offsets = np.concatenate([[0], np.cumsum(self.dims)[:-1]])
        assert np.all(self.dims[self.var_type == "SE3Point"] == 3)

    def blocks_of_type(self, t):
        return np.flatnonzero(self.var_type == t)


# ---------------------------------------------------------------------------------------- curves
def block_errors(ex, D, e):
    """||e_i||_{D_i} for all blocks."""
    out = np.zeros(ex.n_blocks)
    for d, (blk, Dd) in D.items():
        E = e[ex.offsets[blk][:, None] + np.arange(d)]
        out[blk] = np.sqrt(np.maximum(np.einsum("ni,nij,nj->n", E, Dd, E), 0))
    return out


def floor_and_rate(t, err):
    t, err = np.asarray(t, float), np.asarray(err, float)
    T = t[-1]
    le = np.log(np.maximum(err, 1e-300))
    tail = t >= 0.8 * T
    rho_tail = float(np.exp(np.polyfit(t[tail], le[tail], 1)[0])) if tail.sum() > 2 else np.nan
    F_tail = float(np.median(err[tail]))
    a = np.flatnonzero((t >= 10) & (err <= 0.1 * err[0]))
    ia = a[0] if len(a) else int(np.flatnonzero(t >= 10)[0])
    b = np.flatnonzero(err <= 10 * F_tail)
    ib = int(b[0]) if len(b) else len(t) - 1
    if ib - ia < 10:
        ia, ib = int(np.flatnonzero(t >= 10)[0]), len(t) - 1
    rho = float(np.exp(np.polyfit(t[ia:ib + 1], le[ia:ib + 1], 1)[0])) if ib > ia else np.nan
    reached = bool(np.isfinite(rho) and np.isfinite(rho_tail) and rho < 1
                   and abs(np.log(rho_tail)) <= 0.1 * abs(np.log(rho)))
    if reached:
        F = F_tail
        above = np.flatnonzero(err > 2 * F)
        t_f = float(t[above[-1] + 1]) if len(above) and above[-1] + 1 < len(t) else float(t[0])
    else:
        F, t_f = float(err[-1]), np.nan
    return dict(floor=F, floor_reached=reached, t_floor=t_f, rate=rho, rate_window=[float(t[ia]), float(t[ib])],
                rate_tail=rho_tail, frac_increasing_steps=float(np.mean(np.diff(err) > 0)),
                err_first=float(err[0]), err_min=float(err.min()), err_last=float(err[-1]), T=float(T))


def validate(path, align=False, out_csv=None):
    ex = Export(path)
    res = dict(export=str(path), dataset=ex.meta.get("dataset"), n_dof=ex.n_dof, n_blocks=ex.n_blocks,
               factor_counts=ex.meta.get("factor_counts"), frozen_regularizers=ex.meta.get("frozen_regularizers"))
    H, b = ex.assemble(("meas", "gauge", "single_edge"))
    res["symmetry_rel"] = float(abs(H - H.T).max() / abs(H).max())
    informed = informed_blocks(H, ex)
    res["n_uninformed_blocks"] = int((~informed).sum())
    res["uninformed_blocks"] = [str(v) for v in ex.var_id[~informed]]
    # Uninformed blocks have exactly zero rows/columns of H. They are not part of the model: give them an
    # identity diagonal and zero right-hand side (keeps them decoupled, so the solve on the other blocks
    # is exact), and leave them out of the spectrum and of the error curves.
    # zero-information dofs inside otherwise informed blocks (e.g. the velocity of the only pose before
    # any GP factor exists) are handled the same way; they make H singular and are reported
    unf_dofs = np.flatnonzero((H.diagonal() == 0) | np.repeat(~informed, ex.dims))
    res["n_uninformed_dofs"] = int(len(unf_dofs))
    res["uninformed_dofs_in_informed_blocks"] = int(np.sum((H.diagonal() == 0) & np.repeat(informed, ex.dims)))
    if len(unf_dofs):
        assert abs(H.tocsr()[unf_dofs]).sum() == 0
        H = (H + sp.csr_matrix((np.ones(len(unf_dofs)), (unf_dofs, unf_dofs)), shape=H.shape)).tocsr()
        b = b.copy()
        b[unf_dofs] = 0.0
    D = diag_blocks(H, ex)
    solver = Solver(ex, H)
    xbar, hist = solver.solve(b)
    res["solve_mode"] = solver.mode
    res["rel_residual_history"] = hist
    res["rel_residual"] = hist[-1]
    inf_dofs = np.setdiff1d(np.arange(ex.n_dof), unf_dofs)
    Hs = sp.csr_matrix(H.tocsr()[inf_dofs][:, inf_dofs]) if len(unf_dofs) else H
    lmax = lambda_max(Hs)
    small = smallest_eigs(solver, k=6)  # uninformed placeholders have eigenvalue 1; they never are the smallest here
    res["lambda_max"] = lmax
    res["lambda_smallest"] = [float(v) for v in small]
    res["lambda_min"] = float(small[0])
    res["cond_H"] = float(lmax / small[0])
    dg = np.sqrt(Hs.diagonal())
    B = (sp.diags(1 / dg) @ Hs @ sp.diags(1 / dg)).tocsr()
    Binv = spla.LinearOperator(B.shape, dtype=float,
                               matvec=lambda v: dg * _restrict_solve(solver, ex, inf_dofs, dg * np.asarray(v).ravel()))
    mu_max = spla.eigsh(Binv, k=1, which="LA", return_eigenvectors=False, tol=1e-8)[0]
    res["cond_scaled"] = float(lambda_max(B) * mu_max)
    ex_inf = _SubExport(ex, inf_dofs)
    res["near_null_threshold"] = 1e-8 * lmax
    res["near_null_dim"] = count_below(Hs, ex_inf, 1e-8 * lmax)

    se_blocks = np.array([i for i, _, _ in ex.single_edge], dtype=int)
    res["n_single_edge"] = int(len(se_blocks))

    Z = ex.states()
    if Z is None:
        # unfrozen export (charts at the beliefs, state z = 0): model checks only; x_bar is the
        # Gauss-Newton step of G-solver's Jacobians from the current beliefs
        eb = block_errors(ex, D, -xbar)
        keep = informed & ~np.isin(np.arange(ex.n_blocks), se_blocks)
        res["gn_step_D_max"] = float(eb[keep].max())
        res["gn_step_D_median"] = float(np.median(eb[keep]))
        return res, xbar
    iters = np.load(ex.path / "states_iters.npy") if (ex.path / "states_iters.npy").exists() else np.arange(Z.shape[0])
    mask_se = np.zeros(ex.n_blocks, bool)
    mask_se[se_blocks] = True
    mask_se |= ~informed  # uninformed blocks are not part of the model
    # blocks whose weakest direction carries less information than 100x G-solver's regularizer (1e-6)
    # cannot be resolved by the solver (certify/FREEZE_MODE.md §4); reported as a separate category
    mineig = np.full(ex.n_blocks, np.inf)
    for d, (blk, Dd) in D.items():
        mineig[blk] = np.linalg.eigvalsh(Dd)[:, 0]
    res["n_blocks_min_D_eig_below_1e-4"] = int(np.sum(informed & ~mask_se & (mineig < 1e-4)))
    cat_poses = (ex.var_type == "SE3PoseVel") & ~mask_se
    cat_resolvable = ~mask_se & (mineig >= 1e-4)
    curves = {k: [] for k in ("all_vs_xbar", "non_single_edge", "single_edge_vs_xbar", "poses", "resolvable")}
    if align:
        curves["aligned_non_single_edge"] = []
    for k in range(Z.shape[0]):
        z = np.asarray(Z[k])
        eb = block_errors(ex, D, z - xbar)
        curves["all_vs_xbar"].append(eb.max())
        curves["non_single_edge"].append(eb[~mask_se].max())
        curves["poses"].append(eb[cat_poses].max())
        curves["resolvable"].append(eb[cat_resolvable].max())
        if mask_se.any():
            curves["single_edge_vs_xbar"].append(eb[mask_se].max())
        if align:
            za, _, _ = align_to(ex, z, xbar)
            curves["aligned_non_single_edge"].append(block_errors(ex, D, za - xbar)[~mask_se].max())
    curves = {k: np.array(v) for k, v in curves.items() if len(v)}
    res["curves"] = {k: floor_and_rate(iters, v) for k, v in curves.items()}
    last = np.asarray(Z[-1])
    eb_last = block_errors(ex, D, last - xbar)
    worst = int(np.argmax(np.where(mask_se, -1, eb_last)))
    res["worst_block_last"] = dict(index=worst, id=str(ex.var_id[worst]), type=str(ex.var_type[worst]),
                                   pose_index=int(ex.pose_index[worst]), err=float(eb_last[worst]),
                                   degree=int(ex.degree[worst]))
    if out_csv:
        cols = ["iteration"] + list(curves)
        with open(out_csv, "w") as f:
            f.write(",".join(cols) + "\n")
            for j, t in enumerate(iters):
                f.write(",".join([str(int(t))] + ["%.6e" % curves[c][j] for c in curves]) + "\n")
    return res, xbar


if __name__ == "__main__":
    p = Path(sys.argv[1])
    align = "--align" in sys.argv
    res, xbar = validate(p, align=align, out_csv=p / "gateA_curves.csv")
    np.save(p / "xbar.npy", xbar)
    (p / "gateA.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "rel_residual_history"}, indent=1, default=float))


def a3_analysis(path, thresholds=(1e-10, 1e-8, 1e-6)):
    """Amendment A3: spectrum of the stereo model with and without the explicit gauge prior (and without
    the frozen single-edge priors): lambda_max, lambda_min, and the exact number of eigenvalues below
    1e-8 * lambda_max and below fixed thresholds (Sylvester inertia). Uninformed dofs are left out."""
    ex = Export(path)
    out = {}
    for name, parts in (("with_prior", ("meas", "gauge", "single_edge")),
                        ("without_prior", ("meas", "single_edge")),
                        ("measurements_only", ("meas",))):
        H, _ = ex.assemble(parts)
        d = H.diagonal()
        keep = np.flatnonzero(d != 0)
        Hs = sp.csr_matrix(H.tocsr()[keep][:, keep])
        sub = _SubExport(ex, keep)
        lmax = lambda_max(Hs)
        r = dict(n_dof=int(len(keep)), lambda_max=lmax, near_null_threshold=1e-8 * lmax,
                 near_null_dim=count_below(Hs, sub, 1e-8 * lmax))
        for t in thresholds:
            r["n_below_%g" % t] = count_below(Hs, sub, t)
        r["n_negative"] = count_below(Hs, sub, 0.0)
        try:
            sub_ex = _Relabel(ex, keep)
            solver = Solver(sub_ex, Hs)
            r["cholesky_ok"] = True
            r["lambda_smallest"] = [float(v) for v in smallest_eigs(solver, k=8)]
        except Exception as err:  # not positive definite
            r["cholesky_ok"] = False
            r["cholesky_error"] = str(err)[:200]
        out[name] = r
    return out


class _Relabel:
    """Minimal Export-like view of the kept dofs, for Solver (dense or landmark-Schur)."""

    def __init__(self, ex, keep_dofs):
        s = _SubExport(ex, keep_dofs)
        self.var_type, self.dims, self.offsets = s.var_type, s.dims, s.offsets
        self.n_blocks, self.n_dof = len(self.dims), int(self.dims.sum())

    def blocks_of_type(self, t):
        return np.flatnonzero(self.var_type == t)

    def block_slice(self, i):
        return slice(int(self.offsets[i]), int(self.offsets[i] + self.dims[i]))


def far_landmark_attribution(path, taus=(1e-6, 1e-4, 1e-3), t_index=-1):
    """Diagnostic for the frozen floor: re-solve the model with every SE3Point block whose weakest
    D-direction carries less than tau information held at GBP's final value, and report the floor of the
    remaining (non-single-edge) blocks against that solution. If the floor collapses, it is explained by
    those near-null directions (conditioning), which the solver cannot resolve below its 1e-6 regularizers."""
    ex = Export(path)
    H, b = ex.assemble()
    informed = informed_blocks(H, ex)
    unf = np.flatnonzero((H.diagonal() == 0) | np.repeat(~informed, ex.dims))
    H = (H + sp.csr_matrix((np.ones(len(unf)), (unf, unf)), shape=H.shape)).tocsr()
    b = b.copy()
    b[unf] = 0.0
    D = diag_blocks(H, ex)
    zT = np.asarray(ex.states()[t_index])
    se = np.zeros(ex.n_blocks, bool)
    se[[i for i, _, _ in ex.single_edge]] = True
    se |= ~informed
    mineig = np.full(ex.n_blocks, np.inf)
    for d, (blk, Dd) in D.items():
        mineig[blk] = np.linalg.eigvalsh(Dd)[:, 0]
    poses = ex.var_type == "SE3PoseVel"
    out = []
    for tau in taus:
        far = np.flatnonzero((ex.var_type == "SE3Point") & informed & (mineig < tau))
        fd = np.concatenate([np.arange(ex.offsets[i], ex.offsets[i] + 3) for i in far]) if len(far) else np.zeros(0, int)
        rest = np.setdiff1d(np.arange(ex.n_dof), fd)
        Hr = H[rest][:, rest]
        rhs = b[rest] - H[rest][:, fd] @ zT[fd]
        sub = _Relabel(ex, rest)
        xr = zT.copy()
        xr[rest] = Solver(sub, sp.csr_matrix(Hr)).solve(rhs)[0]
        e = block_errors(ex, D, zT - xr)
        keep = ~se & ~np.isin(np.arange(ex.n_blocks), far)
        out.append(dict(tau=tau, n_fixed=int(len(far)), pose_floor=float(e[poses].max()),
                        other_blocks_floor=float(e[keep].max())))
    return out
