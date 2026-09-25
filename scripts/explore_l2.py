"""Exploration: an l2-aggregated variant of the recursive patch certificate (Theorem 3.5).

C3 bounds the leak of patch P_i by the triangle inequality over exterior blocks:
    ||e_i|| <= s_i + sum_j a_ij u_j,        a_ij = ||X_ij||_2,  X_ij = D_i^1/2 sum_b [H_PP^-1]_ib H_bj D_j^-1/2.
Grouping all exterior blocks of row i and using Cauchy-Schwarz instead:
    ||e_i|| <= s_i + gamma_i ||u_{N(i)}||_2,    gamma_i = ||[X_ij]_{j in N(i)}||_2  (one spectral norm).
F(u)_i = gamma_i ||u_{N(i)}||_2 is monotone, positively homogeneous and subadditive. If some c > 0 has
F(c) <= r c with r < 1, then the iteration c <- s + F(c) from 0 converges to c* and u <= c* (Collatz-
Wielandt argument: with v = (u - c*)_+, v <= F(v), and max_i v_i / c_i = t gives v <= t r c, so t = 0).
This script estimates r (nonlinear power iteration) for K = 1, 2 and compares it with rho(A) of C3, row
sums included. Mixed variant: group the exterior blocks by the boundary node they couple through.
"""

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from certify import certs, phaseC  # noqa: E402
from certify import model as mdl  # noqa: E402
from certify.linalg import SPDSystem  # noqa: E402


def rows_for_block(model, met, i, K):
    """(ext blocks, gamma_i (all exterior), per-boundary groups [(gamma_ib, ext blocks of b)], l1 row sum)."""
    P = model.ball(i, K)
    n = model.n_blocks
    inP = np.zeros(n, bool)
    inP[P] = True
    dP = model.dims[P]
    loc_off = np.zeros(n, int)
    loc_off[P] = np.concatenate([[0], np.cumsum(dP)[:-1]])
    dofs = model.dof_index(P)
    Kpp, Kpp_lo = model.sub(dofs)
    sysm = SPDSystem(Kpp, dP, model.elim[P].copy(), K_lo=Kpp_lo)
    d = int(model.dims[i])
    Ei = np.zeros((len(dofs), d))
    Ei[loc_off[i]:loc_off[i] + d] = np.eye(d)
    Y = sysm.solve(Ei, extended=False, refine=1)
    Adj = model.adj
    rowsP = Adj[P]
    outside = ~inP[rowsP.indices]
    row_of = np.repeat(np.arange(len(P)), np.diff(rowsP.indptr))
    cnt = np.bincount(row_of, weights=outside.astype(float), minlength=len(P))
    bP = P[cnt > 0]
    ext = np.unique(Adj[bP].indices)
    ext = ext[~inP[ext]]
    if not len(ext):
        return ext, 0.0, [], 0.0
    H = model.H
    edofs = model.dof_index(ext)
    e_off = np.concatenate([[0], np.cumsum(model.dims[ext])])
    Cinv_blk = sp.block_diag([met.Cinv[j].T for j in ext], format="csr")
    Ci = met.C[i]
    groups = []
    Wall = np.zeros((d, len(edofs)))
    for b in bP:
        yb = Y[loc_off[b]:loc_off[b] + model.dims[b]]
        Hbe = H[model.dof_index([b])][:, edofs]
        Wb = (Ci.T @ yb.T) @ Hbe.toarray()          # d x |edofs|, only b's exterior neighbours nonzero
        Wb = (Cinv_blk.T @ Wb.T).T                  # right-multiply by blockdiag(C_j^-T)
        Wall += Wb
        nzc = np.flatnonzero(np.abs(Wb).max(axis=0) > 0)
        if len(nzc):
            jb = np.unique(np.searchsorted(e_off, nzc, side="right") - 1)
            groups.append((float(np.linalg.norm(Wb, 2)), ext[jb]))
    gamma = float(np.linalg.norm(Wall, 2))
    l1 = 0.0
    for k, j in enumerate(ext):
        l1 += float(np.linalg.norm(Wall[:, e_off[k]:e_off[k + 1]], 2))
    return ext, gamma, groups, l1


_W = {}


def _worker(args):
    blocks, K = args
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        return [(int(i),) + rows_for_block(_W["m"], _W["met"], int(i), K) for i in blocks]


def nonlinear_radius(n, rows, mode, iters=300):
    """Estimate r = max_i F(c)_i / c_i at the fixed direction of the normalized power iteration."""
    c = np.ones(n)
    r_hist = []
    for _ in range(iters):
        F = np.zeros(n)
        for i, (ext, gamma, groups, l1) in rows.items():
            if mode == "all":
                F[i] = gamma * np.linalg.norm(c[ext]) if len(ext) else 0.0
            else:
                F[i] = sum(g * np.linalg.norm(c[e]) for g, e in groups)
        r = float(np.max(F / c))
        r_hist.append(r)
        c = F + 1e-3 * r * c
        c = c / c.max()
        c = np.maximum(c, 1e-12)
    return r_hist[-1], min(r_hist)


def main(name, K, workers):
    path, cfg = phaseC.DATASETS[name]
    model, info = mdl.from_export(path, p_L=cfg["p_L"])
    met = certs.Metric(model)
    import multiprocessing as mp
    _W.update(m=model, met=met)
    blocks = np.arange(model.n_blocks)
    parts = [(blocks[s:s + 32], K) for s in range(0, len(blocks), 32)]
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_worker, parts)
    rows = {r[0]: r[1:] for part in res for r in part}
    n = model.n_blocks
    poses = model.kinds == "pose"
    l1 = np.array([rows[i][3] for i in range(n)])
    gam = np.array([rows[i][1] for i in range(n)])
    next_ = np.array([len(rows[i][0]) for i in range(n)])
    r_all = nonlinear_radius(n, rows, "all")
    r_grp = nonlinear_radius(n, rows, "group")
    out = dict(dataset=name, K=K, l1_row_sum_poses_median=float(np.median(l1[poses])), l1_row_sum_landmarks_median=float(np.median(l1[~poses])),
               gamma_sqrtN_poses_median=float(np.median((gam * np.sqrt(next_))[poses])),
               gamma_poses_median=float(np.median(gam[poses])), n_ext_poses_median=float(np.median(next_[poses])),
               r_l2_all=r_all, r_l2_grouped=r_grp)
    print(json.dumps(out, indent=1), flush=True)
    d = phaseC.OUT / "explore"
    d.mkdir(parents=True, exist_ok=True)
    (d / ("l2_%s_K%d.json" % (name, K))).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 8)
