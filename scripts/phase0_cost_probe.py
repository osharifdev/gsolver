"""Phase 0 cost probe: time the Python-side oracle and patch computations that
Phase C/D will need, on a structural proxy of the KITTI 06 stereo-BA system.

The proxy has exactly the factor graph of multirobot/06.g2o (477 pose-velocity
blocks of dim 12, 45,555 observed 3-dof landmarks, 157,941 stereo factors, 476
GP factors, one anchor) but random well-conditioned Jacobians, because the real
H only exists after the Phase A export. Timings, not values, are the output.
"""

import sys
import time
from collections import deque

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

G2O = sys.argv[1] if len(sys.argv) > 1 else "examples/data/datasets/stereo_visual/multirobot/06.g2o"
rng = np.random.default_rng(0)

pose_t, obs = [], []
for line in open(G2O):
    s = line.split()
    if not s:
        continue
    if s[0] == "VERTEX_SE3":
        pose_t.append(float(s[1]))
    elif s[0] == "EDGE_STEREO":
        obs.append((float(s[1]), int(s[2])))
pose_t = sorted(pose_t)
pidx = {t: i for i, t in enumerate(pose_t)}
lm_ids = sorted({l for _, l in obs})
lidx = {l: i for i, l in enumerate(lm_ids)}
npose, nlm = len(pose_t), len(lm_ids)
P = np.array([pidx[t] for t, _ in obs])
L = np.array([lidx[l] for _, l in obs])
dims = np.r_[np.full(npose, 12), np.full(nlm, 3)]
off = np.r_[0, np.cumsum(dims)]
N = off[-1]
print(f"proxy: {npose} poses, {nlm} landmarks, {len(obs)} stereo factors, N={N} dof")

t0 = time.time()
rows, cols, vals = [], [], []


def add_factor(var_offsets, var_dims, J):
    Hf = J.T @ J
    idx = np.concatenate([np.arange(o, o + d) for o, d in zip(var_offsets, var_dims)])
    rows.append(np.repeat(idx, len(idx)))
    cols.append(np.tile(idx, len(idx)))
    vals.append(Hf.ravel())


# stereo factors, vectorised: J is 3x15 per factor
Js = rng.standard_normal((len(obs), 3, 15))
Hs = np.einsum("fki,fkj->fij", Js, Js)
idx = np.concatenate([off[P][:, None] + np.arange(12), off[npose + L][:, None] + np.arange(3)], axis=1)
rows.append(np.repeat(idx, 15, axis=1).ravel())
cols.append(np.tile(idx, (1, 15)).ravel())
vals.append(Hs.ravel())
for k in range(npose - 1):  # GP factors, 12x24
    add_factor([off[k], off[k + 1]], [12, 12], rng.standard_normal((12, 24)))
add_factor([off[0]], [12], np.eye(12))  # anchor
H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
H.sum_duplicates()
print(f"assemble H: {time.time()-t0:.1f} s, nnz={H.nnz/1e6:.1f}M ({H.data.nbytes/1e6+H.indices.nbytes/1e6:.0f} MB CSR)")

# --- oracle solve by landmark elimination (reduced camera system) ---
b = rng.standard_normal(N)
t0 = time.time()
npdof = off[npose]
Hpp = H[:npdof, :npdof].toarray()
Hpl = H[:npdof, npdof:].tocsr()
Hll = H[npdof:, npdof:].tocsr()
blocks = np.stack([Hll[3 * i:3 * i + 3, 3 * i:3 * i + 3].toarray() for i in range(nlm)])
Hll_inv = sp.block_diag(np.linalg.inv(blocks), format="csr")
S = Hpp - (Hpl @ Hll_inv @ Hpl.T).toarray()
cS = sla.cho_factor(S)
bp, bl = b[:npdof], b[npdof:]
xp = sla.cho_solve(cS, bp - Hpl @ (Hll_inv @ bl))
xl = Hll_inv @ (bl - Hpl.T @ xp)
x = np.r_[xp, xl]
t_solve = time.time() - t0
print(f"oracle solve (Schur on landmarks): {t_solve:.1f} s, residual {np.linalg.norm(H@x-b)/np.linalg.norm(b):.1e}")

# --- oracle marginals for all blocks ---
t0 = time.time()
Sig_pp = sla.cho_solve(cS, np.eye(npdof))
# landmark marginal: Hll^-1 + Hll^-1 Hlp Sig_pp Hpl Hll^-1, per landmark (3x3), using sparse row blocks
W = (Hll_inv @ Hpl.T).tocsr()  # nlm*3 x npdof, sparse
out = np.empty((nlm, 3, 3))
for s0 in range(0, nlm, 4096):  # chunks keep the dense W rows at ~0.5 GB
    s1 = min(nlm, s0 + 4096)
    Wd = W[3 * s0:3 * s1].toarray()
    WSd = Wd @ Sig_pp
    out[s0:s1] = np.einsum("lij,lkj->lik", WSd.reshape(s1 - s0, 3, npdof), Wd.reshape(s1 - s0, 3, npdof))
    out[s0:s1] += np.linalg.inv(blocks[s0:s1])
t_marg = time.time() - t0
print(f"oracle marginals Sigma_ii for all {npose+nlm} blocks: {t_marg:.1f} s")

# --- patch solves: BFS on the variable graph ---
t0 = time.time()
blk_of = np.repeat(np.arange(npose + nlm), dims)
A = sp.csr_matrix((np.ones(H.nnz), (blk_of[H.nonzero()[0]], blk_of[H.nonzero()[1]])), shape=(npose + nlm,) * 2)
A.data[:] = 1
A = A.tocsr()
print(f"block adjacency: {time.time()-t0:.1f} s")


def patch(i, K):
    seen = {i}
    frontier = [i]
    for _ in range(K):
        nxt = set()
        for u in frontier:
            nxt.update(A.indices[A.indptr[u]:A.indptr[u + 1]])
        nxt -= seen
        seen |= nxt
        frontier = list(nxt)
    return np.array(sorted(seen))


def patch_solve(i, K):
    nodes = patch(i, K)
    rr = np.concatenate([np.arange(off[n], off[n + 1]) for n in nodes])
    HPP = H[rr][:, rr].tocsc()
    lu = spla.splu(HPP)
    k0 = np.searchsorted(rr, off[i])
    Ei = np.zeros((len(rr), dims[i]))
    Ei[k0:k0 + dims[i]] = np.eye(dims[i])
    lu.solve(b[rr])
    lu.solve(Ei)  # i-th block column of H_PP^-1 (needed for a_ij)
    return len(nodes), len(rr)


for kind, cand in (("pose", np.arange(npose)), ("landmark", npose + np.arange(nlm))):
    for K in (1, 2, 3):
        sample = rng.choice(cand, size=min(20, len(cand)), replace=False)
        ts, sizes = [], []
        for i in sample:
            t0 = time.time()
            nb, nd = patch_solve(i, K)
            ts.append(time.time() - t0)
            sizes.append(nd)
        print(f"{kind:8s} K={K}: patch dof median {int(np.median(sizes)):7d} max {max(sizes):7d};"
              f" time/patch median {np.median(ts)*1e3:8.1f} ms  max {max(ts)*1e3:8.1f} ms")
