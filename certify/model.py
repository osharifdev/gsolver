"""The certified linear model (PROOF_NOTE §0.1) with its factor list, for exports and synthetic problems.

    H = sum_f J_f^T Lambda_f J_f + G,     b = sum_f J_f^T Lambda_f r_f + b_G.

Every term, including the gauge/anchor terms G and the landmark priors of decision D2(b), is kept as a
factor (J_f, Lambda_f, r_f) on one or more blocks, so the boundary-cut matrices B_P of Theorem 2.4 and the
anchored subgraphs of Lemma 4.3 can be formed from the list. `kind` labels each factor group; groups with
kind in GAUGE_KINDS are the anchor terms of property P5.

Models built from exports (decisions D2(a)-(c) of the Phase A review, applied in Python; the solver is
untouched):
  (a) every variable with exactly one edge (G-solver never updates its outgoing message, so the frozen
      message acts as a prior N(tau, Lambda^-1) on it) is eliminated exactly, together with that prior,
      by the Schur complement of its block. The result is one factor on its observing variable:
          J = J_p,  Lambda = Lambda_f - Lambda_f J_v H_vv^-1 J_v^T Lambda_f,  r = r_f - J_v tau,
      with H_vv = J_v^T Lambda_f J_v + Lambda_prior (kind "se_unary"). This is exact for all other blocks.
      Variables whose every factor has a zero Jacobian (all stereo observations fail G-solver's in-image
      test) carry no information; they are excluded. Factors with an all-zero Jacobian are dropped.
  (b) every remaining SE3Point landmark gets the prior p_L * I at its initial ORB-SLAM3 position
      (kind "landmark_prior", part of G).
  (c) no explicit stereo gauge prior (CERTIFY_GAUGE_PRIOR factors are dropped if present).
"""

import json
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

GAUGE_KINDS = ("gauge", "landmark_prior", "anchor")


class Model:
    """Block linear model. Blocks are variables with dimension dims[i]; `kinds` is a per-block label
    ("pose", "landmark", "marker", ...). factors: list of dicts with keys
        kind  str
        vars  (n_f, m) int      block indices, in the column order of J
        J     (n_f, r, c) float c = sum of the dims of the m variables (the same for the whole group)
        Lam   (n_f, r, r) float information matrix Lambda_f (symmetric PSD)
        r     (n_f, r) float    residual r_f  (b contribution J^T Lambda r)
    """

    def __init__(self, dims, kinds, factors, pose_index=None, block_ids=None, gp_kind="gp", meta=None, elim=None):
        self.dims = np.asarray(dims, dtype=int)
        self.n_blocks = len(self.dims)
        self.offsets = np.concatenate([[0], np.cumsum(self.dims)[:-1]]).astype(int)
        self.N = int(self.dims.sum())
        self.kinds = np.asarray(kinds)
        self.pose_index = np.asarray(pose_index if pose_index is not None else -np.ones(self.n_blocks), dtype=int)
        self.block_ids = np.asarray(block_ids if block_ids is not None else [str(i) for i in range(self.n_blocks)])
        self.factors = [f for f in factors if len(f["vars"])]
        self.gp_kind = gp_kind
        self.meta = dict(meta or {})
        # blocks that may be eliminated inside local solves: an independent set (no factor joins two of them)
        self.elim = np.zeros(self.n_blocks, bool) if elim is None else np.asarray(elim, bool)
        self._assemble()
        self._structure()
        if self.elim.any():
            both = self.adj[self.elim][:, self.elim]
            assert both.nnz == 0, "eliminable blocks must be an independent set"

    # ------------------------------------------------------------------ assembly
    def dof_index(self, blocks):
        blocks = np.atleast_1d(np.asarray(blocks, dtype=int))
        if len(blocks) == 0:
            return np.zeros(0, int)
        return np.concatenate([np.arange(self.offsets[i], self.offsets[i] + self.dims[i]) for i in blocks])

    def factor_cols(self, f):
        """(n_f, c) global dof indices of each factor's columns."""
        cols = []
        for k in range(f["vars"].shape[1]):
            v = f["vars"][:, k]
            d = int(self.dims[v[0]])
            assert np.all(self.dims[v] == d)
            cols.append(self.offsets[v][:, None] + np.arange(d)[None, :])
        return np.concatenate(cols, axis=1)

    @staticmethod
    def factor_blocks(f):
        """Hf = J^T Lambda J (n_f, c, c) and bf = J^T Lambda r (n_f, c)."""
        JL = np.einsum("nki,nkl->nil", f["J"], f["Lam"])
        return np.einsum("nil,nlj->nij", JL, f["J"]), np.einsum("nil,nl->ni", JL, f["r"])

    def _assemble(self):
        """The model is defined exactly by the factor list: every J^T Lambda J, J^T Lambda r and the sums in H,
        b are formed in extended precision and stored exactly as double hi + lo parts (f["Hf"], f["Hf_lo"],
        H, H_lo, b, b_lo). Solvers factorize the hi parts; refinement residuals use hi + lo (certify/precise)."""
        from certify import precise
        LD = precise.LD
        R, C, V = [], [], []
        b = np.zeros(self.N, dtype=LD)
        for f in self.factors:
            J, L, r = f["J"].astype(LD), f["Lam"].astype(LD), f["r"].astype(LD)
            JL = np.einsum("nki,nkl->nil", J, L)
            Hf = np.einsum("nil,nlj->nij", JL, J)
            Hf = (Hf + np.transpose(Hf, (0, 2, 1))) / 2
            bf = np.einsum("nil,nl->ni", JL, r)
            f["Hf"], f["Hf_lo"] = precise.split_ld(Hf)
            f["bf"] = bf.astype(float)
            idx = self.factor_cols(f)
            c = idx.shape[1]
            R.append(np.repeat(idx, c, axis=1).ravel())
            C.append(np.tile(idx, (1, c)).ravel())
            V.append(Hf.ravel())
            np.add.at(b, idx.ravel(), bf.ravel())
        H = sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(self.N, self.N), dtype=LD)
        H.sum_duplicates()
        self.H, self.H_lo = precise.split_ld(H)
        self.b, self.b_lo = precise.split_ld(b)

    def _structure(self):
        # block adjacency from factors touching two or more blocks (structural, not numerical)
        rows, cols = [], []
        inc_rows, inc_cols = [], []
        base = 0
        for gi, f in enumerate(self.factors):
            m = f["vars"].shape[1]
            for a in range(m):
                inc_rows.append(f["vars"][:, a])
                inc_cols.append(base + np.arange(len(f["vars"])))
                for c in range(m):
                    if a != c:
                        rows.append(f["vars"][:, a])
                        cols.append(f["vars"][:, c])
            base += len(f["vars"])
        n = self.n_blocks
        if rows:
            A = sp.csr_matrix((np.ones(sum(len(r) for r in rows)), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
            A.data[:] = 1.0
            A.sum_duplicates()
            A.data[:] = 1.0
        else:
            A = sp.csr_matrix((n, n))
        self.adj = A.tocsr()
        # block -> factors incidence (global factor ids)
        nf = base
        self.n_factors = nf
        self.incidence = sp.csr_matrix((np.ones(sum(len(r) for r in inc_rows)), (np.concatenate(inc_rows), np.concatenate(inc_cols))),
                                       shape=(n, nf)).tocsr() if inc_rows else sp.csr_matrix((n, 0))
        self.factor_group = np.concatenate([np.full(len(f["vars"]), gi) for gi, f in enumerate(self.factors)]) if self.factors else np.zeros(0, int)
        self.factor_local = np.concatenate([np.arange(len(f["vars"])) for f in self.factors]) if self.factors else np.zeros(0, int)

    # ------------------------------------------------------------------ block access
    def block_slice(self, i):
        return slice(int(self.offsets[i]), int(self.offsets[i] + self.dims[i]))

    def diag_blocks(self):
        """list of dense D_i = H_ii."""
        if not hasattr(self, "_D"):
            H = self.H.tocsr()
            self._D = [H[self.block_slice(i), self.block_slice(i)].toarray() for i in range(self.n_blocks)]
        return self._D

    def chol_D(self):
        """Lower Cholesky factors C_i of D_i (||v||_{D_i} = ||C_i^T v||)."""
        if not hasattr(self, "_CD"):
            self._CD = [np.linalg.cholesky(D) for D in self.diag_blocks()]
        return self._CD

    def block_norms(self, v):
        """||v_i||_{D_i} for all blocks."""
        C = self.chol_D()
        return np.array([np.linalg.norm(C[i].T @ v[self.block_slice(i)]) for i in range(self.n_blocks)])

    def block_dual_norms(self, g):
        """||g_i||_{D_i^-1} = ||C_i^-1 g_i|| for all blocks."""
        C = self.chol_D()
        return np.array([np.linalg.norm(sla.solve_triangular(C[i], g[self.block_slice(i)], lower=True))
                         for i in range(self.n_blocks)])

    def neighbours(self, i):
        return self.adj.indices[self.adj.indptr[i]:self.adj.indptr[i + 1]]

    def ball(self, i, K):
        """Blocks within graph distance K of i (the patch P_i(K) of brief §2.3)."""
        seen = np.zeros(self.n_blocks, bool)
        seen[i] = True
        frontier = np.array([i])
        for _ in range(K):
            if len(frontier) == 0:
                break
            nb = np.unique(self.adj[frontier].indices)
            nb = nb[~seen[nb]]
            seen[nb] = True
            frontier = nb
        return np.flatnonzero(seen)

    def factors_touching(self, blocks):
        """Global factor ids touching any of the blocks."""
        return np.unique(self.incidence[np.atleast_1d(blocks)].indices)

    def factor_vars(self, fid):
        f = self.factors[self.factor_group[fid]]
        return f["vars"][self.factor_local[fid]]

    def gp_pairs(self):
        """(n, 2) block pairs (k, k+1) of the GP chain factors, sorted by pose index."""
        out = [f["vars"] for f in self.factors if f["kind"] == self.gp_kind]
        if not out:
            return np.zeros((0, 2), int)
        P = np.concatenate(out)
        return P[np.argsort(self.pose_index[P[:, 0]])]

    def H_ld(self):
        if not hasattr(self, "_H_ld"):
            from certify import precise
            self._H_ld = precise.to_ld(self.H, self.H_lo)
        return self._H_ld

    def sub(self, rows, cols=None):
        """(hi, lo) parts of H[rows][:, cols] (CSR)."""
        cols = rows if cols is None else cols
        return self.H[rows][:, cols].tocsr(), self.H_lo[rows][:, cols].tocsr()

    def residual(self, X):
        """g = H x - b with the exact model (hi + lo) as a double-double dot product per row, rounded to
        float64; X: N or N x t."""
        from certify import precise
        B = self.b if X.ndim == 1 else np.repeat(self.b[:, None], X.shape[1], axis=1)
        Bl = self.b_lo if X.ndim == 1 else np.repeat(self.b_lo[:, None], X.shape[1], axis=1)
        return -precise.residual(self.H_ld(), X, B, Bl)


# --------------------------------------------------------------------------------------------------
# models from exports
# --------------------------------------------------------------------------------------------------
def _read_landmark_init(g2o):
    out = {}
    with open(g2o) as fh:
        for line in fh:
            if line.startswith("LANDMARK_3D"):
                s = line.split()
                out[int(s[1])] = np.array([float(s[2]), float(s[3]), float(s[4])])
    return out


def from_export(path, p_L=1e-6, eliminate_single_edge=True, drop_zero_info=True, include_certify_gauge=False):
    """Certified model of a frozen export directory (decisions D2(a)-(c)); returns (model, info) where
    info maps model blocks to export blocks and records the counts D2 asks for."""
    from certify.io import Export, GAUGE_PREFIXES

    ex = Export(path)
    kind_of = {"SE3PoseVel": "pose", "SE3Pose": "marker", "SE3Point": "landmark"}
    groups = []
    zero_dropped = {}
    for t, g in ex.factors.items():
        ids = g["ids"]
        keep = np.ones(len(ids), bool)
        if not include_certify_gauge:
            keep &= ~np.array([s.startswith("CERTIFY_GAUGE_PRIOR") for s in ids], bool)
        nz = np.abs(g["J"]).reshape(len(ids), -1).max(axis=1) > 0
        zero_dropped[t] = int(np.sum(keep & ~nz))
        keep &= nz
        gauge = np.array([s.startswith(GAUGE_PREFIXES) for s in ids], bool)
        for is_g in (False, True):
            sel = keep & (gauge == is_g)
            if not sel.any():
                continue
            groups.append(dict(kind="gauge" if is_g else t, vars=g["vars"][sel].astype(int), J=g["J"][sel].astype(float),
                               Lam=np.linalg.inv(g["Sigma"][sel]), r=-g["e"][sel].astype(float), export_type=t,
                               ids=ids[sel]))

    n = ex.n_blocks
    # structural degree in the kept factors
    touched = np.zeros(n, int)
    for f in groups:
        for k in range(f["vars"].shape[1]):
            np.add.at(touched, f["vars"][:, k], 1)

    se = {int(i): (tau, L) for i, tau, L in ex.single_edge}
    info = dict(n_export_blocks=int(n), zero_jacobian_factors_dropped=zero_dropped)
    removed = np.zeros(n, bool)

    if eliminate_single_edge and se:
        new_unary = dict(vars=[], J=[], Lam=[], r=[], ids=[])
        n_elim, n_elim_by_kind = 0, {}
        for gi, f in enumerate(groups):
            if f["vars"].shape[1] != 2:
                continue
            dims = [int(ex.dims[f["vars"][0, k]]) for k in range(2)]
            drop = np.zeros(len(f["vars"]), bool)
            for slot in (0, 1):
                other = 1 - slot
                for row in np.flatnonzero(np.isin(f["vars"][:, slot], list(se))):
                    v, p = int(f["vars"][row, slot]), int(f["vars"][row, other])
                    if ex.degree[v] != 1:
                        continue
                    tau, Lp = se[v]
                    c0 = 0 if slot == 0 else dims[0]
                    Jv = f["J"][row][:, c0:c0 + dims[slot]]
                    cp = dims[0] if slot == 0 else 0
                    Jp = f["J"][row][:, cp:cp + dims[other]]
                    Lam = f["Lam"][row]
                    Hvv = Jv.T @ Lam @ Jv + Lp
                    W = Lam - Lam @ Jv @ np.linalg.solve(Hvv, Jv.T @ Lam)
                    W = 0.5 * (W + W.T)
                    new_unary["vars"].append([p])
                    new_unary["J"].append(Jp)
                    new_unary["Lam"].append(W)
                    new_unary["r"].append(f["r"][row] - Jv @ tau)
                    new_unary["ids"].append(f["ids"][row] + "|elim:" + str(ex.var_id[v]))
                    drop[row] = True
                    removed[v] = True
                    n_elim += 1
                    k = kind_of.get(str(ex.var_type[v]), str(ex.var_type[v]))
                    n_elim_by_kind[k] = n_elim_by_kind.get(k, 0) + 1
            if drop.any():
                for key in ("vars", "J", "Lam", "r", "ids"):
                    f[key] = f[key][~drop]
        # degree-1 variables whose only edge was dropped (zero Jacobian) are simply removed below
        if new_unary["vars"]:
            # group by the dimension of the observing variable and residual
            by = {}
            for k in range(len(new_unary["vars"])):
                key = (new_unary["J"][k].shape[0], new_unary["J"][k].shape[1])
                by.setdefault(key, []).append(k)
            for key, ks in by.items():
                groups.append(dict(kind="se_unary", vars=np.array([new_unary["vars"][k] for k in ks], int),
                                   J=np.array([new_unary["J"][k] for k in ks]), Lam=np.array([new_unary["Lam"][k] for k in ks]),
                                   r=np.array([new_unary["r"][k] for k in ks]), ids=np.array([new_unary["ids"][k] for k in ks])))
        info["single_edge_eliminated"] = int(n_elim)
        info["single_edge_eliminated_by_kind"] = n_elim_by_kind
        info["single_edge_in_export"] = int(len(se))

    groups = [f for f in groups if len(f["vars"])]
    touched = np.zeros(n, int)
    for f in groups:
        for k in range(f["vars"].shape[1]):
            np.add.at(touched, f["vars"][:, k], 1)
    zero_info = (~removed) & (touched == 0)
    info["zero_information_excluded"] = int(zero_info.sum())
    info["zero_information_by_kind"] = {k: int(np.sum(zero_info & (ex.var_type == t))) for t, k in kind_of.items()}
    keep = ~removed & ~zero_info
    if not drop_zero_info:
        assert not zero_info.any()

    # landmark priors, decision D2(b)
    pts = np.flatnonzero(keep & (ex.var_type == "SE3Point"))
    if len(pts) and p_L is not None and p_L > 0:
        init = _read_landmark_init(ex.meta["g2o"]) if "g2o" in ex.meta else {}
        pt_rows = {int(b): k for k, b in enumerate(ex.blocks_of_type("SE3Point"))}
        J, Lam, r = [], [], []
        miss = 0
        for i in pts:
            lid = int(str(ex.var_id[i]).split("_")[1])
            p0 = init.get(lid)
            if p0 is None:
                miss += 1
                p0 = ex.point_p_lin[pt_rows[int(i)]]
            J.append(np.eye(3))
            Lam.append(p_L * np.eye(3))
            r.append(p0 - ex.point_p_lin[pt_rows[int(i)]])  # prior residual in the frozen chart (SE3Point is additive)
        groups.append(dict(kind="landmark_prior", vars=pts[:, None], J=np.array(J), Lam=np.array(Lam), r=np.array(r),
                           ids=np.array(["LANDMARK_PRIOR_" + str(ex.var_id[i]) for i in pts])))
        info["landmark_prior_p_L"] = p_L
        info["landmark_priors"] = int(len(pts))
        info["landmark_prior_missing_init"] = miss

    # relabel kept blocks
    new_of_old = -np.ones(n, int)
    kept = np.flatnonzero(keep)
    new_of_old[kept] = np.arange(len(kept))
    for f in groups:
        f["vars"] = new_of_old[f["vars"]]
        assert np.all(f["vars"] >= 0)
    kinds = np.array([kind_of.get(str(t), str(t)) for t in ex.var_type[kept]])
    elim = kinds == "landmark"
    model = Model(ex.dims[kept], kinds, groups, pose_index=ex.pose_index[kept], block_ids=ex.var_id[kept],
                  meta=dict(export=str(path), dataset=ex.meta.get("dataset"), p_L=p_L), elim=elim)
    info["kept_export_blocks"] = kept
    info["n_blocks"] = int(model.n_blocks)
    info["n_dof"] = int(model.N)
    info["export"] = ex
    return model, info


def state_restrict(info, z):
    """Restrict an export state vector to the model's blocks (same charts)."""
    ex = info["export"]
    kept = info["kept_export_blocks"]
    return np.concatenate([z[ex.offsets[i]:ex.offsets[i] + ex.dims[i]] for i in kept])


def export_states(info):
    """(iterations, memmap states) of the frozen run."""
    ex = info["export"]
    Z = ex.states()
    it = np.load(ex.path / "states_iters.npy")
    return it, Z


def independent_assembly(model):
    """D6: H and b assembled factor by factor with plain dense loops (a second code path, independent of
    the vectorized COO assembly in Model._assemble), as a dict of (i, j) -> block and the dense b."""
    blocks = {}
    b = np.zeros(model.N)
    for f in model.factors:
        dims = [int(model.dims[f["vars"][0, k]]) for k in range(f["vars"].shape[1])]
        cs = np.concatenate([[0], np.cumsum(dims)])
        for n in range(len(f["vars"])):
            J, L, r = f["J"][n], f["Lam"][n], f["r"][n]
            for a, va in enumerate(f["vars"][n]):
                Ja = J[:, cs[a]:cs[a + 1]]
                b[model.block_slice(va)] += Ja.T @ (L @ r)
                for c, vc in enumerate(f["vars"][n]):
                    Jc = J[:, cs[c]:cs[c + 1]]
                    key = (int(va), int(vc))
                    blk = Ja.T @ L @ Jc
                    blocks[key] = blocks[key] + blk if key in blocks else blk
    return blocks, b


def d6_check(model, H_used, b_used):
    """Max abs difference between the independent assembly and the matrices used by the direct solve."""
    blocks, b = independent_assembly(model)
    H = H_used.tocsr()
    worst, scale = 0.0, 0.0
    seen = 0
    for (i, j), blk in blocks.items():
        ref = H[model.block_slice(i), model.block_slice(j)].toarray()
        worst = max(worst, float(np.abs(ref - blk).max()))
        scale = max(scale, float(np.abs(blk).max()))
        seen += np.count_nonzero(ref)
    # nothing in H_used outside the assembled blocks
    extra = int(H.count_nonzero() - seen)
    return dict(max_abs_diff_H=worst, max_abs_H=scale, rel_diff_H=worst / scale if scale else 0.0,
                max_abs_diff_b=float(np.abs(b - b_used).max()), max_abs_b=float(np.abs(b).max()),
                nonzeros_outside_factor_blocks=extra)


def reduce_blocks(model, elim_kinds=("landmark",)):
    """Exact marginalization of every block whose kind is in elim_kinds (an independent set: no factor joins
    two of them) into its neighbours: the reduced model's H is the Schur complement of H onto the kept blocks
    and its solution is the full model's solution restricted to them.

    For an eliminated block l with pairwise factors f (on l and one kept block p_f; J_f = [J_fp, J_fl]) and
    unary factors u on l (landmark priors, J_u, Lambda_u, r_u):
        H_u = sum_u J_u^T Lambda_u J_u,   b_u = sum_u J_u^T Lambda_u r_u,   tau = H_u^+ b_u,
        stack the k pairwise factors: J_P (block rows onto their kept blocks), J_L, Lambda = blkdiag, r;
        H_ll = J_L^T Lambda J_L + H_u;
        new factor on the k kept blocks:  J = J_P,  Lambda' = Lambda - Lambda J_L H_ll^-1 J_L^T Lambda,
                                          r' = r - J_L tau.
    Then J^T Lambda' J = the Schur complement contribution and J^T Lambda' r' = the reduced right-hand side
    contribution (for H_u > 0 the identity is the one of the single-edge elimination in from_export; when
    b_u lies in range(H_u), in particular H_u = 0 and b_u = 0, tau = H_u^+ b_u makes it exact as well).
    Factors on kept blocks only are passed through. Returns (reduced model, kept block indices)."""
    elim = np.isin(model.kinds, elim_kinds)
    kept = np.flatnonzero(~elim)
    new_of_old = -np.ones(model.n_blocks, int)
    new_of_old[kept] = np.arange(len(kept))
    groups = []
    # pass-through factors (no eliminated block)
    per_l = {}
    for gi, f in enumerate(model.factors):
        V = f["vars"]
        has = elim[V]
        keep_rows = ~has.any(axis=1)
        if keep_rows.any():
            g = {k: f[k][keep_rows] for k in ("vars", "J", "Lam", "r")}
            g["vars"] = new_of_old[g["vars"]]
            g["kind"] = f["kind"]
            groups.append(g)
        for row in np.flatnonzero(has.any(axis=1)):
            vs = V[row]
            if has[row].sum() != 1 or len(vs) > 2:
                raise ValueError("factor joins two eliminated blocks or is more than pairwise")
            slot = int(np.flatnonzero(has[row])[0])
            l = int(vs[slot])
            per_l.setdefault(l, []).append((gi, row, slot))
    # marginalized factors, grouped by (number of observers, residual dims pattern)
    buckets = {}
    for l, items in per_l.items():
        dl = int(model.dims[l])
        Hu = np.zeros((dl, dl))
        bu = np.zeros(dl)
        obs = []
        for gi, row, slot in items:
            f = model.factors[gi]
            if f["vars"].shape[1] == 1:
                J, L, r = f["J"][row], f["Lam"][row], f["r"][row]
                Hu += J.T @ L @ J
                bu += J.T @ L @ r
            else:
                other = 1 - slot
                dims = [int(model.dims[f["vars"][row, a]]) for a in range(2)]
                c0 = 0 if slot == 0 else dims[0]
                cp = dims[0] if slot == 0 else 0
                J = f["J"][row]
                obs.append((int(f["vars"][row, other]), J[:, cp:cp + dims[other]], J[:, c0:c0 + dl], f["Lam"][row], f["r"][row]))
        if not obs:
            continue  # a block with only unary factors decouples: nothing to marginalize into
        tau = np.linalg.lstsq(Hu, bu, rcond=None)[0] if np.any(Hu) else np.zeros(dl)
        JP_blocks = [o[1] for o in obs]
        JL = np.vstack([o[2] for o in obs])
        Lam = sla.block_diag(*[o[3] for o in obs])
        r = np.concatenate([o[4] for o in obs])
        Hll = JL.T @ Lam @ JL + Hu
        W = Lam - Lam @ JL @ np.linalg.solve(Hll, JL.T @ Lam)
        W = 0.5 * (W + W.T)
        rr = r - JL @ tau
        rows = [b.shape[0] for b in JP_blocks]
        cols = [b.shape[1] for b in JP_blocks]
        JP = np.zeros((sum(rows), sum(cols)))
        ro = np.concatenate([[0], np.cumsum(rows)])
        co = np.concatenate([[0], np.cumsum(cols)])
        for k, b in enumerate(JP_blocks):
            JP[ro[k]:ro[k + 1], co[k]:co[k + 1]] = b
        key = (len(obs), tuple(rows), tuple(cols))
        B = buckets.setdefault(key, dict(vars=[], J=[], Lam=[], r=[], ids=[]))
        B["vars"].append([new_of_old[o[0]] for o in obs])
        B["J"].append(JP)
        B["Lam"].append(W)
        B["r"].append(rr)
        B["ids"].append(str(model.block_ids[l]))
    for key, B in buckets.items():
        groups.append(dict(kind="marginalized", vars=np.array(B["vars"], int), J=np.array(B["J"]), Lam=np.array(B["Lam"]),
                           r=np.array(B["r"]), ids=np.array(B["ids"])))
    red = Model(model.dims[kept], model.kinds[kept], groups, pose_index=model.pose_index[kept], block_ids=model.block_ids[kept],
                gp_kind=model.gp_kind, meta=dict(model.meta, reduced_from=int(model.n_blocks), eliminated=list(elim_kinds)))
    return red, kept
