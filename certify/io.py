"""Loader for G-solver exports written by certify/cpp/certify_common.h (Phase A).

An export directory holds one linear model in fixed charts (PROOF_NOTE.md §0.1):

    H = sum_f J_f^T Lambda_f J_f + G,     b = sum_f J_f^T Lambda_f r_f + b_G,     r_f = -e_f,

with Lambda_f = Sigma_f^{-1}, plus the state x (tangent coordinates in the same charts). The model is
split into labelled parts so that each can be included or left out explicitly:

    "meas"         every factor that is not a gauge term (GP priors, stereo, pose-landmark, ...)
    "gauge"        anchor priors: PRIOR_ANCHOR_* (the ChArUco drivers' own anchors) and
                   CERTIFY_GAUGE_PRIOR (the explicit stereo prior of amendment A3)
    "single_edge"  the frozen outgoing message of every variable with exactly one edge; G-solver never
                   updates it (variable_node.cpp:26-28), so it acts as a prior N(tau, Lambda^-1) on
                   that variable in the rest of the graph (but not in the variable's own belief)

Charts: SE3PoseVel tangent [t(3), r(3), v(6)] with x [+] tau = (T * Exp(tau[:6]), v + tau[6:]),
Exp(t, r) = (Rodrigues(r), t) as in gsolver/maths/geometry3d.h; SE3Pose likewise; SE3Point additive.
"""

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

GAUGE_PREFIXES = ("PRIOR_ANCHOR", "CERTIFY_GAUGE_PRIOR")


class Export:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        rows = [l.rstrip("\n").split("\t") for l in (self.path / "variables.tsv").read_text().splitlines()[1:]]
        self.var_id = np.array([r[1] for r in rows])
        self.var_type = np.array([r[2] for r in rows])
        self.dims = np.array([int(r[3]) for r in rows])
        self.offsets = np.array([int(r[4]) for r in rows])
        self.pose_index = np.array([int(r[5]) for r in rows])
        self.timestamp = np.array([float(r[6]) if r[6] != "null" else np.nan for r in rows])
        self.n_blocks = len(rows)
        self.n_dof = int(self.offsets[-1] + self.dims[-1]) if rows else 0
        self.degree = np.load(self.path / "degree.npy")
        self.z = np.load(self.path / "z.npy")

        self.factors = {}
        for f in sorted(self.path.glob("f_*_vars.npy")):
            t = f.name[2:-9]
            g = {k: np.load(self.path / f"f_{t}_{k}.npy") for k in ("vars", "J", "e", "Sigma", "msg_tau")}
            for opt in ("dt", "inlier"):
                p = self.path / f"f_{t}_{opt}.npy"
                if p.exists():
                    g[opt] = np.load(p)
            g["ids"] = np.array((self.path / f"f_{t}_ids.txt").read_text().split())
            self.factors[t] = g

        se_idx = np.load(self.path / "single_edge_index.npy")
        tau_flat = np.load(self.path / "single_edge_tau_flat.npy")
        lam_flat = np.load(self.path / "single_edge_Lambda_flat.npy")
        self.single_edge = []  # (block, tau, Lambda)
        pt, pl = 0, 0
        for i in se_idx:
            d = int(self.dims[i])
            self.single_edge.append((int(i), tau_flat[pt:pt + d], lam_flat[pl:pl + d * d].reshape(d, d)))
            pt += d
            pl += d * d

        self.pose_T_lin = np.load(self.path / "pose_T_lin.npy")
        self.pose_v_lin = np.load(self.path / "pose_v_lin.npy")
        self.se3_T_lin = np.load(self.path / "se3_T_lin.npy")
        self.point_p_lin = np.load(self.path / "point_p_lin.npy")

    # ------------------------------------------------------------------ indexing
    def block_slice(self, i):
        return slice(int(self.offsets[i]), int(self.offsets[i] + self.dims[i]))

    def blocks_of_type(self, t):
        return np.flatnonzero(self.var_type == t)

    def factor_dof_index(self, t):
        """(n_f, d) global dof indices of each factor's variables, in the factor's column order."""
        g = self.factors[t]
        cols = []
        for k in range(g["vars"].shape[1]):
            v = g["vars"][:, k]
            d = int(self.dims[v[0]]) if len(v) else 0
            cols.append(self.offsets[v][:, None] + np.arange(d)[None, :])
        return np.concatenate(cols, axis=1) if cols else np.zeros((0, 0), int)

    def is_gauge(self, t):
        return np.array([s.startswith(GAUGE_PREFIXES) for s in self.factors[t]["ids"]], dtype=bool)

    # ------------------------------------------------------------------ assembly
    def part_triplets(self, part):
        """COO triplets (rows, cols, vals) of H and the dense b for one part."""
        rows, cols, vals = [], [], []
        b = np.zeros(self.n_dof)
        if part in ("meas", "gauge"):
            for t, g in self.factors.items():
                sel = self.is_gauge(t) if part == "gauge" else ~self.is_gauge(t)
                if not sel.any():
                    continue
                J, e, S = g["J"][sel], g["e"][sel], g["Sigma"][sel]
                W = np.linalg.inv(S)
                JW = np.einsum("nmi,nmk->nik", J, W)
                Hf = np.einsum("nik,nkj->nij", JW, J)
                bf = -np.einsum("nik,nk->ni", JW, e)
                idx = self.factor_dof_index(t)[sel]
                d = idx.shape[1]
                rows.append(np.repeat(idx, d, axis=1).ravel())
                cols.append(np.tile(idx, (1, d)).ravel())
                vals.append(Hf.ravel())
                np.add.at(b, idx.ravel(), bf.ravel())
        elif part == "single_edge":
            for i, tau, L in self.single_edge:
                s = np.arange(self.offsets[i], self.offsets[i] + self.dims[i])
                rows.append(np.repeat(s, len(s)))
                cols.append(np.tile(s, len(s)))
                vals.append(L.ravel())
                b[s] += L @ tau
        else:
            raise ValueError(part)
        if not rows:
            return np.zeros(0, int), np.zeros(0, int), np.zeros(0), b
        return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), b

    def assemble(self, parts=("meas", "gauge", "single_edge")):
        """H (CSR, symmetrized exactly as the sum of symmetric factor blocks) and b for the given parts."""
        R, C, V, b = [], [], [], np.zeros(self.n_dof)
        for p in parts:
            r, c, v, bp = self.part_triplets(p)
            R.append(r)
            C.append(c)
            V.append(v)
            b += bp
        H = sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(self.n_dof, self.n_dof))
        H.sum_duplicates()
        return H, b

    # ------------------------------------------------------------------ states
    def states(self):
        """Memory-mapped (T+1, n_dof) array of frozen-run states z^(t), t = 0..T, or None."""
        p = self.path / "states_all.f64"
        if not p.exists():
            return None
        m = np.memmap(p, dtype="<f8", mode="r")
        return m.reshape(-1, self.n_dof)


def informed_blocks(H, ex):
    """Blocks with a nonzero diagonal block. A block is uninformed when every factor touching it has a
    zero Jacobian at the linearization point (e.g. all its stereo factors fail G-solver's in-image test,
    se3_posevel_stereo_point.cpp:40-60) and it has no prior: its rows and columns of H are exactly zero,
    it is decoupled from the model, and GBP leaves it decoupled too."""
    d = H.diagonal()
    return np.array([np.any(d[ex.block_slice(i)] != 0) for i in range(ex.n_blocks)])


def restrict(H, b, ex, keep_blocks):
    """H, b restricted to the dofs of the kept blocks, and the dof index map."""
    dofs = np.concatenate([np.arange(ex.offsets[i], ex.offsets[i] + ex.dims[i]) for i in np.flatnonzero(keep_blocks)])
    H = H.tocsr()[dofs][:, dofs]
    return H, b[dofs], dofs


def diag_blocks(H, ex):
    """Dense diagonal blocks D_i = H_ii, returned as {dim: (block_indices, array (n_i, d, d))}."""
    H = H.tocsr()
    out = {}
    for d in np.unique(ex.dims):
        blk = np.flatnonzero(ex.dims == d)
        base = ex.offsets[blk][:, None, None]
        r = (base + np.arange(d)[None, :, None]).repeat(d, axis=2)
        c = (base + np.arange(d)[None, None, :]).repeat(d, axis=1)
        vals = np.asarray(H[r.ravel(), c.ravel()]).reshape(len(blk), d, d)
        out[int(d)] = (blk, vals)
    return out
