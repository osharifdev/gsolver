"""Fast per-round construction of the certified pose model for the Phase D sidecar.

Builds, from one frozen-chart export of the live solver (the current single-chart linearization and state), the
model of decision D2 exactly as certify.model.from_export does:
  - factors with an all-zero Jacobian and CERTIFY_GAUGE_PRIOR factors are dropped;
  - every single-edge variable is eliminated exactly into its neighbour together with its frozen single-edge
    message (G-solver never updates it), W = Lam - Lam Jv (Jv' Lam Jv + Lp)^-1 Jv' Lam, r' = r - Jv tau;
  - variables left without factors are excluded;
  - every remaining SE3Point landmark gets the prior p_L I at its initial (g2o LANDMARK_3D) position;
then eliminates every non-pose block exactly (landmarks, and ChArUco markers) and returns the reduced pose model
    S = H_PP - H_PL H_LL^-1 H_LP,   b_red = b_P - H_PL H_LL^-1 b_L,
with poses ordered by time. With true_jacobian=True the stereo factors' rotation columns (pose tangent 3:6) are
halved first (Theorem T0: J is the solver's J~ without the factor 2 of se3_posevel_stereo_point.cpp:77), giving the
reduced Fisher information used for S_lo. Double precision throughout (the sidecar's precision; the three sampled
rounds are re-validated with the extended-precision Phase C pipeline).
"""

import numpy as np
import scipy.sparse as sp

from certify.io import Export

POSE = "SE3PoseVel"


def _landmark_init(g2o):
    out = {}
    with open(g2o) as fh:
        for line in fh:
            if line.startswith("LANDMARK_3D"):
                s = line.split()
                out[int(s[1])] = np.array([float(s[2]), float(s[3]), float(s[4])])
    return out


class PoseModelBuilder:
    """Holds what does not change between rounds (the g2o landmark positions)."""

    def __init__(self, p_L=None, g2o=None):
        self.p_L = p_L
        self.init = _landmark_init(g2o) if (g2o and p_L) else {}

    def build(self, path, true_jacobian=False, ex=None):
        ex = ex if ex is not None else Export(path)
        n, dims, vt = ex.n_blocks, ex.dims, ex.var_type
        se = {i: (tau, L) for i, tau, L in ex.single_edge}
        groups = []
        for t, g in ex.factors.items():
            keep = ~np.char.startswith(g["ids"].astype(str), "CERTIFY_GAUGE_PRIOR")
            J = g["J"].astype(float)
            keep &= np.abs(J).reshape(len(J), -1).max(axis=1) > 0
            if not keep.any():
                continue
            J = J[keep].copy()
            if true_jacobian and t == "stereo":
                J[:, :, 3:6] *= 0.5
            groups.append(dict(type=t, vars=g["vars"][keep].astype(int), J=J, Lam=np.linalg.inv(g["Sigma"][keep]),
                               r=-g["e"][keep].astype(float)))

        # exact elimination of single-edge variables with their frozen message (D2(a))
        removed = np.zeros(n, bool)
        unary = []
        for f in groups:
            if f["vars"].shape[1] != 2:
                continue
            d = [int(dims[f["vars"][0, k]]) for k in range(2)]
            drop = np.zeros(len(f["vars"]), bool)
            for slot in (0, 1):
                other = 1 - slot
                rows = np.flatnonzero((ex.degree[f["vars"][:, slot]] == 1) & np.isin(f["vars"][:, slot], list(se)))
                if not len(rows):
                    continue
                cv = slice(0, d[0]) if slot == 0 else slice(d[0], d[0] + d[1])
                cp = slice(d[0], d[0] + d[1]) if slot == 0 else slice(0, d[0])
                Jv, Jp, Lam, r = f["J"][rows][:, :, cv], f["J"][rows][:, :, cp], f["Lam"][rows], f["r"][rows]
                v = f["vars"][rows, slot]
                tau = np.array([se[int(i)][0] for i in v])
                Lp = np.array([se[int(i)][1] for i in v])
                LJ = Lam @ Jv
                Hvv = np.einsum("nea,neb->nab", Jv, LJ) + Lp
                W = Lam - LJ @ np.linalg.solve(Hvv, np.transpose(LJ, (0, 2, 1)))
                W = 0.5 * (W + np.transpose(W, (0, 2, 1)))
                unary.append(dict(type="se_unary", vars=f["vars"][rows, other][:, None], J=Jp, Lam=W,
                                  r=r - np.einsum("nea,na->ne", Jv, tau)))
                drop[rows] = True
                removed[v] = True
            if drop.any():
                for k in ("vars", "J", "Lam", "r"):
                    f[k] = f[k][~drop]
        groups = [f for f in groups + unary if len(f["vars"])]

        touched = np.zeros(n, int)
        for f in groups:
            for k in range(f["vars"].shape[1]):
                np.add.at(touched, f["vars"][:, k], 1)
        keep = ~removed & (touched > 0)

        # landmark priors (D2(b))
        pts = np.flatnonzero(keep & (vt == "SE3Point"))
        if len(pts) and self.p_L:
            prow = {int(b): k for k, b in enumerate(np.flatnonzero(vt == "SE3Point"))}
            plin = ex.point_p_lin[[prow[int(i)] for i in pts]]
            p0 = np.array([self.init.get(int(str(ex.var_id[i]).split("_")[1]), plin[k]) for k, i in enumerate(pts)])
            groups.append(dict(type="landmark_prior", vars=pts[:, None], J=np.repeat(np.eye(3)[None], len(pts), 0),
                               Lam=np.repeat(self.p_L * np.eye(3)[None], len(pts), 0), r=p0 - plin))

        # block roles: poses (ordered by time) and eliminated blocks
        poses = np.flatnonzero(keep & (vt == POSE))
        poses = poses[np.argsort(ex.timestamp[poses], kind="stable")]
        pidx = -np.ones(n, int)
        pidx[poses] = np.arange(len(poses))
        elim = np.flatnonzero(keep & (vt != POSE))
        lidx = -np.ones(n, int)
        lidx[elim] = np.arange(len(elim))
        ldim = dims[elim]
        loff = np.concatenate([[0], np.cumsum(ldim)])
        npo = len(poses)

        PP = np.zeros((npo, npo, 12, 12))
        bP = np.zeros((npo, 12))
        LLd = {int(d): np.zeros((int(np.sum(ldim == d)), d, d)) for d in np.unique(ldim)}
        bLd = {int(d): np.zeros((int(np.sum(ldim == d)), d)) for d in np.unique(ldim)}
        lsub = np.zeros(len(elim), int)  # index within its dimension group
        for d in LLd:
            lsub[ldim == d] = np.arange(int(np.sum(ldim == d)))
        PL_rows, PL_cols, PL_vals = [], [], []  # the H_PL blocks, as (pose dof, elim dof, value) triples

        for f in groups:
            V = f["vars"]
            k = V.shape[1]
            d = [int(dims[V[0, s]]) for s in range(k)]
            off = np.concatenate([[0], np.cumsum(d)])
            LJ = f["Lam"] @ f["J"]
            Lr = np.einsum("nef,nf->ne", f["Lam"], f["r"])
            for a in range(k):
                Ja = f["J"][:, :, off[a]:off[a + 1]]
                va = V[:, a]
                ba = np.einsum("nea,ne->na", Ja, Lr)
                pa = pidx[va]
                if np.all(pa >= 0):
                    np.add.at(bP, pa, ba)
                else:
                    assert np.all(pa < 0), "mixed block roles within one factor slot"
                    la = lidx[va]
                    np.add.at(bLd[d[a]], lsub[la], ba)
                for b in range(k):
                    blk = np.einsum("nea,neb->nab", Ja, LJ[:, :, off[b]:off[b + 1]])
                    vb = V[:, b]
                    pb = pidx[vb]
                    if np.all(pa >= 0) and np.all(pb >= 0):
                        np.add.at(PP, (pa, pb), blk)
                    elif np.all(pa >= 0) and np.all(pb < 0):
                        lb = lidx[vb]
                        rr = (12 * pa)[:, None, None] + np.arange(12)[None, :, None]
                        cc = loff[lb][:, None, None] + np.arange(d[b])[None, None, :]
                        PL_rows.append(np.broadcast_to(rr, blk.shape).ravel())
                        PL_cols.append(np.broadcast_to(cc, blk.shape).ravel())
                        PL_vals.append(blk.ravel())
                    elif np.all(pa < 0) and np.all(pb < 0):
                        assert np.all(va == vb), "a factor couples two eliminated blocks"
                        np.add.at(LLd[d[a]], lsub[lidx[va]], blk)
                    # (elim, pose) pairs are the transposes of the (pose, elim) ones

        N = 12 * npo
        S = PP.transpose(0, 2, 1, 3).reshape(N, N)
        b = bP.reshape(N).copy()
        if len(elim):
            # H_LL = C C' per eliminated block; U = H_PL C^-T, S -= U U', b -= U C^-1 b_L
            Ld = {dd: np.linalg.cholesky(LLd[dd]) for dd in LLd}
            Linv = {dd: np.linalg.inv(Ld[dd]) for dd in Ld}
            NL = int(loff[-1])
            T_rows, T_cols, T_vals = [], [], []  # the block-diagonal C^-T as a sparse matrix
            w = np.zeros(NL)
            for dd in Ld:
                sel = np.flatnonzero(ldim == dd)
                base = loff[sel]
                Mt = np.transpose(Linv[dd], (0, 2, 1))  # C^-T
                rr = base[:, None, None] + np.arange(dd)[None, :, None]
                cc = base[:, None, None] + np.arange(dd)[None, None, :]
                T_rows.append(np.broadcast_to(rr, Mt.shape).ravel())
                T_cols.append(np.broadcast_to(cc, Mt.shape).ravel())
                T_vals.append(Mt.ravel())
                wl = np.einsum("nab,nb->na", Linv[dd], bLd[dd])
                w[(base[:, None] + np.arange(dd)[None, :]).ravel()] = wl.ravel()
            Tm = sp.csr_matrix((np.concatenate(T_vals), (np.concatenate(T_rows), np.concatenate(T_cols))), shape=(NL, NL))
            HPL = sp.csr_matrix((np.concatenate(PL_vals), (np.concatenate(PL_rows), np.concatenate(PL_cols))), shape=(N, NL))
            U = (HPL @ Tm).tocsr()
            S = S - (U @ U.T).toarray()
            b = b - U @ w
        S = 0.5 * (S + S.T)
        z = ex.z
        xP = np.concatenate([z[ex.offsets[i]:ex.offsets[i] + 12] for i in poses]) if npo else np.zeros(0)
        return dict(S=S, b=b, x=xP, pose_blocks=poses, pose_ids=ex.var_id[poses], times=ex.timestamp[poses],
                    n_eliminated=int(len(elim)), n_removed_single_edge=int(removed.sum()),
                    n_zero_info=int(np.sum(~removed & (touched == 0))), export=ex)
