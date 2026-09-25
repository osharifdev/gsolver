"""Synthetic problems for Phase B (brief §9, Phase B).

Every GP prior, pose prior, hub observation and point observation is a G-solver factor, built and
linearized by G-solver itself through its Python bindings (build-certify/python/_gsolver): the variables
are created at a linearization point, the graph is frozen (GbpSolver.freeze), and each factor's frozen
Jacobian J_lin, error e_lin and covariance Sigma are read back. Terms that G-solver has no factor for
(the 12-dof anchor of the pure chain, the isotropic per-pose measurement information m*I of the brief)
are added in Python as unary factors.

Parameters "as in G-solver": Q_C = the kitti_05 configuration (the stereo experiments), Delta t = the
KITTI 06 keyframe intervals in order. p (the prior link precision scale) = the mean diagonal entry of a
GP factor's contribution to a pose's diagonal block, averaged over the chain's links:
    p = mean_f (tr(J_{f,k}^T Lambda_f J_{f,k}) + tr(J_{f,k+1}^T Lambda_f J_{f,k+1})) / 24.
"""

import sys
from pathlib import Path

import numpy as np

from certify.model import Model

REPO = Path(__file__).resolve().parents[1]
QC_KITTI05 = np.array([0.732725, 0.0825203, 1.1062, 0.0456018, 0.0135428, 0.0337611])


def gs():
    p = str(REPO / "build-certify" / "python")
    if p not in sys.path:
        sys.path.insert(0, p)
    import _gsolver
    return _gsolver


def kitti_dts(n):
    kt = [float(l.split()[1]) for l in open(REPO / "examples/data/datasets/stereo_visual/multirobot/06.g2o")
          if l.startswith("VERTEX_SE3")]
    d = np.diff(kt)
    return np.resize(d, n)


def compute_Qd(qc, dt):
    Qc = np.diag(np.maximum(qc, 1e-4))
    Q = np.zeros((12, 12))
    Q[0:6, 0:6] = dt ** 3 / 3.0 * Qc
    Q[0:6, 6:12] = dt ** 2 / 2.0 * Qc
    Q[6:12, 0:6] = dt ** 2 / 2.0 * Qc
    Q[6:12, 6:12] = dt * Qc
    return Q


def trajectory(n, dts, rng, v0=(1.0, 0.0, 0.0, 0.0, 0.0, 0.1), jitter=0.02):
    """Pose-velocity linearization points: constant body velocity v0 integrated with G-solver's exp map,
    with small random perturbations (so the GP errors at the linearization point are nonzero)."""
    g = gs()
    T = [np.eye(4)]
    v = [np.array(v0, float)]
    for k in range(1, n):
        T.append(T[-1] @ np.asarray(g.exp_map_se3(dts[k - 1] * v[-1])))
        v.append(np.array(v0, float) + jitter * rng.standard_normal(6))
    Tp = []
    for k in range(n):
        d = jitter * rng.standard_normal(6)
        Tp.append(T[k] @ np.asarray(g.exp_map_se3(d)))
    return T, Tp, v


class GraphBuilder:
    """Builds a G-solver factor graph through the bindings and reads its frozen linear model."""

    def __init__(self):
        self.g = gs()
        self.fg = self.g.FactorGraph()
        self.var_ids, self.var_dims, self.var_kinds, self.pose_index = [], [], [], []
        self.factors = []  # (kind, factor object, [var indices])
        self.index = {}

    def pose(self, T, v, t):
        g = self.g
        val = g.SE3PoseVelValue()
        val.pose = T
        val.vel = np.asarray(v, float)
        vid = "POSE_%.9f" % t
        self.fg.add_variable(g.SE3PoseVel(vid, val, np.eye(12), float(t)))
        return self._add(vid, 12, "pose", len([k for k in self.var_kinds if k == "pose"]))

    def marker(self, T, name):
        g = self.g
        vid = "LANDMARK_%s" % name
        self.fg.add_variable(g.SE3Pose(vid, T, np.eye(6)))
        return self._add(vid, 6, "marker", -1)

    def point(self, p, name):
        g = self.g
        vid = "POINT_%s" % name
        self.fg.add_variable(g.SE3Point(vid, np.asarray(p, float), np.eye(3)))
        return self._add(vid, 3, "landmark", -1)

    def _add(self, vid, d, kind, pidx):
        self.index[vid] = len(self.var_ids)
        self.var_ids.append(vid)
        self.var_dims.append(d)
        self.var_kinds.append(kind)
        self.pose_index.append(pidx)
        return len(self.var_ids) - 1

    def gp(self, a, b, dt, qc):
        f = self.g.SE3PoseVelGP("GP_%d_%d" % (a, b), float(dt), compute_Qd(qc, dt))
        self.fg.add_factor(f, [self.var_ids[a], self.var_ids[b]])
        self.factors.append(("gp", f, [a, b]))
        return f

    def pose_prior(self, a, Z, Sigma6):
        f = self.g.SE3PoseVelPrior("PRIOR_%d" % a, Z, Sigma6)
        self.fg.add_factor(f, [self.var_ids[a]])
        self.factors.append(("prior", f, [a]))
        return f

    def pose_marker(self, a, m, Z, Sigma6):
        f = self.g.SE3PoseVelPose("OBS_%d_%d" % (a, m), Z, Sigma6)
        self.fg.add_factor(f, [self.var_ids[a], self.var_ids[m]])
        self.factors.append(("hub_obs", f, [a, m]))
        return f

    def pose_point(self, a, l, z, Sigma3):
        f = self.g.SE3PoseVelPoint("PT_%d_%d" % (a, l), np.asarray(z, float), Sigma3)
        self.fg.add_factor(f, [self.var_ids[a], self.var_ids[l]])
        self.factors.append(("point_obs", f, [a, l]))
        return f

    def freeze(self):
        self.solver = self.g.GbpSolver(self.g.SolverScheduleType.SYNCHRONOUS)
        self.solver.freeze(self.fg, False)

    def linear_factors(self):
        """Factor groups (kind -> dict) from the frozen Jacobians: J = J_lin, Lam = Sigma^-1, r = -e_lin."""
        groups = {}
        for kind, f, vs in self.factors:
            key = (kind, tuple(self.var_dims[v] for v in vs))
            G = groups.setdefault(key, dict(kind=kind, vars=[], J=[], Lam=[], r=[]))
            G["vars"].append(vs)
            G["J"].append(np.asarray(f.J_lin))
            G["Lam"].append(np.linalg.inv(np.asarray(f.Sigma)))
            G["r"].append(-np.asarray(f.error_lin))
        out = []
        for G in groups.values():
            out.append(dict(kind=G["kind"], vars=np.array(G["vars"], int), J=np.array(G["J"]), Lam=np.array(G["Lam"]),
                            r=np.array(G["r"])))
        return out


def _unary(kind, blocks, dims, lam, r):
    blocks = np.asarray(blocks, int)
    d = int(dims)
    return dict(kind=kind, vars=blocks[:, None], J=np.repeat(np.eye(d)[None], len(blocks), 0),
                Lam=np.repeat((lam * np.eye(d))[None], len(blocks), 0) if np.isscalar(lam) else lam, r=r)


def link_scale(factors, dims):
    """p = mean over GP factors of (tr(J_k^T L J_k) + tr(J_k1^T L J_k1)) / 24."""
    vals = []
    for f in factors:
        if f["kind"] != "gp":
            continue
        Hf = np.einsum("nki,nkl,nlj->nij", f["J"], f["Lam"], f["J"])
        vals.append((np.trace(Hf[:, :12, :12], axis1=1, axis2=2) + np.trace(Hf[:, 12:, 12:], axis1=1, axis2=2)) / 24.0)
    return float(np.mean(np.concatenate(vals)))


def wnoa_chain(n, m_over_p, seed=0, qc=QC_KITTI05, blind=None, hub=None, short_tracks=None, anchor_scale=1e4,
               pose_priors=None, python_terms=True, return_builder=False):
    """Pure WNOA chain (GP factors from G-solver) with a 12-dof anchor on pose 0 and isotropic per-pose
    measurement information m*I, m = m_over_p * p (brief §9 Phase B).
      blind          (start, length): no measurement on those poses
      hub            info scale h/p: one 6-dof marker (SE3Pose) observed by every pose through G-solver's
                     SE3PoseVelPose factor with information (h/p) * p * I_6
      short_tracks   (per_pose, info/p): 3-dof points each observed by 2-4 consecutive poses through
                     G-solver's SE3PoseVelPoint factor
      pose_priors    info/p for G-solver SE3PoseVelPrior factors on every pose (the F4 test chains)
      python_terms   False: leave out the Python-side anchor and m*I (graph = G-solver factors only)"""
    rng = np.random.default_rng(seed)
    dts = kitti_dts(n)
    times = np.concatenate([[0.0], np.cumsum(dts[:n - 1])])
    T, Tp, v = trajectory(n, dts, rng)
    # p from the GP factors alone (a separate G-solver graph on the same linearization points)
    g0 = GraphBuilder()
    ps0 = [g0.pose(Tp[k], v[k], times[k]) for k in range(n)]
    for k in range(n - 1):
        g0.gp(ps0[k], ps0[k + 1], times[k + 1] - times[k], qc)
    g0.freeze()
    p = link_scale(g0.linear_factors(), g0.var_dims)

    gb = GraphBuilder()
    poses = [gb.pose(Tp[k], v[k], times[k]) for k in range(n)]
    for k in range(n - 1):
        gb.gp(poses[k], poses[k + 1], times[k + 1] - times[k], qc)
    if pose_priors is not None:
        for k in range(n):
            gb.pose_prior(poses[k], T[k], np.eye(6) / (pose_priors * p))
    if hub is not None:
        Th = np.eye(4)
        Th[:3, 3] = [5.0, 2.0, 1.0]
        hb = gb.marker(Th, "hub")
        for k in range(n):
            gb.pose_marker(poses[k], hb, np.linalg.inv(T[k]) @ Th, np.eye(6) / (hub * p))
    if short_tracks is not None:
        per_pose, info = short_tracks
        c = 0
        for k in range(n - 1):
            for _ in range(per_pose):
                L = int(rng.integers(2, 5))
                obs = list(range(k, min(n, k + L)))
                if len(obs) < 2:
                    continue
                pw = T[k][:3, :3] @ np.array([rng.uniform(-3, 3), rng.uniform(-2, 2), rng.uniform(4, 12)]) + T[k][:3, 3]
                l = gb.point(pw + 0.05 * rng.standard_normal(3), str(c))
                c += 1
                for a in obs:
                    z = T[a][:3, :3].T @ (pw - T[a][:3, 3]) + 0.01 * rng.standard_normal(3)
                    gb.pose_point(poses[a], l, z, np.eye(3) / (info * p))
    gb.freeze()
    factors = gb.linear_factors()
    dims = np.array(gb.var_dims)
    if python_terms:
        # anchor: 12-dof prior on pose 0 (the pure chain has a constant-velocity null space otherwise)
        factors.append(_unary("anchor", [poses[0]], 12, anchor_scale * p, np.zeros((1, 12))))
        m = m_over_p * p
        if m > 0:
            meas = [k for k in range(n) if blind is None or not (blind[0] <= k < blind[0] + blind[1])]
            r = rng.standard_normal((len(meas), 12)) / np.sqrt(m)
            factors.append(_unary("meas", [poses[k] for k in meas], 12, m, r))
    elim = np.array([k == "landmark" for k in gb.var_kinds])
    model = Model(dims, gb.var_kinds, factors, pose_index=gb.pose_index, block_ids=gb.var_ids,
                  meta=dict(n=n, m_over_p=m_over_p, p=p, hub=hub, short_tracks=short_tracks, blind=blind), elim=elim)
    if return_builder:
        return model, gb
    return model


def random_spd(n_blocks, seed=0, dims=(3, 6, 12), edge_prob=None, unary=0.05):
    """Random SPD block model: random pairwise factors (random Jacobians, residual dim 2-6) on a random
    sparse graph, plus weak unary factors that make H positive definite."""
    rng = np.random.default_rng(seed)
    d = rng.choice(dims, size=n_blocks)
    edge_prob = edge_prob or min(1.0, 3.0 / n_blocks)
    pairs = [(a, b) for a in range(n_blocks) for b in range(a + 1, n_blocks) if rng.random() < edge_prob]
    # connect the graph (a path through a random permutation)
    perm = rng.permutation(n_blocks)
    pairs += [tuple(sorted((int(perm[k]), int(perm[k + 1])))) for k in range(n_blocks - 1)]
    pairs = sorted(set(pairs))
    groups = {}
    for a, b in pairs:
        r = int(rng.integers(2, 7))
        key = (r, int(d[a]), int(d[b]))
        G = groups.setdefault(key, dict(kind="pair", vars=[], J=[], Lam=[], r=[]))
        G["vars"].append([a, b])
        G["J"].append(rng.standard_normal((r, d[a] + d[b])))
        Q = rng.standard_normal((r, r))
        G["Lam"].append(Q @ Q.T / r + 0.1 * np.eye(r))
        G["r"].append(rng.standard_normal(r))
    factors = [dict(kind=G["kind"], vars=np.array(G["vars"]), J=np.array(G["J"]), Lam=np.array(G["Lam"]), r=np.array(G["r"]))
               for G in groups.values()]
    for dd in np.unique(d):
        blk = np.flatnonzero(d == dd)
        factors.append(_unary("unary", blk, dd, unary, rng.standard_normal((len(blk), dd))))
    return Model(d, np.array(["node"] * n_blocks), factors, meta=dict(kind="random_spd", n=n_blocks, seed=seed))


def iterates(model, xbar, rng, kind="gs", keep=(0, 1, 2, 5, 10, 20, 50), scale=1.0):
    """Block-Jacobi ("jacobi") or block-Gauss-Seidel ("gs") iterations on H x = b from a random
    perturbation of xbar (size ~scale in each block's D-metric). Returns {iteration: x}."""
    H = model.H.tocsr()
    b = model.b
    D = model.diag_blocks()
    Dinv = [np.linalg.inv(Di) for Di in D]
    x = xbar.copy()
    for i in range(model.n_blocks):
        s = model.block_slice(i)
        C = np.linalg.cholesky(D[i])
        x[s] += scale * np.linalg.solve(C.T, rng.standard_normal(len(C)))
    out = {}
    last = max(keep)
    rows = [H[model.block_slice(i)] for i in range(model.n_blocks)]
    for t in range(last + 1):
        if t in keep:
            out[t] = x.copy()
        if kind == "jacobi":
            g = H @ x - b
            dx = np.concatenate([Dinv[i] @ g[model.block_slice(i)] for i in range(model.n_blocks)])
            x = x - dx
        else:
            for i in range(model.n_blocks):
                s = model.block_slice(i)
                gi = rows[i] @ x - b[s]
                x[s] = x[s] - Dinv[i] @ gi
    return out
