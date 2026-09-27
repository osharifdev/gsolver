"""Phase D, WP6b: the nonlinear residual after the exact local step (certify/NL_PROBE.md).

The residual term of Theorem T1 (PROOF_NOTE_PART_II.md), measured without curvature constants. For pose i with the
patch P the certificate used at that round (sidecar Engine):

  1. Step. On the reduced pose model, dP = -S_PP^-1 g_P (g = S x - b at the current state, the same factorization as
     the certificate). Every eliminated block l observed by P (landmarks; ChArUco markers) takes its back-substituted
     step dl = -H_ll^-1 (g_l + H_lP dP) in the full model of decision D2 (poses plus the eliminated blocks, the landmark
     priors p_L and the frozen single-edge messages as pseudo-factors), so the linear model's gradient vanishes on P and
     on those blocks. z1 = z + d on these targets, the current state z elsewhere.
  2. Residual. F(z1) = sum_f J~_f(x)' Lambda_f e_f(x) at x = x_lin (+) z1 on the targets: every factor touching them is
     re-evaluated by its own C++ code (certify_common.h ProbeServer), plus the linear pseudo-factors.
  3. s_i^NL = || [S_PP^-1 F~_P]_i ||_{D_i},  F~_P = F_P - H_PL H_LL^-1 F_L: the next local step of the same patch.
Diagnostic: s^NL_state, the same norm of F(z) - F_lin(z) at the current state (no step), and the sizes of the
eliminated blocks' back-substituted steps.

Attribution by factor class c: F(z1) = sum_c (F_nl[c] - F_lin[c]) + F_lin(z1), where F_lin(z1), the linear model's
gradient at z1, vanishes on the targets up to rounding (checked per probe); s^c is s^NL computed from F_nl[c] - F_lin[c].
Attribution by mechanism: F_nl - F_lin = (F_jac - F_lin) + (F_true - F_jac) + (F_nl - F_true):
  Jacobian change  (J~(x) - J~_lin)' Lambda (e_lin + J~_lin tau)       (Gauss-Newton, first order times the residual)
  model mismatch   J~(x)' Lambda (J_lin - J~_lin) tau                   (first order: the stereo factor 2, Theorem T0)
  curvature        J~(x)' Lambda (e(x) - e_lin - J_lin tau)             (second order)
with J the true Jacobian (stereo rotation columns halved), tau the offsets from the linearization points.
"""

import os
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from certify.sidecar_model import POSE, _landmark_init

DOF = 12


class FullModel:
    """The D2 model before any elimination, in the export's dof numbering (restricted to kept blocks)."""

    def __init__(self, ex, p_L=None, g2o=None, init=None):
        self.ex = ex
        n, dims, vt = ex.n_blocks, ex.dims, ex.var_type
        se = {i: (tau, L) for i, tau, L in ex.single_edge}
        rows, cols, vals = [], [], []
        g = np.zeros(ex.n_dof)
        touched = np.zeros(n, int)
        self.pseudo = []  # (block, Lambda (d x d), anchor (d,)): the pseudo-factor Lambda (z_v - anchor)
        for t, grp in ex.factors.items():
            keep = ~np.char.startswith(grp["ids"].astype(str), "CERTIFY_GAUGE_PRIOR")
            J = grp["J"].astype(float)
            keep &= np.abs(J).reshape(len(J), -1).max(axis=1) > 0
            if not keep.any():
                continue
            J = J[keep]
            V = grp["vars"][keep].astype(int)
            Lam = np.linalg.inv(grp["Sigma"][keep])
            e = grp["e"][keep].astype(float)
            idx = np.concatenate([ex.offsets[V[:, k]][:, None] + np.arange(dims[V[0, k]])[None, :] for k in range(V.shape[1])], axis=1)
            LJ = Lam @ J
            Hf = np.einsum("nma,nmb->nab", J, LJ)
            zf = ex.z[idx]
            gf = np.einsum("nma,nm->na", LJ, e + np.einsum("nma,na->nm", J, zf))
            d = idx.shape[1]
            rows.append(np.repeat(idx, d, axis=1).ravel())
            cols.append(np.tile(idx, (1, d)).ravel())
            vals.append(Hf.ravel())
            np.add.at(g, idx.ravel(), gf.ravel())
            for k in range(V.shape[1]):
                np.add.at(touched, V[:, k], 1)
        # single-edge variables (degree 1 in the solver graph, with a frozen message) whose one factor is kept
        single = np.array([i for i in se if ex.degree[i] == 1 and touched[i] > 0], dtype=int)
        for i in single:
            tau, L = se[int(i)]
            self.pseudo.append((int(i), L, tau))
        # landmark priors p_L at the g2o LANDMARK_3D positions, for the other kept SE3Point blocks
        pts = np.flatnonzero((touched > 0) & (vt == "SE3Point"))
        pts = pts[~np.isin(pts, single)]
        if len(pts) and p_L:
            init = init if init is not None else (_landmark_init(g2o) if g2o else {})
            prow = {int(b): k for k, b in enumerate(np.flatnonzero(vt == "SE3Point"))}
            plin = ex.point_p_lin[[prow[int(i)] for i in pts]]
            for k, i in enumerate(pts):
                p0 = init.get(int(str(ex.var_id[i]).split("_")[1]), plin[k])
                self.pseudo.append((int(i), p_L * np.eye(3), p0 - plin[k]))  # (p_lin + z) - p0 = z - (p0 - p_lin)
        for i, L, anchor in self.pseudo:
            s_ = np.arange(ex.offsets[i], ex.offsets[i] + dims[i])
            rows.append(np.repeat(s_, len(s_)))
            cols.append(np.tile(s_, len(s_)))
            vals.append(np.asarray(L).ravel())
            g[s_] += L @ (ex.z[s_] - anchor)
        self.pseudo_of = {i: (L, a) for i, L, a in self.pseudo}
        H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ex.n_dof, ex.n_dof))
        H.sum_duplicates()
        self.H, self.g = H, g
        keep = touched > 0
        poses = np.flatnonzero(keep & (vt == POSE))
        self.poses = poses[np.argsort(ex.timestamp[poses], kind="stable")]
        self.elim = np.flatnonzero(keep & (vt != POSE))
        self.pdofs = (ex.offsets[self.poses][:, None] + np.arange(DOF)[None, :]).ravel()
        self.edofs = np.concatenate([np.arange(ex.offsets[i], ex.offsets[i] + dims[i]) for i in self.elim]) if len(self.elim) else np.zeros(0, int)
        # eliminated blocks: per-block inverse of H_ll, and the block-level pose -> eliminated-block adjacency
        self.einv = {}
        for i in self.elim:
            s_ = slice(ex.offsets[i], ex.offsets[i] + dims[i])
            Hll = H[s_, s_].toarray()
            self.einv[int(i)] = np.linalg.inv(0.5 * (Hll + Hll.T))
        HPL = H[self.pdofs][:, self.edofs].tocsr()
        blk_e = np.repeat(self.elim, dims[self.elim])
        self.adj = []
        for k in range(len(self.poses)):
            sub = HPL[DOF * k:DOF * k + DOF]
            self.adj.append(np.unique(blk_e[sub.indices]) if sub.nnz else np.zeros(0, int))

    def reduced(self):
        """S and g of the reduced pose model (for the check against the sidecar's model)."""
        H = self.H
        Hpp = H[self.pdofs][:, self.pdofs].toarray()
        Hpl = H[self.pdofs][:, self.edofs]
        Hlp = H[self.edofs][:, self.pdofs]
        Minv = sp.block_diag([self.einv[int(i)] for i in self.elim], format="csr")
        S = Hpp - (Hpl @ (Minv @ Hlp)).toarray()
        gred = self.g[self.pdofs] - Hpl @ (Minv @ self.g[self.edofs])
        return 0.5 * (S + S.T), gred


class ProbeClient:
    """Talks to `certify_stereo --probe-export EXPORT --probe-dir DIR` (certify_common.h ProbeServer)."""

    def __init__(self, cmd, pdir, timeout=120.0):
        self.dir = Path(pdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        for f in self.dir.glob("*"):
            if f.is_file():
                f.unlink()
        self.proc = subprocess.Popen(cmd, stdout=open(self.dir / "server.log", "w"), stderr=subprocess.STDOUT)
        t0 = time.time()
        while not (self.dir / "server_ready").exists():
            if self.proc.poll() is not None:
                raise RuntimeError("probe server exited: " + (self.dir / "server.log").read_text()[-2000:])
            if time.time() - t0 > timeout:
                raise RuntimeError("probe server did not start")
            time.sleep(0.05)
        import json
        self.load_check = json.loads((self.dir / "load_check.json").read_text())
        self.classes = (self.dir / "probe_classes.txt").read_text().split()
        self.req = 0

    def evaluate(self, probes):
        """probes: list of (target blocks (int array), z1 (flat float array)); returns [(F_nl, F_lin, F_jac, F_true (each C x L), stats)]."""
        targets = np.concatenate([p[0] for p in probes]).astype(np.int64)
        offs = np.concatenate([[0], np.cumsum([len(p[0]) for p in probes])]).astype(np.int64)
        z = np.concatenate([p[1] for p in probes]).astype(float)
        for name, arr in (("probe_targets", targets), ("probe_offsets", offs), ("probe_z", z)):
            np.save(self.dir / (name + ".tmp.npy"), arr)
            os.replace(self.dir / (name + ".tmp.npy"), self.dir / (name + ".npy"))
        self.req += 1
        (self.dir / "probe_request.tmp").write_text("%d\n" % self.req)
        os.replace(self.dir / "probe_request.tmp", self.dir / "probe_request")
        while True:
            try:
                if int((self.dir / "probe_done").read_text().split()[0]) == self.req:
                    break
            except (FileNotFoundError, ValueError, IndexError):
                pass
            if self.proc.poll() is not None:
                raise RuntimeError("probe server exited: " + (self.dir / "server.log").read_text()[-2000:])
            time.sleep(0.001)
        F = np.load(self.dir / "probe_F.npy")
        st = np.load(self.dir / "probe_stats.npy")
        out, o = [], 0
        for k, p in enumerate(probes):
            n = len(p[1])
            out.append((F[0, :, o:o + n], F[1, :, o:o + n], F[2, :, o:o + n], F[3, :, o:o + n], st[k]))
            o += n
        return out

    def fd_check(self, n_per_class=200, h=1e-6):
        """Central-difference Jacobian check at the linearization points (ProbeServer::fdCheck); returns the rows."""
        import csv
        (self.dir / "fd_done").unlink(missing_ok=True)
        (self.dir / "fd_request.tmp").write_text("%d %.17g\n" % (n_per_class, h))
        os.replace(self.dir / "fd_request.tmp", self.dir / "fd_request")
        while not (self.dir / "fd_done").exists():
            if self.proc.poll() is not None:
                raise RuntimeError("probe server exited: " + (self.dir / "server.log").read_text()[-2000:])
            time.sleep(0.01)
        return list(csv.DictReader(open(self.dir / "fd_check.tsv"), delimiter="\t"))

    def close(self):
        (self.dir / "probe_quit").write_text("1\n")
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def reduce_and_norm(prober, pr, F):
    """||[S_PP^-1 (F_P - H_PL H_LL^-1 F_L)]_i||_{D_i} for a vector F on the probe's targets."""
    fm, ex = prober.fm, prober.ex
    npd = len(pr["pd"])
    FP, FL = F[:npd], F[npd:]
    y = np.zeros(len(pr["ed"]))
    o = 0
    for l in pr["elim"]:
        d = int(ex.dims[l])
        y[o:o + d] = fm.einv[int(l)] @ FL[o:o + d]
        o += d
    red = FP - pr["HLP"].T @ y
    return float(np.linalg.norm(prober.L["C"][pr["i"]].T @ (pr["row"].Z.T @ red)))


class Prober:
    """s_i^NL for the poses of one certified round (Engine.last) on the full model of the same export."""

    def __init__(self, full, engine_last, pose_ids):
        self.fm, self.L = full, engine_last
        self.ex = full.ex
        # the engine's pose order is the builder's (poses by time), which is FullModel.poses
        assert [str(p) for p in self.ex.var_id[full.poses]] == [str(p) for p in pose_ids]

    def build(self, i):
        """The probe of reduced pose index i: its patch rows, the targets and z1."""
        fm, ex, last = self.fm, self.ex, self.L
        row = last["rows"][i]
        P = np.asarray(row.P)
        dofP = row.dofP
        cf = last["fcache"].get(P.tobytes())
        if cf is None:  # a row carried over without a factorization this round
            cf = sla.cho_factor(last["S"][np.ix_(dofP, dofP)], lower=True, check_finite=False)
            last["fcache"][P.tobytes()] = cf
        dP = -sla.cho_solve(cf, last["g"][dofP], check_finite=False)
        blocksP = fm.poses[P]
        elim = np.unique(np.concatenate([fm.adj[k] for k in P])) if len(P) else np.zeros(0, int)
        pd = (ex.offsets[blocksP][:, None] + np.arange(DOF)[None, :]).ravel()
        ed = np.concatenate([np.arange(ex.offsets[l], ex.offsets[l] + ex.dims[l]) for l in elim]) if len(elim) else np.zeros(0, int)
        HLP = fm.H[ed][:, pd]
        rhs = fm.g[ed] + HLP @ dP
        dL = np.zeros(len(ed))
        o = 0
        for l in elim:
            d = int(ex.dims[l])
            dL[o:o + d] = -fm.einv[int(l)] @ rhs[o:o + d]
            o += d
        targets = np.concatenate([blocksP, elim]).astype(np.int64)
        z1 = np.concatenate([ex.z[pd] + dP, ex.z[ed] + dL])
        z0 = np.concatenate([ex.z[pd], ex.z[ed]])  # the current state on the targets (no step)
        return dict(i=i, P=P, row=row, cf=cf, dP=dP, dL=dL, elim=elim, pd=pd, ed=ed, HLP=HLP, targets=targets, z1=z1, z0=z0)

    def finish_state(self, pr, Fnl, Flin, Fjac, Ftrue):
        """At the current state (no step): the nonlinear equation's departure from the linear model, F(z) - F_lin(z)
        (the pseudo-factors are linear and cancel), and its three parts."""
        n = lambda F: reduce_and_norm(self, pr, F)
        return dict(s_nl_state=n((Fnl - Flin).sum(axis=0)), s_nl_state_jacobian=n((Fjac - Flin).sum(axis=0)),
                    s_nl_state_mismatch=n((Ftrue - Fjac).sum(axis=0)), s_nl_state_curvature=n((Fnl - Ftrue).sum(axis=0)))

    def finish(self, pr, Fnl, Flin, Fjac, Ftrue, stats, classes):
        """s^NL, its attribution per factor class, and the linear-gradient residue of the probe."""
        fm, ex, last = self.fm, self.ex, self.L
        i, row, pd, ed = pr["i"], pr["row"], pr["pd"], pr["ed"]
        npd = len(pd)
        # pseudo-factors (linear): Lambda (z1_v - anchor) on the eliminated targets
        Fps = np.zeros(npd + len(ed))
        o = npd
        for l in pr["elim"]:
            d = int(ex.dims[l])
            if int(l) in fm.pseudo_of:
                Lm, a = fm.pseudo_of[int(l)]
                Fps[o:o + d] = Lm @ (pr["z1"][o:o + d] - a)
            o += d

        def reduce(F):
            FP, FL = F[:npd], F[npd:]
            y = np.zeros(len(ed))
            o = 0
            for l in pr["elim"]:
                d = int(ex.dims[l])
                y[o:o + d] = fm.einv[int(l)] @ FL[o:o + d]
                o += d
            return FP - pr["HLP"].T @ y

        C = last["C"][i]
        Z = row.Z  # [S_PP^-1]_{:, i}, rows in the order of row.dofP (poses of P in increasing index, like pd)

        def s_of(F):
            return float(np.linalg.norm(C.T @ (Z.T @ reduce(F))))

        F_total = Fnl.sum(axis=0) + Fps
        F_lin_total = Flin.sum(axis=0) + Fps
        out = dict(s_nl=s_of(F_total), s_lin_residue=s_of(F_lin_total),
                   s_nl_jacobian=s_of((Fjac - Flin).sum(axis=0)), s_nl_mismatch=s_of((Ftrue - Fjac).sum(axis=0)),
                   s_nl_curvature=s_of((Fnl - Ftrue).sum(axis=0)))
        for c, name in enumerate(classes):
            out["s_nl_" + name] = s_of(Fnl[c] - Flin[c])
        # the eliminated blocks' back-substituted steps (translation part, metres)
        if len(pr["elim"]):
            dims = ex.dims[pr["elim"]]
            offs = np.concatenate([[0], np.cumsum(dims)])
            st = np.array([np.linalg.norm(pr["dL"][offs[k]:offs[k] + 3]) for k in range(len(dims))])
            out.update(step_elim_median=float(np.median(st)), step_elim_p99=float(np.percentile(st, 99)), step_elim_max=float(st.max()),
                       elim_step_over_1m=int(np.sum(st > 1.0)))
        out["factors_evaluated"] = int(stats[0])
        out["inlier_flips"] = int(stats[1])
        out["targets_pose"] = int(len(pr["P"]))
        out["targets_elim"] = int(len(pr["elim"]))
        dP = pr["dP"].reshape(-1, DOF)
        out["step_max_phys"] = float(np.max(np.linalg.norm(dP[:, :6], axis=1))) if len(dP) else 0.0
        return out
