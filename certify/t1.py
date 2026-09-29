"""Stage 2b: the solver's nonlinear estimating equation on the full D2 model, in the chart at a stop state (certify/STAGE2B.md).

F(z) = sum_f J_f(x)' Lambda_f e_f(x) + sum_pseudo Lambda (z - a) at x = x_c (+) z over the D2 model's blocks (poses and
kept landmarks; stereo factors that are inliers at x_c, GP factors; the landmark priors p_L and the frozen single-edge
messages as linear pseudo-factors), with the corrected (true) Jacobians. Per-factor values, exact derivatives F'_f and
outward-rounded interval enclosures come from `certify_stereo --t1-export` (certify/cpp/t1_server.h).

Stage 2c's clean model (export meta drop_single_landmarks / live_anchor_precision): the server rebuilds the clean graph, and
the anchor prior on the first keyframe is a third factor type, "prior" (eval mode only).
"""

import os
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from certify.io import Export
from certify.nonlinear_probe import FullModel
from certify.sidecar_model import POSE, _landmark_init

REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "build-certify/certify/cpp/certify_stereo"
GT = {"06": "06", "03": "03", "04": "04", "10": "10"}


class T1Client:
    def __init__(self, export_dir, pdir, true_jacobians=True, timeout=300.0):
        ex = Export(export_dir)
        self.dir = Path(pdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        for f in self.dir.glob("*"):
            if f.is_file():
                f.unlink()
        seq = Path(ex.meta["g2o"]).stem
        cmd = [str(BIN), "--g2o", str(REPO / ex.meta["g2o"]), "--gt", str(REPO / f"examples/data/datasets/stereo_visual/gt/{seq}.tum"),
               "--keyframes", str(ex.meta["keyframes"]), "--config", ex.meta["config"], "--out", str(self.dir / "out"),
               "--t1-export", str(export_dir), "--t1-dir", str(self.dir)] + (["--true-jacobians"] if true_jacobians else [])
        if ex.meta.get("drop_single_landmarks"):
            cmd.append("--drop-single-landmarks")
        if ex.meta.get("live_anchor_precision"):
            cmd += ["--live-anchor", repr(float(ex.meta["live_anchor_precision"]))]
        env = dict(os.environ, OMP_STACKSIZE="256M")
        self.proc = subprocess.Popen(cmd, stdout=open(self.dir / "server.log", "w"), stderr=subprocess.STDOUT, env=env)
        t0 = time.time()
        while not (self.dir / "server_ready").exists():
            if self.proc.poll() is not None:
                raise RuntimeError("t1 server exited: " + (self.dir / "server.log").read_text()[-2000:])
            if time.time() - t0 > timeout:
                raise RuntimeError("t1 server did not start")
            time.sleep(0.05)
        import json
        self.load_check = json.loads((self.dir / "load_check.json").read_text())
        tf = self.dir / "t1_types.txt"
        self.types = tuple(tf.read_text().split()) if tf.exists() else ("stereo", "gp")
        self.req = 0

    def _request(self, mode, z, rad=None):
        np.save(self.dir / "t1_z.tmp.npy", np.asarray(z, float))
        os.replace(self.dir / "t1_z.tmp.npy", self.dir / "t1_z.npy")
        if rad is not None:
            np.save(self.dir / "t1_rad.tmp.npy", np.asarray(rad, float))
            os.replace(self.dir / "t1_rad.tmp.npy", self.dir / "t1_rad.npy")
        self.req += 1
        (self.dir / "t1_request.tmp").write_text("%d %s\n" % (self.req, mode))
        os.replace(self.dir / "t1_request.tmp", self.dir / "t1_request")
        while True:
            try:
                if int((self.dir / "t1_done").read_text().split()[0]) == self.req:
                    break
            except (FileNotFoundError, ValueError, IndexError):
                pass
            if self.proc.poll() is not None:
                raise RuntimeError("t1 server exited: " + (self.dir / "server.log").read_text()[-2000:])
            time.sleep(0.002)

    def eval(self, z):
        self._request("eval", z)
        return {t: dict(F=np.load(self.dir / f"t1_F_{t}.npy"), dF=np.load(self.dir / f"t1_dF_{t}.npy"),
                        flags=np.load(self.dir / f"t1_flags_{t}.npy")) for t in self.types}

    def enclose(self, z, rad):
        self._request("enclose", z, rad)
        out = {}
        for t in ("stereo", "gp"):
            out[t] = dict(Flo=np.load(self.dir / f"t1_Flo_{t}.npy"), Fhi=np.load(self.dir / f"t1_Fhi_{t}.npy"),
                          dFlo=np.load(self.dir / f"t1_dFlo_{t}.npy"), dFhi=np.load(self.dir / f"t1_dFhi_{t}.npy"),
                          flags=np.load(self.dir / f"t1_flags_{t}.npy"))
        return out

    def close(self):
        (self.dir / "t1_quit").write_text("1\n")
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


class Problem:
    """The D2 model at a belief-chart export: blocks, factor inclusion, pseudo-factors and F / F' assembly."""

    def __init__(self, export_dir, p_L=1e-6):
        self.ex = ex = Export(export_dir)
        assert np.abs(ex.z).max() == 0.0, "expects a belief-chart export (state 0)"
        g2o = str(REPO / ex.meta["g2o"])
        self.fm = FullModel(ex, p_L=p_L, g2o=g2o, init=_landmark_init(g2o))
        fm = self.fm
        # blocks of the model: poses (time order) then kept landmarks
        self.blocks = np.concatenate([fm.poses, fm.elim]).astype(int)
        self.npose = len(fm.poses)
        dims = ex.dims[self.blocks]
        self.boff = np.concatenate([[0], np.cumsum(dims)])
        self.n = int(self.boff[-1])
        self.pos = -np.ones(ex.n_blocks, int)
        self.pos[self.blocks] = np.arange(len(self.blocks))
        # export dof -> model dof
        self.dof_of_export = -np.ones(ex.n_dof, int)
        for k, b in enumerate(self.blocks):
            self.dof_of_export[ex.offsets[b]:ex.offsets[b] + ex.dims[b]] = np.arange(self.boff[k], self.boff[k + 1])
        # included factors: not gauge, nonzero Jacobian at x_c (the D2 zero-information exclusion), all variables kept
        self.fac = {}
        for t in ("stereo", "gp", "prior"):
            if t not in ex.factors:
                continue
            g = ex.factors[t]
            keep = ~np.char.startswith(g["ids"].astype(str), "CERTIFY_GAUGE_PRIOR")
            keep &= np.abs(g["J"]).reshape(len(g["J"]), -1).max(axis=1) > 0
            V = g["vars"].astype(int)
            keep &= np.all(self.pos[V] >= 0, axis=1)
            idx = np.concatenate([ex.offsets[V[:, k]][:, None] + np.arange(ex.dims[V[0, k]])[None, :] for k in range(V.shape[1])], axis=1)
            self.fac[t] = dict(keep=keep, vars=V, eidx=idx, midx=self.dof_of_export[idx])
        # pseudo-factors (linear): Lambda (z - a) on landmark blocks
        self.pseudo = [(self.boff[self.pos[b]], L, a) for b, (L, a) in fm.pseudo_of.items() if self.pos[b] >= 0]

    def export_z(self, zm):
        """Model vector -> export-order vector (the server's input); blocks outside the model stay 0."""
        z = np.zeros(self.ex.n_dof)
        m = self.dof_of_export >= 0
        z[m] = zm[self.dof_of_export[m]]
        return z

    def assemble(self, out, want_dF=True):
        """F (model dofs) and F' (sparse, model dofs) from a server `eval` answer."""
        F = np.zeros(self.n)
        rows, cols, vals = [], [], []
        flags = 0
        for t in self.fac:
            f = self.fac[t]
            k = f["keep"]
            mi = f["midx"][k]
            np.add.at(F, mi.ravel(), out[t]["F"][k].ravel())
            flags |= int(np.bitwise_or.reduce(out[t]["flags"][k])) if k.any() else 0
            if want_dF:
                d = mi.shape[1]
                rows.append(np.repeat(mi, d, axis=1).ravel())
                cols.append(np.tile(mi, (1, d)).ravel())
                vals.append(out[t]["dF"][k].ravel())
        return F, rows, cols, vals, flags

    def full(self, z_model, client):
        out = client.eval(self.export_z(z_model))
        F, rows, cols, vals, flags = self.assemble(out)
        for o, L, a in self.pseudo:
            d = len(a)
            F[o:o + d] += L @ (z_model[o:o + d] - a)
            s = np.arange(o, o + d)
            rows.append(np.repeat(s, d))
            cols.append(np.tile(s, d))
            vals.append(np.asarray(L).ravel())
        dF = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(self.n, self.n))
        dF.sum_duplicates()
        return F, dF, flags

    def pose_slice(self, i):
        return slice(self.boff[i], self.boff[i + 1])


def schur_solve(A, b, npose_dof, lblocks):
    """Solve A x = b for the full model: dense Schur complement on the pose dofs, the landmark blocks (lblocks: list of
    (offset, dim)) eliminated exactly block by block (they couple only to poses)."""
    A = A.tocsr()
    P = np.arange(npose_dof)
    App = A[:npose_dof][:, :npose_dof].toarray()
    Apl = A[:npose_dof][:, npose_dof:].tocsr()
    Alp = A[npose_dof:][:, :npose_dof].tocsr()
    nL = A.shape[0] - npose_dof
    inv_blocks = []
    Ls = np.zeros((nL, nL)) if False else None
    rows, cols, vals = [], [], []
    for o, d in lblocks:
        s = slice(o - npose_dof, o - npose_dof + d)
        blk = A[o:o + d][:, o:o + d].toarray()
        Bi = np.linalg.inv(blk)
        rr = np.arange(o - npose_dof, o - npose_dof + d)
        rows.append(np.repeat(rr, d))
        cols.append(np.tile(rr, d))
        vals.append(Bi.ravel())
    Linv = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nL, nL))
    S = App - (Apl @ (Linv @ Alp)).toarray()
    rhs = b[:npose_dof] - Apl @ (Linv @ b[npose_dof:])
    xp = np.linalg.solve(S, rhs)
    xl = Linv @ (b[npose_dof:] - Alp @ xp)
    return np.concatenate([xp, xl])
