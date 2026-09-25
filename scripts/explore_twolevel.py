"""Exploration on one export (default: stereo prefix frozen at the normal-run cap):

(1) Oracle early-exit curve: with the exact error e_i(t) and the exact statistical radius
    S_i = sqrt(lambda_min(D_i^1/2 Sigma_ii D_i^1/2)) q_d(0.05), the fraction of poses with
    ||e_i||_{D_i} <= 0.1 S_i at each iteration. No certificate can stop a pose earlier than this.
(2) Two-level split of the error: coarse space Z = rigid motions (6 dof) of trajectory segments of s
    keyframes, applied to the segment's poses and to the landmarks whose median observer lies in the segment.
    In the frozen charts a world rigid motion (rho, phi) moves pose k by tau_t = R_k^T (phi x t_k + rho),
    tau_r = R_k^T phi, tau_v = 0, and landmark l by tau_p = phi x p_l + rho. The coarse error component
    Z c, c = (Z^T H Z)^-1 Z^T g, is exactly computable from GBP's residual g; w = e - Z c is the rest.
    Reports ||w_i|| / ||e_i|| for poses and the energy fraction captured.
"""

import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from certify import certs, phaseC, radii  # noqa: E402
from certify import model as mdl  # noqa: E402
from certify.linalg import GlobalMarginals, SPDSystem  # noqa: E402


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def coarse_basis(model, info, seg):
    ex = info["export"]
    kept = info["kept_export_blocks"]
    poses = np.flatnonzero(model.kinds == "pose")
    order = poses[np.argsort(model.pose_index[poses])]
    rank = -np.ones(model.n_blocks, int)
    rank[order] = np.arange(len(order))
    seg_of = -np.ones(model.n_blocks, int)
    seg_of[order] = rank[order] // seg
    for l in np.flatnonzero(model.kinds == "landmark"):
        obs = [j for j in model.neighbours(l) if rank[j] >= 0]
        seg_of[l] = int(np.median(rank[obs])) // seg if obs else 0
    nseg = int(seg_of.max() + 1)
    T = ex.pose_T_lin
    pts_rows = {int(b): k for k, b in enumerate(ex.blocks_of_type("SE3Point"))}
    rows, cols, vals = [], [], []
    for b in range(model.n_blocks):
        s = seg_of[b]
        if s < 0:
            continue
        o = model.offsets[b]
        eb = kept[b]
        if model.kinds[b] == "pose":
            Tk = T[ex.pose_index[eb]]
            R, t = Tk[:3, :3], Tk[:3, 3]
            Bt = np.hstack([R.T, R.T @ (-skew(t))])       # tau_t = R^T rho + R^T (phi x t) = R^T rho - R^T [t]x phi
            Br = np.hstack([np.zeros((3, 3)), R.T])
            B = np.vstack([Bt, Br, np.zeros((6, 6))])
        elif model.kinds[b] == "landmark":
            p = ex.point_p_lin[pts_rows[int(eb)]]
            B = np.hstack([np.eye(3), -skew(p)])           # tau_p = rho + phi x p = rho - [p]x phi
        else:
            continue
        r_, c_ = np.nonzero(B)
        rows.append(o + r_)
        cols.append(6 * s + c_)
        vals.append(B[r_, c_])
    Z = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(model.N, 6 * nseg))
    return Z, nseg


def main(name="kitti06_prefix_conv"):
    path, cfg = phaseC.DATASETS[name]
    model, info = mdl.from_export(path, p_L=cfg["p_L"])
    d2 = json.loads((phaseC.OUT / "d2" / ("%s.json" % name)).read_text())
    mrec = [m for m in d2["models"] if m["p_L"] == cfg["p_L"]][0]
    iters = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 1000, 2000, 3000]
    avail = set(np.load(Path(path) / "states_iters.npy").tolist())
    iters = [t for t in iters if t in avail]
    sysm = SPDSystem(model.H, model.dims, model.elim, refine=3, K_lo=model.H_lo)
    X = phaseC.load_states(model, info, iters)
    G = model.residual(X)
    E = sysm.solve(G)
    met = certs.Metric(model)
    err = met.norms(E)
    poses = np.flatnonzero(model.kinds == "pose")
    gm = GlobalMarginals(sysm)
    lam_min = np.array([np.linalg.eigvalsh(met.C[b].T @ gm.block(int(model.offsets[b]), int(model.dims[b])) @ met.C[b])[0] for b in poses])
    S = np.sqrt(lam_min) * radii.q(12)
    out = dict(dataset=name, iters=iters, pose_floor=float(mrec["pose_floor"]["floor"]),
               S_poses=[float(q) for q in np.quantile(S, [0, 0.1, 0.5, 0.9, 1])],
               err_poses_median=[float(np.median(err[poses, j])) for j in range(len(iters))],
               err_poses_max=[float(np.max(err[poses, j])) for j in range(len(iters))],
               oracle_stop_fraction=[float(np.mean(err[poses, j] <= radii.ETA * S)) for j in range(len(iters))])
    print("statistical radius S (poses): min/10%%/median/90%%/max = %s" % np.round(out["S_poses"], 2))
    print("iters:", iters)
    print("median pose error:", np.round(out["err_poses_median"], 4))
    print("max pose error:   ", np.round(out["err_poses_max"], 4))
    print("oracle stop fraction (err <= 0.1 S):", np.round(out["oracle_stop_fraction"], 3))
    H = model.H.tocsr()
    out["twolevel"] = {}
    for seg in (161, 40, 20, 10, 5):
        Z, nseg = coarse_basis(model, info, seg)
        ZtHZ = (Z.T @ (H @ Z)).toarray()
        ZtHZ = 0.5 * (ZtHZ + ZtHZ.T)
        c = sla.solve(ZtHZ, (Z.T @ G), assume_a="pos")
        W = E - Z @ c
        wn = met.norms(W)
        ratio = wn[poses] / np.maximum(err[poses], 1e-300)
        en_e = np.einsum("ij,ij->j", E, H @ E)
        en_w = np.einsum("ij,ij->j", W, H @ W)
        rec = dict(segments=nseg, coarse_dim=6 * nseg,
                   w_over_e_poses_median=[float(np.median(ratio[:, j])) for j in range(len(iters))],
                   w_poses_max=[float(np.max(wn[poses, j])) for j in range(len(iters))],
                   energy_fraction_fine=[float(en_w[j] / en_e[j]) for j in range(len(iters))])
        out["twolevel"][seg] = rec
        print("segments of %3d keyframes (%2d segments, coarse dim %3d): median ||w_i||/||e_i|| %s ; fine energy fraction %s ; max ||w_i|| %s" % (
            seg, nseg, 6 * nseg, np.round(rec["w_over_e_poses_median"], 3), np.round(rec["energy_fraction_fine"], 4), np.round(rec["w_poses_max"], 3)))
    d = phaseC.OUT / "explore"
    d.mkdir(parents=True, exist_ok=True)
    (d / ("twolevel_%s.json" % name)).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
