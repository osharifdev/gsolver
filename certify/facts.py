"""Numerical checks of the Phase A facts on exported models.

F3: certify/interp.py (the transcription of G-solver's GP interpolation) reproduces G-solver's own
    interpolation of the freeze-time beliefs at the GT timestamps (interp_at_freeze.npy).
F4: on a GP chain (a tree) frozen and run to convergence, the stored message from the later GP factor
    f(k, k+1) to k satisfies G_k = H_kk - Lambda^f_kk + Lambda_{f->k} (PROOF_NOTE Def. 1.1).
F5: the rotation block of every exported stereo Jacobian equals twice the right-perturbation
    derivative of the stereo projection (se3_posevel_stereo_point.cpp:77); the other blocks match.
"""

import json
import sys

import numpy as np

from certify.io import Export
from certify.interp import interpolate
from certify.phaseA import so3_exp


def f3_check(path):
    ex = Export(path)
    rows = np.load(ex.path / "interp_at_freeze.npy")
    blk = ex.blocks_of_type("SE3PoseVel")
    times = ex.timestamp[blk]
    T, v = ex.pose_T_lin, ex.pose_v_lin
    qc = np.array(ex.meta["qc_diag"])
    dt_err, dr_err, dv_err = [], [], []
    for r in rows:
        tau = r[0]
        Tg, vg = r[1:17].reshape(4, 4), r[17:23]
        Tp, vp = interpolate(times, T, v, qc, tau)
        dt_err.append(np.linalg.norm(Tp[:3, 3] - Tg[:3, 3]))
        dr_err.append(np.linalg.norm(Tp[:3, :3] - Tg[:3, :3]))
        dv_err.append(np.linalg.norm(vp - vg))
    in_range = (rows[:, 0] >= times[0]) & (rows[:, 0] <= times[-1])
    return dict(queries=int(len(rows)), inside_knot_range=int(in_range.sum()),
                max_translation_diff=float(np.max(dt_err)), max_rotation_diff=float(np.max(dr_err)),
                max_velocity_diff=float(np.max(dv_err)))


def f4_check(path):
    ex = Export(path)
    H, _ = ex.assemble()
    H = H.toarray()
    n = ex.n_blocks
    blk = lambda i, j: H[12 * i:12 * i + 12, 12 * j:12 * j + 12]
    G = [None] * n
    G[n - 1] = blk(n - 1, n - 1)
    for k in range(n - 2, -1, -1):
        G[k] = blk(k, k) - blk(k, k + 1) @ np.linalg.solve(G[k + 1], blk(k + 1, k))
    F = [None] * n
    F[0] = blk(0, 0)
    for k in range(1, n):
        F[k] = blk(k, k) - blk(k, k - 1) @ np.linalg.solve(F[k - 1], blk(k - 1, k))
    g = ex.factors["gp"]
    L = np.load(ex.path / "gp_f2v_Lambda.npy")
    rG, rF, rLit, dev = [], [], [], []
    for f, (k, k1) in enumerate(g["vars"]):
        Lf = g["J"][f].T @ np.linalg.inv(g["Sigma"][f]) @ g["J"][f]
        pred = blk(k, k) - Lf[:12, :12] + L[f, 0]
        rG.append(np.linalg.norm(pred - G[k]) / np.linalg.norm(G[k]))
        rF.append(np.linalg.norm(blk(k1, k1) - Lf[12:, 12:] + L[f, 1] - F[k1]) / np.linalg.norm(F[k1]))
        rLit.append(np.linalg.norm(blk(k, k) + L[f, 0] - G[k]) / np.linalg.norm(G[k]))
        dev.append(np.abs(pred - G[k]).max())
    return dict(poses=n, rel_err_G_median=float(np.median(rG)), rel_err_G_max=float(np.max(rG)),
                rel_err_F_median=float(np.median(rF)), rel_err_F_max=float(np.max(rF)),
                max_abs_dev_in_units_of_eps=float(np.max(dev) / 1e-6),
                literal_Hkk_plus_message_rel_err_median=float(np.median(rLit)))


def _stereo_h(T, p, cam):
    fx, fy, cx, cy, bf = cam
    pc = T[:3, :3].T @ (p - T[:3, 3])
    xl, y, z = fx * pc[0] + cx * pc[2], fy * pc[1] + cy * pc[2], pc[2]
    xr = xl - bf
    return np.array([xl / z, y / z, (xl - xr) / z])


def f5_check(path, n_sample=2000, h=1e-6, seed=0):
    ex = Export(path)
    g = ex.factors["stereo"]
    cam = ex.meta["camera"]
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(g["inlier"] == 1)
    idx = rng.choice(idx, size=min(n_sample, len(idx)), replace=False)
    pts = ex.blocks_of_type("SE3Point")
    pt_row = {int(b): k for k, b in enumerate(pts)}
    ratio_rot, err_t, err_p, err_v = [], [], [], []
    for f in idx:
        pv, lv = g["vars"][f]
        T = ex.pose_T_lin[ex.pose_index[pv]]
        p = ex.point_p_lin[pt_row[int(lv)]]
        J = g["J"][f]
        h0 = _stereo_h(T, p, cam)
        Jfd = np.zeros((3, 15))
        for c in range(6):
            d = np.zeros(6)
            d[c] = h
            Tp = T.copy()
            Tp[:3, :3] = T[:3, :3] @ so3_exp(d[3:6][None])[0]
            Tp[:3, 3] = T[:3, 3] + T[:3, :3] @ d[0:3]
            Jfd[:, c] = -(_stereo_h(Tp, p, cam) - h0) / h  # error = z - h
        for c in range(3):
            dp = np.zeros(3)
            dp[c] = h
            Jfd[:, 12 + c] = -(_stereo_h(T, p + dp, cam) - h0) / h
        scale = np.abs(Jfd[:, 3:6]).max()
        ratio_rot.append(np.sum(J[:, 3:6] * Jfd[:, 3:6]) / np.sum(Jfd[:, 3:6] ** 2))
        err_t.append(np.abs(J[:, 0:3] - Jfd[:, 0:3]).max() / max(np.abs(Jfd[:, 0:3]).max(), 1e-12))
        err_p.append(np.abs(J[:, 12:15] - Jfd[:, 12:15]).max() / max(np.abs(Jfd[:, 12:15]).max(), 1e-12))
        err_v.append(np.abs(J[:, 6:12]).max())
        _ = scale
    ratio_rot = np.array(ratio_rot)
    return dict(factors_checked=int(len(idx)), rotation_block_ratio_median=float(np.median(ratio_rot)),
                rotation_block_ratio_min=float(ratio_rot.min()), rotation_block_ratio_max=float(ratio_rot.max()),
                translation_block_rel_err_max=float(np.max(err_t)), landmark_block_rel_err_max=float(np.max(err_p)),
                velocity_block_abs_max=float(np.max(err_v)))


if __name__ == "__main__":
    which, path = sys.argv[1], sys.argv[2]
    print(json.dumps({"f3": f3_check, "f4": f4_check, "f5": f5_check}[which](path), indent=1))
