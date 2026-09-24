"""Trajectory accuracy metrics (ATE / ARE) for TUM-format trajectories.

Definitions follow the paper's Table II protocol as reconstructed in
reproduction/src/repro_common.h: every ground-truth pose is matched with the
estimate at the same timestamp; ATE is the RMSE of the translation error and
ARE the RMSE of the rotation angle ||Log(R_gt^T R_est)||. Both are reported raw
and after an SE(3) Umeyama alignment of the estimated positions onto the
ground-truth positions (no scale).

Usage: python -m certify.tools.traj_eval <estimate.tum> <groundtruth.tum>
"""

import sys

import numpy as np


def read_tum(path):
    """Return (timestamps[n], translations[n,3], rotations[n,3,3])."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            if len(vals) < 8:
                continue
            rows.append([float(v) for v in vals[:8]])
    a = np.asarray(rows, dtype=float)
    t, p, q = a[:, 0], a[:, 1:4], a[:, 4:8]  # q = (qx, qy, qz, qw)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    R = np.empty((len(a), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return t, p, R


def rotation_angles(Ra, Rb):
    """Angles of Ra[k]^T Rb[k], robust near 0 and pi."""
    D = np.einsum("kji,kjl->kil", Ra, Rb)
    cos = np.clip((np.trace(D, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    skew = np.stack([D[:, 2, 1] - D[:, 1, 2], D[:, 0, 2] - D[:, 2, 0], D[:, 1, 0] - D[:, 0, 1]], axis=1)
    sin = 0.5 * np.linalg.norm(skew, axis=1)
    return np.arctan2(sin, cos)


def umeyama_se3(src, dst):
    """R, t minimising sum ||R src_k + t - dst_k||^2 (no scale)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    C = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    return R, mu_d - R @ mu_s


def evaluate(est_path, gt_path, tol=1e-6):
    te, pe, Re = read_tum(est_path)
    tg, pg, Rg = read_tum(gt_path)
    # match by timestamp (the drivers write the estimate at exactly the GT timestamps)
    idx = np.searchsorted(tg, te)
    idx = np.clip(idx, 0, len(tg) - 1)
    ok = np.abs(tg[idx] - te) < tol
    if not np.any(ok):
        raise ValueError("no matching timestamps between estimate and ground truth")
    pe, Re, pg, Rg = pe[ok], Re[ok], pg[idx[ok]], Rg[idx[ok]]
    finite = np.all(np.isfinite(pe), axis=1) & np.all(np.isfinite(Re.reshape(len(Re), -1)), axis=1)
    out = {"matched": int(ok.sum()), "nonfinite": int((~finite).sum())}
    if not np.all(finite):
        out.update(ate=np.nan, are=np.nan, ate_aligned=np.nan, are_aligned=np.nan)
        return out
    out["ate"] = float(np.sqrt(np.mean(np.sum((pe - pg) ** 2, axis=1))))
    out["are"] = float(np.sqrt(np.mean(rotation_angles(Rg, Re) ** 2)))
    Ra, ta = umeyama_se3(pe, pg)
    pa = pe @ Ra.T + ta
    Rea = np.einsum("ij,kjl->kil", Ra, Re)
    out["ate_aligned"] = float(np.sqrt(np.mean(np.sum((pa - pg) ** 2, axis=1))))
    out["are_aligned"] = float(np.sqrt(np.mean(rotation_angles(Rg, Rea) ** 2)))
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    r = evaluate(sys.argv[1], sys.argv[2])
    print(" ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
