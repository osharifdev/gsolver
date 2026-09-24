"""G-solver's GP interpolation (mean), transcribed from gsolver/maths/gp_interpolation.h:266-402.

For tau in [t_i, t_{i+1}] (fact F3):

    Phi(t, s)   = [[I, (t - s) I], [0, I]]                                   (lines 16-21)
    Q(s, t)     = [[dt^3/3 Qc, dt^2/2 Qc], [dt^2/2 Qc, dt Qc]],  dt = t - s    (lines 23-31)
    Psi(tau)    = Q(t_i, tau) Phi(t_{i+1}, tau)^T Q(t_i, t_{i+1})^-1           (line 330)
    Lambda(tau) = Phi(tau, t_i) - Psi(tau) Phi(t_{i+1}, t_i)                   (line 331)
    gamma_i     = [0; v_i]                                                     (line 339)
    gamma_i+1   = [Log(T_i^-1 T_i+1); blkdiag(R, R) v_i+1],  R = R_i^T R_i+1     (lines 341-350)
    T(tau)      = T_i Exp(Lambda_1 gamma_i + Psi_1 gamma_i+1)                  (line 352)
    w(tau)      = blkdiag(R^T, R^T)(Lambda_2 gamma_i + Psi_2 gamma_i+1),  R = R_i^T R(tau)   (354-358)

with Qc = diag(qc) and 1e-4 added to entries below 1e-4 (lines 277-281); Log/Exp are the SO(3) x R^3
maps of geometry3d.h (translation part additive, rotation by Rodrigues). The map is linear in the
local variables (gamma_i, gamma_i+1) but not in the endpoint states: gamma_i+1 contains the Log of
the relative pose and a rotated velocity, and the result is retracted with Exp. Outside the knots,
extrapolation T_i Exp((tau - t_i) v_i), w = v_i (lines 296-311).
"""

import numpy as np

from certify.phaseA import so3_exp, so3_log


def PHI(ti, tj):
    phi = np.eye(12)
    phi[0:6, 6:12] = (ti - tj) * np.eye(6)
    return phi


def Q_i(Qc, ti, tj):
    dt = tj - ti
    Q = np.zeros((12, 12))
    Q[0:6, 0:6] = dt ** 3 / 3.0 * Qc
    Q[0:6, 6:12] = dt ** 2 / 2.0 * Qc
    Q[6:12, 0:6] = dt ** 2 / 2.0 * Qc
    Q[6:12, 6:12] = dt * Qc
    return Q


def lambda_psi(qc_diag, ti, ti1, tau):
    Qc = np.diag(qc_diag).astype(float)
    Qc[np.diag_indices(6)] = np.where(np.diag(Qc) < 1e-4, np.diag(Qc) + 1e-4, np.diag(Qc))
    Psi = Q_i(Qc, ti, tau) @ PHI(ti1, tau).T @ np.linalg.inv(Q_i(Qc, ti, ti1))
    Lam = PHI(tau, ti) - Psi @ PHI(ti1, ti)
    return Lam, Psi


def _exp(x):
    T = np.eye(4)
    T[:3, :3] = so3_exp(x[3:6][None])[0]
    T[:3, 3] = x[0:3]
    return T


def _log(T):
    return np.concatenate([T[:3, 3], so3_log(T[None, :3, :3])[0]])


def interpolate(times, T, v, qc_diag, tau):
    """Pose (4x4) and body velocity (6) at tau, given sorted knot times, poses T (n,4,4), velocities v (n,6)."""
    n = len(times)
    idx = 0
    for j in range(n):
        if times[j] <= tau:
            idx = j
        else:
            break
    if tau < times[0] or idx == n - 1:
        i = 0 if tau < times[0] else idx
        return T[i] @ _exp((tau - times[i]) * v[i]), v[i].copy()
    Lam, Psi = lambda_psi(qc_diag, times[idx], times[idx + 1], tau)
    Ti, Tn = T[idx], T[idx + 1]
    gi = np.concatenate([np.zeros(6), v[idx]])
    R = Ti[:3, :3].T @ Tn[:3, :3]
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[3:, 3:] = R
    gn = np.concatenate([_log(np.linalg.inv(Ti) @ Tn), A @ v[idx + 1]])
    Tt = Ti @ _exp(Lam[:6] @ gi + Psi[:6] @ gn)
    R = Ti[:3, :3].T @ Tt[:3, :3]
    A = np.zeros((6, 6))
    A[:3, :3] = R.T
    A[3:, 3:] = R.T
    return Tt, A @ (Lam[6:] @ gi + Psi[6:] @ gn)
