"""Statistical radii (Proposition 2.7; brief §4.3) and the global energy check (Proposition 5.7; brief §8.5).

    q_d(delta) = sqrt(chi2.ppf(1 - delta, d)),   S_lo = sqrt(mu_lo) q_d(delta),   S_hi = sqrt(mu_hi) q_d(delta).

eta = 0.1 and delta = 0.05 are fixed by the brief (§0, §8.1).
"""

import numpy as np
from scipy.stats import chi2

ETA = 0.1
DELTA = 0.05


def q(d, delta=DELTA):
    return float(np.sqrt(chi2.ppf(1.0 - delta, d)))


def S_lo(mu_lo, d, delta=DELTA):
    return float(np.sqrt(mu_lo)) * q(d, delta)


def S_hi(mu_hi, d, delta=DELTA):
    return float(np.sqrt(mu_hi)) * q(d, delta) if np.isfinite(mu_hi) else np.inf


def global_lower_bounds(H, g, steps=(1, 5)):
    """Gauss-rule lower bounds on ||e||_H^2 = g^T H^-1 g (Proposition 5.7(b)): Lanczos on g with full
    reorthogonalization; the l-step bound is ||g||^2 e_1^T T_l^-1 e_1. The 1-step value is
    (g^T g)^2 / (g^T H g)."""
    out = {}
    nrm = np.linalg.norm(g)
    if nrm == 0:
        return {s: 0.0 for s in steps}
    Q = [g / nrm]
    alpha, beta = [], []
    lmax = max(steps)
    for k in range(lmax):
        w = H @ Q[k]
        a = float(Q[k] @ w)
        alpha.append(a)
        w = w - a * Q[k] - (beta[-1] * Q[k - 1] if k > 0 else 0.0)
        Qm = np.array(Q)
        for _ in range(2):
            w = w - Qm.T @ (Qm @ w)
        bnext = float(np.linalg.norm(w))
        T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
        if (k + 1) in steps:
            e1 = np.zeros(k + 1)
            e1[0] = 1.0
            out[k + 1] = float(nrm ** 2 * np.linalg.solve(T, e1)[0])
        if bnext <= 1e-14 * abs(a):
            for s in steps:
                if s > k + 1:
                    e1 = np.zeros(k + 1)
                    e1[0] = 1.0
                    out[s] = float(nrm ** 2 * np.linalg.solve(T, e1)[0])
            break
        beta.append(bnext)
        Q.append(w / bnext)
    return out
