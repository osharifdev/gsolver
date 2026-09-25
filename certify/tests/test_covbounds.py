"""Brief §4 unit tests: lower bounds (trivial, patch, Krylov) below the oracle marginal and monotone,
upper bounds (boundary cut, hub-aware) above it and monotone, radii, and the global energy bounds of §8.5."""

import numpy as np

from certify import covbounds, radii


def _rows(chk, prefix):
    return [r for r in chk.rows if r["check"].startswith(prefix)]


def test_lower_bounds_below_oracle_and_monotone(ev):
    model, chk, info, keys = ev
    rows = _rows(chk, "patch lower") + _rows(chk, "krylov") + _rows(chk, "trivial lower")
    assert rows
    for r in rows:
        assert r["ok"], r


def test_upper_bounds_above_oracle_and_monotone(ev):
    model, chk, info, keys = ev
    rows = _rows(chk, "cut upper") + _rows(chk, "chain cut") + _rows(chk, "Prop4.7")
    for r in rows:
        assert r["ok"], r


def test_radii_sandwich(ev):
    """S_lo <= S_hi, and mu_lo <= lambda_min, mu_hi >= lambda_max of D^1/2 Sigma D^1/2 (Proposition 2.7)."""
    model, chk, info, keys = ev
    mu = info["mu_oracle"]
    for K, r3 in info["C3"].items():
        for i in range(model.n_blocks):
            C = np.linalg.cholesky(model.diag_blocks()[i])
            lo = covbounds.mu_extremes(C, r3["patch_lower"][i])[0]
            assert lo >= 1 - 1e-9  # patch bound dominates the trivial one (mu_lo >= 1)
            assert lo <= mu[i, 0] * (1 + 1e-9)
            if r3["cut_upper"][i] is not None:
                hi = covbounds.mu_extremes(C, r3["cut_upper"][i])[1]
                assert hi >= mu[i, 1] * (1 - 1e-9)
                d = int(model.dims[i])
                assert radii.S_lo(lo, d) <= radii.S_hi(hi, d)


def test_global_energy_bounds(ev):
    model, chk, info, keys = ev
    for r in _rows(chk, "global"):
        assert r["ok"], r


def test_q_values():
    assert abs(radii.q(12) - np.sqrt(21.02606981748307)) < 1e-12
    assert abs(radii.q(3) - np.sqrt(7.814727903251178)) < 1e-12
