"""Brief §5 unit tests: C1-C4 invariants on every synthetic family and iterate; C2 reduces to C1 at
Pbb = D; C2 is exact on the pure chain; C3 with a patch equal to the whole graph is exact."""

import numpy as np
import pytest

from certify import certs, check, synth


def _rows(chk, prefix):
    return [r for r in chk.rows if r["check"].startswith(prefix)]


def test_c1_invariant(ev):
    model, chk, info, keys = ev
    for r in _rows(chk, "C1 "):
        assert r["ok"], r


def test_c2_invariant(ev):
    model, chk, info, keys = ev
    for r in _rows(chk, "C2 "):
        assert r["ok"], r


def test_c3_invariant(ev):
    model, chk, info, keys = ev
    rows = _rows(chk, "C3 ")
    for r in rows:
        assert r["ok"], r


def test_c4_invariant(ev):
    model, chk, info, keys = ev
    for r in _rows(chk, "C4 ") + _rows(chk, "Prop3.6"):
        assert r["ok"], r


def test_every_check_passes(ev):
    model, chk, info, keys = ev
    bad = chk.failed()
    assert not bad, bad


@pytest.mark.parametrize("seed", range(3))
def test_c2_reduces_to_c1(seed):
    """Setting Pbb = D reproduces C1 exactly (brief §5 C2)."""
    model = synth.wnoa_chain(25, 0.5, seed=seed, short_tracks=(1, 1.0))
    _, _, _, xbar = check.dense_oracle(model)
    xs = synth.iterates(model, xbar, np.random.default_rng(seed), kind="gs", keep=(1, 5))
    G = model.residual(np.stack(list(xs.values()), 1))
    met = certs.Metric(model)
    r1 = certs.c1(model, met, G)
    r2 = certs.c2(model, met, G, bb=certs.Backbone(model, use_chain=False))
    M1, M2 = r1["M"].toarray(), r2["M"].toarray()
    assert np.abs(M1 - M2).max() <= 1e-12 * max(M1.max(), 1e-300)
    assert abs(r1["rho"] - r2["rho"]) <= 1e-10 * max(r1["rho"], 1e-300)
    np.testing.assert_allclose(r1["nu"], r2["nu"], rtol=1e-12, atol=0)
    if r1["E"] is not None:
        np.testing.assert_allclose(r1["E"], r2["E"], rtol=1e-9, atol=0)


@pytest.mark.parametrize("n,m_over_p", [(20, 0.0), (50, 0.0), (50, 1.0), (100, 0.1)])
def test_c2_exact_on_pure_chain(n, m_over_p):
    """No landmark couplings: Mhat = 0 and E_i = ||[Pbb^-1 g]_i||_{D_i} equals the true error (ratio 1 +- 1e-8)."""
    model = synth.wnoa_chain(n, m_over_p, seed=n)
    _, osys, _, xbar = check.dense_oracle(model)
    xs = synth.iterates(model, xbar, np.random.default_rng(n), kind="jacobi", keep=(0, 1, 5, 20))
    X = np.stack(list(xs.values()), 1)
    G = model.residual(X)
    met = certs.Metric(model)
    r2 = certs.c2(model, met, G)
    assert r2["M"].nnz == 0 and r2["rho"] == 0.0
    u = met.norms(osys.solve(G))
    ok = u > 1e-12
    ratio = r2["E"][ok] / u[ok]
    assert np.abs(ratio - 1).max() <= 1e-8, np.abs(ratio - 1).max()


@pytest.mark.parametrize("name", ["chain_tracks", "random"])
def test_c3_whole_graph_patch_is_exact(name):
    """Brief §7: with a patch equal to the whole graph the local solve is the oracle error (ratio 1)."""
    model = synth.wnoa_chain(12, 0.1, seed=9, short_tracks=(1, 1.0)) if name == "chain_tracks" else synth.random_spd(15, seed=9)
    _, osys, _, xbar = check.dense_oracle(model)
    xs = synth.iterates(model, xbar, np.random.default_rng(1), kind="gs", keep=(1, 3))
    X = np.stack(list(xs.values()), 1)
    G = model.residual(X)
    met = certs.Metric(model)
    K = model.n_blocks  # every patch is the whole (connected) graph
    P = certs.patch_pass(model, met, K, G)
    r3 = certs.c3_from_patches(model, P)
    assert r3["A"].nnz == 0
    u = met.norms(osys.solve(G))
    np.testing.assert_allclose(r3["E"], u, rtol=1e-8, atol=1e-14)


def test_one_step_inequalities(ev):
    """u <= nu + M u (C1), u <= nuhat + Mhat u (C2), u <= s + A u (C3, Theorem 3.5), u~ <= s + A u~ (C4):
    these hold whatever rho is, so they test the coefficients even where a certificate is vacuous."""
    model, chk, info, keys = ev
    rows = _rows(chk, "C1 one-step") + _rows(chk, "C2 one-step") + _rows(chk, "C3 one-step") + _rows(chk, "C4 one-step")
    assert rows
    for r in rows:
        assert r["ok"], r


@pytest.mark.parametrize("K", [1, 2])
def test_c3_coefficients_brute_force(K):
    """s_i and a_ij of Definition 3.4 against a dense evaluation of the formulas."""
    model = synth.wnoa_chain(10, 0.5, seed=3, short_tracks=(1, 1.0))
    _, osys, _, xbar = check.dense_oracle(model)
    x = synth.iterates(model, xbar, np.random.default_rng(2), kind="gs", keep=(2,))[2]
    g = model.residual(x)
    met = certs.Metric(model)
    H = model.H.toarray()
    P = certs.patch_pass(model, met, K, g[:, None])
    for p in P:
        i = p["i"]
        blocks = model.ball(i, K)
        dofs = model.dof_index(blocks)
        Hinv = np.linalg.inv(H[np.ix_(dofs, dofs)])
        loc = {b: np.searchsorted(dofs, model.dof_index([b])) for b in blocks}
        Ci = met.C[i]
        s_ref = np.linalg.norm(Ci.T @ (Hinv @ g[dofs])[loc[i]])
        assert abs(s_ref - p["s"][0]) <= 1e-9 * max(s_ref, 1e-300)
        inP = set(blocks.tolist())
        bnd = [b for b in blocks if any(j not in inP for j in model.neighbours(b))]
        ext = sorted({j for b in bnd for j in model.neighbours(b) if j not in inP})
        assert sorted(p["a_cols"].tolist()) == ext
        for j, a in zip(p["a_cols"], p["a_vals"]):
            X = sum(Hinv[np.ix_(loc[i], loc[b])] @ H[np.ix_(model.dof_index([b]), model.dof_index([j]))] for b in bnd)
            ref = np.linalg.norm(Ci.T @ X @ np.linalg.inv(met.C[j]).T, 2)
            assert abs(ref - a) <= 1e-9 * max(ref, 1e-300)


@pytest.mark.parametrize("seed", range(3))
def test_reduce_blocks_is_exact_schur(seed):
    """Landmark elimination: H_red = H_pp - H_pl H_ll^-1 H_lp and b_red = b_p - H_pl H_ll^-1 b_l, so the reduced
    model's solution is the full model's pose solution."""
    from certify.model import reduce_blocks
    model = synth.wnoa_chain(20, 0.5, seed=seed, short_tracks=(2, 1.0))
    # give the points a weak prior like the stereo model's p_L (a unary factor on each point)
    pts = np.flatnonzero(model.kinds == "landmark")
    rng = np.random.default_rng(seed)
    factors = model.factors + [dict(kind="landmark_prior", vars=pts[:, None], J=np.repeat(np.eye(3)[None], len(pts), 0),
                                    Lam=np.repeat((1e-3 * np.eye(3))[None], len(pts), 0), r=rng.standard_normal((len(pts), 3)))]
    from certify.model import Model
    full = Model(model.dims, model.kinds, [dict(f) for f in factors], pose_index=model.pose_index, elim=model.kinds == "landmark")
    red, kept = reduce_blocks(full)
    H = full.H.toarray()
    kd = full.dof_index(kept)
    ld = full.dof_index(np.flatnonzero(full.kinds == "landmark"))
    S = H[np.ix_(kd, kd)] - H[np.ix_(kd, ld)] @ np.linalg.solve(H[np.ix_(ld, ld)], H[np.ix_(ld, kd)])
    bS = full.b[kd] - H[np.ix_(kd, ld)] @ np.linalg.solve(H[np.ix_(ld, ld)], full.b[ld])
    Hr = red.H.toarray()
    assert np.abs(Hr - S).max() <= 1e-9 * np.abs(S).max()
    assert np.abs(red.b - bS).max() <= 1e-9 * np.abs(bS).max()
    xf = np.linalg.solve(H, full.b)[kd]
    xr = np.linalg.solve(Hr, red.b)
    assert np.abs(xf - xr).max() <= 1e-8 * max(np.abs(xf).max(), 1e-300)
