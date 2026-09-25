"""Brief §3 unit tests (Lemmas 1.2-1.6) and fact F4 (errata 2026-09-25 (i)) on G-solver's own GP chain."""

import numpy as np
import pytest

from certify import chain as ch
from certify import synth
from certify.covbounds import psd_violation
from certify.model import Model


def random_chain_model(n, seed, dims=(3, 6, 12)):
    """Block-tridiagonal SPD model built from factors: a random pairwise factor on every link (k, k+1)
    (kind "gp", so the chain structure is known), random unary factors, random sizes."""
    rng = np.random.default_rng(seed)
    d = rng.choice(dims, size=n)
    factors = []
    for k in range(n - 1):
        r = int(rng.integers(max(d[k], d[k + 1]), d[k] + d[k + 1] + 1))
        Q = rng.standard_normal((r, r))
        factors.append(dict(kind="gp", vars=np.array([[k, k + 1]]), J=rng.standard_normal((1, r, d[k] + d[k + 1])),
                            Lam=(Q @ Q.T / r + 0.2 * np.eye(r))[None], r=rng.standard_normal((1, r))))
    for k in range(n):
        s = 10 ** rng.uniform(-2, 1)
        factors.append(dict(kind="unary", vars=np.array([[k]]), J=rng.standard_normal((1, d[k], d[k])),
                            Lam=(s * np.eye(d[k]))[None], r=rng.standard_normal((1, d[k]))))
    return Model(d, np.array(["pose"] * n), factors, pose_index=np.arange(n))


def blocks_of(model):
    H = model.H.toarray()
    sl = [model.block_slice(k) for k in range(model.n_blocks)]
    Pd = [H[s, s] for s in sl]
    Po = [H[sl[k], sl[k + 1]] for k in range(model.n_blocks - 1)]
    return H, sl, Pd, Po


def seeds(model, n):
    """Lemma 1.4 seeds: P_mm minus the diagonal contribution of the link factor (m, m+1) (backward) or
    (m-1, m) (forward)."""
    H, sl, Pd, Po = blocks_of(model)
    back, fwd = [P.copy() for P in Pd], [P.copy() for P in Pd]
    for f in model.factors:
        if f["kind"] != "gp":
            continue
        for row, (a, b) in enumerate(f["vars"]):
            Hf = f["Hf"][row]
            da = int(model.dims[a])
            back[a] -= Hf[:da, :da]
            fwd[b] -= Hf[da:, da:]
    return back, fwd


@pytest.mark.parametrize("seed", range(5))
def test_lemma_1_2_product_formula(seed):
    m = random_chain_model(12, seed)
    H, sl, Pd, Po = blocks_of(m)
    Sigma = np.linalg.inv(H)
    G, F = ch.backward_schur(Pd, Po), ch.forward_schur(Pd, Po)
    Psi, Phi = ch.transfer_backward(G, Po), ch.transfer_forward(F, Po)
    Sd = ch.diag_marginals(Pd, Po, G, F)
    for i in range(m.n_blocks):
        assert np.linalg.norm(Sd[i] - Sigma[sl[i], sl[i]]) <= 1e-10 * np.linalg.norm(Sigma[sl[i], sl[i]])
        for j in range(m.n_blocks):
            ref = Sigma[sl[i], sl[j]]
            if j > i:
                got = ch.cov_product(Sigma[sl[i], sl[i]], Psi, i, j)
            elif j < i:
                got = ch.cov_product_left(Sigma[sl[i], sl[i]], Phi, i, j)
            else:
                continue
            assert np.linalg.norm(got - ref) <= 1e-10 * max(np.linalg.norm(Sigma[sl[i], sl[i]]), np.linalg.norm(ref))


@pytest.mark.parametrize("seed", range(5))
def test_lemma_1_3_1_4_truncated_lower_bounds(seed):
    m = random_chain_model(15, seed)
    H, sl, Pd, Po = blocks_of(m)
    G, F = ch.backward_schur(Pd, Po), ch.forward_schur(Pd, Po)
    back, fwd = seeds(m, m.n_blocks)
    C = [np.linalg.cholesky(P) for P in Pd]
    for mm in range(m.n_blocks):
        assert psd_violation(C[mm], back[mm], G[mm]) <= 1e-10  # Lemma 1.4
        L = ch.truncated_backward(Pd, Po, mm, back[mm])
        for k in L:
            assert psd_violation(C[k], L[k], G[k]) <= 1e-10  # Lemma 1.3
        Lf = ch.truncated_forward(Pd, Po, mm, fwd[mm])
        for k in Lf:
            assert psd_violation(C[k], Lf[k], F[k]) <= 1e-10


@pytest.mark.parametrize("seed", range(5))
def test_lemma_1_6_decay(seed):
    m = random_chain_model(12, seed)
    H, sl, Pd, Po = blocks_of(m)
    Sigma = np.linalg.inv(H)
    back, _ = seeds(m, m.n_blocks)
    n = m.n_blocks
    for window in (1, 2, 4, n):
        psi = np.zeros(n - 1)
        for k in range(n - 1):
            mm = min(k + 1 + window, n - 1)
            L = ch.truncated_backward(Pd, Po, mm, back[mm], k0=k + 1)
            psi[k] = ch.psibar(Pd[k + 1], L[k + 1], Po[k].T, Pd[k])[0]
        for i in range(n):
            lam = ch.lam_max_scaled(Pd[i], Sigma[sl[i], sl[i]])
            for j in range(i + 1, n):
                lhs = ch.scaled_norm(Pd[i], Sigma[sl[i], sl[j]], Pd[j])
                assert lhs <= lam * np.prod(psi[i:j]) * (1 + 1e-9)


def _f4_errors(n, prior_info, seed, eps):
    g = synth.gs()
    g.set_frozen_regularizer(eps)
    try:
        model, gb = synth.wnoa_chain(n, 0.0, seed=seed, pose_priors=prior_info, python_terms=False, return_builder=True)
        for _ in range(4 * n):  # a chain is a tree: synchronous GBP is exact after (diameter) iterations
            gb.solver.perform_iteration(gb.fg)
        H, sl, Pd, Po = blocks_of(model)
        G, F = ch.backward_schur(Pd, Po), ch.forward_schur(Pd, Po)
        rel_G, rel_F, dev = [], [], []
        for kind, f, vs in gb.factors:
            if kind != "gp":
                continue
            a, b = vs
            J = np.asarray(f.J_lin)
            Lf = J.T @ np.linalg.inv(np.asarray(f.Sigma)) @ J
            msg_a = np.asarray(f.edge(0).factor_to_variable_message.Lambda)  # message_{k+1 -> k}
            msg_b = np.asarray(f.edge(1).factor_to_variable_message.Lambda)  # message_{k -> k+1}
            pred_G = H[sl[a], sl[a]] - Lf[:12, :12] + msg_a                  # errata (i)
            pred_F = H[sl[b], sl[b]] - Lf[12:, 12:] + msg_b                  # its mirror
            rel_G.append(np.linalg.norm(pred_G - G[a]) / np.linalg.norm(G[a]))
            rel_F.append(np.linalg.norm(pred_F - F[b]) / np.linalg.norm(F[b]))
            dev.append(max(np.abs(pred_G - G[a]).max(), np.abs(pred_F - F[b]).max()))
            # the literal remark "G_k = P_kk + message" is off by the forward factor's block
        lit = [np.linalg.norm(H[sl[vs[0]], sl[vs[0]]] + np.asarray(f.edge(0).factor_to_variable_message.Lambda) - G[vs[0]])
               / np.linalg.norm(G[vs[0]]) for kind, f, vs in gb.factors if kind == "gp"]
        return np.array(rel_G), np.array(rel_F), np.array(dev), np.array(lit)
    finally:
        g.set_frozen_regularizer(1e-6)


@pytest.mark.parametrize("n,prior_info,seed", [(20, 0.01, 0), (40, 0.1, 1), (60, 1.0, 2), (40, 10.0, 3)])
def test_f4_corrected_identity_exact(n, prior_info, seed):
    """G_k = (H_kk - Lambda^f_kk) + message_{k+1->k} with G-solver's frozen GBP messages on its own GP
    chain; G-solver's four regularizers set to 1e-12 so the identity is tested exactly (to O(1e-12))."""
    rel_G, rel_F, dev, lit = _f4_errors(n, prior_info, seed, 1e-12)
    assert rel_G.max() <= 1e-10, rel_G.max()
    assert rel_F.max() <= 1e-10, rel_F.max()
    assert np.median(lit) > 1e-3  # the uncorrected remark is visibly wrong


@pytest.mark.parametrize("n,prior_info,seed", [(40, 0.1, 1), (60, 1.0, 2)])
def test_f4_identity_as_shipped(n, prior_info, seed):
    """With the shipped regularizers (1e-6 at all four sites) the identity holds up to O(eps) (decision D1)."""
    rel_G, rel_F, dev, lit = _f4_errors(n, prior_info, seed, 1e-6)
    print("F4 as shipped: max abs deviation = %.2f eps, max rel = %.2e" % (dev.max() / 1e-6, max(rel_G.max(), rel_F.max())))
    assert dev.max() <= 1e3 * 1e-6
