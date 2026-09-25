"""Brief §6 unit tests: Theorem 4.2 decomposition (1e-10), Lemma 4.3 (local hub precision below the exact
one), Lemma 4.4 (hub error bound), Lemma 4.5 (influence coefficient), Theorem 4.6 (hub-corrected
certificate), Proposition 4.7, the block-tridiagonal structure of H_TT (Lemma 4.1), and the one-at-a-time
hub construction (certify/hubs.py docstring)."""

import numpy as np
import pytest

from certify import hubs, synth
from certify.tests.conftest import evaluated


def _rows(chk, prefix):
    return [r for r in chk.rows if r["check"].startswith(prefix)]


@pytest.mark.parametrize("name", ["chain_hub_n24", "chain_tracks_n24", "chain_hub_meas_n24", "chain_tracks_meas_n24"])
def test_hub_theorems(name):
    model, chk, info, keys = evaluated(name)
    assert len(info["partition"].S) > 0
    names = ["Thm4.2", "Lemma4.4", "Lemma4.5", "Thm4.6", "Prop4.7", "m_SJ", "sec4.7", "||e_S||"]
    if name == "chain_hub_n24":
        names.append("Lemma4.3")
    for nm in names:
        rows = _rows(chk, nm)
        assert rows, nm
        for r in rows:
            assert r["ok"], r


@pytest.mark.parametrize("name", ["chain_hub_n24", "chain_tracks_n24", "chain_hub_meas_n24", "chain_tracks_meas_n24"])
def test_one_hub_at_a_time(name):
    """Available only where the hub blocks' C3 certificate exists (rho(A) < 1 at some K)."""
    model, chk, info, keys = evaluated(name)
    rows = _rows(chk, "one-at-a-time") + _rows(chk, "Lemma4.5 single hub")
    c3_valid = any(r["E"] is not None for r in info["C3"].values())
    assert bool(rows) == c3_valid
    for r in rows:
        assert r["ok"], r


@pytest.mark.parametrize("w", [1, 2, 3, 5])
def test_lemma_4_1_block_tridiagonal(w):
    model = synth.wnoa_chain(30, 0.1, seed=11, short_tracks=(2, 1.0))
    part = hubs.Partition(model, w)
    assert part.max_gap <= 1
    # every short-track landmark sits in the earlier supernode it touches
    for j in np.flatnonzero((model.kinds == "landmark") & ~part.hub):
        obs = [k for k in model.neighbours(j) if model.kinds[k] == "pose"]
        assert part.sn[j] == min(part.sn[k] for k in obs)
        assert part.span[j] <= w
    assert np.all(part.span[part.S] > w)
