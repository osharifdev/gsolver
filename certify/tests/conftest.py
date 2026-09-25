import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # G-solver's OpenMP loops on toy graphs

import numpy as np
import pytest

from certify import check, synth


def suite():
    """Small instances of every Phase B problem family (the full sweep is scripts/phaseB_run.py)."""
    return {
        "chain_pure_n20": dict(model=lambda: synth.wnoa_chain(20, 0.0, seed=0)),
        "chain_meas_n30": dict(model=lambda: synth.wnoa_chain(30, 1.0, seed=1)),
        "chain_blind_n50": dict(model=lambda: synth.wnoa_chain(50, 1.0, seed=2, blind=(15, 20))),
        "chain_hub_n24": dict(model=lambda: synth.wnoa_chain(24, 0.1, seed=3, hub=1.0)),
        "chain_tracks_n24": dict(model=lambda: synth.wnoa_chain(24, 0.1, seed=4, short_tracks=(2, 1.0))),
        "chain_tracks_meas_n24": dict(model=lambda: synth.wnoa_chain(24, 1.0, seed=7, short_tracks=(2, 1.0))),
        "chain_hub_meas_n24": dict(model=lambda: synth.wnoa_chain(24, 1.0, seed=8, hub=1.0)),
        "random_spd_30a": dict(model=lambda: synth.random_spd(30, seed=5)),
        "random_spd_30b": dict(model=lambda: synth.random_spd(30, seed=6, dims=(3, 12))),
    }


_CACHE = {}


def evaluated(name):
    if name not in _CACHE:
        spec = suite()[name]
        model = spec["model"]()
        _, _, _, xbar = check.dense_oracle(model)
        rng = np.random.default_rng(100)
        xs = {}
        for kind in ("gs", "jacobi"):
            its = synth.iterates(model, xbar, rng, kind=kind, keep=(0, 1, 2, 5, 10, 20, 50))
            for t, x in its.items():
                xs[(kind, t)] = x
        keys = sorted(xs)
        xs_idx = {k: xs[keys[k]] for k in range(len(keys))}
        window = None
        if name == "chain_hub_n24":
            # Lemma 4.3 window: the anchor pose and the first 8 observers (+ factors among them)
            window = np.flatnonzero((model.kinds == "pose") & (model.pose_index < 8))
        chk, info = check.evaluate(model, xs_idx, label=name, hub_window=window)
        _CACHE[name] = (model, chk, info, keys)
    return _CACHE[name]


@pytest.fixture(scope="session", params=list(suite()))
def ev(request):
    return evaluated(request.param)
