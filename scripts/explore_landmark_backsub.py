"""Landmark certificates from pose certificates (back-substitution, exact identity + triangle inequality):

    e_l = H_ll^-1 (g_l - sum_{p in O(l)} H_lp e_p)            (landmark row of H e = g, full model)
    ||e_l||_{D_l} <= ||C_l^-1 g_l|| + sum_p ||C_l^-1 H_lp C_{S,p}^-T|| E_p,

with D_l = H_ll = C_l C_l^T (full model), E_p >= ||e_p||_{S_pp} the reduced-model pose certificate and
S_pp = C_{S,p} C_{S,p}^T. Uses the pose certificates written by scripts/phaseC_reduced.py (C3, K = 1, 2).
Reports validity against the full model's exact landmark errors and the tightness ratio distribution.

Usage: python3 scripts/explore_landmark_backsub.py dataset
"""

import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as sla

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from certify import certs, phaseC  # noqa: E402
from certify import model as mdl  # noqa: E402
from certify.linalg import SPDSystem  # noqa: E402


def main(name):
    path, cfg = phaseC.DATASETS[name]
    full, info = mdl.from_export(path, p_L=cfg["p_L"])
    red, kept = mdl.reduce_blocks(full)
    rows = list(csv.reader(gzip.open(phaseC.OUT / "reduced" / ("%s_certs.csv.gz" % name), "rt")))
    hdr = rows[0]
    ix = {h: k for k, h in enumerate(hdr)}
    iters = sorted({int(r[ix["iter"]]) for r in rows[1:]})
    Ks = sorted({int(r[ix["K"]]) for r in rows[1:]})
    bid = {str(b): k for k, b in enumerate(red.block_ids)}
    Ep = {K: np.full((red.n_blocks, len(iters)), np.nan) for K in Ks}
    for r in rows[1:]:
        Ep[int(r[ix["K"]])][bid[r[ix["block_id"]]], iters.index(int(r[ix["iter"]]))] = float(r[ix["E_C3"]])
    X = phaseC.load_states(full, info, iters)
    G = full.residual(X)
    sysm = SPDSystem(full.H, full.dims, full.elim, refine=3, K_lo=full.H_lo)
    E = sysm.solve(G)
    metF = certs.Metric(full)
    errF = metF.norms(E)
    CS = [np.linalg.cholesky(D) for D in red.diag_blocks()]
    red_of_full = -np.ones(full.n_blocks, int)
    red_of_full[kept] = np.arange(len(kept))
    lms = np.flatnonzero(full.kinds == "landmark")
    H = full.H.tocsr()
    out = dict(dataset=name, iters=iters, K={})
    for K in Ks:
        Eb = np.full((len(lms), len(iters)), np.nan)
        for k, l in enumerate(lms):
            sl = full.block_slice(l)
            Cl = metF.C[l]
            s = np.linalg.norm(sla.solve_triangular(Cl, G[sl], lower=True), axis=0)
            tot = s.copy()
            for p in full.neighbours(l):
                if full.kinds[p] != "pose":
                    continue
                Hlp = H[sl][:, full.block_slice(p)].toarray()
                Y = sla.solve_triangular(Cl, Hlp, lower=True)
                Y = sla.solve_triangular(CS[red_of_full[p]], Y.T, lower=True).T
                tot = tot + float(np.linalg.norm(Y, 2)) * Ep[K][red_of_full[p]]
            Eb[k] = tot
        u = errF[lms]
        ok = u >= 1e-12
        valid = bool(np.all(Eb >= u * (1 - 1e-9)))
        rat = (Eb / np.where(ok, u, np.nan))
        out["K"][K] = dict(valid=valid, worst_violation=float(np.nanmax((u - Eb) / np.maximum(Eb, 1e-300))),
                           ratio_median=[float(np.nanmedian(rat[:, j])) for j in range(len(iters))],
                           ratio_p95=[float(np.nanpercentile(rat[:, j], 95)) for j in range(len(iters))],
                           E_median=[float(np.median(Eb[:, j])) for j in range(len(iters))],
                           err_median=[float(np.median(u[:, j])) for j in range(len(iters))])
        print("K=%d: landmark certificates valid everywhere: %s (worst relative violation %.2e); ratio median per iteration %s; p95 %s" % (
            K, valid, out["K"][K]["worst_violation"], np.round(out["K"][K]["ratio_median"], 2), np.round(out["K"][K]["ratio_p95"], 1)), flush=True)
    print("iterations:", iters)
    d = phaseC.OUT / "explore"
    d.mkdir(parents=True, exist_ok=True)
    (d / ("landmarks_%s.json" % name)).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
