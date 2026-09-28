"""Stage 2b: diagnosis of the T1 attempts against the Newton reference (diagnosis only; nothing here sets a radius).

Per stop state:
  poses      true distance to x^ against Stage 1's E_i (the c a pose box would need to contain x^), and against eta S_lo
  landmarks  true move to x^ (D_l-norm) against option A's and option B's radius at c = 1 and 16; the split between the
             single-edge landmarks of D2 (pinned by their frozen message, precision >= 1e3) and the others
Writes results/stage2b/<dataset>/diagnosis.json.

Usage: python3 scripts/stage2b_diagnose.py <dataset>
"""

import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as sla

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from certify.t1 import Problem  # noqa: E402
from stage2b_t1 import stage2a_candidate  # noqa: E402

ETA, DOF = 0.1, 12


def q(a):
    a = np.asarray(a, float)
    return dict(n=int(len(a)), median=float(np.median(a)), p90=float(np.percentile(a, 90)), max=float(np.max(a))) if len(a) else dict(n=0)


def main(ds):
    exm, fm_lin, stop, ids, Cp, step, unclipped = stage2a_candidate(ds)
    pb = Problem(REPO / "exports/stage1" / ds / "corrected_stage2a" / "eval" / "certified_stop_beliefs")
    ex = pb.ex
    z = np.load(REPO / f"exports/stage2b/{ds}/z_star.npy")
    E = np.array([stop["patches"][p]["E"] for p in ids])
    S_lo = np.array([stop["patches"][p]["S_lo"] for p in ids])
    td = np.array([np.linalg.norm(Cp[i].T @ z[pb.boff[i]:pb.boff[i + 1]]) for i in range(len(ids))])
    res = dict(dataset=ds, poses=dict(true_over_E=q(td / E), c_needed_for_all_poses=float(np.max(td / E)),
                                      true_over_eta_S_lo=q(td / (ETA * S_lo)), finished=float(np.mean(td <= ETA * S_lo))))
    npose, nb = pb.npose, len(pb.blocks)
    C = []
    for k in range(npose, nb):
        b = pb.blocks[k]
        sl = slice(ex.offsets[b], ex.offsets[b] + ex.dims[b])
        Dl = pb.fm.H[sl][:, sl].toarray()
        C.append(np.linalg.cholesky(0.5 * (Dl + Dl.T)))
    Cpinv = [sla.solve_triangular(c, np.eye(DOF), lower=True) for c in Cp]
    single = np.array([float(np.max(np.linalg.eigvalsh(pb.fm.pseudo_of[int(b)][0]))) >= 1e3 for b in pb.blocks[npose:]])
    move_D, move_m, uA, EB = [], [], [], []
    H0 = fm_lin.H
    for k, l in enumerate(range(npose, nb)):
        b = int(pb.blocks[l])
        sl = slice(ex.offsets[b], ex.offsets[b] + 3)
        zl = z[pb.boff[l]:pb.boff[l + 1]]
        move_D.append(float(np.linalg.norm(C[k].T @ zl)))
        move_m.append(float(np.linalg.norm(zl)))
        uA.append(0.1 * float(np.sqrt(np.linalg.eigvalsh(C[k] @ C[k].T)[-1])))
        Cli = np.linalg.inv(C[k])
        val = float(np.linalg.norm(Cli @ fm_lin.g[sl]))
        row = H0[sl].tocsr()
        for i in range(npose):
            pbk = pb.blocks[i]
            blk = row[:, ex.offsets[pbk]:ex.offsets[pbk] + DOF]
            if blk.nnz:
                val += float(np.linalg.norm(Cli @ blk.toarray() @ Cpinv[i].T, 2)) * E[i]
        EB.append(val)
    move_D, move_m, uA, EB = map(np.array, (move_D, move_m, uA, EB))
    lm = {}
    for name, sel in (("single_edge", single), ("multi_observation", ~single), ("all", np.ones(len(single), bool))):
        lm[name] = dict(count=int(sel.sum()), move_m=q(move_m[sel]), move_D=q(move_D[sel]),
                        outside_option_A_c1=int(np.sum(move_D[sel] > uA[sel])), outside_option_A_c16=int(np.sum(move_D[sel] > 16 * uA[sel])),
                        outside_option_B_c1=int(np.sum(move_D[sel] > EB[sel])), outside_option_B_c16=int(np.sum(move_D[sel] > 16 * EB[sel])),
                        option_B_radius_m_equiv_c1=q((EB[sel] / np.sqrt([np.linalg.eigvalsh(C[k] @ C[k].T)[0] for k in np.flatnonzero(sel)]))))
    res["landmarks"] = lm
    out = REPO / "results/stage2b" / ds
    (out / "diagnosis.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
