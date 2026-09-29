"""Stage 2c, Step 4 (BRIEF_stage2c.md §4): the full nonlinear check at a corrected-variant certified stop on the clean model.

The zero x^ of the solver's estimating equation on the full clean model (poses and remaining landmarks, corrected Jacobians,
the anchor prior and the weak landmark priors p_L), F(z) = sum_f J_f(x)' Lambda_f e_f(x) + sum_l p_L (z_l - a_l) at
x = x_c (+) z (certify/t1.py; exact derivatives from `certify_stereo --t1-export`), by Newton's method from the stop state
z = 0 with direct solves (the landmark blocks eliminated exactly, a dense solve on the poses), until ||dz|| <= 1e-12 ||z||
or 50 iterations.

Reported: iterations, final residual; per pose ||x^_i (-) x_c,i||_{D_i} (D_i = S_ii of the stop's linearization, the
certificate's metric), its ratio to eta S_lo,i and to the certified bound E_i; landmark moves in the D-metric (D_l = the
landmark's diagonal block of the Gauss-Newton matrix at x_c) and in metres, separately for landmarks with two observations
and with more (observations: stereo factors in the clean graph). Also F(0) against the solver's own J' Lambda e at x_c.
With no certified stop in the run, newton.json records that and nothing is computed.

Usage: python3 scripts/stage2c_newton.py <dataset> [run dir, default exports/stage2c/<dataset>/corrected]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from certify.io import Export  # noqa: E402
from certify.sidecar_model import PoseModelBuilder  # noqa: E402
from certify.t1 import T1Client, Problem, schur_solve  # noqa: E402

ETA, DOF = 0.1, 12


def q(a):
    a = np.asarray(a, float)
    if not len(a):
        return dict(n=0)
    return dict(n=int(len(a)), median=float(np.median(a)), p90=float(np.percentile(a, 90)), p99=float(np.percentile(a, 99)),
                max=float(np.max(a)))


def main(ds, run=None):
    t0 = time.perf_counter()
    run = Path(run) if run else REPO / "exports/stage2c" / ds / "corrected"
    ev = run / "eval"
    out = REPO / "results/stage2c" / ds
    out.mkdir(parents=True, exist_ok=True)
    if not (ev / "certified_stop_patches.json").exists():
        rounds = max(int(l.split(",")[0]) for l in open(ev / "eval_summary.csv").read().splitlines()[1:])
        res = dict(dataset=ds, run=str(run.relative_to(REPO)), certified_stop=None,
                   note="no certified stop by round %d: there is no stop state to check" % rounds)
        (out / "newton.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res))
        return
    stop = json.load(open(ev / "certified_stop_patches.json"))
    exm = Export(ev / "certified_stop_model")
    m = PoseModelBuilder(p_L=1e-6, g2o=str(REPO / exm.meta["g2o"])).build(None, ex=exm)
    S = m["S"]
    ids = [str(p) for p in m["pose_ids"]]
    Cp = [np.linalg.cholesky(0.5 * (S[DOF * i:DOF * i + DOF, DOF * i:DOF * i + DOF] + S[DOF * i:DOF * i + DOF, DOF * i:DOF * i + DOF].T))
          for i in range(len(ids))]
    ed = ev / "certified_stop_beliefs"
    pb = Problem(ed)
    ex = pb.ex
    assert [str(p) for p in ex.var_id[pb.fm.poses]] == ids
    assert "prior" in pb.fac and pb.fac["prior"]["keep"].sum() == 1, "the clean model's anchor is missing"
    npd = int(pb.boff[pb.npose])
    lblocks = [(int(pb.boff[k]), int(pb.boff[k + 1] - pb.boff[k])) for k in range(pb.npose, len(pb.blocks))]
    cl = T1Client(ed, f"/dev/shm/gbp_os717_t1c_{ds}", true_jacobians=True)
    assert "prior" in cl.types
    z = np.zeros(pb.n)
    hist, converged = [], False
    F0 = None
    for it in range(1, 51):
        ts = time.perf_counter()
        F, dF, flags = pb.full(z, cl)
        if F0 is None:
            F0 = F.copy()
        dz = schur_solve(dF, -F, npd, lblocks)
        z = z + dz
        rel = np.linalg.norm(dz) / max(np.linalg.norm(z), 1e-300)
        hist.append(dict(iteration=it, residual_before=float(np.linalg.norm(F)), step=float(np.linalg.norm(dz)), rel_step=float(rel),
                         flags=int(flags), seconds=time.perf_counter() - ts))
        print("newton %d: |F| %.3e |dz| %.3e rel %.3e flags %d (%.1fs)" % (it, np.linalg.norm(F), np.linalg.norm(dz), rel, flags,
                                                                         hist[-1]["seconds"]), flush=True)
        if rel <= 1e-12:
            converged = True
            break
    Ffin, _, flfin = pb.full(z, cl)
    cl.close()

    # F(0) against the solver's J' Lambda e at x_c (FullModel's gradient at the belief export, same model and pseudo-factors)
    g_model = pb.fm.g[np.concatenate([np.arange(ex.offsets[b], ex.offsets[b] + ex.dims[b]) for b in pb.blocks])]
    f0_gap = float(np.abs(F0 - g_model).max() / max(np.abs(g_model).max(), 1e-300))

    pose_move = np.array([float(np.linalg.norm(Cp[i].T @ z[pb.boff[i]:pb.boff[i + 1]])) for i in range(len(ids))])
    slo = np.array([stop["patches"][p]["S_lo"] for p in ids])
    E = np.array([stop["patches"][p]["E"] for p in ids])
    within = pose_move <= ETA * slo
    H = pb.fm.H
    lm_D, lm_m, nobs = [], [], []
    for k in range(pb.npose, len(pb.blocks)):
        b = pb.blocks[k]
        sl = slice(ex.offsets[b], ex.offsets[b] + ex.dims[b])
        Dl = H[sl][:, sl].toarray()
        Cl = np.linalg.cholesky(0.5 * (Dl + Dl.T))
        zl = z[pb.boff[k]:pb.boff[k + 1]]
        lm_D.append(float(np.linalg.norm(Cl.T @ zl)))
        lm_m.append(float(np.linalg.norm(zl)))
        nobs.append(int(ex.degree[b]))
    lm_D, lm_m, nobs = map(np.array, (lm_D, lm_m, nobs))
    worst = int(np.argmax(pose_move / (ETA * slo)))
    res = dict(dataset=ds, run=str(run.relative_to(REPO)), certified_stop=stop["round"], poses=len(ids), landmarks=len(pb.blocks) - pb.npose,
               iterations=len(hist), converged=converged, final_residual=float(np.linalg.norm(Ffin)),
               final_residual_max_abs=float(np.abs(Ffin).max()), final_flags=int(flfin), F0_vs_solver_rel=f0_gap, history=hist,
               pose_move_D=q(pose_move), pose_move_over_eta_S_lo=q(pose_move / (ETA * slo)), pose_move_over_E=q(pose_move / E),
               fraction_within_eta_S_lo=float(np.mean(within)), poses_outside=[ids[i] for i in np.flatnonzero(~within)],
               worst_pose=dict(pose=ids[worst], move_D=float(pose_move[worst]), over_eta_S_lo=float(pose_move[worst] / (ETA * slo[worst])),
                               over_E=float(pose_move[worst] / E[worst])),
               landmarks_two_observations=dict(count=int(np.sum(nobs == 2)), move_D=q(lm_D[nobs == 2]), move_m=q(lm_m[nobs == 2])),
               landmarks_more_observations=dict(count=int(np.sum(nobs > 2)), move_D=q(lm_D[nobs > 2]), move_m=q(lm_m[nobs > 2])),
               landmarks_over_0_1m=int(np.sum(lm_m > 0.1)), seconds=time.perf_counter() - t0)
    (out / "newton.json").write_text(json.dumps(res, indent=1))
    ez = REPO / "exports/stage2c" / ds
    ez.mkdir(parents=True, exist_ok=True)
    np.save(ez / "z_star.npy", z)
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
