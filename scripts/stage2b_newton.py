"""Stage 2b, Step 1 (BRIEF_stage2b.md §1): the Newton reference x^ at a certified stop.

Newton's method on F (certify/t1.py: the D2 model's estimating equation in the chart at the stop state x_c, exact
derivatives, the corrected Jacobians) with direct solves on the full model (dense Schur complement on the poses, the
landmark blocks eliminated exactly), from z = 0, until ||dz|| <= 1e-12 ||z|| or 50 iterations.

Also the zero of the solver's implemented equation (its own corrected Jacobian code, `certify_stereo --probe-export`):
simplified Newton with the exact F'(z*) from z*, to measure how far the implemented equation's zero is from x^.

Metrics: poses D_i = S_ii of the stop's linearization (the certificate's metric, Stage 1); landmarks D_l = the landmark
diagonal block of the Gauss-Newton matrix at x_c. Writes results/stage2b/<dataset>/newton.json and exports/stage2b/<dataset>/z_star.npy.

Usage: python3 scripts/stage2b_newton.py <dataset>
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from certify.io import Export  # noqa: E402
from certify.nonlinear_probe import ProbeClient  # noqa: E402
from certify.sidecar_model import PoseModelBuilder  # noqa: E402
from certify.t1 import T1Client, Problem, schur_solve  # noqa: E402

ETA = 0.1
DOF = 12


def metrics(ds):
    run = REPO / "exports/stage1" / ds / "corrected_stage2a" / "eval"
    stop = json.load(open(run / "certified_stop_patches.json"))
    exm = Export(run / "certified_stop_model")
    B = PoseModelBuilder(p_L=1e-6, g2o=str(REPO / exm.meta["g2o"]))
    m = B.build(None, ex=exm)
    S = m["S"]
    ids = [str(p) for p in m["pose_ids"]]
    C = [np.linalg.cholesky(0.5 * (S[DOF * i:DOF * i + DOF, DOF * i:DOF * i + DOF] + S[DOF * i:DOF * i + DOF, DOF * i:DOF * i + DOF].T))
         for i in range(len(ids))]
    return run, stop, ids, C


def main(ds):
    t0 = time.perf_counter()
    run, stop, ids, Cp = metrics(ds)
    ed = run / "certified_stop_beliefs"
    pb = Problem(ed)
    assert [str(p) for p in pb.ex.var_id[pb.fm.poses]] == ids
    npd = int(pb.boff[pb.npose])
    lblocks = [(int(pb.boff[k]), int(pb.boff[k + 1] - pb.boff[k])) for k in range(pb.npose, len(pb.blocks))]
    cl = T1Client(ed, f"/dev/shm/gbp_os717_t1_{ds}", true_jacobians=True)
    z = np.zeros(pb.n)
    hist = []
    converged = False
    for it in range(1, 51):
        F, dF, flags = pb.full(z, cl)
        dz = schur_solve(dF, -F, npd, lblocks)
        z = z + dz
        rel = np.linalg.norm(dz) / max(np.linalg.norm(z), 1e-300)
        hist.append(dict(iteration=it, residual_before=float(np.linalg.norm(F)), step=float(np.linalg.norm(dz)), rel_step=float(rel),
                         flags=int(flags)))
        print("newton %d: |F| %.3e |dz| %.3e rel %.3e flags %d" % (it, np.linalg.norm(F), np.linalg.norm(dz), rel, flags), flush=True)
        if rel <= 1e-12:
            converged = True
            break
    Ffin, dFfin, flfin = pb.full(z, cl)
    cl.close()

    # the implemented equation's zero (the solver's own corrected Jacobian code): simplified Newton from z*
    pdir = f"/dev/shm/gbp_os717_probe_{ds}"
    ex = pb.ex
    cmd = [str(REPO / "build-certify/certify/cpp/certify_stereo"), "--g2o", str(REPO / ex.meta["g2o"]), "--gt",
           str(REPO / f"examples/data/datasets/stereo_visual/gt/{Path(ex.meta['g2o']).stem}.tum"), "--keyframes", str(ex.meta["keyframes"]),
           "--config", ex.meta["config"], "--out", pdir + "/out", "--probe-export", str(ed), "--probe-dir", pdir, "--true-jacobians"]
    pc = ProbeClient(cmd, pdir)
    targets = pb.blocks.astype(np.int64)
    tdofs = np.concatenate([np.arange(ex.offsets[b], ex.offsets[b] + ex.dims[b]) for b in pb.blocks])

    def F_solver(zm):
        (Fnl, _, _, _, st), = pc.evaluate([(targets, pb.export_z(zm)[tdofs])])
        Fm = np.zeros(pb.n)
        Fm[pb.dof_of_export[tdofs]] = Fnl.sum(axis=0)
        for o, L, a in pb.pseudo:
            Fm[o:o + len(a)] += L @ (zm[o:o + len(a)] - a)
        return Fm

    zs = z.copy()
    shist = []
    for it in range(1, 21):
        Fs = F_solver(zs)
        dzs = schur_solve(dFfin, -Fs, npd, lblocks)
        zs = zs + dzs
        shist.append(dict(iteration=it, residual_before=float(np.linalg.norm(Fs)), step=float(np.linalg.norm(dzs))))
        if np.linalg.norm(dzs) <= 1e-12 * np.linalg.norm(zs):
            break
    Fs_fin = F_solver(zs)
    pc.close()

    # moves from x_c to x^, per block
    Dl = pb.fm.H  # Gauss-Newton matrix at x_c (export J), incl. pseudo-factors, export dof order
    pose_move, pose_ratio, pose_solver_gap = [], [], []
    for i, p in enumerate(ids):
        zi = z[pb.boff[i]:pb.boff[i + 1]]
        pose_move.append(float(np.linalg.norm(Cp[i].T @ zi)))
        pose_ratio.append(pose_move[-1] / (ETA * stop["patches"][p]["S_lo"]))
        pose_solver_gap.append(float(np.linalg.norm(Cp[i].T @ (zs - z)[pb.boff[i]:pb.boff[i + 1]])))
    lm_D, lm_m, lm_solver_gap = [], [], []
    for k in range(pb.npose, len(pb.blocks)):
        b = pb.blocks[k]
        sl = slice(ex.offsets[b], ex.offsets[b] + ex.dims[b])
        Dlk = Dl[sl][:, sl].toarray()
        Cl = np.linalg.cholesky(0.5 * (Dlk + Dlk.T))
        zl = z[pb.boff[k]:pb.boff[k + 1]]
        lm_D.append(float(np.linalg.norm(Cl.T @ zl)))
        lm_m.append(float(np.linalg.norm(zl[:3])))
        lm_solver_gap.append(float(np.linalg.norm(Cl.T @ (zs - z)[pb.boff[k]:pb.boff[k + 1]])))
    q = lambda a: dict(median=float(np.median(a)), p90=float(np.percentile(a, 90)), p99=float(np.percentile(a, 99)), max=float(np.max(a)))
    res = dict(dataset=ds, stop_round=stop["round"], poses=len(ids), landmarks=len(pb.blocks) - pb.npose, iterations=len(hist),
               converged=converged, final_residual=float(np.linalg.norm(Ffin)), final_residual_max_abs=float(np.abs(Ffin).max()),
               final_flags=int(flfin), history=hist,
               pose_move_D=q(pose_move), pose_move_over_eta_S_lo=q(pose_ratio),
               poses_truly_finished=float(np.mean(np.array(pose_ratio) <= 1.0)),
               landmark_move_D=q(lm_D), landmark_move_m=q(lm_m),
               landmarks_over_c_0_1m={str(c): int(np.sum(np.array(lm_m) > c * 0.1)) for c in (1, 2, 4, 8, 16)},
               implemented_equation=dict(iterations=len(shist), final_residual=float(np.linalg.norm(Fs_fin)), history=shist,
                                         pose_gap_D=q(pose_solver_gap), landmark_gap_D=q(lm_solver_gap)),
               seconds=time.perf_counter() - t0)
    out = REPO / "results/stage2b" / ds
    out.mkdir(parents=True, exist_ok=True)
    (out / "newton.json").write_text(json.dumps(res, indent=1))
    ed2 = REPO / "exports/stage2b" / ds
    ed2.mkdir(parents=True, exist_ok=True)
    np.save(ed2 / "z_star.npy", z)
    np.save(ed2 / "z_star_implemented.npy", zs)
    print(json.dumps({k: v for k, v in res.items() if k not in ("history",)}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
