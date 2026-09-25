"""Decisions D2 and D6 of the Phase A review, on every frozen export used in Phase C.

For each dataset/snapshot:
  D6  (a) raw model: H, b assembled factor by factor with plain loops from the exported J_f, Sigma_f, e_f,
          the gauge factors and the frozen single-edge messages, against the matrices of the production
          assembler (certify.io.Export.assemble; saved by Phase A in model.npz where it exists);
      (b) certified model (D2): the same independent assembly of the certified factor list against the
          H, b used by this script's direct solve.
  D2  counts (single-observation variables eliminated, zero-information landmarks excluded, landmark
      priors), direct solve residual, lambda_min / lambda_max, near-null dimension (eigenvalues below
      1e-8 lambda_max, exact count by Sylvester inertia), the frozen-GBP pose floor against xbar, and for the
      stereo datasets the p_L sensitivity (p_L in {1e-7, 1e-6, 1e-5}): pose floor, the change of the pose
      solution relative to p_L = 1e-6 (D-metric of the p_L = 1e-6 model), and the number of landmarks whose
      observation-information smallest eigenvalue is below 10 p_L.

Insensitivity rule (stated before computing): poses are insensitive to p_L on [1e-7, 1e-5] when for every
p_L the pose floor is <= 0.05 and max_i ||xbar_i(p_L) - xbar_i(1e-6)||_{D_i} <= 0.05 over poses.

Usage: python3 scripts/phaseC_d2.py [dataset ...]   (default: all)
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from certify import model as mdl  # noqa: E402
from certify.io import Export  # noqa: E402
from certify.linalg import SPDSystem  # noqa: E402
from certify.phaseA import floor_and_rate  # noqa: E402

E = REPO / "exports"
OUT = REPO / "results" / "phaseC" / "d2"
DATASETS = {
    "charuco_islam_025": E / "charuco_islam/frac025",
    "charuco_islam_050": E / "charuco_islam/frac050",
    "charuco_islam_100": E / "charuco_islam/frac100",
    "charuco_ba_init": E / "charuco_ba/init",
    "charuco_ba_end": E / "charuco_ba/end",
    "kitti06_prefix_conv": E / "kitti06_prefix161/conv",
    "kitti06_prefix_init": E / "kitti06_prefix161/init",
    "kitti06_full_conv": E / "kitti06_full/conv_noanchor",
}
P_L = (1e-7, 1e-6, 1e-5)
TOL_FLOOR = 0.05


def raw_independent(ex):
    """D6(a): dict (i, j) -> block and b, factor by factor (meas + gauge + single-edge messages)."""
    blocks, b = {}, np.zeros(ex.n_dof)
    sl = lambda i: slice(int(ex.offsets[i]), int(ex.offsets[i] + ex.dims[i]))
    for t, g in ex.factors.items():
        m = g["vars"].shape[1]
        for n in range(len(g["vars"])):
            vs = g["vars"][n]
            dims = [int(ex.dims[v]) for v in vs]
            cs = np.concatenate([[0], np.cumsum(dims)])
            J, L, r = g["J"][n], np.linalg.inv(g["Sigma"][n]), -g["e"][n]
            for a in range(m):
                Ja = J[:, cs[a]:cs[a + 1]]
                b[sl(vs[a])] += Ja.T @ (L @ r)
                for c in range(m):
                    key = (int(vs[a]), int(vs[c]))
                    blk = Ja.T @ L @ J[:, cs[c]:cs[c + 1]]
                    blocks[key] = blocks[key] + blk if key in blocks else blk
    for i, tau, L in ex.single_edge:
        key = (int(i), int(i))
        blocks[key] = blocks[key] + L if key in blocks else L.copy()
        b[sl(i)] += L @ tau
    return blocks, b


def compare_blocks(blocks, b, H, b_used, offsets, dims):
    H = H.tocsr()
    worst, scale, seen = 0.0, 0.0, 0
    for (i, j), blk in blocks.items():
        ref = H[offsets[i]:offsets[i] + dims[i], offsets[j]:offsets[j] + dims[j]].toarray()
        worst = max(worst, float(np.abs(ref - blk).max()))
        scale = max(scale, float(np.abs(blk).max()))
        seen += int(np.count_nonzero(ref))
    return dict(max_abs_diff_H=worst, max_abs_H=scale, rel_diff_H=worst / scale if scale else 0.0,
                max_abs_diff_b=float(np.abs(b - b_used).max()), max_abs_b=float(np.abs(b_used).max()),
                nonzeros_outside_factor_blocks=int(H.count_nonzero() - seen))


def global_system(model):
    s = SPDSystem(model.H, model.dims, model.elim, refine=3, K_lo=model.H_lo)
    if not s.ok:
        raise RuntimeError("H is not positive definite")
    return s


def count_below(model, tau):
    """Exact number of eigenvalues of H below tau (Sylvester inertia; landmarks eliminated first)."""
    H = model.H.tocsr()
    dof_e = np.repeat(model.elim, model.dims)
    L, A = np.flatnonzero(dof_e), np.flatnonzero(~dof_e)
    if len(L) == 0:
        return int(np.sum(np.linalg.eigvalsh(H.toarray() - tau * np.eye(H.shape[0])) < 0))
    KLL = H[L][:, L].tocoo()
    nl = len(L) // 3
    blk = np.zeros((nl, 3, 3))
    np.add.at(blk, (KLL.row // 3, KLL.row % 3, KLL.col % 3), KLL.data)
    blk = blk - tau * np.eye(3)[None]
    n_ll = int(np.sum(np.linalg.eigvalsh(blk) < 0))
    Linv = sp.block_diag(list(np.linalg.inv(blk)), format="csr")
    HAL = H[A][:, L]
    S = H[A][:, A].toarray() - tau * np.eye(len(A)) - (HAL @ Linv @ HAL.T).toarray()
    return n_ll + int(np.sum(np.linalg.eigvalsh(0.5 * (S + S.T)) < 0))


def spectrum(model, sysm):
    lmax = float(spla.eigsh(model.H, k=1, which="LA", return_eigenvectors=False, tol=1e-8)[0])
    N = model.N
    op = spla.LinearOperator((N, N), dtype=float,
                             matvec=lambda v: sysm.solve(np.asarray(v).ravel(), refine=1, extended=False))
    try:
        mu = spla.eigsh(op, k=6, which="LA", return_eigenvectors=False, tol=1e-8, maxiter=400)
        converged = True
    except spla.ArpackNoConvergence as err:
        # the top of H^-1's spectrum is a cluster of thousands of eigenvalues at 1/p_L (landmark directions held
        # only by their prior); report what converged
        mu = err.eigenvalues if len(err.eigenvalues) else np.array([np.nan])
        converged = False
    small = np.sort(1.0 / np.asarray(mu, float))
    return lmax, small, converged


def obs_info_min_eig(model):
    """Smallest eigenvalue of each landmark's observation information sum_f J_l^T Lambda_f J_l (landmark
    priors excluded)."""
    lm = np.flatnonzero(model.kinds == "landmark")
    acc = {int(l): np.zeros((3, 3)) for l in lm}
    for f in model.factors:
        if f["kind"] == "landmark_prior" or f["vars"].shape[1] < 2:
            continue
        dims = [int(model.dims[f["vars"][0, a]]) for a in range(f["vars"].shape[1])]
        cs = np.concatenate([[0], np.cumsum(dims)])
        for a in range(f["vars"].shape[1]):
            sel = np.flatnonzero(model.kinds[f["vars"][:, a]] == "landmark")
            for r in sel:
                acc[int(f["vars"][r, a])] += f["Hf"][r][cs[a]:cs[a + 1], cs[a]:cs[a + 1]]
    return np.array([np.linalg.eigvalsh(acc[int(l)])[0] for l in lm])


def pose_curve(model, info, xbar, stride=1):
    ex = info["export"]
    kept = info["kept_export_blocks"]
    Z = ex.states()
    iters = np.load(ex.path / "states_iters.npy")
    pose_b = np.flatnonzero(model.kinds == "pose")
    cols = np.concatenate([np.arange(ex.offsets[kept[b]], ex.offsets[kept[b]] + ex.dims[kept[b]]) for b in pose_b])
    mcols = model.dof_index(pose_b)
    C = [np.linalg.cholesky(model.diag_blocks()[b]) for b in pose_b]
    CT = np.array([c.T for c in C])  # n x 12 x 12
    xb = xbar[mcols].reshape(len(pose_b), -1)
    rows = np.arange(0, Z.shape[0], stride)
    err = np.zeros(len(rows))
    worst = np.zeros(len(rows), int)
    for k, r in enumerate(rows):
        z = np.asarray(Z[r][cols]).reshape(len(pose_b), -1)
        e = np.linalg.norm(np.einsum("nij,nj->ni", CT, z - xb), axis=1)
        err[k] = e.max()
        worst[k] = int(pose_b[np.argmax(e)])
    return iters[rows], err, worst


def all_blocks_last(model, info, xbar):
    ex = info["export"]
    kept = info["kept_export_blocks"]
    z = np.asarray(ex.states()[-1])
    zz = mdl.state_restrict(info, z)
    D = model.diag_blocks()
    e = zz - xbar
    return np.array([np.sqrt(max(e[model.block_slice(i)] @ D[i] @ e[model.block_slice(i)], 0.0)) for i in range(model.n_blocks)])


def run(name, path):
    t0 = time.perf_counter()
    ex = Export(path)
    out = dict(dataset=name, export=str(path))
    # ---------------- D6(a): raw model
    blocks, b = raw_independent(ex)
    Hp, bp = ex.assemble(("meas", "gauge", "single_edge"))
    out["d6_raw_vs_production_assembler"] = compare_blocks(blocks, b, Hp, bp, ex.offsets, ex.dims)
    if (path / "model.npz").exists():
        z = np.load(path / "model.npz")
        Hs = sp.csr_matrix((z["H_data"], z["H_indices"], z["H_indptr"]), shape=tuple(z["H_shape"]))
        out["d6_raw_vs_phaseA_model_npz"] = compare_blocks(blocks, b, Hs, z["b"], ex.offsets, ex.dims)
    del blocks
    stereo = "kitti" in name
    runs = {}
    for pL in (P_L if stereo else (None,)):
        model, info = mdl.from_export(path, p_L=pL)
        r = dict(p_L=pL)
        for k in ("single_edge_in_export", "single_edge_eliminated", "single_edge_eliminated_by_kind",
                  "zero_information_excluded", "zero_information_by_kind", "landmark_priors", "landmark_prior_missing_init",
                  "zero_jacobian_factors_dropped", "n_blocks", "n_dof"):
            r[k] = info.get(k)
        r["n_poses"] = int(np.sum(model.kinds == "pose"))
        r["n_landmarks"] = int(np.sum(model.kinds == "landmark"))
        r["n_markers"] = int(np.sum(model.kinds == "marker"))
        # D6(b)
        if pL in (None, 1e-6):
            ib, bb = mdl.independent_assembly(model)
            r["d6_certified"] = compare_blocks(ib, bb, model.H, model.b, model.offsets, model.dims)
            del ib
        sysm = global_system(model)
        xbar = sysm.solve(model.b)
        r["rel_residual"] = float(np.linalg.norm(model.residual(xbar)) / np.linalg.norm(model.b))
        lmax, small, conv = spectrum(model, sysm)
        r["lambda_max"], r["lambda_smallest"], r["lambda_min"] = lmax, [float(v) for v in small], float(small[0])
        r["lanczos_converged"] = conv
        r["near_null_threshold"] = 1e-8 * lmax
        r["near_null_dim"] = count_below(model, 1e-8 * lmax)
        its, err, worst = pose_curve(model, info, xbar)
        fr = floor_and_rate(its, err)
        r["pose_floor"] = fr
        r["pose_floor_worst_block"] = dict(id=str(model.block_ids[worst[-1]]), pose_index=int(model.pose_index[worst[-1]]))
        eb = all_blocks_last(model, info, xbar)
        r["all_blocks_last_max"] = float(eb.max())
        r["landmarks_last_max"] = float(eb[model.kinds == "landmark"].max()) if r["n_landmarks"] else None
        r["markers_last_max"] = float(eb[model.kinds == "marker"].max()) if r["n_markers"] else None
        if stereo:
            lm = obs_info_min_eig(model)
            r["n_landmarks_obs_info_below_10pL"] = int(np.sum(lm < 10 * pL))
            r["obs_info_min_eig_quantiles"] = [float(q) for q in np.quantile(lm, [0, 0.01, 0.1, 0.5])]
        pose_dofs = model.dof_index(np.flatnonzero(model.kinds == "pose"))
        runs[pL] = (r, xbar[pose_dofs], model)
        np.save(OUT / ("%s_xbar_pL%s.npy" % (name, "none" if pL is None else "%g" % pL)), xbar)
        out.setdefault("models", []).append(r)
        print(name, "p_L", pL, "pose floor %.3e (reached %s, t=%s)" % (fr["floor"], fr["floor_reached"], fr["t_floor"]),
              "lmin %.3e nn %d" % (r["lambda_min"], r["near_null_dim"]), flush=True)
    if stereo:
        _, x6, m6 = runs[1e-6]
        C = [np.linalg.cholesky(m6.diag_blocks()[b]) for b in np.flatnonzero(m6.kinds == "pose")]
        sens = {}
        for pL, (r, xp, _) in runs.items():
            d = (xp - x6).reshape(len(C), -1)
            sens["%g" % pL] = float(max(np.linalg.norm(c.T @ v) for c, v in zip(C, d)))
        out["pose_solution_change_vs_1e-6"] = sens
        out["insensitive"] = bool(all(r["pose_floor"]["floor"] <= TOL_FLOOR for r, _, _ in runs.values())
                                  and max(sens.values()) <= TOL_FLOOR)
    out["floor_pass"] = bool(all(m["pose_floor"]["floor"] <= TOL_FLOOR for m in out["models"]))
    out["seconds"] = time.perf_counter() - t0
    (OUT / ("%s.json" % name)).write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or list(DATASETS)
    for n in names:
        run(n, DATASETS[n])
