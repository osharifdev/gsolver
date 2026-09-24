"""Package a Phase A export directory into the A5 format: model.npz + structure.json + states.npz.

The per-factor arrays written by the C++ exporter stay in the directory as the factor list
(f_<type>_{vars,J,e,Sigma,msg_tau,ids}; Lambda_f = Sigma_f^-1, r_f = -e_f). This adds:

  model.npz       H and b (CSR arrays), and each part separately: meas, gauge (G, b_G), single_edge;
                  x_bar (direct solve, from gateA) when available
  structure.json  block partition (id, type, dim, offset, pose index, timestamp), per-landmark
                  observation pose indices, GP parameters (Q_C, every dt), gauge description,
                  single-edge and uninformed blocks, chart conventions, freeze information
  states.npz      frozen exports: z at iterations 0, 1, 2, 5, 10, 20, 50, 100 and at the floor (when
                  gateA.json exists) and the last iteration; unfrozen exports: z = 0 (belief charts)

Usage: python -m certify.pack <export_dir> [<export_dir> ...]
"""

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from certify.io import Export, informed_blocks

CHARTS = {
    "SE3PoseVel": "tau = [t(3), r(3), v(6)];  x [+] tau = (T * Exp(tau[0:6]), v + tau[6:12]),  Exp(t, r) = (Rodrigues(r), t)",
    "SE3Pose": "tau = [t(3), r(3)];  T [+] tau = T * Exp(tau)",
    "SE3Point": "tau = dp(3);  p [+] tau = p + tau",
}
SNAPSHOT_ITERS = [0, 1, 2, 5, 10, 20, 50, 100]


def _csr(prefix, M, out):
    M = sp.csr_matrix(M)
    out[prefix + "_data"], out[prefix + "_indices"], out[prefix + "_indptr"] = M.data, M.indices, M.indptr
    out[prefix + "_shape"] = np.array(M.shape)


def pack(path):
    path = Path(path)
    ex = Export(path)
    out = {}
    H, b = ex.assemble(("meas", "gauge", "single_edge"))
    _csr("H", H, out)
    out["b"] = b
    for part in ("meas", "gauge", "single_edge"):
        Hp, bp = ex.assemble((part,))
        _csr("H_" + part, Hp, out)
        out["b_" + part] = bp
    if (path / "xbar.npy").exists():
        out["x_bar"] = np.load(path / "xbar.npy")
    np.savez_compressed(path / "model.npz", **out)

    informed = informed_blocks(H, ex)
    obs = {}
    for t, g in ex.factors.items():
        for row in g["vars"]:
            poses = [int(ex.pose_index[v]) for v in row if ex.var_type[v] == "SE3PoseVel"]
            for v in row:
                if ex.var_type[v] != "SE3PoseVel":
                    obs.setdefault(int(v), set()).update(poses)
    gauge = []
    for t, g in ex.factors.items():
        for k in np.flatnonzero(ex.is_gauge(t)):
            gauge.append(dict(id=str(g["ids"][k]), type=t, blocks=[int(v) for v in g["vars"][k]],
                              precision_diag=[float(v) for v in np.diag(np.linalg.inv(g["Sigma"][k]))]))
    structure = dict(
        meta=ex.meta,
        n_dof=ex.n_dof,
        blocks=[dict(index=i, id=str(ex.var_id[i]), type=str(ex.var_type[i]), dim=int(ex.dims[i]),
                     offset=int(ex.offsets[i]), pose_index=int(ex.pose_index[i]),
                     timestamp=None if np.isnan(ex.timestamp[i]) else float(ex.timestamp[i]),
                     degree=int(ex.degree[i])) for i in range(ex.n_blocks)],
        landmark_observation_pose_indices={str(k): sorted(v) for k, v in sorted(obs.items())},
        gp=dict(qc_diag=ex.meta.get("qc_diag"),
                qc_floor="factor Q_C = max(qc_diag, 1e-4) (factor_graph_builder.h:195); dt per GP factor in dt",
                dt=[float(v) for v in ex.factors.get("gp", {}).get("dt", [])],
                blocks=[[int(a), int(c)] for a, c in ex.factors.get("gp", {}).get("vars", np.zeros((0, 2), int))]),
        gauge_factors=gauge,
        single_edge_blocks=[int(i) for i, _, _ in ex.single_edge],
        uninformed_blocks=[int(i) for i in np.flatnonzero(~informed)],
        factor_files="f_<type>_{vars,J,e,Sigma,msg_tau,ids}: J_f (m x d, columns in the order of vars), "
                     "e_f at the linearization point (r_f = -e_f), Sigma_f (Lambda_f = Sigma_f^-1), "
                     "msg_tau = incoming variable-to-factor means in the export charts",
        model="H = sum_f J_f^T Sigma_f^-1 J_f + G_gauge + G_single_edge; b likewise (certify/FREEZE_MODE.md §3)",
        charts=CHARTS,
    )
    (path / "structure.json").write_text(json.dumps(structure, indent=1))

    Z = ex.states()
    st = {}
    if Z is not None:
        iters = np.load(path / "states_iters.npy") if (path / "states_iters.npy").exists() else np.arange(Z.shape[0])
        want = list(SNAPSHOT_ITERS)
        if (path / "gateA.json").exists():
            curves = json.loads((path / "gateA.json").read_text())["curves"]
            # the state at the floor of every flat curve (non-single-edge blocks, poses, resolvable blocks)
            for name in ("non_single_edge", "poses", "resolvable"):
                c = curves.get(name, {})
                if c.get("floor_reached") and c.get("t_floor") is not None:
                    want.append(int(c["t_floor"]))
                    st["floor_iteration_" + name] = np.array(int(c["t_floor"]))
        want.append(int(iters[-1]))
        for t in sorted(set(want)):
            j = np.searchsorted(iters, t)
            if j < len(iters) and iters[j] == t:
                st["z_%06d" % t] = np.asarray(Z[j])
    else:
        st["z_000000"] = ex.z
    np.savez_compressed(path / "states.npz", **st)
    return path


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print("packed", pack(p))
