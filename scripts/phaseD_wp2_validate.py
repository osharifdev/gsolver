"""Phase D, WP2: extended-precision validation of the double-precision sidecar on the sampled rounds (1, 10, last).

For every persisted export <run>/export_rNNNNN (the linearization and state before that round's message passing):
  - the certified model is rebuilt by the Phase C pipeline (certify.model.from_export, assembled in extended precision
    and stored hi + lo; decision D2) and reduced exactly to the poses (reduce_blocks);
  - the pose errors are solved with double-double refinement (SPDSystem, the Phase B/C machinery);
  - the sidecar's logged certificates of that round are checked against them: L_i <= ||e_i||_{D_i} <= E_i (1e-9 rel.);
  - the sidecar's double-precision oracle error and its fast reduced model S are compared with the extended ones;
  - the exact yardstick is recomputed from an extended I_F (the export's stereo rotation columns halved) and the
    sidecar's S_lo is checked against it (S_lo <= S_exact, implied by the Loewner bound the sidecar checks in double).

Usage: python3 scripts/phaseD_wp2_validate.py <run_dir> [--stereo]
"""

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from certify import certs, radii  # noqa: E402
from certify import model as mdl  # noqa: E402
from certify.io import Export  # noqa: E402
from certify.linalg import SPDSystem  # noqa: E402
from certify.sidecar_model import PoseModelBuilder  # noqa: E402

Q12 = radii.q(12)


def reduced(path, p_L, elim):
    full, info = mdl.from_export(path, p_L=p_L)
    red, kept = mdl.reduce_blocks(full, elim_kinds=elim)
    return red


def validate(run, stereo):
    run = Path(run)
    p_L = 1e-6 if stereo else None
    elim = ("landmark",) if stereo else ("landmark", "marker")
    rows = list(csv.DictReader(open(run / "sidecar" / "rounds.csv")))
    out = {}
    for ex_dir in sorted(run.glob("export_r*")):
        r = int(ex_dir.name.split("_r")[1])
        R = {x["pose"]: x for x in rows if int(x["round"]) == r}
        if any(x.get("decision") == "A0_FAILS" for x in R.values()):
            out[r] = dict(round=r, a0_fails=1)  # singular linearization: nothing was certified
            print(json.dumps(out[r]), flush=True)
            continue
        red = reduced(ex_dir, p_L, elim)
        ids = [str(b) for b in red.block_ids]
        ex = Export(ex_dir)
        # the export's pose states in the reduced model's block order
        zp = np.concatenate([ex.z[ex.block_slice(int(np.flatnonzero(ex.var_id == i)[0]))] for i in ids])
        G = red.residual(zp[:, None])
        sysm = SPDSystem(red.H, red.dims, None, refine=3, K_lo=red.H_lo)
        e = sysm.solve(G)[:, 0]
        met = certs.Metric(red)
        err_ext = met.norms(e[:, None])[:, 0]
        # fast double model vs extended pipeline
        B = PoseModelBuilder(p_L=p_L, g2o=str(REPO / ex.meta["g2o"]) if stereo else None)
        fast = B.build(None, ex=ex)
        order = [ids.index(str(p)) for p in fast["pose_ids"]]
        idx = np.concatenate([np.arange(red.block_slice(k).start, red.block_slice(k).stop) for k in order])
        H = red.H.toarray()[np.ix_(idx, idx)]
        rel_S = float(np.abs(fast["S"] - H).max() / np.abs(H).max())
        # extended I_F: a copy of the export with the stereo rotation columns halved
        S_ex = {}
        if stereo:
            with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
                t = Path(tmp) / "ex"
                shutil.copytree(ex_dir, t)
                J = np.load(t / "f_stereo_J.npy")
                J[:, :, 3:6] *= 0.5
                np.save(t / "f_stereo_J.npy", J)
                redF = reduced(t, p_L, elim)
            F = redF
        else:
            F = red
        sysF = SPDSystem(F.H, F.dims, None, refine=3, K_lo=F.H_lo)
        for k, pid in enumerate(ids):
            sl = red.block_slice(k)
            Eu = np.zeros((F.N, 12))
            Eu[F.block_slice(k), :] = np.eye(12)
            Sig = sysF.solve(Eu)[sl]
            Sig = 0.5 * (Sig + Sig.T)
            C = met.C[k]
            S_ex[pid] = float(np.sqrt(max(0.0, np.linalg.eigvalsh(C.T @ Sig @ C)[0]))) * Q12
        rec = dict(round=r, poses=len(ids), rel_S_fast_vs_extended=rel_S, upper_checked=0, lower_checked=0, upper_violations=0,
                   lower_violations=0, worst_upper_margin=-np.inf, worst_lower_margin=-np.inf, max_rel_err_double_vs_ext=0.0,
                   S_lo_checked=0, S_lo_violations=0, max_rel_S_exact_double_vs_ext=0.0)
        for k, pid in enumerate(ids):
            x = R.get(pid)
            if x is None or x.get("decision") == "A0_FAILS" or x.get("E") in ("", None):
                continue
            E, L, errd = float(x["E"]), float(x["L"]), float(x["err"])
            u = err_ext[k]
            if np.isfinite(E):
                rec["upper_checked"] += 1
                m = (u - E) / max(E, 1e-300)
                rec["worst_upper_margin"] = max(rec["worst_upper_margin"], m)
                rec["upper_violations"] += int(u > E * (1 + 1e-9) + 1e-12)
            rec["lower_checked"] += 1
            m = (L - u) / max(u, 1e-300)
            rec["worst_lower_margin"] = max(rec["worst_lower_margin"], m)
            rec["lower_violations"] += int(L > u * (1 + 1e-9) + 1e-12)
            if u > 1e-12:
                rec["max_rel_err_double_vs_ext"] = max(rec["max_rel_err_double_vs_ext"], abs(errd - u) / u)
            if x.get("S_lo") not in ("", "nan") and np.isfinite(float(x["S_lo"])):
                rec["S_lo_checked"] += 1
                rec["S_lo_violations"] += int(float(x["S_lo"]) > S_ex[pid] * (1 + 1e-9))
            rec["max_rel_S_exact_double_vs_ext"] = max(rec["max_rel_S_exact_double_vs_ext"], abs(float(x["S_exact"]) - S_ex[pid]) / S_ex[pid])
        out[r] = rec
        print(json.dumps(rec), flush=True)
    (run / "extended_validation.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    validate(sys.argv[1], "--stereo" in sys.argv)
