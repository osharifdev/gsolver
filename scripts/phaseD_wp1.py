"""Phase D, WP1: accuracy of threshold relinearization against G-solver's per-round schedule.

Each run directory is written by `certify_charuco --mode islam` or `certify_stereo` with `--final-export`:
  final/            the linearization at the run's final belief means (state z = 0)
  traj.tum          (stereo) G-solver's GP interpolation of the final beliefs at the ground-truth timestamps
  relin_rounds.csv  (threshold runs) per-round re-freezes and relinearized factors
  relin_counts.csv  (threshold runs) re-freezes per variable
  run.json          wall time of the driver, written by scripts/phaseD_wp1_run.sh

Final pose error (pre-registered, brief WP3's definition): ||xbar_i||_{D_i} for every pose i, where xbar = H^-1 b
is the exact solution of the linearization at the final beliefs (one direct solve; the export state is z = 0)
and D_i = H_ii. It is the move that one exact Gauss-Newton step of the solver's own estimating equation would
still make at the run's final state. Stereo models use decision D2 of Phase C (p_L = 1e-6). The WP1 criterion
compares the median over poses with the per-round schedule's: within 5 %, i.e. median <= 1.05 x default.
ATE/ARE (raw and SE(3)-aligned, certify/tools/traj_eval.py) use the ground-truth timestamps inside the keyframe
time span only.

Usage: python3 scripts/phaseD_wp1.py <out.json> <label>=<run_dir>[,gt=<gt.tum>] ...
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from certify import model as mdl  # noqa: E402
from certify.linalg import SPDSystem  # noqa: E402
from certify.tools.traj_eval import read_tum, rotation_angles, umeyama_se3  # noqa: E402


def pose_time(block_id):
    return float(str(block_id).split("POSE_")[1])


def final_error(export_dir, stereo):
    model, info = mdl.from_export(export_dir, p_L=1e-6 if stereo else None)
    sysm = SPDSystem(model.H, model.dims, model.elim, refine=3, K_lo=model.H_lo)
    if not sysm.ok:
        raise RuntimeError("H is not positive definite at the final state of %s" % export_dir)
    xbar = sysm.solve(model.b, R_lo=getattr(model, "b_lo", None))
    H = model.H.tocsr()
    poses = np.flatnonzero(model.kinds == "pose")
    err, times = [], []
    for i in poses:
        sl = model.block_slice(i)
        D = H[sl][:, sl].toarray()
        x = xbar[sl]
        err.append(float(np.sqrt(max(0.0, x @ D @ x))))
        times.append(pose_time(model.block_ids[i]))
    order = np.argsort(times)
    return np.array(err)[order], np.array(times)[order]


def ate(traj, gt, t0, t1, tol=1e-6):
    te, pe, Re = read_tum(traj)
    tg, pg, Rg = read_tum(gt)
    idx = np.clip(np.searchsorted(tg, te), 0, len(tg) - 1)
    ok = (np.abs(tg[idx] - te) < tol) & (te >= t0 - tol) & (te <= t1 + tol)
    pe, Re, pg, Rg = pe[ok], Re[ok], pg[idx[ok]], Rg[idx[ok]]
    out = dict(matched=int(ok.sum()))
    out["ate"] = float(np.sqrt(np.mean(np.sum((pe - pg) ** 2, axis=1))))
    out["are"] = float(np.sqrt(np.mean(rotation_angles(Rg, Re) ** 2)))
    Ra, ta = umeyama_se3(pe, pg)
    pa = pe @ Ra.T + ta
    out["ate_aligned"] = float(np.sqrt(np.mean(np.sum((pa - pg) ** 2, axis=1))))
    out["are_aligned"] = float(np.sqrt(np.mean(rotation_angles(Rg, np.einsum("ij,kjl->kil", Ra, Re)) ** 2)))
    return out


def relin_stats(run):
    out = {}
    rr = run / "relin_rounds.csv"
    if rr.exists():
        rows = list(csv.DictReader(open(rr)))
        out["rounds"] = len(rows)
        out["relin_poses_total"] = int(sum(int(r["relin_poses"]) for r in rows))
        out["relin_other_total"] = int(sum(int(r["relin_other"]) for r in rows))
        out["relin_factors_total"] = int(sum(int(r["relin_factors"]) for r in rows))
        out["relin_factors_per_round_mean"] = out["relin_factors_total"] / max(1, len(rows))
        out["rounds_with_pose_relin"] = int(sum(int(r["relin_poses"]) > 0 for r in rows))
        out["wall_s_per_round_median"] = float(np.median([float(r["wall_s"]) for r in rows]))
    rc = run / "relin_counts.csv"
    if rc.exists():
        rows = list(csv.DictReader(open(rc)))
        for tag, sel in (("poses", "1"), ("other", "0")):
            R = [r for r in rows if r["is_pose"] == sel]
            if not R:
                continue
            lin = np.array([1 + int(r["relinearizations"]) for r in R])
            life = np.array([int(r["rounds"]) - int(r["created_round"]) + 1 for r in R])
            rpl = life / lin
            out["rounds_per_linearization_%s" % tag] = dict(median=float(np.median(rpl)), p5=float(np.percentile(rpl, 5)),
                                                            min=float(rpl.min()), pooled=float(life.sum() / lin.sum()))
            out["relinearizations_%s" % tag] = dict(median=float(np.median(lin - 1)), max=int((lin - 1).max()),
                                                    never=float(np.mean(lin == 1)))
    rj = run / "run.json"
    if rj.exists():
        out["run"] = json.loads(rj.read_text())
    return out


def main(out_path, specs):
    res = {}
    for spec in specs:
        label, rest = spec.split("=", 1)
        parts = rest.split(",")
        run = Path(parts[0])
        gt = next((p.split("=", 1)[1] for p in parts[1:] if p.startswith("gt=")), None)
        t0 = time.perf_counter()
        stereo = gt is not None
        e, times = final_error(run / "final", stereo)
        rec = dict(run=str(run), poses=int(len(e)), err_median=float(np.median(e)), err_p95=float(np.percentile(e, 95)),
                   err_max=float(e.max()), err_worst_pose_time=float(times[int(np.argmax(e))]), err_mean=float(e.mean()))
        if gt and (run / "traj.tum").exists():
            rec["traj"] = ate(run / "traj.tum", gt, times.min(), times.max())
        rec.update(relin_stats(run))
        rec["analysis_s"] = time.perf_counter() - t0
        np.save(run / "final_pose_error.npy", np.stack([times, e]))
        res[label] = rec
        print("%-28s poses %4d  error median %.4g  p95 %.4g  max %.4g%s%s" % (
            label, rec["poses"], rec["err_median"], rec["err_p95"], rec["err_max"],
            "" if "traj" not in rec else "  ATE %.4f m (aligned %.4f)" % (rec["traj"]["ate"], rec["traj"]["ate_aligned"]),
            "" if "rounds_per_linearization_poses" not in rec else "  rounds/linearization (poses) median %.1f" % (
                rec["rounds_per_linearization_poses"]["median"])), flush=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
