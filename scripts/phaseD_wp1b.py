"""Phase D, WP1b (the WP1 amendment of 2026-09-27): threshold relinearization at equal work.

Definitions, fixed before any WP1b run:
- Final pose error: as in WP1 (scripts/phaseD_wp1.py): per pose ||xbar_i||_{D_i}, xbar the exact solution of the
  linearization at the current belief means, D_i = H_ii; stereo models use decision D2 (p_L = 1e-6).
- Work: cumulative linearizations + message computations (certify/cpp/certify_common.h, RoundRunner). W0 = the
  default schedule's work over its budget (ChArUco islam: the shipped 103 rounds; KITTI-40: the 100 rounds of WP1
  run 2).
- Gate per dataset: at work W0 (the state after the threshold run's last round with cumulative work <= W0),
  median <= 1.05 x the default's final median and 95th percentile <= 1.5 x the default's final 95th percentile.
- Rounds-to-reach: the first curve point (every 10 rounds) on the complete graph (corrected after the ChArUco runs:
  on an incremental sequence earlier points cover fewer poses) whose median is <= the default's final median; runs are
  capped at 1,030 rounds (ChArUco, 10 x its budget) and 300 rounds (KITTI-40, 3 x); "not reached" otherwise.
- Choice of (delta, R): among the settings that pass on both datasets, the fewest factor relinearizations per
  round, measured as the fraction of factors relinearized per round up to W0, averaged over the two datasets;
  ties go to the largest delta.
- Diagnostic curves (KITTI-40): keyframe ATE/ARE (raw and SE(3)-aligned) of the belief means against ground truth
  at the keyframe timestamps, and the belief-cavity gap: for every variable with at least two edges, the median
  over its edges of ||m_{v->f} boxminus mu_v|| in the physical metric (Euclidean tangent norm), then the median over
  variables (all, poses, landmarks).

Usage:
  python3 scripts/phaseD_wp1b.py run <run_dir> [--stereo] [--keep]     analyse one run (curve, at_budget, final)
  python3 scripts/phaseD_wp1b.py gate <out.json> <dataset_dir> ...      gate, rounds-to-reach and choice
"""

import csv
import glob
import json
import os
import shutil
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from phaseD_wp1 import final_error  # noqa: E402
from certify.tools.traj_eval import read_tum, rotation_angles, umeyama_se3  # noqa: E402

GT = REPO / "examples/data/datasets/stereo_visual/gt/06.tum"


def keyframe_ate(export):
    rows = list(csv.DictReader(open(Path(export) / "variables.tsv"), delimiter="\t"))
    t = np.array([float(r["timestamp"]) for r in rows if r["type"] == "SE3PoseVel"])
    T = np.load(Path(export) / "pose_T_lin.npy")
    tg, pg, Rg = read_tum(str(GT))
    idx = np.clip(np.searchsorted(tg, t), 0, len(tg) - 1)
    ok = np.abs(tg[idx] - t) < 1e-6
    pe, Re, pg, Rg = T[ok, :3, 3], T[ok, :3, :3], pg[idx[ok]], Rg[idx[ok]]
    Ra, ta = umeyama_se3(pe, pg)
    pa = pe @ Ra.T + ta
    return dict(keyframes=int(ok.sum()),
                ate=float(np.sqrt(np.mean(np.sum((pe - pg) ** 2, axis=1)))),
                are=float(np.sqrt(np.mean(rotation_angles(Rg, Re) ** 2))),
                ate_aligned=float(np.sqrt(np.mean(np.sum((pa - pg) ** 2, axis=1)))),
                are_aligned=float(np.sqrt(np.mean(rotation_angles(Rg, np.einsum("ij,kjl->kil", Ra, Re)) ** 2))))


def cavity_gap(export):
    """Belief-cavity gap from a belief-chart export (msg_tau = cavity mean offsets from the belief)."""
    ex = Path(export)
    rows = list(csv.DictReader(open(ex / "variables.tsv"), delimiter="\t"))
    dims = np.array([int(r["dim"]) for r in rows])
    kind = np.array([r["type"] for r in rows])
    degree = np.load(ex / "degree.npy")
    per_var = [[] for _ in rows]
    for vf in glob.glob(str(ex / "f_*_vars.npy")):
        stem = vf[:-len("vars.npy")]
        V = np.load(vf)
        tau = np.load(stem + "msg_tau.npy")
        for k in range(len(V)):
            off = 0
            for v in V[k]:
                d = dims[v]
                if degree[v] >= 2:
                    per_var[v].append(float(np.linalg.norm(tau[k, off:off + d])))
                off += d
    med = np.array([np.median(g) if g else np.nan for g in per_var])
    out = {}
    for tag, sel in (("all", np.ones(len(rows), bool)), ("poses", kind == "SE3PoseVel"),
                     ("landmarks", kind == "SE3Point"), ("markers", kind == "SE3Pose")):
        m = med[sel & np.isfinite(med)]
        if len(m):
            out[tag] = float(np.median(m))
    return out


def analyse_export(args):
    export, stereo, extras = args
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        e, t = final_error(export, stereo)
        rec = dict(err_median=float(np.median(e)), err_p95=float(np.percentile(e, 95)), err_max=float(e.max()),
                   err_worst_pose_time=float(t[int(np.argmax(e))]))
        meta = json.loads((Path(export) / "meta.json").read_text())
        rec.update(round=int(meta.get("round", -1)), cumulative_work=float(meta.get("cumulative_work", np.nan)))
        if stereo:
            rec["keyframe_traj"] = keyframe_ate(export)
        if extras:
            rec["cavity_gap"] = cavity_gap(export)
        return export, rec


def run(run_dir, stereo, keep, extras=False, workers=8):
    run_dir = Path(run_dir)
    curve = sorted(glob.glob(str(run_dir / "curve" / "r*")))
    jobs = [(c, stereo, extras) for c in curve]
    for name in ("at_budget", "final"):
        if (run_dir / name).exists():
            jobs.append((str(run_dir / name), stereo, extras))
    with get_context("fork").Pool(workers) as pool:
        res = dict(pool.map(analyse_export, jobs))
    out = dict(run=str(run_dir), curve=[res[c] for c in curve])
    for name in ("at_budget", "final"):
        p = str(run_dir / name)
        if p in res:
            out[name] = res[p]
    work = list(csv.DictReader(open(run_dir / "work.csv")))
    out["rounds"] = len(work)
    out["work_per_round"] = [int(r["linearizations"]) + int(r["messages"]) for r in work]
    out["linearizations_per_round"] = [int(r["linearizations"]) for r in work]
    out["messages_per_round"] = [int(r["messages"]) for r in work]
    out["cumulative_work"] = [float(r["cumulative_work"]) for r in work]
    out["wall_s_per_round"] = [float(r["wall_s"]) for r in work]
    ws = run_dir / "work_summary.json"
    if ws.exists():
        out["work_summary"] = json.loads(ws.read_text())
    rj = run_dir / "run.json"
    if rj.exists():
        out["run_info"] = json.loads(rj.read_text())
    tum = run_dir / "traj.tum"
    if stereo and tum.exists() and "final" in out:
        from phaseD_wp1 import ate
        rows = list(csv.DictReader(open(run_dir / "final" / "variables.tsv"), delimiter="\t"))
        t = np.array([float(r["timestamp"]) for r in rows if r["type"] == "SE3PoseVel"])
        out["final"]["interp_traj"] = ate(tum, str(GT), t.min(), t.max())
    (run_dir / "wp1b_analysis.json").write_text(json.dumps(out, indent=1))
    if not keep:
        shutil.rmtree(run_dir / "curve", ignore_errors=True)
    return out


def gate(out_path, dataset_dirs):
    res = {}
    for dd in dataset_dirs:
        dd = Path(dd)
        runs = {Path(p).parent.name: json.loads(Path(p).read_text()) for p in glob.glob(str(dd / "*" / "wp1b_analysis.json"))}
        d = runs["default"]
        W0 = d["cumulative_work"][-1]
        n_factors_end = d["linearizations_per_round"][-1]
        med0, p95_0 = d["final"]["err_median"], d["final"]["err_p95"]
        # Rounds-to-reach counts only curve points on the complete graph: on an incremental sequence (ChArUco) the
        # early points cover only the poses added so far (correction of 2026-09-27, made after the ChArUco runs).
        lin0 = d["linearizations_per_round"]
        full_round = 1 + next(k for k, x in enumerate(lin0) if x == lin0[-1])
        rec = dict(W0=W0, default_rounds=d["rounds"], default_median=med0, default_p95=p95_0, full_graph_round=full_round,
                   settings={})
        for name, r in sorted(runs.items()):
            if name == "default" or not name.startswith("d"):
                continue
            ab = r.get("at_budget")
            s = dict(at_budget=ab)
            if ab is not None:
                k = ab["round"]
                s.update(rounds_at_W0=k, median_ratio=ab["err_median"] / med0, p95_ratio=ab["err_p95"] / p95_0,
                         passes=bool(ab["err_median"] <= 1.05 * med0 and ab["err_p95"] <= 1.5 * p95_0),
                         relin_fraction_per_round=float(np.mean(np.array(r["linearizations_per_round"][:k]) /
                                                                np.maximum(1, n_factors_end))),
                         linearizations_per_round=float(np.mean(r["linearizations_per_round"][:k])))
            else:
                s.update(passes=False, note="work budget not reached within the cap")
            reach = next((c for c in r["curve"] if c["round"] >= full_round and c["err_median"] <= med0), None)
            s["rounds_to_reach_default_median"] = None if reach is None else reach["round"]
            s["work_to_reach_default_median"] = None if reach is None else reach["cumulative_work"]
            s["cap_rounds"] = r["rounds"]
            rec["settings"][name] = s
        res[dd.name] = rec
    names = set.intersection(*[set(v["settings"]) for v in res.values()])
    passing = [n for n in names if all(res[k]["settings"][n]["passes"] for k in res)]

    def delta_of(n):
        return float(n.split("_")[0][1:])

    cost = {n: float(np.mean([res[k]["settings"][n]["relin_fraction_per_round"] for k in res])) for n in passing}
    choice = min(passing, key=lambda n: (cost[n], -delta_of(n))) if passing else None
    res["choice"] = dict(passing=sorted(passing), cost=cost, chosen=choice)
    Path(out_path).write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        if k == "choice":
            continue
        print("== %s: W0 %.4g (default %d rounds), default median %.4g, p95 %.4g" % (k, v["W0"], v["default_rounds"], v["default_median"], v["default_p95"]))
        for n, s in sorted(v["settings"].items(), key=lambda kv: (delta_of(kv[0]), kv[0])):
            print("  %-12s rounds@W0 %-5s median x%-6s p95 x%-6s pass %-5s relin/round %-8s reach %s" % (
                n, s.get("rounds_at_W0"), None if "median_ratio" not in s else round(s["median_ratio"], 3),
                None if "p95_ratio" not in s else round(s["p95_ratio"], 3), s["passes"],
                None if "linearizations_per_round" not in s else round(s["linearizations_per_round"], 1),
                s["rounds_to_reach_default_median"]))
    print("passing on both:", sorted(passing), "chosen:", choice)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        a = sys.argv[2:]
        out = run(a[0], "--stereo" in a, "--keep" in a, extras="--extras" in a)
        c = out["curve"]
        print("%s: %d rounds, %d curve points%s%s" % (a[0], out["rounds"], len(c),
              "" if "at_budget" not in out else ", at W0 (round %d) median %.4g p95 %.4g" % (out["at_budget"]["round"], out["at_budget"]["err_median"], out["at_budget"]["err_p95"]),
              "" if "final" not in out else ", final median %.4g p95 %.4g" % (out["final"]["err_median"], out["final"]["err_p95"])))
    elif sys.argv[1] == "gate":
        gate(sys.argv[2], sys.argv[3:])
