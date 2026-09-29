"""Stage 1: stops, costs, eta sensitivity and Gate A (BRIEF_stage1_2.md §2.2-§2.6; certify/STAGE1.md).

Per run (exports/stage1/<dataset>/<variant>) it writes results/stage1/<dataset>/<variant>/:
  eval_rounds.csv       (copied from the evaluator) per evaluation round and pose
  eval_summary.csv      (copied) per evaluation round
  stops.csv             per stop criterion: round, remaining Gauss-Newton move (median, p95; solver's Jacobian = WP1's
                        definition, and true Jacobian), ATE, GBP flops, certificate / criterion flops, total, and the same
                        with the belief-minus-incoming GBP baseline
  heuristics.csv, flops.csv (copied), eta_sensitivity.csv, pass_curve.csv, validity.json
Stops:
  certified   first evaluation round (>= stop_from) at which every pose passes E_i <= eta S_lo,i (eta = 0.1)
  fixed       100 rounds (KITTI segments), 103 (ChArUco, the shipped islam budget)
  change_tau  first round (>= change_from) at which the largest belief change in the D-metric is below tau
Costs to a stop at round t: GBP flops of rounds 1..t (as instrumented) plus, for the certified stop, the certificate flops of
its evaluation rounds <= t, and for a change-based stop the criterion's flops of rounds 1..t.
Then results/stage1/gate_A.json from the corrected (and, reported, the shipped) variant of the four KITTI segments.

Usage: python3 scripts/stage1_analyse.py [--stage stage2c] [dataset/variant ...]
       (default: every run under exports/<stage> without a tag; stage1 by default, Stage 2c's clean model with --stage stage2c,
       which reads exports/stage2c and writes results/stage2c)
"""

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from certify.io import Export  # noqa: E402
from certify.sidecar_model import PoseModelBuilder  # noqa: E402
from certify.stage1 import accuracy  # noqa: E402

ETAS = (0.05, 0.1, 0.2, 0.5)
TAUS = ("1e-02", "1e-03", "1e-04")
KITTI = ("kitti06", "kitti03", "kitti04", "kitti10")
GT = {"kitti06": "06", "kitti03": "03", "kitti04": "04", "kitti10": "10", "prefix": "06"}
STAGE = "stage1"


def rd(path):
    return list(csv.DictReader(open(path)))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def analyse(ds, var):
    src = REPO / "exports" / STAGE / ds / var
    dst = REPO / "results" / STAGE / ds / var
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("eval/eval_rounds.csv", "eval/eval_summary.csv", "heuristics.csv", "flops.csv", "work.csv", "run.json",
              "relin_rounds.csv"):
        if (src / f).exists():
            shutil.copy(src / f, dst / Path(f).name)
    summ = rd(src / "eval/eval_summary.csv")
    rows = rd(src / "eval/eval_rounds.csv")
    fl = rd(src / "flops.csv")
    heur = rd(src / "heuristics.csv")
    gbp = np.array([float(r["gbp_flops"]) for r in fl])
    gbp_alt = np.array([float(r["gbp_flops_alt"]) for r in fl])
    crit = np.array([float(r["criterion_flops"]) for r in heur])
    cum = lambda a, t: float(a[:t].sum())
    cert = {int(r["round"]): float(r["certificate_flops"] or 0) for r in summ}
    stop_from = 94 if ds == "charuco" else 0
    change_from = 95 if ds == "charuco" else 1
    budget = 103 if ds == "charuco" else 100
    stereo = ds != "charuco"
    gt = REPO / f"examples/data/datasets/stereo_visual/gt/{GT[ds]}.tum" if stereo else None
    builder = None

    def acc_of_export(path):
        nonlocal builder
        ex = Export(path)
        if builder is None:
            builder = PoseModelBuilder(p_L=1e-6 if stereo else None, g2o=str(REPO / ex.meta["g2o"]) if stereo else None)
        a = accuracy(ex, builder, str(gt) if gt else None)
        return dict(move_median=float(np.median(a["move"])), move_p95=float(np.percentile(a["move"], 95)),
                    move_true_median=float(np.median(a["move_true"])), move_true_p95=float(np.percentile(a["move_true"], 95)),
                    ate_aligned=a.get("ate_aligned", np.nan))

    def acc_of_eval(t):
        r = next(x for x in summ if int(x["round"]) == t)
        mv = [float(x["move"]) for x in rows if int(x["round"]) == t]
        mt = [float(x["move_true"]) for x in rows if int(x["round"]) == t]
        return dict(move_median=float(np.median(mv)), move_p95=float(np.percentile(mv, 95)), move_true_median=float(np.median(mt)),
                    move_true_p95=float(np.percentile(mt, 95)), ate_aligned=fnum(r["ate_aligned"]))

    stops = []
    # certified stop (eta = 0.1) and the pass curve
    curve = [dict(round=int(r["round"]), poses=int(r["poses"]), frac_pass=fnum(r["frac_pass"]), all_pass=int(r["all_pass"]),
                  frac_pass_oracle=fnum(r["frac_pass_oracle_exact"])) for r in summ]
    with open(dst / "pass_curve.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curve[0].keys()))
        w.writeheader()
        w.writerows(curve)
    tc = next((c["round"] for c in curve if c["all_pass"] and c["round"] >= stop_from), None)
    stops.append(dict(criterion="certified", round=tc))
    stops.append(dict(criterion="fixed_budget", round=budget))
    for tau in TAUS:
        col = "below_" + {"1e-02": "0.01", "1e-03": "0.001", "1e-04": "0.0001"}[tau]
        t = next((int(h["round"]) for h in heur if h[col] == "1" and int(h["round"]) >= change_from), None)
        stops.append(dict(criterion="change_tau_" + tau, round=t))
    rounds_run = len(fl)
    for s in stops:
        t = s["round"]
        if t is None:
            s.update(note="not by %d" % rounds_run)
            continue
        if t > rounds_run:
            s.update(note="beyond the run (%d rounds)" % rounds_run)
            continue
        if s["criterion"] == "certified":
            s.update(acc_of_eval(t))
            c = sum(v for k, v in cert.items() if k <= t)
            s.update(evaluations=sum(1 for k in cert if k <= t), extra_flops=c)
        elif s["criterion"] == "fixed_budget":
            if t % 10 == 0 and any(int(x["round"]) == t for x in summ):
                s.update(acc_of_eval(t))
            else:
                s.update(acc_of_export(src / ("beliefs_r%05d" % t)))
            s.update(evaluations=0, extra_flops=0.0)
        else:
            exs = Export(src / ("stop_tau_" + s["criterion"].split("_")[-1]))
            if not (np.isfinite(exs.pose_T_lin).all() and np.isfinite(exs.point_p_lin).all()):
                # Stage 2c: on a diverged (non-finite) state the logged largest change is 0 (NaN is skipped by the max),
                # so the criterion "fires" spuriously; not a stop
                s.update(round=None, note="fires at round %d on a non-finite (diverged) state: not a stop" % t)
                continue
            s.update(acc_of_export(src / ("stop_tau_" + s["criterion"].split("_")[-1])))
            s.update(evaluations=t, extra_flops=cum(crit, t))
        s.update(gbp_flops=cum(gbp, t), gbp_flops_alt=cum(gbp_alt, t))
        s.update(total_flops=s["gbp_flops"] + s["extra_flops"], total_flops_alt=s["gbp_flops_alt"] + s["extra_flops"])
    keys = ["criterion", "round", "note", "move_median", "move_p95", "move_true_median", "move_true_p95", "ate_aligned", "gbp_flops",
            "extra_flops", "total_flops", "gbp_flops_alt", "total_flops_alt", "evaluations"]
    with open(dst / "stops.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(stops)

    # eta sensitivity from the logged E and S_lo
    by_round = {}
    for x in rows:
        by_round.setdefault(int(x["round"]), []).append(x)
    eta_rows = []
    for eta in ETAS:
        t = next((r for r in sorted(by_round) if r >= stop_from and all(fnum(x["E"]) <= eta * fnum(x["S_lo"]) for x in by_round[r])), None)
        e = dict(eta=eta, round=t)
        if t is not None:
            e.update(acc_of_eval(t))
            c = sum(v for k, v in cert.items() if k <= t)
            e.update(gbp_flops=cum(gbp, t), certificate_flops=c, total_flops=cum(gbp, t) + c, total_flops_alt=cum(gbp_alt, t) + c,
                     frac_pass_curve=";".join("%d:%.3f" % (r, np.mean([fnum(x["E"]) <= eta * fnum(x["S_lo"]) for x in by_round[r]]))
                                              for r in sorted(by_round)))
        else:
            e.update(note="not by %d" % rounds_run, frac_pass_last=float(np.mean([fnum(x["E"]) <= eta * fnum(x["S_lo"])
                                                                                  for x in by_round[max(by_round)]])))
        eta_rows.append(e)
    with open(dst / "eta_sensitivity.csv", "w", newline="") as fh:
        k = ["eta", "round", "note", "move_median", "move_p95", "move_true_median", "ate_aligned", "gbp_flops", "certificate_flops",
             "total_flops", "total_flops_alt", "frac_pass_last", "frac_pass_curve"]
        w = csv.DictWriter(fh, fieldnames=k, extrasaction="ignore")
        w.writeheader()
        w.writerows(eta_rows)

    # validity (rounds at which the certificate could not run, A0_FAILS or NONFINITE, are counted and left out)
    skipped = [x for x in rows if x.get("decision") in ("A0_FAILS", "NONFINITE")]
    skipped_rounds = sorted(set(int(x["round"]) for x in skipped))
    rows = [x for x in rows if x.get("decision") not in ("A0_FAILS", "NONFINITE")]
    up = np.array([fnum(x["rel_upper_margin"]) for x in rows])
    lo = np.array([fnum(x["rel_lower_margin"]) for x in rows])
    lw = np.array([fnum(x["loewner_margin"]) for x in rows])
    worst_up = int(np.nanargmax(up)) if np.isfinite(up).any() else None
    worst_lo = int(np.nanargmax(lo)) if np.isfinite(lo).any() else None
    val = dict(pose_evaluations=len(rows), evaluations=len(summ), rounds_not_certifiable=skipped_rounds,
               rounds_nonfinite=sorted(set(int(x["round"]) for x in skipped if x.get("decision") == "NONFINITE")),
               violations_upper=int(sum(int(x["valid_upper"]) == 0 for x in rows if x.get("valid_upper") not in (None, ""))),
               violations_lower=int(sum(int(x["valid_lower"]) == 0 for x in rows if x.get("valid_lower") not in (None, ""))),
               worst_rel_upper=float(np.nanmax(up)) if worst_up is not None else None,
               worst_rel_upper_at=dict(round=rows[worst_up]["round"], pose=rows[worst_up]["pose"]) if worst_up is not None else None,
               worst_rel_lower=float(np.nanmax(lo)) if worst_lo is not None else None,
               worst_rel_lower_at=dict(round=rows[worst_lo]["round"], pose=rows[worst_lo]["pose"]) if worst_lo is not None else None,
               loewner_min=float(np.nanmin(lw)) if np.isfinite(lw).any() else None,
               loewner_rounds=sorted(set(int(x["round"]) for x in rows if np.isfinite(fnum(x["loewner_margin"])))),
               E_over_err_median=float(np.median([fnum(x["E"]) / max(fnum(x["err"]), 1e-300) for x in rows])),
               E_over_err_p95=float(np.percentile([fnum(x["E"]) / max(fnum(x["err"]), 1e-300) for x in rows], 95)),
               certified_vs_oracle_pass_disagreements=int(sum(int(x["passed"]) != int(x["passed_oracle_lo"]) for x in rows)),
               grow_exits={k: int(sum(1 for x in rows if x["grow_why"] == k)) for k in ("stop", "continue", "cap", "steps", "no_c0")})
    (dst / "validity.json").write_text(json.dumps(val, indent=1))
    return dict(stops=stops, eta=eta_rows, validity=val, curve=curve)


def gate(results):
    out = {}
    for var in ("corrected", "shipped"):
        per = {}
        for ds in KITTI:
            r = results.get((ds, var))
            if r is None:
                continue
            S = {s["criterion"]: s for s in r["stops"]}
            c, f = S["certified"], S["fixed_budget"]
            d = dict(certified_round=c["round"], fixed_round=f["round"])
            if c["round"] is None:
                d.update(stop_le_60=False, accuracy_ratio=None, cost_ratio=None, cost_ratio_alt=None, passes=False)
            else:
                d["stop_le_60"] = c["round"] <= 60
                d["accuracy_ratio"] = c["move_median"] / f["move_median"]
                d["cost_ratio"] = c["total_flops"] / f["total_flops"]
                d["cost_ratio_alt"] = c["total_flops_alt"] / f["total_flops_alt"]
                d["passes"] = bool(d["stop_le_60"] and d["accuracy_ratio"] <= 1.05 and d["cost_ratio"] <= 0.8)
                d["passes_alt_baseline"] = bool(d["stop_le_60"] and d["accuracy_ratio"] <= 1.05 and d["cost_ratio_alt"] <= 0.8)
            per[ds] = d
        n = sum(1 for d in per.values() if d["passes"])
        out[var] = dict(segments=per, passing=n, gate_A=n >= 3 if len(per) == 4 else None)
    return out


def main(args):
    global STAGE
    if "--stage" in args:
        k = args.index("--stage")
        STAGE = args[k + 1]
        args = args[:k] + args[k + 2:]
    runs = []
    if args:
        runs = [tuple(a.split("/")) for a in args]
    else:
        for p in sorted((REPO / "exports" / STAGE).glob("*/*")):
            if p.is_dir() and "_" not in p.name and (p / "eval/eval_summary.csv").exists():
                runs.append((p.parent.name, p.name))
    results = {}
    for ds, var in runs:
        results[(ds, var)] = analyse(ds, var)
        print("==", ds, var)
        for s in results[(ds, var)]["stops"]:
            print("   %-18s round %-5s move med %-10s p95 %-10s ATE %-8s total %.3g (alt %.3g)" % (
                s["criterion"], s["round"], "%.4g" % s.get("move_median", np.nan), "%.4g" % s.get("move_p95", np.nan),
                "%.4g" % s.get("ate_aligned", np.nan), s.get("total_flops", np.nan), s.get("total_flops_alt", np.nan)))
    g = gate(results)
    (REPO / "results" / STAGE / "gate_A.json").write_text(json.dumps(g, indent=1, default=str))
    print(json.dumps(g, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
