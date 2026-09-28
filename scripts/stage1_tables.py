"""Stage 1: markdown tables for reports/stage1.md from results/stage1/ (stops, eta sensitivity, validity, costs)."""

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results/stage1"
DS = ["kitti06", "kitti03", "kitti04", "kitti10", "charuco", "prefix"]
NAME = {"kitti06": "KITTI 06 (40)", "kitti03": "KITTI 03 (40)", "kitti04": "KITTI 04 (40)", "kitti10": "KITTI 10 (40)",
        "charuco": "ChArUco islam", "prefix": "KITTI 06 prefix (161)"}


def g(x, f="%.3g"):
    try:
        v = float(x)
        return "–" if not np.isfinite(v) else f % v
    except (TypeError, ValueError):
        return "–" if x in ("", None) else str(x)


def stops_table():
    out = ["| Dataset | Variant | Stop | Round | Remaining GN move: median / p95 | True-J move median | ATE (m) | Total flops | vs fixed | Total flops, alt. baseline |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for ds in DS:
        for var in ("corrected", "shipped"):
            p = R / ds / var / "stops.csv"
            if not p.exists():
                continue
            rows = list(csv.DictReader(open(p)))
            fixed = next(r for r in rows if r["criterion"] == "fixed_budget")
            for r in rows:
                rnd = r["round"] or ("not by " + r["note"].split()[-1] if r["note"].startswith("not by") else "–")
                rel = float(r["total_flops"]) / float(fixed["total_flops"]) if r["total_flops"] not in ("", None) else np.nan
                out.append("| %s | %s | %s | %s | %s / %s | %s | %s | %s | %s | %s |" % (
                    NAME[ds], var, r["criterion"].replace("_", " "), rnd, g(r["move_median"]), g(r["move_p95"]),
                    g(r["move_true_median"]), g(r["ate_aligned"], "%.4f"), g(r["total_flops"]), g(rel, "%.2f×"), g(r["total_flops_alt"])))
    return "\n".join(out)


def eta_table():
    out = ["| Dataset | Variant | η = 0.05 | η = 0.1 | η = 0.2 | η = 0.5 |", "|---|---|---|---|---|---|"]
    for ds in DS:
        for var in ("corrected", "shipped"):
            p = R / ds / var / "eta_sensitivity.csv"
            if not p.exists():
                continue
            cells = []
            for r in csv.DictReader(open(p)):
                if r["round"]:
                    cells.append("r%s: %s / %s" % (r["round"], g(r["move_median"]), g(r["total_flops"])))
                else:
                    cells.append("not by end (%s pass)" % g(r["frac_pass_last"], "%.2f"))
            out.append("| %s | %s | %s |" % (NAME[ds], var, " | ".join(cells)))
    return "\n".join(out)


def validity_table():
    out = ["| Dataset | Variant | Pose-evaluations | Violations upper / lower | Worst upper margin | Worst lower margin | Loewner min (rounds) | E/‖e‖ median / p95 | Certificate = oracle decisions | Growth exits (stop / continue / cap / steps) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for ds in DS:
        for var in ("corrected", "shipped"):
            p = R / ds / var / "validity.json"
            if not p.exists():
                continue
            v = json.load(open(p))
            ge = v["grow_exits"]
            out.append("| %s | %s | %d | %d / %d | %s | %s | %s (%s) | %s / %s | %d disagreements | %d / %d / %d / %d |" % (
                NAME[ds], var, v["pose_evaluations"], v["violations_upper"], v["violations_lower"], g(v["worst_rel_upper"]),
                g(v["worst_rel_lower"]), g(v["loewner_min"]), ",".join(map(str, v["loewner_rounds"])), g(v["E_over_err_median"]),
                g(v["E_over_err_p95"]), v["certified_vs_oracle_pass_disagreements"], ge["stop"], ge["continue"], ge["cap"], ge["steps"]))
    return "\n".join(out)


def curve_table():
    out = ["| Dataset | Variant | Fraction of poses passing at rounds 10, 20, 30, 50, 100, 150, 200, 250, 300 (ChArUco: 100, 200, …, 1000) |", "|---|---|---|"]
    for ds in DS:
        for var in ("corrected", "shipped"):
            p = R / ds / var / "pass_curve.csv"
            if not p.exists():
                continue
            c = {int(r["round"]): float(r["frac_pass"]) for r in csv.DictReader(open(p))}
            pick = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] if ds == "charuco" else [10, 20, 30, 50, 100, 150, 200, 250, 300]
            out.append("| %s | %s | %s |" % (NAME[ds], var, " · ".join("%.2f" % c[k] if k in c else "–" for k in pick)))
    return "\n".join(out)


def cost_table():
    out = ["| Dataset | Variant | GBP flops per round (median) | Alt. baseline per round | Certificate flops per evaluation: median (max) | Criterion (τ) flops per round | Evaluator s per evaluation (median) |",
           "|---|---|---|---|---|---|---|"]
    for ds in DS:
        for var in ("corrected", "shipped"):
            d = R / ds / var
            if not (d / "flops.csv").exists():
                continue
            fl = list(csv.DictReader(open(d / "flops.csv")))
            ev = list(csv.DictReader(open(d / "eval_summary.csv")))
            he = list(csv.DictReader(open(d / "heuristics.csv")))
            cf = np.array([float(r["certificate_flops"] or 0) for r in ev])
            t = np.array([float(r["load_s"]) + float(r["certify_s"]) + float(r["accuracy_s"]) for r in ev])
            out.append("| %s | %s | %s | %s | %s (%s) | %s | %s |" % (
                NAME[ds], var, g(np.median([float(r["gbp_flops"]) for r in fl])), g(np.median([float(r["gbp_flops_alt"]) for r in fl])),
                g(np.median(cf)), g(cf.max()), g(np.median([float(r["criterion_flops"]) for r in he])), g(np.median(t), "%.2f")))
    return "\n".join(out)


if __name__ == "__main__":
    for name, f in (("stops", stops_table), ("eta", eta_table), ("validity", validity_table), ("curve", curve_table), ("cost", cost_table)):
        print("### " + name)
        print(f())
        print()
