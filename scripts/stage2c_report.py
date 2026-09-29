"""Stage 2c: markdown tables for reports/stage2c.md, results/stage2c/comparison_old_vs_clean.csv and gate_C.json.

Old model: Stage 1 (results/stage1), Part A (results/partA), Stage 2b's Newton reference (results/stage2b). Clean model:
results/stage2c (stage1_analyse.py --stage stage2c, stage2c_tables.py, stage2c_newton.py, stage2c_anchor_check.py).
"""

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results/stage2c"
DS = ["kitti06", "kitti03", "kitti04", "kitti10"]
ALL = DS + ["prefix"]
NAME = {"kitti06": "KITTI 06", "kitti03": "KITTI 03", "kitti04": "KITTI 04", "kitti10": "KITTI 10", "prefix": "KITTI 06 prefix (161)"}
ETA = 0.1


def rd(p):
    p = Path(p)
    if not p.exists() and Path(str(p) + ".gz").exists():
        import gzip
        import io
        return list(csv.DictReader(io.TextIOWrapper(gzip.open(str(p) + ".gz"), "utf-8")))
    return list(csv.DictReader(open(p)))


def g(x, f="%.3g"):
    try:
        v = float(x)
        return "–" if not np.isfinite(v) else f % v
    except (TypeError, ValueError):
        return "–" if x in ("", None) else str(x)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def at_round(run_dir, rnd):
    """(unfinished current, unfinished relinearized, move median, ATE) at an evaluation round of a results dir."""
    rows = [r for r in rd(run_dir / "eval_rounds.csv") if int(r["round"]) == rnd]
    summ = [r for r in rd(run_dir / "eval_summary.csv") if int(r["round"]) == rnd]
    if not rows:
        return None
    if any(r.get("decision") in ("NONFINITE", "A0_FAILS") for r in rows):
        return dict(uc=1.0, ur=np.nan, move=np.nan, ate=np.nan, note="diverged")
    f = lambda k: np.array([fnum(r[k]) for r in rows])
    slo = f("S_lo")
    return dict(uc=float(np.mean(f("err") > ETA * slo)), ur=float(np.mean(f("move_S") > ETA * slo)), move=float(np.median(f("move"))),
                ate=fnum(summ[0]["ate_aligned"]) if summ else np.nan)


def stop_round(d):
    p = d / "stops.csv"
    if not p.exists():
        return None
    c = next(r for r in rd(p) if r["criterion"] == "certified")
    return c["round"] or None


def comparison():
    out = []
    cl = json.load(open(R / "cleaning.json"))["datasets"]
    for ds in ALL:
        c = cl.get(ds, {}).get("cleaning")
        if c:
            out.append(dict(quantity="landmarks in the model", dataset=ds, variant="both", old=c["landmarks_before"], clean=c["landmarks_remaining"]))
            out.append(dict(quantity="stereo factors in the model", dataset=ds, variant="both", old=c["observations_before"],
                            clean=c["observations_remaining"]))
        for var in ("corrected", "shipped"):
            o, n = REPO / "results/stage1" / ds / var, R / ds / var
            if not n.exists():
                continue
            out.append(dict(quantity="certified stop round (300-round runs)", dataset=ds, variant=var, old=stop_round(o) or "none",
                            clean=stop_round(n) or "none"))
            for b in (100, 300):
                a, z = at_round(o, b), at_round(n, b)
                if a is None or z is None:
                    continue
                out.append(dict(quantity="unfinished fraction, current problem, at round %d" % b, dataset=ds, variant=var, old=a["uc"], clean=z["uc"]))
                out.append(dict(quantity="median remaining GN move (D) at round %d" % b, dataset=ds, variant=var, old=a["move"], clean=z["move"]))
                out.append(dict(quantity="aligned keyframe ATE (m) at round %d" % b, dataset=ds, variant=var, old=a["ate"], clean=z["ate"]))
            fo, fn = rd(o / "flops.csv"), rd(n / "flops.csv")
            out.append(dict(quantity="GBP flops per round (median)", dataset=ds, variant=var, old=float(np.median([fnum(r["gbp_flops"]) for r in fo])),
                            clean=float(np.median([fnum(r["gbp_flops"]) for r in fn]))))
            so = [fnum(r["certificate_flops"]) for r in rd(o / "eval_summary.csv")]
            sn = [fnum(r["certificate_flops"]) for r in rd(n / "eval_summary.csv")]
            out.append(dict(quantity="certificate flops per evaluation (median)", dataset=ds, variant=var, old=float(np.nanmedian(so)),
                            clean=float(np.nanmedian(sn))))
        nb_old = REPO / "results/stage2b" / ds / "newton.json"
        nb_new = R / ds / "newton.json"
        if nb_old.exists() and nb_new.exists():
            a, z = json.load(open(nb_old)), json.load(open(nb_new))
            dg = json.load(open(REPO / "results/stage2b" / ds / "diagnosis.json"))
            none = "no certified stop" if z.get("certified_stop") is None else None
            out.append(dict(quantity="Newton check: stop state", dataset=ds, variant="corrected", old="round %d" % a["stop_round"],
                            clean=none or "round %d" % z["certified_stop"]))
            out.append(dict(quantity="Newton check: poses within eta S_lo of x^", dataset=ds, variant="corrected", old=a["poses_truly_finished"],
                            clean=none or z["fraction_within_eta_S_lo"]))
            out.append(dict(quantity="Newton check: pose distance / E, median; max", dataset=ds, variant="corrected",
                            old="%.3g; %.3g" % (dg["poses"]["true_over_E"]["median"], dg["poses"]["true_over_E"]["max"]),
                            clean=none or "%.3g; %.3g" % (z["pose_move_over_E"]["median"], z["pose_move_over_E"]["max"])))
            out.append(dict(quantity="Newton check: largest landmark move (m)", dataset=ds, variant="corrected", old=a["landmark_move_m"]["max"],
                            clean=none or z["landmarks_two_observations"]["move_m"].get("max")))
    with open(R / "comparison_old_vs_clean.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["quantity", "dataset", "variant", "old", "clean"])
        w.writeheader()
        w.writerows(out)
    return out


def gate_c():
    per = {}
    for ds in DS:
        n = json.load(open(R / ds / "newton.json"))
        if n.get("certified_stop") is None:
            per[ds] = dict(certified_stop=None, fraction_within_eta_S_lo=None, passes=False, note=n.get("note"))
        else:
            f = n["fraction_within_eta_S_lo"]
            per[ds] = dict(certified_stop=n["certified_stop"], fraction_within_eta_S_lo=f, passes=bool(f >= 0.95))
    k = sum(1 for v in per.values() if v["passes"])
    res = dict(criterion="on at least 3 of the 4 segments, at least 95% of poses at the certified stop are within eta S_lo,i of x^",
               segments=per, segments_passing=k, gate_C=k >= 3)
    (R / "gate_C.json").write_text(json.dumps(res, indent=1))
    return res


def tables():
    L = []
    # Step 1
    cl = json.load(open(R / "cleaning.json"))
    L.append("### Step 1: counts")
    L.append("| Dataset | Landmarks before | Removed (one observation) | Remaining | Stereo factors before | Removed | Remaining | Poses left without observations |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ds in ALL:
        c = cl["datasets"].get(ds, {}).get("cleaning")
        if c:
            L.append("| %s | %d | %d | %d | %d | %d | %d | %d |" % (NAME[ds], c["landmarks_before"], c["landmarks_removed"], c["landmarks_remaining"],
                                                               c["observations_before"], c["observations_removed"], c["observations_remaining"],
                                                               c["poses_without_observations_after"]))
    L.append("")
    L.append("### Step 1.4: anchor check (corrected variant, round-100 state)")
    L.append("| Dataset | Pose solution change 1e4 → 1e6: max relative (worst pose) | max in D / ηS_lo | Landmark max relative | λ_min(H) 1e4 / 1e6 | λ_min(S) 1e4 / 1e6 | Without p_L: max relative; max D / ηS_lo | p_L gap at 1e4: max D / ηS_lo; poses above ηS_lo |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ds in DS:
        d = cl["datasets"][ds]
        a, b, p = d["anchor_check"], d["anchor_check_without_p_L"], d["p_L_gap_at_anchor_1e4"]
        L.append("| %s | %s (%s) | %s | %s | %s / %s | %s / %s | %s; %s | %s; %d of 40 |" % (
            NAME[ds], g(a["pose_solution_rel_change_max"]), a["pose_solution_rel_change_worst_pose"], g(a["pose_solution_change_D_over_eta_S_lo_max"]),
            g(a["landmark_solution_rel_change_max"]), g(a["lambda_min_H"]["1e4"]), g(a["lambda_min_H"]["1e6"]), g(a["lambda_min_S"]["1e4"]),
            g(a["lambda_min_S"]["1e6"]), g(b["pose_solution_rel_change_max"]), g(b["pose_solution_change_D_over_eta_S_lo_max"]),
            g(p["pose_solution_change_D_over_eta_S_lo_max"]), p["poses_over_eta_S_lo"]))
    L.append("")
    # Step 2
    L.append("### Step 2: stops")
    L.append("| Dataset | Variant | Certified | Fixed 100: move median / p95, ATE | τ = 1e-2 | τ = 1e-3 | τ = 1e-4 | Largest belief change, round 300 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ds in ALL:
        for var in ("corrected", "shipped"):
            p = R / ds / var / "stops.csv"
            if not p.exists():
                continue
            S = {r["criterion"]: r for r in rd(p)}
            he = rd(R / ds / var / "heuristics.csv")
            last = he[-1]["max_change"]

            def cell(r):
                return r["round"] if r["round"] else (r["note"] or "–")
            f = S["fixed_budget"]
            L.append("| %s | %s | %s | %s / %s, %s m | %s | %s | %s | %s |" % (
                NAME[ds], var, cell(S["certified"]), g(f["move_median"]), g(f["move_p95"]), g(f["ate_aligned"], "%.3f"),
                cell(S["change_tau_1e-02"]), cell(S["change_tau_1e-03"]), cell(S["change_tau_1e-04"]), g(last)))
    L.append("")
    L.append("### Step 2: validity and decisions")
    L.append("| Dataset | Variant | Pose-evaluations | Violations upper / lower | Worst upper margin (round, pose) | Worst lower margin | Loewner min (rounds) | Certificate ≠ oracle | Rounds not certifiable | Growth exits stop / continue / cap / steps |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for ds in ALL:
        for var in ("corrected", "shipped"):
            p = R / ds / var / "validity.json"
            if not p.exists():
                continue
            v = json.load(open(p))
            ge = v["grow_exits"]
            w = v["worst_rel_upper_at"] or {}
            nc = v.get("rounds_not_certifiable", [])
            L.append("| %s | %s | %d | %d / %d | %s (r%s, %s) | %s | %s (%s) | %d | %s | %d / %d / %d / %d |" % (
                NAME[ds], var, v["pose_evaluations"], v["violations_upper"], v["violations_lower"], g(v["worst_rel_upper"]), w.get("round"),
                w.get("pose"), g(v["worst_rel_lower"]), g(v["loewner_min"]), ",".join(map(str, v["loewner_rounds"])),
                v["certified_vs_oracle_pass_disagreements"], ("%d (%d–%d)" % (len(nc), nc[0], nc[-1])) if nc else "0", ge["stop"], ge["continue"],
                ge["cap"], ge["steps"]))
    L.append("")
    L.append("### Step 2: η sensitivity (first round at which every pose passes E_i ≤ η S_lo,i)")
    L.append("| Dataset | Variant | η = 0.05 | η = 0.1 | η = 0.2 | η = 0.5 |")
    L.append("|---|---|---|---|---|---|")
    for ds in ALL:
        for var in ("corrected", "shipped"):
            p = R / ds / var / "eta_sensitivity.csv"
            if not p.exists():
                continue
            cells = []
            for r in rd(p):
                cells.append("r%s" % r["round"] if r["round"] else "none (%s pass at the end)" % g(r["frac_pass_last"], "%.2f"))
            L.append("| %s | %s | %s |" % (NAME[ds], var, " | ".join(cells)))
    L.append("")
    L.append("### Step 2: pass-fraction curve and oracle")
    L.append("| Dataset | Variant | Certified pass fraction at rounds 10, 50, 100, 200, 300 | Oracle pass fraction, same rounds | E/‖e‖ median |")
    L.append("|---|---|---|---|---|")
    for ds in ALL:
        for var in ("corrected", "shipped"):
            p = R / ds / var / "pass_curve.csv"
            if not p.exists():
                continue
            c = {int(r["round"]): r for r in rd(p)}
            pick = [10, 50, 100, 200, 300]
            v = json.load(open(R / ds / var / "validity.json"))
            L.append("| %s | %s | %s | %s | %s |" % (NAME[ds], var, " · ".join(g(c[k]["frac_pass"], "%.2f") if k in c else "–" for k in pick),
                                                 " · ".join(g(c[k]["frac_pass_oracle"], "%.2f") if k in c else "–" for k in pick),
                                                 g(v["E_over_err_median"], "%.6f")))
    L.append("")
    L.append("### Step 2: costs")
    L.append("| Dataset | Variant | GBP flops per round (median) | Belief-minus-incoming baseline per round | Certificate flops per evaluation: median (max) | Criterion flops per round | Evaluator s per evaluation (median) | GBP flops, 300 rounds | Certificate flops, 30 evaluations |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for ds in ALL:
        for var in ("corrected", "shipped"):
            d = R / ds / var
            if not (d / "flops.csv").exists():
                continue
            fl, ev, he = rd(d / "flops.csv"), rd(d / "eval_summary.csv"), rd(d / "heuristics.csv")
            cf = np.array([fnum(r["certificate_flops"]) for r in ev])
            t = np.array([fnum(r["load_s"]) + fnum(r["certify_s"]) + fnum(r["accuracy_s"]) for r in ev])
            L.append("| %s | %s | %s | %s | %s (%s) | %s | %s | %s | %s |" % (
                NAME[ds], var, g(np.median([fnum(r["gbp_flops"]) for r in fl])), g(np.median([fnum(r["gbp_flops_alt"]) for r in fl])),
                g(np.nanmedian(cf)), g(np.nanmax(cf)), g(np.median([fnum(r["criterion_flops"]) for r in he])), g(np.nanmedian(t), "%.1f"),
                g(sum(fnum(r["gbp_flops"]) for r in fl)), g(np.nansum(cf))))
    L.append("")
    # Step 3
    L.append("### Step 3.1: budgets (fraction of poses unfinished: current / relinearized; median and p95 remaining move; ATE)")
    L.append("| Dataset | Variant | 10 | 50 | 100 | 200 | 300 |")
    L.append("|---|---|---|---|---|---|---|")
    B = rd(R / "budgets.csv")
    for ds in ALL:
        for var in ("corrected", "shipped"):
            rows = {int(r["budget"]): r for r in B if r["dataset"] == ds and r["variant"] == var}
            if not rows:
                continue
            cells = []
            for b in (10, 50, 100, 200, 300):
                r = rows.get(b)
                if r is None:
                    cells.append("–")
                elif r["note"]:
                    cells.append("%s / – (diverged)" % g(r["unfinished_current"], "%.2f"))
                else:
                    cells.append("%s / %s; %s, %s; %s m" % (g(r["unfinished_current"], "%.2f"), g(r["unfinished_relinearized"], "%.2f"),
                                                           g(r["move_median"]), g(r["move_p95"]), g(r["ate_aligned"], "%.3f")))
            L.append("| %s | %s | %s |" % (NAME[ds], var, " | ".join(cells)))
    L.append("")
    # Step 4
    L.append("### Step 4: Gate C")
    gc = json.load(open(R / "gate_C.json"))
    L.append("| Dataset | Certified stop | Poses within ηS_lo of x̂ | Passes |")
    L.append("|---|---|---|---|")
    for ds in DS:
        s = gc["segments"][ds]
        L.append("| %s | %s | %s | %s |" % (NAME[ds], s["certified_stop"] or "none by round 300", g(s["fraction_within_eta_S_lo"], "%.3f"),
                                            "yes" if s["passes"] else "no"))
    L.append("")
    L.append("### Old model against the clean model")
    L.append("| Quantity | Dataset | Variant | Old | Clean |")
    L.append("|---|---|---|---|---|")
    for r in comparison():
        L.append("| %s | %s | %s | %s | %s |" % (r["quantity"], NAME[r["dataset"]], r["variant"], g(r["old"]), g(r["clean"])))
    (R / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def worst_cases():
    """The single worst case of every invariant and ratio over the clean-model runs, with its location."""
    L = ["### Worst cases", "| Quantity | Worst value | Where |", "|---|---|---|"]
    best = {}

    def upd(k, v, where, larger=True):
        if not np.isfinite(v):
            return
        if k not in best or (v > best[k][0] if larger else v < best[k][0]):
            best[k] = (v, where)
    for ds in ALL:
        for var in ("corrected", "shipped"):
            p = R / ds / var / "eval_rounds.csv"
            if not p.exists():
                continue
            last = None
            for r in rd(p):
                if r.get("decision") in ("A0_FAILS", "NONFINITE"):
                    continue
                w = "%s %s, round %s, %s" % (NAME[ds], var, r["round"], r["pose"])
                upd("validity, upper: (‖e‖ − E)/E (tolerance 1e-9)", fnum(r["rel_upper_margin"]), w)
                upd("validity, lower: (L − ‖e‖)/‖e‖ (tolerance 1e-9)", fnum(r["rel_lower_margin"]), w)
                upd("patch lower bound vs exact marginal, Loewner margin (≥ 0 required)", fnum(r["loewner_margin"]), w, larger=False)
                last = int(r["round"])
            for r in rd(p):
                if int(r["round"]) != last or r.get("decision") in ("A0_FAILS", "NONFINITE"):
                    continue
                w = "%s %s, round %s, %s" % (NAME[ds], var, r["round"], r["pose"])
                upd("oracle ‖e_i‖ / ηS_lo at the last certifiable evaluation (largest)", fnum(r["err"]) / (ETA * fnum(r["S_lo"])), w)
                upd("oracle ‖e_i‖ / ηS_lo at the last certifiable evaluation (smallest: the pose closest to passing)",
                    fnum(r["err"]) / (ETA * fnum(r["S_lo"])), w, larger=False)
                upd("certified E_i / ηS_lo at the last certifiable evaluation (smallest)", fnum(r["E"]) / (ETA * fnum(r["S_lo"])), w, larger=False)
    fd = json.load(open(R / "fdcheck.json"))
    for run, v in fd.items():
        if "corrected" in run:
            upd("Jacobian vs central differences, corrected variant (requirement 1e-5)", v["worst_rel"], "%s: %s" % (run.split("stage2c/")[-1], v["worst_block"]))
    cl = json.load(open(R / "cleaning.json"))["datasets"]
    for ds in DS:
        a = cl[ds]["anchor_check"]
        upd("anchor 1e4 → 1e6: pose solution change / ηS_lo (with p_L)", a["pose_solution_change_D_over_eta_S_lo_max"], NAME[ds])
        upd("anchor 1e4 → 1e6: relative pose solution change (with p_L)", a["pose_solution_rel_change_max"], "%s, %s" % (NAME[ds], a["pose_solution_rel_change_worst_pose"]))
        b = cl[ds]["anchor_check_without_p_L"]
        upd("anchor 1e4 → 1e6: relative pose solution change (without p_L)", b["pose_solution_rel_change_max"], "%s, %s" % (NAME[ds], b["pose_solution_rel_change_worst_pose"]))
        g_ = cl[ds]["p_L_gap_at_anchor_1e4"]
        upd("certificate model vs solver graph (p_L gap) / ηS_lo", g_["pose_solution_change_D_over_eta_S_lo_max"], NAME[ds])
        upd("certificate model vs solver graph (p_L gap), largest landmark difference (m)", g_["landmark_solution_change_m_max"], NAME[ds])
    for k, (v, w) in best.items():
        L.append("| %s | %s | %s |" % (k, g(v), w))
    (R / "worst_cases.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    print(json.dumps(gate_c(), indent=1))
    tables()
    worst_cases()
