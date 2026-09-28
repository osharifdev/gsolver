"""Stage 2b: tables for reports/stage2b.md, results/stage2b/intervals.json and gate_B2.json."""

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results/stage2b"
DS = ["kitti06", "kitti03", "kitti04", "kitti10"]


def g(x, f="%.3g"):
    try:
        return f % float(x)
    except (TypeError, ValueError):
        return str(x)


def main():
    lines = []
    lines.append("### Newton")
    lines.append("| Stop | Iterations | Converged | Final ‖F‖ | Pose move (D): median / p90 / p99 / max | Pose move / ηS_lo: median / max | Poses truly finished | Landmark move (D): median / p90 / p99 / max | Landmark move (m): median / p90 / p99 / max | Landmarks > c·0.1 m (c = 1, 2, 4, 8, 16) | Implemented equation's zero vs x̂ (poses, D): median / max |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for ds in DS:
        p = R / ds / "newton.json"
        if not p.exists():
            continue
        n = json.load(open(p))
        pm, pr, lm, lmm = n["pose_move_D"], n["pose_move_over_eta_S_lo"], n["landmark_move_D"], n["landmark_move_m"]
        lines.append("| %s r%d | %d | %s | %s | %s / %s / %s / %s | %s / %s | %.1f %% | %s / %s / %s / %s | %s / %s / %s / %s | %s | %s / %s |" % (
            ds, n["stop_round"], n["iterations"], "yes" if n["converged"] else "no", g(n["final_residual"]), g(pm["median"]), g(pm["p90"]),
            g(pm["p99"]), g(pm["max"]), g(pr["median"]), g(pr["max"]), 100 * n["poses_truly_finished"], g(lm["median"]), g(lm["p90"]), g(lm["p99"]),
            g(lm["max"]), g(lmm["median"]), g(lmm["p90"]), g(lmm["p99"]), g(lmm["max"]), " / ".join(str(v) for v in n["landmarks_over_c_0_1m"].values()),
            g(n["implemented_equation"]["pose_gap_D"]["median"]), g(n["implemented_equation"]["pose_gap_D"]["max"])))
    lines.append("")
    lines.append("### T1 attempts")
    lines.append("| Stop | Option | c | Verified | Landmarks with s_l > ū_l | Unbounded factor enclosures | Poses with infinite Ψ | Finite Ψ/ū, poses: max | s_i/ū_i, poses: median / max |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    gate = {}
    intervals = dict(library="own (certify/cpp/t1_ad.h): intervals of doubles, every +, -, *, / in round-to-nearest with both bounds "
                             "moved outward by one ulp (std::nextafter); no libm (SO(3) maps as power series with enclosed remainders)",
                     rounding="round-to-nearest plus one-ulp outward widening per operation", per_stop={})
    for ds in DS:
        for opt in ("A", "B"):
            p = R / ds / opt / "boxes.json"
            if not p.exists():
                continue
            b = json.load(open(p))
            for a in b["attempts"]:
                spu = a.get("s_pose_over_u", {})
                lines.append("| %s | %s | %d | %s | %d | %d | %s | %s | %s / %s |" % (
                    ds, opt, a["c"], "**yes**" if a["verified"] else "no", a["landmarks_s_over_u"], a["unbounded_factors"],
                    a.get("poses_with_infinite_Psi", "–"), g(a["pose_ratio_max"]), g(spu.get("median", "")), g(spu.get("max", ""))))
            ver = [a for a in b["attempts"] if a["verified"]]
            gate.setdefault(opt, {})[ds] = dict(box_verifies=bool(ver), c=ver[0]["c"] if ver else None)
            intervals["per_stop"]["%s/%s" % (ds, opt)] = [dict(c=a["c"], unbounded_factors=a["unbounded_factors"], delta=a["delta_stats"])
                                                           for a in b["attempts"]]
    for opt in gate:
        n_ok = sum(1 for d in gate[opt].values() if d["box_verifies"])
        gate[opt]["segments_passing"] = n_ok
        gate[opt]["gate_B2"] = False if n_ok < 3 else "requires validity, finish fraction and tightness"
    (R / "gate_B2.json").write_text(json.dumps(gate, indent=1))
    (R / "intervals.json").write_text(json.dumps(intervals, indent=1, default=float))
    lines.append("")
    lines.append("### Evaluator checks")
    lines.append("| Stop | F(0) vs solver, stereo: median / max | F(0) vs solver, GP: median / max | F′ vs central differences | Point enclosure contains doubles | Random box points outside enclosure |")
    lines.append("|---|---|---|---|---|---|")
    for ds in DS:
        p = R / ds / "evaluator_checks.json"
        if not p.exists():
            continue
        c = json.load(open(p))
        e = c["enclosures"]
        lines.append("| %s | %s / %s | %s / %s | %s | %s | %d / %d |" % (
            ds, g(c["F0_vs_solver_stereo"]["median"]), g(c["F0_vs_solver_stereo"]["max"]), g(c["F0_vs_solver_gp"]["median"]),
            g(c["F0_vs_solver_gp"]["max"]), g(c["dF_vs_central_differences_max_rel"]),
            "yes" if e["stereo"]["point_contains_double"] and e["gp"]["point_contains_double"] else "NO",
            e["stereo"]["box_random_points_outside"], e["gp"]["box_random_points_outside"]))
    (R / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps(gate, indent=1))


if __name__ == "__main__":
    main()
