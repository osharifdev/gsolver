"""Markdown tables of the D2/D6 results (results/phaseC/d2/*.json)."""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
D = REPO / "results" / "phaseC" / "d2"
NAMES = ["charuco_islam_025", "charuco_islam_050", "charuco_islam_100", "charuco_ba_init", "charuco_ba_end",
         "kitti06_prefix_conv", "kitti06_prefix_init", "kitti06_full_conv"]


def g(x, f="%.3g"):
    return "-" if x is None else (f % x if isinstance(x, (int, float)) else str(x))


def main():
    print("| dataset | p_L | blocks | poses | landmarks/markers | single-obs eliminated | zero-info excluded | "
          "rel. residual | lambda_min | lambda_max | near-null dim | pose floor | reached | t_floor | worst pose | "
          "#landmarks obs-info < 10 p_L |")
    print("|" + "---|" * 16)
    for n in NAMES:
        p = D / ("%s.json" % n)
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for m in d["models"]:
            f = m["pose_floor"]
            print("| %s | %s | %d | %d | %s | %s | %s | %.1e | %.3e | %.3e | %d | %.3e | %s | %s | %s | %s |" % (
                n, g(m["p_L"]), m["n_blocks"], m["n_poses"], "%d / %d" % (m["n_landmarks"], m["n_markers"]),
                m["single_edge_eliminated"], m["zero_information_excluded"], m["rel_residual"], m["lambda_min"],
                m["lambda_max"], m["near_null_dim"], f["floor"], f["floor_reached"], g(f["t_floor"], "%.0f"),
                m["pose_floor_worst_block"]["pose_index"], g(m.get("n_landmarks_obs_info_below_10pL"), "%d")))
    print()
    print("| dataset | D6 raw: max abs diff H (rel) | D6 raw: max abs diff b | D6 vs Phase A model.npz (rel) | "
          "D6 certified: rel diff H | D6 certified: max abs diff b | pose change vs p_L=1e-6 (1e-7 / 1e-5) | insensitive | floor <= 0.05 |")
    print("|" + "---|" * 9)
    for n in NAMES:
        p = D / ("%s.json" % n)
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        raw = d["d6_raw_vs_production_assembler"]
        npz = d.get("d6_raw_vs_phaseA_model_npz")
        cert = [m for m in d["models"] if "d6_certified" in m][0]["d6_certified"]
        sens = d.get("pose_solution_change_vs_1e-6")
        print("| %s | %.2e (%.1e) | %.2e | %s | %.1e | %.2e | %s | %s | %s |" % (
            n, raw["max_abs_diff_H"], raw["rel_diff_H"], raw["max_abs_diff_b"],
            "-" if npz is None else "%.1e" % npz["rel_diff_H"], cert["rel_diff_H"], cert["max_abs_diff_b"],
            "-" if sens is None else "%.3f / %.3f" % (sens["1e-07"], sens["1e-05"]), g(d.get("insensitive")), d["floor_pass"]))
    for n in ("kitti06_prefix_conv", "kitti06_prefix_init", "kitti06_full_conv"):
        p = D / ("attribution_%s.json" % n)
        if not p.exists():
            continue
        a = json.loads(p.read_text())
        print("\n%s (p_L = 1e-6), last frozen iteration %d: no holding floor %.3e (pose %d, %d poses > 0.05)" % (
            n, a["last_iteration"], a["no_holding"]["pose_floor"], a["no_holding"]["worst_pose"], a["no_holding"]["n_poses_above_0_05"]))
        print("| obs-info lambda_min < | landmarks | floor, held at GBP's last state | worst pose | median | poses > 0.05 | floor, clamped at linearization point |")
        print("|---|---|---|---|---|---|---|")
        for h in a["held"]:
            print("| %g | %d | %.3e | %d | %.2e | %d | %s |" % (h["tau"], h["n_held"], h["pose_floor"], h["worst_pose"], h["median"],
                  h["n_poses_above_0_05"], "%.3e" % h["clamped_at_linpoint_pose_floor"] if "clamped_at_linpoint_pose_floor" in h else "-"))


if __name__ == "__main__":
    main()
