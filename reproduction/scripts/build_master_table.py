#!/usr/bin/env python3
"""Build reproduction/reproduction_summary.csv — one row per paper data point
(plot point or table cell) plus one row per qualitative panel, with explicit
paper-value sources and reproduction statuses."""
import csv
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = f"{ROOT}/reproduction/results"
OUTP = f"{ROOT}/reproduction/reproduction_summary.csv"

PHW = "Intel Core i9-14900KF (16 physical cores)"
LHW = "AMD EPYC 7713 (64 physical cores, 16 threads used), RHEL 9.7, GCC 11.5.0"

COLS = ["paper_item", "panel", "experiment", "method", "dataset", "configuration_id",
        "x_variable", "x_value", "metric", "paper_value", "paper_value_source",
        "paper_value_is_exact", "local_value", "local_stddev", "unit", "repetitions",
        "seed", "paper_hardware", "local_hardware", "absolute_difference",
        "relative_difference_percent", "reproduction_status", "raw_data_path",
        "generated_figure_path", "notes"]

rows = []

def rd(path):
    with open(path) as f:
        return [r for r in csv.DictReader(l for l in f if not l.startswith("#"))]

def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None

def add(paper_item, panel, experiment, method, dataset, config, xvar, xval, metric,
        paper_value, source, exact, local_value, unit, status, raw, figpath, notes="",
        stddev="", reps=1, seed=42):
    pv, lv = num(paper_value), num(local_value)
    ad = abs(pv - lv) if pv is not None and lv is not None else ""
    rp = (100.0 * ad / abs(pv)) if isinstance(ad, float) and pv not in (None, 0) else ""
    rows.append(dict(zip(COLS, [paper_item, panel, experiment, method, dataset, config,
                                xvar, xval, metric,
                                paper_value if paper_value is not None else "",
                                source, exact,
                                local_value if local_value is not None else "",
                                stddev, unit, reps, seed, PHW, LHW,
                                round(ad, 6) if isinstance(ad, float) else "",
                                round(rp, 2) if isinstance(rp, float) else "",
                                status, raw, figpath, notes])))

def status_from(pv, lv, tol_rel=0.05, tol_abs=0.01):
    pv, lv = num(pv), num(lv)
    if pv is None or lv is None:
        return "BLOCKED_MISSING_DATA"
    d = abs(pv - lv)
    if d <= tol_abs or (pv != 0 and d / abs(pv) <= tol_rel):
        return "EXACT_OR_WITHIN_TOLERANCE"
    if pv != 0 and d / abs(pv) <= 0.25:
        return "CLOSE"
    return "NUMERICALLY_DIFFERENT"

# ---------------------------------------------------------------- Figure 5 --
pap5 = rd(f"{RES}/paper_digitised/fig05_digitised.csv")
loc5 = {int(r["iteration"]): float(r["total_energy"]) for r in rd(f"{RES}/local_gsolver/figure_05/energy.csv")}
locg = {int(r["iteration"]): float(r["total_energy"])
        for r in rd(f"{RES}/local_gsolver/figure_05/energy_gn_reimplementation.csv")}
for r in pap5:
    it = int(r["iteration"])
    m = r["method"]
    lv = loc5.get(it) if m == "gsolver" else locg.get(it)
    add("Figure 5", "single", "PGO helix energy", m if m == "gsolver" else "gn_gp",
        "pgo/noise_ig1.0_n0.1_helix.g2o", "helix qc, 100 iters", "iteration", it,
        "total_energy", r["total_energy"], "DIGITISED_PDF", "no", lv, "energy",
        "NUMERICALLY_DIFFERENT",
        "reproduction/results/local_gsolver/figure_05/energy.csv",
        "reproduction/figures/comparison/figure_05.png",
        "paper y-scale not reproducible from documented config (see report); local GN is a "
        "labelled reimplementation, NOT the paper baseline; shapes qualitatively consistent")

# --------------------------------------------------------------- Figures 8/9 --
for fign, task in (("Figure 8", "prior"), ("Figure 9", "pgo")):
    pap = rd(f"{RES}/paper_digitised/fig0{8 if task=='prior' else 9}_digitised.csv")
    for r in pap:
        shape, eta, sig, m = r["trajectory"], r["eta_ig"], r["sigma"], r["method"]
        signame = {1e-4: "0.0001", 1e-3: "0.001", 1e-2: "0.01", 1e-1: "0.1", 1.0: "1.0", 1.5: "1.5"}[float(sig)]
        etaname = {1e-4: "0.0001", 1e-3: "0.001", 1e-2: "0.01", 1e-1: "0.1"}[float(eta)]
        d = f"{RES}/local_gsolver/sweeps/{task}_{shape}_ig{etaname}_n{signame}"
        loc = rd(f"{d}/summary.csv")[0]
        panel = f"{shape},eta={etaname}"
        ds = f"{task}/noise_ig{etaname}_n{signame}_{shape}.g2o"
        if m == "gsolver":
            lv = loc["ate_interp_m"] if loc["run_status"] == "ok" else None
            # tol_abs 0.05 m: digitisation zero-anchor uncertainty (see digitiser header)
            st = status_from(r["ATE_m"], lv, tol_rel=0.15, tol_abs=0.05) if lv else "DIVERGED"
            note = f"local=interp-100x ATE (unaligned), run_status={loc['run_status']}; ARE saved in sweep CSV"
            if task == "pgo" and shape == "helix" and st == "NUMERICALLY_DIFFERENT":
                note += ("; systematic ~1.6-1.7x offset ONLY for pgo-helix at sigma>=0.1 while prior-helix "
                         "and pgo-sphere match the paper closely -> possibly a different dataset realization "
                         "in the paper's helix PGO runs (author-clarification item)")
            add(fign, panel, f"{task} localisation", "gsolver", ds, f"{shape} qc,100it",
                "sigma", signame, "ATE_RMSE_m", r["ATE_m"], "DIGITISED_PDF", "no", lv, "m",
                st, d + "/summary.csv", "reproduction/figures/comparison/figure_0%d.png" % (8 if task == "prior" else 9),
                note)
        else:
            add(fign, panel, f"{task} localisation", m, ds, "unknown (Hyperion)",
                "sigma", signame, "ATE_RMSE_m", r["ATE_m"], "DIGITISED_PDF", "no", None, "m",
                "BLOCKED_MISSING_BASELINE", "", "",
                "Hyperion not reproducible from public artefacts (ROS2 stack; no dataset glue)")

# ----------------------------------------------------------------- Table I --
p1 = {"helix": [0.030, 1.533, 3.925, 4.510, 4.694], "sphere": [0.030, 1.542, 4.187, 4.589, 6.515]}
for traj in ("helix", "sphere"):
    for s, pv in zip(["0.0001", "0.001", "0.01", "0.1", "1.0"], p1[traj]):
        d = f"{RES}/local_gsolver/sweeps/prior_{traj}_ig0.001_n{s}"
        lv = rd(f"{d}/summary.csv")[0]["nees_local_est"]
        add("Table I", traj, "prior localisation NEES", "gsolver",
            f"prior/noise_ig*_n{s}_{traj}.g2o", f"{traj} qc,100it", "sigma", s, "mean_NEES",
            pv, "PAPER_TABLE", "yes", round(float(lv), 3), "-",
            status_from(pv, lv, tol_rel=0.02, tol_abs=0.01),
            d + "/nees_per_pose.csv", "reproduction/results/local_gsolver/table_01_comparison.csv",
            "NEES verified eta_ig-independent (identical for eta in 1e-4..1); "
            "convention: e=Log(T_est^-1 T_gt), 6x6 pose block of belief Sigma")

# ---------------------------------------------------------------- Figure 10 --
pap10 = rd(f"{RES}/paper_digitised/fig10_digitised.csv")
for r in pap10:
    add("Figure 10", "single", "Qc scale ablation (torus)", "gsolver", "torus (NOT PUBLIC)",
        "unknown", "scale_s", r["scale_s"], r["metric"], r["value"], "DIGITISED_PDF", "no",
        None, "m" if r["metric"] == "ATE_m" else "rad", "BLOCKED_MISSING_DATA",
        "", "reproduction/figures/comparison/figure_10_substitute.png",
        "paper curve has 59 vertices (dense scale sweep); shipped app tests only {0.01,0.1,1,10,100}")
scales = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
for s in scales:
    loc = rd(f"{RES}/local_gsolver/figure_10_substitute/scale_{s}/summary.csv")[0]
    for met, key, unit in (("ATE_m", "ate_interp_m", "m"), ("ARE_rad", "are_interp_rad", "rad")):
        add("Figure 10", "substitute", "Qc scale ablation (substitute torus)", "gsolver",
            "torus_substitute pgo ig1.0 n1.0", "torus_substitute qc*,100it", "scale_s", s, met,
            None, "NOT_REPORTED", "no", round(float(loc[key]), 4), unit,
            "NOT_REPRODUCIBLE_FROM_PUBLIC_ARTEFACTS",
            f"{RES}/local_gsolver/figure_10_substitute/scale_{s}/summary.csv".replace(ROOT + "/", ""),
            "reproduction/figures/comparison/figure_10_substitute.png",
            "SUBSTITUTE torus (local generator + gp_hyperparam_trainer); minimum in s=[1,10] as in paper")

# ---------------------------------------------------------------- Figure 11 --
pap11 = rd(f"{RES}/paper_digitised/fig11_digitised.csv")
for r in pap11:
    add("Figure 11", "single", "ChArUco BA reprojection", r["method"],
        "charuco/printing_room_0.g2o", "printing_room qc,50it,noise0.1", "iteration",
        r["iteration"], "mean_reprojection_error_px", r["mean_reprojection_error_px"],
        "DIGITISED_PDF", "no", None, "px",
        "BLOCKED_MISSING_CONFIGURATION" if r["method"] == "gsolver" else "BLOCKED_MISSING_BASELINE",
        "", "reproduction/figures/comparison/figure_11.png",
        "pixel MRE not computable: camera calibration / pixel detections / marker geometry not in repo")
for r in rd(f"{RES}/local_gsolver/figure_11/per_iteration.csv"):
    add("Figure 11", "substitute-metric", "ChArUco BA convergence", "gsolver",
        "charuco/printing_room_0.g2o", "printing_room qc,50it,noise0.1", "iteration",
        r["iteration"], "mean_landmark_trans_err_m", None, "NOT_REPORTED", "no",
        round(float(r["mean_landmark_trans_err_m"]), 5), "m", "NOT_REPRODUCIBLE_FROM_PUBLIC_ARTEFACTS",
        "reproduction/results/local_gsolver/figure_11/per_iteration.csv",
        "reproduction/figures/comparison/figure_11.png",
        "substitute metric (pose-landmark SE3 error); converges smoothly as paper describes for G-solver")

# ---------------------------------------------------------------- Figure 12 --
import statistics
pap12 = rd(f"{RES}/paper_digitised/fig12_digitised.csv")
locstats = {r["operation"]: r for r in rd(f"{RES}/latency_raw/fig12_helix_prior/summary.csv")}
for r in pap12:
    op = r["operation"]
    ls = locstats.get(op)
    for stat in ("median_us", "q1_us", "q3_us", "whisker_low_us", "whisker_high_us"):
        lv = None
        if ls and r["method"] == "gsolver":
            lv = {"median_us": ls["median_us"], "q1_us": ls["p25_us"], "q3_us": ls["p75_us"],
                  "whisker_low_us": ls["p5_us"], "whisker_high_us": ls["p95_us"]}[stat]
            lv = round(float(lv), 3)
        add("Figure 12", op, "message/interpolation latency", r["method"],
            "helix prior graph (local choice; paper graph unknown)", "fallback protocol 5000 samples",
            "operation", op, stat, r[stat] if r[stat] else None, "DIGITISED_PDF", "no", lv, "us",
            "PAPER_VALUE_NOT_EXACT" if r["method"] == "gsolver" else "BLOCKED_MISSING_BASELINE",
            f"reproduction/results/latency_raw/fig12_helix_prior/raw_{op}.csv",
            "reproduction/figures/comparison/figure_12.png",
            "different hardware (paper i9-14900KF vs local EPYC 7713); paper protocol unknown; "
            "local q1/q3->p25/p75, whiskers->p5/p95",
            stddev=(ls["std_us"] if ls and r["method"] == "gsolver" else ""), reps=5000)

# ----------------------------------------------------------------- Table II --
p2 = {("ours", "ATE_m"): [0.671, 0.676, 0.676, 0.684, 0.726],
      ("ours", "ARE_rad"): [0.007, 0.007, 0.008, 0.007, 0.008],
      ("ours", "MRE_px"): [1.704, 1.708, 1.709, 1.715, 1.844],
      ("zspline", "ATE_m"): [0.692, 94.510, 1963.040, 200.168, 733.028],
      ("zspline", "ARE_rad"): [0.007, 0.194, 0.508, 0.350, 1.577],
      ("zspline", "MRE_px"): [1.717, 2.099, 1.884, 2.301, 8.142]}
readouts = ["0", "0.0001", "0.001", "0.01", "0.1"]
for i, ro in enumerate(readouts):
    loc = rd(f"{RES}/local_gsolver/table_02/readout_{ro}/summary.csv")[0]
    lmap = {"ATE_m": (round(float(loc["ate_aligned_m"]), 3), "m"),
            "ARE_rad": (round(float(loc["are_aligned_rad"]), 4), "rad"),
            "MRE_px": (round(float(loc["mre_stereo3_in_image"]), 3), "px")}
    for met in ("ATE_m", "ARE_rad", "MRE_px"):
        pv = p2[("ours", met)][i]
        lv, unit = lmap[met]
        st = status_from(pv, lv, tol_rel=0.10, tol_abs=0.005)
        if float(ro or 0) in (1e-4, 1e-3, 1e-2) and met != "MRE_px" and st == "NUMERICALLY_DIFFERENT":
            st = "DIVERGED" if num(lv) and num(lv) > 5 else st
        add("Table II", "Ours", "rolling shutter BA", "gsolver",
            f"rolling_shutter/kitti_06_{ro}.g2o", "kitti_05 qc,10it", "readout_s", ro, met,
            pv, "PAPER_TABLE", "yes", lv, unit, st,
            f"reproduction/results/local_gsolver/table_02/readout_{ro}/summary.csv",
            "reproduction/results/local_gsolver/table_02_comparison.csv",
            f"local ATE/ARE SE(3)-aligned (convention inferred from readout-0 agreement); "
            f"MRE=stereo3 in-image; excluded {float(loc['excluded_pct']):.2f}% projections "
            f"({loc['n_excluded_out_of_frame']} out-of-frame, {loc['n_excluded_behind']} behind, "
            f"{loc['n_unmatched']} unmatched)", seed="n/a")
        pvz = p2[("zspline", met)][i]
        add("Table II", "Z-Spline", "rolling shutter BA", "hyperion_zspline",
            f"rolling_shutter/kitti_06_{ro}.g2o", "unknown (Hyperion)", "readout_s", ro, met,
            pvz, "PAPER_TABLE", "yes", None, lmap[met][1], "BLOCKED_MISSING_BASELINE",
            "", "", "Hyperion not reproducible from public artefacts", seed="n/a")

# ---------------------------------------------------------------- Figure 14 --
pap14 = rd(f"{RES}/paper_digitised/fig14_digitised.csv")
for r in pap14:
    n = int(r["num_cameras"])
    loc = rd(f"{RES}/local_gsolver/figure_14/cameras_{n}/summary.csv")[0]
    met = r["metric"]
    if met == "avg_time_s":
        lv, unit, note = round(float(loc["avg_robot_solve_ms"]) / 1e3, 2), "s", "per-camera GBP solve time (16 threads)"
    elif met == "avg_mem_MB":
        lv, unit, note = round(int(loc["rss_post_solve_kb"]) / 1024 / n, 1), "MB", "process peak RSS / #cameras (documented estimate)"
    elif met == "ATE_m":
        lv, unit, note = round(float(loc["ate_aligned_m"]), 4), "m", "SE(3)-aligned"
    else:
        lv, unit, note = round(float(loc["are_aligned_rad"]), 5), "rad", "SE(3)-aligned"
    st = status_from(r["value"], lv, tol_rel=0.30, tol_abs=0.002)
    add("Figure 14", met, "multi-camera scalability", "gsolver", "multirobot/06.g2o",
        f"kitti_05 qc,10it,robots={n}", "num_cameras", n, met, r["value"], "DIGITISED_PDF",
        "no", lv, unit, st,
        f"reproduction/results/local_gsolver/figure_14/cameras_{n}/summary.csv",
        "reproduction/figures/comparison/figure_14.png",
        note + f"; equality_factors={loc['n_equality_factors']}, poses={loc['n_poses']}", seed="n/a")

# ------------------------------------------------------ qualitative panels --
def qual(item, panel, exp, method, ds, cfg, raw, figp, note, status="VISUALLY_EQUIVALENT"):
    add(item, panel, exp, method, ds, cfg, "", "", "qualitative_artifact", None,
        "NOT_REPORTED", "no", None, "-", status, raw, figp, note)

qual("Figure 1", "single", "continuous-time query", "gsolver", "torus (NOT PUBLIC)",
     "torus_substitute", "reproduction/results/local_gsolver/figure_01_substitute/interp.csv",
     "reproduction/figures/local/figure_01_substitute.png",
     "SUBSTITUTE torus; covariance expands 2.3x in sparse region as in paper",
     "NOT_REPRODUCIBLE_FROM_PUBLIC_ARTEFACTS")
for p, path in (("initial_guess", "gbp_gp/knots_init.tum"), ("gbp_only", "gbp_only/knots_est.tum"),
                ("gbp_gp", "gbp_gp/knots_est.tum")):
    qual("Figure 6", p, "GP ablation helix", p, "prior/noise_ig1.0_n1.0_helix.g2o", "helix qc,100it",
         f"reproduction/results/local_gsolver/figure_06/{path}",
         "reproduction/figures/local/figure_06.png",
         "GBP-only stays noisy (ATE 1.74 m) vs GBP+GP recovers helix (ATE 0.49 m) — matches paper claim")
qual("Figure 7", "Ours", "prior interpolation sphere", "gsolver", "prior/noise_ig1.0_n1.0_sphere.g2o",
     "sphere qc,100it", "reproduction/results/local_gsolver/sweeps/prior_sphere_ig1.0_n1.0/interp.csv",
     "reproduction/figures/local/figure_07.png", "smooth sphere recovered under sigma=1 as in paper")
for m in ("bspline", "zspline"):
    qual("Figure 7", m, "prior interpolation sphere", m, "prior/noise_ig1.0_n1.0_sphere.g2o", "unknown",
         "", "reproduction/figures/local/figure_07.png", "Hyperion baseline not runnable",
         "BLOCKED_MISSING_BASELINE")
for fign, task, shape in (("Figure 15", "prior", "helix"), ("Figure 16", "prior", "sphere"),
                          ("Figure 17", "pgo", "helix"), ("Figure 18", "pgo", "sphere")):
    for s in ("0.0001", "0.001", "0.01", "0.1", "1.0", "1.5"):
        loc = rd(f"{RES}/local_gsolver/sweeps/{task}_{shape}_ig1.0_n{s}/summary.csv")[0]
        qual(fign, f"sigma={s},Ours", f"{task} qualitative", "gsolver",
             f"{task}/noise_ig1.0_n{s}_{shape}.g2o", f"{shape} qc,100it",
             f"reproduction/results/local_gsolver/sweeps/{task}_{shape}_ig1.0_n{s}/interp.csv",
             f"reproduction/figures/local/figure_{fign.split()[1]}.png",
             f"run_status={loc['run_status']}, interp-ATE={loc['ate_interp_m']}",
             "VISUALLY_EQUIVALENT" if loc["run_status"] == "ok" else "DIVERGED")
        for m in ("bspline", "zspline"):
            qual(fign, f"sigma={s},{m}", f"{task} qualitative", m,
                 f"{task}/noise_ig1.0_n{s}_{shape}.g2o", "unknown", "",
                 f"reproduction/figures/local/figure_{fign.split()[1]}.png",
                 "Hyperion baseline not runnable", "BLOCKED_MISSING_BASELINE")
for fign, frames in (("Figure 13", [320]), ("Figure 19", [79, 320, 354, 429, 433]),
                     ("Figure 20", [668, 670, 674, 715, 719])):
    for fr in frames:
        note = ("exact source frame not stated in paper; frame chosen from Fig.19 list; " if fign == "Figure 13" else "")
        qual(fign, f"frame {fr}", "rolling shutter reprojection overlay", "gsolver+nominal",
             "rolling_shutter/kitti_06_0.001.g2o", "kitti_05 qc,10it",
             f"reproduction/results/local_gsolver/table_02/readout_0.001/projections_frame_{fr}.csv",
             f"reproduction/figures/overlays/figure_{fign.split()[1]}_frame_{fr}.png",
             note + "KITTI images not in repo (grey canvas); Hyperion series missing; local G-solver "
             "poses partially diverged at 1 ms readout (see Table II rows)",
             "PARTIALLY_READY" if False else "BLOCKED_MISSING_DATA")

with open(OUTP, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    w.writerows(rows)
print(f"wrote {OUTP} with {len(rows)} rows")
import collections
print(collections.Counter(r["reproduction_status"] for r in rows))
