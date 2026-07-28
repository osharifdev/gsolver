#!/usr/bin/env python3
"""Build every reproduction figure and the master comparison table.

All plots are generated exclusively from machine-readable CSV/TUM files under
reproduction/results/. Local G-solver series are drawn in the paper's own
matplotlib tab10 colours (solid, filled markers at every sampled point);
paper-reference series obtained by PDF digitisation are drawn dashed with open
markers and grey-outlined legend entries labelled "paper (digitised)".
Substitute artefacts (torus) are watermarked SUBSTITUTE.

Usage: build_all_figures.py [--only fig05,fig08,...]  (default: everything)
"""
import csv
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = f"{ROOT}/reproduction/results"
FIG = f"{ROOT}/reproduction/figures"
for sub in ("local", "paper_reference", "comparison", "overlays"):
    os.makedirs(f"{FIG}/{sub}", exist_ok=True)

C = {"gsolver": "#1f77b4", "bspline": "#ff7f0e", "zspline": "#2ca02c",
     "gn": "#000000", "red": "#d62728", "teal": "#008080", "purple": "#800080"}
PAPER_STYLE = dict(linestyle="--", color="#555555", markerfacecolor="none")

def rd(path):
    with open(path) as f:
        rows = [r for r in csv.DictReader(l for l in f if not l.startswith("#"))]
    return rows

def rd_tum(path):
    out = []
    for l in open(path):
        if l.startswith("#") or not l.strip():
            continue
        v = [float(x) for x in l.split()]
        out.append(v)
    return np.array(out)

def savefig(fig, name, subdir="local"):
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG}/{subdir}/{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"figure {subdir}/{name}")

def watermark(ax, text="SUBSTITUTE (original torus not public)"):
    ax.text(0.5, 0.02, text, transform=ax.transAxes, ha="center", fontsize=7,
            color="#a00000", alpha=0.85)

# ---------------------------------------------------------------- Figure 1 --
def fig01():
    rows = rd(f"{RES}/local_gsolver/figure_01_substitute/interp.csv")
    t = np.array([float(r["t"]) for r in rows])
    x = np.array([float(r["x"]) for r in rows])
    y = np.array([float(r["y"]) for r in rows])
    z = np.array([float(r["z"]) for r in rows])
    sig = np.sqrt(np.array([float(r["cov_xx"]) + float(r["cov_yy"]) + float(r["cov_zz"]) for r in rows]))
    knots = rd_tum(f"{RES}/local_gsolver/figure_01_substitute/knots_est.tum")
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    step = 3
    p = ax.scatter(x[::step], y[::step], z[::step], c=sig[::step], cmap="Blues", s=np.clip(2 + 400 * sig[::step] ** 2, 2, 60),
                   alpha=0.5, linewidths=0)
    ax.plot(x, y, z, color="#1f77b4", lw=0.7)
    ax.scatter(knots[::6, 1], knots[::6, 2], knots[::6, 3], color="k", s=4, label="pose knots (1/6 shown)")
    fig.colorbar(p, ax=ax, shrink=0.55, label="position std [m] (marker size ∝ covariance)")
    ax.set_title("Fig. 1 SUBSTITUTE — continuous-time mean + covariance\n"
                 "(local torus; sparse priors in trajectory fraction 0.55–0.80)")
    ax.legend(loc="upper left", fontsize=7)
    savefig(fig, "figure_01_substitute")

# ---------------------------------------------------------------- Figure 5 --
def fig05():
    loc = rd(f"{RES}/local_gsolver/figure_05/energy.csv")
    gn = rd(f"{RES}/local_gsolver/figure_05/energy_gn_reimplementation.csv")
    pap = rd(f"{RES}/paper_digitised/fig05_digitised.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    it = [int(r["iteration"]) for r in loc if int(r["iteration"]) <= 25]
    E = [float(r["total_energy"]) for r in loc if int(r["iteration"]) <= 25]
    a1.plot(it, E, "o-", ms=3.5, color=C["gsolver"], label="G-solver (local, Eq.18)")
    itg = [int(r["iteration"]) for r in gn if int(r["iteration"]) <= 25]
    Eg = [float(r["total_energy"]) for r in gn if int(r["iteration"]) <= 25]
    a1.plot(itg, Eg, "s-", ms=3.5, color=C["gn"], label="GN+GP reimplementation (local, supplementary)")
    a1.set_yscale("log")
    a1.set_xlabel("# iteration"); a1.set_ylabel("$E_{tot}$ (Eq. 18, log)")
    a1.set_title("Local (documented config: PGO helix,\nσ=0.1 m/0.01 rad, $\\eta_{ig}$=1 m/0.1 rad)")
    a1.legend(fontsize=7); a1.grid(alpha=0.3)
    for m, mk in (("gsolver", "o"), ("gn_gp", "s")):
        pts = sorted((int(r["iteration"]), float(r["total_energy"])) for r in pap if r["method"] == m)
        a2.plot([p[0] for p in pts], [p[1] for p in pts], mk + "--", ms=4,
                color=C["gsolver"] if m == "gsolver" else C["gn"], markerfacecolor="none",
                label=f"{'Ours' if m=='gsolver' else 'GN+GP'} — paper (digitised)")
    a2.set_xlabel("# iteration"); a2.set_ylabel("$E_{tot}$")
    a2.set_title("Paper Fig. 5 (digitised approximation\nfrom published figure)")
    a2.legend(fontsize=7); a2.grid(alpha=0.3)
    fig.suptitle("Figure 5 — total energy over iterations (absolute scales differ; see report §Fig.5)", fontsize=9, y=1.06)
    fig.tight_layout()
    savefig(fig, "figure_05", "comparison")

# ---------------------------------------------------------------- Figure 6 --
def fig06():
    base = f"{RES}/local_gsolver/figure_06"
    panels = [("Initial Guess", f"{base}/gbp_gp/knots_init.tum", "k"),
              ("GBP-only", f"{base}/gbp_only/knots_est.tum", "#8b4513"),
              ("Ours (GBP+GP)", f"{base}/gbp_gp/knots_est.tum", C["gsolver"])]
    fig = plt.figure(figsize=(10.5, 3.6))
    for i, (title, path, col) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        d = rd_tum(path)
        ax.plot(d[:, 1], d[:, 2], d[:, 3], ".", ms=1.5, color=col)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-15, 15); ax.set_ylim(-15, 15)
    fig.suptitle("Figure 6 — Ablation on GP (helix, prior-based, σ=1 m/0.1 rad, $\\eta_{ig}$=1 m/0.1 rad) — local", fontsize=9)
    savefig(fig, "figure_06")

# ---------------------------------------------------- Figures 7 and 15-18 --
def qual_panels(figname, task, shape, sigmas, title):
    fig = plt.figure(figsize=(10.5, 3.2 * len(sigmas)))
    for si, s in enumerate(sigmas):
        d = f"{RES}/local_gsolver/sweeps/{task}_{shape}_ig1.0_n{s}"
        rows = rd(f"{d}/interp.csv")
        x = [float(r["x"]) for r in rows][::5]
        y = [float(r["y"]) for r in rows][::5]
        z = [float(r["z"]) for r in rows][::5]
        ax = fig.add_subplot(len(sigmas), 3, 3 * si + 1, projection="3d")
        ax.plot(x, y, z, lw=0.5, color=C["gsolver"])
        ax.set_title(f"Ours — σ={s}", fontsize=8)
        for j, (nm, lab) in enumerate((("bspline", "B-Spline"), ("zspline", "Z-Spline"))):
            axb = fig.add_subplot(len(sigmas), 3, 3 * si + 2 + j, projection="3d")
            axb.text2D(0.5, 0.5, f"{lab}\nMISSING_BASELINE\n(Hyperion not reproducible\nfrom public artefacts)",
                       ha="center", va="center", fontsize=7, color="#777777", transform=axb.transAxes)
            axb.set_axis_off()
    fig.suptitle(title + " — local G-solver (100× interpolation); baselines not runnable", fontsize=9)
    savefig(fig, figname)

def fig07():
    qual_panels("figure_07", "prior", "sphere", ["1.0"],
                "Figure 7 — prior-based interpolation, sphere, σ=1, $\\eta_{ig}$=1")

def fig15_18():
    sigmas = ["0.0001", "0.001", "0.01", "0.1", "1.0", "1.5"]
    qual_panels("figure_15", "prior", "helix", sigmas, "Figure 15 — prior-based, helix, $\\eta_{ig}$=1")
    qual_panels("figure_16", "prior", "sphere", sigmas, "Figure 16 — prior-based, sphere, $\\eta_{ig}$=1")
    qual_panels("figure_17", "pgo", "helix", sigmas, "Figure 17 — PGO, helix, $\\eta_{ig}$=1")
    qual_panels("figure_18", "pgo", "sphere", sigmas, "Figure 18 — PGO, sphere, $\\eta_{ig}$=1")

# ------------------------------------------------------------- Figures 8/9 --
SIGMAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1.5]
SIGNAMES = ["0.0001", "0.001", "0.01", "0.1", "1.0", "1.5"]
ETAS = ["0.0001", "0.001", "0.01", "0.1"]

def sweep_local(task, shape, eta, signame):
    d = f"{RES}/local_gsolver/sweeps/{task}_{shape}_ig{eta}_n{signame}"
    r = rd(f"{d}/summary.csv")[0]
    return r

def fig0809(task, figname, papercsv):
    pap = rd(f"{RES}/paper_digitised/{papercsv}")
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True)
    for ri, shape in enumerate(("helix", "sphere")):
        for ci, eta in enumerate(ETAS):
            ax = axes[ri][ci]
            ate = [float(sweep_local(task, shape, eta, sn)["ate_interp_m"]) for sn in SIGNAMES]
            ax.plot(SIGMAS, ate, "o-", ms=4, color=C["gsolver"], label="Ours (local)")
            for m, lab in (("gsolver", "Ours"), ("bspline", "B-Spline"), ("zspline", "Z-Spline")):
                pts = [(float(r["sigma"]), float(r["ATE_m"])) for r in pap
                       if r["trajectory"] == shape and float(r["eta_ig"]) == float(eta) and r["method"] == m]
                pts.sort()
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], "^--", ms=4,
                            color=C[m], markerfacecolor="none", alpha=0.85,
                            label=f"{lab} — paper (digitised)")
            ax.set_xscale("log")
            ax.grid(alpha=0.3)
            if ri == 0:
                ax.set_title(f"$\\eta_{{ig}}$ = {eta}", fontsize=9)
            if ci == 0:
                ax.set_ylabel(f"{shape.capitalize()} ATE [m]")
            if ri == 1:
                ax.set_xlabel("σ")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"{figname.replace('_',' ').title()} — {task} experiments: local G-solver vs digitised paper curves "
                 "(Hyperion baselines not runnable locally)", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    savefig(fig, figname, "comparison")

def fig08():
    fig0809("prior", "figure_08", "fig08_digitised.csv")

def fig09():
    fig0809("pgo", "figure_09", "fig09_digitised.csv")

# ---------------------------------------------------------------- Figure 10 --
def fig10():
    scales = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    ate, are = [], []
    for s in scales:
        r = rd(f"{RES}/local_gsolver/figure_10_substitute/scale_{s}/summary.csv")[0]
        ate.append(float(r["ate_interp_m"]))
        are.append(float(r["are_interp_rad"]))
    pap = rd(f"{RES}/paper_digitised/fig10_digitised.csv")
    p_ate = sorted((float(r["scale_s"]), float(r["value"])) for r in pap if r["metric"] == "ATE_m")
    p_are = sorted((float(r["scale_s"]), float(r["value"])) for r in pap if r["metric"] == "ARE_rad")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    a1b = a1.twinx()
    a1.plot(scales, ate, "o-", color=C["zspline"], ms=4, label="ATE [m] (local substitute)")
    a1b.plot(scales, are, "s-", color=C["red"], ms=4, label="ARE [rad] (local substitute)")
    off = [s for s in (0.01, 0.1, 1.0, 10.0, 100.0)]
    for s in off:
        a1.axvline(s, color="#bbbbbb", lw=0.5, zorder=0)
    a1.set_xscale("log"); a1.set_xlabel("hyperparameters scale s")
    a1.set_ylabel("ATE [m]", color=C["zspline"]); a1b.set_ylabel("ARE [rad]", color=C["red"])
    a1.set_title("Local SUBSTITUTE torus (official app scales marked;\ndual axis matches paper layout)", fontsize=8)
    watermark(a1)
    a2b = a2.twinx()
    a2.plot([p[0] for p in p_ate], [p[1] for p in p_ate], "--", color=C["zspline"], label="ATE (paper, digitised)")
    a2b.plot([p[0] for p in p_are], [p[1] for p in p_are], "--", color=C["red"], label="ARE (paper, digitised)")
    a2.set_xscale("log"); a2.set_xlabel("hyperparameters scale s")
    a2.set_ylabel("ATE [m]", color=C["zspline"]); a2b.set_ylabel("ARE [rad]", color=C["red"])
    a2.set_title("Paper Fig. 10 (digitised approximation;\n59 curve vertices in published figure)", fontsize=8)
    fig.suptitle("Figure 10 — hyperparameter scale ablation (original torus not public)", fontsize=10)
    fig.tight_layout()
    savefig(fig, "figure_10_substitute", "comparison")

# ---------------------------------------------------------------- Figure 11 --
def fig11():
    loc = rd(f"{RES}/local_gsolver/figure_11/per_iteration.csv")
    pap = rd(f"{RES}/paper_digitised/fig11_digitised.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    it = [int(r["iteration"]) for r in loc]
    a1.plot(it, [float(r["mean_landmark_trans_err_m"]) for r in loc], "o-", ms=3, color=C["gsolver"],
            label="mean marker translation err [m]")
    a1b = a1.twinx()
    a1b.plot(it, [float(r["mean_landmark_rot_err_rad"]) for r in loc], "s-", ms=3, color=C["red"],
             label="mean marker rotation err [rad]")
    a1.set_xlabel("Iteration"); a1.set_ylabel("translation err [m]", color=C["gsolver"])
    a1b.set_ylabel("rotation err [rad]", color=C["red"])
    a1.set_title("Local SUBSTITUTE metric (pose–landmark error;\npixel MRE not computable: no calibration in repo)", fontsize=8)
    a1.grid(alpha=0.3)
    for m, lab in (("gsolver", "Ours"), ("bspline", "B-Spline"), ("zspline", "Z-Spline")):
        pts = sorted((float(r["iteration"]), float(r["mean_reprojection_error_px"])) for r in pap if r["method"] == m)
        if pts:
            a2.plot([p[0] for p in pts], [p[1] for p in pts], "--", marker="^", ms=3, color=C[m],
                    markerfacecolor="none", label=f"{lab} — paper (digitised)")
    a2.set_xlabel("Iteration"); a2.set_ylabel("Mean Reprojection Error [px]")
    a2.set_ylim(0, 150)
    a2.set_title("Paper Fig. 11 (digitised approximation;\nB-Spline divergence retained)", fontsize=8)
    a2.legend(fontsize=7); a2.grid(alpha=0.3)
    fig.suptitle("Figure 11 — ChArUco BA convergence (σ=0.1, seed 42)", fontsize=10)
    fig.tight_layout()
    savefig(fig, "figure_11", "comparison")

# ---------------------------------------------------------------- Figure 12 --
def fig12():
    ops = ["var_to_fac", "fac_to_var", "interpolation"]
    raws = {op: np.array([float(r["ns"]) for r in rd(f"{RES}/latency_raw/fig12_helix_prior/raw_{op}.csv")]) / 1e3
            for op in ops}
    pap = rd(f"{RES}/paper_digitised/fig12_digitised.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4))
    pos = {op: i for i, op in enumerate(ops)}
    bp = ax.boxplot([raws[op] for op in ops], positions=[pos[o] - 0.18 for o in ops], widths=0.28,
                    showfliers=False, patch_artist=True)
    for b in bp["boxes"]:
        b.set(facecolor=C["gsolver"], alpha=0.75)
    for m, ck in (("gsolver", "gsolver"), ("hyperion", "bspline")):
        for r in pap:
            if r["method"] != m:
                continue
            xi = pos[r["operation"]] + 0.18
            q1, q3 = float(r["q1_us"]), float(r["q3_us"])
            wl, wh = float(r["whisker_low_us"]), float(r["whisker_high_us"])
            med = float(r["median_us"]) if r["median_us"] else None
            face = "none"
            edge = C[ck]
            ax.add_patch(plt.Rectangle((xi - 0.12 + (0.0 if m == "gsolver" else 0.0), q1), 0.24, q3 - q1,
                                       fill=False, edgecolor=edge, linestyle="--"))
            ax.plot([xi, xi], [wl, q1], "--", color=edge, lw=1)
            ax.plot([xi, xi], [q3, wh], "--", color=edge, lw=1)
            if med:
                ax.plot([xi - 0.12, xi + 0.12], [med, med], "--", color=edge, lw=1)
    ax.set_xticks(list(pos.values()))
    ax.set_xticklabels(ops)
    ax.set_ylabel("runtime [µs]")
    ax.set_yscale("log")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color=C["gsolver"], lw=6, alpha=0.75, label="Ours — local raw samples (EPYC 7713, fallback protocol)"),
        Line2D([], [], color=C["gsolver"], lw=1, ls="--", label="Ours — paper box (digitised, i9-14900KF)"),
        Line2D([], [], color=C["bspline"], lw=1, ls="--", label="Hyperion — paper box (digitised, i9-14900KF)")],
        fontsize=7)
    ax.set_title("Figure 12 — message/interpolation latencies. Hardware differs; protocol for paper values unknown", fontsize=9)
    savefig(fig, "figure_12", "comparison")

# ------------------------------------------------------- Figures 13/19/20 --
def rs_overlays():
    frames = {13: [320], 19: [79, 320, 354, 429, 433], 20: [668, 670, 674, 715, 719]}
    for fg, fl in frames.items():
        for fr in fl:
            path = f"{RES}/local_gsolver/table_02/readout_0.001/projections_frame_{fr}.csv"
            if not os.path.exists(path):
                print("missing", path)
                continue
            rows = rd(path)
            fig, ax = plt.subplots(figsize=(9.8, 3.2))
            ax.add_patch(plt.Rectangle((0, 0), 1226, 370, facecolor="#d9d9d9", edgecolor="k"))
            uo = [float(r["u_obs"]) for r in rows]; vo = [float(r["v_obs"]) for r in rows]
            un = [float(r["u_nominal_optlm"]) for r in rows]; vn = [float(r["v_nominal_optlm"]) for r in rows]
            ug = [float(r["u_gsolver"]) for r in rows]; vg = [float(r["v_gsolver"]) for r in rows]
            ax.plot(un, vn, "o", ms=3.5, mfc="none", color=C["red"], label="nominal global-shutter projection")
            ax.plot(ug, vg, "+", ms=5, color=C["gsolver"], label="G-solver projection (local)")
            ax.plot(uo, vo, ".", ms=2, color="#333333", label="observed pixels (measurements)")
            ax.set_xlim(0, 1226); ax.set_ylim(370, 0)
            ax.set_title(f"Fig. {fg} — KITTI 06 frame {fr}, readout 1 ms. Camera image NOT in repo (grey canvas). "
                         "Hyperion series missing (baseline not runnable)", fontsize=8)
            ax.legend(fontsize=7, loc="lower right")
            savefig(fig, f"figure_{fg}_frame_{fr}", "overlays")

# ---------------------------------------------------------------- Figure 14 --
def fig14():
    cams = [1, 2, 4, 8]
    loc = {n: rd(f"{RES}/local_gsolver/figure_14/cameras_{n}/summary.csv")[0] for n in cams}
    pap = rd(f"{RES}/paper_digitised/fig14_digitised.csv")
    pv = {(r["metric"], int(r["num_cameras"])): float(r["value"]) for r in pap}
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    series = [
        ("avg runtime / camera [s]", [float(loc[n]["avg_robot_solve_ms"]) / 1e3 for n in cams],
         [pv[("avg_time_s", n)] for n in cams], C["teal"]),
        ("avg peak mem / camera [MB]\n(local: process RSS / #cams)", [int(loc[n]["rss_post_solve_kb"]) / 1024 / n for n in cams],
         [pv[("avg_mem_MB", n)] for n in cams], C["purple"]),
        ("ATE [m] (local: SE(3)-aligned)", [float(loc[n]["ate_aligned_m"]) for n in cams],
         [pv[("ATE_m", n)] for n in cams], C["red"]),
        ("ARE [rad] (local: SE(3)-aligned)", [float(loc[n]["are_aligned_rad"]) for n in cams],
         [pv[("ARE_rad", n)] for n in cams], C["zspline"]),
    ]
    xs = np.arange(4)
    for ax, (title, lv, pvv, col) in zip(axes, series):
        ax.bar(xs - 0.18, lv, width=0.34, color=col, label="local")
        ax.bar(xs + 0.18, pvv, width=0.34, facecolor="none", edgecolor="#555555", hatch="//",
               label="paper (digitised)")
        ax.set_xticks(xs); ax.set_xticklabels(cams); ax.set_xlabel("#cameras")
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=6)
    fig.suptitle("Figure 14 — multi-camera scalability, KITTI 06 (local: 16 threads, EPYC 7713 | paper: i9-14900KF)",
                 fontsize=9)
    fig.tight_layout()
    savefig(fig, "figure_14", "comparison")

# ------------------------------------------------------------------ tables --
def tables():
    # Table I
    paper_t1 = {"helix": [0.030, 1.533, 3.925, 4.510, 4.694],
                "sphere": [0.030, 1.542, 4.187, 4.589, 6.515]}
    out = [["trajectory", "sigma", "paper_NEES", "local_NEES", "abs_diff"]]
    for traj in ("helix", "sphere"):
        for s, pv in zip(SIGNAMES[:5], paper_t1[traj]):
            r = sweep_local("prior", traj, "0.001", s)  # NEES eta-independent (verified)
            lv = float(r["nees_local_est"])
            out.append([traj, s, pv, round(lv, 3), round(abs(lv - pv), 3)])
    with open(f"{RES}/local_gsolver/table_01_comparison.csv", "w", newline="") as f:
        csv.writer(f).writerows(out)
    print("wrote table_01_comparison.csv")

    paper_t2 = {
        ("ours", "ATE_m"): [0.671, 0.676, 0.676, 0.684, 0.726],
        ("ours", "ARE_rad"): [0.007, 0.007, 0.008, 0.007, 0.008],
        ("ours", "MRE_px"): [1.704, 1.708, 1.709, 1.715, 1.844],
        ("zspline", "ATE_m"): [0.692, 94.510, 1963.040, 200.168, 733.028],
        ("zspline", "ARE_rad"): [0.007, 0.194, 0.508, 0.350, 1.577],
        ("zspline", "MRE_px"): [1.717, 2.099, 1.884, 2.301, 8.142],
    }
    readouts = ["0", "0.0001", "0.001", "0.01", "0.1"]
    out = [["method", "metric", "readout_s", "paper_value", "local_value", "local_variant", "notes"]]
    for i, ro in enumerate(readouts):
        r = rd(f"{RES}/local_gsolver/table_02/readout_{ro}/summary.csv")[0]
        out.append(["ours", "ATE_m", ro, paper_t2[("ours", "ATE_m")][i], round(float(r["ate_aligned_m"]), 3),
                    "SE3-aligned (unaligned=" + str(round(float(r["ate_m"]), 3)) + ")",
                    "alignment inferred from readout-0 agreement"])
        out.append(["ours", "ARE_rad", ro, paper_t2[("ours", "ARE_rad")][i], round(float(r["are_aligned_rad"]), 4),
                    "SE3-aligned", ""])
        out.append(["ours", "MRE_px", ro, paper_t2[("ours", "MRE_px")][i], round(float(r["mre_stereo3_in_image"]), 3),
                    "stereo3 [u_l,v,disp] (2D=" + str(round(float(r["mre_px_in_image"]), 3)) + ")",
                    f"excluded {float(r['excluded_pct']):.2f}% of projections"])
        for m in ("zspline",):
            for k in ("ATE_m", "ARE_rad", "MRE_px"):
                out.append([m, k, ro, paper_t2[(m, k)][i], "", "", "MISSING_BASELINE (Hyperion not runnable)"])
    with open(f"{RES}/local_gsolver/table_02_comparison.csv", "w", newline="") as f:
        csv.writer(f).writerows(out)
    print("wrote table_02_comparison.csv")

ALL = {"fig01": fig01, "fig05": fig05, "fig06": fig06, "fig07": fig07, "fig08": fig08,
       "fig09": fig09, "fig10": fig10, "fig11": fig11, "fig12": fig12,
       "rs": rs_overlays, "fig14": fig14, "fig15_18": fig15_18, "tables": tables}

if __name__ == "__main__":
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].split(",")
    for name, fn in ALL.items():
        if only and name not in only:
            continue
        try:
            fn()
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            raise
