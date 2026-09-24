"""Phase A: validate every export with the final validator, run the fact checks, pack (A5), and write
the summary tables in results/phaseA/.

Usage: python scripts/phaseA_validate_all.py [--only SUBSTRING]
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from certify import facts  # noqa: E402
from certify.pack import pack  # noqa: E402
from certify.phaseA import a3_analysis, far_landmark_attribution, validate  # noqa: E402

E = REPO / "exports"
OUT = REPO / "results" / "phaseA"
OUT.mkdir(parents=True, exist_ok=True)
only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else ""

FROZEN = [  # (dataset, snapshot, directory, stereo)
    ("charuco_islam", "25% of the sequence", "charuco_islam/frac025", False),
    ("charuco_islam", "50% of the sequence", "charuco_islam/frac050", False),
    ("charuco_islam", "100% (end of normal run)", "charuco_islam/frac100", False),
    ("charuco_ba", "iteration 0 (initial guess)", "charuco_ba/init", False),
    ("charuco_ba", "iteration 50 (end of normal run)", "charuco_ba/end", False),
    ("kitti06_prefix", "iteration 0 (initial guess)", "kitti06_prefix/init", True),
    ("kitti06_prefix", "normal-run cap (200, not converged)", "kitti06_prefix/conv", True),
    ("kitti06_full", "normal-run cap (200, not converged)", "kitti06_full/conv", True),
]
DIAG = [  # regularizer diagnostics: (base directory, variant directory, eps or site setting)
    ("charuco_ba/end", "charuco_ba/end_eps1e-5", "all sites 1e-5"),
    ("charuco_ba/end", "charuco_ba/end_eps1e-7", "all sites 1e-7"),
    ("charuco_ba/end", "charuco_ba/end_eps1e-8", "all sites 1e-8"),
    ("charuco_ba/end", "charuco_ba/end_eps1e-9", "all sites 1e-9"),
    ("charuco_ba/end", "charuco_ba/end_eps1e-12", "all sites 1e-12"),
    ("charuco_ba/end", "charuco_ba/end_site_belief", "belief 1e-7, others 1e-6"),
    ("charuco_ba/end", "charuco_ba/end_site_v2f", "variable-to-factor 1e-7, others 1e-6"),
    ("charuco_ba/end", "charuco_ba/end_site_marg", "marginalization 1e-7, others 1e-6"),
    ("charuco_ba/end", "charuco_ba/end_site_f2v", "factor-to-variable 1e-7, others 1e-6"),
    ("charuco_islam/frac025", "charuco_islam/frac025_eps1e-7", "all sites 1e-7"),
    ("charuco_islam/frac050", "charuco_islam/frac050_eps1e-7", "all sites 1e-7"),
    ("charuco_islam/frac100", "charuco_islam/frac100_eps1e-7", "all sites 1e-7"),
    ("charuco_ba/init", "charuco_ba/init_eps1e-7", "all sites 1e-7"),
    ("kitti06_prefix/conv", "kitti06_prefix/conv_eps1e-7", "all sites 1e-7"),
    ("kitti06_prefix/conv", "kitti06_prefix/conv_eps1e-8", "all sites 1e-8"),
]
UNFROZEN = [("charuco_islam", E / "charuco_islam/unfrozen_run/unfrozen"),
            ("kitti06_prefix", E / "kitti06_prefix/unfrozen_run/unfrozen")]


def run_validate(d, stereo):
    r, x = validate(d, align=stereo, out_csv=d / "gateA_curves.csv")
    np.save(d / "xbar.npy", x)
    (d / "gateA.json").write_text(json.dumps(r, indent=1, default=float))
    return r


def curve(r, k):
    c = r["curves"].get(k)
    return (c["floor"], c["floor_reached"], c["t_floor"], c["rate"]) if c else (np.nan, None, np.nan, np.nan)


rows = []
for ds, snap, rel, stereo in FROZEN:
    d = E / rel
    if only not in rel or not (d / "frozen_run.json").exists():
        continue
    print("validating", rel, flush=True)
    r = run_validate(d, stereo)
    extra = {}
    if stereo:
        a3 = a3_analysis(d)
        (d / "a3.json").write_text(json.dumps(a3, indent=1, default=float))
        extra["a3"] = a3
        fl = far_landmark_attribution(d)
        (d / "far_landmarks.json").write_text(json.dumps(fl, indent=1))
        extra["far"] = fl
        f5 = facts.f5_check(d)
        (d / "f5.json").write_text(json.dumps(f5, indent=1))
        if (d / "interp_at_freeze.npy").exists():
            (d / "f3.json").write_text(json.dumps(facts.f3_check(d), indent=1))
    pack(d)
    row = dict(dataset=ds, snapshot=snap, export=rel, n_dof=r["n_dof"], n_blocks=r["n_blocks"],
               n_single_edge=r["n_single_edge"], n_uninformed_dofs=r["n_uninformed_dofs"],
               n_blocks_min_D_eig_below_1e_4=r["n_blocks_min_D_eig_below_1e-4"],
               symmetry_rel=r["symmetry_rel"], rel_residual=r["rel_residual"], lambda_min=r["lambda_min"],
               lambda_max=r["lambda_max"], cond_H=r["cond_H"], cond_scaled=r["cond_scaled"],
               near_null_dim=r["near_null_dim"])
    for k in ("non_single_edge", "poses", "resolvable", "aligned_non_single_edge", "single_edge_vs_xbar"):
        f, reached, tf, rate = curve(r, k)
        row[k + "_floor"], row[k + "_reached"], row[k + "_t_floor"], row[k + "_rate"] = f, reached, tf, rate
    if "far" in extra:
        for x in extra["far"]:
            row["pose_floor_far_fixed_tau%g" % x["tau"]] = x["pose_floor"]
            row["n_far_fixed_tau%g" % x["tau"]] = x["n_fixed"]
    rows.append(row)
if rows:
    with open(OUT / ("gateA_summary.csv" if not only else "gateA_summary_%s.csv" % only.replace("/", "_")), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)

drows = []
for base, rel, setting in DIAG:
    d = E / rel
    if only not in rel or not (d / "frozen_run.json").exists():
        continue
    print("validating diagnostic", rel, flush=True)
    r = run_validate(d, "kitti" in rel)
    b = json.loads((E / base / "gateA.json").read_text()) if (E / base / "gateA.json").exists() else None
    for k in ("non_single_edge", "poses", "resolvable"):
        f, reached, tf, rate = curve(r, k)
        fb = b["curves"][k]["floor"] if b and k in b["curves"] else np.nan
        drows.append(dict(base=base, variant=rel, setting=setting, category=k, floor=f, floor_reached=reached,
                          t_floor=tf, rate=rate, base_floor_eps1e_6=fb, ratio_to_base=f / fb if fb else np.nan))
if drows:
    with open(OUT / ("eps_diagnostics.csv" if not only else "eps_diagnostics_%s.csv" % only.replace("/", "_")), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(drows[0]))
        w.writeheader()
        w.writerows(drows)

urows = []
for ds, root in UNFROZEN:
    if only not in str(root) or not root.exists():
        continue
    for d in sorted(root.glob("it*")):
        print("validating unfrozen", d.relative_to(E), flush=True)
        r, x = validate(d)
        np.save(d / "xbar.npy", x)
        (d / "gateA.json").write_text(json.dumps(r, indent=1, default=float))
        pack(d)
        urows.append(dict(dataset=ds, normal_iteration=int(d.name[2:]), n_dof=r["n_dof"], n_single_edge=r["n_single_edge"],
                          n_uninformed_dofs=r["n_uninformed_dofs"], rel_residual=r["rel_residual"],
                          lambda_min=r["lambda_min"], lambda_max=r["lambda_max"], cond_H=r["cond_H"],
                          cond_scaled=r["cond_scaled"], near_null_dim=r["near_null_dim"],
                          gn_step_D_max=r["gn_step_D_max"], gn_step_D_median=r["gn_step_D_median"]))
if urows:
    with open(OUT / ("unfrozen_summary.csv" if not only else "unfrozen_summary_%s.csv" % only.replace("/", "_")), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(urows[0]))
        w.writeheader()
        w.writerows(urows)

if only in "chain_helix" and (E / "chain_helix").exists():
    (OUT / "f4_chain.json").write_text(json.dumps(facts.f4_check(E / "chain_helix"), indent=1))
print("done")
