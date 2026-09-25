"""Markdown tables for reports/phaseB.md from results/phaseB (problems.csv, invariants.csv, rhoM_curve.csv)."""

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
B = REPO / "results" / "phaseB"


def load(fn):
    with open(fn) as f:
        return list(csv.DictReader(f))


def fnum(x, fmt="%.3g"):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "-"
    return "-" if not np.isfinite(v) else fmt % v


def main():
    P = load(B / "problems.csv")
    I = load(B / "invariants.csv")
    g = json.loads((B / "gateB.json").read_text())
    print("problems %d, invariant checks %d, failed %d\n" % (g["problems"], g["checks"], g["failed"]))
    # worst violation per check family (strip K / l suffixes)
    fam = {}
    for r in I:
        name = r["check"]
        key = name.split(" K=")[0].split(" l=")[0]
        w = float(r["worst"]) if r["worst"] not in ("", "-inf") else -np.inf
        f = fam.setdefault(key, dict(n=0, worst=-np.inf, label="", ok=True))
        f["n"] += int(r["n"])
        f["ok"] &= r["ok"] == "True"
        if w > f["worst"]:
            f["worst"], f["label"] = w, r["label"] + " " + name
    print("| check | evaluations | worst relative violation | where | pass |")
    print("|---|---|---|---|---|")
    for k in sorted(fam):
        f = fam[k]
        print("| %s | %d | %s | %s | %s |" % (k, f["n"], fnum(f["worst"], "%.2e"), f["label"], "yes" if f["ok"] else "**NO**"))
    print()
    # rho(M) curve
    rows = load(B / "rhoM_curve.csv")
    ns = sorted({int(r["n"]) for r in rows})
    mps = sorted({float(r["m_over_p"]) for r in rows})
    print("| n | cos(pi/(n+1)) | " + " | ".join("m/p=%g" % m for m in mps) + " |")
    print("|---|---|" + "---|" * len(mps))
    for n in ns:
        vals = {float(r["m_over_p"]): float(r["rho_M"]) for r in rows if int(r["n"]) == n}
        print("| %d | %.6f | " % (n, np.cos(np.pi / (n + 1))) + " | ".join("%.6f" % vals[m] for m in mps) + " |")
    print()
    # per family: spectral radii and ratios
    cols = ["label", "family", "n", "n_blocks", "m_over_p", "rho_M", "rho_Mhat", "rho_A_K1", "rho_A_K2", "rho_A_K3",
            "rho_A4_K1", "rho_A4_K2", "rho_A4_K3", "ratio_C2_median", "ratio_C3_K1_median", "ratio_C3_K2_median",
            "ratio_C3_K3_median", "ratio_C3_K3_p95", "n_hubs", "theta_S_oracle"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in P:
        print("| " + " | ".join(fnum(r.get(c)) if c not in ("label", "family") else r.get(c, "") for c in cols) + " |")


if __name__ == "__main__":
    main()
