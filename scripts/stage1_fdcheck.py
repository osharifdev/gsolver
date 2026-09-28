"""Stage 1: summary of the central-difference Jacobian check of a run (OUT/fdcheck/fd_check.tsv, --fd-check-at).

Per factor class, variable slot and block of three tangent columns: the largest relative difference
max|J_solver - J_fd| / max|J_fd| over the factors (blocks whose central difference is 0 use the absolute difference),
and the least-squares ratio range. The corrected variant must be within 1e-5 on every block.

Usage: python3 scripts/stage1_fdcheck.py <run dir> [<run dir> ...] [--json out.json]
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np


def summarise(run):
    rows = list(csv.DictReader(open(Path(run) / "fdcheck" / "fd_check.tsv"), delimiter="\t"))
    out = {}
    for r in rows:
        k = "%s slot %s cols %s" % (r["class"], r["slot"], r["cols"])
        a = out.setdefault(k, dict(factors=0, rel_max=0.0, ratio=[]))
        a["factors"] += 1
        m, d = float(r["max_abs_fd"]), float(r["max_abs_diff"])
        a["rel_max"] = max(a["rel_max"], d / m if m > 1e-12 else d)
        if r["ratio"] != "nan":
            a["ratio"].append(float(r["ratio"]))
    for a in out.values():
        a["ratio_min"] = float(min(a["ratio"])) if a["ratio"] else None
        a["ratio_max"] = float(max(a["ratio"])) if a["ratio"] else None
        del a["ratio"]
    return dict(factors_checked=len(set((r["class"], r["factor"]) for r in rows)), blocks=out,
                worst_block=max(out, key=lambda k: out[k]["rel_max"]), worst_rel=max(a["rel_max"] for a in out.values()))


if __name__ == "__main__":
    a = sys.argv[1:]
    js = a[a.index("--json") + 1] if "--json" in a else None
    runs = [x for x in a if x != "--json" and x != js]
    res = {r: summarise(r) for r in runs}
    for r, s in res.items():
        print(r, "factors", s["factors_checked"], "worst", s["worst_block"], "%.3g" % s["worst_rel"])
        for k, v in s["blocks"].items():
            print("   %-32s n=%-6d rel_max %.3g  ratio [%s, %s]" % (k, v["factors"], v["rel_max"], v["ratio_min"], v["ratio_max"]))
    if js:
        Path(js).parent.mkdir(parents=True, exist_ok=True)
        Path(js).write_text(json.dumps(res, indent=1))
