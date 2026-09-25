"""Phase C runner: certify.phaseC.run_dataset + certify.phaseC_out.write_all for each dataset/snapshot.

Usage: python3 scripts/phaseC_run.py [--workers N] dataset [dataset ...]
Datasets: see certify/phaseC.py DATASETS. KITTI full runs only if its D2 pose floor is <= 0.05.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from certify import phaseC, phaseC_out  # noqa: E402


def main():
    args = sys.argv[1:]
    workers = 16
    force = "--allow-floor-above-tol" in args
    if force:
        args.remove("--allow-floor-above-tol")
    if "--workers" in args:
        k = args.index("--workers")
        workers = int(args[k + 1])
        del args[k:k + 2]
    for name in args:
        snapshot = name.split("_")[-1]
        d2 = json.loads((phaseC.OUT / "d2" / ("%s.json" % name)).read_text())
        if not d2["floor_pass"]:
            if not force:
                print("%s: D2 pose floor above 0.05; Phase C not run (decision D2)" % name, flush=True)
                continue
            print("%s: D2 pose floor above 0.05 -- running anyway (--allow-floor-above-tol; flagged in summary.json)" % name,
                  flush=True)
        t0 = time.perf_counter()
        try:
            R = phaseC.run_dataset(name, workers=workers)
            R["res"]["floor_tolerance_met"] = bool(d2["floor_pass"])
            S = phaseC_out.write_all(R, name, snapshot)
            bad = [r for r in S["invariants"] if not r["ok"]]
            print("%s: done in %.0fs, %d certs rows, %d invariant checks, %d failed" % (
                name, time.perf_counter() - t0, S["certs_rows"], len(S["invariants"]), len(bad)), flush=True)
            for r in bad:
                print("   FAILED", r, flush=True)
        except Exception:
            traceback.print_exc()
            sys.stdout.flush()


if __name__ == "__main__":
    main()
