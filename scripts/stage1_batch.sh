#!/bin/bash
# Stage 1: a sequence of runs, each capped at 6 minutes, each followed by its gate check (validity on stereo).
#   scripts/stage1_batch.sh <rounds> <tag or -> <dataset/variant> ...
REPO=$(cd "$(dirname "$0")/.." && pwd)
R=$1; T=$2; shift 2
for dv in "$@"; do
  ds=${dv%/*}; v=${dv#*/}
  if [ "$T" = "-" ]; then TAG= timeout 360 "$REPO/scripts/stage1_run.sh" $ds $v $R; else TAG=$T timeout 360 "$REPO/scripts/stage1_run.sh" $ds $v $R; fi
  D=$REPO/exports/${STAGE:-stage1}/$ds/$v${T:+_$T}; [ "$T" = "-" ] && D=$REPO/exports/${STAGE:-stage1}/$ds/$v
  python3 - "$D" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1] + "/eval/eval_summary.csv")))
vu = sum(int(r["violations_upper"] or 0) for r in rows); vl = sum(int(r["violations_lower"] or 0) for r in rows)
first = next((r["round"] for r in rows if r["all_pass"] == "1"), None)
print("GATE %s: %d evaluations, violations %d/%d -> %s; first all-pass round %s; last frac_pass %s" % (
    "/".join(sys.argv[1].split("/")[-3:]), len(rows), vu, vl, "continue" if vu + vl == 0 else "STOP", first, rows[-1]["frac_pass"]), flush=True)
PY
done
echo BATCHDONE
