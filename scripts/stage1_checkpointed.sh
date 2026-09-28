#!/bin/bash
# Stage 1: a run in checkpointed segments (compute rule: no segment over six minutes without a checkpoint and a gate).
#   scripts/stage1_checkpointed.sh <dataset> <variant> <rounds> <pause_every> <tol> [extra driver args]
# At every checkpoint the pre-registered gate is evaluated on the evaluator's log: continue unless a validity violation
# whose relative size exceeds <tol> (stereo: 1e-9, every violation stops; ChArUco: rounding-level exceedances up to
# <tol> are reported and the run continues, as in WP2). Each segment is also capped at 6 minutes.
REPO=$(cd "$(dirname "$0")/.." && pwd)
DS=$1; VAR=$2; R=$3; N=$4; TOL=$5; shift 5
OUT=$REPO/exports/stage1/$DS/$VAR${TAG:+_$TAG}
"$REPO/scripts/stage1_run.sh" $DS $VAR $R --eval-pause-every $N "$@" &
RUN=$!
seen=0; seg_start=$(date +%s)
while kill -0 $RUN 2>/dev/null; do
  sleep 2
  n=$(grep -c "waiting for" "$OUT/stdout.txt" 2>/dev/null); n=${n:-0}
  if [ "$n" -gt "$seen" ]; then
    seen=$n
    verdict=$(python3 - "$OUT" "$TOL" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1] + "/eval/eval_rounds.csv")))
tol = float(sys.argv[2])
bad = [r for r in rows if max(float(r["rel_upper_margin"] or "nan") if r["rel_upper_margin"] not in ("", "nan") else -1,
                              float(r["rel_lower_margin"] or "nan") if r["rel_lower_margin"] not in ("", "nan") else -1) > tol]
small = [r for r in rows if r.get("valid_upper") == "0" or r.get("valid_lower") == "0"]
last = max(int(r["round"]) for r in rows)
print("%s round %d: exceedances above tol %d, all flagged %d" % ("continue" if not bad else "STOP", last, len(bad), len(small)))
PY
)
    echo "CHECKPOINT $DS/$VAR $(date +%H:%M:%S) segment $(( $(date +%s) - seg_start ))s: $verdict"
    if [[ $verdict == continue* ]]; then touch "$OUT/continue"; else touch "$OUT/abort"; fi
    seg_start=$(date +%s)
  fi
  if [ $(( $(date +%s) - seg_start )) -gt 360 ]; then
    echo "CHECKPOINT $DS/$VAR: segment over 6 minutes, aborting"; touch "$OUT/abort"; kill $RUN
  fi
done
wait $RUN
echo "RUNDONE $DS/$VAR"
