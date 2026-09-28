#!/bin/bash
# Stage 1 (certify/STAGE1.md): one run of periodic certified termination.
#   scripts/stage1_run.sh <dataset> <variant> <rounds> [extra driver args...]
# dataset: kitti06 | kitti03 | kitti04 | kitti10 (first 40 keyframes) | prefix (KITTI 06, 161 keyframes) | charuco (islam)
# variant: shipped | corrected (--true-jacobians)
# Driver outputs (flops, heuristics, work log, change-based stop exports, fd check) and the evaluator's logs go to
# exports/stage1/<dataset>/<variant>[_<tag>]; set TAG for one-tenth versions.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
B=${BUILD_DIR:-$REPO/build-certify}
DS=$1; VAR=$2; ROUNDS=$3; shift 3
TAG=${TAG:-}
OUT=$REPO/exports/stage1/$DS/$VAR${TAG:+_$TAG}
EDIR=/dev/shm/gbp_os717_s1_${DS}_${VAR}${TAG:+_$TAG}
S=$REPO/examples/data/datasets/stereo_visual
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
rm -rf "${OUT:?}" "${EDIR:?}"
mkdir -p "$OUT" "$EDIR"
JAC=""; [ "$VAR" = corrected ] && JAC="--true-jacobians"
MID=$(( (ROUNDS / 2) / 10 * 10 )); [ $MID -lt 10 ] && MID=10
COMMON="--relin-threshold 0.003 --relin-every 1 --relin-metric physical --count-flops --eval-dir $EDIR --eval-every 10 --change-taus 1e-2,1e-3,1e-4 --fd-check-at 10 $JAC"
case $DS in
  kitti06|kitti03|kitti04|kitti10|prefix)
    SEQ=${DS#kitti}; KF=40; CFG=kitti_$SEQ
    [ "$DS" = prefix ] && { SEQ=06; KF=161; }
    [ "$SEQ" = 06 ] && CFG=kitti_05
    GT=$S/gt/$SEQ.tum
    EVAL_ARGS="--p-L 1e-6 --gt $GT --sampled 10,$MID,$ROUNDS"
    DRIVER="$B/certify/cpp/certify_stereo --g2o $S/multirobot/$SEQ.g2o --gt $GT --keyframes $KF --config $CFG --normal-iters $ROUNDS --dynamic-schedule --out $OUT --tum $OUT/traj.tum $COMMON"
    ;;
  charuco)
    EXTRA=$(( ROUNDS - 103 )); [ $EXTRA -lt 0 ] && EXTRA=0
    EVAL_ARGS="--p-L none --sampled 10,$MID,$ROUNDS --stop-from 94"
    DRIVER="$B/certify/cpp/certify_charuco --mode islam --g2o $REPO/examples/data/datasets/charuco/printing_room_0.g2o --out $OUT --tum $OUT/traj.tum --extra-final-rounds $EXTRA --change-from 95 --export-beliefs-at 103 $COMMON"
    ;;
  *) echo "unknown dataset $DS"; exit 2;;
esac
python3 "$REPO/scripts/stage1_evaluator.py" "$EDIR" "$OUT/eval" $EVAL_ARGS > "$OUT/evaluator.log" 2>&1 &
EV=$!
t0=$(date +%s.%N)
$DRIVER "$@" > "$OUT/stdout.txt" 2>&1
rc=$?
[ $rc -ne 0 ] && touch "$EDIR/abort"
wait $EV; erc=$?
printf '{"dataset": "%s", "variant": "%s", "rounds": %d, "wall_s": %.3f, "driver_exit": %d, "evaluator_exit": %d}\n' \
  "$DS" "$VAR" "$ROUNDS" "$(echo "$(date +%s.%N) - $t0" | bc)" $rc $erc > "$OUT/run.json"
cat "$OUT/run.json"
