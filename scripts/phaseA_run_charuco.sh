#!/bin/bash
# Phase A exports for ChArUco (amendment A5): frozen snapshots of incremental SLAM at 25/50/100% of
# the sequence and of BA at iteration 0 and at the end of its normal run; unfrozen exports of the
# incremental run at iterations 1,2,5,10,20,50,100. Each frozen snapshot is also rerun with the
# frozen-mode regularizers at 1e-7 (diagnostic: the floor must scale with them if they explain it).
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
B=${BUILD_DIR:-$REPO/build-certify}/certify/cpp/certify_charuco
E=${EXPORTS:-$REPO/exports}
G=$REPO/examples/data/datasets/charuco/printing_room_0.g2o
export OMP_NUM_THREADS=16
cd "$REPO"
FROZEN="--freeze --frozen-iters 30000 --dump-stride 10"
run() { local out=$1; shift; rm -rf "$out"; "$B" --g2o "$G" --out "$out" "$@" > "$out.log" 2>&1; echo "$? $out"; }
mkdir -p $E/charuco_islam $E/charuco_ba
run $E/charuco_islam/frac025 --mode islam --fraction 0.25 $FROZEN &
run $E/charuco_islam/frac050 --mode islam --fraction 0.50 $FROZEN &
run $E/charuco_islam/frac100 --mode islam --fraction 1.00 $FROZEN &
wait
run $E/charuco_ba/init --mode ba --freeze-at 0 $FROZEN &
[ -d $E/charuco_ba/end ] || run $E/charuco_ba/end --mode ba --freeze-at 50 $FROZEN &
run $E/charuco_islam/unfrozen_run --mode islam --unfrozen-export-iters 1,2,5,10,20,50,100 &
wait
for s in charuco_islam/frac025:"--mode islam --fraction 0.25" charuco_islam/frac050:"--mode islam --fraction 0.50" \
         charuco_islam/frac100:"--mode islam --fraction 1.00" charuco_ba/init:"--mode ba --freeze-at 0"; do
  d=${s%%:*}; a=${s#*:}
  run $E/${d}_eps1e-7 $a $FROZEN --frozen-eps 1e-7 &
done
wait
