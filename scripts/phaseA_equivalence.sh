#!/bin/bash
# Phase A: prove the frozen-mode changes leave the default solver path bit-identical, and that the
# certify drivers replicate the shipped drivers. All comparisons are byte comparisons of trajectories.
#  1. shipped drivers rebuilt against the modified library  vs  Phase 0 outputs (unmodified library)
#  2. certify drivers without --freeze                      vs  shipped drivers
#  3. certify islam with unfrozen exports enabled           vs  without (exports must not perturb the run)
#  4. GbpSolver::dynamic_schedule_ (performance option)     vs  default static schedule
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
B=${BUILD_DIR:-$REPO/build-certify}
OUT=${OUT:-$REPO/results/phaseA/equivalence}
# driver output directories (state dumps) go to the gitignored exports/ tree
RUNS=${RUNS:-$REPO/exports/equivalence_runs}
P0=$REPO/results/phase0
D=$REPO/examples/data/datasets
export OMP_NUM_THREADS=16
mkdir -p "$OUT" "$RUNS"
cd "$REPO"
ok=0; bad=0
same() { # label, file a, file b
  if cmp -s <(grep -v '^#' "$2") <(grep -v '^#' "$3"); then echo "IDENTICAL  $1"; ok=$((ok+1)); else echo "DIFFERENT  $1"; bad=$((bad+1)); fi
}
C=$D/charuco/printing_room_0.g2o
$B/examples/cpp/charuco_experiment_islam $C $OUT/shipped_islam.tum > /dev/null
$B/examples/cpp/charuco_experiment_ba $C $OUT/shipped_ba0.tum 0.0 > /dev/null
$B/examples/cpp/charuco_experiment_ba $C $OUT/shipped_ba01.tum 0.1 > /dev/null
same "1. shipped islam (modified lib) vs Phase 0" $OUT/shipped_islam.tum $P0/charuco_islam.tum
same "1. shipped BA noise 0 (modified lib) vs Phase 0" $OUT/shipped_ba0.tum $P0/charuco_ba_noise0.tum
same "1. shipped BA noise 0.1 (modified lib) vs Phase 0" $OUT/shipped_ba01.tum $P0/charuco_ba_noise0.1.tum
$B/certify/cpp/certify_charuco --mode islam --g2o $C --out $RUNS/cert_islam --tum $OUT/cert_islam.tum > /dev/null
$B/certify/cpp/certify_charuco --mode ba --g2o $C --out $RUNS/cert_ba0 --tum $OUT/cert_ba0.tum > /dev/null
$B/certify/cpp/certify_charuco --mode ba --noise 0.1 --g2o $C --out $RUNS/cert_ba01 --tum $OUT/cert_ba01.tum > /dev/null
same "2. certify islam vs shipped islam" $OUT/cert_islam.tum $OUT/shipped_islam.tum
same "2. certify BA noise 0 vs shipped" $OUT/cert_ba0.tum $OUT/shipped_ba0.tum
same "2. certify BA noise 0.1 vs shipped" $OUT/cert_ba01.tum $OUT/shipped_ba01.tum
$B/certify/cpp/certify_charuco --mode islam --g2o $C --out $RUNS/cert_islam_exp --tum $OUT/cert_islam_exp.tum \
  --unfrozen-export-iters 1,2,5,10,20,50,100 > /dev/null
same "3. certify islam with unfrozen exports vs without" $OUT/cert_islam_exp.tum $OUT/cert_islam.tum
S=$D/stereo_visual
$B/examples/cpp/stereo_visual $S/multirobot/06.g2o $S/gt/06.tum $OUT/shipped_stereo.tum kitti_05 > /dev/null
same "1. shipped stereo KITTI 06 (modified lib) vs Phase 0" $OUT/shipped_stereo.tum $P0/stereo_ba_kitti06.tum
$B/certify/cpp/certify_stereo --g2o $S/multirobot/06.g2o --gt $S/gt/06.tum --out $RUNS/cert_stereo --tum $OUT/cert_stereo.tum > /dev/null
same "2. certify stereo vs shipped stereo" $OUT/cert_stereo.tum $OUT/shipped_stereo.tum
$B/certify/cpp/certify_stereo --g2o $S/multirobot/06.g2o --gt $S/gt/06.tum --out $RUNS/cert_stereo_dyn --tum $OUT/cert_stereo_dyn.tum --dynamic-schedule > /dev/null
same "4. certify stereo, dynamic schedule vs Phase 0" $OUT/cert_stereo_dyn.tum $P0/stereo_ba_kitti06.tum
$B/certify/cpp/certify_charuco --mode islam --g2o $C --out $RUNS/cert_islam_dyn --tum $OUT/cert_islam_dyn.tum --dynamic-schedule > /dev/null
same "4. certify islam, dynamic schedule vs Phase 0" $OUT/cert_islam_dyn.tum $P0/charuco_islam.tum
for m in static dyn; do f=""; [ $m = dyn ] && f="--dynamic-schedule"
  $B/certify/cpp/certify_charuco --mode ba --g2o $C --out $RUNS/frozen_ba_$m --freeze --freeze-at 50 --frozen-iters 2000 $f > /dev/null; done
if cmp -s $RUNS/frozen_ba_static/states_all.f64 $RUNS/frozen_ba_dyn/states_all.f64; then echo "IDENTICAL  4. frozen BA state dumps (2000 iterations), static vs dynamic schedule"; ok=$((ok+1)); else echo "DIFFERENT  4. frozen BA state dumps"; bad=$((bad+1)); fi
echo "identical: $ok  different: $bad"
