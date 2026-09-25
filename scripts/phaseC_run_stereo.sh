#!/bin/bash
# Stereo exports for Phases B-C, exactly as run (decisions D2(c), D3, D5 of the Phase A review).
#  kitti06-prefix = the first 161 keyframes (D3: all keyframes before the first keyframe whose ground-truth
#  position is within 20 m of a keyframe more than 50 keyframes earlier; certify/prefix.py).
#  No A3 gauge prior (D2(c)): the frozen single-observation messages fix the gauge.
#  Dynamic OpenMP schedule (D5; bit-identical to the static one), OMP_NUM_THREADS=16.
#  Normal-run rule as in Phase A: iterate until the max belief-mean step < 1e-3, cap 200.
# The two chains below run in parallel.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
B=${BUILD_DIR:-$REPO/build-certify}/certify/cpp/certify_stereo
E=${EXPORTS:-$REPO/exports}
S=$REPO/examples/data/datasets/stereo_visual
export OMP_NUM_THREADS=16
cd "$REPO"
COMMON="--g2o $S/multirobot/06.g2o --gt $S/gt/06.tum --dynamic-schedule"
P="$COMMON --keyframes 161"
mkdir -p $E/kitti06_prefix161 $E/kitti06_full

(
  $B $COMMON --converge 1e-3 --normal-max 200 --freeze --frozen-iters 2000 --dump-stride 10 \
     --out $E/kitti06_full/conv_noanchor
) > $E/kitti06_full/conv_noanchor.log 2>&1 &

(
  $B $P --converge 1e-3 --normal-max 200 --freeze --frozen-iters 3000 --dump-stride 10 --out $E/kitti06_prefix161/conv
  $B $P --normal-iters 0 --freeze --frozen-iters 3000 --dump-stride 10 --out $E/kitti06_prefix161/init
  $B $P --normal-iters 100 --unfrozen-export-iters 1,2,5,10,20,50,100 --out $E/kitti06_prefix161/unfrozen_run
) > $E/kitti06_prefix161/runs.log 2>&1 &
wait
