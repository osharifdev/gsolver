#!/bin/bash
# Phase A exports for KITTI 06 stereo BA (amendments A1, A3, A5), exactly as run.
#  kitti06-prefix = first 20 keyframes (A1 definition; certify/phaseA.py and reports/phaseA.md §3.1).
#  Normal-run rule: iterate until the max belief-mean step (belief precision metric) < 1e-3, cap 200.
#  Frozen models get the A3 gauge prior (SE(3) on the first keyframe, precision 1e4) via --anchor-precision.
# Notes: the prefix runs used the default (static) OpenMP schedule; KITTI full used --dynamic-schedule
# (bit-identical, ~10x faster). The KITTI full frozen run was stopped by hand at iteration 3,740 of the
# planned 5,000 (host overloaded by other jobs; floor flat since t ~ 360); see its frozen_run.json.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
B=${BUILD_DIR:-$REPO/build-certify}/certify/cpp/certify_stereo
E=${EXPORTS:-$REPO/exports}
S=$REPO/examples/data/datasets/stereo_visual
export OMP_NUM_THREADS=16
cd "$REPO"
COMMON="--g2o $S/multirobot/06.g2o --gt $S/gt/06.tum"
P="$COMMON --keyframes 20"
mkdir -p $E/kitti06_prefix $E/kitti06_full

# kitti06-prefix: frozen at the normal-run cap, frozen at iteration 0, unfrozen exports
$B $P --converge 1e-3 --normal-max 200 --freeze --anchor-precision 1e4 --frozen-iters 2000 --dump-stride 10 --out $E/kitti06_prefix/conv
$B $P --normal-iters 0 --freeze --anchor-precision 1e4 --frozen-iters 3000 --dump-stride 10 --out $E/kitti06_prefix/init
$B $P --normal-iters 100 --unfrozen-export-iters 1,2,5,10,20,50,100 --out $E/kitti06_prefix/unfrozen_run
# regularizer diagnostics (frozen mode only)
for e in 1e-7 1e-8; do
  $B $P --converge 1e-3 --normal-max 200 --freeze --anchor-precision 1e4 --frozen-iters 2000 --dump-stride 10 \
     --frozen-eps $e --out $E/kitti06_prefix/conv_eps$e
done

# KITTI 06 full: frozen at the normal-run cap, plus the regularizer diagnostic
$B $COMMON --converge 1e-3 --normal-max 200 --freeze --anchor-precision 1e4 --frozen-iters 5000 --dump-stride 10 \
   --dynamic-schedule --out $E/kitti06_full/conv
$B $COMMON --converge 1e-3 --normal-max 200 --freeze --anchor-precision 1e4 --frozen-iters 1000 --dump-stride 10 \
   --dynamic-schedule --frozen-eps 1e-7 --out $E/kitti06_full/conv_eps1e-7

# F4 chain check (pose chain of the synthetic prior experiment: a tree)
$REPO/build-certify/certify/cpp/certify_chain --g2o $REPO/examples/data/datasets/synthetic/prior/trajectories_1hz/noise_ig1.0_n0.001_helix.g2o \
   --config helix --out $E/chain_helix --normal-iters 100 --frozen-iters 3000
