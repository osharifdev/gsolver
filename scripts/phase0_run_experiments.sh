#!/bin/bash
# Phase 0 (environment check): run every experiment named in the brief once,
# with the repository's default settings (docs/USAGE.md commands, no -visualize),
# recording printed output, wall time and peak RSS via /usr/bin/time -v.
#
# Usage: BUILD_DIR=<cmake build dir with examples> scripts/phase0_run_experiments.sh
# Extra runs at OMP_NUM_THREADS=16 check that results are thread-count independent
# and give timings closer to the paper's 16-core machine.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
BIN=${BUILD_DIR:?set BUILD_DIR}/examples/cpp
OUT=$REPO/results/phase0
DATA=$REPO/examples/data/datasets
mkdir -p "$OUT"
cd "$REPO"

run() { # name, then the command; env overrides may precede the binary
  local name=$1; shift
  echo "=== $name: $*"
  /usr/bin/env "$@" > "$OUT/$name.stdout" 2> "$OUT/$name.stderr_time"
  echo "exit=$?"
  grep -E "Elapsed \(wall|Maximum resident" "$OUT/$name.stderr_time"
}
T="/usr/bin/time -v"

# --- ChArUco (config fixed to printing_room by the drivers) ---
run charuco_islam              $T "$BIN/charuco_experiment_islam" $DATA/charuco/printing_room_0.g2o "$OUT/charuco_islam.tum"
run charuco_ba_noise0          $T "$BIN/charuco_experiment_ba"    $DATA/charuco/printing_room_0.g2o "$OUT/charuco_ba_noise0.tum" 0.0
run charuco_ba_noise0.1        $T "$BIN/charuco_experiment_ba"    $DATA/charuco/printing_room_0.g2o "$OUT/charuco_ba_noise0.1.tum" 0.1
run charuco_islam_omp16        OMP_NUM_THREADS=16 $T "$BIN/charuco_experiment_islam" $DATA/charuco/printing_room_0.g2o "$OUT/charuco_islam_omp16.tum"

# --- KITTI 06 stereo BA, single camera (= rolling-shutter readout 0; files are identical) ---
SV=$DATA/stereo_visual
run stereo_ba_kitti06          $T "$BIN/stereo_visual" $SV/multirobot/06.g2o $SV/gt/06.tum "$OUT/stereo_ba_kitti06.tum" kitti_05
run stereo_ba_kitti06_omp16    OMP_NUM_THREADS=16 $T "$BIN/stereo_visual" $SV/multirobot/06.g2o $SV/gt/06.tum "$OUT/stereo_ba_kitti06_omp16.tum" kitti_05

# --- Multi-camera (README default: 8 cameras) ---
run multicam8_kitti06          $T "$BIN/stereo_visual" $SV/multirobot/06.g2o $SV/gt/06.tum "$OUT/multicam8_kitti06.tum" kitti_05 8

# --- Rolling shutter (README default readout 0.1 s, plus the other Table II readouts) ---
for r in 0.1 0.01 0.001 0.0001; do
  run rs_kitti06_$r            $T "$BIN/stereo_visual" $SV/rolling_shutter/kitti_06_$r.g2o $SV/gt/06.tum "$OUT/rs_kitti06_$r.tum" kitti_05
done

# --- accuracy against KITTI ground truth ---
for f in stereo_ba_kitti06 stereo_ba_kitti06_omp16 multicam8_kitti06 rs_kitti06_0.1 rs_kitti06_0.01 rs_kitti06_0.001 rs_kitti06_0.0001; do
  printf "%-26s " "$f"
  python3 -m certify.tools.traj_eval "$OUT/$f.tum" $SV/gt/06.tum
done | tee "$OUT/ate_are.txt"

echo "=== thread-count independence (byte comparison of outputs)"
cmp "$OUT/charuco_islam.tum" "$OUT/charuco_islam_omp16.tum" && echo "charuco_islam: 64 vs 16 threads identical"
cmp "$OUT/stereo_ba_kitti06.tum" "$OUT/stereo_ba_kitti06_omp16.tum" && echo "stereo_ba: 64 vs 16 threads identical"
