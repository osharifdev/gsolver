#!/bin/bash
# Phase 5: authoritative latency runs (minimal instrumentation: --pure) in two
# thread modes: paper_parallelism (16) and local_native (64). 10 repetitions.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="$ROOT/reproduction/build/repro_synthetic"
SBIN="$ROOT/reproduction/build/repro_stereo"
DATA="$ROOT/examples/data/datasets/synthetic"
KDATA="$ROOT/examples/data/datasets/stereo_visual"
PARAMS="$ROOT/examples/config/experiment_params.yaml"
OUT="$ROOT/reproduction/results/latency_raw/end_to_end"
mkdir -p "$OUT"

for mode in 16 64; do
  for rep in $(seq 1 10); do
    for task in prior pgo; do
      for shape in helix sphere; do
        d="$OUT/${task}_${shape}_t${mode}_rep${rep}"
        [ -f "$d/summary.csv" ] && continue
        OMP_NUM_THREADS=$mode "$BIN" \
          --input "$DATA/$task/trajectories_1hz/noise_ig0.001_n0.001_${shape}.g2o" \
          --gt "$DATA/gt/${shape}_1hz.tum" --gt100 "$DATA/gt/${shape}_100hz.tum" \
          --config "$shape" --params "$PARAMS" --task "$task" \
          --interp --pure --out "$d" > /dev/null
      done
    done
  done
done

# KITTI rolling-shutter 1ms end-to-end (3 reps per mode; heavier)
for mode in 16 64; do
  for rep in 1 2 3; do
    d="$OUT/kitti_rs0.001_t${mode}_rep${rep}"
    [ -f "$d/summary.csv" ] && continue
    OMP_NUM_THREADS=$mode "$SBIN" \
      --input "$KDATA/rolling_shutter/kitti_06_0.001.g2o" \
      --gt "$KDATA/gt/06.tum" --config kitti_05 --params "$PARAMS" \
      --img-w 1226 --img-h 370 --out "$d" > /dev/null
  done
done
echo "latency runs complete"
