#!/bin/bash
# Figure 14: multi-camera scalability on KITTI 06 (multirobot/06.g2o), 1/2/4/8 cameras.
# Each configuration runs in its own process so peak RSS is per-configuration.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="$ROOT/reproduction/build/repro_stereo"
DATA="$ROOT/examples/data/datasets/stereo_visual"
PARAMS="$ROOT/examples/config/experiment_params.yaml"
OUT="$ROOT/reproduction/results/local_gsolver/figure_14"
THREADS=${OMP_NUM_THREADS:-16}

for n in 1 2 4 8; do
  dir="$OUT/cameras_$n"
  [ -f "$dir/summary.csv" ] && continue
  OMP_NUM_THREADS=$THREADS "$BIN" \
    --input "$DATA/multirobot/06.g2o" \
    --gt "$DATA/gt/06.tum" \
    --config kitti_05 --params "$PARAMS" \
    --robots "$n" --img-w 1226 --img-h 370 \
    --out "$dir" 2>&1 | tail -1
done
