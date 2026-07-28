#!/bin/bash
# Table II (all readout times) + Figs. 13/19/20 projection dumps (1 ms readout).
# Config kitti_05 (as the paper: GP hyperparameters initialized from KITTI 05).
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="$ROOT/reproduction/build/repro_stereo"
DATA="$ROOT/examples/data/datasets/stereo_visual"
PARAMS="$ROOT/examples/config/experiment_params.yaml"
OUT="$ROOT/reproduction/results/local_gsolver/table_02"
THREADS=${OMP_NUM_THREADS:-16}

for r in 0 0.0001 0.001 0.01 0.1; do
  dir="$OUT/readout_$r"
  [ -f "$dir/summary.csv" ] && continue
  EXTRA=()
  if [ "$r" == "0.001" ]; then
    EXTRA=(--frames 79,320,354,429,433,668,670,674,715,719)
  fi
  OMP_NUM_THREADS=$THREADS "$BIN" \
    --input "$DATA/rolling_shutter/kitti_06_$r.g2o" \
    --gt "$DATA/gt/06.tum" \
    --config kitti_05 --params "$PARAMS" \
    --img-w 1226 --img-h 370 \
    --out "$dir" "${EXTRA[@]}" 2>&1 | tail -1
done
