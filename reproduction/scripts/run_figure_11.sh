#!/bin/bash
# Figure 11: ChArUco BA per-iteration metrics, initial noise 0.1 (seed 42).
# NOTE: exact pixel MRE not computable from public artefacts (no calibration /
# pixel detections); substitute pose-landmark metrics are recorded.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} "$ROOT/reproduction/build/repro_charuco" \
  --input "$ROOT/examples/data/datasets/charuco/printing_room_0.g2o" \
  --params "$ROOT/examples/config/experiment_params.yaml" \
  --noise 0.1 \
  --out "$ROOT/reproduction/results/local_gsolver/figure_11"
