#!/bin/bash
# Figure 6: helix prior-based, sigma=1 (rot 0.1), eta_ig=1:
# initial guess / GBP-only (no GP factors) / GBP+GP.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
G2O="$ROOT/examples/data/datasets/synthetic/prior/trajectories_1hz/noise_ig1.0_n1.0_helix.g2o"
COMMON=(--input "$G2O" --gt "$ROOT/examples/data/datasets/synthetic/gt/helix_1hz.tum" \
  --gt100 "$ROOT/examples/data/datasets/synthetic/gt/helix_100hz.tum" \
  --config helix --params "$ROOT/examples/config/experiment_params.yaml")
BASE="$ROOT/reproduction/results/local_gsolver/figure_06"
OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} "$ROOT/reproduction/build/repro_synthetic" \
  "${COMMON[@]}" --task prior --out "$BASE/gbp_gp" --interp --nees --energy
OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} "$ROOT/reproduction/build/repro_synthetic" \
  "${COMMON[@]}" --task prior_nogp --out "$BASE/gbp_only" --energy
