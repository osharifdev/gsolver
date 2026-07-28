#!/bin/bash
# Figure 5: total energy vs iteration, PGO helix, sigma=0.1 (rot 0.01), eta_ig=1.
# G-solver energy is instrumented read-only; --gn additionally runs the
# independently implemented GN+GP (labelled supplementary, NOT the paper baseline).
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/reproduction/results/local_gsolver/figure_05"
OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} "$ROOT/reproduction/build/repro_synthetic" \
  --input "$ROOT/examples/data/datasets/synthetic/pgo/trajectories_1hz/noise_ig1.0_n0.1_helix.g2o" \
  --gt "$ROOT/examples/data/datasets/synthetic/gt/helix_1hz.tum" \
  --gt100 "$ROOT/examples/data/datasets/synthetic/gt/helix_100hz.tum" \
  --config helix --params "$ROOT/examples/config/experiment_params.yaml" \
  --task pgo --out "$OUT" --energy --interp --nees --gn --gn-iters 40
