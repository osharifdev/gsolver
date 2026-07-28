#!/bin/bash
# Figure 12: message/interpolation latency microbenchmarks (fallback protocol:
# no original harness in repo). Graph choice (documented): helix prior,
# sigma=1e-3 / eta=1e-3. 5000 samples per op, raw samples saved.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} "$ROOT/reproduction/build/repro_bench" \
  --input "$ROOT/examples/data/datasets/synthetic/prior/trajectories_1hz/noise_ig0.001_n0.001_helix.g2o" \
  --config helix --params "$ROOT/examples/config/experiment_params.yaml" \
  --samples 5000 --warmup-iters 10 \
  --out "$ROOT/reproduction/results/latency_raw/fig12_helix_prior"
