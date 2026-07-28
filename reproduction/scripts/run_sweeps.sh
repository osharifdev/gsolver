#!/bin/bash
# Sweep runner for Figs. 8, 9, 15-18 and Table I.
# Runs repro_synthetic over task x shape x eta_ig x sigma with metrics enabled.
# Numerical results are thread-count independent (synchronous schedule);
# authoritative timing runs are done separately (run_latency.sh).
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="$ROOT/reproduction/build/repro_synthetic"
DATA="$ROOT/examples/data/datasets/synthetic"
PARAMS="$ROOT/examples/config/experiment_params.yaml"
OUT="$ROOT/reproduction/results/local_gsolver/sweeps"
THREADS_PER_RUN=${THREADS_PER_RUN:-8}
CONCURRENCY=${CONCURRENCY:-8}
mkdir -p "$OUT"

ETAS=(0.0001 0.001 0.01 0.1 1.0)
SIGMAS=(0.0001 0.001 0.01 0.1 1.0 1.5)
SHAPES=(helix sphere)
TASKS=(prior pgo)

JOBS=()
for task in "${TASKS[@]}"; do
  for shape in "${SHAPES[@]}"; do
    for eta in "${ETAS[@]}"; do
      for sig in "${SIGMAS[@]}"; do
        g2o="$DATA/$task/trajectories_1hz/noise_ig${eta}_n${sig}_${shape}.g2o"
        dir="$OUT/${task}_${shape}_ig${eta}_n${sig}"
        [ -f "$dir/summary.csv" ] && continue
        JOBS+=("OMP_NUM_THREADS=$THREADS_PER_RUN $BIN --input $g2o \
          --gt $DATA/gt/${shape}_1hz.tum --gt100 $DATA/gt/${shape}_100hz.tum \
          --config $shape --params $PARAMS --task $task --out $dir \
          --interp --nees > $dir.log 2>&1")
      done
    done
  done
done

printf '%s\n' "${JOBS[@]}" | xargs -P "$CONCURRENCY" -I {} bash -c '{}'
echo "sweep complete: $(ls "$OUT" | grep -c summary || true) dirs"
