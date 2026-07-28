#!/bin/bash
# Figures 1 and 10 SUBSTITUTE runs (original torus dataset is not public).
# 1) generate substitute torus GT (1 Hz + 100 Hz) with the labelled generator
# 2) learn Qc* with the OFFICIAL gp_hyperparam_trainer
# 3) write a reproduction-local config (repo config untouched)
# 4) Fig. 10: ablation scales from the official app {0.01,0.1,1,10,100} on a
#    torus PGO dataset (ig 1.0 / n 0.001, mirroring the USAGE.md example config)
#    + denser supplementary scales
# 5) Fig. 1: prior dataset with a sparse-observation gap; interpolate mean+cov
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GEN="$ROOT/reproduction/build/torus_generator"
BIN="$ROOT/reproduction/build/repro_synthetic"
TRAINER="$ROOT/build/examples/cpp/gp_hyperparam_trainer"
DATA="$ROOT/reproduction/results/local_gsolver/torus_substitute"
CFG="$ROOT/reproduction/configs"
mkdir -p "$DATA" "$CFG"

# k=24 poloidal loops, tube radius 4: dynamic enough that the learned Qc* sits
# well above the library's 1e-4 clamp (the paper torus's dynamics are unknown)
"$GEN" --out "$DATA/torus_gt_1hz"   --mode gt --hz 1   --k 24 --r 4
"$GEN" --out "$DATA/torus_gt_100hz" --mode gt --hz 100 --k 24 --r 4

QC=$("$TRAINER" "$DATA/torus_gt_1hz.tum" | grep -o "qc_diag: \[.*\]" | tail -1 | sed 's/qc_diag: //')
echo "learned torus Qc* = $QC"
python3 - "$ROOT/examples/config/experiment_params.yaml" "$CFG/experiment_params_repro.yaml" "$QC" <<'EOF'
import sys, yaml
src, dst, qc = sys.argv[1], sys.argv[2], yaml.safe_load(sys.argv[3])
cfg = yaml.safe_load(open(src))
cfg["torus_substitute"] = {"description": "SUBSTITUTE torus (local generator; not the paper torus)",
                           "qc_diag": qc, "num_iterations": 100}
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print("wrote", dst)
EOF

# --- Fig. 10 substitute: PGO torus, ig 1.0 / meas 1.0 (rot = /10).
# Noise level chosen (documented assumption) so the substitute operates where
# the paper's Fig. 10 sits (ATE ~ 1 m at s=1); the original config is not public.
"$GEN" --out "$DATA/torus_pgo_ig1.0_n1.0" --mode pgo --hz 1 --k 24 --r 4 \
  --ig-t 1.0 --ig-r 0.1 --sigma-t 1.0 --sigma-r 0.1 --seed 42
for s in 0.01 0.1 1.0 10.0 100.0 0.02 0.05 0.2 0.5 2.0 5.0 20.0 50.0; do
  OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} "$BIN" \
    --input "$DATA/torus_pgo_ig1.0_n1.0.g2o" \
    --gt "$DATA/torus_gt_1hz.tum" --gt100 "$DATA/torus_gt_100hz.tum" \
    --config torus_substitute --params "$CFG/experiment_params_repro.yaml" \
    --task ablation --qc-scale "$s" --interp --nees \
    --out "$ROOT/reproduction/results/local_gsolver/figure_10_substitute/scale_$s"
done

# --- Fig. 1 substitute: prior torus with sparse gap (fraction 0.55-0.80, keep 1/5).
# NOTE: with 1-in-10 priors in the gap this GBP configuration diverges (NaN);
# 1-in-5 is the sparsest stable setting found — recorded as a stability limit.
"$GEN" --out "$DATA/torus_prior_fig1" --mode prior --hz 1 --k 24 --r 4 \
  --ig-t 0.1 --ig-r 0.01 --sigma-t 0.1 --sigma-r 0.01 --seed 42 \
  --gap-start 0.55 --gap-end 0.80 --gap-keep 5
OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} "$BIN" \
  --input "$DATA/torus_prior_fig1.g2o" \
  --gt "$DATA/torus_gt_1hz.tum" --gt100 "$DATA/torus_gt_100hz.tum" \
  --config torus_substitute --params "$CFG/experiment_params_repro.yaml" \
  --task prior --interp --nees \
  --out "$ROOT/reproduction/results/local_gsolver/figure_01_substitute"
