#!/bin/bash
# Complete reproduction pipeline from a clean checkout (commit e50581f).
# See docs/PAPER_REPRODUCTION_REPORT.md for per-figure details.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# 0) build main project (Release) — requires yaml-cpp findable via CMAKE_PREFIX_PATH
mkdir -p build && cmake -S . -B build -DGSOLVER_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

# 1) build reproduction tooling (compiles the unmodified gsolver sources)
cmake -S reproduction -B reproduction/build -DCMAKE_BUILD_TYPE=Release
cmake --build reproduction/build -j"$(nproc)"

# 2) generate the synthetic datasets (seed 42, official script)
(cd examples/data/generators && ./generate_all_datasets.sh)

# 3) experiments
THREADS_PER_RUN=6 CONCURRENCY=8 ./reproduction/scripts/run_sweeps.sh   # Figs 8/9/15-18, Tab I
./reproduction/scripts/run_figure_05.sh
./reproduction/scripts/run_figure_06.sh
./reproduction/scripts/run_figure_11.sh
./reproduction/scripts/run_figure_12.sh
./reproduction/scripts/run_figure_01_10_torus_substitute.sh            # labelled substitutes
./reproduction/scripts/run_table_02_and_rs_figures.sh                  # Tab II, Figs 13/19/20
./reproduction/scripts/run_figure_14.sh
./reproduction/scripts/run_latency.sh                                  # Phase-5 timing (16/64 threads)

# 4) paper-figure digitisation (requires docs/2606.06250v1 GBP CPU.pdf + pymupdf)
python3 reproduction/scripts/digitise_paper_figures.py

# 5) figures + master table
python3 reproduction/scripts/build_all_figures.py
python3 reproduction/scripts/build_master_table.py

echo "Reproduction complete. See reproduction/reproduction_summary.csv and reproduction/figures/."
