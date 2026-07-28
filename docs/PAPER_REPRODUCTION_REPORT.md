# Paper Reproduction Report — "Breaking Time: A Fully Gaussian Framework for Distributed and Continuous-Time SLAM"

Paper: arXiv:2606.06250v1 (RA-L 2026), Ceriola, Ferrari, Di Giammarino, Brizi, Grisetti.
Reproduction of every experimental artefact (Figures 1, 5–20; Tables I, II) from the
public repository, CPU-only. Companion documents:
- Phase-1 audit: [`docs/PAPER_REPRODUCTION_AUDIT.md`](PAPER_REPRODUCTION_AUDIT.md)
- Per-figure manifest with final statuses: [`reproduction/figure_manifest.yaml`](../reproduction/figure_manifest.yaml)
- **Master comparison table (790 rows, one per paper point/panel):
  [`reproduction/reproduction_summary.csv`](../reproduction/reproduction_summary.csv)**

## 1. Environment

| | Paper machine | Local machine |
|---|---|---|
| CPU | Intel Core i9-14900KF, 16 physical cores | AMD EPYC 7713, 64 physical cores (SMT off) |
| RAM | not stated | 251 GiB |
| OS | not stated (repo Docker: Ubuntu 22.04) | RHEL 9.7, kernel 5.14.0-611.49.1 |
| Compiler | not stated | GCC 11.5.0, `-O3 -DNDEBUG -std=c++17 -fopenmp` |
| CMake / Eigen / yaml-cpp / OpenMP | — | 3.26.5 / 3.4.0 (pinned) / 0.8.0 (local build) / libgomp 4.5 |
| Rerun SDK | unpinned "latest" in CMake | resolved 0.35.0 (build-only; never linked into measurements) |
| Python (plotting only) | — | 3.9.25; numpy 2.0.2, matplotlib 3.9.4, pymupdf 1.26.5 |
| Governor / threads | — | `performance`; experiments at OMP_NUM_THREADS=16 ("paper_parallelism") and 64 ("local_native") |

Local runtimes are **not** expected to match the paper machine; both are reported side
by side. The synchronous GBP schedule is numerically deterministic w.r.t. thread count
(verified in audit §4), so accuracy results are thread-independent.

## 2. Repository state

Commit **`e50581f5f00cbe6239d498512a1b0aa9fa2d03ac`** (branch `main`), working tree
clean apart from the paper PDF. No plotting scripts, raw results, torus assets or
baseline code exist anywhere in the history (full-history scan in audit §1).
No shipped file was modified; all new code lives under `reproduction/`
(instrumented drivers that reuse the unmodified library and builder code paths,
read-only metric evaluation; schedule/precision/factors/priors/initialisation/
iteration counts untouched).

## 3. Figure-by-figure status

Full details per entry in the manifest. Summary (G-solver side / baseline side):

| Artefact | G-solver reproduction | Baselines / other blockers |
|---|---|---|
| Fig. 1 | **substitute reproduced** (labelled; covariance grows 2.3× in sparse region) | original torus not public → BLOCKED |
| Fig. 5 | curve shape reproduced; **paper y-scale not derivable from documented config** | GN+GP baseline missing; labelled reimplementation supplied |
| Fig. 6 | **REPRODUCED** (GBP-only stays noisy, ATE 1.74 m; GBP+GP recovers helix, 0.49 m) | — |
| Fig. 7 | Ours panel **REPRODUCED** | Hyperion panels MISSING_BASELINE |
| Fig. 8 | **REPRODUCED**: 44/48 points within tolerance, 4 CLOSE | Hyperion curves digitised only |
| Fig. 9 | sphere **REPRODUCED**; helix σ≥0.1 ~1.7× high (suspected different dataset realization) | Hyperion digitised only |
| Tab. I | **REPRODUCED** (max abs diff 0.056 = 0.9%; η_ig proven irrelevant) | — |
| Fig. 10 | substitute reproduced with the paper's U-shape (min in s∈[1,10]) | torus + dense scale list not public |
| Fig. 11 | run reproduced; **pixel metric BLOCKED** (no calibration/pixels in repo); substitute metric converges smoothly as described | Hyperion digitised only |
| Fig. 12 | local fallback-protocol boxes produced (raw samples saved) | Hyperion digitised; paper protocol unknown |
| Fig. 13 | overlays + all projected coordinates produced | image backdrop (KITTI) + Hyperion blocked; exact frame not stated |
| Tab. II | readouts 0 and 0.1: **EXACT** (aligned); readouts 1e-4/1e-3/1e-2: **DIVERGED locally** — key clarification item | Z-Spline rows preserved from printed table |
| Fig. 14 | **REPRODUCED** (time/memory halve per camera doubling; ATE/ARE flat) | — |
| Figs. 15–18 | Ours panels **REPRODUCED** (all 6 σ, both tasks, both shapes, 100× interpolation saved) | Hyperion panels MISSING_BASELINE |
| Figs. 19–20 | all 10 frames: coordinates + overlays produced | backdrops + Hyperion blocked; local RS solution partly diverged |

Counts over the 790 master-table rows: 106 EXACT_OR_WITHIN_TOLERANCE, 12 CLOSE,
66 NUMERICALLY_DIFFERENT, 3 DIVERGED, 28 VISUALLY_EQUIVALENT, 15 PAPER_VALUE_NOT_EXACT,
314 BLOCKED_MISSING_BASELINE, 134 BLOCKED_MISSING_DATA, 34 BLOCKED_MISSING_CONFIGURATION,
78 NOT_REPRODUCIBLE_FROM_PUBLIC_ARTEFACTS (substitute/supplementary rows).

## 4. Generated figures

Local: `reproduction/figures/local/` — figure_01_substitute, figure_06, figure_07,
figure_15 … figure_18 (each .png + .pdf).
Local-vs-paper comparisons: `reproduction/figures/comparison/` — figure_05,
figure_08, figure_09, figure_10_substitute, figure_11, figure_12, figure_14.
Rolling-shutter overlays: `reproduction/figures/overlays/` — figure_13_frame_320,
figure_19_frame_{79,320,354,429,433}, figure_20_frame_{668,670,674,715,719}.
All plots are drawn from the machine-readable CSV/TUM files under
`reproduction/results/`; every sampled point carries a visible marker; digitised
paper series are dashed with open markers and labelled "paper (digitised)".

## 5. Key value comparisons

### Table I (NEES, prior-based localisation) — paper values are PAPER_TABLE (exact)

| Trajectory | σ | Paper | Local | Δ |
|---|---|---|---|---|
| Helix | 1e-4 / 1e-3 / 1e-2 / 1e-1 / 1 | 0.030 / 1.533 / 3.925 / 4.510 / 4.694 | 0.030 / 1.533 / 3.923 / 4.504 / 4.685 | ≤0.009 |
| Sphere | 1e-4 / 1e-3 / 1e-2 / 1e-1 / 1 | 0.030 / 1.542 / 4.187 / 4.589 / 6.515 | 0.030 / 1.539 / 4.188 / 4.592 / 6.459 | ≤0.056 |

Protocol determined by reproduction: single run (seed 42 datasets), average over the
600 poses of one trajectory; error ξ = Log(T_est⁻¹·T_gt) against the 6×6 pose block
of the belief covariance (tangent at the estimate). **η_ig is irrelevant**: identical
NEES to 6 decimal places for every η_ig ∈ {1e-4…1} (the estimator converges to the
same posterior). Data: `reproduction/results/local_gsolver/table_01_comparison.csv`.

### Table II (rolling shutter, KITTI 06; config kitti_05, 10 iterations)

| Readout [s] | Paper Ours ATE/ARE/MRE | Local ATE(aligned)/ARE(aligned)/MRE(stereo-3) | Status |
|---|---|---|---|
| 0 | 0.671 / 0.007 / 1.704 | **0.668 / 0.0074 / 1.470** | EXACT (ATE, ARE), CLOSE (MRE) |
| 1e-4 | 0.676 / 0.007 / 1.708 | 76 139 / 1.95 / 2.570 | **DIVERGED locally** |
| 1e-3 | 0.676 / 0.008 / 1.709 | 21.0 / 0.313 / 3.096 | **DIVERGED locally** |
| 1e-2 | 0.684 / 0.007 / 1.715 | 5.39 / 0.152 / 2.022 | **DIVERGED locally** |
| 1e-1 | 0.726 / 0.008 / 1.844 | **0.662 / 0.0081 / 1.427** | EXACT (ATE, ARE), CLOSE (MRE) |

Notes recorded per row in the master table: (i) ATE/ARE alignment convention is not
stated in the paper; the SE(3)-aligned values match at readout 0 almost exactly
(0.668 vs 0.671) while unaligned gives 1.30, so alignment (or an equivalent gauge
fix) was evidently used; (ii) the MRE definition is unstated — both 2-D pixel
distance and the stereo 3-vector [u_l, v, disparity] are reported (stereo-3 is
closer to the printed values); the in-image rule uses 1226×370 (KITTI 06
resolution, not in the repo); excluded projections are 0.8–2.7% (counts per row).
The mid-readout divergence is *reproducible and worsens with iterations*
(10 it → ATE 21 m; 30 it → NaN; 100 it → full divergence at 1 ms readout), i.e.
the public pipeline at the documented configuration does not deliver the paper's
"Ours" values for readouts 1e-4…1e-2 (see §9, question 2). Z-Spline rows are kept
exclusively from the printed table (baseline not runnable locally).

### Figures 8/9 (ATE sweeps; paper values digitised, ±~0.05 m)

Local Ours vs digitised paper Ours: Fig. 8 — 44/48 within tolerance, 4 CLOSE
(helix σ=1: 0.483 local vs 0.415 digitised). Fig. 9 — 36/48 within tolerance,
5 CLOSE, 7 NUMERICALLY_DIFFERENT: sphere panels match throughout (σ=1: 0.838
local vs ~0.83 paper), helix panels agree for σ≤0.1 but run ~1.6–1.7× high at
σ≥1 (0.468 vs 0.273 at σ=1; 0.624 vs 0.402 at σ=1.5). Since
prior-helix and PGO-sphere match, a different random realization of the paper's
helix-PGO dataset is the leading explanation (§9, question 3). All 288 digitised
paper points and all local points are in the master table; local ARE is stored in
each sweep summary.

### Figure 14 (multi-camera scalability; paper values digitised)

| #cameras | avg time/cam [s] paper→local | avg mem/cam [MB] paper→local | ATE [m] paper→local | ARE [rad] paper→local |
|---|---|---|---|---|
| 1 | 91.2 → 111.4 | 674 → 1272 | 0.703 → 0.668 | 0.0067 → 0.0074 |
| 2 | 46.1 → 55.3 | 342 → 631 | 0.703 → 0.664 | 0.0067 → 0.0075 |
| 4 | 23.1 → 27.7 | 171 → 315 | 0.703 → 0.665 | 0.0067 → 0.0075 |
| 8 | 11.5 → 13.9 | 86 → 157 | 0.703 → 0.667 | 0.0067 → 0.0075 |

Both halving trends and the flat accuracy reproduce. Local memory is process peak
RSS divided by #cameras (documented estimate; the paper's attribution method is
unstated — likely why absolute memory differs ~2×). Equality factors: 9 526 /
10 308 / 11 784 for 2/4/8 cameras; per-segment counts in `per_robot.csv`.

## 6. Latency

### Figure 12 microbenchmarks (local fallback protocol: helix prior graph, 10 warm-up solver iterations, 5 000 samples/op, steady_clock, raw samples saved)

| Operation | Local median (EPYC 7713) | Paper box median, digitised (i9-14900KF) |
|---|---|---|
| variable→factor message | 2.31 µs | ≈1.35 µs |
| factor→variable message | 2.41 µs | ≈1.55 µs |
| interpolation (mean+covariance query) | 50.3 µs | ≈4.35 µs |
| factor update / variable update (extra) | 3.05 µs / 4.71 µs | not in paper |

Message latencies land in the same regime (slower core ⇒ ~1.6×). The interpolation
gap is structural: the local query includes the full 12×12/24×24 covariance
propagation *and* `GPInterpAtTimestamps`' O(N) knot scan; the paper's benchmark
protocol (graph, what a "query" includes) is not recoverable (§9, question 5).
Raw samples: `reproduction/results/latency_raw/fig12_helix_prior/raw_*.csv`.

### End-to-end (10 repetitions synthetic / 3 KITTI; minimal instrumentation `--pure`; mean±std of GBP solve)

| Workload | 16 threads | 64 threads |
|---|---|---|
| prior helix (600 poses, 100 it) | 239 ± 2 ms | 5 362 ± 40 ms |
| PGO helix (600 poses, 100 it) | 336 ± 59 ms | 5 452 ± 18 ms |
| KITTI RS 1 ms (2 379 blocks, 45 k landmarks, 158 k stereo factors, 10 it) | 31.2 ± 0.4 s | 12.0 ± 0.03 s |

Notable: on the small synthetic graphs 64 threads is ~17× *slower* than 16
(OpenMP fork/join dominates the tiny per-node work), while the large KITTI graph
gains 2.6×. Dataset-load / graph-build / interpolation / metric phase timings for
every run: `reproduction/results/latency_raw/end_to_end/aggregate.csv`.
Do not read paper-vs-local runtimes as a correctness check — different machines.

## 7. Missing artefacts and ambiguities (complete list)

1. **Torus** trajectory, generator, Qc\* (Figs. 1, 10) — absent from repo and history.
2. **Hyperion B-/Z-Spline** experiment glue — upstream (VIS4ROB-lab/hyperion,
   commit 33465c9) requires a ROS 2/ament + Ceres + abseil stack (not installable
   here without root) and ships no consumer for these datasets or the paper's
   synchronous-schedule comparison; spline configs unstated → MISSING_BASELINE.
3. **GN+GP baseline** — not public; independent LM-damped reimplementation supplied,
   clearly labelled, never presented as the paper baseline.
4. **ChArUco camera calibration / pixel detections / marker geometry** — absent →
   Fig. 11's pixel MRE not computable from public artefacts.
5. **KITTI 06 images** — not in repo (licensing) → Figs. 13/19/20 backdrops blocked.
6. **Protocol gaps**: Table I η_ig (resolved: irrelevant); Table II
   alignment + MRE definition (inferred, documented); Fig. 5 energy normalisation
   (unresolved); Fig. 10 noise config & task and the dense scale list (figure shows
   59 vertices vs 5 scales in shipped app); Fig. 12 benchmark protocol; Fig. 13
   source frame; Fig. 14 memory attribution; image resolution for the in-image rule.
7. Rerun SDK unpinned ("latest") in the build system.
8. Paper caption vs generator noise mismatch: captions say σ/η apply "both [m] and
   [rad]"; `generate_configs.py` (authoritative) pairs rotation σ = translation σ/10.

## 8. Digitisation register

Every digitised value is stored under `reproduction/results/paper_digitised/` with
the header “digitised approximation from published figure”, produced by
`reproduction/scripts/digitise_paper_figures.py` (vector-path extraction from the
arXiv PDF; calibration via grid lines + tick labels; per-figure method in-file):
fig05 (52 marker points), fig08/fig09 (144 curve points each; y zero-anchor
uncertainty ±~0.05 m documented), fig10 (118 vertices — revealing a ~59-value scale
sweep), fig11 (76 points; B-Spline divergence retained), fig12 (6 box summaries),
fig14 (16 bar values). Values rendered off-axis by the paper's clip mask were
recovered from the unclipped path data and flagged `rendered_off_axis`.
No digitised value is ever presented as an original raw measurement.

## 9. Questions for the authors

1. **Fig. 5**: what normalisation does E_tot use? A strict Eq.-18 evaluation of the
   documented config gives E₀ ≈ 6.5×10⁶, the figure starts at ≈450/530 — and the
   two methods start from different energies at iteration 0 despite sharing the
   initial state.
2. **Table II**: which exact pipeline produced Ours at readouts 1e-4…1e-2? The
   public code + datasets + kitti_05 config (10 iterations) diverge locally
   (worsening with iterations), while readouts 0 and 0.1 match exactly.
3. **Fig. 9 helix**: was the helix PGO dataset regenerated with a different seed or
   noise realization than `generate_configs.py` seed 42?
4. **Fig. 10**: torus generator/Qc\* and the actual scale grid (~59 values in the
   figure vs `{0.01,0.1,1,10,100}` in the shipped app), plus the noise config.
5. **Fig. 12**: benchmark graph, sample counts, and whether the interpolation query
   includes covariance.
6. ATE/ARE alignment convention for KITTI (SE(3) alignment inferred), and the MRE
   pixel-error definition (2-D vs stereo 3-vector).
7. ChArUco calibration + detections for Fig. 11's pixel metric.
8. Was NEES averaging single-run per σ (as reproduced) or multi-seed?

## 10. Rerun commands

Everything from a clean checkout: `./reproduction/scripts/run_all.sh`
(build → datasets → sweeps → per-figure scripts → latency → digitisation → figures
→ master table). Individual artefacts: `reproduction/scripts/run_figure_05.sh`,
`run_figure_06.sh`, `run_sweeps.sh` (Figs 8/9/15–18 + Table I),
`run_figure_01_10_torus_substitute.sh`, `run_figure_11.sh`, `run_figure_12.sh`,
`run_table_02_and_rs_figures.sh` (Table II + Figs 13/19/20), `run_figure_14.sh`,
`run_latency.sh`; then `digitise_paper_figures.py`, `build_all_figures.py`,
`build_master_table.py`.
