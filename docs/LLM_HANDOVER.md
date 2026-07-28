# LLM Handover — G-solver paper reproduction → GPU accelerator project

**Read this first if you are picking up this work cold.** It is written for a model
or engineer with no prior context. Every number here was verified against files on
disk on 2026-07-26. Where something is uncertain or unverified, it says so.

---

## 0. Orientation in 60 seconds

- **Repo:** `/home/os717/gsolver` — a fork of `rvp-group/gsolver`, the official
  implementation of a 2026 RA-L paper on continuous-time SLAM.
- **What was done:** a complete, honest CPU reproduction of every experimental
  figure and table in that paper, from the public artefacts only, with zero
  modifications to the shipped code.
- **Outcome:** the paper's load-bearing claims reproduce, several to 3 decimal
  places. A defined set of artefacts could not be reproduced, all documented.
- **Where it's going:** the user's goal is a **GPU accelerator** for this solver.
  The reproduction was the due-diligence step. Verdict reached: the foundation is
  sufficient to start; four characterization tasks should precede any CUDA.
- **Do not** re-run the reproduction. It's done. Build on it.

Key documents in order of usefulness to a newcomer:
1. `docs/REPRODUCTION_SUMMARY_READABLE.md` — plain-language, section-by-section verdicts
2. `docs/PAPER_REPRODUCTION_REPORT.md` — the formal report (env, per-figure status, latency, author questions)
3. `docs/PAPER_REPRODUCTION_AUDIT.md` — Phase-1 audit of the repo before any experiments
4. `reproduction/reproduction_summary.csv` — 790 rows, one per paper data point
5. `reproduction/figure_manifest.yaml` — per-figure manifest with final statuses

---

## 1. What the paper is

**"Breaking Time: A Fully Gaussian Framework for Distributed and Continuous-Time
SLAM"** — Ceriola, Ferrari, Di Giammarino, Brizi, Grisetti (Sapienza Rome),
arXiv:2606.06250v1, RA-L 2026. PDF at `docs/2606.06250v1 GBP CPU.pdf` (untracked).

It combines two ideas:

1. **Gaussian Process motion priors** give a *continuous-time* trajectory. State at
   each knot is pose ⊕ body velocity (12-dim, `SE3PoseVel`). A constant-velocity GP
   prior between consecutive knots lets you query pose **and covariance** at any
   timestamp. Hyperparameters are a 6-vector `Qc` (acceleration PSD), learnable
   from data via the shipped `gp_hyperparam_trainer`.
2. **Gaussian Belief Propagation** solves it with only local message passing — no
   global factorization — which is what makes it distributable across cameras/robots.

**Their central argument vs the competitor (Hyperion, ECCV 2024, spline-based GBP):**
a spline couples each measurement to ~4 control points, creating dense short loops
that destabilize GBP; the GP prior chains only *consecutive* poses, so the graph is
near-tree and GBP converges. Our reproduction supports this: without the GP prior
GBP cannot recover the trajectory (Fig 6 — ATE 1.74 m vs 0.49 m with it).

**Solver anatomy (this is your accelerator workload).** `GbpSolver::performIteration`
runs four bulk-synchronous OpenMP phases per iteration:
1. every factor `updateFactor()` — relinearize at incoming message means, form
   `η = Jᵀ Σ⁻¹(−e)`, `Λ = Jᵀ Σ⁻¹ J`, then condition on connected edges
2. every factor `sendAllMessages()` — Schur-complement marginalization to each
   neighbour (recipient excluded from the conditioning set — differs from Hyperion)
3. every variable `updateMuAndSigma()` — sum incoming precisions/information,
   add `1e-6·I`, invert (12×12 for poses, 3×3 for points), retract onto SE(3)
4. every variable `sendAllMessages()` — cavity distribution per edge

All small dense linear algebra over a sparse static graph. Robust kernel exists
(`kRHuber`) but **is commented out** — `factor_node.cpp` hardcodes `double k = 1;`.
`SolverScheduleType::ASYNCHRONOUS` is an **empty stub**.

**The GBP problem types (your workload menu):**

| Problem | Graph shape | Verified size | Real data? |
|---|---|---|---|
| Prior-based localisation | GP chain + unary priors | 600 knots, ~1.2k factors | synthetic |
| PGO | chain + odometry + random loop closures (loopy) | 600 knots, ~1.5k factors | synthetic |
| ChArUco BA | poses ↔ marker landmarks + GP chain | 94 poses, 617 factors | real, tiny |
| **Stereo visual BA (KITTI)** | bipartite poses ↔ 3D points + GP chain | **477 poses, 45k landmarks, 158k stereo factors** | **real, full scale** |
| Rolling-shutter BA | one pose per image row-block | up to 2,379 blocks @0.25 ms spacing | real images, *simulated* shutter |
| Multi-camera distributed | KITTI split + landmark equality factors | 1–8 subgraphs, 9.5k–11.8k equality factors | real data, *synthetic* partitioning |

---

## 2. What was done, and the rules that were obeyed

A full reproduction of Figures 1, 5–20 and Tables I–II (Figs 2–4 are diagrams).

**Hard constraints that were honoured throughout — preserve these if you extend:**
- **Zero modifications to shipped code.** `gsolver/`, `examples/`, `python/` are
  untouched. All new code lives in `reproduction/`, compiled against the *same*
  unmodified library sources.
- Solver semantics unchanged: synchronous schedule, message order, fp64, factor
  definitions, GP priors, initialisation, interpolation, iteration counts, metrics.
- All instrumentation is **read-only between iterations** — it snapshots belief
  means and evaluates metrics; it never writes graph state.
- Paper values are never invented. Every one is tagged `PAPER_TABLE` (printed) or
  `DIGITISED_PDF` (extracted from vector paths, labelled *"digitised approximation
  from published figure"*). No digitised value is ever presented as raw measurement.

**Frozen commit:** `e50581f5f00cbe6239d498512a1b0aa9fa2d03ac` (branch `main`).
Full-history scan (`git log --all --name-only`, `--diff-filter=D`) confirmed: **no
plotting scripts, no raw results, no torus assets, no baseline code have ever
existed** in this repository. Everything quantitative had to be regenerated or
digitised.

---

## 3. Environment and build (exact — includes non-obvious gotchas)

| | Value |
|---|---|
| Machine | AMD EPYC 7713, 64 physical cores (SMT off), 251 GiB RAM |
| OS | RHEL 9.7, kernel 5.14.0-611.49.1.el9_7, governor `performance` |
| Compiler | GCC 11.5.0; flags `-O3 -DNDEBUG -std=c++17 -fopenmp` |
| CMake / Python | 3.26.5 / 3.9.25 (numpy 2.0.2, matplotlib 3.9.4, pymupdf 1.26.5, pandas, scipy) |
| Eigen | 3.4.0, pinned by FetchContent |
| Rerun SDK | 0.35.0 — **unpinned "latest"** in upstream CMake; build-only, never linked into measurements |
| yaml-cpp | 0.8.0, **built locally** at `/home/os717/gsolver-deps/install` (absent from system) |
| Hyperion (baseline attempt) | cloned at `/home/os717/gsolver-deps/hyperion`, commit `33465c9a1ec1d8efa78cd800e784d510f8dae0e8` |

**Paper's machine:** Intel Core i9-14900KF, 16 physical cores. Everything else
about their environment is unstated.

### Build gotchas you will hit

1. **Main project** — bare `yaml-cpp` target doesn't propagate includes here:
   ```bash
   cmake -S . -B build -DGSOLVER_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_PREFIX_PATH=/home/os717/gsolver-deps/install \
     "-DCMAKE_CXX_FLAGS=-isystem /home/os717/gsolver-deps/install/include" \
     -DCMAKE_EXE_LINKER_FLAGS=-L/home/os717/gsolver-deps/install/lib64
   cmake --build build -j$(nproc)
   ```
2. **Reproduction tooling** — two traps, both already solved in
   `reproduction/CMakeLists.txt`:
   - You **cannot** `file(GLOB)` the library sources. `instantiations.cpp`
     `#include`s the template `.cpp` files, so the source list must mirror
     `gsolver/CMakeLists.txt` exactly *and* `gsolver/src` must be on the private
     include path.
   - Link `yaml-cpp::yaml-cpp` (namespaced), not bare `yaml-cpp`.
   ```bash
   cmake -S reproduction -B reproduction/build -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_PREFIX_PATH=/home/os717/gsolver-deps/install \
     "-DCMAKE_CXX_FLAGS=-isystem /home/os717/gsolver-deps/install/include"
   cmake --build reproduction/build -j$(nproc)
   ```
3. **Rerun is deliberately never linked** into reproduction binaries — no
   visualisation anywhere in timed paths.

### Determinism — the single most important property for your project

The synchronous schedule produces **bit-identical results at any thread count**.
Each phase writes disjoint per-node state and reads only the previous phase's
output; per-node summation order is fixed by edge insertion order. Verified
empirically at 16 and 64 threads. **Thread count affects timing only, never
numbers.** This is what makes exact GPU validation possible.

---

## 4. Artefacts produced (file map)

```
docs/
  LLM_HANDOVER.md                     ← this file
  REPRODUCTION_SUMMARY_READABLE.md    ← plain-language section-by-section verdicts
  PAPER_REPRODUCTION_REPORT.md        ← formal report: env, statuses, latency, author questions
  PAPER_REPRODUCTION_AUDIT.md         ← Phase-1 repo audit (deps, executables, seeds, gaps)
reproduction/
  figure_manifest.yaml                ← per-figure manifest, final statuses
  reproduction_summary.csv            ← MASTER TABLE, 790 rows
  CMakeLists.txt                      ← standalone build of unmodified lib + new drivers
  configs/experiment_params_repro.yaml← repo config + substitute-torus entry (repo config untouched)
  src/
    repro_common.h        ← read-only eval: ATE/ARE (raw + Umeyama SE(3)-aligned),
                             NEES (3 conventions), Eq.18 energy, TUM I/O, peak RSS
    repro_synthetic.cpp   ← tasks prior|pgo|prior_nogp|ablation; energy/interp/NEES;
                             optional LM-damped GN+GP reimplementation
    repro_charuco.cpp     ← ChArUco BA, per-iteration substitute metrics
    repro_stereo.cpp      ← KITTI stereo BA, multi-robot split, MRE (2D + stereo-3),
                             per-frame projection dumps, per-robot timing, RSS
    repro_bench.cpp       ← microbenchmarks: 5 ops, raw samples, fallback protocol
    torus_generator.cpp   ← SUBSTITUTE torus (clearly labelled in all output)
  scripts/
    run_all.sh                              ← full pipeline from clean checkout
    run_sweeps.sh                           ← 120 runs: Figs 8/9/15–18 + Table I
    run_figure_05.sh / 06 / 11 / 12 / 14    ← per-figure
    run_figure_01_10_torus_substitute.sh    ← substitutes
    run_table_02_and_rs_figures.sh          ← Table II + Figs 13/19/20
    run_latency.sh                          ← Phase-5 timing, 16 & 64 threads, 10 reps
    digitise_paper_figures.py               ← PDF vector-path extraction
    build_all_figures.py                    ← all figures from CSV only
    build_master_table.py                   ← reproduction_summary.csv
  results/
    local_gsolver/   ← all local runs (sweeps/, figure_*/, table_02/, comparison CSVs)
    paper_exact/     ← printed Table I & II values (source: PAPER_TABLE)
    paper_digitised/ ← digitised figure data, every file header-labelled
    local_baselines/ ← GN+GP reimplementation output + README on what's missing
    latency_raw/     ← 25k raw microbenchmark samples + end_to_end/aggregate.csv
  figures/
    local/ comparison/ overlays/ paper_reference/   ← 25 figures, each .png + .pdf
```

---

## 5. Verified results

### Reproduced essentially exactly

**Table I (NEES, statistical consistency)** — paper values are printed, so this is
an exact-vs-exact comparison:

| | σ=1e-4 | 1e-3 | 1e-2 | 1e-1 | 1 |
|---|---|---|---|---|---|
| Helix paper | 0.030 | 1.533 | 3.925 | 4.510 | 4.694 |
| Helix local | 0.030 | 1.533 | 3.923 | 4.504 | 4.685 |
| Sphere paper | 0.030 | 1.542 | 4.187 | 4.589 | 6.515 |
| Sphere local | 0.030 | 1.539 | 4.188 | 4.592 | 6.459 |

Worst error 0.056 (0.9%). **Bonus finding:** the paper never states which η_ig was
used; we ran all five and NEES is *identical* for every one — the omission is
harmless. Convention determined: ξ = Log(T_est⁻¹·T_gt) against the 6×6 pose block
of the belief covariance, averaged over 600 poses, single run.

**Figure 8** (prior-based ATE sweeps): 44/48 points within tolerance, 4 CLOSE
(helix σ=1: 0.483 local vs 0.415 digitised). Curves overlay in all 8 panels.

**Figure 6** (GP ablation): GBP-only 1.7398 m, GBP+GP 0.4855 m — the paper's claim.

**Figure 14** (multi-camera scalability), local at 16 threads vs digitised paper:

| #cams | time/cam local→paper | mem/cam local→paper | ATE local→paper | equality factors |
|---|---|---|---|---|
| 1 | 111.4 s → 91.2 | 1272 MB → 674 | 0.668 → 0.703 | 0 |
| 2 | 55.3 → 46.1 | 631 → 342 | 0.664 → 0.703 | 9,526 |
| 4 | 27.7 → 23.1 | 315 → 171 | 0.665 → 0.703 | 10,308 |
| 8 | 13.9 → 11.5 | 157 → 85.5 | 0.667 → 0.703 | 11,784 |

Perfect halving; accuracy flat. Time offset = our slower cores. Memory offset ~2×
because local metric is process peak RSS ÷ #cameras (paper's attribution unstated).

**Table II endpoints** — readout 0: local 0.668 m / 0.0074 rad vs paper 0.671 /
0.007. Readout 0.1: 0.662 / 0.0081 vs 0.726 / 0.008.

### Reproduced qualitatively, numbers differ

**Figure 9** (PGO sweeps): 36/48 exact, 5 CLOSE, 7 NUMERICALLY_DIFFERENT. Sphere
matches throughout (σ=1: 0.838 vs ~0.83). Helix matches to σ=0.1 then runs
~1.6–1.7× high (σ=1: 0.468 vs 0.273; σ=1.5: 0.624 vs 0.402). Hypothesis: different
random draw of their helix PGO dataset. **Untested** — sweeping the generator seed
would confirm or kill it in an afternoon.

**Figure 5** (energy vs iteration): shape reproduces (GN collapses in 2–3
iterations, GBP descends gradually to a similar basin). Absolute scale does not:

| iteration | local G-solver | local GN reimpl | paper G-solver (digitised) | paper GN |
|---|---|---|---|---|
| 0 | 6.53e6 | 6.53e6 | 444.15 | 529.99 |
| 1 | 1.05e7 | 2.01e5 | 322.96 | 29.16 |
| 2 | 2.50e6 | 2.06e3 | 241.67 | 9.31 |
| 10 | 1.09e5 | — | 13.56 | 8.81 |
| 100 | 1.99e3 | 1975.4 (@40) | — | — |

Two anomalies in the paper's own figure: (a) its scale isn't Eq. 18 on the
documented dataset, (b) its two methods start at *different* energies at iteration 0
despite sharing an initial state. Some undocumented normalisation is in play.
**Untested hypothesis:** the caption says σ applies to "both [m] and [rad]", but
`generate_configs.py` forces rotation σ = translation σ/10. A dataset with equal
translation/rotation noise would give very different energies.

**Figure 12** (latencies) — local fallback protocol (10 warm-up solver iterations,
5,000 samples/op, `steady_clock`, all raw samples saved):

| operation | local median | local mean | paper box (digitised) |
|---|---|---|---|
| var→fac message | 2.31 µs | 2.38 | q1 1.20 / q3 1.88 |
| fac→var message | 2.41 µs | 4.74 | q1 0.81 / q3 2.47 |
| factor update | 3.05 µs | 6.40 | not reported |
| variable update | 4.71 µs | 4.76 | not reported |
| interpolation (mean+cov) | 50.25 µs | 32.55 | ≈4.35 |

Messages are hardware-consistent. Interpolation is 10× off — structural, not
hardware: our query includes full 12×12/24×24 covariance propagation *and* an O(N)
knot scan. Their protocol is unknown.

### Not reproduced

**Table II middle columns** — the headline anomaly:

| readout | paper ATE / ARE / MRE | local (SE3-aligned) | verdict |
|---|---|---|---|
| 0 | 0.671 / 0.007 / 1.704 | 0.668 / 0.0074 / 1.470 | ✅ exact |
| 1e-4 | 0.676 / 0.007 / 1.708 | **76,139** / 1.951 / 2.570 | ❌ diverged |
| 1e-3 | 0.676 / 0.008 / 1.709 | **21.01** / 0.313 / 3.096 | ❌ diverged |
| 1e-2 | 0.684 / 0.007 / 1.715 | **5.39** / 0.152 / 2.022 | ❌ diverged |
| 1e-1 | 0.726 / 0.008 / 1.844 | 0.662 / 0.0081 / 1.427 | ✅ exact |

Same code, same shipped data, deterministic solver. Divergence *worsens* with more
iterations (10 → ATE 21 m; 30 → NaN; 100 → total). Excluded projections 0.80–2.68%.
Two conventions had to be inferred: SE(3) alignment (unaligned readout-0 ATE is
1.301 vs paper 0.671, aligned is 0.668 → alignment is clearly what they used) and
the MRE definition (stereo 3-vector [u_l, v, disparity] is closer to their numbers
than 2-D pixel distance; both are recorded).

**Untried leads — this is the most valuable open experiment:**
1. **The Huber robust kernel is commented out** in `gsolver/src/graph/factor_node.cpp`
   (`double k = 1;`). These configs create factors with precisions ~10¹¹; an
   unrobustified outlier there would plausibly cause exactly this divergence.
2. Config key: we used `kitti_05` (paper says hyperparameters come from KITTI 05),
   but `kitti_06` exists in the config with different values.

### End-to-end latency (10 reps synthetic, 3 KITTI; minimal instrumentation)

| workload | 16 threads | 64 threads |
|---|---|---|
| prior helix (600 poses, 100 it) | 239 ms | 5,362 ms |
| prior sphere | 240 ms | 5,357 ms |
| PGO helix | 336 ms | 5,452 ms |
| PGO sphere | 317 ms | 5,449 ms |
| KITTI RS 1 ms (2,379 blocks, 158k factors, 10 it) | 31,242 ms | 12,045 ms |

**Two findings that matter enormously for a GPU port:**
- On small graphs, 64 threads is **~16× slower** than 16. Synchronous GBP at fine
  granularity is fork/join- and synchronization-bound, not compute-bound.
- On the big graph, 4× the cores buys only 2.6×. Sublinear.
- **Amdahl hazard:** KITTI graph *construction* takes ~14.5 s while the 64-thread
  solve takes ~12.0 s. Accelerating only the solver by 100× yields ~1.8× end-to-end.

### Master table status counts (790 rows)

`BLOCKED_MISSING_BASELINE` 314 · `BLOCKED_MISSING_DATA` 134 ·
`EXACT_OR_WITHIN_TOLERANCE` 106 · `NOT_REPRODUCIBLE_FROM_PUBLIC_ARTEFACTS` 78 ·
`NUMERICALLY_DIFFERENT` 66 · `BLOCKED_MISSING_CONFIGURATION` 34 ·
`VISUALLY_EQUIVALENT` 28 · `PAPER_VALUE_NOT_EXACT` 15 · `CLOSE` 12 · `DIVERGED` 3

(The large blocked counts are dominated by the ~314 Hyperion baseline points, which
were never runnable — not by G-solver failures.)

---

## 6. Gap register

**Tier 1 — impossible without the authors.** No workaround exists.
- Torus dataset + its learned `Qc*` (Figs 1, 10). Generator supports helix/sphere only.
- ChArUco camera intrinsics + pixel detections (Fig 11 y-axis). The shipped g2o has
  pose→marker SE(3) transforms, not pixels. Pixel MRE is unrecoverable.
- GN+GP baseline implementation (Fig 5 black curve). Never released.
- Fig 5's energy normalisation.

**Tier 2 — fixable, simply not done in this session.**
- **KITTI 06 images** for Figs 13/19/20 backdrops. These are *freely downloadable*
  from the KITTI odometry benchmark (registration, ~22 GB). All projected pixel
  coordinates are already saved; dropping real photos underneath is ~1 hour.
  Overlays currently render on grey canvas. **This was an omission, not a wall.**
- **Hyperion baselines.** Code is public and BSD-3 (cloned, commit `33465c9`).
  Blockers: needs ROS 2/ament + Ceres + abseil + glog + fmt (no root here), and
  nobody has written a g2o→Hyperion converter. Days of work, not impossible.
- **Fig 9 helix seed hypothesis.** Cheap to test.

**Tier 3 — open scientific question.** Table II middle columns (see §5).

---

## 7. Foundation for the GPU accelerator — what's usable and why

**Assets you now have:**

1. **A bit-exact correctness oracle.** Determinism means a GPU port can be validated
   *message by message against intermediate state*, not just on final ATE. Rare and
   valuable. Preserve per-node summation order in fp64 and you get bitwise equality.
2. **CPU baselines on your own hardware** with per-phase breakdowns. Note: the
   paper's i9 is irrelevant to your speedup claim — you compare CPU-G-solver vs
   GPU-G-solver on the *same* machine.
3. **Verified accuracy targets** you know are attainable: Table I to 3 decimals,
   KITTI ATE 0.668 m, Fig 8 point-for-point.
4. **Eleven real KITTI sequences.** The repo ships stereo BA graphs for 00–10
   (~79k to ~728k lines, roughly 10× size range) — the paper used only 06. An
   accelerator evaluation on all eleven is strictly more thorough than the original.
5. **A measured motivation** for GPU: the 16×-slowdown-at-64-threads result and
   sublinear scaling on the large graph.
6. **Kernel shapes:** four bulk-synchronous phases of tiny dense ops (12×12 and
   24×24 inverses, 3–12-dim messages, Schur complements) over a static sparse
   bipartite graph. About as accelerator-friendly as SLAM gets.

**Risks:**

- **Novelty positioning is the real threat, not the reproduction gaps.** "GPU
  acceleration of GBP" is not novel on its own: Davison & Ortiz framed GBP as
  *the* algorithm for graph-parallel hardware (FutureMapping 2; Ortiz's 2023
  Imperial thesis ran GBP on Graphcore IPUs), and Hyperion bills itself as a
  high-performance GBP framework. Sharper hooks available from this work:
  (a) **topology** — GP chain priors vs spline dense loops is a *memory-access*
  argument nobody has measured; (b) **precision** — GBP message numerics on stiff
  real factor graphs are uncharacterized and you hold the pathological datasets;
  (c) **scheduling** — the async schedule is an empty stub and sync GBP's four
  global barriers per iteration are exactly what kills GPU occupancy.
- Amdahl: graph construction ≈ solve time (see §5).
- Rolling-shutter divergence: a bit-exact GPU port will faithfully diverge too.
- Async scheduling would be a *contribution*, but you lose the bit-exact oracle
  there and need a separate convergence argument.

---

## 8. Next steps, prioritized

**Before writing any CUDA (days, not weeks):**

1. **Per-phase profile of the real KITTI solve.** We have per-op microbenchmarks on
   a 600-node synthetic graph, but *not* the proportion of the 31 s spent in each of
   the four phases on KITTI. You cannot pick optimization targets without it.
   Tooling exists in `repro_stereo.cpp`; add per-phase timers around the four
   `performIteration` stages.
2. **Precision sensitivity study on the CPU reference.** fp64 vs fp32 vs mixed.
   This is *the* question for a GPU port and it is unanswered. The rolling-shutter
   datasets (precisions ~10¹¹) will answer it emphatically. Doing this before the
   port avoids discovering fp32 fails after the kernels are written.
3. **Amdahl breakdown across all eleven sequences.** Load / build / solve /
   interpolate / evaluate, so you know what an accelerated solver actually buys
   end-to-end and can scope honestly.
4. **Resolve or scope out Table II.** Test the Huber kernel and `kitti_06`
   hypotheses (one day). Highest-value single experiment remaining.

**Cheap wins worth folding in:**
- Download KITTI 06 images, re-render Figs 13/19/20 with real backdrops (~1 hour).
- Sweep the generator seed to test the Fig 9 helix hypothesis (~afternoon).
- Email the authors the eight questions in `PAPER_REPRODUCTION_REPORT.md` §9. The
  rolling-shutter recipe is the one that materially affects paper claims.

**Then:** scope the accelerator paper around **KITTI stereo BA** (real robotics
problem, real scale, 11 sequences, exact reference values), with the synthetic
sweeps as controlled microbenchmarks. Consider adding one genuinely-real
asynchronous-sensor dataset (true rolling-shutter or event camera) — the paper's
hard conditions are *simulated*, so real ones would exceed it.

---

## 9. Open questions for the paper's authors

Full list in `docs/PAPER_REPRODUCTION_REPORT.md` §9. The four that matter most:
1. **Table II**: what produced the "Ours" values at readouts 1e-4…1e-2? The public
   code diverges there. Was the robust kernel enabled?
2. **Fig 5**: what normalisation does the energy axis use, and why do the two
   methods start at different energies from the same initial state?
3. **Fig 9**: was the helix PGO dataset generated with a seed other than 42?
4. **Fig 10**: the torus generator, its `Qc*`, and the actual scale grid (the
   published curve has ~59 vertices; the shipped app tests 5).

---

## 10. Traps for whoever continues

- **Don't `file(GLOB)` the gsolver sources** — `instantiations.cpp` includes
  template `.cpp` files; mirror `gsolver/CMakeLists.txt` and put `gsolver/src` on
  the include path.
- **`yaml-cpp::yaml-cpp`**, not bare `yaml-cpp`.
- **Don't modify shipped code** — the reproduction's credibility rests on it. Put
  everything in `reproduction/` (or a new sibling directory for the GPU work).
- **Rolling-shutter g2o files list timestamps in *descending* order within a frame.**
  `getPoseVariableNodes()` sorts by timestamp, so the chain is correct — but any
  new code that trusts file order will silently build a broken graph.
- **`add_gp_prior_factor_node` clamps `Qc` with `cwiseMax(1e-4)`; `GPInterp*` instead
  *adds* `1e-4` to entries below `1e-4`.** These two floors differ. Preserved as-is;
  do not "fix" it without recording that you changed solver semantics.
- **The paper's captions say σ applies to "both [m] and [rad]"**, but
  `generate_configs.py` (authoritative) pairs rotation σ = translation σ/10. All
  reproduction work follows the code.
- **Rerun SDK is fetched as "latest"** — an upstream release can change your build.
- Metrics are thread-independent; timings are not. Never mix a 64-thread timing run
  with a 16-thread one in the same comparison.
