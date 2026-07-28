# Paper Reproduction Audit — "Breaking Time: A Fully Gaussian Framework for Distributed and Continuous-Time SLAM"

Paper: arXiv:2606.06250v1 [cs.RO], 4 Jun 2026 (RA-L 2026).
Authors: D. Ceriola*, S. Ferrari*, L. Di Giammarino, L. Brizi, G. Grisetti.
Local PDF: `docs/2606.06250v1 GBP CPU.pdf` (14 pages, untracked file in working tree).

This audit was produced BEFORE any experiment execution or code addition, per the
reproduction protocol. It records what the public repository provides, what is
missing, and how each paper artefact maps to code.

---

## 1. Repository state (frozen)

| Item | Value |
|---|---|
| Remote | `https://github.com/osharifdev/gsolver.git` (fork of `rvp-group/gsolver`) |
| Branch | `main` (== `origin/main`) |
| **Commit** | **`e50581f5f00cbe6239d498512a1b0aa9fa2d03ac`** — "Consistent documentation, generation of experiments and consistency with code." (Luca Di Giammarino, 2026-06-25) |
| Working tree at audit time | clean except untracked `docs/2606.06250v1 GBP CPU.pdf` |
| Tags | none |
| Branches | `main` only |

Full history (4 commits): `07270f8` Initial commit → `c9d6aa5` initial commit
(core lib + synthetic/charuco experiments) → `c03cabf` Real world experiments
release (stereo_visual, KITTI data, GP interpolation) → `e50581f` docs.

**History search result:** `git log --all --name-only` over all commits contains
NO plotting scripts, NO raw result files (CSV/JSON), NO paper figure sources, NO
torus generator, NO Hyperion/GN baseline code, NO benchmark harness. Deleted-file
scan (`--diff-filter=D`) is empty. Everything quantitative in the paper must
therefore be regenerated from the experiment executables, or taken from values
printed in the paper (Tables I, II), or digitised from the PDF.

## 2. Build instructions (official)

Native (used here; Docker is the documented alternative):

```bash
mkdir build && cd build
cmake .. -DGSOLVER_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
make -j
```

`CMAKE_BUILD_TYPE` defaults to Release. Local deviation (environment-level only,
no repo change): system yaml-cpp is absent on this host, so yaml-cpp 0.8.0 was
built locally and injected via `CMAKE_PREFIX_PATH=/home/os717/gsolver-deps/install`
plus `-DCMAKE_CXX_FLAGS="-isystem /home/os717/gsolver-deps/install/include"`
`-DCMAKE_EXE_LINKER_FLAGS=-L/home/os717/gsolver-deps/install/lib64`.
Resulting compile flags: `-O3 -DNDEBUG -std=c++17 -fopenmp`.

## 3. Dependencies and versions

| Dependency | Version | Source | Pinned? |
|---|---|---|---|
| Eigen | 3.4.0 | CMake FetchContent (URL + MD5) | yes |
| Rerun C++ SDK | 0.35.0 (resolved at configure time) | FetchContent `releases/latest` | **NO — unpinned "latest"; reproducibility caveat** |
| yaml-cpp | 0.8.0 (local build) | system dep in official docs (Ubuntu 22.04 ships 0.7.0) | no (system) |
| OpenMP | GCC libgomp, OpenMP 4.5 | system | n/a |
| nanobind | v2.7.0 | FetchContent (only if `GSOLVER_BUILD_PYTHON=ON`) | yes |
| Compiler (paper) | unknown (Docker base Ubuntu 22.04 → GCC 11.x likely) | | |
| Compiler (local) | GCC 11.5.0 (RHEL 9.7) | | |
| CMake (local) | 3.26.5 | | |
| Python (local) | 3.9.25; numpy 2.0.2, matplotlib 3.9.4, pyyaml 5.4.1, pandas 2.3.3, scipy 1.13.1 | plotting only | |

## 4. Hardware

| | Paper | Local |
|---|---|---|
| CPU | Intel Core i9-14900KF, 16 physical cores | AMD EPYC 7713, 64 physical cores (SMT off: 1 thread/core) |
| RAM | not stated | 251 GiB |
| OS | not stated (Docker: Ubuntu 22.04) | RHEL 9.7, kernel 5.14.0-611.49.1.el9_7 |
| Governor | not stated | `performance` |

Latency experiments are run in two modes: `paper_parallelism` (OMP_NUM_THREADS=16)
and `local_native` (OMP_NUM_THREADS=64). All apps call
`omp_set_num_threads(omp_get_max_threads())`, so `OMP_NUM_THREADS` controls the
thread count without code changes. **The synchronous schedule computes identical
numerical results for any thread count** (each phase writes disjoint per-node
state and reads only the previous phase's outputs; per-node summation order is
fixed), so thread count affects timing only.

## 5. C++ executables (all under `examples/cpp/apps/`, built to `build/examples/cpp/`)

| Executable | Purpose | CLI |
|---|---|---|
| `data_generator` | synthetic helix/sphere dataset generation | `<config.yaml> <helix\|sphere> <output_prefix>` |
| `prior_experiment` | prior-based localisation (GP + absolute pose priors) | `<in.g2o> <out.tum> <traj-type> [-visualize]` |
| `pgo_experiment` | PGO (GP + odometry factors) | same |
| `hyperparam_ablation_experiment` | Qc scale ablation, scales `{0.01, 0.1, 1.0, 10.0, 100.0}` hard-coded | `<in.g2o> <out_folder> <traj-type> [-visualize]` |
| `charuco_experiment_ba` | ChArUco BA (pose–landmark SE3 factors) | `<in.g2o> <out.tum> <noise_level> [-visualize]` |
| `charuco_experiment_islam` | incremental SLAM variant | `<in.g2o> <out.tum> [-visualize]` |
| `stereo_visual` | KITTI stereo BA; multi-robot splitting; GP interpolation to GT stamps | `<in.g2o> <gt.tum> <out.tum> <traj-type> [num_robots] [-visualize]` |
| `gp_hyperparam_trainer` | learn `qc_diag` from GT TUM (acceleration variance) | `<in.tum>` |

Python mirrors exist for pgo/prior/ablation/charuco under `examples/python/apps/`
(no stereo_visual). None of the C++ or Python apps computes ATE/ARE/NEES/energy/
reprojection metrics or writes anything beyond TUM trajectories. **All paper
metrics require new evaluation tooling** (added under `reproduction/`, read-only
w.r.t. solver state; the shipped apps and library are not modified).

## 6. Configuration

`examples/config/experiment_params.yaml` — per-trajectory `qc_diag` (6-vector)
and `num_iterations`:
- `helix`: qc_diag `[0.00514381, 0.00430491, 0.00550036, 0, 0, 0.000175324]`, 100 iters
- `sphere`: qc_diag `[0.0377749, 0.0255242, 0.0204898, 1.1784e-4×3]`, 100 iters
- `printing_room`: qc_diag `[1.22596, 3.14923, 2.49061, 0.539503, 1.88673, 0.0562069]`, 50 iters
- `kitti_00..kitti_10`: per-sequence qc_diag, 10 iters. Paper uses **kitti_05**
  hyperparameters for the KITTI 06 experiments (rolling shutter and multi-camera),
  matching `kitti_05: [0.732725, 0.0825203, 1.1062, 0.0456018, 0.0135428, 0.0337611]`.
- **No `torus` entry** (referenced by paper Figs. 1 and 10) — see §10.

`add_gp_prior_factor_node` clamps `Qc_diag` entries to a minimum of `1e-4`
(`cwiseMax(1e-4)`); `GPInterp*` adds `+1e-4` to entries `< 1e-4`. Note these two
floors differ slightly (max vs add) — recorded as-is.

## 7. Datasets

### Shipped
- `examples/data/datasets/charuco/printing_room_0.g2o` (94 poses, 617
  `EDGE_POSE_LANDMARK_SE3`, ~1.5 s spacing); `_measurements.g2o`, `_poses.g2o`
  variants. **No camera intrinsics, no pixel detections, no marker-size metadata**
  → the pixel-domain "mean reprojection error" of Fig. 11 is NOT computable from
  shipped artefacts (see §10).
- `examples/data/datasets/stereo_visual/gt/00..10.tum` — KITTI GT (06: 1101 poses).
- `examples/data/datasets/stereo_visual/multirobot/00..10.g2o` — ORB-SLAM3 stereo
  keyframe graphs (06: 477 keyframe poses, 45k landmarks, 158k stereo edges;
  `CAMERA_PARAMS 707.0911865 707.0911865 601.8873291 183.1103973 379.8145142`).
  `06_1.g2o` is an additional variant of 06.
- `examples/data/datasets/stereo_visual/rolling_shutter/kitti_06_{0,0.0001,0.001,0.01,0.1}.g2o`
  — readout-time datasets used by Table II / Figs. 13, 19, 20.
  `kitti_06_0.g2o` is byte-identical to `multirobot/06.g2o`. Non-zero readout
  files contain multiple pose blocks per keyframe (e.g. 2379 poses for 1 ms) with
  per-block timestamps offset by readout/4 within a frame; all blocks of a frame
  are initialised to the same ORB-SLAM3 pose. Timestamps within a frame appear in
  DESCENDING order in the file; `FactorGraph::getPoseVariableNodes()` sorts by
  timestamp, so GP chain construction and interpolation are unaffected.
- **KITTI 06 camera images are NOT shipped** (needed as background of Figs. 13,
  19, 20) and are not freely redistributable → image backdrops are blocked;
  projected-point overlays can still be produced.

### Generated (`examples/data/generators/generate_all_datasets.sh`)
- GT: `helix|sphere` × `1hz,100hz` TUM (600 knots at 1 Hz; 59901 samples at 100 Hz).
- `prior/trajectories_1hz/noise_ig{IG}_n{N}_{shape}.g2o` and
  `pgo/trajectories_1hz/...`: full 7×7 grid over translation noise levels
  `{0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1.5}` (98 files each).
- **Noise convention (authoritative, from `generate_configs.py`):** each level is
  a (translation σ [m], rotation σ [rad]) PAIR: `(0,0), (1e-4,1e-5), (1e-3,1e-4),
  (1e-2,1e-3), (1e-1,1e-2), (1.0,1e-1), (1.5,0.15)` — i.e. rotation σ =
  translation σ / 10. The paper text/captions say noise is applied to "both
  positions [m] and orientations [rad]" with a single symbol; the generator is
  the code-level ground truth and is what we use. Recorded as a paper↔code
  ambiguity.
- σ grid for Figs. 8/9: nonzero levels `{1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1.5}` —
  σ=1.5 EXISTS in the generator (task expectation confirmed); σ=0.0 also exists
  but the paper plots log-σ so 0 cannot appear.
- η_ig grid in Figs. 8/9 panels: `{1e-4, 1e-3, 1e-2, 1e-1}`; η_ig = 1.0 used by
  Figs. 6, 7, 15–18; η_ig ∈ {0, 1.5} generated but unused by any paper figure.
- Helix params: path_length 200, radius 10, height 10, 3 circles × 200 steps
  (600 knots, 1 Hz). Sphere: 600 Fibonacci-spiral points, radius 10, z∈[-9.5,9.5].
- PGO datasets: 599 consecutive odometry edges + 300 random loop closures
  (50% interconnection percentage).

### Random number generators / seeds
- `data_generator`: `std::mt19937 gen(config["random_seed"])`; all configs are
  generated with **seed 42** (`generate_configs.py` default). One realisation per
  noise config; no repetition seeds anywhere in the repo → paper points are
  single-realisation (repetitions = 1) unless the authors used private scripts.
- `charuco_experiment_ba::perturbPoses`: `std::default_random_engine gen(42)`,
  translation σ = `noise_level`, rotation σ = `noise_level × 0.1`; also inflates
  the pose covariance by `I·(1/noise_level)`.
- Loop-closure pair selection: same `mt19937` stream as measurement noise.
- No other RNG use in library or apps.

## 8. Solver internals relevant to reproduction

- Synchronous schedule (`GbpSolver::performIteration`): (1) all factors
  `updateFactor` (relinearise at incoming variable-to-factor message means, robust
  kernel DISABLED — `k=1`, Huber code commented out), (2) all factor-to-variable
  messages (Schur complement, recipient excluded from conditioning), (3) all
  variable `updateMuAndSigma` (belief = sum of incoming messages + 1e-6·I
  regulariser, i.e. no persistent initial prior), (4) all variable-to-factor
  messages (cavity distribution; variables with a single edge never send).
  All four phases OpenMP-parallel; numerically deterministic w.r.t. thread count.
- Factor energies (Eq. 31: E = ½‖e‖²_Σ⁻¹) with error definitions:
  - `SE3PoseVelPrior` (6-dim): e = LogSE3(z⁻¹·x); Σ = meas cov + 1e-6·I.
  - `SE3PoseVelPoseVel` (6-dim): e = LogSE3(z⁻¹·(x_f⁻¹·x_t)); Σ = cov + 1e-6·I.
  - `SE3PoseVelGP` (12-dim, Eq. 4): e = [LogSE3(Exp(Δt·v_f)⁻¹·x_f⁻¹·x_t);
    Adj_R·v_t − v_f]; Σ = Q_i (Eq. 10) built from clamped Qc.
  - `SE3PoseVelPose` (6-dim, charuco): pose-landmark relative SE3.
  - `SE3PoseVelStereoPoint` (3-dim): e = z − [u_l, v, disparity]; behind-camera
    or negative-disparity projections zero the error/Jacobian (soft gating);
    Σ = information⁻¹ (isotropic per-edge from g2o).
  - Total energy (Eq. 18) = Σ measurement-factor energies + Σ GP-prior energies.
- LogSE3/tangent ordering: [translation(3); rotation(3)].
- Initial velocities: finite differences of consecutive initial poses
  (`add_pose_variable_nodes`); initial belief covariance 3·I (12×12).
- GP interpolation (`gsolver/maths/gp_interpolation.h`): `GPInterp(graph, Qc,
  hertz=100)` → per-interval `hertz` samples ⇒ exactly the paper's "100× finer
  than control-point spacing"; also `GPInterpAtTimestamps` for arbitrary stamps
  (used by `stereo_visual` at GT stamps). Returns pose, velocity and 3×3
  position covariance (position block of the 12×12 interpolated covariance).

## 9. Timers / benchmark infrastructure

Shipped apps time ONLY the whole optimisation loop with
`std::chrono::high_resolution_clock` and print milliseconds. There is NO
microbenchmark harness for Fig. 12 (message/interpolation latencies) in the repo
→ the local fallback protocol from the reproduction task specification applies
(Release build, ≥10 warm-up, ≥1000 samples, monotonic clock, raw samples saved).

## 10. Gaps blocking exact reproduction (summary)

| Gap | Affected artefacts | Consequence |
|---|---|---|
| No torus trajectory generator, no torus dataset, no torus qc_diag | Fig. 1, Fig. 10 | original experiments BLOCKED; clearly-labelled substitute (own torus generator + `gp_hyperparam_trainer`) |
| No Hyperion B-/Z-Spline code or converted datasets; upstream Hyperion (Hug et al., ECCV 2024) has its own format; no conversion scripts | Figs. 7, 8, 9, 11, 12, 13, 15–20, Table II (baseline rows) | baselines best-effort; paper curves preserved via printed values (Table II) or digitisation |
| No GN+GP baseline implementation | Fig. 5 (GN+GP curve) | MISSING_BASELINE; independent re-implementation provided as separately-labelled supplementary |
| No camera intrinsics / pixel detections / marker geometry for ChArUco | Fig. 11 y-values (pixel MRE) | exact metric NOT computable; substitute metric (marker-relative pose error) clearly labelled |
| No KITTI 06 images in repo (licensing) | Figs. 13, 19, 20 backdrops | overlays produced on blank canvas + saved pixel coordinates; original backdrop BLOCKED unless user supplies KITTI images |
| No NEES tooling, no η_ig documented for Table I | Table I | NEES convention re-derived (documented); η_ig inferred by matching printed values across the generated grid |
| Rerun SDK unpinned ("latest") | build reproducibility | version recorded (0.35.0) |
| No plotting scripts | all figures | new plotting code under `reproduction/scripts/` |

## 11. Paper-figure → code mapping (details in `reproduction/figure_manifest.yaml`)

| Artefact | Experiment code path | Dataset |
|---|---|---|
| Fig. 1 | prior-type solve + `GPInterp` covariance (torus) | MISSING (substitute) |
| Fig. 5 | `pgo_experiment` graph, energy Eq. 18 per iteration | `pgo/.../noise_ig1.0_n0.1_helix.g2o` |
| Fig. 6 | `prior_experiment` (with GP) + prior-only variant + initial guess | `prior/.../noise_ig1.0_n1.0_helix.g2o` |
| Fig. 7 | `prior_experiment` + `GPInterp` ×100 | `prior/.../noise_ig1.0_n1.0_sphere.g2o` |
| Fig. 8 | `prior_experiment` sweep + ATE | `prior/.../noise_ig{η}_n{σ}_{shape}.g2o`, η∈{1e-4..1e-1}, σ∈{1e-4..1.5} |
| Fig. 9 | `pgo_experiment` sweep + ATE | `pgo/...` same grid |
| Tab. I | `prior_experiment` + NEES from belief Σ | `prior/...`, σ∈{1e-4..1.0}, η_ig inferred |
| Fig. 10 | `hyperparam_ablation_experiment` logic (scales from code) | torus MISSING → substitute |
| Fig. 11 | `charuco_experiment_ba` (σ=0.1) per-iteration metric | `charuco/printing_room_0.g2o` |
| Fig. 12 | (new microbenchmark, fallback protocol) | helix prior graph (choice documented) |
| Fig. 13/19/20 | `stereo_visual` on RS dataset + projection dump | `rolling_shutter/kitti_06_0.001.g2o` |
| Tab. II | `stereo_visual` + ATE/ARE at GT stamps + in-image MRE | all 5 RS files, config `kitti_05` |
| Fig. 14 | `stereo_visual` num_robots ∈ {1,2,4,8} + time/RSS/ATE | `multirobot/06.g2o` |
| Fig. 15/16 | Fig. 8 runs at η_ig=1 + `GPInterp` ×100 | `prior/.../noise_ig1.0_n{σ}_{shape}.g2o` |
| Fig. 17/18 | Fig. 9 runs at η_ig=1 + `GPInterp` ×100 | `pgo/...` same |

## 12. Figure assets in repo

`assets/teaser.png` (157 kB) — README teaser, stated to be "Figure 1 from the
paper" (continuous-time mean + posterior covariance on the torus). This is the
only paper-figure asset in the repository; it is a rendered image, not data.
