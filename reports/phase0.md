# Phase 0 report: environment check

Branch `certify` (created from `paper-reproduction` @ `b330450`). Date 2026-09-24. Host `AMD EPYC 7713`, RHEL 9.7.

## 1. Five-line summary

1. gsolver builds from an empty build directory on this host: C++17, CMake 3.26.5 + GNU Make, GCC 11.5.0. The build is 13.8 s configure + 130.0 s compile (`-j48`), with 0 warnings. That covers the library, all 8 example executables and the Python bindings. It needs two host-specific configure flags and no source changes.
2. Every dataset the brief names is in the repository, git-tracked. That is 134.7 kB of ChArUco and 15.3 MB of KITTI 06 stereo, plus 4 × 14.2 MB rolling-shutter files. KITTI ground truth exists only at the 1,101 frame times. ChArUco has no ground truth and no camera intrinsics.
3. All five experiments ran end to end at default settings (exit 0). KITTI 06 stereo BA gives ATE 0.668 m / ARE 0.0074 rad (paper Table II: 0.671 / 0.007). 8-camera BA gives 0.667 m. Rolling-shutter readout 0.1 s gives 0.662 m (paper 0.726). Readouts 1e-2, 1e-3 and 1e-4 s **diverge** (5.39 m, 21.0 m, 76,139 m against the paper's ≤ 0.684 m). This is the same known discrepancy the July reproduction found, and it is bit-identical here.
4. Performance: shipped GBP on KITTI 06 runs on about one core (11.1 s/iteration at any thread count; the likely cause is diagnosed in §3.4). On this shared machine the default 64 OpenMP threads make ChArUco 47× slower than 16 threads. Results are byte-identical across thread counts. ⚑ With plain patch LU, the full KITTI 06 Phase C sweep is estimated at 6–11 h, and a K = 3 certified Phase D run at 15–20 h (§3.5).
5. **Gate 0: PASS.** The data also changes three assumptions in the brief that later phases depend on. ChArUco has 24 SE(3) marker landmarks (23 are hubs at w = 3), not one board. KITTI 06 has 20,436 hub landmarks at w = 3 (loop re-observations), not an empty hub set. The stereo experiments add no gauge anchor. Details are in §5.

## 2. Gate status

| Gate 0 criterion | Result | Evidence |
|---|---|---|
| Repository builds | **PASS** | fresh configure + build of lib, examples and Python bindings, exit 0, 0 warnings (§3.1) |
| Datasets available | **PASS** | all files present in the repo (§3.2); network also available (gitlab, github, pypi, cvlibs all reachable) |
| ChArUco incremental SLAM runs end to end | **PASS** | exit 0; 118 variables, 711 factors; 0.116 s at 16 threads, 5.41 s at the default 64 threads |
| Stereo BA runs end to end | **PASS** | KITTI 06, exit 0, ATE 0.668 m / ARE 0.0074 rad, 125 s wall, 1,252 MiB peak RSS |

## 3. Tables

### 3.1 Build

| Item | Value |
|---|---|
| Language | C++17 (library, examples); Python bindings via nanobind v2.7.0; analysis in Python 3.9.25 |
| Build system | CMake 3.26.5 + GNU Make 4.3 (no Ninja on host) |
| Compiler | GCC 11.5.0 (Red Hat 11.5.0-11), Release: `-O3 -DNDEBUG`, OpenMP 4.5 (`-fopenmp`, libgomp) |
| Configure time | 13.8 s (includes downloading Eigen, Rerun SDK, nanobind) |
| Build time | 130.0 s wall at `-j48` (1,168 s user CPU, 124 s sys, peak RSS 1.5 GB). Most of it is Apache Arrow, which the Rerun SDK builds from source |
| Products | `libgsolver.a`; `charuco_experiment_{ba,islam}`, `stereo_visual`, `pgo_experiment`, `prior_experiment`, `hyperparam_ablation_experiment`, `gp_hyperparam_trainer`, `data_generator`; `_gsolver.cpython-39-x86_64-linux-gnu.so` |
| Warnings / errors | 0 / 0 in the final build (CMake prints one developer warning, CMP0135, from FetchContent) |
| Python bindings check | README minimal example gives `x1` translation `[1, 0, 0]`, as documented |
| Recipe | `scripts/build_gsolver.sh <build_dir>` |

Dependencies:

| Dependency | Source | Status on this host |
|---|---|---|
| Eigen 3.4.0 | FetchContent, gitlab tarball, MD5-pinned | downloaded automatically |
| Rerun C++ SDK | FetchContent, **unpinned "latest"** | resolved to **0.38.1** today (the July build used 0.35.0). Only used by `-visualize`, never on the numerical path: outputs are bit-identical to July's |
| nanobind v2.7.0 | FetchContent (git) | downloaded automatically |
| OpenMP | GCC libgomp | system |
| yaml-cpp 0.8.0 | **not installed system-wide, and no root** | built from source (static, PIC) into `/home/os717/gsolver-deps/install` by the July session and reused, not rebuilt. It needs `-isystem <prefix>/include` and `-L<prefix>/lib64` for executables **and** for the Python shared module |
| Python 3.9 headers | system (`/usr/include/python3.9`) | present; must be pinned (see below) |
| numpy 2.0.2 (OpenBLAS 0.3.27), scipy 1.13.1, pandas 2.3.3, matplotlib 3.9.4, pytest 8.4.2 | system Python | present. scikit-sparse (CHOLMOD) is **not** installed; pip can fetch it if Phase C needs it |

Two build failures before the successful configuration, both host configuration issues with no source change:

1. Configure (verbatim): `Could NOT find Python (missing: Python_INCLUDE_DIRS Python_LIBRARIES Development Development.Module Development.Embed) (found suitable version "3.12.12", minimum required is "3.8")`. CMake picks `/usr/bin/python3.12`, which has no headers. Fix: `-DPython_EXECUTABLE=/usr/bin/python3` (3.9, has headers).
2. Link of the Python module (verbatim): `/usr/bin/ld: cannot find -lyaml-cpp` / `collect2: error: ld returned 1 exit status` / `gmake[2]: *** [python/CMakeFiles/_gsolver.dir/build.make:345: python/_gsolver.cpython-39-x86_64-linux-gnu.so] Error 1`. The `-L` path was given for executables only. Fix: also pass `-DCMAKE_MODULE_LINKER_FLAGS` and `-DCMAKE_SHARED_LINKER_FLAGS`.

### 3.2 Datasets (all present; paths relative to `examples/data/datasets/`)

| Experiment | File(s) | Size | Contents |
|---|---|---|---|
| ChArUco incremental SLAM, ChArUco BA | `charuco/printing_room_0.g2o` (also `_poses`, `_measurements` splits) | 134.7 kB | 94 `VERTEX_SE3` (UcoSLAM poses, t = 1.50–35.97 s, dt 0.017–2.1 s) and 617 `EDGE_POSE_LANDMARK_SE3` to **24 SE(3) markers**. No ground truth, no intrinsics, no pixel detections |
| Stereo BA (KITTI 06) | `stereo_visual/multirobot/06.g2o`, `stereo_visual/gt/06.tum` | 15.3 MB, 86 kB | 477 keyframes, 46,196 `LANDMARK_3D` (45,555 observed), 157,941 `EDGE_STEREO`, `CAMERA_PARAMS`; GT has 1,101 poses at frame times (0–114.29 s) |
| Multi-camera | same `06.g2o`, split into cameras at run time | — | 8 cameras: 59–64 poses each, 11,784 landmark-equality factors |
| Rolling shutter | `stereo_visual/rolling_shutter/kitti_06_{0,0.0001,0.001,0.01,0.1}.g2o` | 15.3 MB, 4 × 14.2 MB | 2,370–2,379 row-block poses (about 5 per keyframe, spaced readout/4); `kitti_06_0.g2o` is byte-identical to `multirobot/06.g2o`. GT only at frame times, not row times |
| Other (not needed) | `stereo_visual/multirobot/{00..10}.g2o` + GT; `synthetic/` (generated in July, gitignored) | 276 MB; 21 MB | — |

Nothing needed to be fetched. KITTI images are not present and not needed. The ChArUco intrinsics and pixel detections that the paper's Fig. 11 metric needs are not public.

### 3.3 Experiment runs (default settings; wall time and peak RSS from `/usr/bin/time -v`)

The drivers print no accuracy metric. ATE/ARE below come from `certify/tools/traj_eval.py`, following the paper's protocol: estimate interpolated at the GT timestamps, RMSE, and SE(3) Umeyama alignment without scale. It reproduces the July reproduction's C++ evaluator exactly (0.667959 m / 0.007374 rad on readout 0).

| Run | Settings | Wall | Peak RSS | Graph build / GBP | Printed output | ATE / ARE aligned (raw) | Paper |
|---|---|---|---|---|---|---|---|
| ChArUco islam | defaults, 64 threads | 5.42 s | 12.5 MiB | incremental, 5,407 ms | 118 variables, 711 factors | no GT | not in paper |
| ChArUco islam | `OMP_NUM_THREADS=16` | 0.12 s | 14.5 MiB | 116 ms | same; output byte-identical | no GT | — |
| ChArUco BA | noise 0.0 (README default), 64 threads | 2.66 s | 15.0 MiB | 50 it, 2,641 ms | 118 variables, 711 factors | no GT | Fig. 11 pixel MRE, not computable |
| ChArUco BA | noise 0.1 (paper setting) | 2.66 s | 15.0 MiB | 50 it, 2,639 ms | same | no GT | Fig. 11, not computable |
| Stereo BA KITTI 06 | `kitti_05`, 1 camera, 64 threads | 125.2 s | 1,252 MiB | 13.2 s / 10 it in 111.3 s | 477 poses, 45,555 landmarks, 157,941 obs | **0.668 m / 0.0074 rad** (1.301 / 0.0136) | 0.671 / 0.007 |
| Stereo BA KITTI 06 | `OMP_NUM_THREADS=16` | 124.2 s | 1,252 MiB | 13.2 s / 110.3 s | output byte-identical | 0.668 / 0.0074 | 0.671 / 0.007 |
| Multi-camera | 8 cameras (README default) | 118.0 s | 1,235 MiB (154 MiB/camera) | 4.5 s / 10 it in 112.8 s (14.1 s per camera) | 11,784 equality factors | **0.667 / 0.0075** (1.308 / 0.0135) | Fig. 14 (digitised in July, approximate): ATE ≈ 0.70, ≈ 11.5 s and ≈ 85.5 MB per camera |
| Rolling shutter | readout 0.1 s (README default) | 29.0 s | 1,283.0 MiB | 15.9 s / 12.4 s | 2,370 poses | **0.662 / 0.0081** (1.307 / 0.0140) | 0.726 / 0.008 |
| Rolling shutter | readout 0.01 s | 28.5 s | 1,287.5 MiB | 15.8 s / 12.0 s | 2,379 poses | **5.388 / 0.152, diverged** | 0.684 / 0.007 |
| Rolling shutter | readout 0.001 s | 27.4 s | 1,288.5 MiB | 14.5 s / 12.1 s | 2,379 poses | **21.01 / 0.313, diverged** | 0.676 / 0.008 |
| Rolling shutter | readout 0.0001 s | 28.4 s | 1,288.0 MiB | 15.6 s / 12.1 s | 2,379 poses | **76,139 / 1.95, diverged** | 0.676 / 0.007 |

All seven KITTI trajectories are **byte-identical** to the July reproduction's outputs (`reproduction/results/local_gsolver/{table_02,figure_14}/*/est_interp_gt_stamps.tum`), although the build uses Rerun 0.38.1 instead of 0.35.0: the solver is deterministic and Rerun is not on the numerical path. The ChArUco outputs are finite and lie 3.2–3.4 cm (aligned RMS) from the UcoSLAM front-end poses. That is a sanity check that nothing diverged, **not** an accuracy measurement.

### 3.4 Hardware and host conditions

| Item | Value |
|---|---|
| CPU | AMD EPYC 7713, 1 socket, 64 cores (SMT off), 1.5–3.72 GHz, governor `performance`, 256 MB L3 |
| RAM | 250 GiB total; about 109 GiB available during the runs (other users hold about 141 GiB) |
| GPU | 4 × NVIDIA RTX A6000 (48 GB each), driver 595.71.05, CUDA 13.2. The brief does not use them |
| Disk | `/home` (ZFS) 2.6 TB, **174 GB free (94 % used)**; `/` 41 GB free |
| Sharing | Shared host. 12–13 cores are continuously busy with other users' multi-week jobs (load average 13–17 during these runs), so wall times are noisy |

Thread-count behaviour, measured:

- **Default 64 threads oversubscribe the machine.** ChArUco incremental SLAM runs 47× slower at 64 threads than at 16 (5.41 s vs 0.116 s), with busy-waiting threads at 4,852 % CPU. Later phases will set `OMP_NUM_THREADS` explicitly (≤ 16).
- **KITTI 06 GBP runs on about one core at any thread count.** It takes 11.1 s/iteration at both 64 and 16 threads, with 115–140 % CPU over the whole run. The likely cause, diagnosed from evidence rather than profiled, has two parts. First, `gbp_solver.cpp:25-35` runs `#pragma omp parallel for` over `variable_nodes_` with the default static schedule, and the 477 poses are the first 477 of 46,032 variables, so all of them fall in the first thread's chunk. Second, each pose's outgoing messages cost O(degree²) (`variable_node.cpp:30-50`), with a mean stereo degree of 331 (max 656, Σdeg² = 5.6e7). The rolling-shutter graph (2,370 poses, mean degree 66, Σdeg² = 1.5e7, poses spread over about 4 chunks) runs 9× faster per iteration (1.24 s). This is shipped solver behaviour; Phases A–C will not change it.

### 3.5 Estimated cost of later phases (flags marked ⚑)

The GBP figures below are the run times measured above. The Python figures come from `scripts/phase0_cost_probe.py`. It times the oracle and patch computations on a proxy with KITTI 06's exact factor graph but random Jacobians, because the real `H` only exists after Phase A. Its output is `results/phase0/cost_probe.txt`.

Probe results (8 BLAS threads, plain `scipy.sparse.linalg.splu` on each patch, 20 random blocks per row):

| Quantity (KITTI 06 proxy: 142,389 dof, nnz(H) = 11.9 M, 142 MB CSR) | Time |
|---|---|
| Assemble H in Python | 26.8 s |
| Oracle solve (eliminate landmarks, dense 5,724-dof reduced camera system) | 11.4 s (relative residual 8e-12) |
| Oracle marginals Σ_ii for all 46,032 blocks | 51.2 s |
| Pose patch solve, K = 1 / 2 / 3 (median patch 990 / 3,603 / 41,755 dof) | 7.3 ms / 136 ms / **20.0 s** median (max 11 ms / 486 ms / 76.6 s) |
| Landmark patch solve, K = 1 / 2 / 3 (median patch 33 / 1,447 / 3,994 dof) | 0.6 ms / 12.8 ms / 199 ms median (max 1.1 ms / 318 ms / 1.7 s) |

Estimates. One "export" is one (snapshot, iteration) state. Phase A asks for 8 iterations per run, and ChArUco has 3 snapshots.

| Phase / task | Basis | Estimate | Flag |
|---|---|---|---|
| A: ChArUco frozen and unfrozen exports | GBP about 1.1 ms/iteration at 16 threads | under a minute of compute | — |
| A: KITTI 06 frozen run to convergence | 11.1 s/iteration, single-core bound (§3.4) | 18.5 min per 100 iterations. The number needed to reach the 1e-6 D-metric Gate A tolerance is unknown; at 1,000 iterations it is 3.1 h per run | ⚑ if more than about 1,000 iterations are needed |
| A: KITTI 06 unfrozen run to 100 iterations | 11.1 s/iteration | about 20 min | — |
| A: export volume | nnz(H) about 12 M plus per-factor Jacobians | about 0.2 GB per KITTI export, about 3–4 GB for 16 exports. Disk is fine (174 GB free), but exports stay out of git | — |
| C: ChArUco, all certificates and bounds | 1,272 dof, dense linear algebra | minutes in total | — |
| C: KITTI 06 oracle per export | probe | about 1 min (solve plus all Σ_ii) | — |
| C: KITTI 06 C3, all 46,032 blocks, K = 1, 2, 3, plain patch LU | probe: 477 × 20.1 s + 45,555 × 0.21 s | **about 5.4 CPU-h per export**; the cut upper bound needs a second factorization per patch (about 2×). 8 frozen exports come to about 90 CPU-h, **about 6–11 h wall** with 8–16 worker processes (worker count is limited by memory: the probe process peaked at 4.6 GB, and about 109 GiB is free) | ⚑ |
| C: same, with landmarks eliminated inside each patch (no theory change; the patch solve is exact either way) | the K = 3 pose patch reduces to about 100–400 poses of dense solve | not measured; expected to be 10–50× cheaper, to be measured in Phase C | — |
| D: ChArUco, 4 schedules × 4 truncations | GBP plus Python sidecar at about 0.1 s/iteration | under 1 h | — |
| D: KITTI 06 baselines with per-iteration oracle errors | 11.1 s GBP + about 30 s export/assembly + 11 s oracle per iteration | about 45 min per 50-iteration schedule | — |
| D: KITTI 06 certified schedule (sidecar) | as above plus certificates for all blocks every iteration | K = 1: about 1 h per 50 iterations; K = 2: about 1.5–2 h; **K = 3: about 15–20 h** with plain patch LU | ⚑ at K = 3 |
| E: rolling shutter 0.1 s (2,370 poses) | GBP 1.24 s/iteration; one export plus tubes at 1,101 GT times | about 1 h | — |

Unflagged items fit in an hour or two. The three flags all come from KITTI 06: GBP there runs on one core (11 s/iteration), and plain K = 3 patch solves are large (about 42k dof).

## 4. Worst cases

| Category | Worst case | Numbers |
|---|---|---|
| Accuracy against the paper | rolling shutter, readout 1e-4 s | ATE 76,139 m / ARE 1.95 rad vs paper 0.676 / 0.007 (diverged; also 1e-3 at 21.0 m and 1e-2 at 5.39 m) |
| Largest gap among non-diverged runs | rolling shutter, readout 0.1 s | ATE 0.662 m vs paper 0.726 m (local is 0.064 m *better*) |
| Wall time | KITTI 06 stereo BA, 1 camera | 125 s wall; 11.1 s per GBP iteration on about 1 core |
| Thread pathology | ChArUco islam at the default 64 threads | 5.41 s vs 0.116 s at 16 threads (47×) |
| Peak memory | rolling shutter, readout 0.001 s | 1,288.5 MiB |
| Build | Python module link | `cannot find -lyaml-cpp`; fixed by configure flags only |

## 5. Facts and code references

The four facts (F1–F4) belong to Phase A. The observations below came up while running Phase 0. They change assumptions in the brief, so they are recorded now for the human's Gate 0 decision. Phase A will verify them.

1. **ChArUco has 24 separate SE(3) marker landmarks, not one board.** `add_landmark_factor` in `examples/cpp/common/factor_graph_builder.h:271-290` creates one 6-dof `SE3Pose` per marker ID and one `SE3PoseVelPose` factor per detection, using a fixed covariance 3·I that ignores the file's covariance. At w = 3, **23 of the 24 markers have track spans longer than 3 poses**, so |S| = 23, not 1. Only 7 markers (2, 7, 9, 12, 37, 39, 40) are seen from the anchored first pose. Brief §2.4 says "On ChArUco, `S` is the board", and brief §6 assumes the board is observed from the anchor frame. With 23 hubs, the anchored-subgraph condition (Lemma 4.3 / P5) needs the joint or one-at-a-time treatment that proof note §4.5 describes.
2. **KITTI 06 stereo BA has a large hub set.** Track spans are bimodal: median 3 poses, 90th percentile 394, maximum 455. 9,683 landmarks are re-observed after a gap of more than 10 poses, because the sequence revisits the same road. Hubs at w = 3 / 5 / 10: **20,436 / 15,649 / 11,886 of 45,555** landmarks. Brief §2.4 expects "On stereo BA with short tracks, `S` may be empty".
3. **The stereo experiments add no gauge anchor.** `stereo_visual.cpp:157-182` adds only pose variables, GP priors, landmarks and stereo factors. The parsed `EDGE_PRIOR_SE3` priors (`all_priors`, line 130) are never used, and `06.g2o` contains none. The stereo and GP errors are invariant under a global SE(3) left-multiplication. With one consistent linearization point and exact Jacobians, Σ JᵀΛJ would therefore have a 6-dimensional null space. In G-solver, linearization points differ per factor (item 5) and the stereo rotation Jacobian is scaled (item 6), so H may be nonsingular but is expected to be severely ill-conditioned in those directions. What breaks the symmetry inside GBP is (a) the 1e-6·I terms added in every variable update and message (`variable_node.cpp:52,111,159`, `factor_node_helper.h:22,69`), and (b) a side effect: a variable with a single edge never sends an updated message (`variable_node.cpp:26-28`). Its variable-to-factor message stays at its creation value, which is its initial mean and initial precision (`factor_graph_helper.h:37-38,75-76`), so it acts as a fixed prior at the initial ORB-SLAM3 position. That prior enters its factor's messages but not the landmark's own belief. KITTI 06 has 4,928 single-observation landmarks. Phase A (F2) must decide what the gauge term G is for stereo BA and measure λ_min(H); (A0) cannot be assumed for H = Σ JᵀΛJ alone.
4. **Anchor stiffness on ChArUco.** The incremental SLAM anchor has covariance 1e-12·I (`charuco_experiment_islam.cpp:155-156`, precision 1e12) and BA uses 1e-6·I (`factor_graph_builder.h:323-331`). H will be badly conditioned, which matters for Gate A's 1e-6 D-metric tolerance.
5. **Relinearization happens per factor, every iteration**, at the means of the incoming variable-to-factor messages, not at the belief means (e.g. `se3_posevel_gp.cpp:14-17`, `se3_posevel_stereo_point.cpp:15-16`). Messages are manifold points re-centred by `boxplus` (`factor_node_helper.h:72-73`). Precisions are not transported between charts (`variable_node.cpp:42-49`). This is the chart-consistency issue the brief warns about; freezing relinearization for Phase A needs instrumentation that fixes every factor's linearization point.
6. **Stereo Jacobian.** `se3_posevel_stereo_point.cpp:77` uses `2.0 * P_ext * skew(p_cam)` for the rotation block, while the derivative of `p_cam` under the right perturbation `T·Exp(δ)` of `geometry3d.h:126-133` is `skew(p_cam)`. This does not affect certificate validity, because the "current model" is whatever G-solver assembles, but GBP's fixed point is then not the MAP of the stated measurement model. The robust kernel is disabled (`factor_node.cpp:20-21`, `k = 1`).

## 6. What could not be done and why

- **ChArUco accuracy against the paper.** The repo has no ChArUco ground truth, and the paper's metric (Fig. 11, mean pixel reprojection error) needs camera intrinsics and pixel detections that are not public; the g2o file stores pose-to-marker SE(3) transforms only. The only check possible is the sanity comparison against the front-end poses in §3.3. ChArUco incremental SLAM is not a paper experiment, so there is no paper value.
- **Rolling-shutter readouts 1e-4, 1e-3 and 1e-2 s do not reproduce the paper.** They diverge with the public code and config. The cause is unknown. The July session's untested leads are the disabled robust kernel and `kitti_05` vs `kitti_06` hyperparameters. For Phase E, only readouts 0 and 0.1 s currently give sane trajectories.
- **Table II's reprojection-error column was not recomputed.** The drivers do not write landmark estimates.
- **Multi-camera runs with 1, 2 and 4 cameras were not repeated.** The README default is 8; the July session has the others. Paper Fig. 14 values are digitised approximations, and its per-camera time and memory definitions are not stated.
- **Wall times are not comparable to the paper** (16-core i9 vs this loaded, shared 64-core EPYC).
- **No dependency was newly installed.** yaml-cpp was reused from July's local source build; its recipe was not re-run.

## 7. File list

- `AGENT_BRIEF.md`, `PROOF_NOTE.md`: the supplied brief and proof note, added to the repo root verbatim.
- `scripts/build_gsolver.sh`: build recipe with the two host flags.
- `scripts/phase0_run_experiments.sh`: runs every experiment with `/usr/bin/time -v`.
- `scripts/phase0_summarize.py`: produces `results/phase0/summary.csv`.
- `scripts/phase0_cost_probe.py`: Python cost probe for the estimates in §3.5.
- `certify/tools/traj_eval.py`: ATE/ARE evaluator, also used later in Phase D.
- `results/phase0/`: per-run `*.stdout`, `*.stderr_time` and `*.tum`; `summary.csv`; `ate_are.txt`; `run_all.log`; `cost_probe.txt` and `cost_probe.stderr_time`; `charuco_ucoslam_frontend.tum`; `build_logs/` (both failed attempts verbatim, plus the final configure and build logs).
- Commit: see `git log` on branch `certify` (the commit that adds this file).
