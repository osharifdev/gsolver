# Phase D, WP1b: threshold relinearization at equal work

Branch `certify`, base commit `f6a23d2`. Date 2026-09-27. This report follows the WP1 amendment of 2026-09-27, which acts as `BRIEF_phaseD.md`'s WP1 amendment:
- δ and d_i are measured in the physical metric W = I (m, rad, m/s, rad/s);
- the grid is δ ∈ {0.003, 0.01, 0.03, 0.1}, R ∈ {1, 3, 10};
- the gate compares at equal work;
- there is a KITTI-40 fixed-point diagnostic.

The WP1 front-end fix (the islam front end reads beliefs, not linearization points) is kept.

## 1. Five-line summary

1. **Exactly one setting passes on both datasets at equal work, and it is chosen: δ = 0.003, R = 1.**
   - ChArUco islam: median ×1.032, p95 ×1.024.
   - First 40 KITTI keyframes: median ×0.426, p95 ×0.392.
   - Default path byte-identical: 12 of 12, on the final binaries.
2. **Full prefix at (0.003, 1), 200 rounds in two checkpointed segments.**
   - At W₀, the shipped schedule's work over 100 rounds (reached at round 111): median 0.378 against 0.553 (×0.68), p95 4.10 against 6.84 (×0.60).
   - At 200 rounds, using 13 % less work than the shipped 200 rounds: median 0.338 against 0.553, p95 3.20 against 4.84.
   - It reaches the shipped final median by round 60. Aligned keyframe ATE is unchanged (0.381 m).
3. **L1 diagnostic.** On KITTI-40 the shipped schedule does not drift, as it does on ChArUco. It stalls:
   - its remaining Gauss–Newton move stays at a median of 0.42–0.43 (p95 1.275) from round 150 to 500;
   - δ = 0, R = 1 converges to 0.021 (p95 0.078), about 20× closer to the Gauss–Newton stationary point;
   - ATE does not tell the two fixed points apart: aligned keyframe ATE is 0.0823 against 0.0814 m, raw 0.5129 against 0.5126 m;
   - the shipped schedule linearizes every factor at cavity means that sit a median 0.13 m (landmarks) and 2.6e-4 (poses) from the beliefs, and this gap does not change with rounds.
4. **The savings are bounded by the message share.** Messages are about 80 % of a round's work on all three datasets, so at W₀ a threshold schedule gets at most about 1.26× the rounds.
   - The chosen setting relinearizes 27 % of factors per round up to W₀ on ChArUco and KITTI-40.
   - On the prefix it relinearizes 50 % up to W₀: all factors in the first ~20 rounds, about 20 % after round 90.
5. **Caveats.**
   - The ChArUco pass set is narrow and non-monotone: 4 of 12 pass, including δ = 0.1, R = 10 at ×1.048 but not δ = 0.03, R = 10.
   - Every δ ≥ 0.03 fails on KITTI-40.
   - Round for round, threshold schedules trail the shipped one early and lead later.
   - The rounds-to-reach definition was corrected after the ChArUco runs (§6).

## 2. Gate status

| Criterion (amended WP1) | Result | Status |
|---|---|---|
| Default path byte-identical (`scripts/phaseA_equivalence.sh`) | 12 of 12 identical on the final binaries (`results/phaseD/wp1b/equivalence.txt`) | **pass** |
| At work W₀: median ≤ 1.05× and p95 ≤ 1.5× the default's, on ChArUco and KITTI-40 | passing on ChArUco: (0.003, 1), (0.003, 10), (0.01, 10), (0.1, 10). Passing on KITTI-40: (0.003, 1), (0.003, 3), (0.01, 1), (0.01, 3). **Passing on both: (0.003, 1)** | **pass**, one setting |
| Choice: fewest factor relinearizations per round, ties to the largest δ | (0.003, 1), the only setting passing on both: 26.7 % of factors per round up to W₀, the mean of the two datasets | chosen |
| Full prefix at the chosen setting, 200 rounds, two checkpointed segments | done. The checkpoint gate at round 100, fixed before the run (continue unless the median is non-finite or above 10× the default's 0.553), read 0.377: continue | done |

**Definitions**, fixed in `scripts/phaseD_wp1b.py` before any WP1b run:
- **Final pose error:** unchanged from WP1. ‖x̄_i‖_{D_i} per pose, where x̄ is the exact solution of the linearization at the current beliefs.
- **Work:** cumulative linearizations plus message computations.
  - The shipped schedule re-evaluates every factor each round.
  - A threshold schedule counts newly added factors and the factors its rule relinearizes.
  - Messages are every factor-to-variable message, plus the variable-to-factor messages of variables with at least two edges.
  - Each round is accounted before it runs, so "at W₀" is the state after the last round with cumulative work ≤ W₀.
- **W₀:** the default's work over its budget. ChArUco: the shipped 103 rounds, W₀ = 217,945. KITTI-40: the 100 rounds of WP1 run 2, W₀ = 5,623,300 (§6).

## 3. Tables

### 3.1 ChArUco islam at W₀ = 217,945

The default runs 103 rounds: median 1.423, p95 19.33. The complete graph exists from round 94.

| δ, R | Rounds at W₀ | Median (× default) | p95 (× default) | Gate | Relinearizations per round (share of factors) | Rounds (work) to reach the default median |
|---|---|---|---|---|---|---|
| 0.003, 1 | 110 | 1.468 (1.032) | 19.79 (1.024) | **pass** | 190.7 (26.8 %) | 130 (2.73e5) |
| 0.003, 3 | 112 | 1.553 (1.091) | 21.21 (1.097) | fail | 145.4 (20.4 %) | 140 (2.98e5) |
| 0.003, 10 | 115 | 1.444 (1.015) | 18.96 (0.981) | pass | 74.6 (10.5 %) | 130 (2.61e5) |
| 0.01, 1 | 114 | 1.511 (1.061) | 20.63 (1.067) | fail | 93.7 (13.2 %) | 140 (2.91e5) |
| 0.01, 3 | 115 | 1.676 (1.178) | 22.73 (1.176) | fail | 70.7 (9.9 %) | 160 (3.45e5) |
| 0.01, 10 | 116 | 1.450 (1.019) | 19.02 (0.984) | pass | 39.8 (5.6 %) | 130 (2.56e5) |
| 0.03, 1 | 116 | 2.141 (1.505) | 32.75 (1.694) | fail | 44.4 (6.2 %) | 240 (5.70e5) |
| 0.03, 3 | 116 | 2.026 (1.424) | 30.74 (1.590) | fail | 34.0 (4.8 %) | 210 (4.83e5) |
| 0.03, 10 | 117 | 1.585 (1.114) | 20.61 (1.066) | fail | 20.3 (2.9 %) | 160 (3.39e5) |
| 0.1, 1 | 117 | 2.266 (1.592) | 30.13 (1.558) | fail | 16.0 (2.3 %) | 270 (6.51e5) |
| 0.1, 3 | 117 | 1.506 (1.058) | 23.80 (1.231) | fail | 12.3 (1.7 %) | 150 (3.09e5) |
| 0.1, 10 | 117 | 1.492 (1.048) | 23.84 (1.233) | pass | 11.7 (1.6 %) | 150 (3.09e5) |

Relinearizations include the first linearization of the factors added during the incremental phase, about 7 per round. Every run is capped at 1,030 rounds. At the cap, every setting's median is 0.14–0.42, against 2.45 for the shipped schedule run to 1,093 rounds in WP1.

### 3.2 First 40 KITTI 06 keyframes at W₀ = 5,623,300

The default runs 100 rounds: median 0.4245, p95 1.951. The graph has 40 poses, 4,679 landmarks and 11,526 factors, with 44,707 messages per round.

| δ, R | Rounds at W₀ | Median (× default) | p95 (× default) | Gate | Relinearizations per round (share) | Rounds (work) to reach the default median |
|---|---|---|---|---|---|---|
| 0.003, 1 | 117 | 0.181 (0.426) | 0.764 (0.392) | **pass** | 3,062 (26.6 %) | 40 (2.09e6) |
| 0.003, 3 | 121 | 0.370 (0.871) | 0.628 (0.322) | pass | 1,673 (14.5 %) | 60 (2.87e6) |
| 0.003, 10 | 121 | 0.951 (2.240) | 16.90 (8.663) | fail | 1,407 (12.2 %) | 100 (4.61e6) |
| 0.01, 1 | 120 | 0.413 (0.972) | 1.084 (0.556) | pass | 2,102 (18.2 %) | 120 (5.62e6) |
| 0.01, 3 | 123 | 0.349 (0.821) | 1.059 (0.543) | pass | 976 (8.5 %) | 40 (1.89e6) |
| 0.01, 10 | 123 | 0.684 (1.611) | 4.746 (2.433) | fail | 998 (8.7 %) | not within 300 |
| 0.03, 1 | 121 | 0.619 (1.457) | 1.575 (0.807) | fail | 1,465 (12.7 %) | not within 300 |
| 0.03, 3 | 123 | 0.559 (1.317) | 2.532 (1.298) | fail | 866 (7.5 %) | not within 300 |
| 0.03, 10 | 124 | 0.914 (2.154) | 9.549 (4.895) | fail | 642 (5.6 %) | not within 300 |
| 0.1, 1 | 123 | 0.924 (2.177) | 2.555 (1.310) | fail | 902 (7.8 %) | not within 300 |
| 0.1, 3 | 124 | 1.158 (2.728) | 5.702 (2.923) | fail | 479 (4.2 %) | not within 300 |
| 0.1, 10 | 124 | 1.700 (4.005) | 9.536 (4.888) | fail | 385 (3.3 %) | not within 300 |

**Error against work**, chosen setting against the default. Each cell is the median (p95) at that cumulative work:

| Work | 0.56e6 | 1.1e6 | 1.6–1.7e6 | 2.1–2.3e6 | 2.6–2.8e6 | 4.5–4.8e6 | 5.6e6 |
|---|---|---|---|---|---|---|---|
| default | 1.90 (7.61) | 0.675 (5.16) | 0.547 (4.24) | 0.482 (3.45) | 0.469 (2.84) | 0.428 (2.22) | 0.424 (1.95) |
| (0.003, 1) | 2.12 (14.4) | 0.685 (10.9) | 0.460 (8.13) | 0.285 (6.04) | 0.246 (4.49) | 0.195 (1.09) | 0.181 (0.76) |

### 3.3 Full prefix (161 keyframes, 17,937 landmarks, 50,640 factors, 198,317 messages per round) at (0.003, 1)

The default's values come from Phase C's exports of the same shipped schedule (`exports/kitti06_prefix161/unfrozen_run/unfrozen/it{010,020,050,100}` and `conv`, frozen after 200 rounds), analysed with the same code. A default round costs 248,957 work.

| Round (work) | Default median / p95 / max | (0.003, 1) median / p95 / max at the same round | Aligned keyframe ATE, default / (0.003, 1) (m) |
|---|---|---|---|
| 10 (2.49e6) | 5.55 / 44.6 / 2,784 | 7.71 / 113.5 / 207 | 0.391 / 0.422 |
| 20 (4.98e6) | 2.06 / 8.50 / 13.7 | 3.74 / 33.6 / 59.8 | 0.386 / 0.385 |
| 50 (1.24e7; threshold 1.19e7) | 0.606 / 5.73 / 19.9 | 0.721 / 8.73 / 31.1 | 0.384 / 0.383 |
| 100 (2.49e7; threshold 2.25e7) | 0.553 / 6.84 / 18.2 | 0.377 / 5.21 / 11.0 | 0.381 / 0.384 |
| **W₀ = 2.49e7** (threshold round 111) | 0.553 / 6.84 / 18.2 (round 100) | **0.378 / 4.10 / 14.7** | 0.381 / 0.381 |
| 200 (4.98e7; threshold 4.34e7) | 0.553 / 4.84 / 11.9 | 0.338 / 3.20 / 6.96 | 0.380 / 0.381 |

- Relinearizations per round: 50,640 (all) in rounds 1–10, 11,420 in rounds 91–100, 10,038 in rounds 191–200.
- Interpolated ATE at the end: 1.403 m raw and 0.378 m aligned, over 346 ground-truth times in the keyframe span.

### 3.4 L1 diagnostic: KITTI-40, 500 rounds each, sampled every 50 rounds

| Round | Shipped: remaining GN move, median / p95 / max | δ = 0, R = 1: same | Keyframe ATE raw / aligned (m), shipped | Same, δ = 0, R = 1 | Belief–cavity gap, median over variables (landmarks / poses), shipped | Same, δ = 0, R = 1 |
|---|---|---|---|---|---|---|
| 50 | 0.469 / 2.84 / 4.64 | 0.0445 / 4.49 / 7.74 | 0.51280 / 0.08455 | 0.51243 / 0.08632 | 0.137 / 2.60e-4 | 0.134 / 2.52e-4 |
| 100 | 0.425 / 1.95 / 2.20 | 0.0327 / 1.09 / 1.91 | 0.51284 / 0.08285 | 0.51252 / 0.08251 | 0.137 / 2.60e-4 | 0.134 / 2.51e-4 |
| 200 | 0.421 / 1.28 / 2.20 | 0.0237 / 0.112 / 0.186 | 0.51287 / 0.08229 | 0.51257 / 0.08144 | 0.137 / 2.61e-4 | 0.134 / 2.51e-4 |
| 300 | 0.421 / 1.28 / 2.20 | 0.0217 / 0.080 / 0.098 | 0.51287 / 0.08225 | 0.51257 / 0.08138 | 0.137 / 2.61e-4 | 0.134 / 2.51e-4 |
| 400 | 0.420 / 1.28 / 2.20 | 0.0215 / 0.078 / 0.094 | 0.51287 / 0.08225 | 0.51257 / 0.08137 | 0.137 / 2.60e-4 | 0.134 / 2.51e-4 |
| 500 | 0.420 / 1.28 / 2.20 | 0.0214 / 0.078 / 0.093 | 0.51287 / 0.08225 | 0.51257 / 0.08137 | 0.137 / 2.60e-4 | 0.134 / 2.51e-4 |

- **Samples.** The shipped median alternates between 0.420 and 0.429 on alternate 50-round samples (rounds 150, 250, 350 and 450, not shown, read 0.429). This is a small oscillation around a fixed point, not a drift.
- **Interpolated ATE at round 500** (78 ground-truth times): 0.5086 / 0.0783 m (shipped) against 0.5083 / 0.0776 m (δ = 0, R = 1).

**Plain statements:**
1. On KITTI-40 the shipped schedule does **not** drift away from the Gauss–Newton stationary point as it does on ChArUco (1.42 → 2.45 over 1,000 extra rounds). It **stalls** at a fixed distance, median 0.42 in D-units, from round 150 on.
2. The per-variable schedule converges to 0.021, a floor consistent with G-solver's O(ε) regularizer bias.
3. ATE does **not** distinguish the two fixed points. Their difference, 0.3 mm raw and 0.9 mm aligned in keyframe ATE, is 0.06–1.1 % of the error to ground truth.
4. The mechanism is visible directly: the shipped schedule linearizes every landmark factor at a cavity mean a median 0.13 m from the landmark's belief, and this gap is stationary.

## 4. Worst cases

| Quantity | Worst | Where |
|---|---|---|
| Final error at W₀, chosen setting | ChArUco 64.7; KITTI-40 1.21; prefix 14.7 | ChArUco pose t = 3.6 s (index 4, as in WP1 and Phase C); KITTI-40 pose t = 0.312 s; prefix pose t = 31.81 s |
| Default final error | ChArUco 35.0 (t = 3.6 s); KITTI-40 2.20 (t = 8.003 s); prefix 18.2 at round 100 (t = 32.44 s) and 11.9 at round 200 (t = 32.96 s) | |
| Largest ratio at W₀, any setting | KITTI-40 (0.003, 10): p95 ×8.66; (0.1, 10): median ×4.00. ChArUco (0.03, 1): p95 ×1.69 | |
| Closest pass | ChArUco (0.1, 10): median ×1.048 against the 1.05 bound | |
| Early-round deficit of the chosen setting | prefix round 10: p95 113.5 against the default's 44.6 | every factor relinearizes in rounds 1–10 while every belief moves more than 3 mm |
| Byte comparisons | 0 of 12 differ | |

## 5. Implementation (since WP1)

- **Physical metric:** `GbpSolver::relinearize_metric_` (`Physical`, the default: ‖z_v‖₂; `HBlock`: WP1's ‖z_v‖_{D_v}). The drivers take `--relin-metric`.
- **Work accounting:** `GbpSolver::planThresholdRound` freezes new nodes and decides the next round without running it, then returns its linearizations. `performIteration` runs exactly the planned round. `certify::RoundRunner` (`certify/cpp/certify_common.h`) accounts every round before it runs. It writes `work.csv` and exports:
  - the state at the beliefs every K rounds (`--curve-every`);
  - at the work budget (`--work-budget`, to `at_budget/`);
  - at a checkpoint (`--pause-at N`: export, then wait for `OUT/continue` or `OUT/abort`). This implements the two checkpointed segments.
- **Belief export mid-run:** `VariableNode::beginEvaluateAtBelief`, `beliefMean`, `cavityOffsetFromBelief`. The belief-chart export now evaluates at the beliefs in threshold mode without changing the solver state; on unfrozen graphs it is unchanged.
  - Checked: a threshold run with exports every 10 rounds writes a byte-identical trajectory to the same run without them.
  - Checked: the mid-run belief export gives exactly WP1's final-error metric (re-freeze everything, then export): maximum difference 0.
- **WP1's front-end fix** (`beliefOf` in `certify_charuco.cpp`) is kept, identical on the shipped path.

## 6. What could not be done, and choices stated plainly

- **Rounds-to-reach was corrected after the ChArUco runs.** As first written, it took the first curve point with median ≤ the default's final median. On the incremental ChArUco sequence, the early curve points cover only the poses added so far, so it returned round 10 for every setting. It now counts only curve points on the complete graph: from round 94 on ChArUco, every point on KITTI-40. The gate and the choice do not use rounds-to-reach and are unaffected.
- **W₀ for KITTI-40 uses the 100 rounds of WP1 run 2.** The shipped stereo application runs 10 rounds. At 10 rounds, every schedule is still in its transient (default median 1.90).
- **The full-prefix default comes from Phase C's exports**, the same shipped schedule and dynamic OpenMP schedule, byte-identical path, rather than from a new run. Its values exist at rounds 10, 20, 50, 100 and 200 only.
- **The full prefix was compared at W₀(100 rounds) and at 200 rounds**, where the threshold run used 13 % less work. It was not compared at W₀(200 rounds), which would need about 230 rounds.
- **ChArUco has no ground truth.** Its accuracy measure is the remaining Gauss–Newton move of the solver's own equation.
- **The chosen setting saves 73 % of linearizations per round but only 15 % of the work of a round.** Messages dominate the work, so the equal-work gate gives threshold schedules at most about 26 % more rounds. Larger savings need fewer messages, which is the job of the sleeping poses in WP2.

## 7. Compute used (minutes)

- Builds: 1.0.
- ChArUco: 13 runs of 0.03 each.
- KITTI-40: default 0.49; one-tenth version 0.10; 12 grid runs of 0.87–0.92 each (10.6 in total).
- Diagnostic: 2.43 (shipped) and 1.51 (δ = 0).
- Prefix: one-tenth version 0.17; main run 1.2 of compute (1.7 wall, including the checkpoint evaluation).
- Analyses: about 6.
- Equivalence: 4.8.
- No run exceeded 6 minutes.

## 8. Files

- Solver: `gsolver/include/gsolver/core/gbp_solver.h`, `gsolver/src/core/gbp_solver.cpp`, `gsolver/include/gsolver/graph/{variable_node_base.h, variable_node.h}`, `gsolver/src/graph/variable_node.cpp`.
- Drivers: `certify/cpp/{certify_common.h, certify_charuco.cpp, certify_stereo.cpp}`.
- Analysis: `scripts/phaseD_wp1b.py` (definitions in its docstring). Gate: `results/phaseD/wp1b/gate.json`.
- Per run: `results/phaseD/wp1b/{charuco,kf40,diag,prefix}/<run>/` (`wp1b_analysis.json` with the curve, `work.csv`, `relin_rounds.csv`, `relin_counts.csv`, `run.json`, `traj.tum`). Default prefix values: `results/phaseD/wp1b/prefix_default_phaseC_exports.json`. Equivalence: `results/phaseD/wp1b/equivalence.txt`.
- Exports (git-ignored): `exports/phaseD/wp1b/`. The curve exports were deleted after analysis; the `at_budget`, `final` and checkpoint exports are kept.
- Commit: the commit that adds this file (`git log -- reports/phaseD_wp1b.md`).
