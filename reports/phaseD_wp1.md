# Phase D, WP1: threshold relinearization

Branch `certify`, base commit `21fece4`. Date 2026-09-27. Decisions D-a and D-b of the WP0 review apply:
- δ and d_v are measured in D_v = H_vv of the current linearization, exponential chart;
- one linearization point per variable, frozen-mode style.

## 1. Five-line summary

1. **Implemented.**
   - Solver options `threshold_relinearization_`, `relinearize_threshold_` (δ) and `relinearize_every_` (R).
   - Each round: variables whose belief moved more than δ from their linearization point (‖z_v‖ in D_v), at most once per R rounds, are re-frozen at their belief, and every factor touching them relinearizes. Then G-solver's unchanged synchronous schedule runs on the linear model (`certify/THRESHOLD_RELIN.md`).
   - **The default path is byte-identical:** 12 of 12 comparisons, run twice.
2. **The gate fails.** No grid δ has a median final error within 5 % of the per-round schedule on ChArUco islam at its budget of 103 rounds.
   - ChArUco: δ = 0.25 is +26 % (1.80 against 1.42), and δ ≥ 0.5 is +60 %.
   - First 40 KITTI keyframes: δ = 0.25 (−1.5 %) and δ = 0.5 (−33 %) pass, δ = 1 and 2 fail.
   - So no δ is fixed, and run 3 (full prefix) was not run.
3. **L1, the certifiable form itself.** Relinearizing every variable at its belief every round (δ = 0, R = 1) matches the shipped cavity-mean schedule at the budget: ChArUco +5.3 %, KITTI-40 −59 %, ATE unchanged. With more rounds it keeps converging, while the shipped schedule does not.
   - On ChArUco the shipped schedule's remaining Gauss–Newton move grows from 1.42 to 2.45 over 1,000 extra rounds.
   - The per-variable schedules fall to 0.33–0.40 (diagnostic, §3.3).
4. **L1, the cost of holding a linearization for R = 10 rounds.**
   - ChArUco: +26 to +61 % median final error at the budget.
   - KITTI-40: medians mixed (−33 % to +89 %), but the tails are much worse (p95 12.6–16.1 against 1.95), and ATE is nearly unchanged (aligned 0.079–0.119 m against 0.079 m).
   - On ChArUco R, not δ, sets the schedule: the median rounds per linearization is 10–15 for δ ≤ 1.
5. **An implementation defect was found and fixed before these numbers.** The incremental ChArUco front end initialized new poses and markers from `mu_`, which in threshold mode is the linearization point, not the belief. Threshold runs were 16–22× worse because of it. The front end now reads the belief (identical on the shipped path; re-verified byte-identical).

## 2. Gate status

| Criterion (brief WP1) | Result | Status |
|---|---|---|
| Default path byte-identical (`scripts/phaseA_equivalence.sh`) | 12 of 12 identical, run once after the solver change and again on the final binaries (`results/phaseD/wp1/equivalence.txt`) | **pass** |
| δ = 0, R = 1 reproduces the shipped output | replaced by decision D-b. The per-variable form relinearizes at belief means, not the shipped cavity means, so it cannot reproduce the shipped output. Its accuracy against the shipped schedule is reported instead (§3, L1). | recorded, as decided |
| A grid δ with median final error ≤ 1.05 × the default's on run 1 (ChArUco) **and** run 2 (KITTI-40) | ChArUco: none (best is δ = 0.25, at 1.265 × the default). KITTI-40: δ = 0.25 (0.985 ×) and 0.5 (0.673 ×). | **fail** |
| Run 3: full prefix at the chosen δ | not run: no δ is chosen | – |

**Final pose error**, defined before any run in `scripts/phaseD_wp1.py`:
- For every pose, ‖x̄_i‖_{D_i}, where x̄ = H⁻¹b is the exact solution of the linearization at the run's final belief means and D_i = H_ii.
- This is the move that one exact Gauss–Newton step of the solver's own estimating equation would still make: brief WP3's "final error per pose against the last round's direct solve". It costs one direct solve, with no reference run.
- The criterion compares medians over poses. ChArUco has no ground truth. For KITTI, ATE/ARE against ground truth in the keyframe time span is reported alongside, with Phase 0's definitions.

## 3. Tables

### 3.1 Run 1: ChArUco islam at the shipped budget

The budget is one round after each new pose plus 10 final rounds: 103 rounds, 94 poses, 24 markers, 711 factors at the end.

| Schedule | Median final error | 95 % | Max | Median vs default | Rounds per linearization, poses (median / pooled) | Re-freezes per pose (median, max) | Factors relinearized per round | Wall (s) |
|---|---|---|---|---|---|---|---|---|
| default (per round, cavity means) | 1.423 | 19.33 | 34.99 | – | 1 | – | every factor | 0.14 |
| δ = 0, R = 1 (belief means) | 1.499 | 20.38 | 66.35 | +5.3 % | 1.0 / 1.0 | 55.5, 102 | 423.5 | 0.29 |
| δ = 0.25, R = 10 | 1.800 | 27.91 | 81.86 | +26.5 % | 10.0 / 12.5 | 3, 9 | 29.5 | 0.17 |
| δ = 0.5, R = 10 | 2.272 | 31.72 | 95.17 | +59.7 % | 11.3 / 14.9 | 2, 9 | 22.8 | 0.16 |
| δ = 1, R = 10 | 2.280 | 31.77 | 95.30 | +60.2 % | 14.6 / 18.2 | 1, 9 | 17.1 | 0.16 |
| δ = 2, R = 10 | 2.286 | 31.88 | 95.15 | +60.6 % | 22.0 / 22.0 | 1, 9 | 12.8 | 0.15 |

The excess error is spread along the sequence. Over the first 84 poses the median goes from 1.15 (default) to 1.65 (δ = 0.25) and 2.05 (δ = 1). The last 10 poses, which had the fewest rounds, rise more: from 2.30 to 3.44 and 5.53.

### 3.2 Run 2: the first 40 KITTI 06 keyframes, 100 rounds

The graph has 40 poses, 4,679 landmarks, 11,487 stereo factors and 39 GP factors.

| Schedule | Median final error | 95 % | Max | Median vs default | ATE raw / aligned (m) | ARE raw / aligned (rad) | Rounds per linearization, poses (median) | Re-freezes per pose (median) | Factors relinearized per round | Wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| default | 0.4245 | 1.951 | 2.201 | – | 0.5086 / 0.0788 | 0.0127 / 0.0339 | 1 | – | 11,526 (all) | 31.0 |
| δ = 0, R = 1 | 0.1722 | 1.007 | 1.802 | −59.4 % | 0.5083 / 0.0785 | 0.0127 / 0.0329 | 1.0 | 99 | 11,410.7 | 19.5 |
| δ = 0.25, R = 10 | 0.4181 | 16.13 | 43.97 | −1.5 % | 0.5066 / 0.0836 | 0.0128 / 0.0297 | 10.0 | 9 | 1,187.2 | 17.9 |
| δ = 0.5, R = 10 | 0.2855 | 12.65 | 65.91 | −32.7 % | 0.5131 / 0.1187 | 0.0130 / 0.0292 | 11.1 | 8 | 1,005.9 | 18.0 |
| δ = 1, R = 10 | 0.6221 | 15.73 | 40.37 | +46.5 % | 0.5073 / 0.0832 | 0.0128 / 0.0296 | 14.3 | 6 | 698.0 | 17.8 |
| δ = 2, R = 10 | 0.8037 | 3.640 | 8.375 | +89.3 % | 0.5083 / 0.0791 | 0.0127 / 0.0328 | 25.0 | 3 | 331.1 | 17.7 |

- Median seconds per threshold round: 0.17–0.19. The default run's wall time includes G-solver re-evaluating every factor each round.
- The median criterion is not monotone in δ here.

### 3.3 Diagnostic, outside the pre-registered protocol: ChArUco with more final rounds

Median (95 %) final error:

| Rounds (final BA rounds) | default | δ = 0, R = 1 | δ = 1, R = 10 |
|---|---|---|---|
| 103 (10, the shipped budget) | 1.423 (19.3) | 1.499 (20.4) | 2.280 (31.8) |
| 193 (100) | 1.450 (22.1) | 1.229 (15.6) | 1.741 (21.3) |
| 1,093 (1,000) | 2.449 (43.5) | 0.334 (6.4) | 0.404 (8.2) |

- **The per-variable schedules keep approaching the fixed point of the solver's estimating equation.** Their decrease matches the linear-GBP convergence rate Phase A measured on ChArUco (0.99886 per round: 19.1 → 6.5 over 1,000 rounds before the front-end fix, as predicted).
- **The shipped schedule does not approach it.** Its remaining Gauss–Newton move grows with more rounds.
- This was run to rule out an implementation fault in threshold mode (§5). It is not part of the gate.

## 4. Worst cases

| Quantity | Worst | Where |
|---|---|---|
| Final error, ChArUco | 95.3 (δ = 1); default 35.0 | the pose at t = 3.6 s (sequence index 4) in **every** run, the largest-error pose in Phase C as well |
| Final error, KITTI-40 | 65.9 (δ = 0.5); default 2.20 (pose t = 8.003 s) | the pose at t = 5.092 s for δ ∈ {0.25, 0.5, 1}; t = 5.508 s for δ = 2; t = 0.312 s for δ = 0 |
| Median ratio to default | 1.606 (ChArUco, δ = 2); 1.893 (KITTI-40, δ = 2) | |
| 95 % ratio to default | 8.3× (KITTI-40, δ = 0.25: 16.13 against 1.951) | |
| Max ratio to default | 29.9× (KITTI-40, δ = 0.5: 65.91 against 2.201) | |
| ATE (aligned) | 0.1187 m against 0.0788 m (KITTI-40, δ = 0.5) | |
| Byte comparisons | 0 of 12 differ | |

## 5. Implementation, and the defect found

**Solver** (`certify/THRESHOLD_RELIN.md`):
- `gsolver/include/gsolver/core/gbp_solver.h`, `gsolver/src/core/gbp_solver.cpp`: the options, per-round bookkeeping, `relinearizeAllAtBeliefs`. The default `performIteration` gains one branch at its top.
- Variable nodes: `freezeAsNew`, `refreezeAtBelief` (message means keep their manifold points and are re-expressed in the new chart; precisions kept, as in frozen mode), `tangentState`.
- Factor nodes: `linearizedInformationBlock` (J_lin,vᵀΣ⁻¹J_lin,v for D_v) and edge accessors; the edge tuple gains `edgeSpan`.

**Drivers:**
- `certify_charuco --mode islam` and `certify_stereo` accept `--relin-threshold`, `--relin-every` and `--final-export`.
- `certify_charuco` also takes the diagnostic `--extra-final-rounds`, default 0.
- They write `relin_rounds.csv` (per round) and `relin_counts.csv` (per variable).

**Defect, found by the diagnostic and fixed.**
- In threshold mode `mu_` is a variable's linearization point; its belief is `mu_lin_ ⊞ z_`.
- The ChArUco front end predicts each new pose from the previous pose's `mu_` (`certify_charuco.cpp`, from the shipped `charuco_experiment_islam.cpp`). It also initializes each new marker from its camera pose's `mu_` (`factor_graph_builder.h:280`).
- In threshold mode both therefore read estimates up to R rounds stale, which made every threshold run 16–22× worse than shipped (δ = 0, R = 1: median 22.9 against 1.42).
- The front end now reads `beliefOf(pose)`: `mu_` when the variable is not frozen (so the shipped path is unchanged), `mu_lin_ ⊞ z_` when it is. After the fix, shipped islam is byte-identical again, with the static and the dynamic schedule. All numbers above are after the fix.
- The stereo runs are batch runs and never read `mu_` during the run.

## 6. What could not be done, plainly

- **The gate's second condition fails, so no δ was fixed and run 3 was not run.** The failure is not specific to thresholding: even per-round per-variable relinearization (δ = 0, R = 1) sits at +5.3 % on ChArUco at its 103-round budget. At larger budgets it is far more accurate than the shipped schedule (§3.3).
- **δ = 0, R = 1 cannot reproduce the shipped output** (decision D-b). The difference between belief-mean and cavity-mean relinearization is the L1 finding in §1.
- **ChArUco has no ground truth.** Its accuracy measure is the remaining Gauss–Newton move of the solver's own equation, not the error to the truth.
- **Only R = 10 was run on the grid, as specified.** A smaller R would bring ChArUco closer to δ = 0, R = 1, which itself misses the 5 % bound by 0.3 points. This is untested.
- **The final error is recorded at the end of each run only**, not per round.

## 7. Decision needed

WP2 needs δ > 0: Theorem C's stop condition d_i + E_i ≤ δ_i is empty at δ = 0. Options:
1. **Fix δ = 0.25, R = 10 from run 2** and record that ChArUco misses the 5 % bound (+26 %). KITTI-40 medians and ATE are kept, but its p95 is 8× the default's.
2. **Re-run runs 1–2 with a smaller R** (R ∈ {1, 3}) on the same δ grid, a few minutes, since R is what binds on ChArUco. Even R = 1 is unlikely to meet the bound there (δ = 0, R = 1 is +5.3 %).
3. **Change the accuracy yardstick** for L1, for example ATE on KITTI or a longer ChArUco budget, where the per-variable schedules win. This is a change of the pre-registered criterion, so it is your call.

**My recommendation is option 1.** The gate's ChArUco failure is a property of the certifiable (per-variable) form at a 1-round-per-pose budget, not of the threshold. WP2's certificates will expose any tail poses that the threshold leaves stale, via RELINEARIZE-NOW.

## 8. Compute used (minutes)

- Builds: 1.3 + 0.7 + 0.1.
- Equivalence: 4.9 and 4.8.
- ChArUco runs: 0.002–0.005 each (diagnostic: up to 0.065).
- KITTI-40 runs: 0.30–0.52 each (six runs, 2.0 in total); one-tenth versions: 0.04 and 0.06.
- Analyses: about 1.5 in total.
- No run exceeded 6 minutes.

## 9. Files

- Design: `certify/THRESHOLD_RELIN.md`.
- Solver: `gsolver/include/gsolver/core/gbp_solver.h`, `gsolver/src/core/gbp_solver.cpp`, `gsolver/include/gsolver/graph/{variable_node_base.h, variable_node.h, factor_node_base.h, factor_node.h}`, `gsolver/include/gsolver/graph/helpers/factor_node_edge_tuple_helper.h`, `gsolver/src/graph/{variable_node.cpp, factor_node.cpp}`.
- Drivers: `certify/cpp/{certify_charuco.cpp, certify_stereo.cpp, certify_common.h}`.
- Analysis: `scripts/phaseD_wp1.py`. Results: `results/phaseD/wp1/{wp1_charuco.json, wp1_kf40.json, equivalence.txt}`, per run under `results/phaseD/wp1/{charuco,kf40}/<run>/` (`run.json`, `relin_rounds.csv`, `relin_counts.csv`, `final_pose_error.npy`, `traj.tum`). Final-state exports (git-ignored): `exports/phaseD/wp1/`.
- Commit: the commit that adds this file (`git log -- reports/phaseD_wp1.md`).
