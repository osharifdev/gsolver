# Phase A report: trustworthy export

Branch `certify`. Date 2026-09-24. Amendments A1–A7 applied: where they differ from the brief, they are followed. All runs use `OMP_NUM_THREADS=16`.

## 1. Five-line summary

1. G-solver now has an opt-in **frozen (linear) mode** (A2): one fixed linearization point per variable, linearized factors, and messages in fixed charts with no `⊞`/`⊟` remap (`certify/FREEZE_MODE.md`). Exports exist for ChArUco islam (25/50/100 %), ChArUco BA (iterations 0 and 50), kitti06-prefix (20 keyframes; iteration 0 and the normal-run cap) and KITTI 06 full (normal-run cap), plus unfrozen exports of islam and the prefix at iterations 1–100. The default solver path is **bit-identical** (12/12 byte comparisons).
2. **Criterion (i) passes everywhere.** Every frozen model solves directly to a relative residual of at most 2.2e-14 (threshold 1e-10). Frozen GBP decreases to a flat floor on every snapshot. It is not strictly monotone: ChArUco has 0–4 increasing steps before its floor, and the prefix frozen at iteration 0 oscillates with period 2 for about 100 iterations under a falling envelope (§3.1).
3. **Criterion (ii) fails at G-solver's settings on every dataset** (floor ≤ 1e-4). Each floor is explained:
   - **ChArUco (5.5e-4 to 9.9e-4)** is G-solver's `1e-6·I` regularizer bias. It is linear in ε over three decades and comes from the two factor-side sites; with ε = 1e-7 the floors are 5.7e-5 to 9.9e-5.
   - **kitti06-prefix (poses 6.1e-2 to 7.0e-2) is conditioning.** Landmarks triangulated at about 10 km carry depth information down to 1.7e-11, far below 1e-6. x̄ moves them hundreds of km and drags the poses with them. Holding the 24–31 landmarks with less than 1e-3 depth information at GBP's values leaves pose floors of 6.2e-5 to 6.5e-5.
   - **KITTI full (poses 3.19)** is the same effect: holding 42 / 159 / 536 such landmarks gives 2.2e-2 / 2.3e-3 / 1.1e-3. The last 1.1e-3 is not attributed: lowering ε to 1e-7 makes the frozen iteration noisy on this graph.
4. **Structural facts that later phases must handle.**
   - Single-observation landmarks' frozen initial messages act as 1e6·I priors on the rest of the graph but not on their own belief. GBP therefore never converges to x̄ for those blocks (4,928 in KITTI 06).
   - A few landmarks have every factor gated out and are exactly decoupled (257 in KITTI full).
   - The stereo rotation Jacobian is 2× in every stereo factor (F5, confirmed numerically).
   - The A3 gauge prior is redundant, because the single-edge priors already fix the gauge.
   - The normal stereo runs do not meet the convergence rule within 200 iterations.
   - G-solver's rotation matrices drift off SO(3) during normal runs (max |RᵀR − I| = 1.2e-6 on KITTI after 200 iterations), because the small-angle branch of `expMapSO3` is first order.
5. **Gate A: FAIL by the letter of A4** (no floor is ≤ 1e-4 at G-solver's ε), with every floor attributed by a controlled experiment. This decision is yours (§2).

## 2. Gate status (A4)

"Non-single-edge" means all informed blocks except single-edge variables, whose GBP belief is not x̄ by construction (§5 F2). "Poses" is the pose-velocity blocks. D-metric throughout.

| Snapshot | (i) residual | Floor, non-single-edge | Floor, poses | Reached at | Rate / iteration | κ(H) / κ(D⁻¹ᐟ²HD⁻¹ᐟ²) | Explained by | ≤ 1e-4? |
|---|---|---|---|---|---|---|---|---|
| islam 25 % | 4.6e-16 | 9.89e-4 | 9.89e-4 | 542 | 0.9823 | 3.2e13 / 4.2e7 | regularizers (ε = 1e-7 → 9.86e-5) | **no** |
| islam 50 % | 5.5e-15 | 5.54e-4 | 5.54e-4 | 2,220 | 0.9952 | 7.6e13 / 5.9e7 | regularizers (→ 5.68e-5) | **no** |
| islam 100 % | 2.1e-14 | 7.41e-4 | 7.41e-4 | 7,020 | 0.9988 | 1.4e14 / 6.6e7 | regularizers (→ 7.39e-5) | **no** |
| BA, iteration 0 | 1.6e-14 | 7.25e-4 | 7.25e-4 | 9,090 | 0.9989 | 1.3e10 / 6.8e7 | regularizers (→ 7.30e-5) | **no** |
| BA, iteration 50 | 2.2e-14 | 7.39e-4 | 7.39e-4 | 8,960 | 0.9989 | 1.3e10 / 6.6e7 | regularizers (→ 7.40e-5; 739·ε for ε ∈ [1e-7, 1e-5]) | **no** |
| prefix, iteration 0 | 3.6e-15 | 1.19 (far landmarks) | 6.05e-2 | 128 | 0.9996 | 1.8e19 / 5.0e6 | conditioning (24 landmarks fixed → 6.5e-5) | **no** |
| prefix, cap 200 | 6.3e-18 | 1.24 (far landmarks) | 6.96e-2 | 115 | 0.9992 | 2.4e19 / 6.8e6 | conditioning (31 fixed → 6.2e-5); ε = 1e-8 only 0.66× | **no** |
| KITTI full, cap 200 | 4.3e-17 | 32.7 (far landmarks) | 3.19 | 115 | 0.9573 | 3.3e22 / 3.2e9 | conditioning (536 held → 1.1e-3); the last 1.1e-3 not attributed (ε = 1e-7 is noise-dominated here) | **no** |

The islam κ(H) values reflect the 1e12 anchor. The prefix and KITTI κ(H) values reflect the far landmarks (λ_min ≈ 2e-11 on the prefix, 2.6e-14 on KITTI full). The D-scaled condition numbers are 5e6 to 7e7, except 3.2e9 on KITTI full. Every direct solve was also refined with an extended-precision residual (ChArUco BA): x̄ moves by at most 1.9e-8 in the D-metric, so the floors are not solve error.

## 3. Tables

### 3.1 Scope (A1) and snapshots (A5)

| Dataset | Graph | Freeze instants (frozen exports) | Unfrozen exports |
|---|---|---|---|
| ChArUco islam | 94 poses, 24 SE(3) markers, 617 + 93 + 1 factors | after the iteration on 24 poses (25 %), 47 poses (50 %), end of the normal run (100 %) | normal iterations 1, 2, 5, 10, 20, 50, 100 |
| ChArUco BA (noise 0.0, README default) | 94 poses, 24 markers | iteration 0 (initial guess) and 50 (end of the normal run) | — |
| **kitti06-prefix** | **20 keyframes (0–3.74 s)**, 2,359 landmarks, 5,973 stereo, 19 GP | iteration 0, and the normal-run cap (see below) | normal iterations 1, 2, 5, 10, 20, 50, 100 |
| KITTI 06 full | 477 keyframes, 45,555 landmarks, 157,941 stereo, 476 GP | normal-run cap | — |

- **kitti06-prefix length: 20 keyframes.** A1's definition (the longest initial run of keyframes with no landmark re-observed after a gap of more than 10 poses) stops at keyframe 20. Landmark 2182 was last seen at keyframe 1 and is seen again at keyframe 20 (gap 19). This is a far-point re-observation, not the loop.
- **Normal-run convergence rule, fixed before any run:** max over blocks of the belief-mean step in the belief-precision metric, below 1e-3, capped at 200 iterations. **Neither stereo run met it.** The prefix ended at a max step of 1.32 (median 5.6e-5); KITTI full at a max of 36.8 (median 9.8e-4). Both were frozen at the cap. ChArUco snapshots are defined by sequence position, so the rule does not apply there.
- The frozen state is dumped at every iteration up to 1,000, then every 10th. Frozen iterations run: ChArUco 30,000; prefix 2,000–3,000. KITTI full ran 3,740 of a planned 5,000: I stopped it by hand at the last complete dump when other jobs overloaded the host (load average above 100), 3,600 iterations after its curve went flat (`frozen_run.json` records this).
- **Shape of the curves before the floor.** ChArUco is nearly monotone: 0–4 increasing steps out of 541–1,808, the largest a 26 % bump early in BA iteration 0. The prefix frozen at iteration 0 alternates up and down on even and odd iterations for about 100 iterations while the envelope falls (poses 6.8 → 1.5 by t = 39), and iteration 1 spikes to 1.7e3 at landmarks. This is the usual behaviour of synchronous GBP on a pose–landmark bipartite graph. Snapshots frozen after a normal run have 0–4 increasing steps.

### 3.2 Freeze mode (A2): what it bypasses, how the regularizers enter

The full account, with line references, is in `certify/FREEZE_MODE.md`. In frozen mode, three things change:

- `updateErrorAndJacobian()` is replaced by `e_f(x̂) + J_f(x̂)·τ` (`factor_node.cpp:18`, shipped line 18).
- `boxplus`/`boxminus` in the factor-to-variable message, the variable-to-factor messages and the belief update become vector operations in the fixed charts (shipped `factor_node_helper.h:73`; `variable_node.cpp:39,42,57,61,98,101,116,120,146,149,164`).
- The belief mean `mu_` is never rewritten after the freeze.

The canonical form, conditioning, Schur marginalization, schedule and single-edge rule are unchanged.

| `1e-6·I` site | Effect | Enters the fixed point? | Measured: that site alone at 1e-7 (BA, iteration 50) |
|---|---|---|---|
| ε1 belief (`variable_node.cpp:189`) | proximal term at the previous mean | no, cancels | floor 7.391e-4 (unchanged) |
| ε2 variable→factor message (`:58,:129`) | prior at the current belief on every outgoing message | no: a prior at z* leaves H⁻¹b fixed | 7.392e-4 (unchanged) |
| ε3 marginalization (`factor_node_helper.h:26`, set at `:70`) | prior at the incoming message's own mean (a cavity mean) | **yes, O(ε)** | 2.46e-4 |
| ε4 factor→variable message (`:74`) | prior at the cavity mean m_{a→f} | **yes, O(ε)** | 9.42e-4 (partly cancels ε3) |

ε3 and ε4 are centred at GBP-internal cavity means, so no fixed (G, b_G) represents them. **They are therefore not in the exported G**; their effect is measured instead.

**Bit-identity of the default path** (`results/phaseA/equivalence.txt`):

- The shipped drivers rebuilt against the modified library match the Phase 0 outputs (islam, BA ×2, stereo): 4/4.
- The certify drivers match the shipped drivers: 4/4.
- Unfrozen exports do not perturb the run: 1/1.
- The opt-in dynamic OpenMP schedule (below) is also bit-identical: stereo vs Phase 0, islam, and frozen state dumps, 3/3.

**Performance-only addition.** I added `GbpSolver::dynamic_schedule_` (default off). It is used for KITTI full, where the shipped static schedule runs all 477 poses on one thread. It gives the same bits and makes iterations about 10× faster (27 s instead of 125 s for the 10-iteration KITTI run).

### 3.3 Regularizer diagnostic (ChArUco BA, iteration 50; all four sites scaled together unless noted)

| ε | 1e-5 | **1e-6 (G-solver)** | 1e-7 | 1e-8 | 1e-9 | 1e-12 |
|---|---|---|---|---|---|---|
| floor | 7.35e-3 | **7.39e-4** | 7.40e-5 | 1.32e-5 | 8.5e-5 (noisy) | 5.3e-2 |
| floor / ε | 735 | 739 | 740 | 1,318 | — | — |

- The floor is linear in ε down to 1e-7. Below about 1e-8, round-off from inverting rank-deficient factor messages, whose conditioning grows like 1/ε, takes over.
- The rate is 0.99886 for every ε.
- The worst blocks are always consecutive poses less than 1/30 s apart (3.600/3.617, 23.550/23.583, …). There the GP prior makes D_i stiff (up to 5e7), so a 6e-7 m offset reads as 7e-4.
- **On kitti06-prefix, ε is not the cause.** The pose floor goes from 6.96e-2 (ε = 1e-6) to 6.63e-2 (1e-7) to 4.59e-2 (1e-8).
- **On KITTI full with ε = 1e-7**, the pose floor is 3.18, against 3.19 at 1e-6. With the 536 far landmarks held, the remainder fluctuates between 1.3e-2 and 3.5e-2 over t = 600–1,000, and the state still moves by about 1.5e-2 D-units every 50 iterations. That is the round-off regime ChArUco entered only below 1e-8; KITTI's messages are stiffer (1e6 single-edge priors, ~300 observations per pose). **No ε gives KITTI full a floor below 1e-4**, and the 1.1e-3 that remains at ε = 1e-6 after holding the far landmarks is not attributed by this test.

### 3.4 Conditioning diagnostic: landmarks at infinity (stereo)

Each row re-solves the model with every landmark whose weakest D-direction carries less than τ information held at GBP's final value, then compares GBP's state with that solution.

| Snapshot | Pose floor vs x̄ | τ = 1e-6 (n) | τ = 1e-4 (n) | τ = 1e-3 (n) |
|---|---|---|---|---|
| prefix, iteration 0 | 6.05e-2 | 4.05e-4 (2) | 2.75e-4 (6) | **6.51e-5 (24)** |
| prefix, cap 200 | 6.96e-2 | 1.66e-3 (2) | 6.14e-4 (8) | **6.15e-5 (31)** |
| KITTI full, cap 200 | 3.19 | 2.20e-2 (42) | 2.27e-3 (159) | 1.12e-3 (536) |

Worst case, LANDMARK_643 (prefix):

- It lies at 10.3 km depth, is seen from 7 keyframes, and its D-block eigenvalues are [1.7e-11, 3.8e-3, 3.8e-3].
- x̄ places it 2.97e5 m further along the depth direction. That is an O(1) offset in the D-metric (1.24) in a direction the solver cannot resolve, because 1.7e-11 is far below ε.
- In KITTI full the worst block is LANDMARK_169675 (6 observations; 32.7 D-units). The worst pose error is 1.2 cm and 3e-4 rad, at pose t = 80.2 s. The median pose translation offset is 57 µm.
- SE(3) alignment does not help (fitted transform 7.5e-7 rad, 9.4e-5 m).

### 3.5 Gauge (A3) and the near-null space

- **ChArUco G:** the drivers' own anchor. Islam uses precision 1e12; BA uses 1e6.
- **Stereo G:** our modelling choice (A3, owned by us). It is a `CERTIFY_GAUGE_PRIOR`: an SE(3) prior on the first keyframe at its initial ORB-SLAM3 pose, precision 1e4·I, added to the frozen model only via `freezeNewFactor`.
- The frozen single-edge messages form a second G block (§5 F2).

| Model (prefix, cap 200) | λ_min | λ_max | Eigenvalues < 1e-8·λ_max (4.2) | < 1e-10 | < 1e-8 | < 1e-6 | Cholesky |
|---|---|---|---|---|---|---|---|
| with CERTIFY_GAUGE_PRIOR | 1.73e-11 | 4.20e8 | 952 | 1 | 1 | 2 | ok |
| without it (single-edge priors kept) | 1.73e-11 | 4.20e8 | 952 | 1 | 1 | 2 | ok |
| measurements only (no single-edge priors) | ≈ 0 | 4.20e8 | 1,416 | 4 | 6 | 9 | fails |

- **KITTI full**:
  - With the prior: λ_min = 2.64e-14, λ_max = 8.64e8; 26,501 eigenvalues below 1e-8·λ_max (8.6); 13 / 36 / 42 below 1e-10 / 1e-8 / 1e-6; Cholesky succeeds.
  - Without the prior: λ_min = 2.64e-14 (identical to 4 digits); 26,502; 13 / 36 / 42.
  - Measurements only: 31,261; 17 / 42 / 53; 2 eigenvalues numerically negative at round-off level. Cholesky still succeeds here: the gauge directions sit at about 1e-14, below double-precision resolution of this matrix (eps·λ_max ≈ 1e-7).
- **The explicit prior does not change λ_min or the near-null count.** The 595 (prefix) and 4,928 (full) single-observation landmarks already fix the gauge through their frozen 1e6·I messages. Without those priors as well, the gauge null space appears. The count below 1e-8 rises from 1 to 6 on the prefix and from 36 to 42 on KITTI full.
- **A3's near-null dimension (952) is dominated by weakly measured landmark directions, not by gauge.** λ_max (4.2e8) is set by the 1e6 single-edge priors and ~300 observations per pose, so the 1e-8·λ_max = 4.2 threshold sits above most landmark information.
- **λ_min is a far-landmark depth direction.** It needed Lanczos on H⁻¹ through the scaled factorization; a dense eigensolver on H only resolves eigenvalues above about 4e-8.

### 3.6 Unfrozen exports (charts at the belief means, state z = 0; x̄ = Gauss–Newton step with G-solver's Jacobians)

| Normal iteration | islam dof | islam GN step, D-metric max | prefix GN step, max / median | prefix uninformed dofs |
|---|---|---|---|---|
| 1 | 54 | 1.5e-24 (one pose: velocity has zero information, 6 dofs, H singular there) | 1,621 / 45.5 | 123 |
| 2 | 72 | 2.85 | 154 / 0.40 | 15 |
| 5 | 114 | 24.7 | 125 / 0.23 | 15 |
| 10 | 186 | 114 | 11.0 / 0.23 | 18 |
| 20 | 336 | 46.4 | 9.25 / 0.12 | 18 |
| 50 | 726 | 46.7 | 4.88 / 0.080 | 18 |
| 100 | 1,272 | 35.4 | 3.01 / 0.070 | 18 |

All unfrozen models solve to a relative residual of at most 1.3e-14. Per-iteration details are in `results/phaseA/unfrozen_summary.csv`.

### 3.7 Timing (16 threads, shared host)

| | Normal run | Frozen iterations |
|---|---|---|
| ChArUco | ≤ 0.1 s | 0.9–1.9 ms/iteration (30,000 in 26–56 s) |
| kitti06-prefix | 200 iterations in 96 s | 213–221 ms/iteration (static schedule) |
| KITTI full | 200 iterations in 271 s (dynamic schedule) | 3,740 iterations (stopped; planned 5,000), 1.2 s/iteration unloaded and up to about 5 s/iteration under the other jobs' load |
| Python validation | — | ChArUco seconds; prefix ≈ 2 min; KITTI full about 12 min (direct Schur solve about 4 min; the A3 spectra about 3 min) |

## 4. Worst cases

| Check | Worst case | Numbers |
|---|---|---|
| (i) direct-solve residual | BA, iteration 50 | 2.2e-14 (all ≤ 1e-10) |
| (ii) floor, poses | KITTI full, cap 200, POSE_80.228940 | 3.19 D-units (1.21 cm, 3.0e-4 rad, velocity 4.0e-3); median pose offset 57 µm |
| (ii) floor, non-single-edge | KITTI full, a far landmark | 32.7 D-units |
| floor after excluding conditioning | KITTI full, 536 landmarks held | 1.12e-3 |
| single-edge offset from x̄ | KITTI full, cap 200 | 2.69e5 D-units (prefix 2.35e5): their GBP belief omits their own 1e6·I prior |
| regularizer floor at G-solver's ε | islam 25 % | 9.89e-4 |
| slowest convergence | ChArUco BA | rate 0.99886; floor reached after about 9,000 iterations |
| conditioning | KITTI full, cap 200 | κ(H) = 3.3e22, scaled 3.2e9 (λ_min = 2.6e-14) |
| equivalence | — | 0 differences in 12 byte comparisons |

## 5. Facts and code references

Shipped code is at `b330450`; line numbers below refer to it.

**F1. Relinearization: every factor, every iteration, at its own incoming message means.** `performIteration` runs, synchronously: all factors `updateFactor()`, all factors `sendAllMessages()`, all variables `updateMuAndSigma()`, all variables `sendAllMessages()` (`gbp_solver.cpp:8-35`). `updateFactor()` starts with `updateErrorAndJacobian();` (`factor_node.cpp:18`). Every factor type reads its linearization point from the incoming variable-to-factor messages:

```
Message<SE3PoseVel> message_from = edge_from->variable_to_factor_message_;   // se3_posevel_gp.cpp:14
SE3PoseVel::EstimateType x_from  = message_from.mu_;                          // se3_posevel_gp.cpp:16
const SE3PoseVel::EstimateType& x_posevel = edge_pose->variable_to_factor_message_.mu_;  // se3_posevel_stereo_point.cpp:15
```

Those means are the cavity means (belief without that factor, plus εI; `variable_node.cpp:30-61`), so **each factor linearizes each variable at a different point**, and all of them do so every iteration. Timing is synchronous; linearization points are per factor. A new factor's first linearization is at the creation-time message `mu_ = variable_node->mu_` (`factor_graph_helper.h:37-38`). In frozen mode (A2) all of this is replaced by one point per variable.

**F2. Gauge.**

- **ChArUco islam:** `SE3PoseVelPrior(prior_id, Identity, Identity*1e-12)` on the first pose (`charuco_experiment_islam.cpp:153-157`), precision 1e12, pose part only (`derivatives.h:21-26`, velocity columns zero).
- **ChArUco BA:** `fix_first_pose(factor_graph)` with default `sigma = 1e-6` (`charuco_experiment_ba.cpp:129`, `factor_graph_builder.h:323-331`), precision 1e6.
- **Stereo, single and multi-robot: no anchor.** The graph is `add_pose_variable_nodes`, `add_gp_prior_factor_nodes`, `add_landmark_variable_nodes` and `add_stereo_factor_nodes` (`stereo_visual.cpp:156-158,181-182`). The parsed priors (`:130`) are never used, and `06.g2o` has none.
- **What fixes the stereo gauge is a side effect.** Every `LANDMARK_3D` covariance in the file is 0, so `add_landmark_variable_nodes` gives each landmark `sigma = lm.covariance + 1e-6*I` (`factor_graph_builder.h:360`), i.e. an initial precision of 1e6·I. A variable with one edge never updates its outgoing message:
  ```
  if (edges_.size() == 1) { return; }   // variable_node.cpp:26-28
  ```
  So each single-observation landmark keeps its creation-time message N(ORB-SLAM3 position, 1e-6·I) forever. At the fixed point this acts as a 1 mm prior on the rest of the graph, but not on that landmark's own belief, which is the product of factor messages only (`variable_node.cpp:132-167`).
  - KITTI 06 has 4,928 such landmarks; the prefix has 595.
  - They are exported as a separate G block (`single_edge`), with the caveat that GBP's belief of those blocks differs from x̄ by construction (§4: up to 2.69e5 D-units on KITTI full).
- **Distributed mode:** each robot graph is built the same way, with no anchor. Landmark copies are tied by `SE3PointEquality` factors with covariance `1e-6*I` (precision 1e6; `factor_graph_builder.h:436-446`). Robots iterate sequentially in one process (`stereo_visual.cpp:216-220`). Each robot's gauge is fixed by its own single-edge landmarks and the equality factors.
- **The A3 prior is our choice** for the frozen stereo models. It is numerically redundant (§3.5).

**F3. Interpolation (`gp_interpolation.h:266-402`, `GPInterpAtTimestamps`; `GPInterp`, lines 33-253, is the same).** For τ ∈ [t_i, t_{i+1}]:

- Φ(t, s) = [[I, (t−s)I], [0, I]] (`:16-21`)
- Q(s, t) = [[dt³/3·Qc, dt²/2·Qc], [dt²/2·Qc, dt·Qc]] (`:23-31`), with 1e-4 added to Qc entries below 1e-4 (`:277-281`). The factors instead use `cwiseMax(1e-4)`; neither floor is active for `printing_room` or `kitti_05`.
- **Ψ(τ) = Q(t_i, τ) Φ(t_{i+1}, τ)ᵀ Q(t_i, t_{i+1})⁻¹** and **Λ(τ) = Φ(τ, t_i) − Ψ(τ) Φ(t_{i+1}, t_i)** (`:330-331`), both 12×12.
- The local variables are γ_i = [0; v_i] and γ_{i+1} = [Log(T_i⁻¹T_{i+1}); blkdiag(R, R)·v_{i+1}] with R = R_iᵀR_{i+1} (`:338-350`).
- The outputs are T(τ) = T_i·Exp(Λ₁γ_i + Ψ₁γ_{i+1}) and ω(τ) = blkdiag(Rᵀ, Rᵀ)(Λ₂γ_i + Ψ₂γ_{i+1}) (`:352-358`).

**It is affine (linear) in the local variables (γ_i, γ_{i+1}), not in the endpoint states.** γ_{i+1} contains the SO(3)×ℝ³ Log of the relative pose and a rotated velocity, and the result is retracted with Exp. In the fixed charts of a frozen model, the map (z_i, z_{i+1}) ↦ δx(τ) is therefore affine only to first order, with its Jacobians at x̂. Theorem 5.6's hypothesis holds only in linearized form.

The covariance (`:360-390`) uses block-diagonal endpoint covariances. `P_1n2` comes from the two beliefs' `Sigma_`, ignoring their cross-covariance; a commented-out alternative used `H_inv`, `:207`. It also drops the `Pinv_comp` term (commented out, `:229-230`).

**Verified:** `certify/interp.py` transcribes the mean formula. At all 1,101 GT timestamps it reproduces G-solver's own interpolation of the freeze-time beliefs (`interp_at_freeze.npy`, written by the driver) to within 2.3e-13 m, 2.5e-16 in rotation and 1.0e-13 in velocity (prefix, iteration 0: 37 queries inside the knots, the rest extrapolated). **KITTI full** (1,101 queries, 1,100 inside the knots) agrees to 1.1e-13 m, 3.2e-16 and 8.2e-13. That required transcribing one more G-solver detail: `Isometry3d::inverse()` uses Rᵀ. A general 4×4 inverse differs by up to 2.8e-6 m, because **G-solver's rotation matrices are not exactly orthonormal**. The small-angle branch of `expMapSO3` is first order (`R = I + W` for θ² < 1e-8, `geometry3d.h:85-86`), and every boxplus applies it, so rotations drift off SO(3): max |RᵀR − I| = 1.2e-6 on KITTI full after 200 normal iterations (median 2.7e-8), 1.5–2.4e-7 on ChArUco, and 0 before the first iteration. The frozen charts x̂ inherit this; the linearization is still G-solver's own code at x̂.

**F4. Messages vs. G_k.** The factor-to-variable message from GP factor f(k, k+1) to k is computed by `computeMessage`/`marginalize` (`factor_node_helper.h:8-27,41-75`). It is the factor's own block Λ^f_kk minus Λ^f_{k,k+1}(Λ^f_{k+1,k+1} + Λ_{k+1→f} + εI)⁻¹Λ^f_{k+1,k}, plus εI. The incoming precision Λ_{k+1→f} is the product of k+1's other messages plus εI (`variable_node.cpp:30-52`). At convergence on a chain this is the future Schur complement, so

  **G_k = H_kk − Λ^f_kk + (stored message f(k,k+1)→k)** (and likewise F_k with the earlier factor), up to O(ε).

- **Not "P_kk + message":** G-solver's factor message already contains the factor's own diagonal block Λ^f_kk, so adding it to H_kk double-counts it.
- **Verified** on a 600-pose GP chain (poses, unary priors and GP priors; a tree), frozen and run to convergence: relative error median 4.9e-12, max 1.0e-11 (G_k; F_k 1.1e-11). The absolute deviation is ≤ 7.7ε entrywise, which is the regularizers. The literal "H_kk + message" is off by 10.7 %.
- **Only on trees:** on the loopy stereo and ChArUco graphs the messages are not exact Schur complements.

**F5. Stereo rotation Jacobian.** There is one stereo factor type, `SE3PoseVelStereoPoint`, and one line:

```
J_proj.block<4, 3>(0, 3) =  2.0 * P_ext.block<4, 3>(0, 0) * skew(p_cam);   // se3_posevel_stereo_point.cpp:77
```

Every stereo experiment (single-robot, multi-robot, rolling shutter) builds its factors through `add_stereo_factor_nodes` (`factor_graph_builder.h:392`), so the factor 2 is present in every stereo factor. The Python bindings do not expose the stereo factor. The non-stereo point factor `SE3PoseVelPoint` uses `-skew(h)` with no factor 2 (`se3_posevel_point.cpp:31`). The line was not changed.

**Confirmed numerically on exported models:** over 2,000 KITTI stereo factors (prefix), the rotation block of J is 2.000000 ± 1.1e-6 times the finite-difference derivative of the stereo projection under G-solver's right perturbation. The translation and landmark blocks match to 1.3e-6 (finite-difference error), and the velocity block is 0. KITTI full gives the same (2,000 factors: 2.000000 ± 1.1e-6).

## 6. What could not be done and why

- **Gate A's floor ≤ 1e-4 is not met at G-solver's settings on any dataset.** The causes are identified above, but the frozen GBP of the shipped solver does not reach x̄ to 1e-4. ChArUco reaches it with ε = 1e-7, which is a diagnostic change of the frozen mode, not of G-solver. The stereo models reach it only if the ~30 unresolvable far landmarks per 20 keyframes are held fixed; KITTI full does not reach it even then (1.1e-3 with 536 held), and no ε setting helps there.
- **Single-edge blocks have no exact model reference on loopy graphs.** Their GBP belief is their factor's message computed from the neighbour's GBP cavity, which depends on GBP's approximate precisions. A model-marginal formula is exact only on trees, and on the 25 % islam snapshot it missed by 2.2e-2, so that metric was dropped. They are reported only as an offset from x̄.
- **The stereo normal runs never met the convergence rule** within 200 iterations. The KITTI full snapshot "at convergence of the normal run" is therefore at the cap.
- **Uninformed variables are left out of the model.** These are landmarks whose every stereo factor fails G-solver's in-image test at x̂ (6 in the prefix, 257 in KITTI full), and the velocity of the only pose at islam iteration 1. Their rows of H are exactly zero, so A0 fails there; they are excluded from H and from the curves, and counted.
- **F4 holds exactly only on trees**, and **F3's affine form only to first order** (above).
- **The floor rule was revised once, before any Gate A result was recorded.** The first version declared a floor whenever the tail stayed within 2× of its median, and on the first test export (ChArUco BA, 3,000 iterations) it labelled a still-decreasing curve a floor. The rule now requires the tail slope to be under 10 % of the main-phase slope (`certify/phaseA.py` docstring).
- **Exports are not in git** (3.8 GB).
- **KITTI full's frozen run was stopped at 3,740 of 5,000 planned iterations** (host overloaded by other jobs). Its floor was already flat from t = 115.

## 7. File list

- **Library (opt-in, default path bit-identical):**
  - `gsolver/include/gsolver/core/frozen_mode.h`
  - `gbp_solver.{h,cpp}` (`freeze`, `dynamic_schedule_`)
  - `edge.h` (`tau_`, `frozen_`)
  - `variable_node{_base.h,.h,.cpp}`, `factor_node{_base.h,.h,.cpp}`
  - `helpers/factor_node_helper.h`, `helpers/factor_node_edge_tuple_helper.h`
  - `CMakeLists.txt` (`GSOLVER_BUILD_CERTIFY`)
- **Drivers:** `certify/cpp/{certify_common.h, certify_charuco.cpp, certify_stereo.cpp, certify_chain.cpp, CMakeLists.txt}`.
- **Python:** `certify/{io.py, phaseA.py, pack.py, interp.py, facts.py}`.
- **Documentation:** `certify/FREEZE_MODE.md`.
- **Scripts:** `scripts/{phaseA_equivalence.sh, phaseA_run_charuco.sh, phaseA_run_stereo.sh, phaseA_validate_all.py}`; `scripts/build_gsolver.sh` now enables the certify drivers.
- **Results:** `results/phaseA/{gateA_summary.csv, eps_diagnostics.csv, unfrozen_summary.csv, f4_chain.json, equivalence.txt, validate_all.log, validate_kitti06_full.log}`, plus `results/phaseA/equivalence/` (compared trajectories).
- **Exports (A5; not in git):** `/home/os717/gsolver/exports/`, 3.8 GB, on this machine (charuco_ba 403 MB, charuco_islam 151 MB, kitti06_prefix 317 MB, kitti06_full 2.7 GB, chain_helix 169 MB, equivalence_runs 44 MB). There is no other download location.
  - One directory per snapshot: `charuco_islam/{frac025,frac050,frac100}`, `charuco_ba/{init,end}`, `kitti06_prefix/{init,conv}`, `kitti06_full/conv`, the unfrozen exports `*/unfrozen_run/unfrozen/it{001..100}`, and the regularizer diagnostics `*_eps*`, `*_site_*`.
  - The regularizer diagnostics keep only the raw export, curves and `gateA.json`: their model is identical to their base snapshot's; only the frozen-mode ε differs.
  - Each snapshot and unfrozen directory holds `model.npz` (H, b, and the meas / gauge / single_edge parts, x̄), `structure.json` (block partition, pose indices, per-landmark observation pose indices, GP Q_C and every Δt, gauge factors, single-edge and uninformed blocks, charts), `states.npz` (z at iterations 0, 1, 2, 5, 10, 20, 50, 100, the floor, and the last), and the factor list `f_<type>_{vars,J,e,Sigma,msg_tau,ids}` (Λ_f = Σ_f⁻¹, r_f = −e_f).
  - Also: `gateA.json` / `gateA_curves.csv`, and for stereo `a3.json`, `far_landmarks.json`, `f3.json`, `f5.json`.
- **Commit:** the commit that adds this file on branch `certify`.

**A7 decisions recorded for later phases (no work done):**

- **ChArUco:** S = all 24 markers jointly (144 dof); hub window = the whole graph.
- **KITTI 06 full:** no hub decomposition. C3 only, K ∈ {1, 2} on all blocks and K = 3 on a random sample of 200 poses, with landmarks eliminated inside each patch (same exact solve).
- **kitti06-prefix:** full chain treatment, C4 with w = 3, and any residual hubs handled jointly.
- **Consequences of this phase that those decisions will meet:**
  - The prefix is only 20 keyframes (7 supernodes at w = 3).
  - Single-edge and uninformed blocks need an explicit convention in C3/C4.
  - The far landmarks bound how small any certificate can be in their directions.
