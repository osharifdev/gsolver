# Brief: G-solver Phase D — Certified Stop-or-Relinearize on the Live Solver

Date: 2026-09-27. Branch: `certify`. Read this file in full, then `PROOF_NOTE.md` (Parts I–II) and `STOP_OR_RELINEARIZE.md` (§1–§3) in the repository root. Every theorem in those files is proved; treat them as correct. If an invariant fails, the implementation is at fault: stop, find it, report it. Never relax a tolerance or a threshold.

## 0. What we are trying to learn, and the compute constraint

The goal is a G-solver that decides, per pose and from its own GBP state, when further message passing is no longer useful, with a certified bound on the remaining move, and that sleeps those poses and wakes them on new information. The linear certificate (Theorem A) is already validated on frozen models. What is **not yet known**, in order of value:

1. **L1.** Does G-solver keep its accuracy when it relinearizes on a threshold instead of every round?
2. **L2.** On a live, threshold-relinearizing run, are the Theorem C decisions valid every round, and what are their dynamics: how many poses stop, how often a stopped pose is woken by neighbours, how often "relinearize-now" fires?
3. **L3.** How much computation does the certified schedule save against the default schedule and against a message-change heuristic, at equal accuracy?
4. **L4.** How often does a pose's patch need refactorizing (the cost driver)?
5. **L5.** Is the nonlinear layer going to be tight? A cheap probe answers this before any curvature theory is implemented: the nonlinear residual after the exact local step.

**Compute rule.** No unattended run longer than **six minutes** before a checkpoint is written and a gate is evaluated. Every run that could take longer is preceded by a one-tenth-size version. There are **no long reference runs** in this brief: every oracle needed here is a per-round direct solve of the *current linearization*, which is exactly what Theorem C is certified against. Measured costs on this host, for planning: ChArUco islam ≈ 1 ms per round; KITTI prefix ≈ 1.2 s per round with the dynamic schedule; Phase C decision-driven patches ≈ 14 ms per pose per round in double precision (about 2.3 s per round for the 161-pose prefix); a direct sparse solve of the reduced pose model < 1 s. Stay within these.

**Precision.** The live sidecar runs in ordinary double precision. Extended-precision validation (the Phase B/C machinery) is applied only to three sampled rounds per run, chosen in advance (rounds 1, 10, and the last), to confirm the double-precision certificates.

## 1. Definitions used below (from STOP_OR_RELINEARIZE.md §1–§3)

For the current linearization: linearization points `x_lin`, assembled system `H x = b` in the tangent charts at `x_lin`, exact solution `xbar = H^{-1} b`, current iterate `x` (tangent coordinates relative to `x_lin`). Reduced pose model: landmarks eliminated exactly as in Phase C (`certify.model.reduce_blocks`). Per pose `i`:

- `d_i = ||x_i||_W` — displacement since linearization, in the solver's belief-movement metric W (see WP0).
- `E_i, L_i` — the two-sided certificate of Theorem A on the reduced pose model with decision-driven patches (parameters r = 0.3, θ = 0.02, cap 250, 40 growth steps); converted to W by `κ_i = ||W_i^{1/2} D_i^{-1/2}||`.
- `S_lo_i` — Lemma 2.1 patch lower bound on the yardstick (from H if WP0 finds the factor of 2 is a convention; from the true-Jacobian Fisher information per Theorem T0 otherwise), with `q_12(0.05)`.
- **Decision** (Theorem C), with pre-registered `η = 0.1`:
  - STOP if `d_i + κ_i E_i ≤ δ_i` and `E_i ≤ η S_lo_i`;
  - RELINEARIZE-NOW if `d_i − κ_i E_i ≥ δ_i`;
  - CONTINUE otherwise.
- **Oracle per round:** `e_i = x_i − xbar_i` from a direct sparse solve of the current linearization; the oracle decision uses `||e_i||` in place of `E_i` and `L_i`.

Invariants checked every round: `L_i ≤ ||e_i||_{D_i} ≤ E_i` for every pose; the patch lower bound ⪯ the exact marginal covariance on the three sampled rounds.

## 2. Work packages

### WP0 — Code read: the rotation chart (no compute, report first)

Determine whether G-solver's rotation ⊞ is the exponential map (`R(δ) ≈ I + [δ]_x`) or the unit-quaternion vector part (`R(δq) ≈ I + 2[δq]_x`). Cite file and line. Consequence: quaternion ⇒ the stereo Jacobian's factor of 2 is correct and `J̃ = J` for stereo; exponential ⇒ it is a modification and `S_lo` must use the true Jacobian (Theorem T0). Also state which chart and metric the solver uses when it measures belief movement, because δ is stated in it. **Write `reports/phaseD_wp0.md` and stop for the decision point.**

### WP1 — Threshold relinearization option (seconds on ChArUco, ≤ 4 min per prefix run)

Add solver options `relinearize_threshold δ` and `relinearize_every R`: a factor relinearizes only when a connected belief has moved more than δ from that factor's linearization point, at most once per R rounds. Default off; the default code path must stay byte-identical (rerun `scripts/phaseA_equivalence.sh`; with δ = 0, R = 1 the option must also reproduce the shipped output).

Runs, staged:
1. ChArUco islam, δ ∈ {0.25, 0.5, 1, 2} (D-metric units, or the W equivalent per WP0), R = 10, to the default round budget: log rounds per linearization and final pose error against the default schedule. Seconds.
2. **First 40 keyframes** of the KITTI prefix, same grid, 100 rounds each. About 30 s per run.
3. Full prefix, **only the chosen δ**, 200 rounds. About 4 min.

Fix δ as the smallest grid value whose final error is within 5% of the default schedule's on runs 1 and 2. **Gate:** equivalence holds; such a δ exists. Report `reports/phaseD_wp1.md`.

### WP2 — Theorem C on the live solver via the sidecar (≤ 6 min per run)

Implement the sidecar loop: each round the solver exports the current linearization (the Phase A unfrozen hook: H, b, x, linearization points, block partition); the sidecar reduces landmarks, computes `d_i, E_i, L_i, S_lo_i` and the decision for every pose, and writes back an active-set mask and a relinearize-now set; the solver freezes sleeping poses (no belief or message updates; last messages stay in place), honours the relinearize-now set, and refactorizes a pose's patch only when a factor inside it relinearizes. Sleeping poses run the cheap recheck of Proof Note Theorem 5.4 on their accumulated patch-residual change and wake when it exceeds their slack, or when a patch factor relinearizes.

Runs, staged, each with a per-round oracle solve and the invariants:
1. ChArUco islam, full sequence, incremental. Seconds per hundred rounds.
2. First 40 keyframes of the prefix, 100 rounds. About 2 min including the sidecar.
3. Full prefix, 200 rounds, checkpoint every 50 rounds. About 12 min total, so run it as **two 100-round segments resumed from checkpoint**, gate-checking between them.

Log per round per pose: `d_i, E_i, L_i, ||e_i||, S_lo_i, decision, asleep, woke_this_round, woke_reason (recheck | relinearization | new_factor), refactorized_this_round, patch_size`. **Gate:** 100% validity of `L_i ≤ ||e_i|| ≤ E_i` on every round; certified stop fraction within 5 points of the oracle stop fraction at every round; refactorizations per pose per 100 rounds below 10. Report `reports/phaseD_wp2.md` with the three learnings L2 and L4 stated plainly, and stop for the decision point.

### WP3 — Baselines and savings (parallel with WP2 stage 3; ≤ 4 min per run)

Schedules on ChArUco islam and the 40-keyframe prefix first, full prefix last:
- (a) default: per-round relinearization, fixed round budget;
- (b) threshold relinearization (WP1 δ, R = 10), fixed round budget;
- (c) threshold relinearization + message-change heuristic: a pose sleeps when its belief change in the D-metric falls below τ, with τ chosen as the value that matches (a)'s median final error on ChArUco, stated in the report;
- (d) threshold relinearization + Theorem C (WP2).

Metrics: cumulative node updates and message computations, relinearizations, refactorizations, wall time excluding the sidecar, final error per pose against the last round's direct solve, ATE where ground truth exists, fraction asleep per round. **Gate:** (d) uses at least 2× fewer node updates than (a) and at least 1.3× fewer than (c) at equal or lower final error. Also search the logs for the **early-stop failure case**: any pose that (c) puts to sleep while its certified lower bound `L_i` exceeds `η S_lo_i`; report it with numbers if it exists. Report `reports/phaseD_wp3.md`.

### WP4 — Loop-closure wake (seconds on ChArUco; prefix optional, ≤ 4 min)

On ChArUco islam at the end of a (d) run with all poses asleep, add one long-range factor (a re-observation of a marker from a late frame, or a synthetic pose–pose constraint between far-apart poses with realistic information). Log the wake set, each woken pose's graph distance from the new factor, recertification cost, and final error against a full direct re-solve. Repeat once on the 40-keyframe prefix with a synthetic constraint. **Gate:** the wake set is bounded and decays with distance; final error at recertification is within the certificate. Report inside `reports/phaseD_wp3.md`.

### WP5a — The nonlinear residual probe (minutes; no curvature theory)

This decides whether the nonlinear layer (Theorem T1) is worth implementing, at almost no cost. At every pose that STOPs under Theorem C in the WP2 runs:
1. Form the candidate step `z1_i = −[H_PP^{-1} F_P(0)]_i` (the patch step; it is the exact local solve already computed).
2. Evaluate the **nonlinear** estimating-equation residual at the stepped point, `F(z1)`, by re-evaluating every factor at `x_lin ⊞ z1` through the Python bindings (each factor's error and Jacobian at that point; this is what the solver does each round anyway). No curvature constants are needed.
3. Compute `s_i^NL = ||[H_PP^{-1} F_P(z1)]_i||_{D_i}` and log the ratio `s_i^NL / E_i` and `s_i^NL / (η S_lo_i)`.

Interpretation: `s_i^NL` is the residual term of T1 and the dominant part of its remainder at a δ-consistent stop. If it is below one tenth of `η S_lo_i` for most stopped poses, the nonlinear layer will be tight and T1 is worth implementing; if it is comparable to `E_i`, report which factor type dominates the residual (GP or stereo) by per-factor attribution. Report `reports/phaseD_wp5a.md`. **Do not derive or implement curvature constants in this brief.**

## 3. Order and stop points

WP0 → stop. WP1 → stop. WP2 stages 1–2 → report → WP2 stage 3 with WP3 and WP4 on ChArUco in parallel → stop. WP5a on the WP2 logs → stop. Nothing beyond this brief without instruction.

## 4. Logging schema

`results/phaseD/<schedule>_<dataset>/rounds.csv` with the per-round-per-pose columns of WP2, plus `summary.json` per run with the gate numbers, and `checkpoints/` every 50 rounds. `results/phaseD/wp5a_probe.csv` with `dataset, round, pose, E_i, S_lo_i, s_NL, s_NL_over_E, s_NL_over_etaSlo, dominant_factor_type`.

## 5. Report format

Five-line summary; gate status with numbers; tables of medians and 95th percentiles; the single worst case for every invariant and ratio, with its location; what could not be done, plainly, with no workaround presented as success; file list and commit hash. Compute used per run, in minutes.
