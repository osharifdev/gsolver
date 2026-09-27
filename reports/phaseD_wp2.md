# Phase D, WP2: Theorem C on the live solver

Branch `certify`, base commit `faa90ce`. Date 2026-09-27. Setting fixed by WP1b: δ = 0.003 in the physical metric W = I, R = 1. The WP2 amendment applies:
- STOP and CONTINUE only;
- a sleeping pose's patch is refactorized when a factor inside it relinearizes, and the pose is kept asleep or woken;
- certificates only for poses with d_i ≤ δ;
- S_lo from the true-Jacobian Fisher information, η = 0.1, δ_s = 0.05;
- operation counts for the certificate work.

The protocol and the certificate are in `certify/SIDECAR.md`.

## 1. Five-line summary

1. **The sidecar loop runs on the live, threshold-relinearizing solver.** The pieces:
   - each round the driver exports the current linearization (in RAM);
   - the Python sidecar builds decision D2's exactly reduced pose model and the true-Jacobian Fisher information (0.6–1.3 s on the prefix; the Phase C pipeline takes 61 s);
   - it certifies every pose and decides STOP or CONTINUE, and the solver sleeps the stopped poses (no belief update and no messages to or from them).
   - **RELINEARIZE-NOW is not implemented:** at R = 1 the solver relinearizes as soon as d_i > δ, so it cannot fire.
   - The default path is byte-identical (12 of 12).
2. **Validity.**
   - KITTI-40 and the full prefix: 0 violations of L_i ≤ ‖e_i‖ ≤ E_i in 36,200 pose-round checks (double precision). The extended-precision re-check of rounds 1, 10 and last also finds 0.
   - **ChArUco fails at the 1e-9 tolerance, by double-precision rounding.** The per-round check flags round 7 (a lower bound 1.24e-9 above the double oracle). The extended re-check of rounds 7 and 10 finds the true error above the double certificate by 1.46e-9 and 1.06e-9 relative (one pose each). The double Loewner check at round 103 is −1.70e-9.
   - All are certificates that are exact in exact arithmetic (whole-graph patches) on the 10¹²-anchored model, whose double-precision oracle is itself off by 1.37e-9.
3. **Stop fractions.**
   - Full prefix: certified 34.2 % against the oracle's 42.9 % at round 200. It is within 5 points through round 192 and 5.6–9.3 points behind in rounds 193–200, **so the gate fails there**.
   - KITTI-40: 10 % against 12.5 %, with a largest gap of 5.0 points, so it passes at the bound.
   - ChArUco: 0 against 0. Nothing can stop at the islam budget, because the remaining move in physical units (κ_i E_i, median 0.51) is 170× δ.
4. **L2 and L4:**
   - The δ condition is what binds. It holds in 6 % (KITTI-40) and 24 % (prefix) of the eligible pose-rounds, against 72 % and 84 % for the η condition.
   - Wakes: 15 on the prefix, all because a factor in the patch relinearized; 0 on the recheck; 0 from new factors.
   - At R = 1 every patch is refactorized every round: **99.8–100 per pose per 100 rounds**, against the brief's 10. So the Theorem 5.4 cheap recheck never applied.
   - The certificate costs 8.3e10 flops per round on the prefix. Sleeping 34 % of the poses cut GBP's messages by 17 %.
5. **Cause of the late prefix gap:** a growth rule, not validity.
   - At round 197, 13 of the 15 poses the oracle stops but the certificate does not have loose certificates (E/e 6.7–61, t = 24.7–27.6 s). Their decision-driven growth exited through its "s_i ≥ threshold ⇒ continue" rule, inherited from Phase C: with δ's threshold of about 0.007–0.01 D-units, a 45–51-pose patch's local term (about 0.2) exceeds it while the true error (about 0.005) does not.
   - The other 2 have tight certificates that miss the δ margin by less than 1e-5.
   - **Side finding (L1):** at the shipped schedule's round-500 state on KITTI-40, 45 % of the poses have a remaining Gauss–Newton move above η S_lo,i.

## 2. Gate status

| Criterion (brief WP2 as amended) | ChArUco islam (103 rounds) | KITTI-40 (100 rounds) | Full prefix (200 rounds, four checkpointed segments) |
|---|---|---|---|
| 100 % validity of L_i ≤ ‖e_i‖_{D_i} ≤ E_i, every round (1e-9 relative) | **fail**: double 1 lower flag (round 7, 1.24e-9); extended upper 1.46e-9 (round 7) and 1.06e-9 (round 10), one pose each; rounding on the 10¹²-anchored model (§4) | **pass**: 0 of 4,000 (double); extended rounds 1, 10, 100: 0 (worst +1.35e-11) | **pass**: 0 of 32,200 (double); extended rounds 1, 10, 200: 0 (worst −1.8e-3) |
| Patch lower bound ⪯ exact marginal on rounds 1, 10, last | round 1: A0 fails (no certificate). Round 10: −9.3e-10 (pass). Round 103: **−1.70e-9 (fail)** | pass: +2.8e-13 and −3.2e-14 | pass: +3.4e-14 and −2.3e-14 |
| Certified stop fraction within 5 points of the oracle's, every round | pass (0 and 0) | pass (largest gap 5.0 points) | **fail**: 8 of 200 rounds (193–200), largest 9.3 points (round 197) |
| Refactorizations per pose per 100 rounds (bound 10; report the number) | 100.0 | 100.0 | 99.8 |

The prefix checkpoints (rounds 50, 100, 150) each evaluated the validity gate, fixed before the run as "continue unless a violation": 0 every time, so the run continued. The largest stop-fraction gap so far was 0.0, 1.2 and 1.9 points at those checkpoints; it passed 5 only after round 192.

## 3. Tables

### 3.1 Runs

| | ChArUco islam | KITTI-40 | Full prefix |
|---|---|---|---|
| Rounds; poses at the end | 103; 94 | 100; 40 | 200; 161 |
| Round where A0 fails (singular linearization) | 1 (one pose, no velocity information) | none | none |
| Certified / oracle stop at rounds 50, 100, 150, 200 | 0 / 0 throughout | 5.0 / 7.5 %, 10.0 / 12.5 % | 0 / 0, 23.6 / 24.2, 30.4 / 31.7, 34.2 / 42.9 % |
| Eligible pose-rounds (not relinearized, d_i ≤ δ) | 3,297 | 3,106 | 23,431 |
| E/e over eligible pose-rounds: median / 95 % / max | 1.000 / 2.84 / 37.9 | 2.40 / 17.7 / 52.0 | 1.64 / 8.31 / 65.1 |
| Share of eligible pose-rounds with the δ margin > 0 / the η margin > 0 | 0 % / 10.3 % | 6.2 % / 72.1 % | 23.8 % / 83.9 % |
| Pose-rounds asleep; kept by the Theorem 5.4 recheck; kept after refactorization | 0; 0; 0 | 194; 0; 190 | 5,584; 0; 5,514 |
| Wakes: recheck / relinearization / new factor | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 15 / 0 |
| Refactorizations per pose per 100 rounds | 100.0 | 100.0 | 99.8 |
| Certificate flops per round, median (max) | 2.1e9 (1.6e10) | 6.4e8 (1.1e9) | 8.3e10 (1.0e11) |
| Factorizations per round, median (plus shared) | 29 (91) | 26 (14) | 156 (32) |
| Sidecar seconds per round, median (Python, including the model build and the oracle) | 0.25 | 0.35 | 4.48 |
| GBP seconds per round (C++, 16 threads) | 0.001–0.003 | 0.17 | 0.31 |
| GBP messages per round, first → last | 23 → 2,842 (the graph grows) | 44,707 → 42,709 | 198,317 → 164,747 (−17 %) |
| Variable updates per round, first → last | 8 → 118 | 4,719 → 4,715 | 18,098 → 18,043 |

- **Certificate flops** count the factorizations (n³/3), the solves (2n² per right-hand side), the leak products, the recursion sweeps and the yardstick factorizations.
- **GBP's own flop count was not instrumented;** its work is given as messages and updates.
- **Sleeping only poses saves little GBP work.** Landmarks are 99 % of the variables, but each pose has hundreds of edges.

### 3.2 Which STOP condition binds

STOP needs d_i + κ_i E_i ≤ δ (the δ margin) and E_i ≤ η S_lo,i (the η margin).

- **The δ condition binds everywhere.** κ_i = λ_min(D_i)^{-1/2} has median 0.37 (ChArUco) and 0.17 (stereo). So the δ condition asks for E_i ≲ (0.003 − d_i)/κ_i ≈ 0.008–0.018 D-units, whereas η S_lo ≈ 0.4–0.8.
- **On ChArUco nothing can stop at the islam budget.** κ_i E_i has median 0.51 m/rad-scale: the solve is far from converged when the budget ends. The oracle stops nothing either, since 0 % of pose-rounds satisfy the δ condition even with the exact error.

### 3.3 Extended-precision validation of the double-precision sidecar (rounds 1, 10, last; ChArUco also 7)

Margins are (true − bound)/bound: negative means the certificate holds with room.

| Run, round | Fast S vs extended assembly | Worst upper margin | Worst lower margin | Double oracle vs extended | S_lo ≤ S_exact |
|---|---|---|---|---|---|
| ChArUco 1 | – | A0 fails | | | |
| ChArUco 7 | 1.2e-16 | **+1.46e-9** (fail) | +7.8e-10 | 1.37e-9 | 1 of 1 |
| ChArUco 10 | 1.2e-16 | **+1.06e-9** (fail) | −4.5e-10 | 9.0e-10 | 1 of 1 |
| ChArUco 103 | 1.2e-16 | +8.1e-10 | −5.0e-10 | 8.1e-10 | 84 of 84 |
| KITTI-40 1 / 10 / 100 | 1.1e-14 / 1.1e-14 / 2.1e-14 | −6.1e-2 / −2.3e-2 / +1.35e-11 | −4.9e-2 / −2.2e-2 / +3.9e-12 | 2.2e-13 / 7.9e-14 / 1.4e-11 | 40 / – / 38 |
| Prefix 1 / 10 / 200 | 1.5e-14 / 1.2e-14 / 2.4e-14 | −6.8e-2 / −8.0e-3 / −1.8e-3 | −5.3e-2 / −8.2e-3 / −1.9e-3 | 2.7e-13 / 1.1e-12 / 3.1e-11 | 161 / – / 137 |

### 3.4 Side computation (L1): the shipped schedule's fixed point, KITTI-40 round 500

Using the sidecar's conventions (reduced pose model, D_i = S_ii, S_lo on the base patch from the true-Jacobian Fisher information, η = 0.1):

| Remaining GN move, median / 95 % / max | S_lo median (S_exact) | Poses with move > η S_lo | move / (η S_lo), median / max |
|---|---|---|---|
| 0.363 / 0.939 / 1.889 | 4.115 (4.115) | **45 %** (18 of 40) | 0.86 / 4.85 (pose t = 8.003 s) |

**This is a self-consistency finding, not an accuracy claim.** By the certificate's own yardstick, the shipped schedule's stationary state would not be stopped for 45 % of the poses: its distance to the fixed point of its own estimating equation exceeds an η-fraction of their uncertainty. ATE does not distinguish the two fixed points (WP1b).

## 4. Worst cases

| Quantity | Worst | Where |
|---|---|---|
| Validity (double), ChArUco | L exceeds the double oracle by 1.24e-9 relative | round 7, pose t = 1.65 s. Its patch is the whole graph (7 poses), so L = E = ‖e‖ in exact arithmetic |
| Validity (extended), ChArUco | the true error exceeds E by 1.46e-9 (round 7) and 1.06e-9 (round 10) | one pose each, both whole-graph patches. The double oracle is itself off by 1.37e-9 and 9.0e-10 on those rounds: the double-precision floor κu on the 10¹²-anchored model |
| Loewner check, ChArUco | −1.70e-9 | round 103, whole-graph patch |
| Validity, KITTI-40 / prefix | +1.35e-11 / −1.8e-3 (extended) | KITTI-40 round 100, an exact certificate |
| Stop-fraction gap | 9.3 points | prefix round 197: oracle 67 poses, certified 52 |
| Loosest certificate that decided CONTINUE where the oracle stops | E/e = 61.5 | prefix round 197, pose t = 26.20 s; growth exit "continue" at 45 poses |
| Tightest miss of the δ margin | −6.9e-7 (E/e = 1.016) | prefix round 197, pose t = 6.96 s |
| Largest E/e (any eligible pose-round) | 65.1 (prefix), 52.0 (KITTI-40), 37.9 (ChArUco) | |
| Certificate cost | 1.0e11 flops in one round | prefix |
| Side computation, move / (η S_lo) | 4.85 | shipped KITTI-40 round 500, the last pose (t = 8.003 s) |

## 5. Implementation

- **Solver** (`gsolver/src/core/gbp_solver.cpp`, `gsolver/include/gsolver/graph/*`):
  - `applyThresholdRelinearization`, which relinearizes before the round's messages so the sidecar sees the linearization the messages use;
  - `setAsleep`, backed by the edges' `asleep_` flag. Factors skip their messages to sleeping variables, and sleeping variables neither update nor send. On the default path the flag is always false, and the path stays byte-identical (`results/phaseD/wp2/equivalence.txt`).
- **Drivers** (`certify/cpp/certify_common.h`, `RoundRunner::sidecarRound`): `--sidecar`, `--persist-rounds`, `--pause-every`. The work log adds variable updates and the number of sleeping poses.
- **Sidecar:**
  - `certify/sidecar_model.py`: the D2 model, reduced exactly to the poses, vectorized. It is checked against `from_export` + `reduce_blocks` to 1e-14 on every validated round.
  - `certify/sidecar.py`: the certificate engine.
  - `scripts/phaseD_sidecar.py`: the process loop.
  - `scripts/phaseD_wp2_validate.py`: the extended-precision validation.
  - `scripts/phaseD_wp2_selfconsistency.py`: the side computation.
- **Amendment choices, stated plainly:**
  1. "Certificate only for d_i ≤ δ" is applied to the expensive part: the yardstick, the decision-driven growth and the decision. **Every pose still gets a base-patch row**, because Theorem A's recursion is closed over all poses: an eligible pose's leak terms need its neighbours' certificates.
  2. d_i is the solver's own displacement at its relinearization test. A pose it relinearizes this round (d_i > δ) is not eligible.
  3. Identical patches within a round share one factorization: an implementation saving that leaves the certificates unchanged, checked byte-identical on ChArUco.

## 6. What could not be done, and why

- **Validity at 1e-9 on ChArUco is not achieved in double precision.** On the 10¹²-anchored ChArUco model, the double-precision certificate and the double-precision oracle are both accurate only to about 1.5e-9 relative. Where the certificate is exact (whole-graph patches), rounding decides the comparison at 1e-9.
  - The cause is the brief's precision design, not a formula. The certificate formulas are exact, the fast model agrees with the extended assembly to 1e-16, and the extended re-check finds the double oracle off by the same 1.4e-9.
  - I did not loosen the tolerance. Two remedies:
    - (a) run the ChArUco sidecar in extended precision (the Phase C pipeline, about 5 s per round there);
    - (b) add a certified rounding margin to the double-precision certificates (Part II's "unless rounding terms are added").
- **The prefix stop-fraction gap exceeds 5 points in rounds 193–200.** The growth rule's "continue" exit is the cause (§1 item 5). Fixing it is a change of the decision-driven growth, so I have not made it. A candidate: exit on "continue" only when s_i exceeds the threshold by more than the patch's leak bound; otherwise keep growing.
- **Refactorization bound (10 per pose per 100 rounds): not met, about 100.** At R = 1 with δ = 3 mm, some factor inside every patch relinearizes every round. Consequently the Theorem 5.4 cheap recheck never runs, and every sleeping pose is re-certified from a new factorization each round.
- **RELINEARIZE-NOW** was not implemented (R = 1).
- **GBP's own flops were not instrumented**, so the certificate-to-GBP cost ratio is not stated in flops. The sidecar is Python and GBP is C++, so the per-round seconds (4.5 against 0.31 on the prefix) are not a like-for-like cost either.

## 7. Decision needed

1. **ChArUco validity.** Choose remedy (a) extended-precision sidecar, (b) a certified rounding margin, or (c) record the ChArUco result as a double-precision limitation.
2. **The growth rule's "continue" exit.** Change it (a method change), or keep it and accept the late prefix gap.
3. **Cost.** At R = 1 the certificate is re-factorized every round for every pose. A larger R or δ would let patches persist, but WP1b fixed (0.003, 1) on accuracy grounds.

## 8. Compute used (minutes)

- **Builds:** 2.
- **ChArUco:**
  - development runs: 2.1 before shared factorizations; 0.55 after;
  - final runs: 0.67 each. The run was repeated once to persist round 7; it is deterministic, with identical logs.
- **KITTI-40:** one-tenth version 0.11; run 0.94.
- **Prefix:**
  - one-tenth version 1.4;
  - run 16.7 wall, including the checkpoint waits;
  - segments 2.7 / 3.9 / 4.6 / 4.3 of computation, each under 6.
- **Extended validation:** KITTI-40 about 1; prefix about 14 (three exports, about 4.7 each); ChArUco about 1.
- **Side computation:** 0.3.
- **Equivalence:** 4.8.

## 9. Files

- Protocol and certificate: `certify/SIDECAR.md`.
- Code: `certify/sidecar_model.py`, `certify/sidecar.py`, `scripts/phaseD_sidecar.py`, `scripts/phaseD_wp2_validate.py`, `scripts/phaseD_wp2_selfconsistency.py`; solver and drivers as in §5.
- Results, `results/phaseD/wp2/{charuco,kf40,prefix}/`:
  - `rounds.csv.gz` (per round and pose: d, E, L, ‖e‖, S_lo, S_exact, κ, both STOP margins, decision, asleep, woke and reason, refactorized, patch sizes, growth exit, oracle stop, validity);
  - `summary.csv` (per round, with operation counts and timings) and `summary.json` (gate numbers);
  - `extended_validation.json`, `work.csv`, `relin_rounds.csv`, `relin_counts.csv`, `traj.tum`, `sidecar.log`.
- Side computation: `results/phaseD/wp2/selfconsistency_kf40_shipped_r500.json`. Equivalence: `results/phaseD/wp2/equivalence.txt`.
- Exports (git-ignored): `exports/phaseD/wp2/`, with the persisted rounds and the final states.
- Commit: the commit that adds this file.
