# Stage 1: certified termination on a corrected G-solver

Branch `certify`, base commit `e57bc70`. Date 2026-09-28.

Brief: `BRIEF_stage1_2.md`, §0–§2. The file is not in the repository root; I worked from the text in your message.

**Setting:**
- threshold relinearization at δ = 0.003 in the physical metric, R = 1, per-variable linearization points (WP1b);
- the statistical stop rule only: E_i ≤ ηS_lo,i, with η = 0.1;
- S_lo from the true-Jacobian Fisher information, δ_s = 0.05, q₁₂;
- growth whose only target is the statistical decision.

Every run was done on both variants: **shipped** (G-solver's Jacobians) and **corrected** (`--true-jacobians`).

## 1. Five-line summary

1. **The corrected variant is implemented and validated.**
   - The stereo rotation block drops its factor 2.
   - The GP pose error's derivative with respect to the first knot's velocity is now analytic, for the decoupled Exp/Log the solver uses.
   - Central differences agree to ≤ 2.1e-6 on every stereo and GP block, across all six datasets. The worst block anywhere is 9.4e-6: the ChArUco pose–marker factor, which is unmodified and within the 1e-5 requirement.
   - The default path is byte-identical (12 of 12).
   - The correction speeds convergence a lot. The median remaining Gauss–Newton move after 100 rounds, corrected against shipped:
     - 06: 0.032 against 0.195;
     - 03: 8.4 against 28;
     - 04: 1.17 against 4.8;
     - 10: 0.082 against 4.9.
2. **Gate A fails on both variants: 0 of 4 KITTI segments pass.**
   - Corrected, the certified stops come at rounds 130, 250, 170 and 50 (06, 03, 04, 10). Only sequence 10 stops by round 60, and there its median remaining move is 5.5× the fixed budget's (0.45 against 0.082). η = 0.1 of the statistical radius is coarser than 100 rounds reach on that segment.
   - On the other three, the stop waits for the slowest pose. It costs 1.4–2.6× the 100-round budget and leaves 0.04–0.88× its median move.
   - Shipped: stops at 150, none, none, 240.
3. **Validity holds everywhere, and the certificate decides exactly as the oracle.**
   - 0 violations of L_i ≤ ‖e_i‖ ≤ E_i in 19,260 stereo pose-evaluations (KITTI segments and prefix) and 18,590 on ChArUco.
   - The certified pass/fail equals the oracle's in every pose-evaluation.
   - The only exceedance is ChArUco's Loewner check at −5.4e-9 (rounding level on the 10¹²-anchored model, as in WP2).
4. **What a certified stop means depends on the variant.**
   - Corrected: at every certified stop, 0 % of poses have a relinearized Gauss–Newton move above ηS_lo. At the fixed budget, 5 %, 97.5 %, 90 % and 0 % are still statistically unfinished (06, 03, 04, 10).
   - Shipped: at its stops, 15 % (06) and 25 % (10) of poses exceed ηS_lo once relinearized.
   - The change-based criterion with τ = 1e-2 fires on all four corrected segments, but late (rounds 142–265). τ = 1e-4 never fires. On the shipped segments only 06 fires (τ = 1e-2 at 166, 1e-3 at 263).
5. **§2.5, the never-finishes result with the corrected yardstick.**
   - At the shipped schedule's round-500 state on KITTI 06, 45 % of poses have a remaining move above 0.1 S_lo (90 % at η = 0.05, 12.5 % at η = 0.2, 0 at η = 0.5).
   - The change-based criterion does not accept that state either: the largest per-round belief change is still 4.6 D-units at round 500 (a landmark), and it never falls below 1e-2 in 500 rounds.
   - ChArUco reaches no certified stop by round 1,030 on either variant (78 % of poses pass).
   - The 161-keyframe prefix confirms the KITTI pattern: corrected stop at round 130; shipped none by 300 (91 % pass).

## 2. Gate status

### Gate A (pre-registered, §2.6)

Stage 1 passes if, on at least three of the four KITTI segments on the corrected variant, all three hold:
- the certified stop comes at or before round 60;
- its median remaining Gauss–Newton move is at most 1.05× the fixed budget's;
- its total cost (GBP plus certificate flops, as implemented) is at most 0.8× the fixed budget's.

| Segment | Variant | Certified stop | ≤ 60 | Median move ÷ budget's (≤ 1.05) | Cost ÷ budget's (≤ 0.8) | Cost ÷ budget's, belief-minus-incoming baseline | Passes |
|---|---|---|---|---|---|---|---|
| 06 | corrected | 130 | no | 0.875 | 1.36 | 1.69 | no |
| 03 | corrected | 250 | no | 0.042 | 2.62 | 3.51 | no |
| 04 | corrected | 170 | no | 0.307 | 1.78 | 2.45 | no |
| 10 | corrected | **50** | yes | **5.54** | 0.53 | 0.73 | no |
| 06 | shipped | 150 | no | 0.886 | 1.57 | 1.99 | no |
| 03 | shipped | none by 300 | no | – | – | – | no |
| 04 | shipped | none by 300 | no | – | – | – | no |
| 10 | shipped | 240 | no | 0.072 | 2.51 | 3.31 | no |

**Gate A: fail.** 0 of 4 segments pass on the corrected variant, and 0 of 4 on the shipped. It would also fail against the belief-minus-incoming baseline, which is not part of the gate.

### Invariants

| Invariant | Result |
|---|---|
| L_i ≤ ‖e_i‖_{D_i} ≤ E_i every evaluation round, 1e-9 relative (stereo) | **0 violations** in 19,260 pose-evaluations. Worst margins: +2.2e-14 (shipped KITTI 10), within tolerance |
| Same, ChArUco (rounding-level exceedances reported, not stopped) | **0 exceedances** at 1e-9 in 18,590 pose-evaluations. Worst +2.2e-10 |
| Patch lower bound ⪯ exact marginal at the first, middle and last evaluation | stereo: worst −1.2e-13 (pass). **ChArUco: −5.4e-9 (corrected) and −3.5e-9 (shipped), above 1e-9**: rounding level on the 10¹²-anchored model, as in WP2 |
| Corrected Jacobians against central differences, every block, 1e-5 relative | **pass.** Worst 9.35e-6 (ChArUco pose–marker marker-rotation block, unmodified; 9.5e-6 on shipped). Stereo and GP blocks ≤ 2.1e-6 (`results/stage1/fdcheck.json`) |
| Default path byte-identical | **12 of 12** (`results/stage1/equivalence.txt`) |
| Checkpoint gates (ChArUco and prefix segments) | all "continue": 0 exceedances at every checkpoint |

## 3. Tables

### 3.1 Stops: round, accuracy, cost

"Move" is the remaining Gauss–Newton move per pose at the stop state (WP1's definition, the solver's own Jacobian), median and p95. "True-J move" uses the true Jacobian. ATE is the SE(3)-aligned keyframe ATE in metres. Costs are GBP flops (as implemented) plus certificate flops at the evaluation rounds, or the change criterion's flops. "Alt." replaces GBP's variable-to-factor messages with the belief-minus-incoming arithmetic.

| Dataset | Variant | Stop | Round | Move: median / p95 | True-J move median | ATE | Total flops | vs fixed | Total, alt. |
|---|---|---|---|---|---|---|---|---|---|
| KITTI 06 (40) | corrected | certified | 130 | 0.0279 / 0.334 | 0.0279 | 0.0820 | 2.44e11 | 1.36× | 4.32e10 |
| | | fixed budget | 100 | 0.0319 / 0.651 | 0.0319 | 0.0823 | 1.80e11 | 1.00× | 2.55e10 |
| | | change τ = 1e-2 / 1e-3 / 1e-4 | 142 / 237 / – | 0.0265 / 0.261; 0.0222 / 0.087 | | 0.0819; 0.0817 | 2.56e11; 4.27e11 | 1.42×; 2.37× | |
| KITTI 06 (40) | shipped | certified | 150 | 0.173 / 0.633 | 0.336 | 0.0817 | 2.82e11 | 1.57× | 5.07e10 |
| | | fixed budget | 100 | 0.195 / 1.09 | 0.448 | 0.0825 | 1.80e11 | 1.00× | 2.55e10 |
| | | change τ = 1e-2 / 1e-3 / 1e-4 | 166 / 263 / – | 0.170 / 0.633; 0.113 / 0.633 | | 0.0816; 0.0814 | 2.99e11; 4.74e11 | 1.66×; 2.63× | |
| KITTI 03 (40) | corrected | certified | 250 | 0.357 / 0.591 | 0.357 | 0.2474 | 6.74e11 | 2.62× | 1.07e11 |
| | | fixed budget | 100 | 8.40 / 16.4 | 8.40 | 0.2454 | 2.57e11 | 1.00× | 3.06e10 |
| | | change τ = 1e-2 / 1e-3 | 265 / – | 0.260 / 0.430 | | 0.2475 | 6.82e11 | 2.65× | |
| KITTI 03 (40) | shipped | certified | none by 300 | | | | | | |
| | | fixed budget | 100 | 28.2 / 196 | 34.6 | 0.2267 | 2.57e11 | 1.00× | 3.06e10 |
| | | change | none by 300 | | | | | | |
| KITTI 04 (40) | corrected | certified | 170 | 0.359 / 0.582 | 0.359 | 0.1331 | 5.54e11 | 1.78× | 8.27e10 |
| | | fixed budget | 100 | 1.17 / 1.80 | 1.17 | 0.1328 | 3.11e11 | 1.00× | 3.38e10 |
| | | change τ = 1e-2 / 1e-3 | 168 / – | 0.370 / 0.599 | | 0.1331 | 5.23e11 | 1.68× | |
| KITTI 04 (40) | shipped | certified | none by 300 | | | | | | |
| | | fixed budget | 100 | 4.84 / 15.1 | 4.84 | 0.1361 | 3.11e11 | 1.00× | 3.38e10 |
| | | change | none by 300 | | | | | | |
| KITTI 10 (40) | corrected | certified | 50 | 0.454 / 0.786 | 0.454 | 0.0850 | 1.18e11 | 0.53× | 2.04e10 |
| | | fixed budget | 100 | 0.0819 / 0.258 | 0.0819 | 0.0853 | 2.24e11 | 1.00× | 2.81e10 |
| | | change τ = 1e-2 / 1e-3 / 1e-4 | 209 / 273 / – | 0.0119 / 0.029; 0.0053 / 0.013 | | 0.0853; 0.0853 | 4.68e11; 6.12e11 | 2.09×; 2.73× | |
| KITTI 10 (40) | shipped | certified | 240 | 0.351 / 0.898 | 0.545 | 0.0902 | 5.63e11 | 2.51× | 9.29e10 |
| | | fixed budget | 100 | 4.89 / 19.5 | 4.91 | 0.1009 | 2.24e11 | 1.00× | 2.81e10 |
| | | change | none by 300 | | | | | | |
| ChArUco islam | corrected | certified | none by 1,030 | | | | | | |
| | | fixed budget (shipped 103) | 103 | 1.42 / 19.0 | 1.42 | – | 2.10e9 | 1.00× | 1.81e9 |
| | | change | none by 1,030 | | | | | | |
| ChArUco islam | shipped | certified | none by 1,030 | | | | | | |
| | | fixed budget | 103 | 1.51 / 20.5 | 1.49 | – | 2.10e9 | 1.00× | 1.81e9 |
| | | change | none by 1,030 | | | | | | |
| KITTI 06 prefix (161) | corrected | certified | 130 | 0.0072 / 0.267 | 0.0072 | 0.3824 | 1.32e12 | 1.54× | 3.55e11 |
| | | fixed budget | 100 | 0.0151 / 0.624 | 0.0151 | 0.3827 | 8.56e11 | 1.00× | 1.12e11 |
| | | change τ = 1e-2 / 1e-3 / 1e-4 | 194 / 252 / – | 0.0042 / 0.132; 0.0041 / 0.097 | | 0.3823; 0.3823 | 1.66e12; 2.16e12 | 1.94×; 2.52× | |
| KITTI 06 prefix (161) | shipped | certified | none by 300 | | | | | | |
| | | fixed budget | 100 | 0.377 / 5.21 | 0.519 | 0.3842 | 8.56e11 | 1.00× | 1.12e11 |
| | | change | none by 300 | | | | | | |

ATE barely distinguishes any two stops on the same segment. The largest spread on one segment is 0.0902 against 0.1009 m (shipped KITTI 10).

### 3.2 η sensitivity (from the logs; round: median move / total flops)

| Dataset | Variant | η = 0.05 | η = 0.1 | η = 0.2 | η = 0.5 |
|---|---|---|---|---|---|
| KITTI 06 (40) | corrected | r160: 0.0245 / 3.00e11 | r130: 0.0279 / 2.44e11 | r100: 0.0319 / 1.88e11 | r60: 0.0401 / 1.13e11 |
| KITTI 06 (40) | shipped | r190: 0.154 / 3.57e11 | r150: 0.173 / 2.82e11 | r120: 0.181 / 2.26e11 | r90: 0.220 / 1.70e11 |
| KITTI 03 (40) | corrected | r290: 0.158 / 7.80e11 | r250: 0.357 / 6.74e11 | r220: 0.662 / 5.94e11 | r180: 1.53 / 4.86e11 |
| KITTI 03 (40) | shipped | none (0.00 pass at 300) | none (0.23) | none (0.40) | none (0.55) |
| KITTI 04 (40) | corrected | r230: 0.157 / 7.48e11 | r170: 0.359 / 5.54e11 | r140: 0.564 / 4.57e11 | r100: 1.17 / 3.26e11 |
| KITTI 04 (40) | shipped | none (0.12) | none (0.72) | none (0.88) | r250: 0.764 / 8.10e11 |
| KITTI 10 (40) | corrected | r90: 0.102 / 2.11e11 | r50: 0.454 / 1.18e11 | r40: 0.671 / 9.51e10 | r20: 1.23 / 4.74e10 |
| KITTI 10 (40) | shipped | r250: 0.292 / 5.86e11 | r240: 0.351 / 5.63e11 | r200: 0.738 / 4.70e11 | r170: 0.974 / 4.01e11 |
| ChArUco | corrected / shipped | none (0.65 / 0.65) | none (0.78 / 0.78) | none (0.87 / 0.86) | none (0.93 / 0.91) |
| KITTI 06 prefix | corrected | r150: 0.0051 / 1.51e12 | r130: 0.0072 / 1.32e12 | r100: 0.0151 / 1.04e12 | r60: 0.0387 / 6.55e11 |
| KITTI 06 prefix | shipped | none (0.86) | none (0.91) | none (0.92) | none (0.98) |

- **The logged E_i were grown for η = 0.1.** A run tuned for η = 0.05 could tighten some of them, so the η = 0.05 rounds are upper bounds.
- **Even at η = 0.5** only KITTI 06 (round 60) and KITTI 10 (round 20) stop by round 60 on the corrected variant.

### 3.3 Fraction of poses passing (the certified-stop curve)

| Dataset | Variant | Rounds 10 · 20 · 30 · 50 · 100 · 150 · 200 · 250 · 300 (ChArUco: 100 · 200 · … · 1000) |
|---|---|---|
| KITTI 06 (40) | corrected | 0.42 · 0.80 · 0.82 · 0.85 · 0.95 · 1.00 · 1.00 · 1.00 · 1.00 |
| KITTI 06 (40) | shipped | 0.00 · 0.55 · 0.78 · 0.82 · 0.85 · 1.00 · 1.00 · 1.00 · 1.00 |
| KITTI 03 (40) | corrected | 0.00 · 0.00 · 0.00 · 0.00 · 0.03 · 0.23 · 0.38 · 1.00 · 1.00 |
| KITTI 03 (40) | shipped | 0.00 · 0.00 · 0.00 · 0.00 · 0.00 · 0.00 · 0.00 · 0.20 · 0.23 |
| KITTI 04 (40) | corrected | 0.00 · 0.00 · 0.00 · 0.05 · 0.10 · 0.82 · 1.00 · 1.00 · 1.00 |
| KITTI 04 (40) | shipped | 0.00 · 0.00 · 0.00 · 0.00 · 0.00 · 0.03 · 0.07 · 0.35 · 0.72 |
| KITTI 10 (40) | corrected | 0.00 · 0.15 · 0.33 · 1.00 · 1.00 · 1.00 · 1.00 · 1.00 · 1.00 |
| KITTI 10 (40) | shipped | 0.00 · 0.00 · 0.00 · 0.03 · 0.03 · 0.28 · 0.75 · 1.00 · 1.00 |
| ChArUco | corrected | 0.17 · 0.32 · 0.48 · 0.62 · 0.71 · 0.73 · 0.77 · 0.77 · 0.77 · 0.78 |
| ChArUco | shipped | 0.17 · 0.28 · 0.41 · 0.54 · 0.68 · 0.71 · 0.74 · 0.77 · 0.77 · 0.77 |
| KITTI 06 prefix | corrected | 0.16 · 0.53 · 0.57 · 0.73 · 0.99 · 1.00 · 1.00 · 1.00 · 1.00 |
| KITTI 06 prefix | shipped | 0.00 · 0.00 · 0.18 · 0.51 · 0.84 · 0.91 · 0.89 · 0.91 · 0.91 |

Every curve at every evaluation round is in `results/stage1/<dataset>/<variant>/pass_curve.csv`. The curves are not monotone in general (for example, prefix shipped 0.91 → 0.89): the linearization changes between evaluations.

### 3.4 Statistical finish of the relinearized solution

Fraction of poses whose remaining Gauss–Newton move at the beliefs (solver's Jacobian, D_i = S_ii) exceeds ηS_lo,i (η = 0.1):

| Dataset | Corrected at its certified stop | Corrected at the fixed budget (r100) | Shipped at its certified stop | Shipped at r100 |
|---|---|---|---|---|
| KITTI 06 (40) | 0 (r130) | 0.05 | 0.15 (r150) | 0.30 |
| KITTI 03 (40) | 0 (r250) | 0.975 | – | 1.00 |
| KITTI 04 (40) | 0 (r170) | 0.90 | – | 1.00 |
| KITTI 10 (40) | 0 (r50) | 0 | 0.25 (r240) | 1.00 |
| KITTI 06 prefix | 0 (r130) | 0.012 | – (0.26 at r300) | 0.34 |
| ChArUco | – (0.22 at r1030) | 0.83 (r100) | – (0.22 at r1030) | 0.83 |

- **On the corrected variant, a certified stop is also a statistical finish of the relinearized problem.**
- **On the shipped variant it is not.** The certificate is exact for the current linearization, but relinearizing with the factor-2 stereo Jacobian moves the target (WP6b's first-order mismatch).

### 3.5 §2.5: the never-finishes result, corrected

The shipped schedule (per-round relinearization at cavity means) on KITTI 06, first 40 keyframes, was rerun for 500 rounds with the new logging. Its round-500 beliefs are identical to WP1b's persisted state.

| State | η = 0.05 | η = 0.1 | η = 0.2 | η = 0.5 | S_lo median |
|---|---|---|---|---|---|
| Shipped schedule, round 500, **corrected yardstick** | 0.90 | **0.45** | 0.125 | 0 | 3.71 |
| … with WP2's yardstick (GP Jacobian as the solver's) | 0.90 | 0.45 | 0.05 | 0 | 4.12 |
| Corrected variant's certified stop (r130) | 0.05 | **0** | 0 | 0 | 4.85 |
| Shipped variant's certified stop (r150), corrected yardstick | 0.375 | 0.15 | 0 | 0 | 3.71 |

- **The first three rows** are the fraction of poses whose remaining GN move exceeds ηS_lo,i. For the certified stops, 0 at η = 0.1 is expected by construction, up to the difference between the certificate at the linearization and the move at the beliefs; it is 0.
- **The change-based criterion at the shipped round-500 state:** the largest per-round belief change in the D-metric is 4.57 (LANDMARK_10179). So τ = 1e-2, 1e-3 and 1e-4 all say "continue", and none was reached in any of the 500 rounds.
- **This corrects WP6d's framing** ("every change-based heuristic accepts it"). Measured in the D-metric over all variables, the shipped schedule's landmark beliefs never settle. Its remaining GN move stalls at 0.36 (median) while the beliefs keep moving.

### 3.6 Cost per round and per evaluation

| Dataset | GBP flops per round (as implemented) | Alt. baseline | Certificate flops per evaluation: median (max) | Certificate ÷ GBP of the 10 rounds between evaluations | Change criterion per round | Evaluator s per evaluation |
|---|---|---|---|---|---|---|
| KITTI 06 (40) | 1.80e9 | 2.55e8 | 6.4e8 (2.2e9) corrected; 6.6e8 shipped | 3.5 % | 1.9e6 | 0.6 |
| KITTI 03 (40) | 2.57e9 | 3.06e8 | 1.2e9 (2.5e9); 1.3e9 | 4.6–5.1 % | 2.2e6 | 0.8 |
| KITTI 04 (40) | 3.11e9 | 3.38e8 | 1.25e9 (3.1e9); 1.2e9 | 4.0 % | 2.5e6 | 0.8 |
| KITTI 10 (40) | 2.24e9 | 2.81e8 | 6.7e8 (1.6e9); 9.6e8 | 3.0–4.3 % | 2.1e6 | 0.6 |
| ChArUco | 3.48e7 | 3.00e7 | 6.6e9 (6.3e10) | 19× | 1.7e5 | 0.4 |
| KITTI 06 prefix | 8.56e9 | 1.12e9 | 9.5e9 (7.0e10); 1.0e10 | 11 % | 8.2e6 | 3.8 |

- **Per evaluation, the certificate is a small part of the cost on stereo.** The certified stop's cost is decided by the round at which it fires.
- **Against the belief-minus-incoming baseline** the certificate adds 24–43 % per 10 rounds on the 40-keyframe segments and 85 % on the prefix.

### 3.7 Validity and certificate tightness

| Dataset | Variant | Pose-evaluations | Violations (upper / lower) | Worst upper margin | Loewner min | E/‖e‖ median / p95 | Certified = oracle decisions | Growth exits (stop / continue / cap / steps) |
|---|---|---|---|---|---|---|---|---|
| KITTI 06 (40) | corrected | 1,200 | 0 / 0 | 7.9e-16 | 2.3e-13 | 1.47 / 8.0 | all | 1123 / 77 / 0 / 0 |
| KITTI 06 (40) | shipped | 1,200 | 0 / 0 | −4.3e-16 | 3.9e-13 | 1.60 / 11.0 | all | 1076 / 124 / 0 / 0 |
| KITTI 03 (40) | corrected | 1,200 | 0 / 0 | 1.6e-15 | 2.8e-16 | 1.23 / 2.42 | all | 449 / 751 / 0 / 0 |
| KITTI 03 (40) | shipped | 1,200 | 0 / 0 | 6.3e-15 | 5.5e-17 | 1.07 / 1.86 | all | 97 / 1103 / 0 / 0 |
| KITTI 04 (40) | corrected | 1,200 | 0 / 0 | 1.0e-15 | −1.05e-13 | 1.31 / 3.21 | all | 715 / 485 / 0 / 0 |
| KITTI 04 (40) | shipped | 1,200 | 0 / 0 | 1.2e-15 | −1.17e-13 | 1.12 / 1.73 | all | 191 / 1009 / 0 / 0 |
| KITTI 10 (40) | corrected | 1,200 | 0 / 0 | −1.1e-16 | 1.1e-14 | 1.33 / 2.44 | all | 1085 / 115 / 0 / 0 |
| KITTI 10 (40) | shipped | 1,200 | 0 / 0 | 2.2e-14 | −4.4e-16 | 1.23 / 2.55 | all | 539 / 661 / 0 / 0 |
| ChArUco | corrected | 9,295 | 0 / 0 | 2.2e-10 | **−5.4e-9** | 1.00 / 1.00 | all | 5630 / 3665 / 0 / 0 |
| ChArUco | shipped | 9,295 | 0 / 0 | 6.6e-11 | **−3.5e-9** | 1.00 / 1.00 | all | 5402 / 3893 / 0 / 0 |
| KITTI 06 prefix | corrected | 4,830 | 0 / 0 | −1.5e-4 | 3.2e-13 | 1.90 / 9.4 | all | 4383 / 447 / 0 / 0 |
| KITTI 06 prefix | shipped | 4,830 | 0 / 0 | −7.0e-4 | 3.7e-13 | 2.41 / 26.9 | all | 3725 / 1105 / 0 / 0 |

- **How to read the columns.**
  - A positive upper margin means ‖e‖ exceeded E by that relative amount; every one is below the 1e-9 tolerance.
  - The Loewner check was run at the first, middle and last evaluation rounds (10, 150, 300; ChArUco 10, 510, 1030).
  - Growth always ended on a certified exit: no pose hit the cap or the step limit.
- **Why the certificates are tight.** On the 40-keyframe graphs many patches grow to the whole graph, so E ≈ ‖e‖ there.

### 3.8 The corrected Jacobians

`certify/TRUE_JACOBIANS.md`. Central differences (h = 1e-6) of every factor's own error, along the solver's charts, at the evaluation after round 10 (ChArUco: round 1,030, all 711 factors).

| Dataset | Factors checked | Corrected: worst block (relative) | Shipped: worst block |
|---|---|---|---|
| KITTI 06 (40) | 11,526 | 1.3e-6 (stereo landmark columns) | GP ω of the first knot 1.14; stereo rotation ratio 2.0 |
| KITTI 03 (40) | 13,846 | 1.1e-6 | GP ω 2.11 |
| KITTI 04 (40) | 15,306 | 3.8e-7 | GP ω 1.34 |
| KITTI 10 (40) | 12,719 | 2.1e-6 | GP ω 2.35 |
| KITTI 06 prefix | 50,640 | 2.1e-6 | GP ω 1.92 |
| ChArUco | 711 | 9.35e-6 (pose–marker, marker rotation; unmodified factor) | GP ω 0.089 |

## 4. Worst cases

| Invariant or ratio | Worst | Where |
|---|---|---|
| Upper validity margin (stereo) | +2.2e-14 (below 1e-9) | shipped KITTI 10 |
| Upper validity margin (ChArUco) | +2.2e-10 (below 1e-9) | corrected ChArUco |
| Loewner, patch bound vs exact marginal | −5.4e-9 (rounding level, 10¹² anchor) | corrected ChArUco, sampled round |
| Loewner, stereo | −1.2e-13 | shipped KITTI 04 |
| Corrected Jacobian vs central differences | 9.35e-6 | ChArUco pose–marker, marker rotation (unmodified factor) |
| Corrected stereo or GP block vs central differences | 2.1e-6 | KITTI 10 / prefix, GP first-knot rotation columns |
| E/‖e‖, 95th percentile | 26.9 | shipped prefix |
| Gate A accuracy ratio (corrected) | 5.54 | KITTI 10, stop at round 50 |
| Gate A cost ratio (corrected) | 2.62 | KITTI 03, stop at round 250 |
| Latest certified stop (corrected) | 250 | KITTI 03 |
| No certified stop by the cap | ChArUco (both variants, 1,030); shipped KITTI 03, 04 and prefix (300) | |
| Relinearized move above ηS_lo at a certified stop | 25 % | shipped KITTI 10, round 240 |
| Largest per-round belief change at the shipped round-500 state | 4.57 D-units | LANDMARK_10179 |

## 5. What could not be done, and notes

- **The brief files** (`BRIEF_stage1_2.md`, `BRIEF_phaseD_wp6.md`) are not in the repository root.
- **All four KITTI sequences loaded** (03, 04 and 10 with their own configs, `kitti_03`, `kitti_04`, `kitti_10`; 06 with `kitti_05`, as throughout), so no substitution of 01 was needed.
- **Sequences 03 and 04 start far from convergence.** Their median remaining move at round 30 is 18–34 D-units, against 0.05 on 06. The shipped variant does not finish them in 300 rounds.
- **ChArUco:** stops count only from round 94, when the last pose has been added (evaluations and the pass curve run from round 10). The change criterion counts from round 95. ChArUco has no ground truth, so it has no ATE.
- **The change criterion** has no value at round 1, which has no previous belief to compare with. It counts every variable, landmarks included; that is why it rarely falls below τ.
- **The certificate's pass/fail agreed with the oracle** (‖e_i‖ ≤ ηS_lo,i) in every pose-evaluation, and with the exact-yardstick oracle in every evaluation's pass fraction. On these graphs the certificate is, in effect, exact for the decision.
- **Checkpointing.** ChArUco ran in 300-round segments and the prefix in 50-round segments, each gated on validity; every segment was under 2 minutes. One-tenth versions (30 rounds; ChArUco 103) preceded every full run.
- **The §2.5 rerun** reproduced WP1b's round-500 state exactly. That was needed to get the true GP Jacobians at that state and the D-metric change log.
- **Nothing in Stage 2 was started.**

## 6. Compute used (minutes)

- **Builds:** about 6.
- **Equivalence:** 4.9.
- **One-tenth runs:** KITTI 0.2 each (8 runs); ChArUco 0.5 each (2); prefix 1.2 and 1.4 (2).
- **Full KITTI runs:**

  | Sequence | Shipped | Corrected |
  |---|---|---|
  | 06 | 1.3 | 1.3 |
  | 03 | 2.2 | 2.2 |
  | 04 | 2.1 | 2.2 |
  | 10 | 2.1 | 2.1 |

- **ChArUco:** 1.5 (shipped) and 2.0 (corrected), in four segments each.
- **Prefix:** 5.4 (shipped) and 4.9 (corrected), in six segments each, none over 1.8.
- **§2.5 shipped schedule:** 0.3 (50 rounds) and 2.5 (500 rounds).
- **Analysis:** about 4.

## 7. Files

- **Solver:**
  - `gsolver/include/gsolver/core/jacobian_mode.h` (new);
  - `se3_posevel_stereo_point.cpp` and `derivatives.h` (the switchable true Jacobians);
  - `flop_count.h` (the belief-minus-incoming count);
  - `factor_node_base.h` and `factor_node.h` (`currentInformationBlock`);
  - `variable_node.cpp`.
- **Drivers:** `certify/cpp/certify_common.h`:
  - `f_*_Jtrue.npy` in every export;
  - the change criterion, periodic evaluation, beliefs exports and live Jacobian check in `RoundRunner`;
  - the flags of `certify_stereo` / `certify_charuco`.
- **Python:**
  - `certify/stage1.py` (new: `StatEngine`, accuracy);
  - `certify/sidecar_model.py` (`Jtrue`, pose blocks before elimination);
  - `certify/io.py`.
- **Scripts:** `scripts/stage1_evaluator.py`, `stage1_run.sh`, `stage1_checkpointed.sh`, `stage1_batch.sh`, `stage1_analyse.py`, `stage1_never_finishes.py`, `stage1_fdcheck.py`, `stage1_tables.py`.
- **Notes:** `certify/STAGE1.md`, `certify/TRUE_JACOBIANS.md`.
- **Results:**
  - `results/stage1/<dataset>/<variant>/`: `eval_rounds.csv` (gzipped for ChArUco and the prefix), `eval_summary.csv`, `stops.csv`, `eta_sensitivity.csv`, `pass_curve.csv`, `heuristics.csv`, `flops.csv`, `work.csv`, `validity.json`, `run.json`;
  - `results/stage1/`: `gate_A.json`, `never_finishes.json`, `fdcheck.json`, `equivalence.txt`, `tables.md`, `analyse.log`.
- **Git-ignored runs:** `exports/stage1/`.
- **Commit:** the commit that adds this file.
