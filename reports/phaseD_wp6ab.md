# Phase D, WP6a and WP6b: GBP flops, and the nonlinear residual probe

Branch `certify`, base commit `c5b870e`. Date 2026-09-27.

Brief: `BRIEF_phaseD_wp6.md`. The file is not in the repository root; I worked from the text in your message. It supersedes the WP2b message (WP2b was not run).

Fixed setting (WP1b): δ = 0.003 in the physical metric W = I, R = 1, per-variable linearization points.

**Growth rule decided after WP2**, applied throughout: the decision-driven growth decides CONTINUE only when the certified lower bound L_i = s_i − Σ_j a_ij c0_j exceeds the tighter of its two thresholds. This is `Engine(continue_rule="L")`, the default; `"s"` reproduces WP2.

## 1. Five-line summary

1. **GBP flops per round** (threshold schedule, no sleeping; the counters are validated in §5):

   | Run | GBP flops per round | WP2 certificate flops per round (median) | Certificate ÷ GBP |
   |---|---|---|---|
   | ChArUco islam | 2.2e7 (median) | 2.1e9 | 94 |
   | KITTI-40 | 1.80e9 | 6.4e8 | 0.35 |
   | Full prefix | 8.56e9 | 8.3e10 | 9.7 |

   - Variable-to-factor messages are 89–90 % of GBP's flops on stereo. G-solver re-sums a variable's other incoming messages for every outgoing one, which is O(degree²) per variable, and a pose has about 300 edges.
   - Belief-minus-incoming, the usual implementation, would cut the prefix's GBP flops 7.6× (to 1.1e9) and raise the ratio to about 74 (arithmetic, §3.2).
2. **The corrected growth rule reaches the oracle** on the re-certified persisted rounds:
   - prefix round 200: 69 certified stops against the oracle's 69 (WP2: 55);
   - KITTI-40 round 100: 5 against 5 (WP2: 4).

   It costs 2–4× WP2's certificate flops at those rounds: 1.9e11 on prefix round 200, 22× GBP.
3. **L5 (WP6b): criterion (N) essentially never holds.** (N) asks for s_i^NL ≤ ηS_lo,i/10. The ratio below is r = s^NL/(ηS_lo).
   - Pose-rounds the δ condition holds back (η margin > 0, δ margin < 0):
     - prefix: 68, r median 0.60, max 1.93;
     - KITTI-40: 29, median 0.43;
     - **none has r < 0.1.**
   - Pose-rounds the oracle stops: 1 of 69 on the prefix and 0 of 5 on KITTI-40 have r < 0.1.
   - The residual term alone is below ηS_lo (r < 1, necessary for T1 to release a pose) for 72–76 % of the held-back pose-rounds.
   - Stereo carries it: 99.9 % of s^NL in the worst decile, 7 of 7 on the prefix. GP carries 0.6–1.4 %.
4. **Two mechanisms drive the residual.**
   - *(a) The stereo Jacobian mismatch.* The solver's stereo Jacobian is not the derivative of its own error: central differences give exactly 2.0× in the rotation columns. So the linear model is wrong at first order in the offsets from the linearization points (≤ δ).
     - At the current state, before any step, this mismatch is 99.9 % of the nonlinear departure and gives r = 0.50 by itself (median, prefix held-back).
     - Its order is 1.00 (slope), against 2.00 for the stereo curvature and about 1e-3 of the total for the Gauss–Newton Jacobian change.
   - *(b) The landmark steps.* The exact local step moves weakly determined landmarks by metres to kilometres: every probe moves at least one landmark more than 1 m, and the largest move is 2.0 km. This adds a further 1.2–1.5× to s^NL.
5. **New finding: the GP factor's Jacobian is not its error's derivative** in one block: the first knot's angular velocity versus the translation-error rows, where the solver's entries reach 1.8 and the derivative's are at most 0.35.
   - **Theorem T0 assumes it is**, so S_lo inherits the error.
   - Rebuilt with the central-difference GP Jacobian, the exact yardstick is smaller for every pose: median −9.7 %, worst −34 % (prefix round 200).
   - **The sidecar's S_lo is therefore not a certified lower bound on the true Cramér–Rao radius.** This needs your decision before WP6c (§7).

## 2. Gate status

WP6a and WP6b have no gate. Every invariant I checked is listed below.

| Check | Result |
|---|---|
| Hand count of one operation of every kind against the solver's counters (ChArUco round 50: 53 kinds; prefix round 10: 306) | **match, all 359** |
| Counted f2v + v2f messages and relinearizations equal `work.csv`'s independent counts, every round | **equal, 403 of 403 rounds** |
| Default path byte-identical after the instrumentation | **12 of 12 identical** (`results/phaseD/wp6/equivalence.txt`) |
| Probe export load: J and e recomputed in C++ at the export's linearization points vs the export | **0.0 difference** (all 6 rounds, both factor types) |
| Probe identity: every variable at z = 0 ⇒ F_nl = F_jac = F_true = F_lin | **0.0 difference** (KITTI-40 r100, prefix r200) |
| Linear gradient after the exact local step, on the targets (relative to s_i) | max **3.1e-12** |
| Full D2 model reduced vs the sidecar's S and g | S ≤ 1.2e-13, g ≤ 3.8e-13 relative |
| Certificate validity L_i ≤ ‖e_i‖ ≤ E_i on the re-certified rounds (1e-9) | **0 violations** in 603 pose-rounds |
| Order of the residual terms (state scaled by 1, ½, ¼, ⅛) | stereo mismatch 1.00, stereo curvature 2.00, GP remainder 1.00 (prefix r200; KITTI-40 r100 the same) |

## 3. WP6a: GBP flops

### 3.1 Per round

| | ChArUco islam (103 rounds) | KITTI-40 (100) | Full prefix (200) |
|---|---|---|---|
| GBP flops per round: median (first → last) | 2.19e7 (2.8e5 → 3.48e7; the graph grows) | 1.80e9 (constant) | 8.56e9 (constant) |
| Shares: factor update / f2v / belief / v2f | 24 / 48 / 3 / 25 % | 1.7 / 9.3 / 0.4 / 88.6 % | 1.6 / 8.6 / 0.3 / 89.5 % |
| WP2 certificate flops per round: median (max) | 2.06e9 (1.59e10) | 6.35e8 (1.05e9) | 8.26e10 (1.02e11) |
| Certificate ÷ GBP per round: median (max) | 94 (456) | 0.35 (0.58) | 9.7 (11.9) |
| Certificate ÷ GBP over the run | 190 | 0.37 | 7.0 |
| Certificate flops with the corrected growth rule, persisted rounds | – | r1 1.05e9, r10 1.10e9, r100 2.29e9 (WP2: 1.05e9, 3.3e8, 6.3e8) | r1 2.06e10, r10 1.84e10, r200 1.92e11 (WP2: 2.04e10, 5.7e9, 9.1e10) |
| Relinearizations per round: median (max) | 210 (382) | 1,368 (11,526) | 11,077 (50,640) |
| Relinearization flops, upper bound per round | 3.8e6 (≤ 11 % of GBP; allowance-based) | 8.1e6 (0.45 %) | 3.5e7 (0.4 %) |

- **The runs.** Threshold mode with `--count-flops`, no sleeping (the WP6c setting).
  - The ChArUco trajectory is byte-identical to WP2's run, in which nothing slept.
  - The WP2 KITTI-40 and prefix runs slept poses, so their GBP work was lower: the prefix's messages fell 17 % by round 200.
- **WP2's certificate** decided every pose every round. The ChArUco minimum ratio is 0 because round 1 fails A0 and has no certificate.
- **The corrected-rule numbers come from a fresh engine per round** (no base-patch history), so part of the increase is the base patches regrown from scratch.
- **Relinearization bound.** The stereo factor's `updateErrorAndJacobian` is hand-counted at 665 flops (certify/FLOPS.md). GP and ChArUco factors get an allowance of 1e4 each; that is not counted.

### 3.2 Where GBP's flops go

- **A pose's v2f work is O(degree²).** `VariableNode::sendMessage(i)` re-sums the other deg − 1 incoming messages for each of its deg outgoing messages (`variable_node.cpp`, lines 17–70). For a stereo pose with d = 12 and deg ≈ 300, that is about 1.4e5 flops per message, 4e7 per pose per round.
- **The usual GBP implementation** forms each outgoing message as belief minus one incoming message: 2d³ + 5d² + 3d flops per message, independent of degree. Computed from the final graphs' degrees:

  | | v2f per round (shipped → belief-minus-incoming) | GBP per round |
  |---|---|---|
  | Prefix | 7.66e9 → 2.20e8 (35× less) | 8.56e9 → 1.12e9 |
  | KITTI-40 | 1.59e9 → 5.0e7 | 1.80e9 → 2.55e8 |
  | ChArUco | 8.6e6 → 3.8e6 | 3.48e7 → 3.00e7 |

- **With that implementation**, WP2's certificate would be about 74× GBP on the prefix and 2.5× on KITTI-40. This is arithmetic, not a run.
- **The WP6c cost accounting** will use the measured, as-implemented GBP flops unless you say otherwise.

## 4. WP6b: the nonlinear residual probe

**The measurement** (certify/NL_PROBE.md):
- **Rounds.** Each persisted WP2 round (prefix 1, 10, 200; KITTI-40 1, 10, 100) is re-certified with the corrected growth rule, using the solver's d_i and relinearization flags from the WP2 log.
- **Step.** For every pose, z₁ is the exact step on its certificate patch: the same Cholesky factor, with landmarks back-substituted in the full D2 model.
- **Residual.** F(z₁) is evaluated by re-running every factor that touches the patch's poses and landmarks in C++, in `updateErrorAndJacobian`, the solver's own code. Then s^NL = ‖[S_PP⁻¹F̃_P(z₁)]_i‖_{D_i}.

**(N) is a diagnostic criterion, not a certified one:** s^NL is T1's residual term without its curvature constants and without the leak recursion.

**Where the selected pose-rounds are.** Only the last persisted rounds have any:
- round 1: no pose is within the η margin;
- round 10: every pose was relinearized.

### 4.1 Distribution of r = s^NL / (η S_lo)

| Pose-rounds | n | median | p90 | max | r < 0.1 (N) | r < 1 |
|---|---|---|---|---|---|---|
| Prefix r200, held back: η margin > 0, δ margin < 0 (recomputed) | 68 | 0.60 | 1.51 | 1.93 | **0 %** | 76 % |
| … the same, selected by the WP2 log's margins | 82 | 0.62 | 1.49 | 1.93 | 0 % | 73 % |
| Prefix r200, oracle stops | 69 | 0.47 | 1.14 | 1.82 | **1.4 %** (1) | 80 % |
| KITTI-40 r100, held back (recomputed) | 29 | 0.43 | 1.34 | 2.08 | **0 %** | 72 % |
| … selected by the WP2 log | 30 | 0.44 | 1.40 | 2.08 | 0 % | 70 % |
| KITTI-40 r100, oracle stops | 5 | 0.40 | 1.30 | 1.80 | 0 % | 80 % |
| Prefix r200, held back, with the GP-corrected yardstick (§4.4) | 68 | 0.69 | 1.60 | 2.09 | 0 % | 72 % |

Earlier rounds, all poses, for scale: r median 4.7 at prefix round 1 and 30 at round 10. Steps are large there: summed over those rounds' probes, 4,607 (prefix) and 661 (KITTI-40) factor evaluations change inlier status at the stepped point, against 0 at the late rounds.

### 4.2 Attribution

| | Prefix r200, held back | KITTI-40 r100, held back |
|---|---|---|
| Worst decile (n; r range) | 7; 1.55–1.93 | 3; 1.39–2.08 |
| Worst decile: s^NL from stereo factors ÷ s^NL, median | 1.000 | 0.999 |
| Worst decile: s^NL from GP factors ÷ s^NL, median | 0.006 | 0.014 |
| Worst decile: stereo larger than GP | 7 of 7 | 3 of 3 |
| At the current state (no step): r median | 0.50 | 0.33 |
| At the current state: mismatch / Jacobian change / curvature, share of s^NL_state (median) | 0.999 / 0.001 / 0.048 | 0.999 / 0.002 / 0.053 |
| s^NL ÷ s^NL_state, median | 1.24 | 1.47 |
| Landmark back-substituted step: per-probe maximum, median | 356 m | 1,835 m |
| Landmark step: per-probe 99th percentile, median | 14 m | 14 m |
| Probes that move some landmark more than 1 m | 68 of 68 | 29 of 29 |

- **What the columns decompose.** Each factor class's contribution F_nl − F_lin, split into:
  - the Gauss–Newton Jacobian change (J̃(x) − J̃_lin)ᵀΛr_lin;
  - the model mismatch J̃(x)ᵀΛ(J − J̃)τ, with J the true Jacobian;
  - the curvature J̃(x)ᵀΛ(e(x) − e_lin − Jτ).
- **Norm shares do not add to 1:** the three terms partly cancel.
- **Reading.**
  - Before any step, the solver's nonlinear equation departs from its linear model almost entirely through the stereo factor's doubled rotation columns. That departure is first order in the offsets from the linearization points (≤ δ), which is Proposition D's O(κ_GN δ) warning with a much larger constant than the Gauss–Newton term.
  - The exact local step then extrapolates weakly determined landmarks, in depth, far outside any δ-sized box. Their stereo factors' curvature and Jacobian change add the rest.

### 4.3 Does T1 release them? (L5)

**Not at (N)'s margin.**
- **With H₀ = the solver's H**, the brief's definition, the residual term alone is 0.16–1.93 × ηS_lo on the held-back pose-rounds. It never falls below 0.1 ηS_lo there, and does so once in 69 oracle stops.
- **Releasing a pose on the statistical test alone needs at least r < 1.** That holds for 72–76 % of held-back pose-rounds, before T1's curvature remainder and recursion are added.
- **The box B(ū) T1 needs** would have to contain landmark moves of hundreds of metres. At a δ-sized box it cannot verify for those poses.
- **The mismatch term is removable by the choice of H₀.** T1 admits H₀ = J̃ᵀΛJ, the one-sided Gauss–Newton matrix; with it the step itself accounts for the first-order mismatch. That is a method change, so I have not made it (§7).

### 4.4 GP Jacobian versus Theorem T0

**The check.** Central differences of each factor's own error along the solver's charts (h = 1e-6), at the linearization points of the persisted exports. The first 200 factors of each class were checked, and every GP factor (160 on the prefix).
- **Stereo:** the rotation columns are exactly 2.0× the derivative, in all 200 factors checked. Every other block matches to 1e-6 relative, which confirms WP0.
- **GP:** every block matches to 1e-5, except the first knot's angular-velocity columns (9–11) in the translation-error rows (0–2).
  - There the solver's entries reach 1.8 where the derivative's are at most 0.35.
  - Relative difference: median 0.50, maximum 1.13. In the rotation-error rows 3–5 of the same columns the difference is at most 0.12, in entries up to 0.6.
  - The source is `jacob_dDinvP1invP2_de1e2_gp` (`gsolver/include/gsolver/maths/derivatives.h:37`). It carries a rotation–translation coupling of order dt·[v]× that the error, as the solver evaluates it, does not have.
- **Consequence.** T0 builds I_F with every non-stereo Jacobian taken as the solver's. Rebuilt with the central-difference GP Jacobians (stereo still halved), the exact yardstick S_i = √λ_min(C_iᵀ[I_F⁻¹]_ii C_i)·q₁₂(0.05) changes:

  | | Median | Range | Worst pose |
  |---|---|---|---|
  | Prefix r200 | −9.7 % | −34 % to −0.07 % | the first pose, −34 % |
  | KITTI-40 r100 | −8.6 % | −34 % to −0.1 % | the first pose, −34 % |

- **The sidecar's S_lo is therefore not a certified lower bound on the true (1 − δ_s) radius.**
  - Every S_lo in WP2 and here was built with the solver's GP Jacobian.
  - WP2's check S_lo ≤ S_exact compared two quantities with the same error, so it could not see this.
  - The linear certificate (E, L against the exact solution of the solver's linear model) is not affected.

## 5. Worst cases

| Invariant | Worst | Where |
|---|---|---|
| Hand count vs counters | 0 mismatches | 359 kinds |
| Flop logs vs work log | 0 mismatched rounds | 403 rounds |
| Default path byte-identical | 0 of 12 comparisons differ | `results/phaseD/wp6/equivalence.txt` |
| Probe load (J, e) | 0.0 | all 6 exports |
| Probe identity | 0.0 | KITTI-40 r100 (4,719 variables), prefix r200 (18,098) |
| Linear-gradient residue after the step | 3.1e-12 of s_i | prefix |
| Full model vs sidecar | 3.8e-13 (g) | prefix round 1 |
| Certificate validity, re-certified rounds | 0 violations | 603 pose-rounds |
| Largest r among held-back pose-rounds | 1.93 | prefix r200, pose t = 23.08 s: s^NL 0.896, ηS_lo 0.465, E 0.0134, d 0.0017 |
| Smallest r among oracle stops (the only one with r < 0.1) | 0.095 | prefix r200, pose t = 13.83 s |
| Largest landmark back-substituted step | 2.0 km | prefix r200 |
| Yardstick with the true GP Jacobian | −34 % | the first pose, both runs |

## 6. What could not be done, and why

- **WP6b uses only the persisted rounds.** No other WP2 states exist. Held-back and oracle-stopped poses occur only at the last persisted round of each run.
  - The re-certification is a fresh engine per round, without WP2's base-patch history or sleep records, so its margins differ from the WP2 log. Both selections are reported.
  - WP6c computes s^NL at every evaluation round, as the brief says.
- **ChArUco has no probe.** The brief asks for the prefix, and an islam graph cannot be rebuilt from an export without replaying its front end. For WP6c the probe server has to run inside the live driver: the server's `poll()` is ready for that, but not yet wired into `sidecarRound`.
- **The flop counters cover the frozen code path only**, the one the threshold schedule uses. Relinearizations are counted as operations and bounded, not counted in flops.
- **The GP Jacobian discrepancy is located but not traced to a derivation step.** The fix is either a corrected analytic Jacobian or central differences in the certificate's I_F. That is your decision (§7).
- **BRIEF_phaseD_wp6.md is not in the repository root.** I did not create it.

## 7. Decisions needed

1. **GP Jacobian and the yardstick (before WP6c).** S_lo is up to 34 % too large because T0's I_F uses the solver's GP Jacobian. The options:
   - (a) build I_F with the true GP Jacobian, from central differences of the GP error in C++ (as §4.4 does) or an analytic correction, as the stereo halving does for stereo;
   - (b) keep it and record it as a limitation of the yardstick.

   (a) changes (S) and (C) in WP6c, since both use S_lo.
2. **Criterion (N) at δ = 0.003.** It will almost certainly report "not by 300" in WP6c. Keep it as specified, or add either of these as diagnostics:
   - (i) T1 with H₀ = J̃ᵀΛJ, which removes the first-order stereo mismatch from the residual after the step;
   - (ii) a residual with the landmark steps restricted to a δ-sized box.
3. **GBP cost baseline for WP6c's break-even.** The measured as-implemented flops, dominated by O(degree²) v2f, or also the belief-minus-incoming arithmetic of §3.2, which raises the certificate's relative cost 7–8×.

## 8. Compute used (minutes)

- **Builds:** about 4.
- **WP6a runs:** ChArUco 0.003, KITTI-40 0.29, prefix 1.1.
- **WP6b probes** (three passes: development, the three-way split, the no-step diagnostic):
  - KITTI-40: about 0.1–0.2 per round per pass;
  - prefix: about 0.6, 0.6 and 1.4 per round per pass;
  - about 10 in total.
- **Checks:** order and central-difference checks about 2; GP-yardstick rebuild about 2; one debugging probe 0.3.
- **Equivalence:** 4.8.

Every run was under 6 minutes. Each prefix round was run as its own invocation with a 6-minute cap and its own output file, which served as the checkpoint.

## 9. Files

- **Solver:**
  - `gsolver/include/gsolver/core/flop_count.h` (new);
  - the counting sites in `factor_node.cpp`, `factor_node_edge_tuple_helper.h`, `variable_node.cpp`;
  - `flop_class_` in `factor_node_base.h` and `variable_node_base.h`.
- **Drivers:** `certify/cpp/certify_common.h`:
  - flop classes and the flop log in `RoundRunner`;
  - `readNpy`, `loadFrozenState`, `ProbeServer` with `fdCheck`;
  - `certify_charuco` / `certify_stereo` flags `--count-flops`, `--flop-trace-round`, `--probe-export`, `--probe-dir`.
- **Python:**
  - `certify/sidecar.py`: `continue_rule`, `yard_all`, `last`;
  - `certify/sidecar_model.py`: `J_override`;
  - `certify/nonlinear_probe.py` (new);
  - `scripts/phaseD_sidecar.py`: `--continue-rule`.
- **Scripts:** `scripts/phaseD_wp6a_handcount.py`, `phaseD_wp6a_flops.py`, `phaseD_wp6b_probe.py`, `phaseD_wp6b_checks.py`, `phaseD_wp6b_gp_jacobian.py`, `phaseD_wp6b_analyse.py`.
- **Notes:** `certify/FLOPS.md`, `certify/NL_PROBE.md`.
- **Results:**
  - `results/phaseD/wp6/flops/{charuco,kf40,prefix}/`: `flops.csv` (per round: GBP flops by category, message counts, the WP2 certificate flops and the ratio), `flops_by_class.csv`, `flops_trace.csv`, `work.csv`, `relin_rounds.csv`, `run.json`;
  - `results/phaseD/wp6/flops/summary.json`, `handcount_charuco_r50.json`, `handcount_prefix_r10.json`;
  - `results/phaseD/wp6/probe/{prefix,kf40}/`: `probe_rNNNNN.csv` (per pose: certificate, margins, decision, oracle stop, s^NL and its parts, no-step diagnostic, landmark steps, WP2 log values), `probe_checks_rNNNNN.json`;
  - `results/phaseD/wp6/probe/`: `summary.json`, `summary.txt`, `checks_kf40_r100.json`, `checks_prefix_r200.json`, `gp_jacobian_{kf40,prefix}.json`;
  - `results/phaseD/wp6/equivalence.txt`.
- **Commit:** the commit that adds this file.
