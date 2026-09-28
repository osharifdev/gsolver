# Stage 2b: Theorem T1 at the certified stops

Branch `certify`, base commit `d998764` (Stage 2a). Brief: `BRIEF_stage2b.md` (it replaces §3 of `BRIEF_stage2.md`). Scope: the four corrected-variant certified stops (KITTI 06 at round 130, 03 at 250, 04 at 170, 10 at 50), rebuilt exactly in Stage 2a. Method: `certify/STAGE2B.md`.

## 1. Five-line summary

1. **The Newton reference converges quadratically at all four stops.**
   - 7–8 iterations, final ‖F‖ = 2–6e-8, and no factor changes inlier status.
   - **Poses at the true zero x̂:** 95 %, 97.5 %, **57.5 %** and 100 % are within ηS_lo,i of the stop state (06, 03, 04, 10). Their distance to x̂ is 0.58–3.8× (median) Stage 1's linear bound E_i.
   - **Landmarks:** the median moves 1–6 mm, but 477–961 per stop move more than 0.1 m, and the largest 1.6 km.
2. **Gate B2 fails for both options.**
   - Theorem T1's box check Ψ(ū) ≤ ū verifies at no stop and for no c ∈ {1, 2, 4, 8, 16}.
   - No pose receives a certified bound on its distance to x̂.
3. **Why option A fails:** 112–705 landmarks per attempt have a residual step larger than their box (s_l > ū_l), which fails the check whatever the curvature bounds are.
   - At c = 16 these are almost all decision D2's single-edge landmarks: 114–253 per stop, against 0–1 multi-observation landmarks.
   - D2 pins each of them to its frozen message, up to 339 m from the solver's belief, and the pre-registered candidate clips their step to 0.1 m.
4. **Why option B fails:** its boxes contain every landmark's zero at c = 16, but in the D_l-norm they reach into regions where G-solver switches the stereo factor off (behind the camera or off the right image).
   - 20–1,508 factor enclosures per attempt are unbounded, which makes every pose's Ψ infinite.
   - The pose rows alone would fit: s_i/ū_i ≤ 0.104 at c = 16 at every stop.
5. **The tooling is validated, and it shows the claim is weaker than Stage 2a suggested.**
   - The interval/AD evaluator matches central differences to ≤ 3.3e-8, and no random point of a test box falls outside its enclosure (0 of 200 per stop).
   - The implemented corrected Jacobians differ from the exact derivative by ≤ 5.3e-6 (GP) and ≤ 5e-8 (stereo), which moves the zero by ≤ 1.2e-5 D-units.
   - Independently of T1: on KITTI 04, 42.5 % of poses are farther than ηS_lo from x̂, so the certified stop is not a statistical finish of the D2 nonlinear problem there. Stage 2a's Gate B1 pass on 04 (100 %) did not predict this.

## 2. Step 1: the Newton reference

**Method.** Newton's method on F, with the exact derivative F′, from the stop state (`scripts/stage2b_newton.py`):
- the full D2 model: poses and kept landmarks, the landmark priors p_L and the frozen single-edge messages; the stereo factors that are inliers at x_c, and the GP factors;
- the corrected (true) Jacobians;
- the chart at the stop state x_c;
- direct solves: a dense Schur complement on the poses, with the landmark blocks eliminated exactly;
- stopping at ‖dz‖ ≤ 1e-12‖z‖.

**Metrics:** poses in D_i = S_ii (Stage 1's metric); landmarks in D_l = the landmark's Gauss–Newton diagonal block at x_c.

| Stop | Iterations | Converged | Final ‖F‖ | Pose move (D): median / p90 / p99 / max | Pose move ÷ ηS_lo: median / max | Poses within ηS_lo | Pose move ÷ Stage 1 E_i: median / max |
|---|---|---|---|---|---|---|---|
| KITTI 06, r130 | 7 | yes | 2.8e-8 | 0.149 / 0.366 / 0.513 / 0.515 | 0.31 / 1.03 | **95.0 %** | 3.78 / 6.76 |
| KITTI 03, r250 | 8 | yes | 5.5e-8 | 0.151 / 0.341 / 0.452 / 0.498 | 0.31 / 1.02 | **97.5 %** | 0.58 / 2.23 |
| KITTI 04, r170 | 8 | yes | 2.3e-8 | 0.417 / 0.837 / 0.963 / 0.978 | 0.87 / 1.96 | **57.5 %** | 1.50 / 2.90 |
| KITTI 10, r50 | 7 | yes | 3.4e-8 | 0.341 / 0.439 / 0.465 / 0.465 | 0.69 / 0.93 | **100 %** | 0.89 / 1.33 |

| Stop | Landmarks | Move (D): median / p90 / p99 / max | Move (m): median / p90 / p99 / max | Moving > c·0.1 m for c = 1 / 2 / 4 / 8 / 16 |
|---|---|---|---|---|
| KITTI 06 | 4,676 | 0.025 / 511 / 1.6e4 / 3.4e5 | 0.0021 / 0.56 / 17.1 / 1,620 | 961 / 726 / 551 / 389 / 266 |
| KITTI 03 | 4,614 | 0.018 / 113 / 5.8e3 / 3.2e5 | 0.0011 / 0.17 / 9.3 / 1,050 | 516 / 440 / 342 / 249 / 162 |
| KITTI 04 | 4,642 | 0.031 / 0.28 / 8.6e3 / 1.1e5 | 0.0021 / 0.12 / 8.6 / 154 | 477 / 419 / 332 / 263 / 189 |
| KITTI 10 | 4,222 | 0.092 / 116 / 4.3e3 / 1.8e5 | 0.0059 / 0.19 / 5.0 / 175 | 536 / 408 / 299 / 199 / 127 |

**Which landmarks move** (`results/stage2b/<dataset>/diagnosis.json`):

| Stop | Single-edge landmarks (D2 frozen-message pin, precision ≥ 1e3): count; median / max move | Multi-observation landmarks: count; median / max move |
|---|---|---|
| KITTI 06 | 1,397; 0.20 m / 339 m | 3,279; 1.1 mm / 1,620 m |
| KITTI 03 | 533; 0.58 m / 319 m | 4,081; 0.9 mm / 1,050 m |
| KITTI 04 | 463; 1.04 m / 107 m | 4,179; 1.9 mm / 154 m |
| KITTI 10 | 545; 0.39 m / 175 m | 3,677; 5.0 mm / 152 m |

**The implemented equation.** Its zero, found by simplified Newton on the solver's own corrected-Jacobian code from x̂, lies within 4.0e-7 to 1.2e-5 D-units of x̂ for poses (maximum 1.2e-5, KITTI 10).

**Reading:**
- **Pose distance vs E_i.** E_i bounds the distance to the linearization's solution. The distance to the nonlinear zero is larger, up to 6.8× on KITTI 06.
- **KITTI 04.** The certified stop sits close to the threshold there (E_i/ηS_lo median 0.75, Stage 2a), so a 1.5× (median) gap leaves 17 of 40 poses beyond ηS_lo.
- **The single-edge pin is not the explanation for 04.** On 04, a pose's distance to x̂ is not significantly associated with how far its single-edge landmarks move (Spearman 0.19, p = 0.2). On 03 the association is strong (0.66).

## 3. Step 2: Theorem T1

### 3.1 Ingredients, as implemented

- **Candidate.** z₁ is the Stage 2a global candidate (patch steps; landmark steps clipped to 0.1 m), taken as a step in the x_c chart.
- **H₀.** The exact derivative F′(z₁), with the second-order terms, assembled in double precision.
- **Rows:**
  - pose i: patch P_i (its Stage 1 final patch) ∪ every landmark P_i observes, with s_i, a_ij and ℓ_ip from H₀ by exact block elimination of the landmarks;
  - landmark l: patch {l}.
- **ρ(A).** ρ(A_PP) < 1 by a Collatz–Wielandt witness at every stop: row sums ≤ 0.30, witness ratio 0.20–0.25.
- **Δ.** Computed from outward-rounded interval enclosures of every factor's F′_f over z₁ + B(ū), obtained by nested forward-mode automatic differentiation in C++ (§3.4). Factors whose inlier conditions are not guaranteed on the box are set to Δ = ∞.
- **Radii:**
  - poses: c·E_i;
  - landmarks, option A: c·0.1·√λ_max(D_l);
  - landmarks, option B: c·E_l, with E_l = ‖D_l^{−1/2}g_l‖ + Σ_p‖D_l^{−1/2}H_lpD_p^{−1/2}‖E_p at the stop's linearization;
  - c ∈ {1, 2, 4, 8, 16}.
  - The radii were fixed by these formulas alone; the Newton result was not used.

### 3.2 Gate B2 (pre-registered)

Gate B2 requires all four of the following on at least three of the four KITTI segments, separately for each option:
- the box verifies;
- validity holds for 100 % of certified poses;
- at least 80 % of poses have E_i^NL ≤ ηS_lo,i;
- median E_i^NL / true distance ≤ 3.

| Option | KITTI 06 | KITTI 03 | KITTI 04 | KITTI 10 | Segments passing | Gate B2 |
|---|---|---|---|---|---|---|
| A | box does not verify (c = 1–16) | does not verify | does not verify | does not verify | 0 | **fail** |
| B | does not verify (c = 1–16) | does not verify | does not verify | does not verify | 0 | **fail** |

Because no box verifies, none of the other quantities of §3 of the brief exists for any pose:
- the fraction of poses with a finite certified bound is 0;
- validity against x̂, tightness E^NL/true distance, the share of u⁺ from s versus Δ, and E^NL ≤ ηS_lo are therefore undefined.

`probe_T1.csv` records, for every pose, ‖z₁‖, s_i and the true distance, with "not certified: Ψ(ū) ≤ ū not verified for c in 1..16".

### 3.3 The attempts

| Stop | Option | c | Landmarks with s_l > ū_l | Unbounded factor enclosures | Poses with infinite Ψ | s_i / ū_i, poses: median / max |
|---|---|---|---|---|---|---|
| KITTI 06 | A | 1 / 2 / 4 / 8 / 16 | 705 / 604 / 478 / 346 / 245 | 247 / 681 / 2,142 / 6,385 / 9,192 | 40 at every c | 0.84 / 1.67 at c = 1; 0.053 / 0.10 at c = 16 |
| | B | 1 / 2 / 4 / 8 / 16 | 0 | 69 / 120 / 230 / 332 / 553 | 40 | same |
| KITTI 03 | A | 1 … 16 | 418 / 358 / 294 / 207 / 134 | 445 / 1,376 / 3,937 / 9,775 / 12,613 | 40 | 0.16 / 0.42 at c = 1; 0.010 / 0.026 at c = 16 |
| | B | 1 … 16 | 0 | 20 / 52 / 123 / 255 / 490 | 40 | same |
| KITTI 04 | A | 1 … 16 | 390 / 341 / 302 / 244 / 180 | 505 / 1,373 / 3,512 / 9,515 / 13,455 | 40 | 0.23 / 0.73 at c = 1; 0.014 / 0.046 at c = 16 |
| | B | 1 … 16 | 0 | 49 / 95 / 164 / 248 / 382 | 40 | same |
| KITTI 10 | A | 1 … 16 | 355 / 306 / 238 / 168 / 112 | 700 / 2,170 / 6,474 / 10,650 / 11,826 | 40 | 0.082 / 0.31 at c = 1; 0.005 / 0.019 at c = 16 |
| | B | 1 … 16 | 0 | 34 / 106 / 333 / 753 / 1,508 | 40 | same |

- **The unbounded enclosures** are stereo factors whose inlier conditions (camera depth > 0, disparity > 0, right-image column ≥ 0) cannot be guaranteed on the box. There G-solver's F switches the factor off and is not smooth. At c = 16, GP enclosures left the series domain of the SO(3) maps: 2 on KITTI 03 and 1 on KITTI 10.
- **An infinite Δ** on any stereo factor reaches, through the patch weights ℓ and (I − A)⁻¹, every pose.
- **Option A** also fails directly on landmarks whose residual step exceeds their box: s_l/ū_l reaches 3,393 (KITTI 06, c = 1). That is independent of Δ.

**The pose rows on their own would fit.** s_i ranges over 0.031–0.065 (median) and 0.049–0.155 (max) across the four stops, and s_i/ū_i ≤ 0.104 at c = 16 everywhere. The Newton diagnosis says c ≥ 7 (06), 3 (03), 3 (04) and 2 (10) would be needed for the pose boxes to contain x̂.

### 3.4 Precision and the evaluator

- **Intervals** (`certify/cpp/t1_ad.h`; own implementation, since no interval library is installed):
  - every +, −, × and ÷ is computed in round-to-nearest, and both bounds are then moved outward by one ulp with `std::nextafter`, which contains the exact result on the endpoints;
  - division by an interval containing 0 is flagged.
- **The SO(3) maps** are power series with rigorous remainder bounds on the value and its first two derivatives, plus a 1e-11 margin for the rounding of the stored coefficients. No libm function is used.
- **Derivatives.** Nested forward-mode duals give F_f and F′_f exactly (in z and in the local perturbation).
- **What was not built.** The verified linear algebra that would make s, a and ℓ rigorous upper bounds (interval residuals of the patch solves, a verified inverse bound) is only needed if a check verifies. None did, so no bound is claimed.
- **The failures are not rounding effects.** Each is infinite (an unbounded enclosure) or orders of magnitude (s_l/ū_l up to 3.4e3).

| Evaluator check (`results/stage2b/<dataset>/evaluator_checks.json`) | 06 | 03 | 04 | 10 |
|---|---|---|---|---|
| F(0) vs the solver's JᵀΛe, stereo: median / max | 1.0e-9 / 1.4e-8 | 2.2e-9 / 2.1e-8 | 4.4e-10 / 5.2e-9 | 8.5e-13 / 5.0e-8 |
| F(0) vs the solver's JᵀΛe, GP: median / max | 1.8e-7 / 4.1e-6 | 1.7e-7 / 4.0e-6 | 2.2e-7 / 5.3e-6 | 4.5e-8 / 4.8e-6 |
| F′ vs central differences (12 columns) | 2.2e-8 | 1.6e-8 | 3.3e-8 | 3.3e-9 |
| Point enclosure contains the double F, F′ | yes | yes | yes | yes |
| Random points of a ±1e-3 box outside its enclosure (200 points) | 0 | 0 | 0 | 0 |
| Load: J, e recomputed at x_c vs the export | 0.0 | 0.0 | 0.0 | 0.0 |

**Where the evaluator differs from the solver, and why.** The equation certified is the exactly differentiated one. The implemented corrected solver differs from it in two places:
- **GP:** its rotation block is accurate to about |φ|²/12 relative, with |φ| ≈ 1–4e-3, which passed Stage 1's 1e-5 check.
- **Stereo:** its translation block assumes RᵀR = I, while its stored rotations are orthonormal only to about 1e-8, because of the small-angle branch of `expMapSO3`.

Step 1 measures the resulting distance between the two zeros: ≤ 1.2e-5 D-units.

## 4. Worst cases

| Invariant or ratio | Worst | Where |
|---|---|---|
| Newton final ‖F‖ | 5.5e-8 | KITTI 03 |
| Implemented equation's zero vs x̂ (poses) | 1.2e-5 D-units | KITTI 10 |
| Evaluator F vs the solver's | 5.3e-6 (GP), 5.0e-8 (stereo) | KITTI 04 (GP), KITTI 10 (stereo) |
| F′ vs central differences | 3.3e-8 | KITTI 04 |
| Enclosure containment (point, random box points) | always contained | all stops |
| Pose distance to x̂ ÷ ηS_lo | 1.96 | KITTI 04 |
| Poses within ηS_lo of x̂ | 57.5 % | KITTI 04 |
| Pose distance to x̂ ÷ Stage 1 E_i | 6.76 | KITTI 06 |
| Landmark move to x̂ | 1,620 m (multi-observation); 339 m (single-edge) | KITTI 06 |
| s_l / ū_l, option A, c = 1 | 3,393 | KITTI 06 |
| Unbounded enclosures, one attempt | 13,455 (A, c = 16); 1,508 (B, c = 16) | KITTI 04; KITTI 10 |
| Box verification | none of 40 attempts | all stops, both options |

## 5. What could not be done, plainly

**No certified bound on the distance to x̂ was obtained for any pose.** T1, with the pre-registered candidate and box rules, does not verify at any certified stop. Claim 7 cannot be made a proof this way.

**Why, as far as the numbers show:**
1. **The landmark components of the nonlinear zero are far from the stop state.**
   - This comes above all from D2's single-edge landmarks, which the model pins (precision ≥ 1e3, typically 1e6) to their frozen message means, up to 339 m from the solver's beliefs, where the chart is centred.
   - Weak-depth multi-observation landmarks move up to 1.6 km.
   - The 0.1 m clip keeps the candidate far from those zeros, so option A fails.
   - Option B's radii cover them, but the D_l-norm boxes then include points where the stereo factor is not smooth, so the enclosures are unbounded.
2. **The pose boxes c·E_i would need c ≥ 2–7** to contain the poses of x̂. This is because E_i bounds the distance to the linear solution, not to the nonlinear zero.

**Independently of T1:** on KITTI 04, 42.5 % of poses are farther than ηS_lo from x̂. With the D2 nonlinear model, a certified stop is therefore not a statistical finish on that segment. Stage 2a's necessary check (with the 0.1 m cap) passed there, so it was not sufficient.

**Not attempted:** a diagnostic Newton run without D2's single-edge pins. It diverged (single-observation landmarks become nearly free), so I removed it and report nothing from it.

**Decisions this needs from you:**
- (a) **Which nonlinear equation to certify.** D2's full-model F pins single-edge landmarks at their frozen messages. The relinearizing solver instead keeps linearizing their factors at the landmark's belief, so the full-model zero may not be the solver's fixed point for the poses observing them.
- (b) **Whether to apply T1 to the pose-only problem, with landmarks eliminated nonlinearly.** Each landmark would be enclosed by interval Newton on its own 3-dimensional equation over the pose box. That removes the landmark boxes, which are what failed here.
- (c) **The pose radii.** Pose boxes would need c ≥ 7 on KITTI 06.

## 6. Compute used (minutes)

- **Builds:** about 3 (two rebuilds of `certify_stereo`, each about 1).
- **Per stop:**

  | Stop | Checks | Newton | T1 |
  |---|---|---|---|
  | KITTI 06 | 0.9 | 0.4 | 3.4 and 3.5 (rerun after the Δ = ∞ correction for flagged factors) |
  | KITTI 03 | 1.0 | 0.6 | 4.0 |
  | KITTI 04 | 1.0 | 0.7 | 4.0 |
  | KITTI 10 | 0.9 | 0.5 | 3.3 |

- **Diagnosis:** about 2. Development and validation: about 12.
- **Equivalence:** 5.6. The default path is byte-identical: 12 of 12 (`results/stage2b/equivalence.txt`).
- **Compute rule:** no single run exceeded 4.0 minutes, each was capped at 6 by `timeout`, and each stop's runs were checkpointed by their own outputs. KITTI 06 ran first as the pilot. No run could exceed six minutes (T1 is ten enclosure calls of 2–3 s plus a fixed setup), so no one-tenth versions were needed.

## 7. Files

- **C++:**
  - `certify/cpp/t1_ad.h` (new): intervals, nested duals, series with rigorous remainders, the stereo and GP errors;
  - `certify/cpp/t1_server.h` (new);
  - `certify/cpp/certify_stereo.cpp`: `--t1-export`, `--t1-dir`.
- **Python:**
  - `certify/t1.py` (new): the D2 full-model problem, the client, the Schur solve;
  - `scripts/stage2b_checks.py`, `stage2b_newton.py`, `stage2b_t1.py`, `stage2b_diagnose.py`, `stage2b_tables.py`.
- **Note:** `certify/STAGE2B.md`.
- **Results:**
  - `results/stage2b/<dataset>/`: `newton.json`, `evaluator_checks.json`, `diagnosis.json`, `t1_summary.json`, `A/` and `B/` (each with `probe_T1.csv` and `boxes.json`);
  - `results/stage2b/`: `intervals.json`, `gate_B2.json`, `tables.md`, `equivalence.txt`.
- **Git-ignored:** `exports/stage2b/<dataset>/z_star.npy` and `z_star_implemented.npy`, with logs.
- **Commit:** the commit that adds this file.
