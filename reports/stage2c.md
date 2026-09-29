# Stage 2c: the clean model

Branch `certify`, base commit `b40529c` (Stage 2b). Brief: `BRIEF_stage2c.md`. Protocol: Stage 1's (`certify/STAGE1.md`), unchanged. The clean model and the tooling: `certify/STAGE2C.md`. Settings as fixed in §0 of the brief:
- δ = 0.003 (physical), R = 1, per-variable linearization points;
- η = 0.1, S_lo from the true-Jacobian Fisher information (δ_s = 0.05, q₁₂);
- decision-driven patches with the L-based continue exit;
- evaluation every 10 rounds, 300-round cap;
- p_L = 1e-6 on the remaining landmarks;
- anchor precision 1e4.

## 1. Five-line summary

1. **The cleaning is exactly D2's single-edge set, and ChArUco is unaffected.**
   - Removed: 1,397, 533, 463 and 545 single-observation landmarks with their factors (KITTI 06, 03, 04, 10; 30, 12, 10 and 13 % of the landmarks), and 4,243 on the 161-keyframe prefix.
   - Every ChArUco marker has at least 3 pose–marker factors, so ChArUco is not rerun.
   - The anchor (1e4·I on the first keyframe's initial pose) is in the solver's graph from round 1, and in the certificate's model.
2. **The anchor check does not come out at rounding level.**
   - At the round-100 state, raising the anchor from 1e4 to 1e6 moves the linear solution by up to **5.7 / 0.20 / 0.47 / 2.9 × ηS_lo** (06, 03, 04, 10). λ_min of the pose system is the anchored gauge: 0.07–0.20 at 1e4, 3–8 at 1e6.
   - The cause is the weak landmark prior p_L. It is absolute (it holds landmarks at their initial positions), and the anchor must balance it. Without p_L the change is ≤ 1.6e-3 × ηS_lo.
   - **The same prior separates the certificate's model from the solver's own graph**, which has no p_L. Their solutions differ by up to **8.7 / 0.28 / 0.57 / 3.0 × ηS_lo**, which is more than ηS_lo for 39 and 37 of 40 poses on 06 and 10.
3. **No certified stop, on any segment or variant, by round 300.**
   - No pose passes at any evaluation, even at η = 0.5, and the oracle agrees: 0 poses pass ‖e_i‖ ≤ ηS_lo,i.
   - Validity holds throughout: 0 violations in 18,580 certified pose-evaluations (8,920 on the segments, 9,660 on the prefix); the worst margin is +2.8e-10 (tolerance 1e-9); the Loewner checks are at rounding level (≥ −1.9e-9 on the segments, −1.8e-8 on the prefix at cond 5.8e10).
   - **Corrected variant:** ATE is as on the old model (0.08–0.26 m). But at round 300 the oracle error is 8–333 × ηS_lo (median 15–130), and it grows along the window away from the anchor. The largest belief change per round decays by only 0.991–0.996 per round.
   - **Shipped variant:** diverges on KITTI 03 (ATE 200 m at round 70, non-finite from round 150).
4. **Every pose is unfinished at every budget.** Steps 3.2–3.3 cannot be formed.
   - At every budget from 10 to 300 rounds, 100 % of poses are unfinished, in both senses, for both variants and all four segments (the old model had 0–97.5 % at 100 rounds).
   - "Rounds needed" is therefore undefined on every segment, so the common budget, the held-out budgets and all their flop ratios do not exist.
5. **Gate C is not met: 0 of 4.** There is no certified stop, so the Newton check has no state to start from.
   - **The 161-keyframe prefix (Step 5)** tells the same story: no stop on either variant, and 100 % unfinished at every budget. On the old model the corrected prefix stopped at round 130.
   - At round 300 the corrected prefix is 13–1,008 × ηS_lo from its optimum (median 312). The shipped prefix is not converging.

## 2. Step 1: the clean model

### 2.1 Counts (`results/stage2c/cleaning.json`)

| Dataset | Landmarks before | Removed (one observation) | Remaining | Stereo factors before | Removed | Remaining | Poses left without observations |
|---|---|---|---|---|---|---|---|
| KITTI 06 (40) | 4,679 | 1,397 | 3,282 | 11,487 | 1,397 | 10,090 | 0 |
| KITTI 03 (40) | 4,628 | 533 | 4,095 | 13,807 | 533 | 13,274 | 0 |
| KITTI 04 (40) | 4,652 | 463 | 4,189 | 15,267 | 463 | 14,804 | 0 |
| KITTI 10 (40) | 4,228 | 545 | 3,683 | 12,680 | 545 | 12,135 | 0 |
| KITTI 06 prefix (161) | 17,937 | 4,243 | 13,694 | 50,480 | 4,243 | 46,237 | 0 |

- **The removed sets match D2's single-edge landmarks:** the counts equal Stage 2b's (1,397 / 533 / 463 / 545).
- **The clean exports contain 0 single-edge variables**, so D2's elimination of frozen messages no longer applies anywhere.
- **The remaining landmarks** have 2 observations (1,916 / 2,097 / 2,072 / 1,853) or more.
- **ChArUco** (`results/stage2c/charuco_markers.json`): 24 markers with 3–51 pose–marker factors each (the minimum is 3, for 2 markers). None has one factor, so ChArUco is unaffected and is not rerun.

### 2.2 The anchor

- **The factor:** `SE3PoseVelPrior` `PRIOR_ANCHOR_POSE_0.000000000`, on the first keyframe at its initial ORB-SLAM3 pose, precision 1e4·I on position and rotation.
- **Where it lives:** in the solver's graph from round 1, and exported like every other factor. The certificate's model keeps it, as it keeps ChArUco's `PRIOR_ANCHOR`.
- **Its Jacobian** is G-solver's. Central differences agree to ≤ 4.4e-6 (worst: KITTI 10, rotation block), within the Stage 1 requirement of 1e-5. The remainder comes from the O(θ²) small-angle branch of `jacob_dlogv_dv_SO3`.

### 2.3 The anchor check (§1.4; `scripts/stage2c_anchor_check.py`)

**Method.**
- **State:** the corrected variant's evaluation after round 100 (frozen charts, as the certificate sees it).
- **Solves:** the linear system is solved twice, with the anchor's precision at 1e4 and at 1e6. Two systems:
  - the full model H e = g (poses and landmarks, with p_L; sparse LU; relative residual ≤ 4e-14);
  - the certificate's reduced pose model S.
- **Reported:** the change in each pose's solution, relative and in D_i-units against ηS_lo,i, and λ_min(H) and λ_min(S).

| Dataset | Pose solution change, 1e4 → 1e6: max relative (worst pose) | Max change ÷ ηS_lo | Landmark: max relative | λ_min(H), 1e4 / 1e6 | λ_min(S), 1e4 / 1e6 |
|---|---|---|---|---|---|
| KITTI 06 | 5.2e-2 (POSE_5.299899) | **5.69** | 0.152 | 1.0e-6 / 1.0e-6 | 0.086 / 7.52 |
| KITTI 03 | 8.1e-4 | 0.20 | 0.013 | 1.0e-6 / 1.0e-6 | 0.128 / 8.19 |
| KITTI 04 | 1.7e-3 | 0.47 | 0.011 | 1.5e-6 / 1.5e-6 | 0.069 / 5.54 |
| KITTI 10 | 0.148 (POSE_10.275010) | **2.85** | 0.531 | 1.0e-6 / 1.0e-6 | 0.204 / 2.96 |

- **The expectation of rounding level is not met.** On 06 and 10 the pose solution moves by more than the pass threshold itself.
- **What the eigenvalues say:**
  - λ_min(S) scales with the anchor (×15–90), so the weakest direction of the pose system is the gauge that only the anchor holds.
  - λ_min(H) is at p_L's level (1.0–1.5e-6): a landmark with essentially no depth information, held only by its prior.
  - λ_max(S) is 0.7–1.5e8, so cond(S) is about 1e9 at 1e4.

**Attribution** (diagnostic; the same solves without p_L):

| Dataset | Without p_L: max relative change; max ÷ ηS_lo | λ_min(H) without p_L | Certificate's model (p_L) vs solver's graph (no p_L), anchor 1e4: max ÷ ηS_lo; median; poses above ηS_lo | Largest landmark difference |
|---|---|---|---|---|
| KITTI 06 | 1.1e-6; 1.5e-4 | 4.1e-10 | **8.71**; 3.14; **39 of 40** | 64 km |
| KITTI 03 | 5.8e-8; 1.4e-5 | 1.8e-8 | 0.28; 0.14; 0 | 1.8 km |
| KITTI 04 | 5.5e-6; 1.6e-3 | 5.5e-7 | 0.57; 0.23; 0 | 242 m |
| KITTI 10 | 2.7e-6; 7.5e-5 | 2.8e-13 | **3.01**; 1.47; **37 of 40** | 2,470 km |

- **The claim holds for the model without p_L.** Without the landmark priors, the anchor's strength does not change the solution beyond 1e-6 relative. That residue is not rounding level; plausibly it is the gauge invariance broken at the level of the corrected Jacobians' own approximations (about 1e-6).
- **The change is p_L's.** It holds every landmark at its initial position, and on the clean model the only other absolute term is the one anchored pose. Its pull along the gauge (a torque through landmarks tens of metres away) is balanced by the anchor, so a stronger anchor gives a different solution.
- **The consequence matters more than the check:**
  - the solver's graph has no p_L, but the certificate's model (D2(b), "unchanged") does;
  - on 06 and 10 their solutions are more than ηS_lo apart for almost every pose;
  - so a certified stop there requires the solver's state to be where the solver is not going, even after full convergence.
  - On the old model the fake pins (precision about 1e6 on 1,397 landmarks) dwarfed p_L, which is why this never showed.

## 3. Step 2: Stage 1 on the clean model (300 rounds, both variants)

Everything is exactly as in Stage 1: the same drivers, evaluator, logs and invariants, with evaluations every 10 rounds. Files: `results/stage2c/<dataset>/<variant>/` (the Stage 1 file set).

### 3.1 Invariants

| Dataset | Variant | Pose-evaluations | Violations upper / lower | Worst upper margin (‖e‖−E)/E (round, pose) | Worst lower margin (L−‖e‖)/‖e‖ | Loewner min (rounds 10, 150, 300) | Certified ≠ oracle decisions | Rounds not certifiable |
|---|---|---|---|---|---|---|---|---|
| KITTI 06 | corrected | 1,200 | 0 / 0 | 2.1e-12 (r10, POSE_2.285652) | 1.9e-12 | −1.2e-9 | 0 | 0 |
| KITTI 06 | shipped | 1,200 | 0 / 0 | 9.7e-12 (r10, POSE_2.285652) | 4.5e-12 | −6.1e-10 | 0 | 0 |
| KITTI 03 | corrected | 1,200 | 0 / 0 | 7.7e-11 (r10, POSE_0) | 4.6e-11 | −2.1e-10 | 0 | 0 |
| KITTI 03 | shipped | 520 | 0 / 0 | 2.8e-10 (r20, POSE_7.141701) | 2.8e-10 | −3.8e-10 (r10 only) | 0 | 17 (r140: A0 fails; r150–300: non-finite) |
| KITTI 04 | corrected | 1,200 | 0 / 0 | 7.7e-13 (r10, POSE_0) | 6.2e-13 | −1.1e-9 | 0 | 0 |
| KITTI 04 | shipped | 1,200 | 0 / 0 | 1.6e-11 (r10, POSE_3.854131) | 7.3e-12 | −1.9e-9 | 0 | 0 |
| KITTI 10 | corrected | 1,200 | 0 / 0 | 4.5e-12 (r20, POSE_9.652155) | 5.1e-12 | −3.1e-10 | 0 | 0 |
| KITTI 10 | shipped | 1,200 | 0 / 0 | 1.0e-11 (r50, POSE_10.586510) | 1.6e-11 | −3.1e-10 | 0 | 0 |

- **Validity (L_i ≤ ‖e_i‖_{D_i} ≤ E_i at 1e-9 relative): 0 violations** in 8,920 certified pose-evaluations.
  - Positive margins occur, all rounding level: up to +2.8e-10 upper and +2.8e-10 lower. They are reported, and no tolerance was relaxed.
  - The Loewner check of the patch lower bound against the exact marginal is at rounding level: min −1.9e-9 relative, KITTI 04 shipped, round 10, POSE_6.040889. That is the same order as ChArUco's −5.4e-9 in Stage 1. On the prefix it is −1.8e-8 (§6).
- **The certificate's decision equals the oracle's everywhere:** every pose fails both, at every evaluation.
- **Every growth exit is "continue"** (the certified lower bound already exceeds ηS_lo).
- **Every base patch is the whole window (40 poses) from the first evaluation.** On the old model the median was 22–32 poses.
  - Without the pins, no smaller patch meets the leak criterion (row sum ≤ 0.3), so the certificate degenerates to a global solve. It is exact: E/‖e‖ = 1.000000 (median).
- **KITTI 03 shipped diverges.**
  - At round 140 the reduced matrix is numerically indefinite (Theorem A's precondition A0 fails), and from round 150 the state is non-finite.
  - These rounds are logged as such (`certify/STAGE2C.md`) and left out of the validity counts.

### 3.2 Stops

| Dataset | Variant | Certified | Fixed budget 100: move median / p95; ATE | Change-based τ = 1e-2 / 1e-3 / 1e-4 | Largest belief change at round 300 (D) |
|---|---|---|---|---|---|
| KITTI 06 | corrected | none by 300 | 27.5 / 75.2; 0.080 m | none / none / none | 0.355 |
| KITTI 06 | shipped | none by 300 | 51.7 / 116; 0.072 m | none / none / none | 0.515 |
| KITTI 03 | corrected | none by 300 | 106 / 242; 0.284 m | none / none / none | 5.08 |
| KITTI 03 | shipped | none (diverged) | 4,200 / 79,300; 11.6 m | spurious at r151 on the NaN state (not a stop) | – |
| KITTI 04 | corrected | none by 300 | 37.2 / 104; 0.129 m | none / none / none | 0.025 |
| KITTI 04 | shipped | none by 300 | 66.6 / 185; 0.197 m | none / none / none | 0.112 |
| KITTI 10 | corrected | none by 300 | 12.4 / 20.8; 0.081 m | none / none / none | 6.12 |
| KITTI 10 | shipped | none by 300 | 102 / 193; 0.196 m | none / none / none | 0.807 |

- **The η sensitivity (0.05, 0.1, 0.2, 0.5) is empty:** no η gives a stop, and the pass fraction at round 300 is 0.00 for every η.
- **The pass-fraction curve** is 0.00 at every evaluation, for every run (`pass_curve.csv`).
- **The change-based criterion** fires nowhere by round 300, except KITTI 03 shipped: there the logged change is 0 on the NaN state, because the max skips NaN, and the analysis marks that as not a stop.
- **KITTI 04 corrected shows why such a criterion would mislead:** its largest per-round change is down to 0.025 D-units at round 300, while every pose is still 7.8–185 × ηS_lo from the solution of its own linear model.

### 3.3 What the runs look like (`results/stage2c/dynamics.json`; descriptive only)

| Dataset | Variant | Oracle ‖e‖/ηS_lo at round 300: median (min–max) | First 10 poses / last 10 (median) | Largest belief change at rounds 100 / 200 / 300 | Decay per round (fit, rounds 100–300) | Relinearized at round 300: poses / landmarks | Rigid-motion share of the error at round 100 |
|---|---|---|---|---|---|---|---|
| KITTI 06 | corrected | 43.8 (11.4–116) | 17.2 / 94.6 | 2.09 / 0.886 / 0.355 | 0.9913 | 1 / 208 | 19 % |
| KITTI 03 | corrected | 130 (9.2–333) | 27.7 / 196 | 23.7 / 10.2 / 5.08 | 0.9924 | 32 / 3,025 | 0.7 % |
| KITTI 04 | corrected | 51.4 (7.8–185) | 14.5 / 161 | 0.130 / 0.032 / 0.025 | 0.9934 | 1 / 74 | 6 % |
| KITTI 10 | corrected | 15.5 (11.2–23.3) | 16.0 / 19.7 | 14.6 / 8.32 / 6.12 | 0.9959 | 33 / 2,798 | 31 % |
| KITTI 06 | shipped | 94.1 (28.9–236) | 36.8 / 181 | 0.593 / 0.591 / 0.515 | 0.9966 | 1 / 321 | – |
| KITTI 03 | shipped | diverged (r130: 25,200) | – | – | – | – | – |
| KITTI 04 | shipped | 113 (22.8–345) | 33.1 / 308 | 0.263 / 0.145 / 0.112 | 0.9997 | 0 / 358 | – |
| KITTI 10 | shipped | 214 (126–283) | 192 / 210 | 11.6 / 1.04 / 0.807 | 0.9851 | 15 / 1,017 | – |

- **The trajectories are as accurate as before** (ATE at round 300, corrected variant: 0.078–0.258 m, against 0.082–0.248 m on the old model).
- **But they are far from the optimum of their own problem.**
  - The error grows along the window away from the anchored first pose.
  - It is mostly not a rigid motion (0.7–31 % of it is).
  - It shrinks very slowly.
  - This is the pattern of a weakly held, long-wavelength deformation, which GBP propagates slowly once no landmark pins the map locally.
- **Two of the four segments are still relinearizing most of the graph every round at round 300:** 03 and 10 re-freeze 2,800–3,000 landmarks a round.
- **Whether a longer run would stop cannot be told from these runs.** Two readings fit: very slow convergence, or relinearization churn under the fixed δ = 0.003. Separating them needs a relinearization experiment, which is out of scope.
  - Independently, the p_L gap (§2.3) puts the certificate's target more than ηS_lo away from the solver's target for most poses on 06 and 10.

### 3.4 Costs

| Dataset | Variant | GBP flops per round (median) | Belief-minus-incoming baseline per round | Certificate flops per evaluation: median (max, first evaluation) | Change criterion flops per round | Evaluator s per evaluation (median) |
|---|---|---|---|---|---|---|
| KITTI 06 | both | 1.42e9 | 2.25e8 | 5.17e8 (4.4e9) | 1.64e6 | 0.5 |
| KITTI 03 | both | 2.39e9 | 2.94e8 | 5.17e8 (4.3e9) | 2.14e6 | 0.6–0.7 |
| KITTI 04 | both | 2.95e9 | 3.28e8 | 5.17e8 (5.0e9) | 2.38e6 | 0.7 |
| KITTI 10 | both | 2.07e9 | 2.69e8 | 5.17e8 (4.3e9) | 1.96e6 | 0.6 |

- **GBP is 5–21 % cheaper per round than on the old model** (06: 1.80e9 → 1.42e9), since there are fewer factors.
- **The certificate's cost per evaluation is identical on every segment.** Its patches are all the whole 40-pose window, and one shared factorization serves every pose.
- Per-run totals and the certificate's flops at every evaluation are in `flops.csv` and `eval_summary.csv`.

## 4. Step 3: the numbers the paper uses

### 4.1 Budgets (`results/stage2c/budgets.csv`)

Each cell: the fraction of poses unfinished in the current / relinearized problem; the median and p95 remaining Gauss–Newton move (D); aligned keyframe ATE.

| Dataset | Variant | 10 | 50 | 100 | 200 | 300 |
|---|---|---|---|---|---|---|
| KITTI 06 | corrected | 1.00 / 1.00; 30.5, 77.6; 0.087 m | 1.00 / 1.00; 28.4, 76.9; 0.083 m | 1.00 / 1.00; 27.5, 75.2; 0.080 m | 1.00 / 1.00; 27.2, 71.9; 0.078 m | 1.00 / 1.00; 27.6, 69.7; 0.077 m |
| KITTI 06 | shipped | 1.00 / 1.00; 44.6, 111; 0.095 m | 1.00 / 1.00; 51.4, 117; 0.079 m | 1.00 / 1.00; 51.7, 116; 0.072 m | 1.00 / 1.00; 52.3, 116; 0.066 m | 1.00 / 1.00; 53.2, 116; 0.063 m |
| KITTI 03 | corrected | 1.00 / 1.00; 98.4, 258; 0.454 m | 1.00 / 1.00; 115, 244; 0.332 m | 1.00 / 1.00; 106, 242; 0.284 m | 1.00 / 1.00; 113, 229; 0.266 m | 1.00 / 1.00; 113, 220; 0.258 m |
| KITTI 03 | shipped | 1.00 / 1.00; 2,930, 17,800; 1.84 m | 1.00 / 1.00; 53,700, 91,700; 3.71 m | 1.00 / 1.00; 4,200, 79,300; 11.6 m | diverged | diverged |
| KITTI 04 | corrected | 1.00 / 1.00; 38.1, 104; 0.138 m | 1.00 / 1.00; 37.2, 106; 0.131 m | 1.00 / 1.00; 37.2, 104; 0.129 m | 1.00 / 1.00; 37.3, 102; 0.128 m | 1.00 / 1.00; 36.9, 99.7; 0.127 m |
| KITTI 04 | shipped | 1.00 / 1.00; 118, 300; 0.232 m | 1.00 / 1.00; 61.2, 188; 0.202 m | 1.00 / 1.00; 66.6, 185; 0.197 m | 1.00 / 1.00; 70.9, 183; 0.195 m | 1.00 / 1.00; 71.5, 180; 0.193 m |
| KITTI 10 | corrected | 1.00 / 1.00; 24.6, 41.1; 0.077 m | 1.00 / 1.00; 13.7, 27.9; 0.077 m | 1.00 / 1.00; 12.4, 20.8; 0.081 m | 1.00 / 1.00; 11.4, 15.7; 0.083 m | 1.00 / 1.00; 10.9, 15.5; 0.083 m |
| KITTI 10 | shipped | 1.00 / 1.00; 477, 689; 0.389 m | 1.00 / 1.00; 104, 228; 0.219 m | 1.00 / 1.00; 102, 193; 0.196 m | 1.00 / 1.00; 119, 171; 0.160 m | 1.00 / 1.00; 132, 162; 0.140 m |

On the old model (Part A), the unfinished fraction at 100 rounds was 5 / 97.5 / 90 / 0 % (corrected) and at 300 rounds 0 % on all four corrected segments.

### 4.2 Equal-guarantee comparison and leave-one-out (`equal_guarantee.csv`, `leave_one_out.csv`)

**As §3.4 of the brief requires:** these two are evaluation protocols defined after the pre-registered Stage 1 comparison. Their definitions, exactly as given:
- **Rounds needed** = the first evaluation round at which every pose passes the oracle test ‖e_i‖_{D_i} ≤ ηS_lo,i (resolution 10 rounds).
- **Common budget** = the largest rounds needed over the four segments.
  - Per segment: the certified stop round; its total flops (GBP up to that round plus the certificate at every evaluation round up to it); the common budget's total flops (GBP only); the ratio; and the same with the belief-minus-incoming GBP baseline.
- **Leave-one-out:** for each segment in turn, the held-out budget = the largest rounds needed over the other three segments.
  - Whether it finishes the held-out segment (the fraction of poses unfinished at that round, in both senses); the certified stop's total flops against the held-out budget's; the ratio.

**Result: every quantity is undefined.**
- On every segment, no evaluation up to round 300 has every pose passing the oracle test. The oracle pass fraction is 0.00 at all 30 evaluations of all four corrected runs.
- So rounds needed are > 300 on all four segments, and neither the common budget nor any held-out budget exists within the runs.
- No segment has a certified stop either.
- Both CSVs record this per segment; no ratio is computed.

## 5. Step 4: the full nonlinear check and Gate C

**Gate C (pre-registered):** on at least three of the four segments, at least 95 % of poses at the certified stop are within ηS_lo,i of x̂.

| Dataset | Certified stop | Poses within ηS_lo of x̂ | Passes |
|---|---|---|---|
| KITTI 06 | none by round 300 | – | no |
| KITTI 03 | none by round 300 | – | no |
| KITTI 04 | none by round 300 | – | no |
| KITTI 10 | none by round 300 | – | no |

- **Gate C: not met, 0 of 4** (`results/stage2c/gate_C.json`). There is no certified stop, so there is no stop state to start Newton from. `scripts/stage2c_newton.py` records this in `results/stage2c/<dataset>/newton.json`.
- **The tooling is ready and checked on a clean export.**
  - At the round-100 beliefs of KITTI 06, the evaluator's F(0), anchor included, equals the solver's own JᵀΛe to 2.4e-8 relative (the anchor rows to 7e-9).
  - An undamped Newton iteration from that non-stop state diverges within three steps (first step about 1.8 km on weakly determined landmarks, then stereo factors behind the camera). This is a tooling observation only.

### 5.1 The old model against the clean model (`results/stage2c/comparison_old_vs_clean.csv`)

| Quantity (corrected variant unless noted) | KITTI 06: old → clean | KITTI 03 | KITTI 04 | KITTI 10 |
|---|---|---|---|---|
| Landmarks / stereo factors | 4,679 / 11,487 → 3,282 / 10,090 | 4,628 / 13,807 → 4,095 / 13,274 | 4,652 / 15,267 → 4,189 / 14,804 | 4,228 / 12,680 → 3,683 / 12,135 |
| Certified stop, corrected | r130 → none | r250 → none | r170 → none | r50 → none |
| Certified stop, shipped | r150 → none | none → none (diverges) | none → none | r240 → none |
| Unfinished at round 100 (current problem) | 5 % → 100 % | 97.5 % → 100 % | 90 % → 100 % | 0 % → 100 % |
| Unfinished at round 300 | 0 % → 100 % | 0 % → 100 % | 0 % → 100 % | 0 % → 100 % |
| Median remaining GN move, round 300 (D) | 0.022 → 27.6 | 0.131 → 113 | 0.064 → 36.9 | 0.0047 → 10.9 |
| Aligned keyframe ATE, round 300 (m) | 0.0817 → 0.0775 | 0.248 → 0.258 | 0.133 → 0.127 | 0.0853 → 0.0834 |
| Base patch size (median) | 23 → 40 (whole window) | 24 → 40 | 32 → 40 | 23 → 40 |
| GBP flops per round (median) | 1.80e9 → 1.42e9 | 2.57e9 → 2.39e9 | 3.11e9 → 2.95e9 | 2.24e9 → 2.07e9 |
| Newton check: stop state | r130 → none | r250 → none | r170 → none | r50 → none |
| Newton check: poses within ηS_lo of x̂ | 95 % → – | 97.5 % → – | 57.5 % → – | 100 % → – |
| Newton check: pose distance ÷ E_i, median; max | 3.78; 6.76 → – | 0.58; 2.23 → – | 1.50; 2.90 → – | 0.89; 1.33 → – |
| Newton check: largest landmark move | 1,620 m → – | 1,050 m → – | 154 m → – | 175 m → – |

**What removing the fake information did.**
- Every "finish" that Stages 1–2b found on these segments depended on the single-observation landmarks' frozen messages. These held the map locally at about 1e6 precision, so the reduced pose problem was local (patches of 22–32 poses) and fast for GBP.
- Without them the same solver, with the same settings, reaches trajectories that are as accurate against ground truth. But within 300 rounds they are nowhere near the optimum of the problem it is solving, by 8–333 × ηS_lo.
- The Stage 2b result on 04 (57.5 % within ηS_lo of x̂) was an early sign.

## 6. Step 5: the 161-keyframe prefix

Both variants ran 300 rounds in checkpointed segments of 10 rounds, each validity-gated. There were 29 checkpoints per run, all "continue", and the longest segment was 205 s.
- The first two evaluations take 135–190 s, because the patches grow from about 90 poses to the whole window.
- Later evaluations take about 4 s, because the patch is the whole window and one factorization is shared.

| Variant | Certified stop | Validity: pose-evaluations, violations | Worst margins upper / lower | Loewner min | Unfinished (current / relinearized) at 10, 50, 100, 200, 300 | Median remaining move at 10 → 300 | ATE at 10 → 300 |
|---|---|---|---|---|---|---|---|
| corrected | none by 300 | 4,830, 0 / 0 | 2.5e-11 / 4.3e-11 | −8.8e-9 | 1.00 / 1.00 at every budget | 258 → 240 | 0.389 → 0.342 m |
| shipped | none by 300 | 4,830, 0 / 0 | 1.9e-10 / 2.0e-10 | −1.8e-8 | 1.00 / 1.00 at every budget | 2,424 → 1,251 | 0.463 → 0.832 m |

- **Round 10:** the base recursion's bound on the spectral radius is 6.2 > 1, so the certificate declines at every pose (E = ∞; the 161 "no_c0" growth exits). That is the certificate refusing, not a violation.
- **The Loewner minima occur where the base patch is the whole window.** There the patch bound and the exact marginal are the same matrix in exact arithmetic, and the margin is the rounding difference of two inverses. cond(S_F) = 5.8e10 at round 100, so ε·cond ≈ 1e-5.
- **Corrected at round 300:**
  - the oracle error is median 312 × ηS_lo (13–1,008), first ten poses 20, last ten 414;
  - the largest belief change decays by 0.9915 per round (38 → 6.9 D-units from round 100 to 300);
  - 30 poses and 3,907 landmarks still relinearize per round;
  - 2 % of the round-100 error is a rigid motion.
- **Shipped:** the largest belief change grows from 449 to 656 D-units between rounds 100 and 300, and the ATE worsens from 0.46 to 0.83 m. It is not converging.
- **Newton check (§5 of the brief):** not run, because there is no certified stop (`results/stage2c/prefix/newton.json`).
- **The old model, for comparison:**
  - corrected: stop at round 130; unfinished 1.2 % at round 100 and 0 % at 300; ATE 0.382 m;
  - shipped: no stop; 9.3 % unfinished at round 300; ATE 0.383 m;
  - per round: GBP 8.56e9 flops, now 7.30e9; the certificate 9.5e9 per evaluation, now 3.4e10.

## 7. Worst cases

| Quantity | Worst value | Where |
|---|---|---|
| Validity, upper: (‖e‖ − E)/E (tolerance 1e-9) | +2.8e-10 | KITTI 03 shipped, round 20, POSE_7.141701 |
| Validity, lower: (L − ‖e‖)/‖e‖ (tolerance 1e-9) | +2.8e-10 | KITTI 03 shipped, round 20, POSE_6.002373 |
| Patch lower bound vs exact marginal, Loewner margin (≥ 0 required) | −1.8e-8 (rounding: the patch is the whole window and cond(S_F) = 5.8e10); on the 40-keyframe segments −1.9e-9 | prefix shipped, round 300, POSE_0.207793; segments: KITTI 04 shipped, round 10, POSE_6.040889 |
| Oracle ‖e_i‖/ηS_lo at the last certifiable evaluation, smallest (the pose closest to passing) | 7.78 | KITTI 04 corrected, round 300, POSE_0 (the anchored pose) |
| Oracle ‖e_i‖/ηS_lo, largest | 333 on the segments, corrected; 1.3e5 on KITTI 03 shipped at round 130, before it diverged | KITTI 03 corrected, round 300 |
| Jacobian vs central differences, corrected (requirement 1e-5) | 4.4e-6 | KITTI 10, anchor, rotation block |
| Anchor 1e4 → 1e6: pose solution change ÷ ηS_lo (with p_L) | 5.69 | KITTI 06 |
| Anchor 1e4 → 1e6: relative pose change (with p_L) | 0.148 | KITTI 10, POSE_10.275010 |
| Anchor 1e4 → 1e6: relative pose change (without p_L) | 5.5e-6 | KITTI 04, POSE_7.394135 |
| Certificate's model vs solver's graph (p_L gap) ÷ ηS_lo | 8.71 | KITTI 06 |
| p_L gap, largest landmark difference | 2,470 km | KITTI 10 (a landmark with no depth information: λ_min(H) = 2.8e-13 without p_L) |
| Prefix: oracle ‖e_i‖/ηS_lo at round 300, corrected (smallest; largest) | 13.1; 1,008 | KITTI 06 prefix |

## 8. What could not be done, plainly

- **No certified stop exists on the clean model** within the 300-round cap, on any segment or variant (nor on the prefix).
  - So nothing in Step 3.2, Step 3.3 or Step 4 could be computed: no rounds needed, no common or held-out budget, no flop ratio, and no Newton check at a stop.
  - Gate C is not met (0 of 4). No setting was changed to obtain a stop.
- **The anchor check's expectation fails.** With p_L = 1e-6 in the model, the anchor's precision changes the solution by up to 5.7 × ηS_lo, and the certificate's model and the solver's graph differ by up to 8.7 × ηS_lo (39 and 37 of 40 poses above ηS_lo on 06 and 10).
  - p_L is a fixed setting ("unchanged from D2"), so it was not changed. This needs a decision (§9).
- **Why GBP does not reach the optimum within 300 rounds is not settled.** The logs fit both slow convergence along weakly held long-wavelength modes and relinearization churn (03 and 10 re-freeze about 3,000 landmarks per round at round 300). Telling them apart needs a relinearization experiment, which is out of scope.
- **The shipped variant diverges on KITTI 03** (ATE 200 m at round 70, non-finite from round 150).
  - The Stage 1 evaluator could not handle this and aborted the run twice, at round 140 (an indefinite matrix in the accuracy step) and at round 150 (an empty model).
  - I made it record such rounds (`certify/STAGE2C.md`) and reran; the other seven runs are not affected by the change.
- **The prefix one-tenth versions** first ran unchecked (one evaluation takes 150–200 s there, so a 30-round run exceeds six minutes) and then in parallel (both hit the six-minute segment cap before their first evaluation).
  - Both attempts were stopped. The runs used are the sequential, checkpointed ones (segments ≤ 205 s).
- **ChArUco** was not rerun (no single-edge markers), as the brief directs.

## 9. Decisions this needs

1. **p_L.** The certificate's model carries landmark priors at the initial positions that the solver's graph does not have. On the clean model they are no longer negligible:
   - they set the solution's gauge against the anchor;
   - they put the certificate's target more than ηS_lo from the solver's target on two of four segments.
   
   The options are to add the same priors to the solver's graph, or to remove them from the certificate's model. The second needs a treatment of landmarks with no depth information: λ_min(H) reaches 2.8e-13 without them.
2. **The iteration budget and the anchor's strength as conditioning.** With the fixed settings, GBP on the clean model is 8–333 × ηS_lo from its optimum at round 300.
   - Should Stage 2c's protocol be rerun with a longer cap, or is "no finish within 300 rounds" the result to take forward?
   - The anchor's strength sets λ_min(S) (0.07–0.20 at 1e4), and plausibly GBP's rate. The brief treats it as conditioning only.
3. **Locality.** On the clean model the certificate's patches are the whole window. The local certificate degenerates to a global solve: 5.17e8 flops per evaluation on 40 poses, and 3.4e10 on the prefix (9.3e11 while the patches grow).

## 10. Compute used (minutes)

| Item | Minutes |
|---|---|
| Builds (driver and evaluator changes) | 1.2 |
| Step 1: counts (5 graph builds) | 0.2 |
| Smoke test, KITTI 06 corrected, 30 rounds | 0.2 |
| One-tenth versions, 30 rounds × 8 (KITTI) | 2.0 (0.17–0.31 each) |
| 300-round runs × 8 (KITTI): 06 c / s, 03 c / s, 04 c / s, 10 c / s | 1.19 / 1.39, 2.16 / 2.02, 2.29 / 2.10, 2.54 / 2.36 = 16.1 |
| KITTI 03 shipped, two aborted attempts (evaluator stopped at round 140, then 150) | 1.1 + 1.1 |
| Prefix one-tenth, first attempt (unchecked; stopped: an evaluation takes 150–200 s) | ≈ 10.5 |
| Prefix one-tenth, parallel checkpointed attempt (both hit the six-minute segment cap) | ≈ 6.5 |
| Prefix one-tenth, sequential and checkpointed: corrected / shipped | 6.5 / 6.5 |
| Prefix 300 rounds, checkpointed: corrected / shipped | 9.7 / 9.4 |
| Anchor check (4 segments; with the p_L attribution) and its preliminary tests | 4.3 + 1.2 |
| Evaluator tooling check with the anchor (Newton from a non-stop state) | 0.1 |
| Analysis, tables, dynamics, fd summaries | ≈ 2 |
| Equivalence check (default path; 12 of 12 byte-identical) | 5.0 |
| **Total** | **≈ 84** |

Nothing ran unattended past six minutes without a checkpoint and gate, apart from the first prefix one-tenth attempt. That attempt was capped by the batch timeout at six minutes per run; both of its runs were discarded.

## 11. Files

**Code:**
- `certify/cpp/certify_stereo.cpp`: `--drop-single-landmarks`, `--live-anchor`; the T1 server lists its factor types.
- `certify/cpp/t1_ad.h`, `certify/cpp/t1_server.h`: the anchor prior in the evaluator.
- `certify/t1.py`: the clean model's flags and the prior type.
- `certify/stage1.py`, `scripts/stage1_evaluator.py`: `--keep-rounds`; diverged rounds recorded.
- `scripts/stage1_run.sh`, `stage1_batch.sh`, `stage1_checkpointed.sh`: `STAGE`, `EXTRA_DRIVER`, `EXTRA_EVAL`.
- `scripts/stage1_analyse.py`: `--stage`; non-certifiable rounds; spurious change stops.
- New scripts: `scripts/stage2c_anchor_check.py`, `stage2c_tables.py`, `stage2c_newton.py`, `stage2c_dynamics.py`, `stage2c_report.py`.
- Note: `certify/STAGE2C.md`.

**Results:**
- `results/stage2c/cleaning.json` (counts, anchor check, attribution, p_L gap) and `charuco_markers.json`;
- per run, `results/stage2c/<dataset>/<variant>/`: `eval_rounds.csv`, `eval_summary.csv`, `stops.csv`, `eta_sensitivity.csv`, `pass_curve.csv`, `heuristics.csv`, `flops.csv`, `validity.json`, `run.json`, `relin_rounds.csv`, `work.csv`;
- `results/stage2c/budgets.csv`, `equal_guarantee.csv`, `leave_one_out.csv`, `gate_C.json`, `comparison_old_vs_clean.csv`, `dynamics.json`, `fdcheck.json`, `worst_cases.md`, `tables.md`, `analyse.log`, `equivalence.txt`;
- `results/stage2c/gate_A.json`: Stage 1's criterion, recomputed by the analysis script as a by-product; not part of this brief. Every segment fails, since there is no stop;
- per dataset, `results/stage2c/<dataset>/newton.json`.

**Exports** (not committed, as before): `exports/stage2c/`.

**Progress log:** `25_09_2026_PROGRESS.md`.

**Commit:** the commit that adds this file (not pushed).
