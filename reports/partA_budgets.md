# Part A: the unfinished fraction at fixed budgets

Branch `certify`, base commit `857a87d` (Stage 1). Brief: `BRIEF_stage2.md` §1. Settings as in Stage 1: δ = 0.003, R = 1, η = 0.1, true-Jacobian yardstick; variants corrected and shipped.

## 1. Five-line summary

1. **At G-solver's shipped stereo default of 10 rounds, most poses are unfinished.**
   - Corrected variant: 57.5 % (06) to 100 % (03, 04, 10) of poses are unfinished at the statistical scale.
   - Shipped variant: 100 % on all four segments.
2. **At 100 rounds**, the corrected variant still leaves 5 % (06), 97.5 % (03), 90 % (04) and 0 % (10) unfinished; the shipped variant 15–100 % (current problem) and 30–100 % (relinearized).
3. **No single budget is right for all segments.** The budget at which every pose is finished ranges from ≤ 50 rounds (corrected 10) to more than 200 (corrected 03); on the shipped variant, 03 and 04 are still unfinished at 300. The certified stop finds each segment's own round: 130, 250, 170 and 50 (corrected).
4. **The two senses coincide on the corrected variant and diverge on the shipped one.**
   - Corrected: the current-problem and relinearized fractions are identical at every budget (the two moves agree to 1 % median, up to 42 %, and fall on the same side of ηS_lo).
   - Shipped: relinearizing adds up to 45 points (KITTI 10 at 200 rounds: 25 % → 70 %). The shipped Jacobian makes the current linearization's answer differ from the relinearized one.
5. **ChArUco:** 82–83 % of poses are unfinished at its shipped budget of 103 rounds, 29–32 % at 500, and 22 % at 1,030.
   - ATE does not reflect any of this: from 50 rounds on it changes by at most 0.007 m on each KITTI segment (corrected).

## 2. Definitions

At a budget B, the state is the one after B rounds, i.e. Stage 1's evaluation after round B. A pose is unfinished:
- **(1) in the current problem** if ‖e_i‖_{D_i} > ηS_lo,i, where e is the evaluation's oracle error with respect to the exact solution of the current linearization;
- **(2) in the relinearized problem** if its remaining Gauss–Newton move at the beliefs (solver's Jacobian, D_i = S_ii; Stage 1 §3.4) exceeds ηS_lo,i.

S_lo,i is the evaluation's yardstick. "Move" is the remaining Gauss–Newton move of WP1's definition. ATE is the SE(3)-aligned keyframe ATE in metres.

**Sources.**
- **KITTI budgets** are all Stage 1 evaluation rounds, so no solver was rerun.
- **ChArUco's budget of 103** is not a multiple of 10. Both ChArUco variants were rerun for 103 rounds with one extra evaluation after round 103 (`--eval-at 103`). The rerun reproduces Stage 1's evaluations 10–100 exactly (553 of 553 pose-evaluations identical).
- **ChArUco's 200, 500 and 1,030** come from Stage 1.

## 3. Unfinished fraction: current problem / relinearized problem

### Corrected variant

| Dataset | 10 | 50 | 100 | 200 | 300 | Certified stop |
|---|---|---|---|---|---|---|
| KITTI 06 (40) | 0.575 / 0.575 | 0.150 / 0.150 | 0.050 / 0.050 | 0 / 0 | 0 / 0 | 130 |
| KITTI 03 (40) | 1.000 / 1.000 | 1.000 / 1.000 | 0.975 / 0.975 | 0.625 / 0.625 | 0 / 0 | 250 |
| KITTI 04 (40) | 1.000 / 1.000 | 0.950 / 0.950 | 0.900 / 0.900 | 0 / 0 | 0 / 0 | 170 |
| KITTI 10 (40) | 1.000 / 1.000 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 50 |

| Dataset | 103 | 200 | 500 | 1,030 | Certified stop |
|---|---|---|---|---|---|
| ChArUco islam | 0.819 / 0.819 | 0.681 / 0.681 | 0.287 / 0.287 | 0.223 / 0.223 | none by 1,030 |

### Shipped variant

| Dataset | 10 | 50 | 100 | 200 | 300 | Certified stop |
|---|---|---|---|---|---|---|
| KITTI 06 (40) | 1.000 / 1.000 | 0.175 / 0.325 | 0.150 / 0.300 | 0 / 0.150 | 0 / 0.150 | 150 |
| KITTI 03 (40) | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.775 / 0.875 | none by 300 |
| KITTI 04 (40) | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.925 / 0.950 | 0.275 / 0.275 | none by 300 |
| KITTI 10 (40) | 1.000 / 1.000 | 0.975 / 0.975 | 0.975 / 1.000 | 0.250 / 0.700 | 0 / 0.225 | 240 |

| Dataset | 103 | 200 | 500 | 1,030 | Certified stop |
|---|---|---|---|---|---|
| ChArUco islam | 0.830 / 0.830 | 0.723 / 0.723 | 0.319 / 0.319 | 0.223 / 0.223 | none by 1,030 |

At the certified stop itself both fractions are 0 on the corrected variant (Stage 1 §3.4). On the shipped variant, the relinearized fraction is 0.15 (06) and 0.25 (10) at its stops.

## 4. Remaining Gauss–Newton move (median / p95) and ATE at each budget

| Dataset | Variant | 10 | 50 | 100 | 200 | 300 |
|---|---|---|---|---|---|---|
| KITTI 06 (40) | corrected | 0.706 / 5.94; 0.0879 | 0.041 / 2.16; 0.0839 | 0.032 / 0.651; 0.0823 | 0.023 / 0.105; 0.0817 | 0.022 / 0.078; 0.0817 |
| KITTI 06 (40) | shipped | 2.12 / 14.4; 0.0982 | 0.246 / 4.49; 0.0863 | 0.195 / 1.09; 0.0825 | 0.154 / 0.633; 0.0814 | 0.113 / 0.633; 0.0814 |
| KITTI 03 (40) | corrected | 42.2 / 511; 0.338 | 22.8 / 55.1; 0.255 | 8.40 / 16.4; 0.245 | 1.01 / 1.74; 0.247 | 0.131 / 0.203; 0.248 |
| KITTI 03 (40) | shipped | 136 / 5,280; **43.7** | 103 / 314; 0.246 | 28.2 / 196; 0.227 | 4.24 / 123; 0.245 | 1.45 / 123; 0.247 |
| KITTI 04 (40) | corrected | 8.74 / 13.1; 0.132 | 3.34 / 4.83; 0.132 | 1.17 / 1.80; 0.133 | 0.235 / 0.375; 0.133 | 0.064 / 0.093; 0.133 |
| KITTI 04 (40) | shipped | 27.1 / 133; 0.200 | 11.0 / 34.4; 0.144 | 4.84 / 15.1; 0.136 | 1.41 / 3.89; 0.134 | 0.415 / 1.06; 0.133 |
| KITTI 10 (40) | corrected | 1.73 / 3.37; 0.0851 | 0.454 / 0.786; 0.0850 | 0.082 / 0.258; 0.0853 | 0.011 / 0.024; 0.0853 | 0.0047 / 0.0128; 0.0853 |
| KITTI 10 (40) | shipped | 13.2 / 25,300; 0.660 | 23.2 / 104; 0.168 | 4.89 / 19.5; 0.101 | 0.738 / 1.05; 0.0904 | 0.270 / 0.862; 0.0901 |
| ChArUco (103 · 200 · 500 · 1,030) | corrected | 1.42 / 19.0 | 1.14 / 13.9 | 0.566 / 10.2 | 0.321 / 6.56 | |
| ChArUco (103 · 200 · 500 · 1,030) | shipped | 1.51 / 20.5 | 1.21 / 15.3 | 0.623 / 10.8 | 0.354 / 6.85 | |

Each KITTI cell is median / p95 move; ATE in metres. ChArUco has no ground truth, so no ATE.

- **ATE separates the variants only early** (shipped KITTI 03 at 10 rounds: 43.7 m, a transient far from the ground truth; shipped KITTI 10 at 10 rounds: 0.66 m).
- **From 50 rounds on**, ATE changes by at most 0.007 m per segment on the corrected variant (0.08 m on shipped KITTI 10), while the unfinished fraction changes by up to 100 points.

## 5. Worst cases

| Quantity | Worst | Where |
|---|---|---|
| Unfinished at 100 rounds, corrected | 97.5 % | KITTI 03 |
| Unfinished at 300 rounds, shipped (relinearized) | 87.5 % | KITTI 03 |
| Relinearized minus current-problem fraction, shipped | +45 points | KITTI 10 at 200 rounds |
| Relinearized minus current-problem fraction, corrected | 0 at every budget | all segments |
| Unfinished at ChArUco's shipped budget (103) | 83 % | shipped |
| ATE at the shipped stereo default (10 rounds) | 43.7 m | shipped KITTI 03 |

## 6. What could not be done

Nothing was left undone. The ChArUco budget of 103 needed the reruns described in §2, about 0.5 min each.

## 7. Compute used (minutes)

- ChArUco reruns to round 103: 0.5 (corrected) and 0.5 (shipped).
- Analysis: under 1.

## 8. Files

- `scripts/partA_budgets.py`.
- `results/partA/<dataset>/<variant>/budgets.csv`, with per-budget current and relinearized fractions, median and p95 move, true-Jacobian move and ATE.
- `results/partA/summary.json`.
- ChArUco reruns (git-ignored): `exports/stage1/charuco/{corrected,shipped}_partA`.
- `--eval-at` in `certify_stereo` / `certify_charuco`.
- Commit: the commit that adds this file.
