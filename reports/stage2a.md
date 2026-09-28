# Stage 2a: the final-answer check at the certified stops

Branch `certify`, base commit `d8dd413` (Part A, on Stage 1's `857a87d`). Brief: `BRIEF_stage2.md` §2. Settings fixed by §0: δ = 0.003, R = 1, η = 0.1, true-Jacobian yardstick, D2 landmark treatment; variants corrected and shipped. Method: `certify/STAGE2A.md`.

## 1. Five-line summary

1. **Gate B1 passes.** On the corrected variant with the 0.1 m landmark cap, the fraction of poses with r_i = s_i^NL/(ηS_lo,i) < 0.1 is 97.5 % (06), 100 % (03), 100 % (04) and 100 % (10): 4 of 4 segments at or above 80 %.
   - Median r is 0.022–0.051 across the four segments; the largest is 0.102, a single pose on 06.
2. **The shipped variant fails the same check:** 0 % of poses have r < 0.1 on both segments with a certified stop (06, 10), with median r 0.28 and 0.58. The shipped stereo Jacobian leaves a first-order residual (WP6b).
3. **The result depends on the landmark cap.** The exact local step moves weakly observed landmarks by up to 2 km (29 km on shipped KITTI 10); 10–20 % of landmarks move more than 0.1 m.
   - With a 1 m cap, the corrected fractions below 0.1 drop to 0–25 %; unclipped, to 0 % on every segment.
   - The fraction below 1 stays at 100 % with the 1 m cap, and at 52.5–100 % unclipped.
4. **Stereo factors carry the residual; GP factors do not** (≤ 0.4 % of s^NL in the worst decile on the corrected variant).
   - With the 0.1 m cap, a large part of the worst decile's residual is linear rather than nonlinear: 0.92 of s^NL on KITTI 10 and 0.57 on 03. It comes from assembling one global candidate out of different patches and from the clipping itself.
5. **Every stop state was reconstructed exactly.**
   - The six Stage 1 runs, repeated to their stop rounds, reproduce Stage 1 bit-for-bit (3,960 of 3,960 pose-evaluations).
   - The rebuilt patch factorizations reproduce the logged E and L to 7e-16.
   - The probe's factor evaluation is exact at the linearization point: F_nl = F_lin, difference 0.0.
   - This is a necessary check for T1, not a certificate; ChArUco has no certified stop and is skipped.

## 2. Gate status

### Gate B1 (pre-registered)

The corrected variant with the 0.1 m cap must have at least 80 % of poses with r_i < 0.1 on at least three of the four KITTI segments.

| Segment | Stop round | Poses | r < 0.1 | Meets 80 % |
|---|---|---|---|---|
| KITTI 06 | 130 | 40 | **97.5 %** | yes |
| KITTI 03 | 250 | 40 | **100 %** | yes |
| KITTI 04 | 170 | 40 | **100 %** | yes |
| KITTI 10 | 50 | 40 | **100 %** | yes |

**Gate B1: pass (4 of 4).**

Shipped, for comparison (not part of the gate):

| Segment | Stop round | r < 0.1 |
|---|---|---|
| KITTI 06 | 150 | 0 % |
| KITTI 10 | 240 | 0 % |

The shipped variant has no certified stop on 03 or 04, and ChArUco has none on either variant, so those states do not exist.

### Checks

| Check | Result |
|---|---|
| Reruns to the stop round reproduce Stage 1 | 3,960 of 3,960 pose-evaluations identical (six runs) |
| Rebuilt certificate from the persisted patches vs the logged E and L | ≤ 7.1e-16 relative (E); ≤ 2.8e-16 absolute (L) |
| Full D2 model's exact reduction vs the sidecar's S and g | S ≤ 4.2e-14; g ≤ 1.1e-11 relative |
| Probe load: J and e recomputed in C++ at x_lin, with the variant's Jacobians | 0.0 difference (all six states) |
| Probe identity at z = 0: F_nl = F_lin | 0.0 difference (all six states) |
| Default path byte-identical | **12 of 12** (`results/stage2a/equivalence.txt`) |

## 3. Tables

### 3.1 Distribution of r_i = s_i^NL / (ηS_lo,i)

| Segment | Variant | Cap | r median | r p95 | r max | r < 0.1 | r < 1 | Worst decile: stereo / GP / linear residue share | Largest landmark step in the worst decile (before clipping) |
|---|---|---|---|---|---|---|---|---|---|
| KITTI 06 | corrected | 0.1 m | 0.030 | 0.062 | 0.102 | 0.975 | 1.000 | 0.99 / 0.002 / 0.08 | 2,041 m |
| | | 1 m | 0.123 | 0.213 | 0.230 | 0.175 | 1.000 | 1.00 / 0.001 / 0.03 | 2,041 m |
| | | none | 0.289 | 0.614 | 0.699 | 0 | 1.000 | 1.00 / 0.000 / 0.004 | 2,041 m |
| KITTI 03 | corrected | 0.1 m | 0.034 | 0.058 | 0.074 | 1.000 | 1.000 | 0.86 / 0.004 / 0.57 | 17.5 m |
| | | 1 m | 0.208 | 0.317 | 0.338 | 0.050 | 1.000 | 0.97 / 0.001 / 0.12 | 1,530 m |
| | | none | 0.478 | 1.17 | 1.25 | 0 | 0.875 | 0.98 / 0.000 / 0.03 | 1,530 m |
| KITTI 04 | corrected | 0.1 m | 0.022 | 0.076 | 0.078 | 1.000 | 1.000 | 0.98 / 0.003 / 0.23 | 260 m |
| | | 1 m | 0.424 | 0.744 | 0.753 | 0 | 1.000 | 1.00 / 0.000 / 0.02 | 260 m |
| | | none | 0.948 | 1.75 | 1.95 | 0 | 0.525 | 1.00 / 0.000 / 0.001 | 260 m |
| KITTI 10 | corrected | 0.1 m | 0.052 | 0.078 | 0.087 | 1.000 | 1.000 | 0.57 / 0.003 / 0.92 | 175 m |
| | | 1 m | 0.173 | 0.230 | 0.257 | 0.250 | 1.000 | 1.11 / 0.002 / 0.25 | 175 m |
| | | none | 0.281 | 0.541 | 0.562 | 0 | 1.000 | 0.95 / 0.001 / 0.10 | 175 m |
| KITTI 06 | shipped | 0.1 m | 0.284 | 1.18 | 2.00 | 0 | 0.850 | 1.00 / 0.026 / 0.003 | 2,164 m |
| | | 1 m | 0.330 | 1.20 | 2.02 | 0 | 0.850 | 1.00 / 0.026 / 0.005 | 2,164 m |
| | | none | 0.425 | 1.57 | 2.12 | 0 | 0.750 | 1.00 / 0.022 / 0.003 | 2,164 m |
| KITTI 10 | shipped | 0.1 m | 0.584 | 1.51 | 2.18 | 0 | 0.775 | 0.99 / 0.002 / 0.009 | 29,400 m |
| | | 1 m | 0.655 | 1.55 | 2.20 | 0 | 0.750 | 0.99 / 0.002 / 0.009 | 29,400 m |
| | | none | 10,400 | 271,000 | 302,000 | 0 | 0.200 | 1.00 / 0.000 / 0.000 | 29,400 m |

- **The shares are norm ratios with the same reduction**, so they are not additive. "Stereo" and "GP" are each factor class's nonlinear part, F_nl − F_lin. "Linear residue" is the linear model's gradient at the global candidate, F_lin(z₁).
- **Shipped KITTI 10, unclipped:** 7 stereo factors change inlier status at the stepped point, since a landmark moved 29 km lands behind the camera. No other state has any flip.

### 3.2 The landmark steps of the global candidate (translation, before clipping)

| Segment | Variant | Landmarks | Median | p99 | Max | > 0.1 m | > 1 m |
|---|---|---|---|---|---|---|---|
| KITTI 06 | corrected | 4,676 | 0.7 mm | 17.1 m | 2,041 m | 949 | 344 |
| KITTI 03 | corrected | 4,614 | 1.1 mm | 9.3 m | 1,530 m | 507 | 212 |
| KITTI 04 | corrected | 4,642 | 1.2 mm | 8.6 m | 260 m | 458 | 245 |
| KITTI 10 | corrected | 4,222 | 5.8 mm | 5.0 m | 175 m | 539 | 172 |
| KITTI 06 | shipped | 4,636 | 0.6 mm | 17.2 m | 2,164 m | 948 | 344 |
| KITTI 10 | shipped | 3,951 | 0.7 mm | 6.1 m | 29,400 m | 465 | 169 |

The median landmark step is millimetres. Ten to twenty percent of landmarks are weakly determined in depth, and the exact local step moves them far outside any box on which the linearization means anything. The pre-registered 0.1 m cap confines exactly those.

### 3.3 The certificate at the same stops, for reference

E_i/(ηS_lo,i) median / p95 (all below 1 by construction):

| Segment | Corrected | Shipped |
|---|---|---|
| KITTI 06 | 0.078 / 0.50 | 0.109 / 0.62 |
| KITTI 03 | 0.60 / 0.90 | – |
| KITTI 04 | 0.75 / 0.98 | – |
| KITTI 10 | 0.77 / 0.99 | 0.27 / 0.60 |

## 4. Worst cases

| Quantity | Worst | Where |
|---|---|---|
| r, corrected, 0.1 m cap | 0.102 (the only pose at or above 0.1) | KITTI 06, pose t = 6.963 s |
| r, corrected, unclipped | 1.95 | KITTI 04 |
| r, shipped, 0.1 m cap | 2.18 | KITTI 10, pose t = 8.718 s |
| Linear-residue share in a worst decile | 0.92 | corrected KITTI 10, 0.1 m cap |
| GP share in a worst decile, corrected | 0.004 | KITTI 03, 0.1 m cap |
| Largest landmark step | 2,041 m (corrected); 29,400 m (shipped) | KITTI 06 corrected; KITTI 10 shipped |
| Rebuilt E vs logged | 7.1e-16 | KITTI 06 corrected |
| Model agreement (g) | 1.1e-11 | KITTI 03 corrected |
| Probe identity at z = 0 | 0.0 | all |

## 5. What could not be done, and notes

- **ChArUco has no certified stop** on either variant (Stage 1), so it has no state to check.
- **The patch factorizations were not persisted in Stage 1.**
  - The six Stage 1 runs were repeated to their stop rounds; each reproduces Stage 1 exactly. At the stop, the linearization and every pose's final and base patch were persisted.
  - The factorizations were rebuilt from those patches. The rebuilt certificate matches the logged E and L to rounding, so the factorizations are the ones used.
- **The candidate's landmark cap applies to the landmark's step from the current state**, not to its offset from the linearization point. The two differ by at most δ = 3 mm.
- **Gate B1 passes only with the 0.1 m cap it was registered with.** With a 1 m cap it would fail on all four segments (0–25 % below 0.1), and unclipped too (0 %).
  - The cap is part of the candidate, and T1 accepts any candidate. Stage 2b will have to certify with this capped candidate, and its box must then contain the capped landmark moves.
- **With the 0.1 m cap, much of the remaining residual is linear**, not curvature: up to 0.92 of s^NL in a worst decile. A candidate that is a single consistent solve rather than assembled from patches would reduce it. Stage 2b's T1 accounts for it automatically, because s_i is computed from the actual F(z₁).
- **This is a necessary condition only.** T1 adds the curvature remainder ℓ∘ρ(u) and the leak recursion to s; nothing here is a certificate of the distance to the nonlinear zero.
- **Stage 2b was not started.**

## 6. Compute used (minutes)

- **Reruns to the stop rounds:**

  | Segment | Corrected | Shipped |
  |---|---|---|
  | KITTI 06 | 0.6 | 0.7 |
  | KITTI 03 | 1.9 | – |
  | KITTI 04 | 1.3 | – |
  | KITTI 10 | 0.4 | 1.7 |

- **Probes:** 0.15–0.23 per stop state (six).
- **Equivalence:** 5.0.
- **Build:** about 2.

## 7. Files

- **Code:**
  - `scripts/stage2a_probe.py`;
  - `scripts/stage1_evaluator.py`, which persists the stop linearization and patches;
  - `certify/cpp/certify_common.h`: `ProbeServer`'s true-Jacobian term under `--true-jacobians`, and `--eval-at`.
- **Note:** `certify/STAGE2A.md`.
- **Results:**
  - `results/stage2a/<dataset>/<variant>/probe.csv`, per pose and cap: E, S_lo, s^NL, r, the stereo, GP and linear-residue parts, the largest landmark step before clipping, and landmarks clipped;
  - `results/stage2a/<dataset>/<variant>/summary.json`;
  - `results/stage2a/gate_B1.json`;
  - `results/stage2a/equivalence.txt`.
- **Git-ignored:** the stop states, in `exports/stage1/<dataset>/<variant>_stage2a/eval/certified_stop_{model,beliefs,patches.json}`.
- **Commit:** the commit that adds this file.
