# Phase C report: tightness on real graphs

Branch `certify`. Date 2026-09-25. This report replaces the earlier "stopped at D2" version. After that stop you decided (14:43) to continue Phase C and to explore beyond the brief where the data called for it. The time-stamped reasoning is in `25_09_2026_PROGRESS.md`. All runs used `OMP_NUM_THREADS=16` and the dynamic OpenMP schedule (D5). Tolerance and thresholds are as fixed in the brief and in the Phase A review: invariants at 1e-9 relative, η = 0.1, δ = 0.05.

**Update, 2026-09-26.** Added the decision-driven patches (I1b) on the prefix and on KITTI full, and the one-factorization statistical-metric script (§1 items 4–5, §3.6, §3.7, §4, §6, §7). Phase B's sweep has finished and Gate B passes (`reports/phaseB.md`).

## 1. Five-line summary

1. **What was run.**
   - The brief's certificates (C1–C4, hub correction, covariance bounds, radii, oracle) on all five ChArUco snapshots.
   - The brief's certificates on the stereo prefix: C1, C2, C3 at K = 1–3 and Krylov bounds on the full 13,778-block model. C4 and hubs were not reached; I stopped that run at 18:08 (§6).
   - **Outside the brief:** C1, C3 and every covariance bound on the *landmark-marginalized pose model* of the stereo prefix (both snapshots) and of KITTI 06 in full.
   - D2's sensitivity check fails on stereo, so those runs are flagged. By your decision they went ahead with p_L = 1e-6.
2. **Validity: 0 invariant violations** in all 8 evaluated runs: 237 checks, 33–42 per ChArUco snapshot and 21 per pose-model run. They include 51,498 certificate ≥ true-error comparisons, plus the one-step and covariance checks. The closest is +9.96e-10 against the 1e-9 tolerance: islam 25 %, patch lower bound ≤ Σ at K = 3, after the extended-precision re-check (§4).
3. **The brief's certificates are vacuous on every real graph except islam 25 % at K = 3. Gate C fails as defined.**
   - C1: ρ(M) = 1.64 on ChArUco and 2.91 on stereo.
   - C2: ρ(M̂) = 2.28 (stereo) to 4.17 (ChArUco).
   - C3: ρ(A) = 6.6–16 on ChArUco and 1.33–2.15 on stereo.
   - C4: ρ(A) = 2.9–48.
   - Hub bound: 5e8 × the true error (median).

   ChArUco has no local certificate at all: C3 first becomes valid at K = 6, where the patch is the whole graph. On stereo the landmark → pose → landmark hops have gain ≈ 1 in the per-block metric, and no patch radius tried brings ρ(A) below 1 (1.014 at the largest).
4. **On the landmark-marginalized pose model, C3 is valid, tight and gives certified early exit.** Eliminating the landmarks exactly (S = H_pp − H_pl H_ll⁻¹ H_lp) leaves the same pose solution (agreement to 1.5e-12).
   - ρ(A) = 0.51–0.56 at K = 1 and 0.095 at K = 2.
   - E/e median 1.23–1.56 at K = 1 and 1.003–1.02 at K = 2.
   - At K = 2 the certified stop fraction is within 0–4 points of the oracle at every iteration. The certified μ_lo is within 3e-5 of λ_min (median, K = 1), so S_lo is effectively exact.
   - C1 on the same model is still vacuous (ρ(M) = 2.25–2.71): the gain comes from exact local solves.
   - **Decision-driven patches** grow a pose's patch only while its stop decision is open (§3.7). On KITTI full their certified stop fraction is within 1.4 points of the oracle at every iteration, the same decisions as K = 3, with median patches of 73–87 poses instead of 274. All certificates are valid.
5. **Open problems.**
   - Patch size: K = 1 spans 27 % of KITTI full and 44 % of the prefix. Decision-driven patches reduce this to about 80 poses, but they depend on the residual, so a solver must refactorize a pose's patch whenever it changes. That cost is not measured.
   - The cut upper bound exists for only 13–48 % of KITTI-full poses.
   - Landmark certificates by back-substitution are about 3× loose.
   - Early exit against G-solver's live schedule is not measured yet (Phase D).

## 2. Gate status (brief §9, Gate C)

| Target | Dataset / certificate | Result | Status |
|---|---|---|---|
| Median ratio ≤ 3 for C3/C4 on stereo BA | full model (the brief's definition), prefix | C3 ρ(A) = 2.15 / 1.78 / 1.33 at K = 1 / 2 / 3; no certificate exists | **fail** |
| | landmark-marginalized pose model (not in the brief) | C3 K = 1: 1.56 / 1.30 / 1.23 (prefix cap / prefix it. 0 / KITTI full); K = 2: 1.02 / 1.01 / 1.003 | pass, on a different model |
| Median ratio ≤ 3 for C3/C4 on ChArUco chain interiors | islam 25 % | C3 K = 3: 1.000 (poses); C4 w = 3, K = 3: 10.3 | C3 pass, C4 fail |
| | islam 50 %, 100 %, BA it. 0 / 50 | C3 ρ(A) ≥ 6.6 and C4 ρ(A) ≥ 8.9 at every K ≤ 3; no certificate | **fail** |
| Ratio ≤ 10 on ChArUco overall with the hub term | islam 25 % (the only snapshot where C4 exists) | hub w = 3, K = 3: median 5.2e8, 95 % 1.6e12 | **fail** |
| μ_lo ≥ 0.5 λ_min at K = 3 or ℓ = 5 | all 8 runs | 100 % of poses; μ_lo/λ_min median 0.998–1.000 | pass |
| μ_hi ≤ 4 λ_max at K = 3 for ≥ 90 % of poses | ChArUco | 100 % (islam 25 %), 85 % (50 %), 43 % (100 %, BA it. 0, BA it. 50) | fail on 4 of 5 |
| | pose model | 96 % / 97 % (prefix cap / it. 0), 38 % (KITTI full) | pass on the prefix, **fail** on KITTI full |
| Report before Phase D if the chosen certificate's ratio > 30 | pose-model C3 | largest ratio anywhere 6.93 (K = 1) | – |

§12 decision logic, as it applies:
- **C1 is vacuous while C3 is tight** (on the pose model). This confirms the project's motivation.
- **On ChArUco the hub term dominates**: its share of E_hub is 1 − 3e-10 (median). The §4.7 energy aggregate is 35× smaller than E_S (median 0.029), so it is the identified fix. The brief's fallback, "continue Phase D with the chain certificate on stereo BA", is not available: the chain certificate is vacuous on the stereo full model. The pose-model C3 is what Phase D should use (§6).
- **ψ̄ ≫ 1 on long segments** (median 9–51), so I reran C3 on islam 100 % at K = 4, 5, 6 (§3.3). It becomes valid only at K = 6, when every patch is the whole graph.

## 3. Tables

Conventions:
- Ratios E/e are over entries whose true error exceeds 10 × the pose floor, as agreed in C-iii.
- "Stop" is the fraction of poses whose certificate satisfies E_i ≤ η · S_lo_i at that iteration. It is pointwise, not cumulative. S_lo comes only from certified lower bounds (trivial, patch, Krylov).
- The oracle row applies the same rule to the exact error and the exact radius.
- Iteration t counts frozen-GBP iterations after the freeze. "cap" snapshots are frozen at the normal run's 200-iteration cap with the messages kept; "it. 0" snapshots are frozen before the first iteration.

### 3.1 Datasets and floors

| Snapshot | Model certified | Blocks | Poses | Pose floor (metric of that model) | Reached (t) | Flag |
|---|---|---|---|---|---|---|
| islam 25 % / 50 % / 100 % | certified model (markers kept) | 38 / 68 / 118 | 24 / 47 / 94 | 9.9e-4 / 5.5e-4 / 7.4e-4 | yes (542 / 2,220 / 7,020) | – |
| BA it. 0 / it. 50 | certified model | 118 / 118 | 94 | 7.3e-4 / 7.4e-4 | yes (9,090 / 8,960) | – |
| kitti06 prefix, cap | full model (D2, p_L = 1e-6) | 13,778 | 161 | 0.108 | no | D2 fails |
| kitti06 prefix, cap | pose model | 161 | 161 | 0.076 | no (GBP still drifting at 3,000) | D2 fails |
| kitti06 prefix, it. 0 | pose model | 161 | 161 | 0.147 | yes (121) | D2 fails |
| kitti06 full, cap | pose model (40,370 landmarks eliminated) | 477 | 477 | 0.126 | yes (361) | D2 fails |

The stereo floors exceed the 0.05 tolerance but are far below the stopping threshold η · S_lo ≈ 0.48. For early exit the floor does not bind.

### 3.2 ChArUco: the brief's certificates

Spectral radii (Collatz–Wielandt upper bounds agree to 1e-10):

| Snapshot | C1 ρ(M) | C2 ρ(M̂) | C3 ρ(A), K = 1 / 2 / 3 | C4 w = 1, K = 1 / 2 / 3 | C4 w = 3, K = 1 / 2 / 3 | θ_S |
|---|---|---|---|---|---|---|
| islam 25 % | 1.643 | 3.71 | 8.91 / 4.94 / **0.719** | 8.77 / 11.95 / 12.40 | 4.08 / 2.87 / **0.830** | 23.4 |
| islam 50 % | 1.638 | 4.17 | 15.31 / 6.83 / 7.48 | 15.20 / 25.42 / 22.65 | 30.42 / 23.13 / 8.95 | 106 |
| islam 100 % | 1.641 | 3.91 | 15.39 / 6.79 / 6.64 | 15.30 / 23.69 / 17.94 | 33.27 / 47.65 / 30.95 | 322 |
| BA it. 0 | 1.643 | 3.91 | 16.10 / 6.82 / 6.66 | 16.03 / 24.20 / 18.04 | 33.47 / 47.52 / 30.79 | 323 |
| BA it. 50 | 1.641 | 3.91 | 15.40 / 6.79 / 6.64 | 15.31 / 23.71 / 17.95 | 33.27 / 47.65 / 30.95 | 322 |

Ratios, shares and ψ̄ where a certificate exists (islam 25 % only):

| Certificate | E/e median | 95 % | max | Local share (median) | Notes |
|---|---|---|---|---|---|
| C3, K = 3, poses | 1.000 | 1.000 | 1.000 | 1.000 | K = 3 patches cover almost the whole 38-block graph |
| C3, K = 3, all blocks | 1.00 | 2.44 | 3.53 | 1.000 | worst is a marker (§4) |
| C4, w = 3, K = 3 (against the hub-clamped error) | 10.3 | 85.6 | 1,067 | 0.52 (supernode) | |
| Hub correction, w = 3, K = 3 | 5.2e8 | 1.6e12 | 1.0e13 | ĉ share 2.6e-10 | hub term / oracle hub influence: 1.1e-10 (median). §4.7 energy aggregate / E_S = 0.029. E_S / ‖e_S‖ = 47 |

ψ̄ quantiles (min / median / max), K = 3:
- w = 1: 0.83 / 9.3 / 2.6e5 (islam 25 %) to 0.86 / 10.3 / 3.4e5 (BA it. 0);
- w = 3: 0.94 / 20.7 / 77 (islam 25 %) to 0.94 / 51 / 4.3e4 (BA it. 0).

No snapshot shows certified decay along the chain.

Covariance bounds at K = 3 (poses):

| Snapshot | μ_lo/λ_min median | fraction ≥ 0.5 | μ_hi(cut)/λ_max median | fraction of poses with μ_hi ≤ 4 λ_max |
|---|---|---|---|---|
| islam 25 % | 1.000 | 100 % | 1.000 | 100 % |
| islam 50 % | 1.000 | 100 % | 1.000 | 85 % |
| islam 100 % | 0.998 | 100 % | 1.000 | 43 % |
| BA it. 0 | 0.998 | 100 % | 1.000 | 43 % |
| BA it. 50 | 0.998 | 100 % | 1.000 | 43 % |

Where the cut bound exists it is essentially exact; where it does not exist, the cut boundary leaves the patch under-determined.

### 3.3 ChArUco: C3 against K (islam 100 %, the §12 rerun)

| K | median patch | ρ(A) | valid |
|---|---|---|---|
| 1 | 10 blocks (8 %) | 15.4 | no |
| 2 | 58 (49 %) | 6.8 | no |
| 3 | 83 (70 %) | 6.6 | no |
| 4 | 114 (97 %) | 4.3 | no |
| 5 | 118 (100 %) | 4.7 | no (a few patches still miss blocks) |
| 6 | whole graph | 0.35 | yes: ratio 1.00 median, 1.13 at the 95th percentile |

The board markers put every frame within 2–5 hops of every other. Even with the exact K = 6 certificate, only 29 % of poses may stop at iteration 1 and 44 % at 100. GBP itself converges at 0.9988 per iteration there, with the floor at t ≈ 7,000.

### 3.4 Stereo prefix (cap), full model: the brief's certificates

| Certificate | ρ | Time |
|---|---|---|
| C1 | 2.91 | 1.4 s |
| C2 | 2.28 | 56 s (structured pose–landmark path; the general path took > 30 min; the two agree to 1e-15) |
| C3, K = 1 / 2 / 3 | 2.15 / 1.78 / 1.33 | 46 s / 404 s / 4,053 s |
| C3, pose radius K_p = 2…6, landmark radius K_l = 1 | 2.12 / 2.03 / 2.03 / 2.02 / 2.02 | |
| C3, pose radius K_p = 2…6, landmark radius K_l = 2 | 1.78 / 1.17 / 1.08 / 1.018 / 1.014 | |

Where ρ(A) > 1 lives, at K = 1:
- Pose rows: median row sum 8.05, of which 6.98 comes from landmark columns.
- Landmark rows: median row sum 1.42. Well-observed landmarks carry 65 % of the Perron mass.
- Removing the 1,819 weakest landmarks leaves ρ = 1.97.

Two tighter aggregations were tried and rejected:
- ℓ2 aggregation of the leak: 4.9, or 3.3 grouped per boundary node, against 2.15 for C3.
- Rigid-motion coarse spaces: 60–99 % of the error energy remains.

### 3.5 Stereo: C3 on the landmark-marginalized pose model

Reduction:
- Each landmark's pairwise factors are stacked and the landmark is eliminated exactly into one marginalized factor on its observers (`certify.model.reduce_blocks`).
- Prefix: 13,617 landmarks → 18,017 factors on 161 poses, 76 s.
- KITTI full: 40,370 landmarks → 45,767 factors on 477 poses, 254 s.
- The reduced model's solution equals the full model's pose solution to 5e-13 (prefix) and 1.5e-12 (KITTI full), in the pose metric. It also matches the direct Schur complement (unit test).

| Run | K | ρ(A) | max row sum | patch median / 95 % / max (poses) | E/e median / 95 % / max | local share | locally certifiable (cut bound exists) | μ_lo/λ_min median | μ_hi/λ_max median | S_lo median |
|---|---|---|---|---|---|---|---|---|---|---|
| prefix, cap | C1 | 2.25 | | | vacuous | | | | | |
| | 1 | 0.506 | 1.031 | 71 / 90 / 108 of 161 | 1.56 / 3.29 / 3.99 | 0.56 | 45 % | 0.99999 | 1.002 | 4.775 |
| | 2 | 0.095 | 0.168 | 125 / 147 / 148 | 1.02 / 1.20 / 1.24 | 0.98 | 65 % | 1.0000 | 1.000 | 4.775 |
| | 3 | 0.005 | 0.014 | 148 / 161 / 161 | 1.000 / 1.001 / 1.002 | 1.00 | 96 % | 1.0000 | 1.000 | 4.775 |
| prefix, it. 0 | C1 | 2.26 | | | vacuous | | | | | |
| | 1 | 0.506 | 1.026 | 71 / 91 / 108 | 1.30 / 2.67 / 3.82 | 0.76 | 45 % | 0.99999 | 1.002 | 4.775 |
| | 2 | 0.096 | 0.169 | 125 / 147 / 148 | 1.01 / 1.19 / 1.52 | 0.99 | 64 % | 1.0000 | 1.000 | 4.775 |
| | 3 | 0.005 | 0.015 | 148 / 161 / 161 | 1.000 / 1.003 / 1.009 | 1.00 | 97 % | 1.0000 | 1.000 | 4.775 |
| KITTI full | C1 | 2.71 | | | vacuous | | | | | |
| | 1 | 0.565 | 1.301 | 128 / 188 / 195 of 477 | 1.23 / 2.78 / 6.93 | 0.79 | 13 % | 0.99997 | 1.004 | 4.787 |
| | 2 | 0.095 | 0.151 | 210 / 266 / 292 | 1.003 / 1.07 / 3.61 | 1.00 | 48 % | 1.0000 | 1.000 | 4.787 |
| | 3 | 0.006 | 0.019 | 274 / 377 / 461 | 1.000 / 1.005 / 1.40 | 1.00 | 38 % | 1.0000 | 1.000 | 4.787 |

Exact λ_min of D^{1/2}ΣD^{1/2} in the pose metric: median 1.08–1.09, max 1.22. S = √λ_min · q₁₂(0.05) ≈ 4.8 for every pose.

### 3.6 Stereo: certified stopping against the oracle (pose model)

Prefix, cap (161 poses):

| Iteration | 1 | 2 | 3 | 5 | 7 | 10 | 15 | 20 | 30 | 50 | 70 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| K = 1 | 50 % | 59 | 59 | 64 | 67 | 73 | 75 | 80 | 87 | 94 | 98 | 100 |
| K = 2 | 62 | 65 | 67 | 70 | 76 | 83 | 87 | 89 | 94 | 96 | 100 | 100 |
| K = 3 | 62 | 65 | 68 | 73 | 78 | 84 | 88 | 91 | 94 | 96 | 100 | 100 |
| decision-driven (§3.7), median patch 33 | 62 | 64 | 66 | 72 | · | 84 | · | 89 | · | 96 | · | · |
| oracle | 62 | 65 | 68 | 73 | 78 | 84 | 88 | 91 | 94 | 96 | 100 | 100 |

(· = not evaluated.)

Prefix, it. 0 (cold start from ORB-SLAM3 values):

| Iteration | 1 | 2 | 3 | 5 | 7 | 10 | 15 | 20 | 30 | 50 | 70 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| K = 1 | 0 % | 0 | 6 | 38 | 52 | 57 | 71 | 71 | 78 | 86 | 89 | 99 |
| K = 2 | 0 | 0 | 11 | 53 | 65 | 63 | 77 | 78 | 83 | 89 | 91 | 99 |
| oracle | 0 | 0 | 12 | 53 | 65 | 64 | 78 | 79 | 86 | 89 | 91 | 99 |

KITTI full, cap (477 poses; the normal run had not converged at the cap, with max step 36.8):

| Iteration | 1 | 10 | 20 | 30 | 50 | 70 | 100 | 150 | 200 | 300 |
|---|---|---|---|---|---|---|---|---|---|---|
| K = 1 | 0 % | 0 | 1 | 3 | 10 | 29 | 54 | 77 | 94 | 100 |
| K = 2 | 2 | 5 | 10 | 16 | 34 | 47 | 65 | 81 | 95 | 100 |
| K = 3 | 6 | 8 | 11 | 19 | 37 | 48 | 65 | 82 | 95 | 100 |
| decision-driven (§3.7), median patch 73–87 | 5 | 8 | 11 | 19 | 37 | 47 | 65 | 81 | 94 | · |
| oracle | 6 | 8 | 11 | 20 | 37 | 48 | 65 | 82 | 95 | 100 |

K = 1 / 2 / 3 patches have median sizes 128 / 210 / 274 poses.

These fractions show how close the certificate comes to perfect knowledge. They are not savings against G-solver's schedule: sleeping poses change GBP's trajectory, and the frozen-run length is not a baseline. Phase D measures savings.

### 3.7 Beyond the brief: improvements tested on the pose model

**Certificate-driven grown patches (I1).**
- Rule: pose i grows its own patch, adding the exterior blocks whose leak coefficient is at least half of the largest, until its row sum Σ_j a_ij ≤ r.
- Theorem 3.5 allows any patch per node, and the row-sum condition certifies ρ(A) ≤ r from each node's own row (Collatz–Wielandt with w = 1).

| Run | Patches | Size median / p90 / max | ρ(A) | E/e median / 95 % | Stop at iterations |
|---|---|---|---|---|---|
| prefix, cap | K = 1 | 71 / 89 / 108 | 0.506 | 1.56 / 3.29 | 1 / 5 / 10 / 20 / 50: 50 / 64 / 73 / 80 / 94 % |
| | grown, r = 0.3 | **33 / 49 / 52** | 0.257 | **1.11 / 2.57** | **52 / 66 / 78 / 88 / 96 %** |
| | grown, r = 0.5 | 21 / 37 / 47 | 0.442 | 1.21 / 3.13 | 26 / 57 / 71 / 88 / 94 % |
| | oracle | | | | 62 / 73 / 84 / 91 / 96 % |
| KITTI full | K = 1 | 128 / 171 / 195 | 0.565 | 1.23 / 2.78 | 70 / 100 / 150 / 200: 29 / 54 / 77 / 94 % |
| | grown, r = 0.3 | 73 / 95 / 130 | 0.272 | 1.40 / 3.67 | 16 / 49 / 74 / 94 % |
| | grown, r = 0.5 | 52 / 71 / 94 | 0.459 | 2.02 / 4.86 | 0 / 34 / 60 / 93 % |
| | oracle | | | | 48 / 65 / 82 / 95 % |

- Prefix: grown patches beat K = 1 on every axis.
- KITTI full: they halve the patch but are looser and stop later. All valid.

**Decision-driven patches (I1b).** Why the grown patches stop late on KITTI full: at iterations 50–100 most errors sit close to the threshold (median error 0.5 at iteration 70, against a threshold of about 0.48). With exterior errors c_j ≈ e_i, the leak Σ_j a_ij c_j is about (row sum) · e_i. So a pose within 10 % of its threshold needs a row sum of about 0.1, which means a big patch. Most poses are far from the threshold, however, and need no such patch. So each pose grows only while its decision is open:
- Start from the grown r = 0.3 patch. The base certificate c⁰ is C3 on those patches (valid).
- At the evaluated iteration, pose i stops growing once its row sum is ≤ r and one of these holds:
  - s_i + Σ_j a_ij c⁰_j ≤ threshold (decided: stop);
  - s_i ≥ threshold (decided: continue);
  - the leak is below θ · s_i (nothing left to gain).
- Otherwise it adds the exterior blocks with the largest a_ij c⁰_j, up to a cap of 250 poses and 40 growth steps.
- The certificate is plain C3 on the final patches. Theorem 3.5 holds for any patches fixed when the certificate is evaluated, so patches that depend on the residual keep validity; the prediction s_i + Σ_j a_ij c⁰_j only steers the growth. ρ(A) ≤ max row sum ≤ r is certified node by node.
- Parameters were fixed before any run: r = 0.3, θ = 0.02, cap 250, 40 steps.

| Run | Patch median (mean), max | ρ(A) | E/e median / 95 % / max | Certified stop / oracle | Closest certificate to its error |
|---|---|---|---|---|---|
| prefix, cap | 33 (34–35), max 83 | 0.250–0.257 | 1.01–1.18 / 1.02–2.74 / 2.87 | iterations 1 / 2 / 3 / 5 / 10 / 20 / 50: 62 / 64 / 66 / 72 / 84 / 89 / 96 % against 62 / 65 / 68 / 73 / 84 / 91 / 96 % | E ≥ 1.008 e |
| KITTI full | 73–87 (75–98), max 259 | 0.241–0.271 | 1.11–2.01 / 1.17–4.32 / 5.43 | iterations 1 / 10 / 20 / 30 / 50 / 70 / 100 / 150 / 200: 5 / 8 / 11 / 19 / 37 / 47 / 65 / 81 / 94 % against 6 / 8 / 11 / 20 / 37 / 48 / 65 / 82 / 95 % | E ≥ 1.007 e |

- **Valid at every iteration** in both runs: every pose's certificate is at least its exact error.
- **Within 1.4 points of the oracle everywhere.** The largest gap is at iteration 30 on KITTI full (18.9 % against 20.3 %). There the base patches certify nothing before iteration 70, and K = 3 needs median patches of 274 poses for the same decisions.
- **Most decisions close at once.** On KITTI full, 450 of 477 poses are "continue" at iteration 1 and 449 are "stop" at iteration 200. The mean number of growth steps is 0.05–3.5, and the size cap binds for at most 2 poses per iteration.
- **Factorization cost proxy:** (|P| / |P_{K=3}|)³ ≈ (80 / 274)³ ≈ 1/40 per pose.
- **Two-sided self-assessment** (prefix run only; the KITTI-full run used an earlier version of the script). The reverse triangle inequality gives ‖e_i‖ ≥ s_i − Σ_j a_ij E_j, a certified lower bound L_i. It is valid at every iteration, and E/L has a median of 1.03–1.33. So the certificate reports its own looseness without the oracle.
- **Cost here:** growth over every (pose, iteration) pair took 16 s (prefix, 1,127 pairs) and 332 s (KITTI full, 4,293 pairs) on 12 cores. In a solver a pose's patch changes only while its decision is open, and each change needs a refactorization of that patch.

**Certificates in the statistical metric (I2).** W_i = L_i⁻¹, with L_i the certified patch lower bound on Σ_ii. The rule E_i^W ≤ η · q₁₂(0.05) guarantees ‖e_i‖_{Σ_ii⁻¹} ≤ η · q₁₂(0.05).

*This is a different criterion from the brief's.* The brief's D-metric rule implies it, but not conversely: it accepts larger errors along poorly determined directions. So its oracle differs as well. Prefix, cap:

| Patches | ρ(A^D) → ρ(A^W) | D-rule: certified / oracle, iterations 1 / 5 / 10 | W-rule: certified / oracle, iterations 1 / 5 / 10 |
|---|---|---|---|
| K = 1 | 0.506 → 0.471 | 50 / 64 / 73 against 62 / 73 / 84 | 64 / 70 / 76 against 71 / 83 / 90 |
| K = 2 | 0.095 → 0.084 | 62 / 70 / 83 against 62 / 73 / 84 | 70 / 79 / 90 against 71 / 83 / 90 |
| grown, r = 0.3 | 0.257 → 0.241 | 52 / 66 / 78 | 64 / 73 / 89 |

All W-certificates are valid (≥ ‖e‖_{W_i} and ≥ ‖e‖_{Σ_ii⁻¹}). The D-certificate converted to W stops exactly the same poses, as derived. The gain needs the recursion computed directly in W.

`scripts/explore_metric.py` computes both metrics from one factorization per patch: it keeps the metric-free pieces [H_PP⁻¹g_P]_i, [H_PP⁻¹]_ii and the raw leak blocks, then applies either metric. On the prefix it reproduces the table above exactly, with all certificates valid, in 90 s in total. It needs 6.8 s for K = 1, 5.5 s for the grown patches and 13 s for K = 2, over 161 poses and 17 iterations each. It has not been run on KITTI full (§6).

**Landmark certificates by back-substitution (prefix, cap).** From the landmark row of H e = g:

‖e_l‖_{D_l} ≤ ‖D_l^{-1/2} g_l‖ + Σ_p ‖D_l^{-1/2} H_lp S_pp^{-1/2}‖ E_p

Valid for all 13,617 landmarks at all iterations; the worst margin is +5e-5 relative, on the safe side. E/e median over iterations: 4.2–6.4 (from K = 1 pose certificates), 2.8–3.8 (K = 2), 2.7–3.4 (K = 3); 95th percentile 8–36.

### 3.8 Timing (offline Python, extended precision; not representative of an in-solver implementation)

| Run | Model reduction | C3 patches, K = 1 / 2 / 3 | Median per pose, K = 1 | Krylov bounds | Total |
|---|---|---|---|---|---|
| prefix, cap | 76 s | 44 s / 45 s / 30 s | 2.4 s | 41 s | 525 s |
| KITTI full | 254 s | 1,828 s / 1,388 s / 1,639 s | 38 s | 184 s | 5,485 s |
| islam 100 % (certified model) | – | 1.1 s / 8.7 s / 14.2 s | 15 ms | 1.4 s | 120 s |

The KITTI-full per-pose cost is dominated by the extended-precision cut solves and the double-double refinement used for validation. In-solver cost per model (§I3 of the progress log):
- factorize the patch once per relinearization, about |P|³/3 ≈ 1e9 flops at K = 1;
- per iteration, one |P| × 12 matrix–vector product (about 2e4 flops per pose).

## 4. Worst cases

Invariants: every check in every run passes. The complete list, with the location of each worst case, is in `results/phaseC/report_tables.md`. The checks closest to violation (positive margin = the bound exceeds the reference by that relative amount):

| Run | Check | Worst margin (pass ≤ 1e-9) | Where | Reading |
|---|---|---|---|---|
| islam 25 % | patch lower ≤ Σ, K = 3 | **+9.96e-10** | pose POSE_12.016666667 | The patch is the whole graph, so the bound equals Σ exactly and the margin measures the reference's accuracy on this 1e12-anchored model. 22 blocks were flagged in double precision and re-checked with the extended-precision marginal; this is the re-checked value. |
| islam 25 % | patch lower ≤ Σ, K = 2 | +7.96e-10 | marker LANDMARK_2 | same mechanism |
| islam 50 % | cut upper ≥ Σ, K = 3 | +6.15e-10 | pose POSE_3.600000000 | exact cut bound |
| islam 25 % | E_C3 ≥ err, K = 3 | +3.35e-11 | pose POSE_10.750000000, iteration 2 | exact certificate |
| islam 50 % / islam 25 % | C3 one-step, K = 3 / K = 2 | +5.8e-11 / +1.9e-11 | POSE_7.400000000, it. 1 / marker LANDMARK_7, it. 2 | exact local solve |
| islam 50 %, islam 25 % | patch lower ≤ Σ K = 3; cut upper ≥ Σ K = 2, 3 | +5.0e-12, +4.3e-12 | POSE_18.766666667; marker LANDMARK_9 | |
| pose model, KITTI full / prefix | patch lower ≤ Σ K = 3 / E_C3 ≥ err K = 3 | +6.5e-15 / +2.4e-15 | not recorded (the pose-model runs keep only the worst margin of each check) | exact at K = 3 |
| every other check, all runs | | ≤ +3.3e-15 | | rounding level |

Worst ratios:

| Run | Certificate | Worst block (position) | Iteration | E / e = ratio | Why |
|---|---|---|---|---|---|
| islam 25 % | C3, K = 3 | marker LANDMARK_28 (degree 2) | 10 | 0.042 / 0.012 = 3.53 | a marker seen by 2 poses, whose K = 3 patch misses part of the graph; pose ratios are 1.000 |
| islam 25 % | C4, w = 3, K = 3 | POSE_3.616666667 (pose 5, degree 10) | 2 | 3,252 / 3.05 = 1,067 | ψ̄ products |
| islam 25 % | hub, w = 3, K = 3 | POSE_1.650000000 (pose 1, degree 9) | 1 | 9.8e11 / 0.097 = 1.0e13 | hub term: Lemma 4.5 multiplies per-link ψ̄ ≈ 10–80 |
| islam 50 %, 100 %, BA | all | – | – | no valid certificate | |
| prefix, cap | C3, K = 1 | pose 139 (t = 29.4 s) | 1 | 3.48 / 0.87 = 3.99 | local term 0.89 ≈ e; leak 2.59 |
| | C3, K = 2 | pose 144 (t = 30.5 s) | 15 | 1.06 / 0.85 = 1.24 | |
| | C3, K = 3 | pose 156 (t = 35.0 s) | 1 | 1.63 / 1.62 = 1.002 | |
| prefix, it. 0 | C3, K = 1 / 2 / 3 | pose 147 (it. 10) / 160 (it. 1) / 106 (it. 1) | | 3.82 / 1.52 / 1.009 | |
| KITTI full | C3, K = 1 | pose 350 (t = 71.9 s) | 2 | 10.8 / 1.56 = 6.93 | leak 9.6 against local 1.25. The pose has a moderate error (rank 276 of 477) and its patch boundary touches the slowly converging stretch at poses 383–403 (errors up to 373 at iteration 1). |
| | C3, K = 2 | pose 353 (t = 72.5 s) | 1 | 4.73 / 1.31 = 3.61 | leak 3.38 against local 1.35 |
| | C3, K = 3 | pose 340 (t = 70.9 s) | 1 | 1.90 / 1.36 = 1.40 | |
| grown r = 0.8, KITTI full | C3 | – | – | max 22.1 | the largest ratio of any valid certificate |
| decision-driven, KITTI full / prefix | C3 | not recorded (the script keeps per-iteration statistics only) | 1 / 2 | max 5.43 / 2.87 | At iteration 1 on KITTI full, 450 of 477 decisions are "continue", which needs no tightness. From iteration 100 on the maximum is ≤ 1.47. |

Other worst cases:
- **Largest true error at the last iteration:** ChArUco POSE_3.600000000 (pose 4) in every snapshot, equal to the floor (5.5e-4 to 9.9e-4). Pose model: 0.127 on KITTI full at t = 2,000.
- **Largest D2 pose sensitivity:** prefix it. 0, 0.357 between p_L = 1e-5 and 1e-6; KITTI full 0.221.
- **Largest D6 difference:** 2.4e-4 absolute on islam (2.4e-16 relative), from the 1e12 anchor.

## 5. Decisions D1–D6 and deviations from the brief

**D1.** ε = 1e-6 is kept. Recorded finding: G-solver's fixed point differs from the assembled model's solution by O(ε). The cause is the two factor-side regularizers centred on cavity means.

**D2.** Single-observation landmarks are eliminated exactly, zero-information landmarks excluded, and p_L priors placed at the ORB-SLAM3 positions. The sensitivity check fails:

| Snapshot | Pose floor at p_L = 1e-7 / 1e-6 / 1e-5 | Pose change vs 1e-6 (1e-7 / 1e-5) | Zero-info excluded |
|---|---|---|---|
| prefix, cap | 0.106 / 0.108 / 0.171 | 0.050 / 0.150 | 80 |
| prefix, it. 0 | 0.154 / 0.188 / 0.433 | 0.057 / 0.357 | 0 |
| KITTI full, cap | 0.143 / 0.157 / 0.236 | 0.033 / 0.221 | 264 |

Attribution: holding the landmarks with observation information below τ at GBP's own state gives the floors in columns (a); clamping them at the linearization point (a fixed model) gives the floors in columns (b).

| τ | prefix cap (a) / (b) | prefix it. 0 (a) / (b) | KITTI full (a) / (b) |
|---|---|---|---|
| none | 0.108 | 0.188 | 0.158 |
| 1e-6 | 0.031 / 1.09 | 0.081 / 1.25 | 0.034 / 1.69 |
| 1e-4 | 7.2e-3 / 1.44 | 1.9e-3 / 7.86 | 1.2e-2 / 10.9 |
| 1e-2 | 2.4e-4 / 5.28 | 2.2e-4 / 8.38 | 1.1e-3 / 35.4 |

The floor is GBP's own O(ε) bias in the weak directions of a few tens of far landmarks. No fixed model reproduces it. For early exit it does not bind (floor ≤ 0.19 against a threshold ≈ 0.48). The certificates bound the distance to the assembled model's solution, which is what they claim.

**D3.** The prefix is keyframes 0–160 (first ground-truth revisit at keyframe 161), with w = 10 and hubs one at a time. No w gives ≤ 2,000 hub dof (1,048 hubs at w = 10).

**D4.** The errata are in both files. **D5.** The dynamic schedule was used for every run. **D6.** The independent assembly agrees to 1e-15 to 2e-17 relative on every dataset, KITTI full included (2.2e-15).

**Deviations from the brief, made on your instruction to explore.**
1. **The certified object for stereo is the landmark-marginalized pose model**, not the full model. Validity is Theorem 3.5 applied to an exact Schur complement. Its residual g_red = g_p − H_pl H_ll⁻¹ g_l is a sum of per-landmark local terms, so it is computable from GBP's state without the global solution. Landmarks are certified afterwards by back-substitution.
2. **Grown patches and the statistical metric** are tested as improvements (§3.7). Neither changes any invariant or threshold.
3. **The stereo runs went ahead despite the failed D2 check**, flagged in every summary (`--allow-floor-above-tol`).

## 6. What could not be done, and why

- **Gate C fails for the brief's certificates.** On ChArUco and on the stereo full model no certificate at K ≤ 3 has ρ < 1, except islam 25 % at K = 3. The hub correction is valid but 1e8–1e13 loose. This is a measurement; nothing was tuned.
- **The stereo full-model run is incomplete.** I stopped it at 18:08 after C1, C2, C3 at K = 1–3 and the Krylov bounds, because its outcome was established (ρ ≥ 1.33 at every radius tried), to free the machine for the pose-model runs. C4, the hub terms, the covariance invariants and `summary.json` for that model do not exist. The ρ values above come from its log (`results/phaseC/run_prefix_conv.log`).
- **KITTI full on the full model was not run.** C3 at K = 3 took 68 min on the 13,778-block prefix, and KITTI full has 40,847 blocks.
- **The cut upper bound on the pose model exists for only 13 % (K = 1) to 48 % (K = 2) of KITTI-full poses.** Cutting the marginalized-landmark factors at the boundary leaves boundary poses under-determined. S_hi, and so Phase E's tubes, needs a better upper bound. Stopping uses S_lo only and is unaffected.
- **Not attempted:**
  - C4 and hubs on the pose model (the I5 idea);
  - the Mahalanobis metric on KITTI full (`scripts/explore_metric.py` is ready and ran on the prefix in 90 s);
  - decision-driven patches in the statistical metric, and the two-sided self-assessment on KITTI full;
  - the refactorization cost of residual-dependent patches inside a solver (Phase D);
  - figures (Phase F);
  - a longer frozen run of the prefix at the cap to find its fixed point (the floor there is not reached).
- **Timings are offline Python with validation-grade precision.** They say nothing about in-solver cost (§3.8).

## 7. Files

- Report: `reports/phaseC.md` (this file). Full invariant and worst-case tables: `results/phaseC/report_tables.md`, generated by `scripts/phaseC_report_tables.py`.
- ChArUco (brief's certificates): `results/phaseC/charuco_{islam_025,islam_050,islam_100,ba_init,ba_end}/{summary.json,graph.json,curves.json,certs.csv.gz}`, from `scripts/phaseC_run.py` (`certify/phaseC.py`, `certify/phaseC_out.py`).
- Stereo full model (incomplete): `results/phaseC/run_prefix_conv.log`.
- Pose model: `results/phaseC/reduced/{kitti06_prefix_conv,kitti06_prefix_init,kitti06_full_conv}.json` and `_certs.csv.gz`, from `scripts/phaseC_reduced.py`; `certify.model.reduce_blocks`.
- Explorations: `results/phaseC/explore/`, with the logs `results/phaseC/explore_*.log`:

  | Output | Script |
  |---|---|
  | `K_charuco_islam_100.json` | `scripts/explore_K.py` |
  | `mixedK_kitti06_prefix_conv.json` | `scripts/explore_mixedK.py` |
  | `l2_kitti06_prefix_conv_K1.json` | `scripts/explore_l2.py` |
  | `twolevel_kitti06_prefix_conv.json` | `scripts/explore_twolevel.py` |
  | `greedy_kitti06_{prefix,full}_conv.json` | `scripts/explore_greedy.py` |
  | `mahalanobis_kitti06_prefix_conv.json` | `scripts/explore_mahalanobis.py` |
  | `landmarks_kitti06_prefix_conv.json` | `scripts/explore_landmark_backsub.py` |
  | `decision_kitti06_{prefix,full}_conv.json` (logs `explore_decision_{prefix,full}.log`) | `scripts/explore_decision.py` |
  | `metric_kitti06_prefix_conv.json` (log `explore_metric_prefix.log`) | `scripts/explore_metric.py` |
- D2/D6:
  - `results/phaseC/d2/<dataset>.json`, `_xbar_pL*.npy`, `attribution_<dataset>.json`;
  - scripts `scripts/phaseC_d2.py`, `scripts/phaseC_d2_attribution.py`, `scripts/phaseC_d2_summary.py`.
- Exports (git-ignored): `exports/kitti06_prefix161/{conv,init,unfrozen_run}` and `exports/kitti06_full/conv_noanchor`, from `scripts/phaseC_run_stereo.sh`.
- Progress log: `25_09_2026_PROGRESS.md`.
- Commit: the commit that adds this file (`git log -- reports/phaseC.md`).
