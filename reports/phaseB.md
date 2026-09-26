# Phase B report: standalone module and toy suite

Branch `certify`. Written 2026-09-26; the sweep finished on 2026-09-25 at 19:55. Brief §9 Phase B, with the Phase A review's requirement that the chain generator build its GP factors through G-solver's Python bindings and that the unit tests check F4's corrected identity.

## 1. Five-line summary

1. **The module is implemented.**
   - `certify/`: `chain.py` (§3), `covbounds.py` (§4.1–4.2), `radii.py` (§4.3, §8.5), `certs.py` (C1, C2, C3), `hubs.py` (supernodes, C4, hub correction).
   - Support: `model.py` (the model as a factor list), `synth.py` (generators), `check.py` (invariants against dense oracles), `linalg.py` and `precise.py` (numerics).
   - `certify/tests/`: 138 pytest tests, all passing (53 s). They cover every unit test named in brief §3–§6, plus fact F4 on G-solver's own GP chains through its bindings.
2. **The synthetic GP, hub and point factors are G-solver's own.** They are built, linearized and frozen by G-solver through its Python bindings; I added read-only accessors to the bindings (§7). F4's corrected identity G_k = (H_kk − Λ^f_kk) + message_{k+1→k} holds on G-solver's frozen GBP messages to relative 1e-15 to 4e-13 with the regularizers at 1e-12. As shipped (1e-6) it holds to O(ε): 79–616 ε absolute, 9e-10 to 4e-8 relative. The uncorrected remark is off by 12–98 %.
3. **Gate B: pass.**
   - 71 synthetic problems × 14 iterates (7 block Gauss–Seidel, 7 block Jacobi): 12,474 invariant checks covering 2,237,438 individual comparisons. None fails its tolerance (1e-9 relative; 1e-10 for the Theorem 4.2 decomposition), and no problem errors. The worst is +4.6e-10 (Lemma 1.3/1.4 on the 500-pose pure chain).
   - C2 is exact on every pure chain: the largest |E/e − 1| over all 12,180 entries is 3.3e-16.
   - C1 is reproduced by C2 with Pbb = D (M identical to 1e-12, E to 1e-9).
   - ρ(M) on the pure chain does **not** follow cos(π/(n+1)): it is 1.64–1.69 ≈ 2·m̄·cos(π/(n+1)). Here m̄ ≈ 0.84 (not ½) is the normalized 12-dof coupling of G-solver's GP link, so C1 is vacuous on every pure chain, even at n = 20 (a finding, §3.2).
4. **The invariants found real bugs in my first implementation, all fixed before this gate** (rule 1; §6):
   - Some boundary-cut matrices are exactly singular, but Cholesky "succeeded" through rounding.
   - Krylov bounds for different ℓ came from bases that were not nested, and the Krylov rank test was wrong.
   - The oracle and the exact certificates were not accurate enough on chains with cond(D⁻¹ᐟ²HD⁻¹ᐟ²) up to 4e13. Meeting 1e-9 needed an exactly assembled model (extended precision, stored hi + lo) and double-double residuals.
5. **Findings for Phase C: local certificates need absolute information in every block.**
   - C3 (ρ(A) < 1) exists on chains with m/p ≥ 1 at every K, and at m/p = 0.1 only at K = 3 for n ≤ 50. On track chains it exists only at m/p = 10 (K = 3, and K = 2 at n = 20). On random SPD graphs it exists only at K = 3, on 5 of the 6 graphs with at most 60 blocks. It never exists on pure chains or chains with a blind segment. On hub chains ρ(A) = 0 at K ≥ 2, but only because the hub puts every pose within two hops, so each patch is the whole graph.
   - C4 exists far more widely: on every hub chain and every track chain at K ≥ 2, and on chains with m/p ≥ 0.1 at K ≥ 2. It never exists on pure chains or blind segments.
   - C1 exists on 7 chains (m/p ≥ 1) but is loose: median E/e up to 35, 95th percentile up to 292.
   - The note gives no formula for "hubs one at a time". The obvious per-hub Lemma 4.4 is not valid; I implemented a valid construction and tested it (§6). Its oracle aggregate θ_S reaches 8.7e3 on track chains with 144 hubs.

## 2. Gate status (brief §9, Gate B)

| Criterion | Result | Status |
|---|---|---|
| All invariants pass on all synthetic problems (1e-9 relative) | 71 problems, 12,474 checks, 2,237,438 comparisons, 0 failures, 0 problem errors. Worst +4.6e-10: Lemma 1.3/1.4 (L ≤ G on H_TT) on the 500-pose pure chain | **pass** |
| C2 exact on the pure chain (ratio 1 ± 1e-8) | largest \|E/e − 1\| = 3.3e-16 over the 12,180 entries of the five pure chains (n = 20–500). The sweep stores only the median and 95th percentile, which are exactly 1 on all 34 problems where C2 exists. I recomputed the maximum from the same problems, seeds and iterates. | **pass** |
| C1 reproduced from C2 with Pbb = D | unit test, 3 models: M equal to 1e-12, ρ to 1e-10, ν to 1e-12, E to 1e-9 | pass |
| ρ(M) on the pure chain vs cos(π/(n+1)) as m/p → 0 | deviates: 1.643, 1.660, 1.685, 1.688, 1.689 for n = 20, 50, 100, 200, 500 (m/p = 0), against 0.989, 0.998, 0.9995, 0.9999, 0.99998 (§3.2) | reported (a finding, not a failure) |

## 3. Tables

Conventions:
- "Worst relative violation" is (lhs − rhs) / max(|lhs|, |rhs|) for an inequality lhs ≤ rhs. A negative value means the inequality holds with that much room. A positive value is a violation by that relative amount; it passes if ≤ 1e-9 and occurs only where the inequality is an equality up to rounding.
- E/e ratios are over the entries whose true error exceeds 1e-12. "–" means no certificate (ρ ≥ 1) or not applicable.
- C2's E/e is exactly 1 at the median and the 95th percentile on all 34 chain and blind-chain problems, so it is not repeated in §3.3.
- The sweep did not record C4's E/e ratios (§6).

### 3.1 Invariants: worst value of each check over all problems

| Check | Comparisons | Worst relative violation | Problem (radius) | Tolerance | Pass |
|---|---|---|---|---|---|
| `C1 E>=err` | 940 | -6.15e-05 | chain_24 | 1e-09 | yes |
| `C1 one-step u<=nu+Mu` | 16,810 | -4.11e-06 | chain_24 | 1e-09 | yes |
| `C2 E>=err` | 5,400 | 3.51e-16 | chain_20 | 1e-09 | yes |
| `C2 one-step u<=nuhat+Mhat u` | 16,810 | 3.51e-16 | chain_20 | 1e-09 | yes |
| `C3 E>=err` | 10,840 | 7.96e-14 | random_spd_64 K=3 | 1e-09 | yes |
| `C3 local<=E` | 10,840 | 0.00e+00 | chain_hub_27 K=2 | 1e-09 | yes |
| `C3 one-step u<=s+Au` | 50,430 | 7.96e-14 | random_spd_64 K=3 | 1e-09 | yes |
| `C4 c*>=err~ (supernode)` | 14,890 | 8.75e-15 | chain_1 K=3 | 1e-09 | yes |
| `C4 one-step u~<=s+Au~` | 23,970 | 6.96e-12 | chain_0 K=3 | 1e-09 | yes |
| `C4 per-block >= err (S empty)` | 7,220 | -4.04e-12 | chain_14 K=3 | 1e-09 | yes |
| `C4 per-block >= err~` | 14,890 | -4.04e-12 | chain_14 K=3 | 1e-09 | yes |
| `Lemma1.4 mirror L<=F (H_TT)` | 683,930 | 9.08e-15 | chain_11 | 1e-09 | yes |
| `Lemma1.4 seed/1.3 L<=G (H_TT)` | 683,930 | 4.62e-10 | chain_20 | 1e-09 | yes |
| `Lemma1.6 decay (H_TT)` | 23,970 | -5.78e-08 | chain_24 K=2 | 1e-09 | yes |
| `Lemma4.4 ||e_S||<=E_S` | 7,670 | -1.42e-01 | chain_hub_30 K=3 | 1e-09 | yes |
| `Lemma4.5 phi<=phibar` | 7,670 | -1.66e-01 | chain_tracks_40 K=3 | 1e-09 | yes |
| `Lemma4.5 phi<=phibar(trunc+tail)` | 7,670 | -3.86e-01 | chain_hub_27 K=3 | 1e-09 | yes |
| `Lemma4.5 single hub` | 35,890 | -2.28e-02 | chain_tracks_40 K=3 | 1e-09 | yes |
| `Prop3.6 a<=bound` | 23,970 | -6.75e-01 | chain_23 K=1 | 1e-09 | yes |
| `Prop4.7 hub-aware upper` | 7,670 | -1.74e-01 | chain_hub_33 K=3 | 1e-09 | yes |
| `Thm4.2 decomposition (rel err)` | 2,590 | 9.35e-15 | chain_tracks_45 | 1e-10 | yes |
| `Thm4.6 per-block E_hub>=err` | 7,670 | -1.42e-01 | chain_hub_30 K=3 | 1e-09 | yes |
| `Thm4.6 supernode` | 7,670 | -4.28e-01 | chain_hub_27 K=3 | 1e-09 | yes |
| `chain cut >= Sigma^T` | 23,970 | 1.05e-10 | chain_20 K=3 | 1e-09 | yes |
| `cut upper monotone` | 17,240 | 1.14e-15 | random_spd_62 K=3 | 1e-09 | yes |
| `cut upper>=Sigma` | 25,860 | 2.62e-11 | chain_20 K=3 | 1e-09 | yes |
| `global lb1<=energy` | 111,986 | -8.86e-03 | chain_19 | 1e-09 | yes |
| `global lb1<=lb5` | 111,986 | -8.53e-03 | chain_19 | 1e-09 | yes |
| `global lb5<=energy` | 111,986 | -2.62e-04 | chain_4 | 1e-09 | yes |
| `krylov lower<=Sigma` | 43,100 | 2.07e-15 | random_spd_70 l=2 | 1e-09 | yes |
| `krylov monotone` | 43,100 | 1.57e-14 | random_spd_70 l=2 | 1e-09 | yes |
| `m_SJ<=1` | 2,590 | -1.56e-01 | chain_tracks_46 | 1e-09 | yes |
| `one-at-a-time per-block` | 2,610 | 4.99e-14 | chain_hub_34 K=2 | 1e-09 | yes |
| `one-at-a-time supernode` | 2,610 | -2.91e-02 | chain_hub_36 K=3 | 1e-09 | yes |
| `patch lower monotone` | 17,240 | 3.39e-15 | random_spd_70 K=2 | 1e-09 | yes |
| `patch lower<=Sigma` | 25,860 | 1.76e-15 | random_spd_70 K=1 | 1e-09 | yes |
| `sec4.7 energy <= E_S` | 7,670 | -1.19e-01 | chain_hub_27 K=3 | 1e-09 | yes |
| `trivial lower<=Sigma` | 8,620 | 5.34e-15 | random_spd_70 | 1e-09 | yes |
| `||e_S|| <= theta*||r_S|| (oracle aggregate)` | 7,670 | -1.27e-02 | chain_hub_30 | 1e-09 | yes |

### 3.2 ρ(M) of the block-Jacobi matrix on chains, against cos(π/(n+1))

| n | cos(π/(n+1)) | m/p = 0 | m/p = 0.01 | m/p = 0.1 | m/p = 1 | m/p = 10 |
|---|---|---|---|---|---|---|
| 20 | 0.98883 | 1.643 | 1.535 | 1.262 | 0.979 | 0.658 |
| 50 | 0.99810 | 1.660 | 1.553 | 1.283 | 0.989 | 0.682 |
| 100 | 0.99952 | 1.685 | 1.587 | 1.343 | 1.046 | 0.793 |
| 200 | 0.99988 | 1.688 | 1.596 | 1.390 | 1.066 | 0.812 |
| 500 | 0.99998 | 1.689 | 1.602 | 1.391 | 1.066 | 0.811 |

The brief expects ρ(M) → cos(π/(n+1)) as m/p → 0. That holds for a scalar chain whose normalized link coupling is ½. G-solver's GP link couples 12-dof pose–velocity blocks with normalized strength m̄ ≈ 0.84, so ρ(M) ≈ 2·m̄·cos(π/(n+1)) instead. C1 needs ρ(M) < 1, which on chains first happens at m/p = 1 for n ≤ 50 and at m/p = 10 for every n.

### 3.3 Per problem: contraction and tightness

| Problem | Family | n | m/p | ρ(M) | C1 E/e median / 95 % | C3 ρ(A), K = 1 / 2 / 3 | C3 E/e median / 95 %, K = 1 · 2 · 3 | C4 ρ(A), K = 1 / 2 / 3 | Hubs | θ_S (oracle) |
|---|---|---|---|---|---|---|---|---|---|---|
| chain_0 | chain | 20 | 0 | 1.643 | – | 3.08 / 4.58 / 6.39 | – · – · – | 7.04 / 11 / 13.9 | 0 | – |
| chain_1 | chain | 20 | 0.01 | 1.535 | – | 2.01 / 2.24 / 2.09 | – · – · – | 1.36 / 0.716 / 0.343 | 0 | – |
| chain_2 | chain | 20 | 0.1 | 1.262 | – | 1.19 / 1.03 / 0.806 | – · – · 3.1 / 7.96 | 0.679 / 0.28 / 0.115 | 0 | – |
| chain_3 | chain | 20 | 1 | 0.979 | 19 / 82.9 | 0.643 / 0.366 / 0.204 | 1.72 / 3.94 · 1.36 / 3.16 · 1.16 / 2.16 | 0.162 / 0.0252 / 0.00392 | 0 | – |
| chain_4 | chain | 20 | 10 | 0.658 | 1.41 / 4.55 | 0.138 / 0.0232 / 0.0089 | 1.02 / 1.6 · 1.01 / 1.24 · 1 / 1.06 | 0.00278 / 3.52e-05 / 1.97e-07 | 0 | – |
| chain_5 | chain | 50 | 0 | 1.660 | – | 3.28 / 5.02 / 6.62 | – · – · – | 11.3 / 19.4 / 41.7 | 0 | – |
| chain_6 | chain | 50 | 0.01 | 1.553 | – | 2.07 / 2.49 / 2.41 | – · – · – | 2.59 / 1.37 / 0.701 | 0 | – |
| chain_7 | chain | 50 | 0.1 | 1.283 | – | 1.24 / 1.13 / 0.907 | – · – · 5.62 / 16.4 | 0.915 / 0.389 / 0.152 | 0 | – |
| chain_8 | chain | 50 | 1 | 0.989 | 34.8 / 292 | 0.672 / 0.402 / 0.245 | 1.82 / 4.1 · 1.47 / 3.53 · 1.17 / 1.91 | 0.264 / 0.0403 / 0.00638 | 0 | – |
| chain_9 | chain | 50 | 10 | 0.682 | 1.81 / 16.4 | 0.151 / 0.0288 / 0.0109 | 1.04 / 1.74 · 1.01 / 1.75 · 1 / 1.06 | 0.011 / 6.33e-05 / 9.07e-07 | 0 | – |
| chain_10 | chain | 100 | 0 | 1.685 | – | 2.55 / 4.11 / 5.79 | – · – · – | 8.18 / 15 / 26.1 | 0 | – |
| chain_11 | chain | 100 | 0.01 | 1.587 | – | 2.07 / 2.67 / 2.65 | – · – · – | 2.97 / 1.68 / 0.965 | 0 | – |
| chain_12 | chain | 100 | 0.1 | 1.343 | – | 1.34 / 1.25 / 1.05 | – · – · – | 1 / 0.48 / 0.23 | 0 | – |
| chain_13 | chain | 100 | 1 | 1.046 | – | 0.872 / 0.539 / 0.387 | 2.02 / 5.82 · 1.56 / 3.94 · 1.2 / 2.27 | 0.334 / 0.0727 / 0.0169 | 0 | – |
| chain_14 | chain | 100 | 10 | 0.793 | 1.94 / 12.3 | 0.433 / 0.0797 / 0.0449 | 1.05 / 1.71 · 1.03 / 1.78 · 1 / 1.08 | 0.0185 / 0.000363 / 8.87e-06 | 0 | – |
| chain_15 | chain | 200 | 0 | 1.688 | – | 2.49 / 4.76 / 6.05 | – · – · – | 11.3 / 17.7 / 36.1 | 0 | – |
| chain_16 | chain | 200 | 0.01 | 1.596 | – | 1.98 / 2.44 / 2.33 | – · – · – | 2.51 / 1.89 / 1.33 | 0 | – |
| chain_17 | chain | 200 | 0.1 | 1.390 | – | 1.32 / 1.26 / 1.15 | – · – · – | 1.16 / 0.702 / 0.452 | 0 | – |
| chain_18 | chain | 200 | 1 | 1.066 | – | 0.937 / 0.787 / 0.626 | 1.89 / 8.48 · 1.5 / 5.59 · 1.17 / 2.7 | 0.676 / 0.261 / 0.129 | 0 | – |
| chain_19 | chain | 200 | 10 | 0.812 | 1.73 / 12.4 | 0.463 / 0.229 / 0.108 | 1.03 / 1.78 · 1.02 / 2.05 · 1 / 1.13 | 0.109 / 0.0106 / 0.00147 | 0 | – |
| chain_20 | chain | 500 | 0 | 1.689 | – | 3.73 / 5.42 / 9.24 | – · – · – | 13.6 / 25.3 / 35.4 | 0 | – |
| chain_21 | chain | 500 | 0.01 | 1.602 | – | 1.97 / 2.39 / 2.3 | – · – · – | 2.44 / 2.02 / 1.3 | 0 | – |
| chain_22 | chain | 500 | 0.1 | 1.391 | – | 1.33 / 1.26 / 1.18 | – · – · – | 1.14 / 0.786 / 0.477 | 0 | – |
| chain_23 | chain | 500 | 1 | 1.066 | – | 0.947 / 0.817 / 0.669 | 1.92 / 10.6 · 1.5 / 7.32 · 1.18 / 2.95 | 0.684 / 0.317 / 0.121 | 0 | – |
| chain_24 | chain | 500 | 10 | 0.811 | 1.69 / 12.6 | 0.47 / 0.242 / 0.116 | 1.04 / 1.9 · 1.01 / 2.12 · 1 / 1.16 | 0.107 / 0.0121 / 0.00121 | 0 | – |
| chain_blind_53 | chain_blind | 50 | 0.1 | 1.653 | – | 2.28 / 3.33 / 4.06 | – · – · – | 5.52 / 5.01 / 3.3 | 0 | – |
| chain_blind_54 | chain_blind | 50 | 1 | 1.649 | – | 2.26 / 3.3 / 4.03 | – · – · – | 5.43 / 4.95 / 3.24 | 0 | – |
| chain_blind_55 | chain_blind | 50 | 10 | 1.651 | – | 2.29 / 3.32 / 4.09 | – · – · – | 5.5 / 4.89 / 3.25 | 0 | – |
| chain_blind_56 | chain_blind | 100 | 0.1 | 1.637 | – | 2.44 / 3.76 / 4.41 | – · – · – | 4.9 / 6.01 / 5.04 | 0 | – |
| chain_blind_57 | chain_blind | 100 | 1 | 1.635 | – | 2.45 / 3.74 / 4.37 | – · – · – | 4.86 / 5.95 / 4.91 | 0 | – |
| chain_blind_58 | chain_blind | 100 | 10 | 1.633 | – | 2.44 / 3.72 / 4.37 | – · – · – | 4.75 / 5.92 / 4.8 | 0 | – |
| chain_blind_59 | chain_blind | 200 | 0.1 | 1.667 | – | 2.18 / 3.08 / 4.08 | – · – · – | 4.45 / 5.76 / 4.71 | 0 | – |
| chain_blind_60 | chain_blind | 200 | 1 | 1.666 | – | 2.14 / 3.03 / 3.94 | – · – · – | 4.48 / 5.78 / 4.77 | 0 | – |
| chain_blind_61 | chain_blind | 200 | 10 | 1.668 | – | 2.21 / 3.09 / 4.03 | – · – · – | 4.43 / 5.81 / 4.72 | 0 | – |
| chain_tracks_37 | chain_tracks | 20 | 0.01 | 2.130 | – | 2.32 / 2.35 / 2.12 | – · – · – | 0.627 / 0.193 / 0.0257 | 10 | 565 |
| chain_tracks_38 | chain_tracks | 20 | 0.1 | 1.903 | – | 1.99 / 1.95 / 1.79 | – · – · – | 0.29 / 0.0454 / 0.00444 | 12 | 247 |
| chain_tracks_39 | chain_tracks | 20 | 1 | 1.681 | – | 1.68 / 1.55 / 1.32 | – · – · – | 0.358 / 0.0851 / 0.02 | 9 | 58.9 |
| chain_tracks_40 | chain_tracks | 20 | 10 | 1.350 | – | 1.36 / 0.986 / 0.789 | – · 45.8 / 146 · 3.31 / 9.52 | 0.0748 / 0.00558 / 0.000127 | 13 | 12.9 |
| chain_tracks_41 | chain_tracks | 50 | 0.01 | 2.147 | – | 2.39 / 2.4 / 2.21 | – · – · – | 0.814 / 0.242 / 0.0285 | 36 | 3.14e+03 |
| chain_tracks_42 | chain_tracks | 50 | 0.1 | 1.905 | – | 2.07 / 1.99 / 1.83 | – · – · – | 0.561 / 0.104 / 0.0121 | 28 | 726 |
| chain_tracks_43 | chain_tracks | 50 | 1 | 1.665 | – | 1.68 / 1.49 / 1.41 | – · – · – | 0.41 / 0.0689 / 0.0118 | 30 | 144 |
| chain_tracks_44 | chain_tracks | 50 | 10 | 1.345 | – | 1.38 / 1.02 / 0.85 | – · – · 4.83 / 13.9 | 0.237 / 0.0224 / 0.00111 | 35 | 25 |
| chain_tracks_45 | chain_tracks | 100 | 0.01 | 2.179 | – | 2.36 / 2.55 / 2.47 | – · – · – | 1.33 / 0.488 / 0.139 | 63 | 5.59e+03 |
| chain_tracks_46 | chain_tracks | 100 | 0.1 | 1.936 | – | 2.08 / 2.05 / 2.04 | – · – · – | 0.583 / 0.16 / 0.0302 | 67 | 1.15e+03 |
| chain_tracks_47 | chain_tracks | 100 | 1 | 1.714 | – | 1.68 / 1.49 / 1.41 | – · – · – | 0.863 / 0.264 / 0.114 | 60 | 158 |
| chain_tracks_48 | chain_tracks | 100 | 10 | 1.353 | – | 1.35 / 1.06 / 0.888 | – · – · 6.16 / 20.6 | 0.198 / 0.0171 / 0.000856 | 54 | 21.5 |
| chain_tracks_49 | chain_tracks | 200 | 0.01 | 2.144 | – | 2.33 / 2.45 / 2.41 | – · – · – | 0.875 / 0.265 / 0.0934 | 144 | 8.75e+03 |
| chain_tracks_50 | chain_tracks | 200 | 0.1 | 1.936 | – | 2.12 / 2.1 / 1.95 | – · – · – | 0.826 / 0.269 / 0.0335 | 141 | 1.36e+03 |
| chain_tracks_51 | chain_tracks | 200 | 1 | 1.730 | – | 1.7 / 1.6 / 1.47 | – · – · – | 0.497 / 0.0723 / 0.012 | 140 | 236 |
| chain_tracks_52 | chain_tracks | 200 | 10 | 1.407 | – | 1.35 / 1.05 / 0.894 | – · – · 6.49 / 19.5 | 0.463 / 0.094 / 0.0188 | 130 | 26.4 |
| chain_hub_25 | chain_hub | 20 | 0.01 | 1.803 | – | 1.91 / 0 / 0 | – · 1 / 1 · 1 / 1 | 0.226 / 0.0157 / 0.00127 | 1 | 19.1 |
| chain_hub_26 | chain_hub | 20 | 0.1 | 1.565 | – | 1.48 / 0 / 0 | – · 1 / 1 · 1 / 1 | 0.153 / 0.0173 / 0.00247 | 1 | 16.5 |
| chain_hub_27 | chain_hub | 20 | 1 | 1.266 | – | 1.05 / 0 / 0 | – · 1 / 1 · 1 / 1 | 0.0169 / 0.000691 / 1.45e-05 | 1 | 8.2 |
| chain_hub_28 | chain_hub | 50 | 0.01 | 1.850 | – | 2.03 / 0 / 0 | – · 1 / 1 · 1 / 1 | 0.235 / 0.0165 / 0.00132 | 1 | 38.6 |
| chain_hub_29 | chain_hub | 50 | 0.1 | 1.610 | – | 1.61 / 0 / 0 | – · 1 / 1 · 1 / 1 | 0.186 / 0.0195 / 0.00291 | 1 | 18.7 |
| chain_hub_30 | chain_hub | 50 | 1 | 1.271 | – | 1.08 / 0 / 0 | – · 1 / 1 · 1 / 1 | 0.0597 / 0.00279 / 8.81e-05 | 1 | 5.42 |
| chain_hub_31 | chain_hub | 100 | 0.01 | 1.883 | – | 2.19 / 0 / 0 | – · 1 / 1 · 1 / 1 | 0.5 / 0.116 / 0.0272 | 1 | 64.1 |
| chain_hub_32 | chain_hub | 100 | 0.1 | 1.634 | – | 1.68 / 0 / 0 | – · 1 / 1 · 1 / 1 | 0.331 / 0.0694 / 0.0159 | 1 | 25.8 |
| chain_hub_33 | chain_hub | 100 | 1 | 1.317 | – | 1.18 / 0 / 0 | – · 1 / 1 · 1 / 1 | 0.091 / 0.00617 / 0.000286 | 1 | 7.57 |
| chain_hub_34 | chain_hub | 200 | 0.01 | 1.896 | – | 2.16 / 0 / 0 | – · 1 / 1 · 1 / 1 | 0.941 / 0.427 / 0.234 | 1 | 126 |
| chain_hub_35 | chain_hub | 200 | 0.1 | 1.671 | – | 1.71 / 0 / 0 | – · 1 / 1 · 1 / 1 | 0.455 / 0.118 / 0.0433 | 1 | 44.3 |
| chain_hub_36 | chain_hub | 200 | 1 | 1.408 | – | 1.35 / 0 / 0 | – · 1 / 1 · 1 / 1 | 0.429 / 0.111 / 0.0395 | 1 | 13.2 |
| random_spd_62 | random_spd | 30 | – | 2.774 | – | 4.27 / 2.55 / 0.715 | – · – · 1.48 / 6.45 | – / – / – | – | – |
| random_spd_63 | random_spd | 30 | – | 3.418 | – | 5 / 2.21 / 0.337 | – · – · 1.11 / 2.38 | – / – / – | – | – |
| random_spd_64 | random_spd | 30 | – | 2.897 | – | 4.12 / 2.34 / 0.897 | – · – · 3.66 / 30.3 | – / – / – | – | – |
| random_spd_65 | random_spd | 60 | – | 3.067 | – | 4.25 / 2.88 / 0.907 | – · – · 8.75 / 39.1 | – / – / – | – | – |
| random_spd_66 | random_spd | 60 | – | 2.923 | – | 4.36 / 3.29 / 0.922 | – · – · 9.14 / 36.7 | – / – / – | – | – |
| random_spd_67 | random_spd | 60 | – | 3.102 | – | 5.15 / 4.17 / 1.5 | – · – · – | – / – / – | – | – |
| random_spd_68 | random_spd | 120 | – | 3.260 | – | 5.63 / 5.46 / 2.83 | – · – · – | – / – / – | – | – |
| random_spd_69 | random_spd | 120 | – | 3.295 | – | 6.46 / 6.56 / 3.65 | – · – · – | – / – / – | – | – |
| random_spd_70 | random_spd | 120 | – | 3.401 | – | 5.82 / 5.57 / 2.57 | – · – · – | – / – / – | – | – |

## 4. Worst cases

The sweep keeps the worst value of each check per problem, not the block where it occurs. Positions are therefore given to the problem, except where I recomputed a quantity.

**Invariants closest to violation.** The complete list is §3.1.

| Check | Worst | Problem | Reading |
|---|---|---|---|
| Lemma 1.3/1.4: seeded truncated recursion L ≤ G on H_TT | +4.6e-10 | chain_20 (pure chain, n = 500) | An equality case up to rounding. On this chain cond(D⁻¹ᐟ²HD⁻¹ᐟ²) = 4.0e13, and rounding in the extended-precision recursions and in the refined reference reaches this level. The six largest values of this check are all on this problem (4.6e-10 to 1.9e-10). |
| chain cut ≥ Σ^T | +1.1e-10 | chain_20, K = 3 | the same problem; an exact case |
| cut upper ≥ Σ | +2.6e-11 | chain_20, K = 3 | exact case: the cut bound of an anchored pure-chain patch equals the marginal |
| C4 one-step ũ ≤ s + Aũ | +7.0e-12 | chain_0, K = 3 | exact local solve |
| C3 E ≥ err, C3 one-step | +8.0e-14 | random_spd_64, K = 3 | exact case |
| every other check | ≤ +5.0e-14 | | rounding level |

**Worst ratios, per certificate.**

| Certificate | Worst median E/e | Worst 95 % | Problem | Reading |
|---|---|---|---|---|
| C1 | 34.8 | 292 | chain_8 (n = 50, m/p = 1) | ρ(M) = 0.989, so (I − M)⁻¹ amplifies up to 1/(1 − 0.989) ≈ 90 |
| C2 | 1 | 1 | all 34 | largest \|E/e − 1\| = 3.3e-16, at block 413 of chain_20, Jacobi iteration 2 (recomputed) |
| C3, K = 1 | 2.02: chain_13 (n = 100, m/p = 1) | 10.6: chain_23 (n = 500, m/p = 1) | | ρ(A) = 0.872 and 0.947 |
| C3, K = 2 | 45.8 | 146 | chain_tracks_40 (n = 20, m/p = 10) | ρ(A) = 0.986: the certificate exists but the leak recursion barely contracts |
| C3, K = 3 | 9.14: random_spd_66 (60 blocks) | 39.1: random_spd_65 (60 blocks) | | ρ(A) = 0.922 and 0.907 |
| C4 | not recorded | | | §6 |
| Hub correction, θ_S (oracle aggregate) | 8.7e3 | | chain_tracks_49 (n = 200, m/p = 0.01, 144 hubs) | the aggregate grows with the number of hubs |

**Other worst cases.**
- **Oracle cross-check.** x − x̄ agrees with e = H⁻¹g to 4.9e-6, relative to the largest error, on chain_20. This is cancellation in x − x̄ at cond 4.0e13, and it is why the oracle uses e = H⁻¹g.
- **Lemma 1.6 decay bound.** On some far-apart supernode pairs the bound λ · ∏ψ̄ overflows to +∞ (two overflow warnings in `results/phaseB/run.log`, from `certify/check.py:243`). `viol_scalar` counts +∞ as never violated, so those comparisons are vacuous rather than wrong. The warning is printed once per code location, so the problems that produced it are not recorded. The tightest finite margin of this check is −5.8e-8 (chain_24).
- **Run time.** The slowest problem is the pure 500-pose chain: 5 h 53 min (21,166 s) of the sweep's 32.6 CPU-hours. Its extended-precision solves need the most refinement steps: the full 6000 × 6000 H_TT covariance took 5 steps of about 14 minutes each.

## 5. Facts and code references

Not applicable to Phase B. The only changes to the repository outside `certify/` are the Python-binding accessors listed in §7. They are read-only, apart from exposing the existing `GbpSolver::freeze` and `setFrozenRegularizer`. The C++ library is unchanged in this phase.

## 6. Notes, decisions, and what could not be done

- **Bugs found by the invariants and fixed.** Rule 1: every violation is a bug; each was found, fixed and re-run.
  1. *Singular boundary-cut matrices.* On a pure chain, a patch without the anchor has no absolute information once its boundary factors are cut, so H_PP − B_P is exactly singular and the honest upper bound is +∞. Cholesky can still succeed through rounding and return a meaningless finite "bound". It violated the cut bound by up to 1.3e-6 in the first sweep. Now a patch is locally certifiable only if the double-double-refined solve with the block's unit columns converges; a singular matrix fails that test.
  2. *Krylov monotonicity.* The bounds for ℓ and ℓ+1 were computed from separately built bases, which rounding made slightly non-nested; ℓ = 5 → 6 failed monotonicity by up to 3.4e-7. Now one block-Lanczos basis per block (D-orthonormal) serves every ℓ. The bound for ℓ uses its leading blocks and the leading block of one extended-precision Cholesky factor, so Lemma 2.2 holds as computed.
  3. *Krylov rank test.* A first version dropped directions by the eigenvalues of the Gram matrix, which squares the condition number. It kept rounding noise and produced a numerically singular compression. It now uses an SVD in Cholesky-scaled coordinates.
  4. *Accuracy at 1e-9 on ill-conditioned chains.* The pure WNOA chain has cond(D⁻¹ᐟ²HD⁻¹ᐟ²) = 3.4e6, 4.2e8, 4.1e9, 1.3e12, 4.0e13 for n = 20…500. Two changes were needed:
     - Residuals in 80-bit precision stall at cond·1.1e-19, which is 4e-6 at n = 500. All refinement residuals are now double-double dot products (Ogita–Rump–Oishi Dot2), and refinement runs to convergence: 2–5 steps to full double accuracy even at n = 500.
     - For an exact certificate (the cut bound of an anchored pure-chain patch equals the marginal), rounding in the stored matrices alone (ε·|H|, amplified by the conditioning) gave 1.2e-9. The model is now assembled exactly: every J_fᵀΛ_fJ_f and every sum in H and b is formed in extended precision and stored as double hi + lo. That case now agrees to 4e-13 (n = 100) and 3e-11 (n = 200); the final sweep's worst over all sizes is 2.6e-11.

     No tolerance was changed. The oracle error is computed as e = H⁻¹g with the exact model, and x − x̄ is kept as a cross-check (§4).
- **Hubs "one at a time" (PROOF_NOTE §4.5).** The note says it is valid but gives no formula. Applying Lemma 4.4 to one hub h with the chain certificate is **not** valid: e_h = [Ŝ⁻¹ r_S]_h depends on the other hubs' residuals through the off-diagonal blocks of Ŝ⁻¹, and Lemma 4.4 for a single hub needs the error of the problem with only that hub clamped. I implemented:
  - E_i = ĉ_i + Σ_h φ̄_ih·E_h for i ∈ T, with ĉ the chain certificate C4 on H_TT (all hubs clamped), φ̄_ih Lemma 4.5 for the single hub h, and E_h ≥ ‖e_h‖_{H_hh} the C3 certificate of hub block h;
  - E_h itself for the hub blocks.

  Validity follows from Theorem 4.2 with S the union of hubs: e_T = ẽ_T − Σ_h H_TT⁻¹H_Th·e_h, since hubs are never directly coupled. Then use the triangle inequality and Lemma 4.5 per hub. It loosens with the number of hubs, as the note says. It is available only where C3 has ρ(A) < 1 at the hub blocks. It passes every invariant where it applies (chain with hub: K = 2, 3). θ_h and the joint oracle θ_S are computed as diagnostics.
- **C4 per block.** C4 bounds the supernode error in the metric D_I = (H_TT)_II. The per-block value multiplies by √κ_i, κ_i = λ_max(H_ii^{1/2}[D_I⁻¹]_ii H_ii^{1/2}) (Cauchy–Schwarz on v_i = E_iᵀv_I); κ_i = 1 for single-block supernodes. C4 alone bounds the hub-clamped error ẽ, and it bounds e only when S = ∅. With hubs the certificate for e is E_hub. Both are logged, with err_clamped as an extra column.
- **Norms in C1/C2.** Block norms are in the D-metric, ‖X_ij‖ = ‖D_i^{1/2}X_ijD_j^{-1/2}‖, as Theorem 3.2 requires for P = D to recover Theorem 3.1. The unit test checks this.
- **ρ < 1** is certified by a Collatz–Wielandt upper bound (a positive x with max_i (Ax)_i/x_i < 1); certificates are computed only then. The monotone iteration of Theorem 3.5 is run and its count logged. That the iterates are nondecreasing and below c* is asserted on every solve.
- **Parameters fixed before any result.**
  - Q_C = kitti_05; Δt = the KITTI 06 keyframe intervals.
  - Anchor: 1e4·p·I₁₂ on pose 0 (the pure chain has a constant-velocity null space without it).
  - p = the mean diagonal of one GP factor's contribution to a pose block.
  - Hub information p·I₆; point information p·I₃, 2 points per pose on tracks of 2–4 poses.
  - Iterates: block Gauss–Seidel and block Jacobi from x̄ plus a unit-D-norm perturbation, kept at iterations 0, 1, 2, 5, 10, 20, 50.
  - w = 3.
- **Not recorded by the sweep:**
  - the block where each worst case occurs (only the worst value per check per problem is kept);
  - C4's E/e ratios (only ρ(A) for C4 is kept).

  Both could be recovered by rerunning single problems. The problems concerned include the 500-pose chains, at up to 6 hours each, so I have not done it.
- **Not done:**
  - the brief's "chain + one hub" at n = 500 (hub chains stop at n = 200 for run time);
  - Krylov bounds beyond ℓ = 6.

## 7. Files

- Module: `certify/{model,linalg,precise,chain,covbounds,radii,certs,hubs,synth,check}.py`.
- Tests: `certify/tests/{conftest,test_chain,test_covbounds,test_certs,test_hubs}.py`. Run `python3 -m pytest certify/tests` (about 1 minute).
- Python bindings:
  - `python/core/bind_gbp_solver.cpp`: `freeze`, `dynamic_schedule`, `set_frozen_regularizer`, `frozen_regularizers`;
  - `python/graph/bind_edge.cpp`: message `tau`, edge `frozen`;
  - `python/graph/types/factors/factor_accessors.h` (new) and `bind_se3_posevel_{gp,prior,pose,point}.cpp`: `J`, `error`, `Sigma`, `J_lin`, `error_lin`, `frozen`, `edge(i)`;
  - `python/graph/types/variables/bind_se3_posevel.cpp`: `frozen`, `mu_lin`, `z`.
- Sweep: `scripts/phaseB_run.py` → `results/phaseB/{problems.csv, invariants.csv, rhoM_curve.csv, gateB.json, partial.jsonl, run.log}`. `superseded_run*_partial.jsonl` are the two earlier sweeps whose failures are described in §6.
- Tables: `scripts/phaseB_summary.py`. F4 numbers: `results/phaseB/f4_bindings.json`.
- Commit: the commit that adds this file (`git log -- reports/phaseB.md`).
