# Agent Brief: Self-Certifying Gaussian Belief Propagation in G-solver

> **Errata, 2026-09-25** (from Phase A, facts F3–F5; theorems unchanged)
>
> (i) **F4.** On the chain, $G_k=(H_{kk}-\Lambda^f_{kk})+\text{message}_{k+1\to k}$, where $\Lambda^f_{kk}$ is the forward GP factor's diagonal contribution to $H_{kk}$ and $\text{message}_{k+1\to k}$ is the stored factor-to-variable message of the GP factor $f(k,k+1)$ to $k$. The remark "$G_k=P_{kk}+$ message" in proof note §1.1 and brief §3 is wrong by that term. Lemma 1.4's seed $P_{mm}^-$ is exactly $H_{mm}-\Lambda^f_{mm}$, consistent with this. Nothing else changes. (Verified to relative error $10^{-11}$ on a 600-pose GP chain; `results/phaseA/f4_chain.json`.)
>
> (ii) **F3.** G-solver's interpolation is affine in the GP local variables, hence in tangent coordinates within a fixed chart only to first order. Theorem 5.6 applies to the linearized interpolant in the frozen charts. Phase E must report the interpolant's linearization error at observed tangent magnitudes as a caveat.
>
> (iii) **F5.** There is a factor of 2 in every stereo rotation Jacobian (`se3_posevel_stereo_point.cpp:77`). It is recorded and left unchanged: the certified model is the assembled one.

You are the implementing and experimenting agent for a research project. This brief is complete and self-contained: it gives the mission, the rules, the proven theory in implementable form, the phased work plan with pass/fail gates, and the report format. The mathematics below has been proved and reviewed; treat every theorem as correct. Your job is to implement it faithfully, test that your implementation satisfies the theorems' invariants, measure how tight the bounds are on real G-solver graphs, and report honestly.

Repository: `rvp-group/gsolver` (already available to you). Work on branch `certify`. Do all analysis in Python (NumPy, SciPy sparse) on data exported from G-solver; touch the C++ solver only to add instrumentation and, in Phase D, the stopping/sleep-wake schedule.

---

## 0. Mission

Continuous-time SLAM in G-solver runs Gaussian Belief Propagation (GBP) on a factor graph whose variables are pose-velocity nodes (12-dimensional tangent) and landmarks, linked by a white-noise-on-acceleration (constant-velocity) Gaussian-process prior between consecutive poses and by visual measurement factors.

We want the solver to know, for each pose and without ground truth, when further inference is no longer worth doing. Concretely, for each variable block `i` we compute two numbers from the factor graph and the current GBP state alone:

1. **A certified error bound** `E_i` such that the exact solution of the current linear model cannot differ from the current estimate at block `i` by more than `E_i`. This bounds how much more computation could change the state.
2. **A certified lower bound on statistical uncertainty** `S_lo_i` — a lower bound on the (1−δ) radius of the exact estimator's own error under the measurement noise model. No amount of computation can reduce the error below this scale.

The stopping rule is `E_i ≤ η · S_lo_i` with `η = 0.1, δ = 0.05` (fixed; do not tune). A pose satisfying it is put to sleep; it wakes when new information could invalidate its certificate.

**Success** is showing on real G-solver graphs that (a) every bound is valid, (b) `E_i` is within a small factor of the true error (target ≤ 3× on stereo BA and chain interiors, ≤ 10× on ChArUco), (c) the stopping rule saves updates versus G-solver's current schedule at equal accuracy, (d) the number of active states stays flat as the trajectory grows, and (e) the certificates extend to interpolated poses at arbitrary sensor timestamps.

**Failure** is a valid bound so loose that no state ever sleeps. If that happens, the logs must say exactly which term is loose (§7 explains how).

---

## 1. Non-negotiable rules

1. **Validity invariants are never relaxed.** Every certificate must be ≥ the true error for every block, iteration and dataset, to floating-point tolerance (`1e-9` relative). Every lower bound on covariance must be ⪯ the exact marginal, every upper bound ⪰ it. A single violation is a bug in your code (the theorems are proved). Stop, find it, fix it, report it. Never add a fudge factor, never drop the offending case, never widen a tolerance.
2. **Do not alter solver mathematics** in Phases A–C. Instrumentation only. In Phase D you add a scheduling layer (which nodes update); the message equations are unchanged.
3. **All thresholds are fixed in this brief.** Do not tune `η`, `δ`, `K`, or any baseline parameter to make results look better. Where a baseline needs a parameter, choose it by a stated rule before seeing results.
4. **Toy tests before real data.** Phase B must pass on synthetic problems with known answers before any real export is used.
5. **Oracle versus local.** Many quantities have an exact "oracle" value (computed from the full matrix in Python) and a "local" certified bound (computed from a patch). Compute and log both, clearly labelled. The oracle is for diagnosis; only local quantities can be claimed as computable in the solver.
6. **Report failures and uncertainties plainly.** A report that says "this could not be done because X" is more valuable than a workaround. Never claim a step succeeded when it did not.
7. **Every report ends with the worst case**, not only the median.

---

## 2. Definitions (use these names in code and logs)

### 2.1 The current linear model

At any instant every factor `f` has a linearization point, a whitened Jacobian `J_f` (rows = whitened residual dimension, columns = tangent coordinates of the variables it touches) and whitened residual `r_f`. The **current model** is

```
H = Σ_f J_fᵀ Λ_f J_f + G        b = Σ_f J_fᵀ Λ_f r_f + b_G
```

where `G ⪰ 0` is the gauge-fixing (anchor/prior) term. `H` must be symmetric positive definite (SPD): this is assumption **(A0)**. The **exact solution of the current model** is `xbar = H⁻¹ b`, computed offline with a direct sparse solver (e.g. `scipy.sparse.linalg.splu` or CHOLMOD). This definition works whether G-solver relinearizes all factors together or one at a time: at any instant the assembled `H, b` are what they are.

If G-solver's GBP works in an information form where the assembled system is not literally formed, you must assemble it from the factor Jacobians. Verify (Phase A) that the direct solve of your assembled system matches G-solver's converged GBP state.

**Chart consistency (the most likely cause of a Gate A failure).** `H` and `b` live in the tangent charts at the factors' linearization points. The exported state `x` must be expressed in the *same* charts: for each variable export its linearization point (the manifold element the chart is centred on) and its current tangent coordinate relative to that point. Errors `e_i = x_i − xbar_i` are then tangent-space differences. Never compare manifold elements directly, and never mix charts from different linearization epochs within one export. If G-solver stores beliefs as manifold means with a tangent remap each iteration, export both the mean and the remap so the tangent coordinate can be recovered exactly.

### 2.2 Blocks, graph, metric

- Variables are blocks `i ∈ V` with dimension `d_i` (12 for pose-velocity nodes; 3 or 6 for landmarks). `H_ij` is the `(i,j)` block.
- The **variable graph** joins `i` and `j` when some factor touches both. `dist(i,j)` is graph distance.
- `D = blkdiag(H_11, …, H_nn)`; `D_i = H_ii`.
- Local metric: `‖v‖_{D_i} = sqrt(vᵀ D_i v)`. Errors of block `i` are always measured in this metric. It is unit-free (whitens rotation/translation/velocity coordinates) and makes the trivial statistical lower bound free (§4.3).
- For any estimate `x` (any GBP iterate): **residual** `g = H x − b`, **error** `e = x − xbar`. Then `H e = g` exactly. `g_i = Σ_j H_ij x_j − b_i` is computable at node `i` from its neighbours.
- `Σ = H⁻¹`; `Σ_ii = [H⁻¹]_ii` is the exact marginal covariance of block `i` (oracle: selected inverse or dense inverse on these problem sizes).

### 2.3 Patches

For node `i` and hop count `K`, the **patch** `P = P_i(K)` is the set of variables within graph distance `K` of `i`, together with every factor whose variables all lie in `P`. `E = V \ P` is the exterior. `∂P` = nodes of `P` adjacent to `E`. `H_PP`, `H_PE`, `g_P` are the corresponding sub-blocks.

**Boundary factors** `F_∂` are the factors touching both `P` and `E`. Each contributes `J_fᵀ Λ_f J_f` with `J_f = [J_{f,P}, J_{f,E}]`. Define

```
B_P = Σ_{f ∈ F_∂} J_{f,P}ᵀ Λ_f J_{f,P}     (⪰ 0; the part of H_PP that came from boundary factors)
```

The patch is **locally certifiable** if `H_PP − B_P ≻ 0` (Cholesky succeeds). Log this flag per patch.

**Blocks inside patches.** In certificate C3 every variable is an ordinary block: landmarks (3-dof points) are included in a patch by graph distance like any other block, and the ChArUco board is a single 6-dof `SE(3)` block whose corner observations are all factors touching that block. In certificate C4 short-track landmarks are absorbed into supernodes (§2.4) and the board is a hub. Both conventions must be used and logged; they are two different certificates.

### 2.4 Supernodes and hubs

Fix a bandwidth `w` (use `w = 3` unless the data suggests otherwise; log the choice). Group consecutive poses into supernodes of `w` poses. A landmark whose observations span at most `w` consecutive poses is assigned to the earlier of the (at most two) supernodes it touches. Let `T` be all poses and all such short-track landmarks, in supernode order. Let `S` (the **hub set**) be everything else: landmarks whose track spans more than `w` consecutive poses, and one endpoint pose of each loop-closure factor whose endpoints are more than `w` poses apart.

Fact (proved): with this grouping `H_TT` is block-tridiagonal in supernode order. Verify this structurally in code (assert no nonzero blocks beyond the first off-diagonal in supernode indexing).

On ChArUco, `S` is the board. On stereo BA with short tracks, `S` may be empty.

---

## 3. Chain lemmas (implement as `chain.py`)

Let `P` be any block-tridiagonal SPD matrix with blocks `P_kl` (in Phase C this is `H_TT` in supernode blocks). `Σ = P⁻¹`.

**Backward Schur complements.**

```
G_n = P_nn
G_k = P_kk − P_{k,k+1} G_{k+1}⁻¹ P_{k+1,k}     for k = n−1, …, 1
```

Forward ones `F_k` are the mirror image. All SPD. On a GBP chain, `G_k = P_kk + (converged message from k+1 to k)`.

**Transfer operators and product formula (Lemma 1.2, proved).** With `Ψ_k = −G_{k+1}⁻¹ P_{k+1,k}`, for `i < j`:

```
Σ_ij = Σ_ii · Ψ_iᵀ Ψ_{i+1}ᵀ ⋯ Ψ_{j−1}ᵀ
```

Unit test: build a random block-tridiagonal SPD matrix, compute `Σ` densely, and check this identity to `1e-10`.

**Local lower bounds on `G_k` (Lemmas 1.3–1.4, proved).** If `L_{k+1} ⪯ G_{k+1}` and both SPD, then `L_k := P_kk − P_{k,k+1} L_{k+1}⁻¹ P_{k+1,k} ⪯ G_k`. Seed at any truncation node `m`: `G_m ⪰ P_mm⁻`, where `P_mm⁻` is `P_mm` with the diagonal contribution of every factor touching a node beyond `m` removed (for the GP chain: remove the diagonal block that the GP factor `(m, m+1)` contributes to `P_mm`). So from any window `[k, m]` you get certified `L_k ⪯ G_k` using only blocks in the window.

Unit test: on a random chain, run the truncated recursion from several `m` and assert `L_k ⪯ G_k` (min eigenvalue of `G_k − L_k ≥ −1e-10`) for all `k < m`.

**Certified decay factor (Definition 1.5).** For link `(k, k+1)` and a lower bound `L_{k+1} ⪯ G_{k+1}`:

```
psibar_k = λ_max( D_{k+1}^{1/2} L_{k+1}⁻¹ D_{k+1}^{1/2} ) · ‖ D_{k+1}^{−1/2} P_{k+1,k} D_k^{−1/2} ‖₂
```

with `D_k = P_kk`. **Decay of influence (Lemma 1.6, proved):** for `i < j`,

```
‖ D_i^{1/2} Σ_ij D_j^{1/2} ‖₂ ≤ λ_max(D_i^{1/2} Σ_ii D_i^{1/2}) · Π_{k=i}^{j−1} psibar_k
```

Unit test: check this inequality on random chains (oracle left side).

Interpretation to keep in mind: the second factor of `psibar_k` is the normalized coupling (≈ ½ on a prior-dominated WNOA link); the first is ≥ 1 and measures how much of node `k+1`'s information survives after marginalizing the future. `psibar_k < 1` iff the node retains more than a coupling's worth of information from its own measurements and its past link. On a pure prior chain `psibar ≈ 1`: no decay. That is correct physics, not a bug.

---

## 4. Marginal covariance bounds and statistical radii (implement as `covbounds.py`)

### 4.1 Lower bounds (Lemma 2.1, proved)

For any matrix `V` with orthonormal columns whose span contains the coordinate directions `E_i` of block `i`:

```
Σ_ii ⪰ E_iᵀ V (Vᵀ H V)⁻¹ Vᵀ E_i
```

Implement three instances, all must be logged:

- **trivial**: `V = E_i` → `Σ_ii ⪰ H_ii⁻¹`.
- **patch**: `V` = coordinates of `P_i(K)` → `Σ_ii ⪰ [(H_PP)⁻¹]_ii`.
- **krylov**: `V = orth([E_i, H E_i, …, H^{ℓ−1} E_i])` (block Krylov, ℓ steps; each step is one sparse product; use ℓ = 3 and 5) → `Σ_ii ⪰ E_iᵀ V (VᵀHV)⁻¹ Vᵀ E_i`.

Monotonicity (proved): larger subspace ⇒ larger bound. Unit test: patch bound at `K+1` ⪰ patch bound at `K`; krylov at `ℓ+1` ⪰ at `ℓ`; all ⪯ oracle `Σ_ii`.

### 4.2 Upper bounds

**Boundary-cut (Theorem 2.4, proved).** If `H_PP − B_P ≻ 0`:

```
Σ_ii ⪯ [(H_PP − B_P)⁻¹]_ii
```

Otherwise the upper bound is `+∞` (log as such). Monotone: larger `K` ⇒ smaller bound.

**Hub-aware (Proposition 4.7, proved; needs §6).** For `i ∈ T`:

```
λ_max(D_i^{1/2} Σ_ii D_i^{1/2}) ≤ mu_hi_T_i + theta_S · phibar_iS²
```

where `mu_hi_T_i` is the cut upper bound computed on `H_TT` (hubs clamped), and `theta_S`, `phibar_iS` are defined in §6.

Unit test: both upper bounds ⪰ oracle `Σ_ii` (on synthetic chain + hub problems and on real exports).

### 4.3 Statistical radii (Proposition 2.7, proved)

Let `q_d(δ) = sqrt(chi2.ppf(1−δ, d))`. With `mu_lo ≤ λ_min(D_i^{1/2} Σ_ii D_i^{1/2})` and `mu_hi ≥ λ_max(...)`:

```
S_lo_i(δ) = sqrt(mu_lo) · q_{d_i}(δ)       S_hi_i(δ) = sqrt(mu_hi) · q_{d_i}(δ)
```

For the trivial subspace `mu_lo = 1`, so `S_lo_i = q_{d_i}(δ)` is always available for free. Patch/krylov give `mu_lo ≥ 1`. Compute `mu_lo` as `λ_min(D_i^{1/2} L D_i^{1/2})` for each lower bound `L`.

---

## 5. Computational certificates (implement as `certs.py`)

All four must be implemented and logged. Each is a per-block scalar `E_i` with the invariant `E_i ≥ ‖e_i‖_{D_i}` (oracle `e = x − xbar`).

### C1 — classical comparison bound (Theorem 3.1; baseline)

```
B = D^{−1/2} H D^{−1/2} = I − R   (R_ii = 0)
M_ij = ‖R_ij‖₂ (i ≠ j), M_ii = 0
if rho(M) < 1:  C = (I − M)⁻¹;   nu_j = ‖ D_j^{−1/2} g_j ‖₂;   E_i = [C nu]_i
```

Log `rho(M)` always; `E_i` only if `rho(M) < 1`.

### C2 — backbone-preconditioned comparison bound (Theorem 3.2)

Backbone `Pbb` = block-tridiagonal SPD matrix formed from: the GP prior chain factors + the block-diagonal part of all measurement information (i.e. `H_ii` for every node) + `G`. Equivalently: `Pbb = H` with every off-block-diagonal entry that is *not* a GP-chain coupling set to zero. `R_H = H − Pbb` (off-diagonal measurement couplings only).

```
Mhat_ij = ‖ [Pbb⁻¹ R_H]_ij ‖₂ (i ≠ j), Mhat_ii = 0        (compute Pbb⁻¹ R_H column-block by column-block with a banded solve)
if rho(Mhat) < 1:  Chat = (I − Mhat)⁻¹;  nuhat_j = ‖[Pbb⁻¹ g]_j‖_{D_j};  E_i = [Chat nuhat]_i
```

Setting `Pbb = D` must reproduce C1 exactly (unit test). On a pure chain (no landmark couplings) `Mhat = 0` and `E_i = ‖[Pbb⁻¹g]_i‖_{D_i}` equals the true error exactly (unit test: ratio = 1 ± 1e-8).

### C3 — recursive patch certificate (Theorem 3.5; the recommended one)

For each node `i`, patch `P_i = P_i(K)`, `K ∈ {1, 2, 3}`:

```
s_i   = ‖ [H_{P_iP_i}⁻¹ g_{P_i}]_i ‖_{D_i}                                  (exact local solve)
a_ij  = ‖ D_i^{1/2} · ( Σ_{b ∈ ∂P_i} [H_{P_iP_i}⁻¹]_ib H_bj ) · D_j^{−1/2} ‖₂    for j ∉ P_i adjacent to ∂P_i; else 0
```

`A = (a_ij)` is sparse nonnegative. If `rho(A) < 1`, the certificate is `c* = (I − A)⁻¹ s ≥ 0`, computed by the monotone iteration `c ← s + A c` from `c = 0` (converges from below; log iteration count) or by a sparse solve. `E_i = c*_i`. Log `rho(A)`, `s_i` (local term) and `c*_i − s_i` (leak term) separately.

Invariant (proved): `c*_i ≥ ‖e_i‖_{D_i}`. Unit test on random SPD block matrices and chains.

### C4 — chain certificate with explicit coefficients (Proposition 3.6)

Same as C3 but on the supernode chain `H_TT` (hubs clamped) with `P_i = [i−K, i+K]` in supernode index. The boundary has two nodes and

```
a_{i, i±(K+1)} = ‖ D_i^{1/2} [H_PP⁻¹]_{i, i±K} H_{i±K, i±(K+1)} D_{i±(K+1)}^{−1/2} ‖₂
```

Also log the certified bound on these coefficients from Lemma 1.6: `λ_max(D_i^{1/2}[H_PP⁻¹]_ii D_i^{1/2}) · psibar^K · m_boundary` (with `psibar` computed inside the patch, seeded at the patch boundary by Lemma 1.4). Log both the coefficient and its certified bound.

### Tightness diagnostics for every certificate

For each `(dataset, snapshot, iteration, block, K)` log `ratio = E_i / ‖e_i‖_{D_i}` and, for C3/C4, `local_share = s_i / E_i`. When `‖e_i‖_{D_i} < 1e-12` (converged block) log the ratio as NaN, not as infinity.

---

## 6. Hubs (implement as `hubs.py`; only active when `S ≠ ∅`)

Partition `H = [[H_TT, H_TS], [H_ST, H_SS]]`, likewise `g`, `e`.

**Exact decomposition (Theorem 4.2, proved).**

```
etilde_T = H_TT⁻¹ g_T                          (error of the hub-clamped problem; oracle for diagnosis)
Shat     = H_SS − H_ST H_TT⁻¹ H_TS             (exact hub marginal precision; oracle)
r_S      = g_S − H_ST etilde_T
e_S      = Shat⁻¹ r_S
e_T      = etilde_T − H_TT⁻¹ H_TS e_S
```

Unit test: reconstruct `e` from this decomposition and compare with the direct solve (`1e-10`).

**Local lower bound on hub precision (Lemma 4.3, proved).** For any subgraph `H'` obtained by deleting factors, with `H' ≻ 0` on its variables and containing the gauge anchor, the marginal precision `Shat_loc` of `S` in `H'` satisfies `Shat_loc ⪯ Shat`. Use `H'` = hub + its observing poses within a window of `K_hub` supernodes on each side of the anchor-containing region (start with the full set of observers on ChArUco; it is small) + factors among them + the anchor. Compute `Shat_loc` by Schur complement. Log the window size.

```
theta_S = λ_max( H_SS^{1/2} Shat_loc⁻¹ H_SS^{1/2} )    (≥ 1; "hub inflation")
```

Also log the oracle `λ_max(H_SS^{1/2} Shat⁻¹ H_SS^{1/2})` for comparison.

**Anchor requirement (property P5).** `Shat_loc` is only SPD if the subgraph `H'` determines the hub's absolute pose, which requires the gauge anchor to be inside `H'`. On ChArUco the anchor is expected to be the first frame and the board is observed from it, so the hub window naturally contains the anchor: verify this (fact F2) and log it. If the anchor is *not* reachable within the hub window, `theta_S` is undefined: do not fake it. Report the situation, treat the hub as an ordinary block in certificate C3 with `K` widened until the patches containing it are locally certifiable, and skip the hub correction for that dataset.

**Hub error bound (Lemma 4.4, proved).** With `N(S)` = supernodes observing `S`, `m_Sj = ‖H_SS^{−1/2} H_Sj D_j^{−1/2}‖₂ (≤ 1)`, and `chat_j ≥ ‖etilde_j‖_{D_j}` any chain certificate (C3/C4 on `H_TT`, `g_T`):

```
E_S = theta_S · ( ‖g_S‖_{H_SS⁻¹} + Σ_{j ∈ N(S)} m_Sj · chat_j )
```

Invariant: `‖e_S‖_{H_SS} ≤ E_S`. Also log the **oracle energy-form aggregate** `theta_S · ‖r_S‖_{H_SS⁻¹}` with `r_S` computed exactly: this is the tightened version proposed if the sum is loose (§7), and its ratio to `E_S` tells us how much the triangle-inequality sum costs.

**Influence coefficient (Lemma 4.5, proved).**

```
phi_iS = ‖ D_i^{1/2} [H_TT⁻¹ H_TS]_i H_SS^{−1/2} ‖₂                (oracle)
phibar_iS = mu_hi_T_i · Σ_{j ∈ N(S)} ( Π_{links i→j} psibar ) · m'_jS ,   m'_jS = ‖D_j^{−1/2} H_jS H_SS^{−1/2}‖₂   (local certified bound)
```

with the tail beyond `K` links bounded by `2 · mu_hi_T_i · psibar^{K+1} / (1 − psibar)` when `psibar < 1` on the tail. Log both `phi_iS` and `phibar_iS`.

**Hub-corrected certificate (Theorem 4.6, proved).**

```
E_i = chat_i + phibar_iS · E_S         for i ∈ T
```

Invariant: `E_i ≥ ‖e_i‖_{D_i}`. Log the three contributions (`chat_i`, `phibar_iS · E_S`, and the oracle `phi_iS · ‖e_S‖_{H_SS}`) separately.

If `S = ∅`, `E_i = chat_i` and none of the above is computed.

---

## 7. Where looseness can come from, and what to do

You do not need to fix looseness; you need to *locate* it. For every certificate report, per dataset, the median and 95th-percentile ratio, and the share of the bound coming from each term. Then apply this table:

| Symptom | Meaning | What to log/do |
| --- | --- | --- |
| C1 ratio huge, `rho(M)` near 1 | Classical bound vacuous on long chains (predicted) | Log `rho(M)` versus trajectory length; this is the motivating figure |
| C3 leak term dominates | Patch too narrow for influence to decay | Log `psibar` along the chain; report where it is ≈ 1 (uninformative segments); rerun with `K+1` |
| C3 local term dominates but ratio still large | Should not happen at patch centre; suspect bug | Check the exact local solve against oracle `e_i` on a patch equal to the whole graph (ratio must be 1) |
| Hub term `phibar_iS·E_S` dominates | Sum over observers loses cancellation | Log the oracle energy aggregate `theta_S·‖r_S‖_{H_SS⁻¹}` versus `E_S`; if much smaller, that is the fix |
| `theta_S` large | Hub poorly determined by local anchored subgraph | Enlarge the hub window; log `theta_S` versus window size |
| Cut upper bound `+∞` for many patches | Property P3 fails: patches not locally certifiable | Log the fraction and the segment lengths; the lower bound and stopping remain valid |

---

## 8. Stopping, sleep/wake, queries

### 8.1 Stopping rule (Theorem 5.1, proved)

A block `i` **stops** at iteration `t` when `E_i(t) ≤ η · S_lo_i(δ)`, `η = 0.1`, `δ = 0.05`. Use the best available `S_lo_i` (max over trivial/patch/krylov lower bounds; all are valid). Log, per iteration, the fraction of blocks that have stopped.

### 8.2 Certified sleep with cheap recheck (Lemma 5.3, Theorem 5.4, proved)

A sleeping node freezes its own state and stops sending updated messages. It maintains a running certificate:

```
E_i_running(t) = s_i(t0) + Σ_{p ∈ P_i} ell_ip · ‖Δg_p‖_{D_p⁻¹} + Σ_j a_ij c_j(t)  [+ phibar_iS · E_S(t) if hubs]
ell_ip = ‖ D_i^{1/2} [H_PP⁻¹]_ip D_p^{1/2} ‖₂     (computed once at sleep time)
Δg_p   = Σ_j H_pj Δx_j                          (accumulated moves of p's neighbours since t0)
```

It wakes (recomputes its exact patch solve and resumes updates) when `E_i_running > η S_lo_i`. This running bound is valid at every `t` (proved). Nodes whose patch is touched by a new factor recompute immediately.

### 8.3 Bounded wake wave (Theorem 5.5, proved)

After adding a factor on variables `V_f`, the change in the recursive certificate at a node at recursion distance `m` from the affected patches is at most `r^m ‖Δs‖_∞ / (1 − r)`, `r = max row sum of A`. Nodes with more slack than this stay asleep. The experiment measures the wake set size directly.

### 8.4 Continuous-time query (Theorem 5.6, conditional on G-solver's interpolation form)

If the interpolated state at `τ ∈ [t_k, t_{k+1}]` is affine in the two endpoint states in local coordinates, `x(τ) = Λ(τ) x_k + Ψ(τ) x_{k+1} + const`, then for any SPD query metric `W` (use `W = D_k`, the precision block of the nearer support state, so that tube widths are comparable to node-level errors):

```
E_τ = ‖W^{1/2} Λ(τ) D_k^{−1/2}‖₂ · E_k + ‖W^{1/2} Ψ(τ) D_{k+1}^{−1/2}‖₂ · E_{k+1}
Cov(xbar(τ) − x*(τ)) = [Λ Ψ] Σ_{(k,k+1)} [Λ Ψ]ᵀ + Q_τ
```

with `Σ_{(k,k+1)}` the joint marginal of the endpoint pair (bounded below/above by treating the pair as one block in §4) and `Q_τ` the conditional process covariance of the interpolant (from the GP prior). Radii as in §4.3 with `d = dim x(τ)`. You must first confirm the affine form from the code (Phase A, fact 3) and extract `Λ(τ), Ψ(τ)` exactly as the code computes them.

### 8.5 Global lower bound on remaining error energy (Proposition 5.7, proved)

```
‖e‖_H² = gᵀ H⁻¹ g ≥ (gᵀg)² / (gᵀ H g)     (and the ℓ-step Gauss rule from Lanczos on g gives larger valid lower bounds)
```

Log the 1-step and 5-step values and the oracle `gᵀH⁻¹g` each iteration. This is a global sanity monitor only.

---

## 9. Work plan

Do the phases in order except where marked parallel. Each phase ends with a report in the format of §11 and a git commit. Do not start a phase whose gate has not been passed. **Gates A and C are decided by the human**: after Phases A and C, write the report, commit, and stop until told to continue. Gates B, D and E you may pass yourself if every stated threshold is met; if any is not met, stop and report.

### Phase 0 — Environment check (first, before anything else)

Before touching the theory, establish and report the state of the environment. Do not assume anything about it.

1. Does the repository build in this environment? Run the build. Report the language, build system, compiler, every dependency that had to be installed, and the build time. If the build fails, report the failure verbatim and what would be needed.
2. Are the datasets present? Locate the data the ChArUco incremental SLAM, ChArUco BA, stereo BA (KITTI or whichever sequence the repo uses), multi-camera and rolling-shutter experiments expect. For each: present or absent, path, size, and if absent where the repo says to obtain it and whether you have network access to fetch it.
3. Do the experiments run? Run each of the above once with default settings. Report the printed metrics, wall time, and peak memory. Compare with the values reported in the paper or README.
4. Hardware: CPU cores, RAM, GPU (if any), disk. Estimate from the run times how long the sweeps in Phases C–D will take and flag anything that would take more than a few hours.
5. Report in the §11 format. **Gate 0:** builds, datasets available, at least ChArUco incremental SLAM and stereo BA run end to end. If the datasets are absent and cannot be fetched, stop and report; the human will supply them.

### Phase A — Trustworthy export (repo)

1. Build G-solver. Run the ChArUco incremental SLAM and the stereo BA experiments with default settings. Record the printed accuracy metrics and compare with the paper/README. Record the language, build system, run time.
2. **Read off four facts from the code, with file:line references and quoted code:**
   - (F1) Relinearization: are all factors relinearized together at fixed times, or independently? When exactly is a factor's Jacobian recomputed?
   - (F2) Gauge: how is the global gauge fixed? How in the distributed/multi-robot mode?
   - (F3) Interpolation: the exact formula for the state at query time `τ` between support states. Is it affine in the two endpoints in local coordinates? Extract `Λ(τ), Ψ(τ)`.
   - (F4) Messages: on a chain segment, does a node's stored message from its later neighbour equal the Schur complement of the future sub-chain onto that node (the `G_k` of §3)? Argue from the message update code.
3. Add an export option that, at chosen GBP iterations, writes: sparse `H`; `b`; the block partition (row ranges per variable, variable type, dimension, pose index for poses, observation pose indices per landmark); the factor list (id, type, variable ids, `J_f` blocks, `Λ_f`, `r_f`); GP prior parameters (`Q_C`, all `Δt`); the gauge term `G, b_G`; the current state `x` (tangent coordinates in the same chart as `H`). Export at iterations `1, 2, 5, 10, 20, 50, 100` and at convergence, with **relinearization frozen** at a fixed point (say how you froze it). Format: `.npz` for matrices + `.json` for structure. Do this for ChArUco incremental SLAM (several snapshots along the run: at 25%, 50%, 100% of the sequence) and stereo BA.
4. Also export **unfrozen** runs (normal relinearization) with the same fields at the same iterations, so Phase D can use them.
5. Validation: for each frozen export, assert `H` symmetric, Cholesky succeeds (report `λ_min` via `eigsh`), solve `H xbar = b` directly, compare with G-solver's converged state: report `max_i ‖x_i^{GBP,∞} − xbar_i‖_{D_i}`. **Gate A:** this is at machine-precision level (≤ 1e-6 in the D-metric). If not, explain why before proceeding (likely: chart mismatch, GBP not converged, or relinearization not frozen).

### Phase B — Standalone module + toy suite (parallel with A; no repo needed)

Implement `chain.py`, `covbounds.py`, `certs.py`, `hubs.py`, `radii.py` against a generic input (`H` sparse, `b`, block partition, factor list, GP chain description). Synthetic generators:

- **Pure WNOA chain**: `n` poses, `Q_C` and `Δt` as in G-solver, anchor prior on pose 0, no measurements; and the same with per-pose measurement information of strength `m` (isotropic `m·I` added to each diagonal). Sweep `n ∈ {20, 50, 100, 200, 500}` and `m/p ∈ {0, 0.01, 0.1, 1, 10}` where `p` is the prior link precision scale.
- **Chain + one hub**: as above plus one 6-dof landmark observed by every pose with a 6×6 information block.
- **Chain + short-track landmarks**: 3-dof landmarks each observed by 2–4 consecutive poses.
- **Random SPD block matrices** with random sparsity.
- **Chain with a blind segment**: measurements on all poses except a run of 20 in the middle.

For each, generate estimates `x` with realistic residual structure by running a few block-Jacobi or block-Gauss–Seidel iterations on `H x = b` from a random perturbation of `xbar` (no GBP implementation is needed; on a chain block-Jacobi has the same error-propagation structure as synchronous GBP). Keep iterates at several stages of convergence. Compute the oracle `xbar` directly, and run every unit test listed in §3–§6 on every iterate.

**Gate B:** all invariants pass on all synthetic problems; C2 is exact on the pure chain; C1 reproduces from C2 with `Pbb = D`; `rho(M)` on the pure chain follows `cos(π/(n+1))` as `m/p → 0` (report the curve; if the 12-dof block version deviates, report the actual curve — that is a finding, not a failure).

### Phase C — Tightness on real graphs

On every frozen export from Phase A: build supernodes and the hub set (§2.4, log `w` and `|S|`); compute all four certificates at `K ∈ {1,2,3}` (C4 and the hub correction only where structure allows), all lower/upper covariance bounds, radii with `δ = 0.05`, oracle `Σ_ii` and `e_i`; log everything in the schema of §10.

**Gate C (the decision gate):** for each dataset and certificate, report median and 95th-percentile `ratio`, the local/leak/hub shares, `rho(M)`, `rho(Mhat)`, `rho(A)`, `psibar` distribution, `theta_S`, fraction of patches locally certifiable, `mu_lo` from each subspace versus oracle `λ_min`, `mu_hi` from each upper bound versus oracle `λ_max`. Then apply §7 to name the loosest term. Targets: median ratio ≤ 3 on stereo BA and on ChArUco chain interiors for C3/C4; ≤ 10 on ChArUco overall with the hub term; `mu_lo ≥ 0.5 · λ_min` at `K = 3` or `ℓ = 5`; `mu_hi ≤ 4 · λ_max` at `K = 3` for ≥ 90% of poses. Do not proceed to Phase D with a certificate whose ratio is > 30 without reporting first.

### Phase D — Stopping and sleep/wake (repo)

1. Choose the certificate with the best Gate-C ratio among those computable locally (C3 or C4+hub expected). Implement it as a **scheduling layer** over G-solver's GBP with the message equations unchanged. Specifics:
   - **Where the certificate is computed.** Two acceptable designs; state which you use. (a) In C++ inside the solver: after each synchronous iteration, each node assembles its `K`-hop patch from its neighbours' current factors and beliefs and performs a small dense solve. (b) A Python **sidecar**: the solver exports the current assembled `H, b, x` each iteration (the Phase A hook, unfrozen), the sidecar computes every node's certificate and writes back a boolean active-set mask that the solver reads before the next iteration. Design (b) is acceptable for measuring update savings and is faster to build; wall-time comparisons must then exclude the sidecar and count node updates and message computations instead. Prefer (b) first; move to (a) only if time permits.
   - **What a sleeping node does** in G-solver's synchronous schedule: it does not recompute its belief, does not recompute its outgoing messages, and re-sends (or leaves in place) its last messages unchanged. Its incoming messages continue to arrive so that `Δg_p` in §8.2 can be accumulated. Factors touching only sleeping nodes are not evaluated.
   - **Live relinearization.** Phase D runs with G-solver's normal relinearization, not frozen. Each relinearization changes the model; certificates are recomputed for the new model (they are cheap; §5.2 of the proof note says per-model validity is automatic). The running bound of §8.2 is reset when a node's patch is relinearized, since `H_PP` changed; the node recomputes its exact patch solve at that point.
   - **Hub broadcast.** If a hub is present, the hub node computes `E_S` each iteration from its observers' `chat_j` and broadcasts one scalar; each observer adds `phibar_iS · E_S`.
   - `K` is fixed at the smallest value that passed Gate C. `S_lo_i` uses the trivial and patch lower bounds (both valid; take the larger).
2. Baselines, parameters fixed before seeing results: (i) G-solver's default fixed iteration count; (ii) **message-change threshold**: a node sleeps when the change of its belief mean between iterations, in the `D_i` metric, is below `τ_msg`, chosen as the value that gives the same *median* final error as the default schedule on the ChArUco run (state the value); (iii) residual threshold: sleep when `‖g_i‖_{D_i⁻¹} < τ_res`, chosen the same way.
3. Metrics per schedule: total node updates, total message computations, wall time, final error versus oracle `xbar` (per block, in `D_i`), ATE/RPE versus ground truth where available, fraction of active nodes per iteration, and for the certified schedule the fraction of nodes that ever woke after sleeping.
4. **Scaling:** run the ChArUco sequence truncated to 25/50/75/100% of its length (and, if available, the longer stereo sequence) and plot active-node count per iteration versus number of nodes.
5. **Loop closure:** let all nodes sleep at convergence, then add one long-range factor (a synthetic pose-pose constraint between two far-apart poses with realistic information, or a re-observation of the board from a late frame). Record the set of nodes that wake, their graph distance from the new factor, the recertification cost, and the final error versus a full re-solve.
6. **Early-stop failure case:** search the exported runs for any block where the message-change or residual baseline sleeps while the oracle error is still above `S_lo_i`; report such cases with numbers. This is the case where a heuristic stops too early and the certificate does not.

**Gate D:** certified schedule achieves ≥ 2× fewer node updates than the default at equal or lower final error; active set flat (or clearly sublinear) in trajectory length; wake set after a loop closure bounded and decaying with distance.

### Phase E — Continuous-time tubes (repo + module)

On the rolling-shutter stereo experiment (and any asynchronous multi-camera run): for a sample of row times `τ`, compute the interpolated pose, `E_τ`, and `S_hi_τ` from §8.4 with `W = D_k`; compare with ground truth at `τ`. **Ground truth at row times may not exist.** Check first. If ground truth is only available at frame times, use one of these, and say which: (i) evaluate coverage only at times where ground truth exists (frame times are legitimate query times too, since they are generally not support-state times); (ii) interpolate ground truth between its samples with a cubic spline and flag the interpolation error as a caveat; (iii) on synthetic data, where the true continuous trajectory is known, evaluate at arbitrary `τ`. Do (i) and (iii) at minimum. Report empirical coverage of `‖x(τ) − x*(τ)‖_W ≤ E_τ + S_hi_τ` (target ≥ 0.95 at `δ = 0.05`) and the ratio of tube half-width to actual error (median, 95th). Produce a figure of the tube along a segment of trajectory.

### Phase F — Figures and results summary

Produce: (1) `ratio` versus trajectory length for C1 (blowing up) and C3/C4 (flat); (2) scatter of certificate versus true error per block, all datasets; (3) active-node count versus iteration for the four schedules, and versus trajectory length; (4) loop-closure wake map (nodes coloured by whether they woke, with the new factor marked); (5) the confidence tube on the rolling-shutter segment; (6) certified `mu_lo`/`mu_hi` versus oracle eigenvalues. Write `RESULTS.md` following §11 with every number needed for a paper.

---

## 10. Logging schema

One CSV `certs.csv` with columns:

```
dataset, snapshot, frozen(bool), iter, block_id, block_type, dim, K,
err_true,                # ‖e_i‖_{D_i}, oracle
E_C1, E_C2, E_C3, E_C4, E_hub,      # NaN where not applicable
C3_local, C3_leak, C4_local, C4_leak, hub_term, hub_term_oracle,
ratio_C1, ratio_C2, ratio_C3, ratio_C4, ratio_hub,
mu_lo_trivial, mu_lo_patch, mu_lo_krylov3, mu_lo_krylov5, lam_min_oracle,
mu_hi_cut, mu_hi_hubaware, lam_max_oracle, locally_certifiable(bool),
S_lo, S_hi, stopped(bool)
```

One CSV `graph.csv` per (dataset, snapshot): `n_blocks, n_factors, w, n_hubs, rho_M, rho_Mhat, rho_A_K1, rho_A_K2, rho_A_K3, theta_S, theta_S_oracle, psibar_quantiles, global_lb_1step, global_lb_5step, global_energy_oracle`.

One CSV `schedule.csv` per Phase-D run: `schedule, dataset, length_fraction, iter, n_active, n_updates_cum, n_messages_cum, wall_time, max_err_true, median_err_true, ate, rpe`.

---

## 11. Report format (every phase, every commit)

1. **Five-line summary.** What was done, what passed, what failed.
2. **Gate status.** Pass/fail against the gate's stated thresholds, with the numbers.
3. **Tables.** The medians and 95th percentiles requested in the phase; per dataset; per certificate.
4. **Worst cases.** The single worst block for every invariant and every ratio, with its numbers and where it sits in the graph.
5. **Facts and code references** (Phase A only).
6. **What could not be done and why.** Plain statements. No workarounds presented as successes.
7. **File list.** CSVs, figures, export directories, commit hash.

---

## 12. Decision logic you are authorized to apply

- If Gate A fails because GBP did not converge to the direct solve under frozen linearization, diagnose the chart mismatch first (tangent coordinates of `H` versus the state export), then convergence, and report; do not proceed.
- If Gate B finds an invariant violation, the code is wrong. Do not proceed to real data until it passes.
- If, at Gate C, C1 is vacuous (`rho(M) ≥ 1` or ratio > 100) while C3/C4 are tight, that confirms the project's motivation: record it prominently.
- If at Gate C the hub term dominates on ChArUco and the oracle energy aggregate is much smaller than `E_S`, report it as the identified fix and continue Phase D with the chain certificate on stereo BA in the meantime.
- If at Gate C `psibar ≈ 1` on long segments, report the segment statistics and rerun with `K = 4, 5` on those datasets before declaring the bound loose.
- Anything else that is ambiguous: stop and report rather than guess.

---

## 13. Facts you may assume

All theorems referenced above (Lemmas 1.2–1.6, 2.1, 2.2, 2.6, 4.1, 4.3, 4.4, 4.5, 5.3; Theorems 2.4, 3.1, 3.2, 3.5, 4.2, 4.6, 5.1, 5.2, 5.4, 5.5, 5.6; Propositions 2.7, 3.6, 3.7, 4.7, 5.7) are proved. The full proof note is the file `PROOF_NOTE.md` in the repository root; consult it whenever a formula here needs its hypothesis or its proof, and cite its theorem numbers in code comments. If an invariant fails, the implementation is at fault. If a bound is valid but loose, that is a measurement, not a failure; §7 tells you how to attribute it.

---

## 14. Deliverables checklist

At the end, the repository branch `certify` must contain:

- `PROOF_NOTE.md` (supplied) and this brief (`AGENT_BRIEF.md`).
- `certify/` Python package: `chain.py`, `covbounds.py`, `certs.py`, `hubs.py`, `radii.py`, `io.py` (export loader), `synth.py` (toy generators), `tests/` (every unit test named in §3–§6, runnable with `pytest`).
- The C++ export hook and, for Phase D, the scheduling layer or sidecar, behind command-line flags, default off.
- `exports/` directory (or a documented download location if large): frozen and unfrozen exports for each dataset and snapshot.
- `results/certs.csv`, `results/graph.csv`, `results/schedule.csv`, `results/tubes.csv`.
- `results/figures/` with the six figures of Phase F.
- `reports/phase0.md` … `reports/phaseF.md`, each in the §11 format, and `RESULTS.md` collecting every number.

What is **not** asked of you: the paper text, related-work positioning, or any change to the theory. If you believe a theorem is wrong, do not fix it; report the counterexample with numbers.
