# Certified Stop-or-Relinearize for Gaussian Belief Propagation

Repository copy, 2026-09-27, of §1–§3 of the working document of the same name (the paper-planning §4–§5 are omitted; they are not needed by the agent). Builds on PROOF_NOTE.md Part I and PROOF_NOTE_PART_II.md. Every theorem below is proved; the one marked conditional depends on Theorem T1 of Part II.

## 1. Setting

### 1.1 Two kinds of relinearizing GBP

A nonlinear factor graph is solved by GBP in an outer–inner loop: linearize every factor at a linearization point, pass messages on the resulting linear Gaussian system, update beliefs, relinearize. Solver families differ in when they relinearize.

- **Threshold relinearization.** A factor's linearization point is updated only when a connected belief has moved more than a set distance δ; many message rounds are spent on one linear system; stopping is a fixed round budget. (Ortiz et al., CVPR 2020, use β = 0.01 with at most one relinearization per 10 rounds; iSAM2 uses β = 0.1.)
- **Per-round relinearization (G-solver as shipped).** Every factor relinearizes at every iteration, at its incoming message means (cavity means), so each linear system is held for one round and the factors of one variable sit at different linearization points.

The certificate of Part I applies to the linear system currently being solved. Under threshold relinearization with **one linearization point per variable** (the frozen-mode bookkeeping of Phase A, re-frozen per variable when it moves more than δ), that system is held for many rounds and the certificate applies directly. This is the form WP1 implements.

### 1.2 Objects

For one linearization: linearization points x_lin,i (one per variable), the assembled SPD system H x = b in the tangent charts at x_lin, its exact solution x̄ = H⁻¹b, and the current iterate x^(t), all expressed as tangent coordinates relative to x_lin. Blocks i ∈ V with dimension d_i; D_i = H_ii; ‖v‖_{D_i} = (vᵀD_iv)^{1/2}. If the solver's belief-movement metric W_i differs from D_i, the conversion factor is κ_i = ‖W_i^{1/2}D_i^{−1/2}‖, so ‖v‖_{W_i} ≤ κ_i‖v‖_{D_i}; with W_i = D_i (the WP0 decision), κ_i = 1.

Three quantities per variable per round:

- **Displacement since linearization** d_i^(t) = ‖x_i^(t)‖_{W_i} (the iterate is a tangent vector at x_lin,i, so this is ‖x_i^(t) ⊟ x_lin,i‖).
- **Certified remaining move** E_i^(t) ≥ ‖x_i^(t) − x̄_i‖_{D_i} (Theorem A) and its certified lower bound L_i^(t).
- **Statistical yardstick** S_lo,i(δ_s) from Theorem T0 with the true Jacobian (Part II), since G-solver's stereo Jacobian is a modification of the true one (WP0).

### 1.3 What is certified, and against what

Everything in §2–§3 is certified against the exact solution of the current linearization. The composition with the nonlinear fixed point is Proposition D in §3, conditional on Theorem T1. Model validity is never certified, only monitored.

### 1.4 Standing assumptions

(A0) H ≻ 0 on the current linearization (gauge fixed). (A1) The solver keeps iterates and linearization points in consistent charts, so that x^(t), x̄ and 0 are comparable tangent vectors at x_lin; the frozen mode is the reference implementation of this bookkeeping. (A2) Patches P_i ∋ i are fixed when a certificate is evaluated (they may depend on the residual, as in decision-driven growth); the leak matrix A ≥ 0 satisfies ρ(A) < 1, verified by a Collatz–Wielandt witness (a positive vector w with Aw < w componentwise).

## 2. The linear certificate

### Theorem A (per-variable two-sided a-posteriori bound)

Let H ≻ 0 be block-sparse, b given, x̄ = H⁻¹b, and x any vector with residual g = Hx − b and error e = x − x̄, so He = g. For each block i fix a patch P_i ∋ i with exterior E_i and boundary ∂P_i (the patch nodes adjacent to E_i), and define

    s_i   = || [ H_{P_iP_i}^{-1} g_{P_i} ]_i ||_{D_i}
    a_ij  = || D_i^{1/2} · ( Σ_{b∈∂P_i} [H_{P_iP_i}^{-1}]_{ib} H_{bj} ) · D_j^{-1/2} ||_2   for j ∉ P_i,   a_ij = 0 for j ∈ P_i.

If A = (a_ij) satisfies ρ(A) < 1, then with c* = (I − A)⁻¹s,

    max(0, s_i − Σ_j a_ij c*_j)  ≤  ||e_i||_{D_i}  ≤  c*_i     for every i.

The upper bound is E_i := c*_i and the lower bound L_i := max(0, s_i − Σ_j a_ij c*_j). Both are computable from the patch and the neighbours' values of c*.

*Proof.* The P_i-rows of He = g give e_{P_i} = H_{P_iP_i}⁻¹(g_{P_i} − H_{P_iE_i}e_{E_i}); extracting block i,
e_i = [H_{P_iP_i}⁻¹g_{P_i}]_i − Σ_{b∈∂P_i}[H_{P_iP_i}⁻¹]_{ib} Σ_{j∈E_i} H_{bj} e_j.
Taking D_i-norms and the triangle inequality, u_i := ‖e_i‖_{D_i} satisfies u_i ≤ s_i + Σ_j a_ij u_j and u_i ≥ s_i − Σ_j a_ij u_j. From u ≤ s + Au with A ≥ 0 and ρ(A) < 1, multiply by (I − A)⁻¹ = Σ_m A^m ≥ 0 to get u ≤ c*. Substituting u_j ≤ c*_j in the reverse inequality gives the lower bound. ∎

*Computation.* c* is the limit of the monotone iteration c ← s + Ac from 0; ρ(A) < 1 is certified by a Collatz–Wielandt witness. Patches may be grown per node while its decision is open (decision-driven patches, Phase C parameters r = 0.3, θ = 0.02, cap 250, 40 steps); Theorem A holds for whatever patches are in force when it is evaluated. On SLAM graphs, apply it to the landmark-marginalized pose model (exact Schur elimination of landmarks, `certify.model.reduce_blocks`), which is where ρ(A) < 1 held in Phase C.

### Theorem B (certified marginal covariance from columns)

Let Σ = H⁻¹ and, for block i, W = H⁻¹E_i (the n × d_i block column), so Σ_ii = [W]_i and Σ_ji = [W]_j. For every block j define t_j = ‖D_j^{1/2}[H_{P_jP_j}⁻¹]_{ji}D_i^{1/2}‖ if i ∈ P_j and t_j = 0 otherwise. If ρ(A) < 1 with the same A as in Theorem A, then with v* = (I − A)⁻¹t,

    || D_j^{1/2} Σ_ji D_i^{1/2} ||  ≤  v*_j   for all j,
    [H_{P_iP_i}^{-1}]_ii  ⪯  Σ_ii  ⪯  [H_{P_iP_i}^{-1}]_ii + ε_i D_i^{-1},    ε_i = Σ_{j∉P_i} a_ij v*_j.

*Proof.* HW = E_i, so the P_j-rows give W_{P_j} = H_{P_jP_j}⁻¹(E_{P_j,i} − H_{P_jE_j}W_{E_j}), where E_{P_j,i} is nonzero only if i ∈ P_j. Extracting block j and taking the operator norm of D_j^{1/2}(·)D_i^{1/2}: v_j := ‖D_j^{1/2}Σ_{ji}D_i^{1/2}‖ ≤ t_j + Σ_k a_jk v_k, the inequality of Theorem A with s replaced by t (the argument is linear, so it holds for matrix blocks with the operator norm). Hence v ≤ v*. The lower bound of the sandwich is Lemma 2.1 of Part I (conditioning on the exterior). For the upper bound take j = i: Σ_ii = [H_{P_iP_i}⁻¹]_ii − Δ with Δ = Σ_{b∈∂P_i}[H_{P_iP_i}⁻¹]_{ib} Σ_{k∈E_i} H_{bk}Σ_{ki}; then ‖D_i^{1/2}ΔD_i^{1/2}‖ ≤ Σ_k a_ik v*_k = ε_i, and a symmetric matrix of D_i-weighted norm at most ε_i is ⪯ ε_iD_i⁻¹ in the Loewner order. ∎

### Corollary (certified anytime stopping)

With S_lo,i(δ_s) from Theorem T0 (true-Jacobian Fisher information, patch or Krylov subspace of Lemma 2.1), the rule E_i ≤ η S_lo,i(δ_s) is valid at every iterate simultaneously within one linearization (Theorem 5.2(a) of Part I) and means: completing the linear solve cannot move block i by more than an η-fraction of a lower bound on the (1 − δ_s) radius of any unbiased estimator's error. The ratio E_i / L_i reports the certificate's own tightness without an oracle.

## 3. The certified stop-or-relinearize rule

### Theorem C (certified stop-or-relinearize)

Under (A0)–(A2), for variable i at round t let d_i = ‖x_i^(t)‖_{W_i} be the displacement since linearization, and let E_i^W = κ_i E_i and L_i^W = L_i / κ'_i be the certificate of Theorem A converted to the solver's metric (κ'_i = ‖D_i^{1/2}W_i^{−1/2}‖; both equal 1 when W_i = D_i). Then:

1. **(Bracket.)** The converged belief of the current linearization satisfies
       max(0, d_i − E_i^W)  ≤  ||x̄_i||_{W_i}  ≤  d_i + E_i^W.
2. **(Certified stop.)** If d_i + E_i^W ≤ δ_i, the converged belief does not cross the relinearization threshold, so the solver's own rule would never relinearize variable i on this linearization. If in addition E_i ≤ η S_lo,i, variable i may stop: its remaining move is at most an η-fraction of its irreducible uncertainty, and the current linearization is final for it.
3. **(Certified relinearize-now.)** If d_i − E_i^W ≥ δ_i, the converged belief certainly crosses the threshold; every further message round spent on the current linearization for variable i is wasted, and relinearization may be triggered immediately.
4. **(Continue.)** Otherwise the decision is open; the width of the open interval is 2E_i^W, so growing the patch (which shrinks E_i) closes it.

*Proof.* All three vectors x_i^(t), x̄_i and 0 are tangent vectors at x_lin,i by (A1). By the triangle inequality, ‖x̄_i‖_{W_i} ≤ ‖x_i^(t)‖_{W_i} + ‖x_i^(t) − x̄_i‖_{W_i} ≤ d_i + κ_iE_i, and ‖x̄_i‖_{W_i} ≥ d_i − κ_iE_i. That is (1); (2)–(4) read off the solver's threshold rule from the bracket, and the stopping clause is the Corollary of §2. ∎

### Sleep semantics and wake

A stopped variable freezes its state and stops sending updated messages. Its certificate stays valid under the cheap recheck of Theorem 5.4 of Part I: with ℓ_ip = ‖D_i^{1/2}[H_PP⁻¹]_ip D_p^{1/2}‖ computed once, s_i can increase by at most Σ_{p∈P} ℓ_ip ‖Δg_p‖_{D_p⁻¹} when the patch residual changes by Δg_P, and Δg_p = Σ_j H_pj Δx_j is computable from the neighbours' moves. The running bound s_i(t₀) + Σ_p ℓ_ip‖Δg_p‖ + Σ_j a_ij c_j(t) is a valid certificate at every t. The variable wakes when it exceeds η S_lo,i, or when any factor in its patch is relinearized (H_PP changes) or added (new information). Theorem 5.5 of Part I bounds the wake wave after a new factor: with r the maximum row sum of A, the change of c* at recursion distance m from the affected patches is at most r^m ‖Δs‖_∞ / (1 − r).

### Amortized cost

Within one linearization H is fixed, so each node factorizes its patch once and afterwards pays one back-solve per round for s_i, plus the scalar recursion. Relinearization of any factor in the patch triggers one refactorization.

### Proposition D (δ-consistent fixed point; composition with T1) [conditional on T1]

Suppose every variable has stopped under Theorem C(2), with state x^s, linearization points x_lin and the exact linear solution x̄. Call the pair (x_lin, x̄) **δ-consistent**: ‖x̄_i‖_{W_i} ≤ δ_i for all i. Then:

1. **(What the scheduling route certifies.)** ‖x^s_i − x̄_i‖_{D_i} ≤ E_i for all i, and no relinearization is pending under the solver's rule. The state is within E of the exact solution of a linearization that the solver itself would not change. δ remains a design parameter.
2. **(What T1 adds.)** Apply Theorem T1 (Part II) with candidate step z₁ = x̄ (or x^s) in the chart at x_lin and box radius ū of order δ. If the box verifies, ‖x^s_i ⊟ x̂_i‖ ≤ E_i + u⁺_i, where x̂ is the zero of the solver's nonlinear estimating equation F and u⁺ = lim Ψ^k(ū).

**Correction (verified literature check, 2026-09-27).** For Newton-type updates or zero-residual problems, u⁺ = O(ωδ²). G-solver is Gauss–Newton with nonzero residuals, so at a δ-consistent state the nonlinear residual is F(x̄) = (J̃(x̄) − J̃(x_lin))ᵀΛ(r_lin + J δx) + O(δ²) = O(L‖r‖δ) + O(δ²): a first-order term proportional to the residual appears, with coefficient κ_GN ≈ ‖H⁻¹S‖, S = Σ_k r_k∇²r_k, the classical Gauss–Newton incompatibility factor. The correct statement is u⁺ = O(κ_GN δ + ωδ²). Theorem T1 itself is unaffected: it computes s_i from the actual residual F(z₁), so the κ_GN δ contribution enters automatically and is certified; only the asymptotic claim changes. Citations: Kelley, Acta Numerica 27 (2018), Theorem 2.3; Deuflhard (2004/2011), Theorem 2.1 and §2.1.2; Dembo, Eisenstat & Steihaug, SIAM J. Numer. Anal. 19(2):400–408 (1982).

**Why this matters for the plan.** With threshold relinearization the drift between reference and current point is bounded by δ by construction, and the T1 box is not a free parameter but the solver's own δ. The residual term s_i of T1, evaluated at the exact local step, is what WP5a measures directly, without curvature constants.
