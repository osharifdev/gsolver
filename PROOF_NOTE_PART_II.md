# Proof Note — Part II: The nonlinear certificate and the correct yardstick

Repository copy, 2026-09-27, of Part II of the working proof note. Part I is the existing `PROOF_NOTE.md` (§0–§7 with the 25 Sept errata). Every result here is proved; status tags: [verified] classical or elementary, checked; [new] proved here.

## II.0 Preamble and errata to Part I

Part I certifies distance to the exact solution of one frozen linear model. Phase A showed that G-solver relinearizes every factor on every iteration, so the frozen model is a transient object; Phase C showed that the exact-local-solve certificates (Lemma 3.3, Theorem 3.5) are tight on the landmark-marginalized pose model while every norm-product bound (Lemma 1.6, Proposition 3.6, Theorem 4.6) is vacuous on real graphs. Part II keeps what worked and extends it to the two objects a user cares about: the solver's nonlinear fixed point x̂ (a zero of its estimating equation) and the true state x*.

**Errata to Part I.** (E1) On the chain, G_k = (H_kk − Λ^f_kk) + message_{k+1→k}, where Λ^f_kk is the forward GP factor's diagonal contribution; the remark "P_kk + message" in §1.1 is wrong by that term; Lemma 1.4's seed is exactly H_mm − Λ^f_mm. (E2) G-solver's interpolation is affine in the GP local variables, so Theorem 5.6 applies to the linearized interpolant in a fixed chart. (E3) The classical coupling is ≈ 0.84 for G-solver's 12-dof link, not ½, so ρ(M) ≈ 1.6–1.7 even at n = 20: Proposition 3.7's conclusion holds more strongly than stated. (E4) Lemma 1.6 and everything built on it is valid but empirically vacuous (ψ̄ medians 9–51). (E5) The statistical statements of §2.3 and §5.1 assume the model matrix is the true Jacobian; on G-solver's stereo factors it is not (WP0: the rotation ⊞ is the exponential map and the stereo rotation Jacobian block carries a factor of 2), and Theorem T0 corrects them.

## II.1 Theorem T0 — the uncertainty yardstick for solvers with approximate Jacobians

**Setting.** Measurements y = h(x*) + ε, ε ~ N(0, Λ⁻¹). J = ∂h/∂x is the true whitened Jacobian in the solver's chart; J̃ is the Jacobian the solver uses (for G-solver's stereo factors: J with the three rotation columns of each pose block multiplied by 2; all other factor types have J̃ = J). The solver's estimate x̂ solves the estimating equation F_y(x̂) = J̃(x̂)ᵀΛ(h(x̂) − y) = 0. Write A = J̃ᵀΛJ, B = J̃ᵀΛJ̃, I_F = JᵀΛJ (Fisher information), all at x*; H = B is the matrix the solver reports as its precision.

**Theorem T0.**
1. *(Sandwich covariance.)* To first order in the noise, x̂ − x* = Kε with K = A⁻¹J̃ᵀΛ, so Cov(x̂) ≈ V := A⁻¹BA⁻ᵀ. The reported covariance H⁻¹ = B⁻¹ equals V only when J̃ = J.
2. *(Gauss–Markov / Cramér–Rao lower bound.)* V ⪰ I_F⁻¹. More generally every linear estimator Kε with KJ = I has covariance ⪰ I_F⁻¹.
3. *(Certified yardstick.)* For any matrix V_i with orthonormal columns whose span contains the coordinate directions E_i of block i (patch or Krylov subspace),
       V_ii  ⪰  [I_F⁻¹]_ii  ⪰  E_iᵀ V_i (V_iᵀ I_F V_i)⁻¹ V_iᵀ E_i  =: L_i,
   so S_lo,i(δ_s) = √λ_min(D_i^{1/2} L_i D_i^{1/2}) · q_{d_i}(δ_s), with q_d(δ) = √χ²_{d,1−δ}, is a valid lower bound on the (1−δ_s) radius of the solver's own error and of any unbiased linearized estimator's error.
4. *(Frozen-mode bias.)* In a frozen linear model built with J̃ at x₀, E[x̄] − x* = (B⁻¹A − I)(x* − x₀) up to second order: a biased estimator of x* unless J̃ = J.

*Proof.* (1) Implicit differentiation of F_y(x̂(y)) = 0 at y = h(x*): ∂_xF = J̃ᵀΛJ + Σ_k(∂J̃_kᵀ)Λr_k, and at the noise-free point every r_k vanishes, so ∂_xF = A; ∂_yF = −J̃ᵀΛ; hence dx̂ = A⁻¹J̃ᵀΛ dy and Cov(Kε) = KΛ⁻¹Kᵀ = V. (2) KJ = I. Let K₀ = I_F⁻¹JᵀΛ, with K₀J = I. Then (K − K₀)Λ⁻¹K₀ᵀ = (K − K₀)J I_F⁻¹ = 0, so KΛ⁻¹Kᵀ = K₀Λ⁻¹K₀ᵀ + (K − K₀)Λ⁻¹(K − K₀)ᵀ ⪰ I_F⁻¹. (3) The first inequality is (2) restricted to block i; the second is Lemma 2.1 of Part I applied to the SPD matrix I_F; the radius statement is Proposition 2.7 of Part I. (4) x̄ = x₀ + B⁻¹J̃ᵀΛ(y − h(x₀)) with y − h(x₀) = J(x* − x₀) + ε + O(‖x* − x₀‖²); take expectations. ∎

**What is classical.** Items 1–2: estimating-equation (M-estimator) sandwich and Gauss–Markov (Godambe 1960; Huber 1967; White 1982). **What is new.** The observation that a GBP SLAM solver with a modified Jacobian is an estimating-equation estimator whose reported H⁻¹ is not its covariance, and the use of the true-Jacobian Cramér–Rao bound as the certified stopping yardstick.

**Implementation.** J is obtained from the exported J̃ by halving the rotation columns of every stereo factor's pose block (`se3_posevel_stereo_point.cpp:77` without the `2.0`). I_F = JᵀΛJ is assembled like H; the patch subspace bound is the Phase C code applied to I_F instead of H.

## II.2 Theorem T1 — the localized nonlinear certificate (a-posteriori form)

### Setting

The live state is x_c. Work in the chart at x_c: x = x_c ⊞ z, z ∈ ℝⁿ with the block structure of the variables. The solver's estimating equation in this chart is

    F(z) = Σ_f J̃_f(x_c ⊞ z)ᵀ Λ_f r_f(x_c ⊞ z),

where J̃_f is the local-perturbation Jacobian the solver evaluates at x_c ⊞ z and the gauge term is a factor. A zero z* of F is a fixed point x̂ = x_c ⊞ z* of the relinearizing solver, up to its O(ε) regularizer bias. The quantity certified is ‖z*_i‖_{D_i}, the remaining move of block i, with D_i any fixed SPD block metric (the diagonal block of the solver's H at x_c).

### Ingredients (all local)

- **Candidate step z₁.** Any vector. Recommended: each node's exact patch step z₁,i = −[H₀,PP⁻¹F_P(0)]_i.
- **Block-invertible matrix H₀** with the graph's sparsity, factorized per patch. Admissible choices, in increasing tightness: the solver's H = J̃ᵀΛJ̃; the one-sided Gauss–Newton matrix J̃ᵀΛJ; the exact derivative F′(z₁). It may be evaluated at a stale reference (drift, below).
- **Residual at the stepped point** F(z₁): one evaluation of every factor at x_c ⊞ z₁.
- **Patches** P_i ∋ i, arbitrary; recommended the covisibility patch (poses within K hops in the reduced pose graph plus every landmark they observe).
- **Leak coefficients and patch weights** from H₀:
      a_ij = || D_i^{1/2} Σ_{b∈∂P_i}[H₀,P_iP_i⁻¹]_{ib}(H₀)_{bj} D_j^{-1/2} ||  (j ∉ P_i),   ℓ_ip = || D_i^{1/2}[H₀,P_iP_i⁻¹]_{ip} D_p^{1/2} ||  (p ∈ P_i).
- **Local residual term** s_i = ‖[H₀,P_iP_i⁻¹ F_{P_i}(z₁)]_i‖_{D_i}.
- **Factor-local curvature constants (H-L).** For a box B(ū) = {w : ‖w_k‖_{D_k} ≤ ū_k ∀k} around z₁, constants Δ_{f,pq} ≥ 0 and L_{f,pqk} ≥ 0 such that for every ζ with ζ − z₁ ∈ B(ū),
      || D_p^{-1/2}(F′_f(ζ) − H₀,f)_{pq} D_q^{-1/2} ||  ≤  Δ_{f,pq} + Σ_{k∈V_f} L_{f,pqk} ||ζ_k − z₁,k||_{D_k}.        (H-L)
  Δ is zero for H₀ = F′(z₁) fresh, proportional to the drift for a stale reference, proportional to the residual size for a Gauss–Newton H₀ (since F′ − J̃ᵀΛJ = Σ(∂J̃ᵀ)Λr). L is the Lipschitz constant of the factor's derivative on the box, analytic per factor type. Constants must be analytic upper bounds, never sampled.
- **Remainder majorant and the map Ψ:**
      ρ_p(u) = Σ_{f∋p} Σ_{q∈V_f} ( Δ_{f,pq} + ½ Σ_{k∈V_f} L_{f,pqk} u_k ) u_q,
      Ψ(u) = (I − A)⁻¹ ( s + ℓ∘ρ(u) ),   [ℓ∘ρ(u)]_i = Σ_{p∈P_i} ℓ_ip ρ_p(u).
  ρ is componentwise monotone and quadratic; Ψ is monotone.

### Statement

**Theorem T1.** Assume A ≥ 0 with ρ(A) < 1, (H-L) on B(ū), and Ψ(ū) ≤ ū componentwise. Let u⁺ be the limit of the decreasing iteration u^{k+1} = Ψ(u^k) from u⁰ = ū. Then:
1. *(Existence.)* F has a zero z* with ‖z*_i − z₁,i‖_{D_i} ≤ u⁺_i for every i; consequently
       | ||z₁,i||_{D_i} − u⁺_i |  ≤  ||z*_i||_{D_i}  ≤  ||z₁,i||_{D_i} + u⁺_i  =: E_i.
2. *(Localization.)* Every zero of F in z₁ + B(ū) lies in z₁ + B(u⁺).
3. *(Uniqueness.)* If in addition ρ((I − A)⁻¹ ℓ∘Λ(ū)) < 1, with Λ(ū) the componentwise Lipschitz matrix of ρ on [0, ū], the zero is unique in z₁ + B(ū) and the simplified-Newton iteration with matrix H₀ converges to it from z₁.

### Proof

For w ∈ ℝⁿ write ζ = z₁ + w. By the fundamental theorem of calculus, F(z₁ + w) = F(z₁) + H₀w + R₁(w) with R₁(w) = ∫₀¹ (F′(z₁ + tw) − H₀) dt · w. Define Φ(w) = −H₀⁻¹(F(z₁) + R₁(w)); then w = Φ(w) iff F(z₁ + w) = 0.

*Componentwise bound on Φ.* Let v = Φ(w), so H₀v = −F(z₁) − R₁(w). The rows of patch P = P_i give v_P = H₀,PP⁻¹(−F_P(z₁) − R₁,P(w) − H₀,PE v_E). Extract block i and take D_i-norms:
    ||v_i||_{D_i} ≤ s_i + || [H₀,PP⁻¹ R₁,P(w)]_i ||_{D_i} + Σ_{j∉P_i} a_ij ||v_j||_{D_j}.
For the middle term, ‖[H₀,PP⁻¹R₁,P]_i‖_{D_i} ≤ Σ_{p∈P} ℓ_ip ‖R₁,p(w)‖_{D_p⁻¹}, and R₁,p(w) = Σ_{f∋p} Σ_{q∈V_f} ∫₀¹ (F′_f(z₁ + tw) − H₀,f)_{pq} dt · w_q. Insert D_p^{−1/2}, D_q^{−1/2}, apply (H-L) with ζ_k − z₁,k = t w_k, and integrate (∫₀¹ t dt = ½): ‖R₁,p(w)‖_{D_p⁻¹} ≤ ρ_p(u(w)), where u(w) is the vector of block norms of w. Collecting, u(v) ≤ s + ℓ∘ρ(u(w)) + A u(v); since A ≥ 0 and ρ(A) < 1, (I − A)⁻¹ ≥ 0 and u(v) ≤ Ψ(u(w)).

*(1) Existence.* If u(w) ≤ ū then u(Φ(w)) ≤ Ψ(ū) ≤ ū: Φ maps the box B(ū), a compact convex set in a Euclidean chart, into itself, and Φ is continuous. Brouwer's theorem gives a fixed point w* ∈ B(ū), i.e. a zero z* = z₁ + w*. The two-sided bound is the triangle inequality plus ‖w*_i‖ ≤ u⁺_i from (2).

*(2) Localization.* If w is a fixed point in B(u^k), then u(w) = u(Φ(w)) ≤ Ψ(u^k) = u^{k+1}; induction from k = 0 gives u(w) ≤ u⁺. The iteration is decreasing because Ψ(ū) ≤ ū and Ψ is monotone; bounded below by 0, so convergent.

*(3) Uniqueness.* For w, w′ ∈ B(ū), Φ(w) − Φ(w′) solves H₀(Φ(w) − Φ(w′)) = −(R₁(w) − R₁(w′)); the same patch argument gives u(Φ(w) − Φ(w′)) ≤ (I − A)⁻¹ ℓ∘Λ(ū) u(w − w′). A nonnegative matrix with spectral radius below one is a contraction in the weighted max-norm defined by its Perron vector, so Φ is a contraction on the box and Banach's theorem applies. ∎

### Remarks

- **Tightness by construction.** The certificate is the exact local step plus a remainder driven by the residual after the step (second order in the step) and by ρ (quadratic in the deviation). This is the a-posteriori form of Kantorovich's theorem, the device of verified Newton solvers (Krawczyk; Rump), localized to each node's row.
- **Two-sided.** ‖z₁,i‖ − u⁺_i is a certified lower bound on the remaining move.
- **Drift and reuse.** With H₀ from a stale reference, Δ grows with the drift of the factor's variables (and, for SE(3) charts, with the chart change, bounded via ‖J_r(w) − I‖ ≤ ½‖w‖). Under threshold relinearization the drift is at most δ by construction.
- **Adaptive box.** Ψ(ū) ≤ ū is checked, not assumed: try ū = c·(linear certificate) for c ∈ {2, 4, 8, …} and keep the smallest that verifies.
- **Special cases.** L = Δ = 0, z₁ = 0: exactly certificate C3 of Part I. z₁ = patch step, L = Δ = 0: C3 applied to the residual after one exact local step.
- **Floating point.** The certificate is in exact arithmetic unless rounding terms are added.
- **Gauss–Newton residual term.** At a δ-consistent state, s_i = O(κ_GN δ + ωδ²) (see the correction in STOP_OR_RELINEARIZE.md, Proposition D); T1 certifies it automatically because s_i is computed from the actual residual F(z₁).

### Novelty boundary (verified literature checks of 2026-09-26/27)

Classical, to cite: Kantorovich (1948) and vector-majorant forms (Ortega–Rheinboldt 1970); a-posteriori verified Newton (Krawczyk 1969; Rump, Acta Numerica 2010, INTLAB verifynlss; Yamamoto/Potra); affine-covariant Newton–Kantorovich and simplified Newton (Deuflhard 2004/2011, Theorem 2.1, §2.1.2); chord-method error estimate with approximate Jacobians (Kelley, Acta Numerica 27, 2018, Theorem 2.3); inexact Newton (Dembo–Eisenstat–Steihaug 1982); Schur-complement existence proofs for PDEs (Sekine–Nakao–Oishi 2020); a-posteriori stopping for inexact Newton in PDEs (Ern–Vohralík 2013). New: the per-variable localization through the patch identity and leak recursion; the drift term and reuse rule; covisibility patches inside a nonlinear certificate; the two-sided remaining-move bound; application to a live relinearizing GBP solver. Existing relinearization rules (iSAM2 β = 0.1, GTSAM default 0.1, Ortiz et al. β = 0.01) are empirically tuned thresholds without an error certificate; no prior work certifies per-variable relinearization or inner-loop stopping in factor-graph or GBP solvers.
