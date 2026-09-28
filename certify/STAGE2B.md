# Stage 2b: Theorem T1 at the certified stops

Brief: `BRIEF_stage2b.md`. Theory: `PROOF_NOTE_PART_II.md` §II.2.

## The equation

F(z) = Σ_f J_f(x)ᵀ Λ_f e_f(x) + Σ_pseudo Λ(z − a), at x = x_c ⊞ z. Its ingredients:
- **Model.** The full D2 model: every pose and kept landmark; the stereo factors that are inliers at x_c, and the GP factors; the landmark priors p_L and the frozen single-edge messages as linear pseudo-factors (`certify/t1.py`, `Problem`).
- **Chart.** Centred at the certified stop's beliefs x_c: T·Exp(τ) with Exp(t, r) = (Rodrigues(r), t); velocities and landmarks additive.
- **Jacobian.** J_f is the exact derivative of e_f along the local perturbation at x. This is the corrected solver's Jacobian in exact arithmetic.
  - The implemented corrected solver differs from it slightly. Its GP rotation block is accurate to about |φ|²/12 relative, and its stereo translation block assumes RᵀR = I, although its stored rotations are orthonormal only to about 1e-8.
  - `results/stage2b/<dataset>/evaluator_checks.json` measures the difference in F at x_c.
  - Step 1 measures the distance between the two equations' zeros.

## The evaluator (C++: `certify/cpp/t1_ad.h`, `t1_server.h`; `certify_stereo --t1-export`)

- **Forward-mode duals, nested.** The outer level differentiates in z, the inner level in the local perturbation δ at δ = 0. From e as a nested dual one reads e, J = ∂e/∂δ, ∂e/∂z and ∂²e/∂z∂δ, hence F_f = JᵀΛe and its exact derivative F′_f = ∂_z(JᵀΛe), with all second-order terms.
- **Intervals.** Every +, −, ×, ÷ is computed in round-to-nearest, and the result's bounds are moved outward by one ulp (`std::nextafter`). This contains the exact result of the operation on the endpoints. Division by an interval containing 0 sets a flag.
- **The only non-rational maps**, the SO(3) exponential and logarithm, are power series:
  - Rodrigues coefficients sin t/t and (1 − cos t)/t² in s = |r|², 12 terms, for |r| ≤ 2;
  - the principal logarithm φ = asin(√q)/√q · w with w = vee(R − Rᵀ)/2, 40 terms, for q ≤ 1/4 and tr R > 1.
  - The truncation remainder of the value and of its first two derivatives is enclosed by a geometric tail bound.
  - A margin of 1e-11 covers the rounding of the stored coefficients.
  - No libm function is used.
- **Inlier conditions.** A stereo factor whose inlier conditions (z > 0, disparity > 0, right-image column ≥ 0) are not guaranteed on the box is flagged "unbounded". There G-solver's F switches the factor off and is not smooth.
- **Requests:**
  - `eval`: double values, used by Newton and for H₀;
  - `enclose`: F at a point and F′ over a box, as interval enclosures.

## Step 1: the Newton reference (`scripts/stage2b_newton.py`)

- **Newton on F.** Direct solves on the full model: a dense Schur complement on the poses, with the landmark blocks eliminated exactly. It starts at z = 0 and stops when ‖dz‖ ≤ 1e-12‖z‖, or after 50 iterations.
- **The implemented equation's zero.** Simplified Newton on the solver's implemented equation (`--probe-export`, its own code) from x̂, which measures how far that zero is from x̂.
- **Metrics:**
  - poses: D_i = S_ii of the stop's linearization, the certificate's metric (Stage 1);
  - landmarks: D_l = the landmark's diagonal block of the Gauss–Newton matrix at x_c.

## Step 2: T1 (`scripts/stage2b_t1.py`)

- **Candidate.** z₁ is the Stage 2a global candidate (landmark steps clipped to 0.1 m), taken as a step in the x_c chart.
- **Iteration matrix.** H₀ = F′(z₁), the exact derivative, assembled in double.
- **Rows:**
  - pose i: patch Q_i = P_i (its Stage 1 final patch) ∪ every landmark P_i observes. s_i, a_ij and ℓ_ip come from H₀ by exact block elimination of the landmarks; the exterior is the poses outside P_i adjacent to Q_i.
  - landmark l: patch {l}, whose leaks go to its observing poses.
  - Landmarks have no leak to landmarks, and pose rows none to landmarks. So ρ(A) = ρ(A_PP), certified by a Collatz–Wielandt witness.
- **Δ_kq** = sup over the box of ‖D_k^{−1/2}(F′(ζ) − H₀)_kq D_q^{−1/2}‖. It is computed from the per-factor enclosures of F′_f over z₁ + B(ū), summed per block pair (the pseudo-factors are linear and exact in H₀). The box is enclosed in coordinates by |w_k,j| ≤ ū_k √[D_k⁻¹]_jj, inflated by 1e-10.
- **Ψ(u)** = (I − A)⁻¹(s + ℓ∘ρ(u)) with ρ_k(u) = Σ_q Δ_kq u_q. The check is Ψ(ū) ≤ ū.
- **Radii.** Poses: c·E_i. Landmarks:
  - option A: c · 0.1 · √λ_max(D_l);
  - option B: c · E_l, with E_l = ‖D_l^{−1/2} g_l‖ + Σ_p ‖D_l^{−1/2} H_lp D_p^{−1/2}‖ E_p at the stop's linearization.
  - c ∈ {1, 2, 4, 8, 16}.
- **Precision.** The check is evaluated in double precision, with rigorous interval enclosures for Δ.
  - A certificate would also need s, a and ℓ as rigorous upper bounds: verified solves of the patch systems with interval residuals, and a verified bound on the patch inverse.
  - That layer is only needed if the check verifies; no bound is claimed otherwise.
