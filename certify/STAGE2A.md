# Stage 2a: the final-answer check at the certified stops

Brief: `BRIEF_stage2.md` §2. This is a necessary condition for Theorem T1 (`PROOF_NOTE_PART_II.md` §II.2): the residual term s_i of T1 at a global candidate, without curvature constants.

## The stop states

The Stage 1 runs were repeated to their certified-stop round with identical settings (`TAG=stage2a scripts/stage1_run.sh`).
- Every evaluation up to the stop reproduces Stage 1 exactly.
- At the first all-pass evaluation, the evaluator (`scripts/stage1_evaluator.py`) persists:
  - the linearization, as `certified_stop_model` (frozen charts);
  - the beliefs;
  - every pose's final and base patch, with E, L and S_lo (`certified_stop_patches.json`).
- The patch factorizations were not persisted; they are **rebuilt from the persisted patches**. That gives the same Cholesky factors of S_PP, the same [S_PP⁻¹]_{:,i} and the same leaks. Recomputing the certificate from them reproduces the logged E and L to ≤ 7e-16.

## The check (`scripts/stage2a_probe.py`)

1. **Global candidate z₁.**
   - Each pose p takes the step of its own patch: [−S_PP⁻¹ g_P]_p.
   - Each landmark takes its back-substituted step in the full D2 model, −H_ll⁻¹(g_l + H_lP ΔP), from the patch of the observing pose whose patch is largest (ties: the earlier pose).
2. **Cap.** Each landmark step is clipped to a Euclidean ball of radius r_L ∈ {0.1 m, 1 m, ∞}. The step is taken from the current state, not from the linearization point, which is at most δ = 3 mm away.
3. **Residual.** F(z₁) = Σ_f J_fᵀΛ_f e_f at x_lin ⊞ z₁, over all poses and landmarks.
   - Every factor is re-evaluated in C++ by its own `updateErrorAndJacobian`, with the variant's Jacobians (`certify_stereo --probe-export … [--true-jacobians]`, `ProbeServer`).
   - The linear D2 pseudo-factors (landmark priors, frozen single-edge messages) are added in Python.
4. **Residual term.** s_i^NL = ‖[S_{P_iP_i}⁻¹(F_{P_i} − H_{P_i L} H_LL⁻¹ F_L)]_i‖_{D_i}, where L is the landmarks observed by P_i. This is T1's s_i for the full-model patch P_i ∪ L with H₀ = the current linearization's H.
5. **Ratio.** r_i = s_i^NL / (ηS_lo,i), with η = 0.1.

## Attribution

Norm shares of s^NL, computed with the same reduction; they are not additive:
- the stereo factors' nonlinear part F_nl − F_lin;
- the GP factors' nonlinear part;
- the linear residue F_lin(z₁). This is nonzero because the global candidate mixes patches, and because clipping moves landmarks off their linear solution.

## Checks per stop state

- The export loads bitwise: J and e are recomputed at x_lin, with the variant's Jacobians.
- F_nl = F_lin exactly at z = 0.
- The full model's exact reduction agrees with the sidecar's S (≤ 4e-14) and g (≤ 1.1e-11).
- The rebuilt certificate reproduces the logged E and L.
