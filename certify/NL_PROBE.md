# The nonlinear residual probe (Phase D, WP6b)

WP6b measures the residual term of Theorem T1 (`PROOF_NOTE_PART_II.md`) without curvature constants. The question is how far the solver's nonlinear estimating equation F is from zero after the exact local step of the current linearization.

## Definition (per pose i, at one round)

1. **Patch and step.** P is the patch the certificate used for pose i at this round (sidecar `Engine`: the decision-driven patch for eligible poses, otherwise the base patch).
   - On the reduced pose model, dP = −S_PP⁻¹ g_P, with the same Cholesky factor as the certificate, and g = S x − b at the current state.
   - Every eliminated block l observed by P (landmarks; ChArUco markers) takes its back-substituted step dl = −H_ll⁻¹(g_l + H_lP dP). This is in the full model of decision D2: the poses and the eliminated blocks, with the landmark priors p_L and the frozen single-edge messages as pseudo-factors.
   - The linear model's gradient then vanishes on P and on those blocks. The **targets** are P ∪ L(P); z₁ = z + d on the targets, and every other variable keeps its current state z.
2. **Residual.** F(z₁) = Σ_f J̃_f(x)ᵀ Λ_f e_f(x) at x = x_lin ⊞ z₁, on the targets.
   - Every factor touching a target is re-evaluated by its own `updateErrorAndJacobian` in C++, the code the solver relinearizes with.
   - The linear pseudo-factors are added in Python.
3. **s_i^NL** = ‖[S_PP⁻¹ F̃_P]_i‖_{D_i}, with F̃_P = F_P − H_PL H_LL⁻¹ F_L: the next local step of the same patch (T1's s_i with H₀ = the solver's H).

## Decomposition

Per factor class, the C++ side returns four sums over the factors touching each target:

| | Formula |
|---|---|
| F_nl | J̃(x)ᵀ Λ e(x) |
| F_lin | J̃_linᵀ Λ (e_lin + J̃_lin τ) |
| F_jac | J̃(x)ᵀ Λ (e_lin + J̃_lin τ) |
| F_true | J̃(x)ᵀ Λ (e_lin + J_lin τ) |

Here τ is the offset from the linearization points, and J is the true Jacobian (the stereo rotation columns halved, Theorem T0). This gives:

- **F(z₁) = Σ_c (F_nl − F_lin)[c] + F_lin(z₁).** F_lin(z₁) vanishes on the targets up to rounding, which is checked on every probe as `s_lin_residue`.
- **F_nl − F_lin = (F_jac − F_lin) + (F_true − F_jac) + (F_nl − F_true):**
  - the Gauss–Newton Jacobian change, first order times the residual;
  - the model mismatch J̃(x)ᵀΛ(J − J̃)τ, first order and nonzero only where J̃ ≠ J;
  - the error's curvature beyond its true linearization, second order.
- **The corresponding norms** are reported: `s_nl_jacobian`, `s_nl_mismatch`, `s_nl_curvature`, and per class `s_nl_gp`, `s_nl_stereo`.

## Implementation

- **`certify_common.h`:**
  - **`loadFrozenState`** puts a persisted frozen-chart export's linearization points and states into a freshly built graph, relinearizes every factor there, and compares the result with the export's J and e. The difference was 0.0 on every run.
  - **`ProbeServer`** answers probe requests through files in `/dev/shm`. It never changes the solver state: message means are restored, and so are each factor's `error_`, `J_` and inlier flag.
  - **`ProbeServer::fdCheck`** compares the solver's Jacobians at the linearization points with central differences of the factors' own errors along the solver's charts.
- **`certify_stereo --probe-export EXPORT --probe-dir DIR`** runs the server on a persisted export.
- **`certify/nonlinear_probe.py`:**
  - `FullModel` is the D2 model before elimination. Its exact reduction is checked against the sidecar's S and g on every round.
  - `ProbeClient` talks to the server.
  - `Prober` builds the step and computes s^NL.
- **Scripts:**
  - `scripts/phaseD_wp6b_probe.py`: persisted WP2 rounds, re-certified with the growth rule decided after WP2;
  - `scripts/phaseD_wp6b_checks.py`: identity, order and Jacobian checks;
  - `scripts/phaseD_wp6b_gp_jacobian.py`: the yardstick with central-difference GP Jacobians.
