# Phase D, WP0: the rotation chart (code read)

Branch `certify`, base commit `e89fc01`. Date 2026-09-27. Compute used: none (code read only; the one numerical fact cited is Phase A's).

**Documents.** `BRIEF_phaseD.md` and `STOP_OR_RELINEARIZE.md` are not on this machine: I searched `/home/os717` and `/tmp` by name and by content ("RELINEARIZE-NOW", "Theorem T0"). I worked from the brief's text in the instruction message. The `PROOF_NOTE.md` in the repository is the 2026-09-24 version with the 25 Sep errata: sections 0–7, with no "Parts I–II", Theorem A, Theorem C, T0 or T1. WP0 needs none of these. WP1 onward does (§4).

## 1. Five-line summary

1. **G-solver's rotation ⊞ is the exponential map, applied on the right.**
   - The pose update is T ⊞ τ = T · Exp(τ₀:₆) on SO(3) × ℝ³: rotation R·Exp(ω), translation t + R·δt. Velocity is additive.
   - Exp is Rodrigues' formula: R(δ) = I + sin θ K + (1 − cos θ) K², exactly I + [δ]× for θ² < 1e-8. So R(δ) ≈ I + [δ]×, not I + 2[δq]×.
   - No quaternion appears anywhere in the library.
2. **So the stereo Jacobian's factor of 2 is a modification, not a chart convention.** It doubles the rotation block relative to the derivative in the solver's own chart. Phase A measured this by finite differences under this ⊞: 2.000000 ± 1.1e-6 on 2,000 factors each of the prefix and KITTI full. By the brief's rule, **S_lo must use the true Jacobian (Theorem T0), and J̃ ≠ J for stereo.** J̃ is J with every stereo factor's three rotation columns halved. Every other factor type is consistent with the exponential chart.
3. **G-solver never measures belief movement.**
   - `solve()` has no stopping criterion (a TODO), and nothing thresholds relinearization: every factor relinearizes every iteration.
   - The per-round belief move is δ_i = Λ'_i⁻¹η_i in the exponential chart at the old mean, where Λ'_i is GBP's belief precision.
   - The only movement measure in our stack is my Phase A driver's, ‖μ_i ⊟ μ_i^prev‖ in Λ'_i. That is our convention, not the solver's.
   - So δ has no native metric, and WP1 must fix one (decision D-a).
4. **G-solver's linearization points are per factor, not per variable.** Each factor linearizes at its incoming variable-to-factor message means, which are cavity means, so the factors of one variable sit at different points in the same round.
   - The brief's §1 assumes one x_lin per variable and one linear system in its charts.
   - Only the per-variable form gives that exact linear model, but the shipped per-factor form is what "δ = 0, R = 1 reproduces the shipped output" requires (decision D-b).
5. **Useful for WP2.** The solver already has a subset update, `performAttentionIteration`, which updates only listed factors and variables. Frozen mode already keeps one fixed chart per variable. Stereo factors are not exposed in the Python bindings, which WP5a needs.

## 2. Gate status

WP0 has no gate. It ends at a decision point (§4).

## 3. Findings, with file and line

### 3.1 The ⊞ and ⊟ of a pose–velocity variable

- `gsolver/src/graph/types/variables/se3_posevel.cpp:20-25` (member) and `:27-32` (static): `pose ← pose * expMapSE3(tau.head(6))`, `vel ← vel + tau.tail(6)`.
- `:13-18`: `tau.head(6) = logMapSE3(pose.inverse() * pose_in)`, `tau.tail(6) = vel_in − vel`.
- Tangent order [t(3), ω(3), v(6)]: `expMapSE3` takes the translation from `x.head(3)` and the rotation from `x.tail(3)`. The stereo factor's comment at `se3_posevel_stereo_point.cpp:71` states the same order.
- `gsolver/include/gsolver/maths/geometry3d.h:126-133`, `expMapSE3`: translation = x₀:₃ directly, rotation = `expMapSO3(x₃:₆)`. This is the product chart SO(3) × ℝ³, not the SE(3) exponential (no V(ω) coupling). Composed on the right: T·Exp(τ) = (R·Exp(ω), t + R·δt).
- `geometry3d.h:79-92`, `expMapSO3`: Rodrigues, `R = I + sin θ K + (1 − cos θ) K²` (line 89). For θ² < 1e-8 it uses `R = I + W` with W = [ω]× (line 86). Phase A recorded the resulting drift off SO(3), up to 1.2e-6.
- `geometry3d.h:94-124`, `logMapSO3`: the inverse, θ/(2 sin θ)·vee(R − Rᵀ). `:135-140`, `logMapSE3` = (t, Log R).
- **No quaternion:** `grep -iE "quaternion|Quaterniond|\.vec\(\)|coeffs\(\)"` over `gsolver/include` and `gsolver/src` finds nothing. The state is an `Eigen::Isometry3d` plus a 6-vector velocity (`se3_posevel.h:10-11`).

**Chart:** exponential, right perturbation, R(δ) = I + [δ]× + O(|δ|²).

### 3.2 The stereo rotation Jacobian

- `se3_posevel_stereo_point.cpp:22`: `p_cam = T_world.inverse() * p_world`. Under T ⊞ δ = T·Exp(δ), p_cam(δ) = p_cam − δt + [p_cam]× δω + O(|δ|²). So the true derivatives are ∂p_cam/∂δω = [p_cam]× and ∂p_cam/∂δt = −I.
- The code's own comment says exactly this (`:72-73`: "d(p_cam)/d(delta_r) = skew(p_cam) (right perturbation)").
  - `:76`: translation block `−P_ext·I`, correct.
  - `:77`: `J_proj.block<4,3>(0,3) = 2.0 * P_ext.block<4,3>(0,0) * skew(p_cam)`, **twice the derivative**.
- Numerically (Phase A, `reports/phaseA.md` F5; `f5.json` in the prefix and KITTI-full exports): the rotation block is 2.000000 ± 1.1e-6 times the finite-difference derivative under G-solver's ⊞. The translation and landmark blocks agree to finite-difference error.
- **It is the only such factor.** A search of all factor sources for 2.0/0.5 multipliers finds only line 77.
  - The point factor uses `-skew(h)` (`se3_posevel_point.cpp:31`).
  - The GP, prior and pose–pose factors form their errors with `logMapSE3` of relative poses (`se3_posevel_gp.cpp:27`, `se3_posevel_prior.cpp:18`, `se3_posevel_pose.cpp:23`), consistent with the exponential chart.

**Consequences.**
- The quaternion reading, in which R ≈ I + 2[δq]× would make the 2 correct, is ruled out: the solver's ⊞ uses δω directly as the rotation vector.
- The assembled model with the 2 is still a well-defined linear model, and GBP solves it. The certificates (E, L) are about that model, as in Phase C. What changes is the yardstick: per the brief, S_lo must come from the true-Jacobian Fisher information (Theorem T0).
- **J̃ is exact and cheap.** Take J with columns 3–5 of the pose block of every stereo factor divided by 2 (`se3_posevel_stereo_point.cpp:77` without the `2.0`). The sidecar can form it from the exported J.
- Phase C's stereo radii used H as assembled (errata F5: "the certified model is the assembled one"). Its stop fractions would change under T0. The direction is not fixed in general, since J̃ᵀΛJ̃ and JᵀΛJ are not ordered in the Loewner sense (the cross terms change sign relative to the diagonal ones).

### 3.3 Relinearization and belief movement

- **No stopping criterion or movement measure.** `GbpSolver::solve` is `while (true) performIteration(...)` with "TODO: Implement stopping criteria" and "FIXME" (`gsolver/src/core/gbp_solver.cpp:106-110`). Iteration counts are set by the caller.
- **Relinearization, every factor, every iteration** (Phase A F1):
  - `performIteration` runs `updateFactor` on all factors, then factor messages, then `updateMuAndSigma`, then variable messages (`gbp_solver.cpp:29-55`; `:9-28` is the same with the dynamic schedule).
  - Outside frozen mode, `updateFactor` calls `updateErrorAndJacobian()` every time (`gsolver/src/graph/factor_node.cpp:24-26`).
- **Belief update** (`gsolver/src/graph/variable_node.cpp:157-205`):
  - Λ'_i = Σ_f Λ_{f→i} + 1e-6·I (`:168-189`);
  - δ_i = Λ'_i⁻¹ Σ_f Λ_{f→i} (m_{f→i} ⊟ μ_i) (`:191`), a vector in the exponential chart at the current mean;
  - μ_i ← μ_i ⊞ δ_i (`:202`), and Σ_i = Λ'_i⁻¹ (`:203-204`).
- **The only movement measure in the stack is ours**: `certify/cpp/certify_common.h:500-515`, s_i = (d_iᵀ Λ'_i d_i)^{1/2} with d_i = μ_i ⊟ μ_i^prev at the new mean. It is Phase A's convergence rule, not G-solver's.

**Available metrics for δ and d_i**, both local to a node:
- D_i = H_ii of the current linearization. This is the certified metric, and κ_i = 1. It is local: D_i = Σ_{f∋i} J_{f,i}ᵀΛ_f J_{f,i} from the node's own factors.
- Λ'_i, GBP's belief precision. It is already at the node, with κ_i = ‖Λ'_i^{1/2} D_i^{-1/2}‖.

### 3.4 Linearization points are per factor (cavity means)

- **Variable-to-factor message mean = cavity mean.** For the out-edge to f, the other incoming messages are combined in the chart at μ_i (+1e-6·I) and mapped back by ⊞ (`variable_node.cpp:103-145`). A variable with a single edge sends nothing (`:89-90`).
- **Each factor's linearization point is its incoming message means.** For example, stereo reads `variable_to_factor_message_.mu_` (`se3_posevel_stereo_point.cpp:15-16`). Factor-to-variable messages are formed in the tangent space at those points and mapped by ⊞ (`gsolver/include/gsolver/graph/helpers/factor_node_helper.h:88-93`).
- **So there is no single linearization point per variable in the shipped solver.** In one round the factors of variable i are linearized at different points: each at the belief without its own message.
- Phase A's unfrozen exports built a single-chart model by re-evaluating every factor at the belief means (`reports/phaseA.md` §3.6). That model is close to what GBP ran, but not it.
- **Frozen mode is the per-variable form.** One fixed point `mu_lin_` per variable. Every factor is linear there (`factor_node.cpp:18-23`: error = error_lin + J_lin·τ), and messages live in fixed charts (`factor_node_helper.h:80-86`; `certify/FREEZE_MODE.md`).

### 3.5 Other facts that bear on WP1–WP5a

- **`performAttentionIteration`** (`gbp_solver.cpp:62-89`) updates only listed factors and variables, in the same four phases. It is a ready base for WP2's sleeping poses. It is serial, with no OpenMP.
- **Single-edge variables** (landmarks seen once) send no messages (`variable_node.cpp:89-90`). Phase C's D2 elimination handles them in the certified model.
- **Stereo factors have no Python bindings** (Phase A F5). WP5a re-evaluates every factor's error and Jacobian at x_lin ⊞ z₁ through the bindings, so the stereo factor must be bound first, or evaluated in C++.

## 4. Decision point: needed before WP1

**D-a. The metric and chart of δ and d_i.** Recommendation: **D_i = H_ii of the current linearization, in the exponential chart at x_lin**, d_i = ‖μ_i ⊟ x_lin,i‖_{D_i}.
- It is the certificates' own metric, so κ_i = 1.
- It is computable at the node from its own factors.
- The brief's grid δ ∈ {0.25, 0.5, 1, 2} is stated in D-metric units.

The alternative, Λ'_i, is GBP's own and already at the node. It is not the certified metric, and on loopy graphs it is GBP's approximation to the marginal precision.

**D-b. One linearization point per variable, or per factor.**
- **(i) Per variable, frozen-mode style.**
  - x_lin,v stays fixed until v's belief moves more than δ from it (at most once per R rounds). Then x_lin,v ← μ_v, and every factor touching v relinearizes at the current x_lin of all its variables.
  - The model is exactly one linear system in one chart per variable, the system that Theorems A and C certify and that the per-round oracle solves. It reuses the frozen-mode code: it is a per-variable re-freeze.
  - With δ = 0 and R = 1 it relinearizes at belief means, not cavity means, so **it cannot reproduce the shipped output**. The option-off path stays byte-identical.
- **(ii) Per factor at cavity means, as shipped.**
  - δ = 0 and R = 1 reproduce the shipped output exactly.
  - But the factors of one variable sit in different charts, so "the current linearization" is not a single linear model. The certificates and the oracle would be computed on a re-evaluated model, as in Phase A, not on what GBP runs, and Theorem C's premise holds only approximately.

**Recommendation: (i).**
- Keep the byte-identity check for the option-off path.
- Replace "δ = 0, R = 1 reproduces the shipped output" with "δ = 0, R = 1 reproduces a belief-mean relinearizing run". Report its accuracy against the shipped schedule as part of L1: the difference between cavity-mean and belief-mean relinearization is itself what L1 must quantify.

**D-c. Missing documents.** Please add `STOP_OR_RELINEARIZE.md`, the proof note with Parts I–II, and `BRIEF_phaseD.md` to the repository root. WP1 does not need them, beyond the brief I have. WP2 needs Theorem C's exact statement (κ_i, per-pose δ_i), Theorem T0's construction of S_lo from J̃, and Theorem 5.4 as it appears in Parts I–II.

## 5. What could not be done

- The two documents above could not be read, because they are not on this machine. Nothing in this report depends on them beyond the brief's own statement of the consequence ("quaternion ⇒ J̃ = J; exponential ⇒ S_lo uses the true Jacobian, Theorem T0").
- WP0 has no worst cases or tables: it is a code read.

## 6. Files

- `reports/phaseD_wp0.md` (this file). Commit: the commit that adds it (`git log -- reports/phaseD_wp0.md`).
- Sources read: `gsolver/src/graph/types/variables/se3_posevel.cpp`, `gsolver/include/gsolver/maths/geometry3d.h`, `gsolver/src/graph/types/factors/se3_posevel_{stereo_point,point,gp,prior,pose}.cpp`, `gsolver/src/graph/variable_node.cpp`, `gsolver/src/graph/factor_node.cpp`, `gsolver/include/gsolver/graph/helpers/factor_node_helper.h`, `gsolver/src/core/gbp_solver.cpp`, `certify/cpp/certify_common.h`, `reports/phaseA.md` (F1, F5, §3.6).
