# Frozen (linear) mode in G-solver

Amendment A2 replaces the brief's "freeze relinearization" with a solver mode in which G-solver's GBP
is Gaussian belief propagation on one fixed linear model. This file states what the mode does, which
of G-solver's code paths it bypasses (line references to the shipped code at `b330450` and to the
current branch), and how G-solver's `1e-6·I` regularizers enter its fixed point.

## 1. Definition

`GbpSolver::freeze(graph, factor_messages_valid)` (`gsolver/src/core/gbp_solver.cpp:91-104`) does three things:

1. **Fix linearization points.** Every variable sets `mu_lin_ = mu_` (its belief mean at the freeze
   instant; `VariableNode::beginFreeze`, `variable_node.cpp:209-222`). Every factor reads its
   linearization point from its incoming variable-to-factor message means. For this one step those
   means are pointed at `mu_lin_`.
2. **Linearize once.** Every factor calls its own `updateErrorAndJacobian()` at those points and stores
   `error_lin_ = e_f(x̂)`, `J_lin_ = J_f(x̂)` (`FactorNode::freezeLinearization`, `factor_node.cpp:53-61`).
   The stereo factor's in-image test (`inlier_`) is therefore also evaluated once, at x̂.
3. **Move to fixed charts.** The message means are restored, and every stored message mean is
   converted once to a tangent coordinate in the variable's fixed chart: `tau_ = mu_lin_ ⊟ mu_msg`
   (`VariableNode::endFreeze`, `variable_node.cpp:231-244`). Precisions are kept as stored; G-solver
   never transports precisions between charts. Each variable's state becomes `z_ = 0`, so the
   variable's value is `x̂ ⊞ z_`. `factor_messages_valid = false` (freezing before the first
   iteration) sets the never-written factor-to-variable means to 0. The first frozen factor update
   overwrites them before any variable reads them.

After the freeze no `⊞`/`⊟` (boxplus/boxminus) is evaluated again: every mean is a vector in a fixed
chart, `mu_` is never rewritten, and `performIteration()` runs G-solver's unchanged synchronous
schedule (`gbp_solver.cpp:8-60`: the shipped loops, plus an optional dynamically scheduled copy of the same four loops, `dynamic_schedule_`, off by default and bit-identical) on the linear model

    e_f(z) = e_f(x̂) + J_f(x̂) z_f,    with charts x ⊞ τ = (T·Exp(τ₀:₆), v + τ₆:₁₂) for SE3PoseVel,
    T·Exp(τ) for SE3Pose, p + τ for SE3Point, and Exp(t, r) = (Rodrigues(r), t) (geometry3d.h:126-133).

A factor added after the freeze (for example the A3 stereo gauge prior) is frozen with
`FactorNode::freezeNewFactor()` (`factor_node.cpp:63-69`). `createEdge` initializes its messages from
the variable's `mu_`, which equals `mu_lin_` because frozen variables never rewrite `mu_`.

## 2. Code paths bypassed in frozen mode

| Where (shipped `b330450`) | Default behaviour (unchanged when not frozen) | Frozen replacement (current branch) |
|---|---|---|
| `factor_node.cpp:18` `updateErrorAndJacobian()` | re-evaluate e_f, J_f at the incoming message means every iteration (all factor types: `se3_posevel_gp.cpp:10-38`, `se3_posevel_pose.cpp:9-27`, `se3_posevel_prior.cpp:9-22`, `se3_posevel_stereo_point.cpp:12-87`, …) | `factor_node.cpp:18-26`: `error_ = error_lin_ + J_lin_ · τ_in` with τ_in the stacked incoming message means (`fillMessageTau`, `factor_node_edge_tuple_helper.h:73-78`), and `J_ = J_lin_` |
| `factor_node_helper.h:73` `VariableType::boxplus(mu_to_update, tau)` | the factor-to-variable mean is a manifold point retracted from the incoming mean | `factor_node_helper.h:80-86`: `message.tau_ = message_in.tau_ + tau` |
| `variable_node.cpp:39,42` (and `98,101`, `146,149`) `boxminus(mu_inward)` | each incoming mean is projected into the chart at the current belief mean `mu_` | `variable_node.cpp:43-44` (and `114-115`, `174-175`): `tau_ − z_` |
| `variable_node.cpp:57,61` (and `116,120`) `boxplus(...)`, write `mu_` of the message | variable-to-factor mean retracted onto the manifold | `variable_node.cpp:62-67` (and `133-138`): write `tau_ = z_ + δ` |
| `variable_node.cpp:164` `mu_ = boxplus(tangent_delta_mu)` | the belief mean moves on the manifold (the chart moves with it) | `variable_node.cpp:193-199`: `z_ = z_ + δ`, `mu_` unchanged |

Unchanged in frozen mode: the factor's canonical form (`eta_ = Jᵀ Σ⁻¹ (−e)`, `Lambdaprime_ = Jᵀ Σ⁻¹ J`,
robust weight `k = 1`) and the conditioning on incoming precisions (`factor_node.cpp:27-36`). Also
unchanged: the rearrangement and Schur-complement marginalization (`factor_node_helper.h:12-31,40-75`),
the precision algebra of beliefs and variable-to-factor messages, the synchronous schedule, and the
single-edge rule (`variable_node.cpp:26-29`: a variable with one edge never sends an updated message).
The four `+ 1e-6·I` sites read `frozen_eps_{belief, v2f, marg, f2v}` (`gsolver/core/frozen_mode.h`),
which default to 1e-6. The default path keeps the literal `1e-6`.

The default path is bit-identical to the shipped solver. Phase A byte-compared trajectories of the
shipped drivers rebuilt against the modified library with the Phase 0 outputs, the certify drivers
with the shipped drivers, runs with and without unfrozen exports, and runs with and without the
dynamic schedule: 12 of 12 identical (`scripts/phaseA_equivalence.sh`, `results/phaseA/equivalence.txt`).

## 3. The model the frozen mode solves

With `f` over all factors, the frozen iteration is Gaussian BP on

    H = Σ_f J_fᵀ Σ_f⁻¹ J_f + Σ_{v: deg v = 1} Λ_v,     b = Σ_f J_fᵀ Σ_f⁻¹ (−e_f) + Σ_{v: deg v = 1} Λ_v τ_v,

in the fixed charts at x̂. The first sum contains the gauge terms: the ChArUco drivers' `PRIOR_ANCHOR`
factors and, for stereo, the A3 `CERTIFY_GAUGE_PRIOR`.

The second sum is G-solver's **frozen single-edge messages**. A variable with exactly one edge never
recomputes its outgoing message (`variable_node.cpp:26-29`), so that message keeps its creation value
(`factor_graph_helper.h:37-38,75-76`: initial mean, initial precision `Sigma⁻¹`). Its factor therefore
sees a fixed prior N(τ_v, Λ_v⁻¹) on v. At the fixed point this prior acts on **every other variable**
exactly like a unary factor on v, so it is part of H. It is **absent from v's own belief**, which is
the product of factor-to-variable messages only (`updateMuAndSigma`). GBP's converged belief of a
single-edge variable is therefore not x̄_v. On a tree it would be the model marginal of v with that
prior divided out, x̃_v = (Σ_vv⁻¹ − Λ_v)⁻¹ (Σ_vv⁻¹ x̄_v − Λ_v τ_v). On a loopy graph it is the factor's
message computed from GBP's cavity of the neighbour, whose precision is approximate, so it matches no
exact model quantity (on the 25 % islam snapshot x̃_v misses it by 2.2e-2). In KITTI 06, every `LANDMARK_3D` covariance in the
file is 0. `add_landmark_variable_nodes` adds `1e-6·I`, so every single-observation landmark carries a
frozen prior of precision 1e6·I (σ = 1 mm) at its ORB-SLAM3 position.

## 4. How the `1e-6·I` regularizers enter the fixed point

With ε at each site, in information form in the fixed charts (`m_{i→f}` the variable-to-factor mean,
the cavity mean of i with respect to f; `z_i` the belief mean):

| Site | Code | Effect | At the fixed point |
|---|---|---|---|
| ε1 belief | `updateMuAndSigma`, `variable_node.cpp:189` | belief × N(z_i^old, ε⁻¹I) (proximal) | cancels: Σ_f Λ_{f→i}(m_{f→i} − z_i*) = 0 |
| ε2 variable→factor | `sendMessage`, `variable_node.cpp:58,129` | every outgoing message × N(z_i, ε⁻¹I) | a unary prior at z* on every variable: GBP on (H + εI) z = b + ε z*, whose solution is H⁻¹b. **No bias** |
| ε3 marginalization | `marginalize`, `factor_node_helper.h:26` (value set at `:70`) | in factor f, every other variable's incoming message gets precision +εI at its own mean m_{b→f} | a prior centred at a cavity mean, a GBP-internal point: **biases the fixed point, O(ε)** |
| ε4 factor→variable | `computeMessage`, `factor_node_helper.h:74` | every factor-to-variable message × N(m_{a→f}, ε⁻¹I) | a prior centred at the cavity mean m_{a→f}: **biases the fixed point, O(ε)** |

ε3 and ε4 are centred at cavity means, which are quantities of the GBP iteration and not of the model.
No fixed (G, b_G) reproduces them, so they are **not** part of the exported G. Their effect is measured
instead (ChArUco BA, frozen after its 50 normal iterations, `results/phaseA/`):

* The floor of max_i ‖z_i − x̄_i‖_{D_i} is 739·ε for every ε in {1e-5, 1e-6, 1e-7}, the same for all four
  sites together: 7.35e-3, 7.39e-4, 7.40e-5. The convergence rate does not change (0.99886 per iteration).
* Reducing one site to 1e-7 with the others at 1e-6: ε1 → floor 7.391e-4 and ε2 → 7.392e-4 (no change,
  as predicted above); ε3 → 2.46e-4; ε4 → 9.42e-4. The bias comes from ε3 and ε4, which partly cancel.
* Below ε ≈ 1e-8 round-off takes over. The inverses of the rank-deficient factor messages, such as
  pose-landmark messages that do not constrain velocity, are conditioned like 1/ε: ε = 1e-8 → 1.3e-5,
  1e-9 → 8.5e-5 (noisy), 1e-12 → 5.3e-2.

The worst blocks are consecutive poses less than 1/30 s apart. For those pairs the GP prior makes D_i
stiff (eigenvalues up to 5e7), so a raw offset of 6e-7 m reads as 7e-4 in the D-metric.

## 5. Python bindings (Phase B)

The frozen mode and the linear model of each factor are reachable from Python (`build-certify/python/_gsolver`; the
certify package adds that directory to `sys.path`, see `certify/synth.py`). Only read access and the existing entry
points are exposed; no solver code changed:

- `GbpSolver.freeze(factor_graph, factor_messages_valid=True)`, `GbpSolver.dynamic_schedule` (read/write),
  `set_frozen_regularizer(eps)`, `frozen_regularizers()` (`python/core/bind_gbp_solver.cpp`);
- messages: `tau` (frozen-chart mean), edges: `frozen` (`python/graph/bind_edge.cpp`);
- `SE3PoseVelGP`, `SE3PoseVelPrior`, `SE3PoseVelPose`, `SE3PoseVelPoint`: `J`, `error`, `Sigma`, `J_lin`, `error_lin`,
  `frozen`; GP and prior factors also `edge(i)` (`python/graph/types/factors/factor_accessors.h`);
- `SE3PoseVel`: `frozen`, `mu_lin`, `z` (`python/graph/types/variables/bind_se3_posevel.cpp`).
