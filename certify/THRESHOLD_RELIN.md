# Threshold relinearization in G-solver (Phase D, WP1)

`GbpSolver::threshold_relinearization_` (default off) replaces G-solver's per-round relinearization with the
per-variable threshold rule of `STOP_OR_RELINEARIZE.md` §1.1, using the frozen-mode bookkeeping of
`certify/FREEZE_MODE.md`. Decisions D-a and D-b of the WP0 review fix the design:

- **One linearization point per variable** (`mu_lin_`), shared by every factor touching it. The state of a
  variable is its tangent coordinate `z_` in the chart at `mu_lin_`, so the current model is exactly one
  linear system H x = b in those charts (assumption A1). It is the model that Theorem A/C certify and the
  per-round oracle solves.
- **Threshold in the D-metric, exponential chart**: d_v = ‖z_v‖_{D_v}, D_v = H_vv of the current model
  (Σ over v's factors of J_lin,vᵀ Σ_f⁻¹ J_lin,v; the 1e-6 regularizers are not part of H), so κ_v = 1.

## One round (`GbpSolver::performThresholdIteration`)

1. **New nodes.** Variables added since the last round are frozen at their current mean
   (`VariableNode::freezeAsNew`); then new factors linearize at their variables' points
   (`FactorNode::freezeNewFactor`). At the first round every node is new, which is the frozen mode's
   `freeze(graph, false)`.
2. **D_v** is recomputed for every variable whose factors were added or relinearized.
3. **Test.** d_v is computed for every variable. v is re-frozen if d_v > δ and at least R rounds have passed
   since its last re-freeze (or its creation).
4. **Re-freeze and relinearize.** For each such v: `mu_lin_ ← mu_lin_ ⊞ z_` (the belief mean), every stored
   message mean on v's edges keeps its manifold point and is re-expressed in the new chart, z_v = 0
   (`VariableNode::refreezeAtBelief`). Every factor touching a re-frozen variable relinearizes at the current
   `mu_lin_` of **all** its variables (`FactorNode::freezeLinearization`). Precisions are kept as stored, as
   in the frozen mode (G-solver never transports precisions between charts).
5. **Message passing.** The unchanged synchronous schedule (four phases, static or dynamic OpenMP) runs on the
   linear model, exactly as in the frozen mode.

Invariant: in threshold mode every edge's incoming-to-factor mean `variable_to_factor_message_.mu_` equals its
variable's `mu_lin_`, so a factor can relinearize at any round. Nothing else writes that field while frozen.

`GbpSolver::relinearizeAllAtBeliefs` re-freezes every variable and relinearizes every factor. The drivers call
it once at the end, so `mu_` holds the final beliefs (for the TUM output) and the frozen export is the
linearization at the final beliefs.

## What it is not

- With δ = 0 and R = 1 every variable that moved is re-frozen every round, so every factor relinearizes at the
  **belief means** each round. The shipped solver relinearizes each factor at its **cavity means** (its
  incoming variable-to-factor message means; WP0 §3.4). The two are different algorithms; their accuracy
  difference is measured in WP1 as part of L1, not asserted equal.
- The option must be enabled before the first iteration of a graph: `freezeAsNew` assumes a variable's edges
  carry no factor messages yet.

## Default path

`performIteration()` gains one branch at its top (`if (threshold_relinearization_)`); with the option off the
shipped schedule runs unchanged. `scripts/phaseA_equivalence.sh` byte-compares the default path.
