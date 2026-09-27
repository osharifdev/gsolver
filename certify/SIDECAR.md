# The Phase D sidecar (WP2)

Theorem C on the live, threshold-relinearizing G-solver (`certify/THRESHOLD_RELIN.md`), with δ = 0.003 in the physical metric W = I and R = 1 (the WP1b choice).

## Protocol (one round)

The driver is `certify_charuco --mode islam` or `certify_stereo`, run with `--relin-threshold 0.003 --relin-every 1 --relin-metric physical --sidecar <dir>`. The sidecar is `scripts/phaseD_sidecar.py <dir> <log dir>`, which uses `certify/sidecar_model.py` and `certify/sidecar.py`.

1. **The driver plans the round and applies its relinearizations** (`GbpSolver::planThresholdRound`, `applyThresholdRelinearization`). Variables whose belief moved more than δ are re-frozen, and their factors are relinearized.
2. **The driver exports the current linearization and state** to `<dir>/model` (frozen charts; `<dir>` lives in `/dev/shm`). It writes `<dir>/round_info.tsv`: per pose, its displacement d_i at the relinearization test, whether it was relinearized, and whether it is asleep. Then it writes `<dir>/ready`.
3. **The sidecar builds the certified model and decides.**
   - It builds decision D2's model and reduces it exactly to the poses: S, b, and the reduced true-Jacobian Fisher information SF (stereo rotation columns halved, Theorem T0). This takes 0.6–1.3 s on the 161-keyframe prefix, and agrees with the Phase C pipeline to 1e-14.
   - It certifies the poses and decides STOP or CONTINUE.
   - It writes `<dir>/decision`: the round number, then the poses that sleep during this round's message passing.
4. **The driver applies the sleep mask** (`GbpSolver::setAsleep`) and runs the synchronous message passing. A sleeping pose neither updates nor sends, and factors do not recompute their messages to it; its last messages stay in place.

`--persist-rounds 1,10` also exports those rounds and the last one for the extended-precision validation (`scripts/phaseD_wp2_validate.py`). `--pause-every N` checkpoints the run.

## Certificate (`certify/sidecar.py`)

**Theorem A on the reduced pose model**, with D_i = S_ii:
- **Base patch, every pose.** Its rows are needed to close the recursion c* = (I − A)⁻¹s. The patch is grown until the leak row sum is ≤ 0.3, kept across rounds, and refactorized only when a block row of S inside it changes. Patches that coincide within a round share one factorization.
- **Decision-driven patch** (Phase C I1b: r = 0.3, θ = 0.02, cap 250, 40 steps), for the poses the solver did not relinearize this round (d_i ≤ δ, the amendment's necessary condition for STOP) and for sleeping poses whose patch changed. The growth threshold is min(η S_lo,i, (δ − d_i)/κ_i).
- **Sleeping poses whose patch did not change** use the running bound of Proof Note Theorem 5.4: s_i(t₀) + Σ_p ℓ_ip ‖C_p⁻¹Δg_p‖.
- **ρ(A) < 1** is certified by Collatz–Wielandt: the maximum row sum, or a Perron vector. Then E = c*, and L_i = max(0, s_i^lo − (AE)_i).

**Yardstick:** S_lo,i = √λ_min(C_iᵀ [SF_PP⁻¹]_ii C_i) · q₁₂(0.05) on the base patch, with κ_i = ‖C_i⁻ᵀ‖ (W = I).

**Decision:**
- **STOP** if d_i + κ_i E_i ≤ δ and E_i ≤ η S_lo,i.
- **CONTINUE** otherwise.
- **RELINEARIZE-NOW is not implemented.** At R = 1 the solver relinearizes as soon as d_i > δ, so it cannot fire.

A sleeping pose that fails its recheck wakes (reason `recheck`). If its patch changed, the patch is refactorized; the pose stays asleep if the recomputed certificate still passes, and otherwise wakes (reason `relinearization`, or `new_factor` when a pose added this round is in or next to the patch).

**If the linearization is not positive definite** (assumption A0; ChArUco's first round, one pose with no velocity information), nothing is certified and every pose stays awake.

## Oracle and invariants, every round

- **Oracle:** e = S⁻¹g by a dense Cholesky of the reduced model, and the exact yardstick from [SF⁻¹]_ii.
- **Oracle STOP:** the same rule with ‖e_i‖ in place of E_i.
- **Invariants:** L_i ≤ ‖e_i‖_{D_i} ≤ E_i for every pose with a certificate, at 1e-9 relative. On rounds 1, 10 and the last, the patch lower bound is also checked against the exact marginal in the Loewner order.
- **Precision:** everything above is double precision. The sampled rounds are re-checked with the extended-precision Phase C pipeline.
