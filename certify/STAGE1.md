# Stage 1: periodic certified termination

Brief: `BRIEF_stage1_2.md` §2, with the decisions of its §0.

## The run

**Schedule.** `certify_stereo` / `certify_charuco --mode islam` run the threshold schedule of WP1b: δ = 0.003 in the physical metric, R = 1, one linearization point per variable. The **shipped** variant uses G-solver's Jacobians. The **corrected** variant adds `--true-jacobians` (`certify/TRUE_JACOBIANS.md`).

`scripts/stage1_run.sh <dataset> <variant> <rounds>` launches the driver and the evaluator. `scripts/stage1_checkpointed.sh` does the same in checkpointed segments and evaluates the gate at each checkpoint.

**Every round**, the driver logs:
- **GBP flops** (`flops.csv`, WP6a counters): as implemented, and with variable-to-factor messages formed as belief minus the one incoming message (`gbp_flops_alt`, the counted arithmetic of that implementation);
- **the change-based criterion** (`heuristics.csv`):
  - c(t) = max over variables present in both rounds of ‖belief(t) ⊟ belief(t−1)‖_{D_v};
  - D_v is the variable's sum of diagonal blocks of this round's factor information, which is H_vv of the current linearization;
  - at the first round below each τ ∈ {1e-2, 1e-3, 1e-4}, the beliefs are exported to `stop_tau_<τ>`;
  - round 1 has no previous belief to compare with, and ChArUco counts only from round 95, the second round on the complete graph.

**After every 10th round** (and after the last), the driver:
1. applies the relinearizations the solver would make before its next round;
2. exports that linearization (frozen charts) and the beliefs (belief charts) to `/dev/shm`;
3. waits for the evaluator, `scripts/stage1_evaluator.py`.

On ChArUco the evaluation is taken before the next round's messages, after the front end has added that round's pose. That matters only in the incremental phase, before the last pose is added at round 94.

## The evaluator

**The certificate** (`certify/stage1.py`, `StatEngine`) is Theorem A on the reduced pose model of decision D2, for every pose.
- **Base patches:** leak row sum ≤ 0.3, regrown each evaluation from the previous patch.
- **Base recursion:** c0.
- **Yardstick:** S_lo,i on the base patch of the true-Jacobian Fisher information, δ_s = 0.05, q₁₂.
- **Growth, whose only target is the statistical decision** (threshold ηS_lo,i, η = 0.1). It exits:
  - on CONTINUE when s_i − Σ a_ij c0_j > ηS_lo,i;
  - on STOP when s_i + Σ a_ij c0_j ≤ ηS_lo,i;
  - otherwise it grows until the cap of 250 poses or 40 steps.
- **Final bounds:** E_i = min(c*_i, s_i + Σ a_ij c0_j), both certified upper bounds; L_i = max(0, s_i − Σ a_ij E_j).
- **Rule:** a pose passes when E_i ≤ ηS_lo,i. The certified stop is the first evaluation round at which every pose passes (on ChArUco, from round 94).

**The oracle**, every evaluation: e = S⁻¹g by dense Cholesky, and the exact yardstick from [I_F⁻¹]_ii. The invariants:
- L_i ≤ ‖e_i‖_{D_i} ≤ E_i, at 1e-9 relative;
- at the first, middle and last evaluation, the patch lower bound against the exact marginal in the Loewner order.

**Accuracy**, every evaluation, from the belief-chart export:
- the remaining Gauss–Newton move ‖x̄_i‖_{D_i}, x̄ = H⁻¹b at the beliefs. This is WP1's definition: D_i = H_ii of the D2 model before landmark elimination, with the solver's own Jacobian. It is also given with the true Jacobian.
- on stereo, the SE(3)-aligned ATE of the keyframe beliefs against the ground truth at the keyframe timestamps.
- The fast computation agrees with WP1's Phase C pipeline to 5e-14 on KITTI-40.

## Analysis

- **`scripts/stage1_analyse.py`:**
  - stops: certified, fixed budget (100 rounds on KITTI, 103 on ChArUco), change-based per τ;
  - costs to each stop: GBP flops, plus the certificate's flops at its evaluation rounds or the change criterion's flops;
  - η sensitivity from the logged E_i and S_lo,i;
  - validity; Gate A.
- **`scripts/stage1_never_finishes.py`:** §2.5.
- **`scripts/stage1_fdcheck.py`:** the Jacobian check.
