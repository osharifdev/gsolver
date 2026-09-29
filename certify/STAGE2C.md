# Stage 2c: the clean model

Brief: `BRIEF_stage2c.md`. The protocol is Stage 1's (`certify/STAGE1.md`), unchanged; only the input graph changes.

## The clean model (`certify_stereo`)

- **`--drop-single-landmarks`:** before round 1, every landmark with exactly one stereo observation in the window is removed, together with that observation.
  - This is applied once, to the input graph, so both the solver and every export see the clean graph.
  - Counts go to `OUT/cleaning.json`: before, removed and remaining, the histogram of observations per landmark, and the poses left without observations.
- **`--live-anchor P`:** a `SE3PoseVelPrior` named `PRIOR_ANCHOR_<pose id>` on the first keyframe, at its initial ORB-SLAM3 pose, with precision P·I (1e4, decision A3).
  - It is added to the solver's graph before round 1.
  - Its error is G-solver's Log(Z⁻¹T) = [R_zᵀ(t − t_z); Log(R_zᵀR)], and its Jacobian is G-solver's. Central differences agree to ≤ 4.4e-6; the remainder is the small-angle branch of `jacob_dlogv_dv_SO3`.
- **The certificate's model** (`PoseModelBuilder`, `FullModel`) keeps `PRIOR_ANCHOR_*` factors as ordinary factors, as on ChArUco. D2 is otherwise unchanged:
  - no single-edge variable is left to eliminate: 0 in every export;
  - the weak priors p_L = 1e-6 at the g2o positions stay on the remaining landmarks;
  - factors with a zero Jacobian (stereo outliers) are dropped.
- **Both flags are recorded in the export's meta** (`drop_single_landmarks`, `live_anchor_precision`). The T1 evaluator rebuilds the same graph from them (`certify/t1.py`), and serves the anchor as a third factor type, "prior" (eval mode only; `t1_ad.h` `priorF`).

## Runs

- **Launching:** `scripts/stage1_run.sh`, `stage1_batch.sh` and `stage1_checkpointed.sh` run in the Stage 2c configuration when given:
  - `STAGE=stage2c`, which writes to `exports/stage2c/`;
  - `EXTRA_DRIVER="--drop-single-landmarks --live-anchor 1e4"`;
  - `EXTRA_EVAL="--keep-rounds 100"`, with which the evaluator keeps the round-100 linearization and beliefs for the anchor check.
- **Evaluator robustness (`scripts/stage1_evaluator.py`, `certify/stage1.py`).** A diverged run (the shipped variant on KITTI 03) is recorded rather than aborting the run:
  - a round whose exports hold non-finite numbers is written as `NONFINITE`;
  - a remaining Gauss–Newton move that cannot be computed (an indefinite reduced matrix at the beliefs) is written as NaN, flagged by `move_undefined`;
  - `stage1_analyse.py` counts such rounds and leaves them out of the validity statistics;
  - a change-based "stop" on a non-finite state (the logged largest change is 0 because the max skips NaN) is marked as not a stop.

## Scripts

- `scripts/stage1_analyse.py --stage stage2c`: the Stage 1 file set in `results/stage2c/<dataset>/<variant>/`.
- `scripts/stage2c_anchor_check.py`: §1.4, plus the attribution without p_L and the gap between the certificate's model and the solver's graph (`results/stage2c/cleaning.json`).
- `scripts/stage2c_tables.py`: §3 (`budgets.csv`, `equal_guarantee.csv`, `leave_one_out.csv`).
- `scripts/stage2c_newton.py`: §4 (`results/stage2c/<dataset>/newton.json`). It records "no certified stop" when there is none.
- `scripts/stage2c_dynamics.py`: descriptive statistics of the runs (`dynamics.json`).
- `scripts/stage2c_report.py`: the report's tables, `gate_C.json`, `comparison_old_vs_clean.csv` and `worst_cases.md`.
