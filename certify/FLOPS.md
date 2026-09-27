# GBP flop counts (Phase D, WP6a)

Per-round flop counts of G-solver's frozen code path, the path the threshold schedule runs (`certify/THRESHOLD_RELIN.md`). They are taken so that the certificate's cost can be set against GBP's in the same unit.

## Instrumentation

- **Counters** live in `gsolver/include/gsolver/core/flop_count.h`. Each site adds its count to a per-thread tally, and the driver sums the tallies after every round.
  - The counts never change what the solver computes. With the counters off (the default), each site is a single test of a global flag.
  - The default path stays byte-identical (`results/phaseD/wp6/equivalence.txt`).
- **Sites**, frozen branches only:

| Category | Where | Counted per |
|---|---|---|
| `factor_update` | `FactorNode::updateFactor` (`factor_node.cpp`) | factor per round: the linear model's error, information form, conditioned joint |
| `f2v` | `FactorEdgeTuple_::sendAllMessages` / `sendMessage`, around `computeMessage` (`factor_node_edge_tuple_helper.h`) | factor-to-variable message actually computed (not to a sleeping variable) |
| `belief` | `VariableNode::updateMuAndSigma` (`variable_node.cpp`) | belief update |
| `v2f` | `VariableNode::sendMessage` (`variable_node.cpp`) | variable-to-factor message (variables with at least two edges) |
| `linearize` | `FactorNode::freezeLinearization` | relinearization: operations only, no flops |

- **Classes.** Each node carries a class (`flop_class_`): the factor type for the factor-side categories, the variable type for the variable-side ones. The per-class tallies go to `flops_by_class.csv`.
- **Driver flags:** `certify_charuco` / `certify_stereo` take `--count-flops` (threshold mode only) and `--flop-trace-round R`. The trace records one operation of every kind, with its dimensions and degree, for the hand-count check.

## Conventions

The counts are textbook dense counts, by block dimensions:
- a product (m × k)(k × n) counts 2mkn;
- an explicit inverse of an n × n matrix counts 2n³, whether Eigen takes its cofactor path (n ≤ 4) or its LU path;
- an elementwise operation counts one flop per entry, and so does `+ εI`, which counts one per diagonal entry;
- copies and block rearrangements count 0.

These match the certificate's counts (`certify/sidecar.py`, `Ops`): n³/3 per Cholesky and 2n² per right-hand side.

Per operation, with a = the outgoing variable's dimension and b = the rest of the factor, d = the variable's dimension, deg = its number of edges, m = the error dimension and n = the stacked variable dimension:

- **f2v:**
  b + 2·(2b³ + 2ab²) + 2ab + a + 2a²b + a² (marginalization; `bb.inverse()` is evaluated twice), plus a + 2a³ + 2a² + a.
  For a unary factor (b = 0) only the second part.
- **v2f**, per outgoing message: (deg − 1)(3d² + 2d) + 2d³ + 2d² + 2d.
  **The other deg − 1 incoming messages are re-summed for every outgoing message, so a variable's v2f work is O(deg²).**
- **belief:** deg(3d² + 2d) + 4d³ + 2d² + 2d (the belief inverse is formed twice: once for the mean, once for Σ).
- **factor_update:** 2mn + m + 2·(2m³ + 2nm²) + 2nm + m + n + 2n²m + n² + Σ d_k².

## Relinearization

Relinearizations are counted as operations, not flops. A hand count of the stereo factor's `updateErrorAndJacobian` gives about 665 flops:
- the point into the camera frame: 42;
- the projection: 28;
- the scalars and h: 10;
- J_hom: 7;
- J_proj: 111;
- J_lm: 72;
- the two Jacobian products: 360;
- the negation: 45.

That is about 5 % of the stereo factor's message work in one round. The GP factor's evaluation (SE(3) exponential, logarithm and their Jacobians) was not hand-counted. The WP6a report bounds relinearization flops with an allowance of 1e4 per GP factor.

## Validation

- **Hand count** (`scripts/phaseD_wp6a_handcount.py`): one operation of every kind is itemized line by line from the code and compared with the traced count.
- **Structure:** on every round, the counted messages (f2v + v2f) and relinearizations equal the independent counts in `work.csv` (`RoundRunner::messageCount`, `planThresholdRound`).
