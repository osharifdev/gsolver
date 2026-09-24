#pragma once

#include <gsolver/graph/factor_graph.h>

namespace gsolver {

  enum class SolverScheduleType { SYNCHRONOUS, ASYNCHRONOUS };

  class GbpSolver {
  public:
    GbpSolver(SolverScheduleType schedule_type);

    SolverScheduleType schedule_type_;

    // Performance-only option, default off: dynamic OpenMP scheduling of the four per-node loops of a
    // synchronous iteration. Every node's update reads only the previous phase's output and writes only
    // its own state, so the result does not depend on which thread runs it (bit-identical; see
    // certify/FREEZE_MODE.md). With the default static schedule all pose nodes, which come first in
    // variable_nodes_ and have O(degree^2) message cost, land on one thread.
    bool dynamic_schedule_ = false;

    void performIteration(FactorGraph& factor_graph);
    void performAttentionIteration(FactorGraph& factor_graph,
                                   const std::vector<std::string>& attention_variables_ids,
                                   const std::vector<std::string>& attention_factors_ids);
    void solve(FactorGraph& factor_graph);

    // Frozen (linear) mode, see certify/FREEZE_MODE.md. Fixes every variable's linearization point at
    // its current belief mean, linearizes every factor there, and moves all messages into the fixed
    // charts; afterwards performIteration() is Gaussian BP on one linear model. Pass
    // factor_messages_valid = false if no iteration has run yet (factor-to-variable messages unset).
    void freeze(FactorGraph& factor_graph, bool factor_messages_valid = true);
  };

} // namespace gsolver
