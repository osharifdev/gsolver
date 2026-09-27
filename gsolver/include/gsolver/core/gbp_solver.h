#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>

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

    // Threshold relinearization, default off; see certify/THRESHOLD_RELIN.md. When on, every variable keeps
    // one linearization point (the frozen-mode bookkeeping above) and performIteration() first freezes the
    // variables and factors added since the last round, then re-freezes every variable v whose belief has
    // moved more than relinearize_threshold_ from its linearization point, d_v = ||z_v||_{D_v} with
    // D_v = H_vv of the current linear model, at most once per relinearize_every_ rounds, and relinearizes
    // every factor touching it at the current points of all its variables. Then it runs the synchronous
    // schedule on the linear model. Enable it before the first iteration of a graph.
    bool threshold_relinearization_ = false;
    double relinearize_threshold_   = 0.0; // delta
    int relinearize_every_          = 1;   // R
    // Metric of d_v: Physical = the Euclidean norm of the tangent vector z_v (metres, radians, m/s and rad/s,
    // i.e. W = I; revised decision D-a of Phase D), HBlock = ||z_v||_{D_v} with D_v = H_vv (WP1).
    enum class RelinMetric { Physical, HBlock };
    RelinMetric relinearize_metric_ = RelinMetric::Physical;

    // Threshold mode: decide the next round (freeze new nodes, test every variable) without running it, and
    // return its linearizations: new factors plus factors relinearized by the rule. performIteration() then
    // runs exactly this round. Lets a caller stop at a work budget.
    int planThresholdRound(FactorGraph& factor_graph);

    struct ThresholdStats {
      int round = 0;                      // threshold rounds planned so far (the current one included)
      std::vector<int> relinearized_vars; // indices into variable_nodes_ re-frozen in the last round
      int relinearized_factors = 0;       // factors relinearized in the last round by the rule
      int new_factors          = 0;       // factors first linearized in the last round
      std::vector<double> displacement;   // d_v per variable at the last round's test
      std::vector<int> relinearizations;  // re-freezes per variable so far
      std::vector<int> created_round;     // round in which each variable was first linearized
    };
    ThresholdStats threshold_stats_;

    // Threshold mode: move every linearization point to its belief mean and relinearize every factor, so
    // mu_ holds the beliefs (e.g. for writeTUM) and the frozen model is the linearization at the beliefs.
    // Not counted in threshold_stats_.
    void relinearizeAllAtBeliefs(FactorGraph& factor_graph);

  private:
    void performThresholdIteration(FactorGraph& factor_graph);
    void performSynchronousPhases(FactorGraph& factor_graph);
    void registerNewNodes(FactorGraph& factor_graph);
    void refreshDiagonalBlocks(FactorGraph& factor_graph);

    // Threshold-mode bookkeeping, indexed like variable_nodes_ / factor_nodes_ (both only grow)
    std::unordered_map<std::string, int> var_index_;
    std::vector<int> last_relin_round_;
    std::vector<Eigen::MatrixXd> D_;
    std::vector<char> D_dirty_;
    std::vector<std::vector<std::pair<int, int>>> var_factors_; // (factor index, edge position)
    std::vector<std::vector<int>> factor_vars_;
    size_t known_vars_    = 0;
    size_t known_factors_ = 0;
    bool planned_         = false; // a round has been planned and not yet run
    std::vector<char> planned_relin_;
    std::vector<int> planned_factors_;
  };

} // namespace gsolver
