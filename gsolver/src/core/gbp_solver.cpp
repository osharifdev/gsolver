#include <gsolver/core/gbp_solver.h>

#include <algorithm>
#include <cmath>

namespace gsolver {

  GbpSolver::GbpSolver(SolverScheduleType schedule_type) : schedule_type_(schedule_type) {
  }

  void GbpSolver::performIteration(FactorGraph& factor_graph) {
    if (threshold_relinearization_) {
      performThresholdIteration(factor_graph);
      return;
    }
    if (schedule_type_ == SolverScheduleType::SYNCHRONOUS && dynamic_schedule_) {
      // Same four phases as below, dynamically scheduled (performance only, bit-identical results)
#pragma omp parallel for schedule(dynamic, 16)
      for (auto& factor_node : factor_graph.factor_nodes_) {
        factor_node->updateFactor();
      }
#pragma omp parallel for schedule(dynamic, 16)
      for (auto& factor_node : factor_graph.factor_nodes_) {
        factor_node->sendAllMessages();
      }
#pragma omp parallel for schedule(dynamic, 16)
      for (auto& variable_node : factor_graph.variable_nodes_) {
        variable_node->updateMuAndSigma();
      }
#pragma omp parallel for schedule(dynamic, 16)
      for (auto& variable_node : factor_graph.variable_nodes_) {
        variable_node->sendAllMessages();
      }
      return;
    }
    if (schedule_type_ == SolverScheduleType::SYNCHRONOUS) {
// Update factors
// std::cout << "Update Factors" << std::endl;
#pragma omp parallel for
      for (auto& factor_node : factor_graph.factor_nodes_) {
        factor_node->updateFactor();
      }
// Factor-to-variable messages
// std::cout << "Factor Messages" << std::endl;
#pragma omp parallel for
      for (auto& factor_node : factor_graph.factor_nodes_) {
        factor_node->sendAllMessages();
      }

// Update variables
// std::cout << "Update Variables" << std::endl;s
#pragma omp parallel for
      for (auto& variable_node : factor_graph.variable_nodes_) {
        variable_node->updateMuAndSigma();
      }

// Variable-to-factor messages
// std::cout << "Variable Messages" << std::endl;
#pragma omp parallel for
      for (auto& variable_node : factor_graph.variable_nodes_) {
        variable_node->sendAllMessages();
      }
    } else if (schedule_type_ == SolverScheduleType::ASYNCHRONOUS) {
      // Implement asynchronous schedule (e.g., random node selection)
      // ...
    }
  }

  void GbpSolver::performAttentionIteration(FactorGraph& factor_graph,
                                            const std::vector<std::string>& attention_variables_ids,
                                            const std::vector<std::string>& attention_factors_ids) {
    if (schedule_type_ == SolverScheduleType::SYNCHRONOUS) {
      // Update factors
      for (std::string factor_id : attention_factors_ids) {
        factor_graph.getFactorNode(factor_id)->updateFactor();
      }

      // Factor-to-variable messages
      for (std::string factor_id : attention_factors_ids) {
        factor_graph.getFactorNode(factor_id)->sendAllMessages();
      }

      // Update variables
      for (std::string variable_id : attention_variables_ids) {
        factor_graph.getVariableNode(variable_id)->updateMuAndSigma();
      }

      // Variable-to-factor messages
      for (std::string variable_id : attention_variables_ids) {
        factor_graph.getVariableNode(variable_id)->sendAllMessages();
      }
    } else if (schedule_type_ == SolverScheduleType::ASYNCHRONOUS) {
      // Implement asynchronous schedule with attention
      // ...
    }
  }

  void GbpSolver::freeze(FactorGraph& factor_graph, bool factor_messages_valid) {
    // 1. fix mu_lin_ = mu_ for every variable and point its outgoing message means at it
    for (auto& variable_node : factor_graph.variable_nodes_) {
      variable_node->beginFreeze();
    }
    // 2. every factor evaluates its own error and Jacobian at the fixed points and becomes linear
    for (auto& factor_node : factor_graph.factor_nodes_) {
      factor_node->freezeLinearization();
    }
    // 3. restore the message means and move all messages into the fixed charts
    for (auto& variable_node : factor_graph.variable_nodes_) {
      variable_node->endFreeze(factor_messages_valid);
    }
  }

  // Threshold relinearization, see certify/THRESHOLD_RELIN.md ----------------------------------------

  void GbpSolver::registerNewNodes(FactorGraph& factor_graph) {
    // Variables and factors are only ever appended. New variables are frozen at their current mean (their
    // linearization point is the initial guess), then new factors linearize at their variables' points.
    auto& st = threshold_stats_;
    for (size_t v = known_vars_; v < factor_graph.variable_nodes_.size(); ++v) {
      auto& node = factor_graph.variable_nodes_[v];
      if (!node->isFrozen())
        node->freezeAsNew();
      var_index_[node->id_] = static_cast<int>(v);
      last_relin_round_.push_back(st.round);
      D_.emplace_back();
      D_dirty_.push_back(1);
      var_factors_.emplace_back();
      st.displacement.push_back(0.0);
      st.relinearizations.push_back(0);
      st.created_round.push_back(st.round);
    }
    known_vars_ = factor_graph.variable_nodes_.size();
    for (size_t f = known_factors_; f < factor_graph.factor_nodes_.size(); ++f) {
      auto& node = factor_graph.factor_nodes_[f];
      if (!node->isFrozen())
        node->freezeNewFactor();
      std::vector<int> vars;
      for (size_t k = 0; k < node->numberOfEdges(); ++k) {
        const int v = var_index_.at(node->edgeVariableId(k));
        vars.push_back(v);
        var_factors_[v].emplace_back(static_cast<int>(f), static_cast<int>(k));
        D_dirty_[v] = 1;
      }
      factor_vars_.push_back(std::move(vars));
    }
    known_factors_ = factor_graph.factor_nodes_.size();
  }

  void GbpSolver::refreshDiagonalBlocks(FactorGraph& factor_graph) {
    // D_v = H_vv = sum over the factors of v of J_lin,vᵀ Σ⁻¹ J_lin,v (the regularizers are not part of H)
    std::vector<int> dirty;
    for (size_t v = 0; v < D_dirty_.size(); ++v)
      if (D_dirty_[v])
        dirty.push_back(static_cast<int>(v));
#pragma omp parallel for schedule(dynamic, 64)
    for (size_t k = 0; k < dirty.size(); ++k) {
      const int v = dirty[k];
      Eigen::MatrixXd D;
      for (const auto& [f, e] : var_factors_[v]) {
        const Eigen::MatrixXd B = factor_graph.factor_nodes_[f]->linearizedInformationBlock(e);
        if (D.size() == 0)
          D = B;
        else
          D += B;
      }
      D_[v]       = D;
      D_dirty_[v] = 0;
    }
  }

  int GbpSolver::planThresholdRound(FactorGraph& factor_graph) {
    auto& st = threshold_stats_;
    if (planned_)
      return st.new_factors + static_cast<int>(planned_factors_.size());
    st.round++;
    st.relinearized_vars.clear();
    st.relinearized_factors = 0;

    const size_t factors_before = known_factors_;
    registerNewNodes(factor_graph);
    st.new_factors = static_cast<int>(known_factors_ - factors_before);
    const bool hblock = (relinearize_metric_ == RelinMetric::HBlock);
    if (hblock)
      refreshDiagonalBlocks(factor_graph);

    // Displacement since linearization and the threshold test
    const size_t n = factor_graph.variable_nodes_.size();
    planned_relin_.assign(n, 0);
#pragma omp parallel for schedule(static)
    for (size_t v = 0; v < n; ++v) {
      const Eigen::VectorXd z = factor_graph.variable_nodes_[v]->tangentState();
      double d                = 0.0;
      if (!hblock)
        d = z.norm();
      else if (D_[v].size() != 0)
        d = std::sqrt(std::max(0.0, z.dot(D_[v] * z)));
      st.displacement[v] = d;
      planned_relin_[v]  = (st.round - last_relin_round_[v] >= relinearize_every_) && (d > relinearize_threshold_);
    }
    std::vector<char> relin_factor(factor_graph.factor_nodes_.size(), 0);
    for (size_t v = 0; v < n; ++v)
      if (planned_relin_[v])
        for (const auto& fe : var_factors_[v])
          relin_factor[fe.first] = 1;
    planned_factors_.clear();
    for (size_t f = 0; f < relin_factor.size(); ++f)
      if (relin_factor[f])
        planned_factors_.push_back(static_cast<int>(f));
    planned_ = true;
    return st.new_factors + static_cast<int>(planned_factors_.size());
  }

  void GbpSolver::performThresholdIteration(FactorGraph& factor_graph) {
    planThresholdRound(factor_graph);
    auto& st = threshold_stats_;

    // Re-freeze the moved variables at their beliefs, then relinearize every factor touching them
    for (size_t v = 0; v < planned_relin_.size(); ++v) {
      if (!planned_relin_[v])
        continue;
      factor_graph.variable_nodes_[v]->refreezeAtBelief();
      last_relin_round_[v] = st.round;
      st.relinearizations[v]++;
      st.relinearized_vars.push_back(static_cast<int>(v));
    }
    const auto& factors = planned_factors_;
#pragma omp parallel for schedule(dynamic, 64)
    for (size_t k = 0; k < factors.size(); ++k)
      factor_graph.factor_nodes_[factors[k]]->freezeLinearization();
    for (int f : factors)
      for (int v : factor_vars_[f])
        D_dirty_[v] = 1;
    st.relinearized_factors = static_cast<int>(factors.size());
    planned_                = false;

    performSynchronousPhases(factor_graph);
  }

  void GbpSolver::relinearizeAllAtBeliefs(FactorGraph& factor_graph) {
    registerNewNodes(factor_graph);
    for (auto& variable_node : factor_graph.variable_nodes_)
      variable_node->refreezeAtBelief();
#pragma omp parallel for schedule(dynamic, 64)
    for (size_t f = 0; f < factor_graph.factor_nodes_.size(); ++f)
      factor_graph.factor_nodes_[f]->freezeLinearization();
    std::fill(D_dirty_.begin(), D_dirty_.end(), 1);
    planned_ = false;
  }

  void GbpSolver::performSynchronousPhases(FactorGraph& factor_graph) {
    // The four phases of performIteration(), on the frozen linear model
    if (dynamic_schedule_) {
#pragma omp parallel for schedule(dynamic, 16)
      for (auto& factor_node : factor_graph.factor_nodes_) {
        factor_node->updateFactor();
      }
#pragma omp parallel for schedule(dynamic, 16)
      for (auto& factor_node : factor_graph.factor_nodes_) {
        factor_node->sendAllMessages();
      }
#pragma omp parallel for schedule(dynamic, 16)
      for (auto& variable_node : factor_graph.variable_nodes_) {
        variable_node->updateMuAndSigma();
      }
#pragma omp parallel for schedule(dynamic, 16)
      for (auto& variable_node : factor_graph.variable_nodes_) {
        variable_node->sendAllMessages();
      }
      return;
    }
#pragma omp parallel for
    for (auto& factor_node : factor_graph.factor_nodes_) {
      factor_node->updateFactor();
    }
#pragma omp parallel for
    for (auto& factor_node : factor_graph.factor_nodes_) {
      factor_node->sendAllMessages();
    }
#pragma omp parallel for
    for (auto& variable_node : factor_graph.variable_nodes_) {
      variable_node->updateMuAndSigma();
    }
#pragma omp parallel for
    for (auto& variable_node : factor_graph.variable_nodes_) {
      variable_node->sendAllMessages();
    }
  }

  void GbpSolver::solve(FactorGraph& factor_graph) {
    // TODO: Implement stopping criteria (e.g., convergence threshold, maximum iterations)
    while (true) { // FIXME: This is a bug, and should be replaced with a proper stopping criterion
      performIteration(factor_graph);
    }
  }

} // namespace gsolver