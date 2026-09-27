#pragma once

#include <Eigen/Dense>
#include <memory>
#include <string>

#include <gsolver/graph/edge_base.h>

namespace gsolver {

  class FactorNodeBase {
  public:
    std::string id_; // Factor node ID
    int flop_class_ = 0; // Phase D, WP6a: class index of this factor in the flop tallies (gsolver/core/flop_count.h)

    virtual ~FactorNodeBase() = default;

    virtual void setEdgeAt(size_t i, std::shared_ptr<EdgeBase> edge) = 0; // Set edge at index i
    virtual void updateFactor()                                      = 0; // Update the factor based on the connected edges
    virtual void sendMessage(const std::string& variable_id)         = 0;
    virtual void sendAllMessages()                                   = 0;

    // Frozen (linear) mode, see certify/FREEZE_MODE.md
    virtual void freezeLinearization() = 0; // store error and Jacobian at the incoming means, go linear
    virtual void freezeNewFactor()     = 0; // freeze a factor added after the graph was frozen

    // Threshold relinearization (GbpSolver::threshold_relinearization_), see certify/THRESHOLD_RELIN.md
    virtual bool isFrozen() const                                = 0;
    virtual size_t numberOfEdges() const                         = 0;
    virtual std::string edgeVariableId(size_t i)                 = 0;
    virtual Eigen::MatrixXd linearizedInformationBlock(size_t i) = 0; // J_lin,iᵀ Σ⁻¹ J_lin,i for edge i
  };

} // namespace gsolver
