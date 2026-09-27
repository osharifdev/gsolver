#pragma once

#include <gsolver/graph/edge_base.h>

namespace gsolver {

  template <typename VariableType_>
  struct Message {
    using VariableType = VariableType_;
    using EstimateType = typename VariableType::EstimateType;
    using MatrixLieDim = typename VariableType::MatrixLieDim;
    using VectorLieDim = typename VariableType::VectorLieDim;

    bool processed_ = false; // Flag indicating if the message has been processed
    MatrixLieDim Lambda_; // Lambda matrix (precision matrix of the factor)
    EstimateType mu_;     // Estimate of the variable

    // Frozen (linear) mode only: the message mean as tangent coordinates in the fixed chart of the
    // variable, i.e. relative to VariableNode::mu_lin_. Unused otherwise (see certify/FREEZE_MODE.md).
    VectorLieDim tau_ = VectorLieDim::Zero();
  };

  template <typename VariableType>
  class Edge : public EdgeBase {
  public:
    using VariableType_ = VariableType;
    using MessageType   = Message<VariableType>;

    MessageType variable_to_factor_message_;
    MessageType factor_to_variable_message_;

    bool frozen_ = false; // Frozen (linear) mode: messages on this edge live in tau_, not mu_
    bool asleep_ = false; // Threshold mode: the variable is asleep, its factor-to-variable message is not recomputed
  };

} // namespace gsolver