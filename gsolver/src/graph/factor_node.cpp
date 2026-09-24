#include <gsolver/graph/factor_node.h>

namespace gsolver {

  template <int Dim_, typename... VariableTypes_>
  FactorNode<Dim_, VariableTypes_...>::FactorNode(const std::string& id, const MatrixErrorDim& Sigma) {
    id_          = id;
    Sigma_       = Sigma;
    eta_         = VectorVariablesDim::Zero();
    Lambdaprime_ = MatrixVariablesDim::Identity();
    error_       = VectorErrorDim::Zero();
    J_           = MatrixJacobianDim::Zero();
  }

  template <int Dim_, typename... VariableTypes_>
  void FactorNode<Dim_, VariableTypes_...>::updateFactor() {
    // Update the error and Jacobian
    if (frozen_) {
      // Frozen mode: the linear model at the fixed point, evaluated at the incoming message means
      VectorVariablesDim tau_in;
      edges_.template fillMessageTau<VariablesDim_>(tau_in, 0);
      error_ = error_lin_ + J_lin_ * tau_in;
      J_     = J_lin_;
    } else {
      updateErrorAndJacobian();
    }
    // Compute the k factor for robust estimation
    // double k = kRHuber(sqrt(error_.transpose() * Sigma_.inverse() * error_));
    double k = 1;
    // Update factor estimate (canonical form)
    eta_         = J_.transpose() * Sigma_.inverse() * (-error_) * k;
    Lambdaprime_ = J_.transpose() * Sigma_.inverse() * J_ * k;

    // Update Lambda by conditioning on the connected edges
    Lambdaprime_conditioned_ = Lambdaprime_;
    Lambdaprime_conditioned_ = edges_.template computeJointDistribution<VariablesDim_>(Lambdaprime_conditioned_);
  }

  template <int Dim_, typename... VariableTypes_>
  void FactorNode<Dim_, VariableTypes_...>::sendMessage(const std::string& variable_id) {
    // Send message from factor node to variable node
    edges_.template sendMessage<VariablesDim_>(variable_id, eta_, Lambdaprime_, Lambdaprime_conditioned_);
  }

  template <int Dim_, typename... VariableTypes_>
  void FactorNode<Dim_, VariableTypes_...>::sendAllMessages() {
    // iterate over all edges and send messages to all connected variable nodes
    edges_.template sendAllMessages<VariablesDim_>(eta_, Lambdaprime_, Lambdaprime_conditioned_);
  }

  // Frozen (linear) mode, see certify/FREEZE_MODE.md -------------------------------------------------

  template <int Dim_, typename... VariableTypes_>
  void FactorNode<Dim_, VariableTypes_...>::freezeLinearization() {
    // VariableNode::beginFreeze() has pointed every incoming message mean at the variable's fixed
    // linearization point, so the factor's own error/Jacobian code is evaluated at that point.
    updateErrorAndJacobian();
    error_lin_ = error_;
    J_lin_     = J_;
    frozen_    = true;
  }

  template <int Dim_, typename... VariableTypes_>
  void FactorNode<Dim_, VariableTypes_...>::freezeNewFactor() {
    // A factor created after the freeze: its edges were initialized by createEdge() with the variables'
    // mu_ (== mu_lin_, never remapped in frozen mode), so it linearizes at the fixed point directly.
    freezeLinearization();
    edges_.markEdgesFrozen();
  }

} // namespace gsolver
