#include <gsolver/core/frozen_mode.h>
#include <gsolver/graph/variable_node.h>

namespace gsolver {

  template <int Dim_, typename EstimateType_, typename EdgeType_>
  VariableNode<Dim_, EstimateType_, EdgeType_>::VariableNode(const std::string& id,
                                                             const EstimateType_& mu,
                                                             const MatrixLieDim& Sigma) {
    id_    = id;
    mu_    = mu;
    Sigma_ = Sigma;
  }

  template <int Dim_, typename EstimateType_, typename EdgeType_>
  void VariableNode<Dim_, EstimateType_, EdgeType_>::sendMessage(size_t outedge_id) {
    VectorLieDim tangent_eta         = VectorLieDim::Zero();
    MatrixLieDim tangent_Lambdaprime = MatrixLieDim::Zero();
    VectorLieDim tangent_delta_mu    = VectorLieDim::Zero();

    // If there are no factors connected to the variable node, skip the message computation
    if (edges_.size() == 0) {
      return;
    }

    // If there are not enough factors skip the message computation
    if (edges_.size() == 1) {
      return;
    }

    // Multiply inward messages from other factors
    for (size_t i = 0; i < edges_.size(); ++i) {
      if (i == outedge_id) {
        // Skip the outgoing edge
        continue;
      }

      // Access messages from edges (TODO: replace with appropriate getter methods based on your edge structure)
      const MatrixLieDim& Lambdaprime_inward = edges_[i]->factor_to_variable_message_.Lambda_;

      // Project to the tangent space (frozen mode: the message mean already is a coordinate in the fixed chart)
      VectorLieDim tangent_delta_mu_inward;
      if (frozen_) {
        tangent_delta_mu_inward = edges_[i]->factor_to_variable_message_.tau_ - z_;
      } else {
        const EstimateType_ mu_inward = edges_[i]->factor_to_variable_message_.mu_;
        tangent_delta_mu_inward       = boxminus(mu_inward);
      }

      // Compute the Lambda and eta
      MatrixLieDim tangent_Lambda_inward = Lambdaprime_inward;
      VectorLieDim tangent_eta_inward    = tangent_Lambda_inward * tangent_delta_mu_inward;

      tangent_eta += tangent_eta_inward;
      tangent_Lambdaprime += tangent_Lambda_inward;
    }
    // Ensure that the tangent_Lambdaprime is invertible
    tangent_Lambdaprime += MatrixLieDim::Identity() * (frozen_ ? frozen_eps_v2f : 1e-6); // Add a small perturbation to make it invertible
    // get the perturbation to be applied to the current estimate
    tangent_delta_mu = tangent_Lambdaprime.inverse() * tangent_eta;

    if (frozen_) {
      // Frozen mode: the outgoing mean stays in the fixed chart (no boxplus remap)
      edges_[outedge_id]->variable_to_factor_message_.Lambda_ = tangent_Lambdaprime;
      edges_[outedge_id]->variable_to_factor_message_.tau_    = z_ + tangent_delta_mu;
      return;
    }

    // compute the new estimate in the manifold
    EstimateType_ mu_outward         = boxplus(tangent_delta_mu);
    MatrixLieDim Lambdaprime_outward = tangent_Lambdaprime;

    edges_[outedge_id]->variable_to_factor_message_.Lambda_ = Lambdaprime_outward;
    edges_[outedge_id]->variable_to_factor_message_.mu_     = mu_outward;
  }

  template <int Dim_, typename EstimateType_, typename EdgeType_>
  void VariableNode<Dim_, EstimateType_, EdgeType_>::sendMessage(const std::string& factor_id) {
    VectorLieDim tangent_eta         = VectorLieDim::Zero();
    MatrixLieDim tangent_Lambdaprime = MatrixLieDim::Zero();
    VectorLieDim tangent_delta_mu    = VectorLieDim::Zero();

    // If there are no factors connected to the variable node, skip the message computation
    if (edges_.size() == 0) {
      return;
    }

    // If there are not enough factors skip the message computation
    if (edges_.size() == 1) {
      return;
    }

    // Find the outgoing edge index
    const auto it = std::find_if(edges_.begin(), edges_.end(), [&](const auto& edge) {
      return edge->factor_id_ == factor_id;
    });
    if (it == edges_.end()) {
      throw std::runtime_error("Edge with the given factor_id not found in VariableNode::sendMessage");
    }
    const size_t outedge_id = static_cast<size_t>(std::distance(edges_.begin(), it));

    // Multiply inward messages from other factors
    for (size_t i = 0; i < edges_.size(); ++i) {
      if (i == outedge_id) {
        // Skip the outgoing edge
        continue;
      }

      // Access messages from edges (TODO: replace with appropriate getter methods based on your edge structure)
      const MatrixLieDim& Lambdaprime_inward = edges_[i]->factor_to_variable_message_.Lambda_;

      // Project to the tangent space (frozen mode: the message mean already is a coordinate in the fixed chart)
      VectorLieDim tangent_delta_mu_inward;
      if (frozen_) {
        tangent_delta_mu_inward = edges_[i]->factor_to_variable_message_.tau_ - z_;
      } else {
        const EstimateType_ mu_inward = edges_[i]->factor_to_variable_message_.mu_;
        tangent_delta_mu_inward       = boxminus(mu_inward);
      }

      // Compute the Lambda and eta
      MatrixLieDim tangent_Lambda_inward = Lambdaprime_inward;
      VectorLieDim tangent_eta_inward    = tangent_Lambda_inward * tangent_delta_mu_inward;

      tangent_eta += tangent_eta_inward;
      tangent_Lambdaprime += tangent_Lambda_inward;
    }
    // Ensure that the tangent_Lambdaprime is invertible
    tangent_Lambdaprime += MatrixLieDim::Identity() * (frozen_ ? frozen_eps_v2f : 1e-6); // Add a small perturbation to make it invertible
    // get the perturbation to be applied to the current estimate
    tangent_delta_mu = tangent_Lambdaprime.inverse() * tangent_eta;

    if (frozen_) {
      // Frozen mode: the outgoing mean stays in the fixed chart (no boxplus remap)
      edges_[outedge_id]->variable_to_factor_message_.Lambda_ = tangent_Lambdaprime;
      edges_[outedge_id]->variable_to_factor_message_.tau_    = z_ + tangent_delta_mu;
      return;
    }

    // compute the new estimate in the manifold
    EstimateType_ mu_outward         = boxplus(tangent_delta_mu);
    MatrixLieDim Lambdaprime_outward = tangent_Lambdaprime;

    edges_[outedge_id]->variable_to_factor_message_.Lambda_ = Lambdaprime_outward;
    edges_[outedge_id]->variable_to_factor_message_.mu_     = mu_outward;
  }

  template <int Dim_, typename EstimateType_, typename EdgeType_>
  void VariableNode<Dim_, EstimateType_, EdgeType_>::sendAllMessages() {
#pragma omp parallel for
    for (size_t i = 0; i < edges_.size(); ++i) {
      sendMessage(i);
    }
  }

  template <int Dim_, typename EstimateType_, typename EdgeType_>
  void VariableNode<Dim_, EstimateType_, EdgeType_>::updateMuAndSigma() {
    VectorLieDim tangent_eta         = VectorLieDim::Zero();
    MatrixLieDim tangent_Lambdaprime = MatrixLieDim::Zero();
    VectorLieDim tangent_delta_mu    = VectorLieDim::Zero();

    // If there are no factors connected to the variable node, skip the update
    if (edges_.size() == 0) {
      return;
    }

    // Multiply inward messages from other factors
    for (size_t i = 0; i < edges_.size(); ++i) {
      // Access messages from edges (replace with appropriate getter methods based on your edge structure)
      const MatrixLieDim& Lambdaprime_inward = edges_[i]->factor_to_variable_message_.Lambda_;

      // Handle the pose (frozen mode: the message mean already is a coordinate in the fixed chart)
      VectorLieDim tangent_delta_mu_inward;
      if (frozen_) {
        tangent_delta_mu_inward = edges_[i]->factor_to_variable_message_.tau_ - z_;
      } else {
        const EstimateType_ mu_inward = edges_[i]->factor_to_variable_message_.mu_;
        tangent_delta_mu_inward       = boxminus(mu_inward);
      }

      // Compute the Lambda and eta
      MatrixLieDim tangent_Lambda_inward = Lambdaprime_inward;
      VectorLieDim tangent_eta_inward    = tangent_Lambda_inward * tangent_delta_mu_inward;

      tangent_eta += tangent_eta_inward;
      tangent_Lambdaprime += tangent_Lambda_inward;
    }
    // Ensure that the tangent_Lambdaprime is invertible
    tangent_Lambdaprime += MatrixLieDim::Identity() * (frozen_ ? frozen_eps_belief : 1e-6); // Add a small perturbation to make it invertible
    // get the perturbation to be applied to the current estimate
    tangent_delta_mu = tangent_Lambdaprime.inverse() * tangent_eta;

    if (frozen_) {
      // Frozen mode: the state is z_ in the fixed chart; mu_ is not remapped (it stays at mu_lin_)
      z_                       = z_ + tangent_delta_mu;
      MatrixLieDim Lambdaprime = tangent_Lambdaprime;
      Sigma_                   = Lambdaprime.inverse();
      return;
    }

    // compute and update the estimate in the manifold
    mu_                      = boxplus(tangent_delta_mu);
    MatrixLieDim Lambdaprime = tangent_Lambdaprime;
    Sigma_                   = Lambdaprime.inverse();
  }

  // Frozen (linear) mode, see certify/FREEZE_MODE.md -------------------------------------------------

  template <int Dim_, typename EstimateType_, typename EdgeType_>
  void VariableNode<Dim_, EstimateType_, EdgeType_>::beginFreeze() {
    // The fixed linearization point is the belief mean at the freeze instant. Every factor reads its
    // linearization point from the incoming variable-to-factor message mean, so those means are pointed
    // at mu_lin_ while the factors linearize; restoreMessageMeans() puts the original means back.
    mu_lin_ = mu_;
    mu_msg_stash_.clear();
    mu_msg_stash_.reserve(edges_.size());
    for (auto& edge : edges_) {
      mu_msg_stash_.push_back(edge->variable_to_factor_message_.mu_);
      edge->variable_to_factor_message_.mu_ = mu_lin_;
    }
  }

  template <int Dim_, typename EstimateType_, typename EdgeType_>
  void VariableNode<Dim_, EstimateType_, EdgeType_>::restoreMessageMeans() {
    for (size_t i = 0; i < edges_.size(); ++i) {
      edges_[i]->variable_to_factor_message_.mu_ = mu_msg_stash_[i];
    }
    mu_msg_stash_.clear();
  }

  template <int Dim_, typename EstimateType_, typename EdgeType_>
  void VariableNode<Dim_, EstimateType_, EdgeType_>::endFreeze(bool factor_messages_valid) {
    restoreMessageMeans();
    // One-time chart change of every stored message mean into the fixed chart at mu_lin_ (== mu_ here).
    // Precisions are kept as they are: G-solver never transports precisions between charts.
    for (auto& edge : edges_) {
      edge->variable_to_factor_message_.tau_ = boxminus(edge->variable_to_factor_message_.mu_);
      edge->factor_to_variable_message_.tau_ =
        factor_messages_valid ? boxminus(edge->factor_to_variable_message_.mu_) : VectorLieDim::Zero();
      edge->frozen_ = true;
    }
    z_      = VectorLieDim::Zero();
    frozen_ = true;
  }

  // Threshold relinearization, see certify/THRESHOLD_RELIN.md ----------------------------------------

  template <int Dim_, typename EstimateType_, typename EdgeType_>
  void VariableNode<Dim_, EstimateType_, EdgeType_>::freezeAsNew() {
    // A variable that has not taken part in any threshold round. Its edges are new (createEdge set their
    // incoming-to-factor means to mu_), so every message mean sits at the fixed point: tau_ = 0. Factor
    // messages on them are written by the factors before any variable reads them.
    mu_lin_ = mu_;
    for (auto& edge : edges_) {
      edge->variable_to_factor_message_.mu_  = mu_lin_;
      edge->variable_to_factor_message_.tau_ = VectorLieDim::Zero();
      edge->factor_to_variable_message_.tau_ = VectorLieDim::Zero();
      edge->frozen_                          = true;
    }
    z_      = VectorLieDim::Zero();
    frozen_ = true;
  }

  template <int Dim_, typename EstimateType_, typename EdgeType_>
  void VariableNode<Dim_, EstimateType_, EdgeType_>::refreezeAtBelief() {
    // The fixed point moves to the belief mean mu_lin_ ⊞ z_ (mu_ == mu_lin_ while frozen). Every stored
    // message mean keeps its manifold point and is re-expressed in the new chart. Precisions are kept as
    // stored, as in endFreeze(). Incoming-to-factor means point at the new mu_lin_, where the factors
    // touching this variable relinearize (FactorNode::freezeLinearization reads them).
    std::vector<EstimateType> v2f, f2v;
    v2f.reserve(edges_.size());
    f2v.reserve(edges_.size());
    for (auto& edge : edges_) {
      v2f.push_back(boxplus(edge->variable_to_factor_message_.tau_));
      f2v.push_back(boxplus(edge->factor_to_variable_message_.tau_));
    }
    mu_     = boxplus(z_);
    mu_lin_ = mu_;
    for (size_t i = 0; i < edges_.size(); ++i) {
      edges_[i]->variable_to_factor_message_.tau_ = boxminus(v2f[i]);
      edges_[i]->factor_to_variable_message_.tau_ = boxminus(f2v[i]);
      edges_[i]->variable_to_factor_message_.mu_  = mu_lin_;
    }
    z_ = VectorLieDim::Zero();
  }

  template <int Dim_, typename EstimateType_, typename EdgeType_>
  EstimateType_ VariableNode<Dim_, EstimateType_, EdgeType_>::beliefMean() {
    return frozen_ ? boxplus(z_) : mu_;
  }

  template <int Dim_, typename EstimateType_, typename EdgeType_>
  typename VariableNode<Dim_, EstimateType_, EdgeType_>::VectorLieDim
  VariableNode<Dim_, EstimateType_, EdgeType_>::offsetFromBelief(const EstimateType_& point) {
    if (!frozen_)
      return boxminus(point);
    const EstimateType_ lin    = mu_;
    const EstimateType_ belief = boxplus(z_);
    mu_                        = belief; // boxminus() is taken at mu_
    const VectorLieDim tau     = boxminus(point);
    mu_                        = lin;
    return tau;
  }

  template <int Dim_, typename EstimateType_, typename EdgeType_>
  typename VariableNode<Dim_, EstimateType_, EdgeType_>::VectorLieDim
  VariableNode<Dim_, EstimateType_, EdgeType_>::cavityOffsetFromBelief(size_t i) {
    // Not frozen: the stash of beginEvaluateAtBelief holds the solver's own incoming-to-factor means (the cavity
    // means). Frozen: the cavity mean is the message's tangent coordinate in the fixed chart.
    if (!frozen_)
      return boxminus(mu_msg_stash_[i]);
    return offsetFromBelief(boxplus(edges_[i]->variable_to_factor_message_.tau_));
  }

  template <int Dim_, typename EstimateType_, typename EdgeType_>
  void VariableNode<Dim_, EstimateType_, EdgeType_>::beginEvaluateAtBelief() {
    if (!frozen_) {
      beginFreeze(); // the belief is mu_; mu_lin_ is unused while not frozen
      return;
    }
    const EstimateType_ belief = boxplus(z_);
    mu_msg_stash_.clear();
    mu_msg_stash_.reserve(edges_.size());
    for (auto& edge : edges_) {
      mu_msg_stash_.push_back(edge->variable_to_factor_message_.mu_);
      edge->variable_to_factor_message_.mu_ = belief;
    }
  }

} // namespace gsolver
