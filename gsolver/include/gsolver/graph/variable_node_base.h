#pragma once

#include <Eigen/Dense>

#include <gsolver/graph/edge_base.h>

class VariableNodeBase {
public:
  std::string id_; // Variable node ID

  virtual void sendMessage(size_t outedge_id) = 0;
  virtual void sendMessage(const std::string& factor_id) = 0;
  virtual void sendAllMessages()              = 0;
  virtual void updateMuAndSigma()             = 0;

  // Frozen (linear) mode, see certify/FREEZE_MODE.md
  virtual void beginFreeze()                         = 0; // fix mu_lin_ = mu_; point incoming-to-factor means at it
  virtual void restoreMessageMeans()                 = 0; // undo beginFreeze() without freezing
  virtual void endFreeze(bool factor_messages_valid) = 0; // restore means, move messages to the fixed chart

  // Threshold relinearization (GbpSolver::threshold_relinearization_), see certify/THRESHOLD_RELIN.md
  virtual bool isFrozen() const                = 0;
  virtual void freezeAsNew()                   = 0; // new variable: mu_lin_ = mu_, z_ = 0, fresh edges frozen
  virtual void refreezeAtBelief()              = 0; // mu_lin_ <- belief mean, messages re-charted, z_ = 0
  virtual Eigen::VectorXd tangentState() const = 0; // z_ (the belief mean in the chart at mu_lin_)
  virtual void beginEvaluateAtBelief()         = 0; // point incoming-to-factor means at the belief (restoreMessageMeans undoes)
  virtual void setEdgesAsleep(bool asleep)     = 0; // mark this variable's edges: factors skip messages to it
};
