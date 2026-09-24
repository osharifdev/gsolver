#pragma once

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
};
