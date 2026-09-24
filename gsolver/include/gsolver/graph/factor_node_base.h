#pragma once

#include <memory>
#include <string>

#include <gsolver/graph/edge_base.h>

namespace gsolver {

  class FactorNodeBase {
  public:
    std::string id_; // Factor node ID

    virtual ~FactorNodeBase() = default;

    virtual void setEdgeAt(size_t i, std::shared_ptr<EdgeBase> edge) = 0; // Set edge at index i
    virtual void updateFactor()                                      = 0; // Update the factor based on the connected edges
    virtual void sendMessage(const std::string& variable_id)         = 0;
    virtual void sendAllMessages()                                   = 0;

    // Frozen (linear) mode, see certify/FREEZE_MODE.md
    virtual void freezeLinearization() = 0; // store error and Jacobian at the incoming means, go linear
    virtual void freezeNewFactor()     = 0; // freeze a factor added after the graph was frozen
  };

} // namespace gsolver
