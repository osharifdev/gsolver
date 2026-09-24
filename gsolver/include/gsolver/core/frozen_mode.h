#pragma once

namespace gsolver {

  // Frozen (linear) mode, see certify/FREEZE_MODE.md.
  //
  // Values of G-solver's "+ 1e-6 * I" regularizers inside the frozen-mode branches only, one per site.
  // G-solver's default (unfrozen) code paths keep the literal 1e-6. Changing these values is a
  // diagnostic: it scales the regularizers whose effect on the frozen fixed point is measured in Phase A.
  inline double frozen_eps_belief = 1e-6; // variable_node.cpp updateMuAndSigma(): belief precision
  inline double frozen_eps_v2f    = 1e-6; // variable_node.cpp sendMessage(): variable-to-factor message
  inline double frozen_eps_marg   = 1e-6; // factor_node_helper.h marginalize(): conditioned bb block
  inline double frozen_eps_f2v    = 1e-6; // factor_node_helper.h computeMessage(): factor-to-variable message

  inline void setFrozenRegularizer(double eps) {
    frozen_eps_belief = frozen_eps_v2f = frozen_eps_marg = frozen_eps_f2v = eps;
  }

} // namespace gsolver
