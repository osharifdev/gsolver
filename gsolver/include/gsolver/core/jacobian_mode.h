#pragma once

namespace gsolver {

  // Stage 1 (certify/TRUE_JACOBIANS.md): with true_jacobians set, the stereo and GP factors use the true derivative of
  // their error along the solver's charts: the stereo rotation block without G-solver's factor 2, and the GP pose error's
  // derivative with respect to the first knot's velocity for the decoupled Exp/Log of geometry3d.h. Default false:
  // G-solver's shipped Jacobians, bit-identical.
  inline bool true_jacobians = false;

} // namespace gsolver
