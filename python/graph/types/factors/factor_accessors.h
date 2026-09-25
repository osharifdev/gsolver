// =============================================================================
// gsolver Python bindings - read-only factor instrumentation (certify/, frozen mode)
// =============================================================================
#pragma once

#include <memory>

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

namespace gsolver {
  namespace python {

    // Read-only access to a factor's current linearization and its frozen linear model.
    template <typename FactorT, typename ClassT>
    void add_factor_accessors(ClassT& cls) {
      cls.def_prop_ro("J", [](const FactorT& f) { return Eigen::MatrixXd(f.J_); }, "Current Jacobian J_")
        .def_prop_ro("error", [](const FactorT& f) { return Eigen::VectorXd(f.error_); }, "Current error error_")
        .def_prop_ro("Sigma", [](const FactorT& f) { return Eigen::MatrixXd(f.Sigma_); }, "Factor covariance Sigma_")
        .def_prop_ro("J_lin", [](const FactorT& f) { return Eigen::MatrixXd(f.J_lin_); }, "Frozen Jacobian J_lin_")
        .def_prop_ro(
          "error_lin", [](const FactorT& f) { return Eigen::VectorXd(f.error_lin_); }, "Frozen error error_lin_")
        .def_prop_ro("frozen", [](const FactorT& f) { return f.frozen_; }, "Frozen (linear) mode flag");
    }

    // Typed edges of a factor whose variables all have the same type (messages in both directions).
    template <typename FactorT, int NumEdges, typename ClassT>
    void add_factor_edges(ClassT& cls) {
      cls.def(
        "edge",
        [](FactorT& f, size_t i) {
          if (i >= NumEdges)
            throw nanobind::index_error("edge index out of range");
          if constexpr (NumEdges == 1) {
            return f.edges_.template at<0>();
          } else {
            return i == 0 ? f.edges_.template at<0>() : f.edges_.template at<1>();
          }
        },
        nanobind::arg("index"),
        "Typed edge at the given slot");
    }

  } // namespace python
} // namespace gsolver
