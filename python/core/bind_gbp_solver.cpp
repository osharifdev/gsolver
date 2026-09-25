// =============================================================================
#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
// gsolver Python bindings - GbpSolver
// =============================================================================


#include <gsolver/core/frozen_mode.h>
#include <gsolver/core/gbp_solver.h>
#include <gsolver/graph/factor_graph.h>

namespace nb = nanobind;

namespace gsolver {
  namespace python {

    void bind_gbp_solver(nb::module_& m) {
      // SolverScheduleType enum
      nb::enum_<SolverScheduleType>(m, "SolverScheduleType", "Solver scheduling type")
        .value("SYNCHRONOUS", SolverScheduleType::SYNCHRONOUS, "Synchronous message passing")
        .value("ASYNCHRONOUS", SolverScheduleType::ASYNCHRONOUS, "Asynchronous message passing")
        .export_values();

      // GbpSolver class
      nb::class_<GbpSolver>(m, "GbpSolver", "Gaussian Belief Propagation Solver")
        .def(nb::init<SolverScheduleType>(), nb::arg("schedule_type"), "Create a GBP solver with the specified schedule type")
        .def_rw("schedule_type", &GbpSolver::schedule_type_, "Solver scheduling type")
        .def("perform_iteration",
             &GbpSolver::performIteration,
             nb::arg("factor_graph"),
             "Perform a single iteration of belief propagation")
        .def("perform_attention_iteration",
             &GbpSolver::performAttentionIteration,
             nb::arg("factor_graph"),
             nb::arg("attention_variable_ids"),
             nb::arg("attention_factor_ids"),
             "Perform an attention-weighted iteration")
        .def("solve", &GbpSolver::solve, nb::arg("factor_graph"), "Solve the factor graph until convergence")
        // Frozen (linear) mode and scheduling instrumentation, see certify/FREEZE_MODE.md
        .def_rw("dynamic_schedule", &GbpSolver::dynamic_schedule_, "Use the dynamically scheduled OpenMP loops (bit-identical)")
        .def("freeze",
             &GbpSolver::freeze,
             nb::arg("factor_graph"),
             nb::arg("factor_messages_valid") = true,
             "Freeze every factor's linearization at the current beliefs (frozen linear mode)");

      m.def("set_frozen_regularizer", &setFrozenRegularizer, nb::arg("eps"),
            "Set the four frozen-mode regularizers (G-solver ships 1e-6)");
      m.def("frozen_regularizers", []() {
        return std::make_tuple(frozen_eps_belief, frozen_eps_v2f, frozen_eps_marg, frozen_eps_f2v);
      }, "Current frozen-mode regularizers (belief, v2f, marg, f2v)");
    }

  } // namespace python
} // namespace gsolver
