/**
 * @file certify_chain.cpp
 * @brief Fact F4 check: a pure GP chain (a tree), frozen, run to convergence, exported with the
 *        messages on every GP edge.
 *
 * Graph construction replicates examples/cpp/apps/prior_experiment.cpp: pose-velocity variables,
 * absolute pose priors from the file, GP priors between consecutive poses.
 *
 * Usage: certify_chain --g2o PATH --config KEY --out DIR [--normal-iters N] [--frozen-iters T]
 */

#include <iostream>
#include <string>

#include "common/data_types.h"
#include "common/experiment_config.h"
#include "common/factor_graph_builder.h"
#include "common/g2o_parser.h"

#include "certify_common.h"

using namespace gsolver;
using namespace examples;

int main(int argc, char** argv) {
  try {
    std::string g2o, config, out;
    int normal_iters = 100, frozen_iters = 3000;
    for (int i = 1; i < argc; ++i) {
      std::string k = argv[i];
      if (i + 1 >= argc)
        throw std::runtime_error("missing value for " + k);
      std::string v = argv[++i];
      if (k == "--g2o") g2o = v;
      else if (k == "--config") config = v;
      else if (k == "--out") out = v;
      else if (k == "--normal-iters") normal_iters = std::stoi(v);
      else if (k == "--frozen-iters") frozen_iters = std::stoi(v);
      else throw std::runtime_error("unknown argument " + k);
    }
    auto params = loadExperimentParams(getDefaultConfigPath(), config);

    std::vector<PoseInit> poses;
    std::vector<OdometryMeas> odometry;
    std::vector<PriorMeas> priors;
    std::vector<LandmarkMeas> landmarks;
    parseG2OFile(g2o, poses, odometry, priors, landmarks);

    FactorGraph graph;
    add_pose_variable_nodes(graph, poses);
    add_prior_factor_nodes(graph, priors);
    add_gp_prior_factor_nodes(graph, params.qc_diag);

    GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
    for (int i = 0; i < normal_iters; ++i)
      solver.performIteration(graph);

    certify::Json meta;
    meta.str("dataset", "chain").str("g2o", g2o).str("config", config).integer("normal_iterations_before_freeze", normal_iters);
    meta.set("qc_diag", certify::jsonArray(std::vector<double>(params.qc_diag.data(), params.qc_diag.data() + 6)));
    solver.freeze(graph, normal_iters > 0);
    certify::makeDirs(out);
    const double secs = certify::runFrozen(graph, solver, frozen_iters, out);
    // export after the frozen run, so the stored messages are the converged ones
    certify::exportModel(graph, out, certify::Charts::Frozen, meta);
    std::cout << "chain: " << poses.size() << " poses, " << frozen_iters << " frozen iterations in " << secs << " s" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 2;
  }
}
