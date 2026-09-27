/**
 * @file certify_stereo.cpp
 * @brief Phase A driver for KITTI stereo BA: normal run, snapshot, freeze, export.
 *
 * The graph construction replicates examples/cpp/apps/stereo_visual.cpp for one camera; --keyframes L
 * keeps only the first L keyframes (and the observations and landmarks they carry), exactly as the
 * shipped driver builds the window of one robot. With L = 0 (all) and --normal-iters 10, the trajectory
 * written by --tum is byte-identical to stereo_visual's output.
 *
 * Usage:
 *   certify_stereo --g2o PATH --gt PATH --out DIR [options]
 *     --config KEY                 experiment_params.yaml key (default kitti_05, as in the paper)
 *     --keyframes L                first L keyframes only (0 = all)
 *     --normal-iters N             normal (relinearizing) iterations before the snapshot (default 10)
 *     --converge TOL               instead: iterate until max_i s_i < TOL, s_i = belief-mean step in the
 *                                  belief precision metric, capped by --normal-max (default 200)
 *     --tum PATH                   GP-interpolate at the GT timestamps and write TUM (no freeze)
 *     --unfrozen-export-iters L    comma list of normal-run iterations to export (charts at beliefs)
 *     --dynamic-schedule           GbpSolver::dynamic_schedule_ (performance only, bit-identical)
 *     --freeze                     freeze at the snapshot, export, run frozen iterations
 *     --anchor-precision P         frozen model only: SE(3) prior on the first keyframe at its initial
 *                                  ORB-SLAM3 pose with precision P*I (A3: 1e4); 0 = none
 *     --frozen-iters T             frozen iterations (default 1000)
 *     --frozen-eps E               frozen-mode regularizer (default 1e-6, as in G-solver)
 *     --frozen-eps-sites SPEC      diagnostic per-site override, e.g. "marg=1e-7,f2v=1e-6"
 *     --dump-stride S              state dump: every iteration up to 1000, then every S-th, and the last
 *     --relin-threshold D          Phase D, WP1: threshold relinearization instead of the per-round schedule,
 *                                  d_v = ||z_v||_{H_vv} > D (certify/THRESHOLD_RELIN.md); runs --normal-iters
 *                                  rounds, writes relin_rounds.csv and relin_counts.csv
 *     --relin-every R              at most one re-freeze per variable per R rounds (default 1)
 *     --final-export DIR           export the linearization at the final belief means (either schedule)
 *     --relin-metric physical|D    metric of d_v: Euclidean tangent norm (default, revised D-a) or ||z_v||_{H_vv}
 *     --work-budget W              export the state after the last round with cumulative work <= W to
 *                                  OUT/at_budget (work = linearizations + messages; certify_common.h)
 *     --curve-every K              export the state at the beliefs every K rounds to OUT/curve/
 *     --pause-at N                 after round N write OUT/checkpoint_rN and wait for OUT/continue (or OUT/abort)
 */

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>

#include "common/data_types.h"
#include "common/experiment_config.h"
#include "common/factor_graph_builder.h"
#include "common/g2o_parser.h"

#include <gsolver/maths/gp_interpolation.h>

#include "certify_common.h"

using namespace gsolver;
using namespace examples;

struct Args {
  std::string g2o, gt, out, tum, config = "kitti_05";
  int keyframes = 0, normal_iters = 10, normal_max = 200, frozen_iters = 1000, dump_stride = 1;
  double converge = 0.0, anchor_precision = 0.0, frozen_eps = 1e-6;
  std::string eps_sites;
  bool freeze = false, dynamic_schedule = false;
  std::vector<int> unfrozen_iters;
  double relin_threshold = -1.0; // >= 0: threshold relinearization (Phase D, WP1)
  int relin_every        = 1;
  std::string final_export;
  std::string relin_metric = "physical";
  long long work_budget  = 0;
  int curve_every = 0, pause_at = 0;
};

static Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto next     = [&]() -> std::string {
      if (i + 1 >= argc)
        throw std::runtime_error("missing value for " + k);
      return argv[++i];
    };
    if (k == "--g2o") a.g2o = next();
    else if (k == "--gt") a.gt = next();
    else if (k == "--out") a.out = next();
    else if (k == "--tum") a.tum = next();
    else if (k == "--config") a.config = next();
    else if (k == "--keyframes") a.keyframes = std::stoi(next());
    else if (k == "--normal-iters") a.normal_iters = std::stoi(next());
    else if (k == "--converge") a.converge = std::stod(next());
    else if (k == "--normal-max") a.normal_max = std::stoi(next());
    else if (k == "--unfrozen-export-iters") a.unfrozen_iters = certify::parseIntList(next());
    else if (k == "--freeze") a.freeze = true;
    else if (k == "--dynamic-schedule") a.dynamic_schedule = true;
    else if (k == "--anchor-precision") a.anchor_precision = std::stod(next());
    else if (k == "--frozen-iters") a.frozen_iters = std::stoi(next());
    else if (k == "--frozen-eps") a.frozen_eps = std::stod(next());
    else if (k == "--dump-stride") a.dump_stride = std::stoi(next());
    else if (k == "--frozen-eps-sites") a.eps_sites = next();
    else if (k == "--relin-threshold") a.relin_threshold = std::stod(next());
    else if (k == "--relin-every") a.relin_every = std::stoi(next());
    else if (k == "--final-export") a.final_export = next();
    else if (k == "--relin-metric") a.relin_metric = next();
    else if (k == "--work-budget") a.work_budget = std::stoll(next());
    else if (k == "--curve-every") a.curve_every = std::stoi(next());
    else if (k == "--pause-at") a.pause_at = std::stoi(next());
    else throw std::runtime_error("unknown argument " + k);
  }
  if (a.g2o.empty() || a.out.empty())
    throw std::runtime_error("usage: certify_stereo --g2o PATH --gt PATH --out DIR [options]");
  if (a.relin_threshold >= 0 && (a.freeze || a.converge > 0 || !a.unfrozen_iters.empty()))
    throw std::runtime_error("--relin-threshold runs a fixed number of rounds (--normal-iters) and cannot be combined "
                             "with --freeze, --converge or --unfrozen-export-iters");
  return a;
}

static std::vector<double> loadGTTimestamps(const std::string& path) {
  std::vector<double> timestamps;
  std::ifstream file(path);
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream ss(line);
    double t;
    if (ss >> t)
      timestamps.push_back(t);
  }
  return timestamps;
}

int main(int argc, char** argv) {
  try {
    Args a = parse(argc, argv);
    certify::makeDirs(a.out);
    auto params = loadExperimentParams(getDefaultConfigPath(), a.config);

    CameraParams camera;
    std::vector<PoseInit> all_poses;
    std::vector<LandmarkInit> all_landmarks;
    std::vector<StereoMeas> all_stereo;
    std::vector<PriorMeas> all_priors;
    parseStereoG2OFile(a.g2o, camera, all_poses, all_landmarks, all_stereo, all_priors);

    // One robot window, as stereo_visual.cpp builds it for num_robots = 1, cut to the first L keyframes
    const size_t end = (a.keyframes > 0) ? std::min(all_poses.size(), size_t(a.keyframes)) : all_poses.size();
    std::vector<PoseInit> robot_poses(all_poses.begin(), all_poses.begin() + end);

    FactorGraph graph;
    GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
    solver.dynamic_schedule_ = a.dynamic_schedule;
    add_pose_variable_nodes(graph, robot_poses);
    add_gp_prior_factor_nodes(graph, params.qc_diag);

    std::set<size_t> robot_lm_ids;
    std::vector<StereoMeas> robot_stereo;
    for (const auto& obs : all_stereo) {
      bool in_window = false;
      for (const auto& p : robot_poses) {
        if (std::abs(p.timestamp - obs.timestamp) < 1e-9) {
          in_window = true;
          break;
        }
      }
      if (in_window) {
        robot_stereo.push_back(obs);
        robot_lm_ids.insert(obs.landmark_id);
      }
    }
    std::vector<LandmarkInit> robot_landmarks;
    for (const auto& lm : all_landmarks)
      if (robot_lm_ids.count(lm.landmark_id))
        robot_landmarks.push_back(lm);
    add_landmark_variable_nodes(graph, robot_landmarks, 0);
    add_stereo_factor_nodes(graph, robot_stereo, camera, 0);
    std::cout << "graph: " << robot_poses.size() << " poses, " << robot_landmarks.size() << " landmarks, "
              << robot_stereo.size() << " observations" << std::endl;

    certify::Json meta;
    meta.str("dataset", a.keyframes > 0 ? "kitti06_prefix" : "kitti06_full").str("g2o", a.g2o).str("config", a.config);
    meta.set("qc_diag", certify::jsonArray(std::vector<double>(params.qc_diag.data(), params.qc_diag.data() + 6)));
    meta.integer("keyframes", static_cast<long long>(robot_poses.size()));
    meta.set("camera", certify::jsonArray(std::vector<double>{camera.fx, camera.fy, camera.cx, camera.cy, camera.bf}));

    // ---- normal (relinearizing) run: G-solver's per-round schedule, or the threshold rule (Phase D, WP1)
    const bool threshold = a.relin_threshold >= 0;
    if (threshold) {
      gsolver::setFrozenRegularizer(a.frozen_eps);
      certify::enableThresholdRelinearization(solver, a.relin_threshold, a.relin_every, a.relin_metric);
      meta.num("relinearize_threshold", a.relin_threshold).integer("relinearize_every", a.relin_every);
      meta.str("relinearize_metric", a.relin_metric);
      meta.str("schedule", "threshold relinearization, one linearization point per variable (certify/THRESHOLD_RELIN.md)");
    } else {
      meta.str("schedule", "G-solver per-round relinearization at the cavity means (shipped)");
    }
    auto vars = certify::orderVariables(graph);
    certify::MeanSnapshot snap;
    std::vector<double> step_max, step_med;
    const int max_iters = (a.converge > 0) ? a.normal_max : a.normal_iters;
    int iteration       = 0;
    bool converged      = false;
    auto t0             = std::chrono::steady_clock::now();
    // Every round goes through the work account (WP1b); performIteration() itself is unchanged
    certify::RoundRunner runner(solver, graph, a.out, a.work_budget, a.curve_every, meta);
    auto maybePause = [&]() {
      return !(a.pause_at > 0 && iteration == a.pause_at && iteration < max_iters) || runner.pause("r" + std::to_string(iteration));
    };
    while (iteration < max_iters) {
      if (threshold) {
        // mu_ stays at the linearization points in threshold mode, so the belief-step monitor does not apply
        runner.round();
        ++iteration;
        const auto& st = solver.threshold_stats_;
        std::cout << "threshold round " << iteration << ": re-frozen " << st.relinearized_vars.size()
                  << " variables, relinearized " << st.relinearized_factors << " factors" << std::endl;
        if (!maybePause())
          break;
        continue;
      }
      snap.take(vars);
      runner.round();
      ++iteration;
      auto [smax, smed] = snap.steps(vars);
      step_max.push_back(smax);
      step_med.push_back(smed);
      std::cout << "normal iteration " << iteration << ": max step " << smax << ", median step " << smed << std::endl;
      if (std::find(a.unfrozen_iters.begin(), a.unfrozen_iters.end(), iteration) != a.unfrozen_iters.end()) {
        char name[32];
        snprintf(name, sizeof(name), "/unfrozen/it%03d", iteration);
        certify::Json m = meta;
        m.integer("normal_iteration", iteration);
        certify::exportModel(graph, a.out + name, certify::Charts::Beliefs, m);
      }
      if (a.converge > 0 && smax < a.converge) {
        converged = true;
        break;
      }
      if (!maybePause())
        break;
    }
    runner.finish();
    const double normal_secs = certify::seconds_since(t0);
    meta.integer("normal_iterations_before_freeze", iteration);
    meta.num("normal_run_wall_s", normal_secs);
    meta.set("normal_step_max", certify::jsonArray(step_max));
    meta.set("normal_step_median", certify::jsonArray(step_med));
    if (a.converge > 0) {
      meta.num("converge_tol", a.converge).integer("normal_max", a.normal_max).boolean("normal_converged", converged);
    }
    if (threshold) {
      // The final beliefs become the linearization points (mu_ = beliefs for the TUM output; the frozen
      // export is then the linearization at the final beliefs, state z = 0).
      certify::writeRelinCounts(a.out + "/relin_counts.csv", solver, graph);
      solver.relinearizeAllAtBeliefs(graph);
    }
    if (!a.final_export.empty()) {
      certify::Json m = meta;
      m.str("final_export", "linearization at the final belief means (state z = 0)");
      certify::exportModel(graph, a.final_export, threshold ? certify::Charts::Frozen : certify::Charts::Beliefs, m);
    }

    if (!a.freeze) {
      if (!a.tum.empty()) {
        std::vector<std::shared_ptr<SE3PoseVel>> pose_nodes;
        for (const auto& var : graph.variable_nodes_)
          if (var->id_.find("POSE_") != std::string::npos)
            if (auto node = std::dynamic_pointer_cast<SE3PoseVel>(var))
              pose_nodes.push_back(node);
        std::sort(pose_nodes.begin(), pose_nodes.end(), [](const auto& x, const auto& y) { return x->timestamp_ < y->timestamp_; });
        const auto gt_timestamps = loadGTTimestamps(a.gt);
        auto interp              = GPInterpAtTimestamps(pose_nodes, gt_timestamps, params.qc_diag);
        std::ofstream output_file(a.tum);
        output_file << std::fixed << std::setprecision(9);
        output_file << "#timestamp tx ty tz qx qy qz qw\n";
        for (size_t i = 0; i < interp.size(); ++i) {
          const Eigen::Vector3d& t = interp[i].pose.translation();
          const Eigen::Quaterniond q(interp[i].pose.linear());
          output_file << gt_timestamps[i] << " " << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << " " << q.y()
                      << " " << q.z() << " " << q.w() << "\n";
        }
      }
      std::ofstream(a.out + "/normal_run.json") << meta.dump() << "\n";
      return 0;
    }

    // ---- freeze
    {
      // G-solver's own GP interpolation of the beliefs at the freeze instant, at the GT timestamps
      // (fact F3: certify/interp.py must reproduce it from the exported linearization points)
      std::vector<std::shared_ptr<SE3PoseVel>> pose_nodes;
      for (const auto& var : graph.variable_nodes_)
        if (auto node = std::dynamic_pointer_cast<SE3PoseVel>(var))
          pose_nodes.push_back(node);
      std::sort(pose_nodes.begin(), pose_nodes.end(), [](const auto& x, const auto& y) { return x->timestamp_ < y->timestamp_; });
      const auto gt_timestamps = loadGTTimestamps(a.gt);
      auto interp              = GPInterpAtTimestamps(pose_nodes, gt_timestamps, params.qc_diag);
      std::vector<double> rows;
      for (size_t i = 0; i < interp.size(); ++i) {
        rows.push_back(gt_timestamps[i]);
        for (int r = 0; r < 4; ++r)
          for (int c = 0; c < 4; ++c)
            rows.push_back(interp[i].pose.matrix()(r, c));
        for (int k = 0; k < 6; ++k)
          rows.push_back(interp[i].vel(k));
      }
      certify::writeNpy(a.out + "/interp_at_freeze.npy", rows, {interp.size(), 23});
    }
    gsolver::setFrozenRegularizer(a.frozen_eps);
    certify::applySiteOverrides(a.eps_sites);
    solver.freeze(graph, iteration > 0);
    meta.boolean("factor_messages_valid_at_freeze", iteration > 0);
    if (a.anchor_precision > 0) {
      // A3: explicit SE(3) prior on the first keyframe at its initial ORB-SLAM3 pose, frozen model only.
      const Eigen::Isometry3d z0          = robot_poses.front().pose;
      const Eigen::Matrix<double, 6, 6> C = Eigen::Matrix<double, 6, 6>::Identity() / a.anchor_precision;
      auto prior = std::make_shared<SE3PoseVelPrior>("CERTIFY_GAUGE_PRIOR", z0, C);
      graph.addFactorNode(prior, {poseId(robot_poses.front().timestamp)});
      prior->freezeNewFactor();
      meta.num("anchor_precision", a.anchor_precision);
      meta.str("gauge",
               "CERTIFY_GAUGE_PRIOR: SE(3) prior on the first keyframe at its initial ORB-SLAM3 pose, precision " +
                 certify::jsonNum(a.anchor_precision) + "*I, added to the frozen model only (modelling choice, A3)");
    } else {
      meta.str("gauge", "none added (shipped driver has no anchor)");
    }
    certify::exportModel(graph, a.out, certify::Charts::Frozen, meta);
    const double secs = certify::runFrozen(graph, solver, a.frozen_iters, a.out, 1000, a.dump_stride);
    std::ofstream(a.out + "/frozen_run.json") << "{\"frozen_iters\": " << a.frozen_iters << ", \"wall_s\": " << secs
                                              << ", \"frozen_regularizer\": " << certify::jsonNum(a.frozen_eps) << "}\n";
    std::cout << "frozen: " << a.frozen_iters << " iterations in " << secs << " s -> " << a.out << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 2;
  }
}
