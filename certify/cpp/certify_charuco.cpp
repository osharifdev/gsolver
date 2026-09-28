/**
 * @file certify_charuco.cpp
 * @brief Phase A driver for the ChArUco experiments: normal run, snapshot, freeze, export.
 *
 * The graph construction and normal schedule replicate examples/cpp/apps/charuco_experiment_islam.cpp
 * (--mode islam) and charuco_experiment_ba.cpp (--mode ba) statement for statement; with no freeze
 * flag the trajectory written by --tum is byte-identical to the shipped drivers' output.
 *
 * Usage:
 *   certify_charuco --mode islam|ba --g2o PATH --out DIR [options]
 *     --tum PATH                   write the normal-run trajectory (writeTUM) at the end
 *     --noise S                    ba: initial-guess perturbation std (shipped default 0.0)
 *     --fraction F                 islam: snapshot after the iteration run on round(F*N) poses
 *                                  (F = 1: after the final BA)
 *     --freeze-at N                ba: snapshot after N normal iterations (0 = before any iteration)
 *     --dynamic-schedule           GbpSolver::dynamic_schedule_ (performance only, bit-identical)
 *     --freeze                     freeze at the snapshot, export the model, run frozen iterations
 *     --frozen-iters T             frozen iterations (default 3000)
 *     --frozen-eps E               frozen-mode regularizer (default 1e-6, as in G-solver)
 *     --frozen-eps-sites SPEC      diagnostic per-site override, e.g. "marg=1e-7,f2v=1e-6"
 *     --dump-stride S              state dump: every iteration up to 1000, then every S-th, and the last
 *     --unfrozen-export-iters L    comma list of normal-run iterations to export (charts at beliefs)
 *     --relin-threshold D          islam, Phase D, WP1: threshold relinearization instead of the per-round
 *                                  schedule, d_v = ||z_v||_{H_vv} > D (certify/THRESHOLD_RELIN.md), same round
 *                                  budget; writes relin_rounds.csv and relin_counts.csv
 *     --relin-every R              at most one re-freeze per variable per R rounds (default 1)
 *     --final-export DIR           islam: export the linearization at the final belief means (either schedule)
 *     --extra-final-rounds N       islam, diagnostic: N rounds added to the final BA (default 0)
 *     --relin-metric physical|D    metric of d_v: Euclidean tangent norm (default, revised D-a) or ||z_v||_{H_vv}
 *     --work-budget W              islam: export the state after the last round with cumulative work <= W to
 *                                  OUT/at_budget (work = linearizations + messages; certify_common.h)
 *     --curve-every K              islam: export the state at the beliefs every K rounds to OUT/curve/
 *     --sidecar DIR                Phase D, WP2 (threshold mode): certify every round with the Python sidecar through
 *                                  DIR (certify/SIDECAR.md); the sidecar decides which poses sleep
 *     --persist-rounds LIST        with --sidecar: also export these rounds (and the last) to OUT/export_rNNNNN
 *     --pause-every N              with --sidecar: checkpoint every N rounds (export, wait for OUT/continue)
 *     --count-flops                Phase D, WP6a (threshold mode): per-round GBP flops to OUT/flops.csv (certify/FLOPS.md)
 *     --flop-trace-round R         with --count-flops: one operation of every kind in round R to OUT/flops_trace.csv
 *     --true-jacobians             Stage 1: the stereo and GP factors use their true derivatives (certify/TRUE_JACOBIANS.md)
 *     --eval-dir DIR --eval-every N  Stage 1 (threshold mode): after every N-th round export the linearization and the beliefs
 *                                  to DIR and wait for the evaluator (scripts/stage1_evaluator.py); --eval-pause-every M
 *                                  checkpoints after every M-th round
 *     --change-taus LIST           Stage 1: log the change-based criterion per round (heuristics.csv) and export the beliefs
 *                                  at its first round below each tau (--change-from R: not before round R)
 *     --export-beliefs-at LIST     export the beliefs after these rounds to OUT/beliefs_rNNNNN
 *     --fd-check-at R              with --eval-every: central-difference check of every factor's Jacobian at the evaluation
 *                                  after round R (OUT/fdcheck/fd_check.tsv)
 */

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <string>

#include "common/data_types.h"
#include "common/experiment_config.h"
#include "common/factor_graph_builder.h"
#include "common/g2o_parser.h"
#include "common/tum_writer.h"

#include "certify_common.h"

using namespace gsolver;
using namespace examples;

// The front end's current estimate of a pose: its belief mean. With threshold relinearization (Phase D, WP1)
// mu_ holds the linearization point and the belief is mu_lin_ ⊞ z_; otherwise mu_ is the belief (shipped path).
static SE3PoseVel::EstimateType beliefOf(const std::shared_ptr<SE3PoseVel>& pose) {
  return pose->frozen_ ? pose->boxplus(pose->z_) : pose->mu_;
}

// From charuco_experiment_islam.cpp, verbatim except that the previous pose is read through beliefOf()
void predictAndAddNewPose(FactorGraph& factor_graph, double new_timestamp, double previous_timestamp) {
  auto pose_nodes                       = factor_graph.getPoseVariableNodes();
  std::shared_ptr<SE3PoseVel> prev_pose = std::dynamic_pointer_cast<SE3PoseVel>(pose_nodes.back());
  const SE3PoseVel::EstimateType prev   = beliefOf(prev_pose);

  double dt                        = new_timestamp - previous_timestamp;
  Eigen::Isometry3d delta_pose     = expMapSE3(prev.vel * dt);
  Eigen::Isometry3d predicted_pose = prev.pose * delta_pose;

  Eigen::Vector<double, 6> velocity = logMapSE3(delta_pose) / dt;

  PoseInit new_pose;
  new_pose.timestamp  = new_timestamp;
  new_pose.pose       = predicted_pose;
  new_pose.covariance = Eigen::Matrix<double, 6, 6>::Identity();

  Eigen::Matrix<double, 12, 12> sigma = Eigen::Matrix<double, 12, 12>::Identity() * 3.0;
  add_pose_variable_node(factor_graph, new_pose, velocity, sigma);
}

// add_landmark_factor (examples/cpp/common/factor_graph_builder.h), verbatim except that a new marker is
// initialized from the camera pose's belief (beliefOf); identical to it on the shipped path.
static void addLandmarkFactor(FactorGraph& factor_graph, const LandmarkMeas& landmark,
                              const Eigen::Matrix<double, 6, 6>& sigma = Eigen::Matrix<double, 6, 6>::Identity() * 3.0) {
  std::string pose_var_id     = poseId(landmark.timestamp);
  std::string landmark_var_id = landmarkId(landmark.landmark_id);
  if (!factor_graph.getVariableNode(landmark_var_id)) {
    auto pose_node       = std::dynamic_pointer_cast<SE3PoseVel>(factor_graph.getVariableNode(pose_var_id));
    Eigen::Isometry3d lm = beliefOf(pose_node).pose * landmark.measurement;
    auto landmark_var    = std::make_shared<SE3Pose>(landmark_var_id, lm, sigma);
    factor_graph.addVariableNode(landmark_var);
  }
  std::string factor_id = timestampToString(landmark.timestamp) + "_" + std::to_string(landmark.landmark_id);
  auto factor           = std::make_shared<SE3PoseVelPose>(factor_id, landmark.measurement, sigma);
  factor_graph.addFactorNode(factor, {pose_var_id, landmark_var_id});
}

// Verbatim from charuco_experiment_ba.cpp
void perturbPoses(std::vector<PoseInit>& poses, double noise_std, int seed = 42) {
  if (noise_std == 0.0)
    return;

  std::default_random_engine gen(seed);
  std::normal_distribution<double> trans_noise(0.0, noise_std);
  std::normal_distribution<double> rot_noise(0.0, noise_std * 0.1);

  for (auto& pose : poses) {
    Eigen::Vector<double, 6> delta;
    delta << trans_noise(gen), trans_noise(gen), trans_noise(gen), rot_noise(gen), rot_noise(gen), rot_noise(gen);
    pose.pose = pose.pose * expMapSE3(delta);
    pose.covariance += Eigen::Matrix<double, 6, 6>::Identity() * (1.0 / noise_std);
  }
}

struct Args {
  std::string mode, g2o, out, tum;
  double noise = 0.0, fraction = 1.0, frozen_eps = 1e-6;
  std::string eps_sites;
  int freeze_at = -1, frozen_iters = 3000, dump_stride = 1;
  bool freeze = false, dynamic_schedule = false;
  std::vector<int> unfrozen_iters;
  double relin_threshold = -1.0; // >= 0: threshold relinearization (Phase D, WP1), islam only
  int relin_every        = 1;
  std::string final_export;
  int extra_final_rounds = 0; // diagnostic: rounds added to the final BA (default 0 = shipped budget)
  std::string relin_metric = "physical";
  long long work_budget  = 0;
  int curve_every        = 0;
  std::string sidecar;
  std::vector<int> persist_rounds;
  int pause_every = 0;
  bool count_flops = false;
  int flop_trace_round = -1;
  bool true_jacobians = false;
  std::string eval_dir;
  int eval_every = 0, eval_pause_every = 0, change_from = 1, fd_check_at = -1;
  std::vector<double> change_taus;
  std::vector<int> beliefs_at;
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
    if (k == "--mode") a.mode = next();
    else if (k == "--g2o") a.g2o = next();
    else if (k == "--out") a.out = next();
    else if (k == "--tum") a.tum = next();
    else if (k == "--noise") a.noise = std::stod(next());
    else if (k == "--fraction") a.fraction = std::stod(next());
    else if (k == "--freeze-at") a.freeze_at = std::stoi(next());
    else if (k == "--freeze") a.freeze = true;
    else if (k == "--dynamic-schedule") a.dynamic_schedule = true;
    else if (k == "--frozen-iters") a.frozen_iters = std::stoi(next());
    else if (k == "--frozen-eps") a.frozen_eps = std::stod(next());
    else if (k == "--dump-stride") a.dump_stride = std::stoi(next());
    else if (k == "--frozen-eps-sites") a.eps_sites = next();
    else if (k == "--unfrozen-export-iters") a.unfrozen_iters = certify::parseIntList(next());
    else if (k == "--relin-threshold") a.relin_threshold = std::stod(next());
    else if (k == "--relin-every") a.relin_every = std::stoi(next());
    else if (k == "--final-export") a.final_export = next();
    else if (k == "--extra-final-rounds") a.extra_final_rounds = std::stoi(next());
    else if (k == "--relin-metric") a.relin_metric = next();
    else if (k == "--work-budget") a.work_budget = std::stoll(next());
    else if (k == "--curve-every") a.curve_every = std::stoi(next());
    else if (k == "--sidecar") a.sidecar = next();
    else if (k == "--persist-rounds") a.persist_rounds = certify::parseIntList(next());
    else if (k == "--pause-every") a.pause_every = std::stoi(next());
    else if (k == "--count-flops") a.count_flops = true;
    else if (k == "--flop-trace-round") a.flop_trace_round = std::stoi(next());
    else if (k == "--true-jacobians") a.true_jacobians = true;
    else if (k == "--eval-dir") a.eval_dir = next();
    else if (k == "--eval-every") a.eval_every = std::stoi(next());
    else if (k == "--eval-pause-every") a.eval_pause_every = std::stoi(next());
    else if (k == "--change-taus") a.change_taus = certify::parseDoubleList(next());
    else if (k == "--change-from") a.change_from = std::stoi(next());
    else if (k == "--export-beliefs-at") a.beliefs_at = certify::parseIntList(next());
    else if (k == "--fd-check-at") a.fd_check_at = std::stoi(next());
    else throw std::runtime_error("unknown argument " + k);
  }
  if (a.relin_threshold >= 0 && (a.mode != "islam" || a.freeze || !a.unfrozen_iters.empty()))
    throw std::runtime_error("--relin-threshold is implemented for --mode islam without --freeze or unfrozen exports");
  if ((a.mode != "islam" && a.mode != "ba") || a.g2o.empty() || a.out.empty())
    throw std::runtime_error("usage: certify_charuco --mode islam|ba --g2o PATH --out DIR [options]");
  return a;
}

// Snapshot: freeze the graph, export the frozen model, run frozen iterations with state dumps.
static void freezeAndRun(FactorGraph& graph, GbpSolver& solver, const Args& a, certify::Json meta, bool f2v_valid) {
  gsolver::setFrozenRegularizer(a.frozen_eps);
  certify::applySiteOverrides(a.eps_sites);
  solver.freeze(graph, f2v_valid);
  meta.boolean("factor_messages_valid_at_freeze", f2v_valid);
  certify::exportModel(graph, a.out, certify::Charts::Frozen, meta);
  const double secs = certify::runFrozen(graph, solver, a.frozen_iters, a.out, 1000, a.dump_stride);
  std::ofstream(a.out + "/frozen_run.json") << "{\"frozen_iters\": " << a.frozen_iters << ", \"wall_s\": " << secs
                                            << ", \"frozen_regularizer\": " << certify::jsonNum(a.frozen_eps) << "}\n";
  std::cout << "frozen: " << a.frozen_iters << " iterations in " << secs << " s -> " << a.out << std::endl;
}

static void maybeExportUnfrozen(FactorGraph& graph, const Args& a, int iteration, certify::Json meta) {
  if (std::find(a.unfrozen_iters.begin(), a.unfrozen_iters.end(), iteration) == a.unfrozen_iters.end())
    return;
  char name[32];
  snprintf(name, sizeof(name), "/unfrozen/it%03d", iteration);
  meta.integer("normal_iteration", iteration);
  certify::exportModel(graph, a.out + name, certify::Charts::Beliefs, meta);
}

int run_islam(const Args& a) {
  auto params = loadExperimentParams(getDefaultConfigPath(), "printing_room");
  const int iterations_per_pose = 1;
  const int final_ba_iterations = 10;

  std::vector<PoseInit> poses;
  std::vector<OdometryMeas> odometry;
  std::vector<PriorMeas> priors;
  std::vector<LandmarkMeas> landmarks;
  parseG2OFile(a.g2o, poses, odometry, priors, landmarks);

  std::set<double> stamps;
  for (auto& l : landmarks)
    stamps.insert(l.timestamp);
  const int n_total     = static_cast<int>(stamps.size());
  const int n_snapshot  = static_cast<int>(std::lround(a.fraction * n_total));
  const bool final_snap = (n_snapshot >= n_total);

  FactorGraph factor_graph;
  GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
  solver.dynamic_schedule_ = a.dynamic_schedule;

  double current_timestamp = landmarks[0].timestamp;
  PoseInit first_pose;
  first_pose.timestamp  = current_timestamp;
  first_pose.pose       = Eigen::Isometry3d::Identity();
  first_pose.covariance = Eigen::Matrix<double, 6, 6>::Identity();
  Eigen::Vector<double, 6> zero_velocity   = Eigen::Vector<double, 6>::Zero();
  Eigen::Matrix<double, 12, 12> init_sigma = Eigen::Matrix<double, 12, 12>::Identity() * 3.0;
  add_pose_variable_node(factor_graph, first_pose, zero_velocity, init_sigma);

  std::string prior_id = "PRIOR_ANCHOR_" + timestampToString(current_timestamp);
  auto anchor_prior =
    std::make_shared<SE3PoseVelPrior>(prior_id, Eigen::Isometry3d::Identity(), Eigen::Matrix<double, 6, 6>::Identity() * 1e-12);
  factor_graph.addFactorNode(anchor_prior, {factor_graph.variable_nodes_[0]->id_});

  certify::Json meta;
  meta.str("dataset", "charuco_islam").str("g2o", a.g2o).str("config", "printing_room");
  meta.set("qc_diag", certify::jsonArray(std::vector<double>(params.qc_diag.data(), params.qc_diag.data() + 6)));
  meta.num("snapshot_fraction", a.fraction).integer("snapshot_poses", n_snapshot).integer("sequence_poses", n_total);
  meta.str("gauge", "PRIOR_ANCHOR factor on the first pose, covariance 1e-12*I (precision 1e12), shipped driver");

  // Phase D, WP1: the same rounds under the threshold rule instead of per-round relinearization
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
  // Every round goes through the work account (WP1b); performIteration() itself is unchanged
  certify::RoundRunner runner(solver, factor_graph, a.out, a.work_budget, a.curve_every, meta);
  if (a.true_jacobians)
    gsolver::true_jacobians = true;
  meta.boolean("true_jacobians", a.true_jacobians);
  if (a.eval_every > 0) {
    if (a.relin_threshold < 0 || a.eval_dir.empty())
      throw std::runtime_error("--eval-every needs --relin-threshold and --eval-dir");
    runner.eval_dir         = a.eval_dir;
    runner.eval_every       = a.eval_every;
    runner.eval_last        = (n_total - 1) * iterations_per_pose + final_ba_iterations + a.extra_final_rounds;
    runner.eval_pause_every = a.eval_pause_every;
    runner.fd_check_at      = a.fd_check_at;
  }
  if (!a.change_taus.empty())
    runner.enableChangeCriterion(a.change_taus, a.change_from);
  runner.beliefs_at = a.beliefs_at;
  if (a.count_flops) {
    if (a.relin_threshold < 0)
      throw std::runtime_error("--count-flops counts the frozen (threshold-mode) operations: it needs --relin-threshold");
    runner.enableFlops(a.flop_trace_round);
  }
  if (!a.sidecar.empty()) {
    if (!threshold)
      throw std::runtime_error("--sidecar needs --relin-threshold");
    runner.sidecar        = a.sidecar;
    runner.persist        = a.out;
    runner.persist_rounds = a.persist_rounds;
    runner.last_round     = (n_total - 1) * iterations_per_pose + final_ba_iterations + a.extra_final_rounds;
    runner.pause_every    = a.pause_every;
  }
  auto round = [&]() { runner.round(); };

  int iteration  = 0;
  int pose_count = 1;
  for (const auto& landmark : landmarks) {
    if (landmark.timestamp == current_timestamp) {
      addLandmarkFactor(factor_graph, landmark);
    } else if (landmark.timestamp > current_timestamp) {
      for (int iter = 0; iter < iterations_per_pose; ++iter) {
        round();
        ++iteration;
        maybeExportUnfrozen(factor_graph, a, iteration, meta);
      }
      if (a.freeze && !final_snap && pose_count == n_snapshot) {
        meta.integer("normal_iterations_before_freeze", iteration);
        meta.str("snapshot", "after the iteration on " + std::to_string(pose_count) + " poses");
        freezeAndRun(factor_graph, solver, a, meta, true);
        return 0;
      }
      predictAndAddNewPose(factor_graph, landmark.timestamp, current_timestamp);
      add_gp_prior_factor_node(factor_graph, current_timestamp, landmark.timestamp, params.qc_diag);
      addLandmarkFactor(factor_graph, landmark);
      current_timestamp = landmark.timestamp;
      pose_count++;
    } else {
      std::cerr << "Landmarks are not sorted by timestamp" << std::endl;
      return 1;
    }
  }
  for (int iter = 0; iter < final_ba_iterations + a.extra_final_rounds; ++iter) {
    round();
    ++iteration;
    maybeExportUnfrozen(factor_graph, a, iteration, meta);
  }
  runner.finish();
  if (threshold) {
    // The final beliefs become the linearization points (mu_ = beliefs for writeTUM; the frozen export is
    // then the linearization at the final beliefs, state z = 0).
    certify::writeRelinCounts(a.out + "/relin_counts.csv", solver, factor_graph);
    solver.relinearizeAllAtBeliefs(factor_graph);
  }
  if (!a.final_export.empty()) {
    meta.integer("rounds", iteration).str("final_export", "linearization at the final belief means (state z = 0)");
    certify::exportModel(factor_graph, a.final_export, threshold ? certify::Charts::Frozen : certify::Charts::Beliefs, meta);
  }
  if (a.freeze && final_snap) {
    meta.integer("normal_iterations_before_freeze", iteration);
    meta.str("snapshot", "end of the normal run (after the final BA)");
    freezeAndRun(factor_graph, solver, a, meta, true);
    return 0;
  }
  if (!a.tum.empty()) {
    std::ofstream output_file(a.tum);
    writeTUM(factor_graph, output_file);
  }
  return 0;
}

int run_ba(const Args& a) {
  auto params = loadExperimentParams(getDefaultConfigPath(), "printing_room");

  std::vector<PoseInit> poses;
  std::vector<OdometryMeas> odometry;
  std::vector<PriorMeas> priors;
  std::vector<LandmarkMeas> landmarks;
  parseG2OFile(a.g2o, poses, odometry, priors, landmarks);

  perturbPoses(poses, a.noise);

  FactorGraph factor_graph;
  add_pose_variable_nodes(factor_graph, poses);
  add_gp_prior_factor_nodes(factor_graph, params.qc_diag);
  add_landmark_factors(factor_graph, landmarks);
  fix_first_pose(factor_graph);

  GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
  solver.dynamic_schedule_ = a.dynamic_schedule;

  certify::Json meta;
  meta.str("dataset", "charuco_ba").str("g2o", a.g2o).str("config", "printing_room").num("noise", a.noise);
  meta.set("qc_diag", certify::jsonArray(std::vector<double>(params.qc_diag.data(), params.qc_diag.data() + 6)));
  meta.str("gauge", "PRIOR_ANCHOR factor on the first pose, covariance 1e-6*I (precision 1e6), shipped driver");

  for (int i = 0; i < params.num_iterations; ++i) {
    if (a.freeze && i == a.freeze_at) {
      meta.integer("normal_iterations_before_freeze", i);
      freezeAndRun(factor_graph, solver, a, meta, i > 0);
      return 0;
    }
    solver.performIteration(factor_graph);
    maybeExportUnfrozen(factor_graph, a, i + 1, meta);
  }
  if (a.freeze && a.freeze_at == params.num_iterations) {
    meta.integer("normal_iterations_before_freeze", params.num_iterations);
    freezeAndRun(factor_graph, solver, a, meta, true);
    return 0;
  }
  if (!a.tum.empty()) {
    std::ofstream output_file(a.tum);
    writeTUM(factor_graph, output_file);
  }
  return 0;
}

int main(int argc, char** argv) {
  try {
    Args a = parse(argc, argv);
    certify::makeDirs(a.out);
    return a.mode == "islam" ? run_islam(a) : run_ba(a);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 2;
  }
}
