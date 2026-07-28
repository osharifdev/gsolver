/**
 * @file repro_charuco.cpp
 * @brief Instrumented ChArUco BA reproduction (paper Fig. 11).
 *
 * Replicates examples/cpp/apps/charuco_experiment_ba.cpp exactly: same parsing,
 * same perturbation (std::default_random_engine seed 42, rot sigma = 0.1 x
 * trans sigma, covariance inflation I*(1/noise)), same graph construction and
 * SYNCHRONOUS schedule.
 *
 * The paper's y-metric (mean PIXEL reprojection error) is NOT computable from
 * public artefacts (no camera intrinsics, pixel detections or marker geometry
 * in the repository). This tool records per-iteration SUBSTITUTE metrics on
 * the available pose-landmark SE3 measurements:
 *   - mean translation error [m] and mean rotation error [rad] of the
 *     estimated pose->landmark relative transform vs the measured one,
 *   - factor-energy breakdown (landmark / GP / anchor), Eq. 18 style.
 *
 * Usage: repro_charuco --input in.g2o --params yaml --out dir --noise 0.1 [--iters N]
 */
#include <filesystem>
#include <iostream>
#include <map>
#include <omp.h>
#include <random>
#include <set>
#include <unordered_map>

#include "common/data_types.h"
#include "common/experiment_config.h"
#include "common/factor_graph_builder.h"
#include "common/g2o_parser.h"

#include <gsolver/core/gbp_solver.h>
#include <gsolver/graph/factor_graph.h>

#include "repro_common.h"

using namespace gsolver;
using namespace examples;
using namespace repro;

// EXACT copy of charuco_experiment_ba.cpp::perturbPoses (same RNG, same seed)
static void perturbPoses(std::vector<PoseInit>& poses, double noise_std, int seed = 42) {
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

int main(int argc, char* argv[]) {
  std::map<std::string, std::string> opt;
  for (int i = 1; i + 1 < argc; i += 2)
    opt[std::string(argv[i]).substr(2)] = argv[i + 1];
  const std::string input_path = opt.at("input");
  const std::string params_path = opt.at("params");
  const std::string out_dir     = opt.at("out");
  const double noise            = std::stod(opt.count("noise") ? opt["noise"] : "0.1");
  std::filesystem::create_directories(out_dir);

  auto params = loadExperimentParams(params_path, "printing_room");
  const int iters = opt.count("iters") ? std::stoi(opt["iters"]) : params.num_iterations;
  omp_set_num_threads(omp_get_max_threads());

  auto t_load0 = Clock::now();
  std::vector<PoseInit> poses;
  std::vector<OdometryMeas> odometry;
  std::vector<PriorMeas> priors;
  std::vector<LandmarkMeas> landmarks;
  parseG2OFile(input_path, poses, odometry, priors, landmarks);
  const double load_ms = ms_since(t_load0);

  perturbPoses(poses, noise); // seed 42, as shipped

  auto t_build0 = Clock::now();
  FactorGraph fg;
  add_pose_variable_nodes(fg, poses);
  add_gp_prior_factor_nodes(fg, params.qc_diag);
  add_landmark_factors(fg, landmarks);
  fix_first_pose(fg);
  const double build_ms = ms_since(t_build0);

  // fast node lookup (read-only)
  std::unordered_map<std::string, std::shared_ptr<VariableNodeBase>> byId;
  for (auto& v : fg.variable_nodes_)
    byId[v->id_] = v;

  const Eigen::Matrix<double, 6, 6> lm_sigma = Eigen::Matrix<double, 6, 6>::Identity() * 3.0; // add_landmark_factors default
  const Eigen::Matrix<double, 6, 6> lm_sigma_inv = lm_sigma.inverse();

  auto evalIter = [&](int it, std::ofstream& f) {
    double sum_t = 0, sum_r = 0, sum_e = 0;
    size_t n = 0;
    for (const auto& lm : landmarks) {
      auto pit = byId.find(poseId(lm.timestamp));
      auto lit = byId.find(landmarkId(static_cast<int>(lm.landmark_id)));
      if (pit == byId.end() || lit == byId.end())
        continue;
      auto pn = std::dynamic_pointer_cast<SE3PoseVel>(pit->second);
      auto ln = std::dynamic_pointer_cast<SE3Pose>(lit->second);
      if (!pn || !ln)
        continue;
      const Eigen::Isometry3d T_rel = pn->mu_.pose.inverse() * ln->mu_;
      const Vector6d e              = logMapSE3(lm.measurement.inverse() * T_rel);
      sum_t += e.head<3>().norm();
      sum_r += e.tail<3>().norm();
      sum_e += 0.5 * e.dot(lm_sigma_inv * e);
      ++n;
    }
    // GP energy over pose chain
    auto states = snapshotStates(fg);
    std::vector<PriorMeas> none;
    std::vector<OdometryMeas> none2;
    const double e_gp = evalEnergy(states, none, none2, params.qc_diag, true).gp;
    f << it << "," << (n ? sum_t / n : 0) << "," << (n ? sum_r / n : 0) << "," << (n ? sum_e / n : 0) << "," << sum_e << ","
      << e_gp << "," << (sum_e + e_gp) << "," << n << "\n";
  };

  std::ofstream f(out_dir + "/per_iteration.csv");
  f << "iteration,mean_landmark_trans_err_m,mean_landmark_rot_err_rad,mean_landmark_energy,"
       "E_landmarks,E_gp,E_total_no_anchor,n_landmark_meas\n";
  f << std::setprecision(10);
  evalIter(0, f);

  GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
  std::vector<double> iter_ms;
  auto t_solve0 = Clock::now();
  for (int i = 0; i < iters; ++i) {
    auto t0 = Clock::now();
    solver.performIteration(fg);
    iter_ms.push_back(ms_since(t0));
    evalIter(i + 1, f);
  }
  const double solve_ms = ms_since(t_solve0);

  // outputs
  std::vector<TumPose> est;
  for (auto& vn : fg.getPoseVariableNodes()) {
    auto n = std::dynamic_pointer_cast<SE3PoseVel>(vn);
    if (n)
      est.push_back({n->timestamp_, n->mu_.pose});
  }
  writeTumFile(out_dir + "/knots_est.tum", est);

  std::ofstream s(out_dir + "/summary.csv");
  s << "input,noise,iterations,threads,n_poses,n_landmark_meas,load_ms,build_ms,solve_ms,avg_iter_ms,peak_rss_kb\n";
  double avg = 0;
  for (double m : iter_ms)
    avg += m;
  avg /= std::max<size_t>(1, iter_ms.size());
  s << input_path << "," << noise << "," << iters << "," << omp_get_max_threads() << "," << poses.size() << ","
    << landmarks.size() << "," << load_ms << "," << build_ms << "," << solve_ms << "," << avg << "," << peakRSSKb() << "\n";

  std::cout << "[repro_charuco] done, " << iters << " iterations, solve_ms=" << solve_ms << "\n";
  return 0;
}
