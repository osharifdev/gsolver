/**
 * @file repro_stereo.cpp
 * @brief Instrumented KITTI stereo BA reproduction
 *        (paper Table II, Figs. 13, 14, 19, 20).
 *
 * Replicates examples/cpp/apps/stereo_visual.cpp exactly: same parsing, same
 * per-robot trajectory split, same graph construction (incl. the landmark
 * time-window filter and equality-factor handshake), same interleaved
 * SYNCHRONOUS schedule, same GP interpolation at ground-truth timestamps.
 * Instrumentation added on top is read-only:
 *   - ATE/ARE of the GT-timestamp interpolated trajectory (unaligned+aligned),
 *   - mean reprojection error of optimized landmarks vs observed pixels on the
 *     left image, counting ONLY projections inside the image plane
 *     (0<=u<=W-1, 0<=v<=H-1, depth>0), with excluded-projection accounting,
 *   - per-robot solve time, per-segment graph statistics, peak RSS,
 *   - per-frame projection dumps for the rolling-shutter overlay figures.
 *
 * Usage:
 *  repro_stereo --input in.g2o --gt gt.tum --config kitti_05 --params yaml
 *     --out dir [--robots N] [--frames 79,320,...] [--img-w 1226] [--img-h 370]
 *     [--iters N]
 */
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <omp.h>
#include <set>
#include <unordered_map>

#include "common/data_types.h"
#include "common/experiment_config.h"
#include "common/factor_graph_builder.h"
#include "common/g2o_parser.h"

#include <gsolver/core/gbp_solver.h>
#include <gsolver/graph/factor_graph.h>
#include <gsolver/maths/gp_interpolation.h>

#include "repro_common.h"

using namespace gsolver;
using namespace examples;
using namespace repro;

int main(int argc, char* argv[]) {
  std::map<std::string, std::string> opt;
  for (int i = 1; i + 1 < argc; i += 2)
    opt[std::string(argv[i]).substr(2)] = argv[i + 1];
  const std::string input_path  = opt.at("input");
  const std::string gt_path     = opt.at("gt");
  const std::string cfg_key     = opt.count("config") ? opt["config"] : "kitti_05";
  const std::string params_path = opt.at("params");
  const std::string out_dir     = opt.at("out");
  const int num_robots          = opt.count("robots") ? std::stoi(opt["robots"]) : 1;
  const double img_w            = opt.count("img-w") ? std::stod(opt["img-w"]) : 1226.0;
  const double img_h            = opt.count("img-h") ? std::stod(opt["img-h"]) : 370.0;
  std::filesystem::create_directories(out_dir);

  std::vector<int> dump_frames;
  if (opt.count("frames")) {
    std::stringstream ss(opt["frames"]);
    std::string tok;
    while (std::getline(ss, tok, ','))
      dump_frames.push_back(std::stoi(tok));
  }

  auto params = loadExperimentParams(params_path, cfg_key);
  const int iters = opt.count("iters") ? std::stoi(opt["iters"]) : params.num_iterations;
  omp_set_num_threads(omp_get_max_threads());

  // ---------------- load ----------------
  auto t_load0 = Clock::now();
  CameraParams camera;
  std::vector<PoseInit> all_poses;
  std::vector<LandmarkInit> all_landmarks;
  std::vector<StereoMeas> all_stereo;
  std::vector<PriorMeas> all_priors;
  parseStereoG2OFile(input_path, camera, all_poses, all_landmarks, all_stereo, all_priors);
  const double load_ms = ms_since(t_load0);
  const long rss_post_load = peakRSSKb();

  auto gt = loadTum(gt_path);

  // evaluation copies (lightweight; graph build below clears the originals as
  // in the shipped app — RSS impact of these copies is ~15 MB, documented)
  std::vector<PoseInit> eval_poses         = all_poses;
  std::vector<LandmarkInit> eval_landmarks = all_landmarks;
  std::vector<StereoMeas> eval_stereo      = all_stereo;

  // ---------------- build (replicates stereo_visual.cpp) ----------------
  auto t_build0 = Clock::now();
  const size_t poses_per_robot = all_poses.size() / static_cast<size_t>(num_robots);

  std::vector<std::unique_ptr<FactorGraph>> graphs(num_robots);
  std::vector<std::unique_ptr<GbpSolver>> solvers(num_robots);
  for (int r = 0; r < num_robots; ++r) {
    graphs[r]  = std::make_unique<FactorGraph>();
    solvers[r] = std::make_unique<GbpSolver>(SolverScheduleType::SYNCHRONOUS);
  }

  struct SegStat {
    size_t poses = 0, landmarks = 0, stereo = 0;
    double t0 = 0, t1 = 0;
  };
  std::vector<SegStat> seg(num_robots);
  std::vector<std::pair<double, double>> robot_windows(num_robots);

  for (int r = 0; r < num_robots; ++r) {
    const size_t start = static_cast<size_t>(r) * poses_per_robot;
    const size_t end   = (r == num_robots - 1) ? all_poses.size() : start + poses_per_robot;
    std::vector<PoseInit> robot_poses(all_poses.begin() + start, all_poses.begin() + end);
    seg[r].poses     = robot_poses.size();
    seg[r].t0        = robot_poses.front().timestamp;
    seg[r].t1        = robot_poses.back().timestamp;
    robot_windows[r] = {seg[r].t0, seg[r].t1};

    add_pose_variable_nodes(*graphs[r], robot_poses);
    add_gp_prior_factor_nodes(*graphs[r], params.qc_diag);

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
    seg[r].landmarks = robot_landmarks.size();
    seg[r].stereo    = robot_stereo.size();

    add_landmark_variable_nodes(*graphs[r], robot_landmarks, r);
    add_stereo_factor_nodes(*graphs[r], robot_stereo, camera, r);
  }

  all_poses.clear();
  all_poses.shrink_to_fit();
  all_landmarks.clear();
  all_landmarks.shrink_to_fit();
  all_stereo.clear();
  all_stereo.shrink_to_fit();

  size_t n_equality = 0;
  if (num_robots > 1) {
    std::vector<size_t> pre(num_robots);
    for (int r = 0; r < num_robots; ++r)
      pre[r] = graphs[r]->factor_nodes_.size();
    std::vector<FactorGraph*> ptrs(num_robots);
    for (int r = 0; r < num_robots; ++r)
      ptrs[r] = graphs[r].get();
    add_landmark_equality_factor_nodes(ptrs);
    for (int r = 0; r < num_robots; ++r)
      n_equality += graphs[r]->factor_nodes_.size() - pre[r];
  }
  const double build_ms     = ms_since(t_build0);
  const long rss_post_build = peakRSSKb();

  // ---------------- solve (interleaved, as shipped) ----------------
  std::vector<double> robot_solve_ms(num_robots, 0.0);
  auto t_solve0 = Clock::now();
  for (int iter = 0; iter < iters; ++iter) {
    for (int r = 0; r < num_robots; ++r) {
      auto t0 = Clock::now();
      solvers[r]->performIteration(*graphs[r]);
      robot_solve_ms[r] += ms_since(t0);
    }
  }
  const double solve_ms     = ms_since(t_solve0);
  const long rss_post_solve = peakRSSKb();

  // ---------------- collect nodes ----------------
  std::vector<std::shared_ptr<SE3PoseVel>> all_pose_nodes;
  std::vector<std::unordered_map<std::string, std::shared_ptr<VariableNodeBase>>> byId(num_robots);
  for (int r = 0; r < num_robots; ++r) {
    for (const auto& var : graphs[r]->variable_nodes_) {
      byId[r][var->id_] = var;
      if (var->id_.find("POSE_") != std::string::npos) {
        auto node = std::dynamic_pointer_cast<SE3PoseVel>(var);
        if (node)
          all_pose_nodes.push_back(node);
      }
    }
  }
  std::sort(all_pose_nodes.begin(), all_pose_nodes.end(),
            [](const auto& a, const auto& b) { return a->timestamp_ < b->timestamp_; });

  // ---------------- GP interpolation at GT timestamps (as shipped) ----------
  std::vector<double> gt_ts;
  for (auto& g : gt)
    gt_ts.push_back(g.t);
  auto t_interp0 = Clock::now();
  auto interp    = GPInterpAtTimestamps(all_pose_nodes, gt_ts, params.qc_diag);
  const double interp_ms = ms_since(t_interp0);

  std::vector<TumPose> est;
  for (size_t i = 0; i < interp.size(); ++i)
    est.push_back({gt_ts[i], interp[i].pose});
  writeTumFile(out_dir + "/est_interp_gt_stamps.tum", est);

  bool nan_found = false;
  for (auto& e : est)
    if (!e.T.matrix().allFinite())
      nan_found = true;

  auto t_metrics0 = Clock::now();
  AteAre m = computeAteAre(est, gt);

  // ---------------- mean reprojection error ----------------
  auto robotOf = [&](double ts) {
    for (int r = 0; r < num_robots; ++r)
      if (ts >= robot_windows[r].first - 1e-9 && ts <= robot_windows[r].second + 1e-9)
        return r;
    return num_robots - 1;
  };

  std::unordered_map<long long, std::shared_ptr<SE3PoseVel>> poseByTsKey;
  auto tsKey = [](double t) { return static_cast<long long>(std::llround(t * 1e9)); };
  for (auto& n : all_pose_nodes)
    poseByTsKey[tsKey(n->timestamp_)] = n;

  // initial landmark positions / frame poses for the nominal series
  std::unordered_map<size_t, Eigen::Vector3d> initLm;
  for (auto& lm : eval_landmarks)
    initLm[lm.landmark_id] = lm.position;

  size_t n_total = 0, n_no_pose = 0, n_behind = 0, n_out = 0, n_incl = 0;
  double sum_err = 0, sum_err_all = 0, sum_err3 = 0; // err3: stereo 3-vector [u_l, v, disparity]
  size_t n_all = 0;
  const double fx = camera.fx, fy = camera.fy, cx = camera.cx, cy = camera.cy;
  const double baseline = camera.baseline();

  auto projectLeft = [&](const Eigen::Isometry3d& T, const Eigen::Vector3d& p, double& u, double& v, double& z) {
    const Eigen::Vector3d pc = T.inverse() * p;
    z                        = pc.z();
    u                        = fx * pc.x() / pc.z() + cx;
    v                        = fy * pc.y() / pc.z() + cy;
  };

  for (const auto& obs : eval_stereo) {
    ++n_total;
    auto pit = poseByTsKey.find(tsKey(obs.timestamp));
    if (pit == poseByTsKey.end()) {
      ++n_no_pose;
      continue;
    }
    const int r  = robotOf(obs.timestamp);
    auto lit     = byId[r].find(landmarkRobotId(obs.landmark_id, r));
    if (lit == byId[r].end()) {
      ++n_no_pose;
      continue;
    }
    auto ln = std::dynamic_pointer_cast<SE3Point>(lit->second);
    double u, v, z;
    projectLeft(pit->second->mu_.pose, ln->mu_, u, v, z);
    if (!(z > 0)) {
      ++n_behind;
      continue;
    }
    const double err = std::hypot(u - obs.observation(0), v - obs.observation(1));
    sum_err_all += err;
    ++n_all;
    if (u < 0 || u > img_w - 1 || v < 0 || v > img_h - 1) {
      ++n_out;
      continue;
    }
    sum_err += err;
    // stereo 3-vector error as modelled by SE3PoseVelStereoPoint: [u_l, v, disp]
    {
      const Eigen::Vector3d pc = pit->second->mu_.pose.inverse() * ln->mu_;
      const double disp_est    = fx * baseline / pc.z();
      const Eigen::Vector3d h(u, v, disp_est);
      const Eigen::Vector3d z(obs.observation(0), obs.observation(1), obs.observation(0) - obs.observation(2));
      sum_err3 += (z - h).norm();
    }
    ++n_incl;
  }
  const double metrics_ms = ms_since(t_metrics0);

  // ---------------- frame projection dumps ----------------
  if (!dump_frames.empty()) {
    // frame index -> gt timestamp; obs -> nearest gt frame
    for (int f : dump_frames) {
      if (f < 0 || f >= static_cast<int>(gt_ts.size()))
        continue;
      const double tf = gt_ts[f];
      // nominal (global-shutter) pose: initial per-frame ORB-SLAM3 pose = the
      // parsed initial pose whose timestamp is nearest to the frame time
      // (all row-blocks of a frame share the same initial pose).
      const PoseInit* nominal = nullptr;
      double best = 1e18;
      for (const auto& p : eval_poses) {
        const double d = std::abs(p.timestamp - tf);
        if (d < best) {
          best    = d;
          nominal = &p;
        }
      }
      std::ofstream ff(out_dir + "/projections_frame_" + std::to_string(f) + ".csv");
      ff << "frame,obs_timestamp,landmark_id,u_obs,v_obs,"
            "u_gsolver,v_gsolver,depth_gsolver,in_image_gsolver,"
            "u_nominal_optlm,v_nominal_optlm,in_image_nominal_optlm,"
            "u_nominal_initlm,v_nominal_initlm,in_image_nominal_initlm\n";
      ff << std::setprecision(10);
      const double half_frame = 0.05; // frames ~10 Hz; RS offsets << this at 1 ms
      for (const auto& obs : eval_stereo) {
        if (std::abs(obs.timestamp - tf) > half_frame)
          continue;
        // confirm this obs's nearest frame is f (not a neighbour)
        auto it   = std::lower_bound(gt_ts.begin(), gt_ts.end(), obs.timestamp);
        int fnear = static_cast<int>(std::distance(gt_ts.begin(), it));
        if (fnear > 0 && (fnear == static_cast<int>(gt_ts.size()) ||
                          std::abs(gt_ts[fnear - 1] - obs.timestamp) < std::abs(gt_ts[fnear] - obs.timestamp)))
          fnear -= 1;
        if (fnear != f)
          continue;
        auto pit = poseByTsKey.find(tsKey(obs.timestamp));
        if (pit == poseByTsKey.end())
          continue;
        const int r = robotOf(obs.timestamp);
        auto lit    = byId[r].find(landmarkRobotId(obs.landmark_id, r));
        if (lit == byId[r].end())
          continue;
        auto ln = std::dynamic_pointer_cast<SE3Point>(lit->second);

        double ug, vg, zg;
        projectLeft(pit->second->mu_.pose, ln->mu_, ug, vg, zg);
        const bool in_g = zg > 0 && ug >= 0 && ug <= img_w - 1 && vg >= 0 && vg <= img_h - 1;
        double un, vn, zn;
        projectLeft(nominal->pose, ln->mu_, un, vn, zn);
        const bool in_n = zn > 0 && un >= 0 && un <= img_w - 1 && vn >= 0 && vn <= img_h - 1;
        double ui = std::numeric_limits<double>::quiet_NaN(), vi = ui;
        bool in_i = false;
        auto lmi  = initLm.find(obs.landmark_id);
        if (lmi != initLm.end()) {
          double zi;
          projectLeft(nominal->pose, lmi->second, ui, vi, zi);
          in_i = zi > 0 && ui >= 0 && ui <= img_w - 1 && vi >= 0 && vi <= img_h - 1;
        }
        ff << f << "," << obs.timestamp << "," << obs.landmark_id << "," << obs.observation(0) << "," << obs.observation(1)
           << "," << ug << "," << vg << "," << zg << "," << in_g << "," << un << "," << vn << "," << in_n << "," << ui << ","
           << vi << "," << in_i << "\n";
      }
    }
  }

  // ---------------- outputs ----------------
  {
    std::ofstream f(out_dir + "/per_robot.csv");
    f << "robot,poses,landmarks,stereo_obs,t_start,t_end,solve_ms\n";
    for (int r = 0; r < num_robots; ++r)
      f << r << "," << seg[r].poses << "," << seg[r].landmarks << "," << seg[r].stereo << "," << seg[r].t0 << "," << seg[r].t1
        << "," << robot_solve_ms[r] << "\n";
  }

  std::ofstream s(out_dir + "/summary.csv");
  s << "input,config,num_robots,iterations,threads,run_status,"
       "n_poses,n_landmarks,n_stereo,n_equality_factors,"
       "ate_m,are_rad,ate_aligned_m,are_aligned_rad,n_matched,"
       "mre_px_in_image,mre_stereo3_in_image,n_included,n_excluded_out_of_frame,n_excluded_behind,n_unmatched,"
       "mre_px_all_valid_depth,n_all_valid_depth,excluded_pct,"
       "load_ms,build_ms,solve_ms,avg_robot_solve_ms,interp_ms,metrics_ms,"
       "rss_post_load_kb,rss_post_build_kb,rss_post_solve_kb,img_w,img_h\n";
  s << std::setprecision(10);
  double avg_robot = 0;
  for (double v : robot_solve_ms)
    avg_robot += v;
  avg_robot /= num_robots;
  size_t tot_lm = 0, tot_st = 0;
  for (auto& sg : seg) {
    tot_lm += sg.landmarks;
    tot_st += sg.stereo;
  }
  s << input_path << "," << cfg_key << "," << num_robots << "," << iters << "," << omp_get_max_threads() << ","
    << (nan_found ? "diverged_nan" : "ok") << "," << eval_poses.size() << "," << tot_lm << "," << tot_st << "," << n_equality
    << "," << m.ate << "," << m.are << "," << m.ate_aligned << "," << m.are_aligned << "," << m.matched << ","
    << (n_incl ? sum_err / n_incl : -1) << "," << (n_incl ? sum_err3 / n_incl : -1) << "," << n_incl << "," << n_out << ","
    << n_behind << "," << n_no_pose << ","
    << (n_all ? sum_err_all / n_all : -1) << "," << n_all << "," << (n_total ? 100.0 * (n_total - n_incl) / n_total : -1) << ","
    << load_ms << "," << build_ms << "," << solve_ms << "," << avg_robot << "," << interp_ms << "," << metrics_ms << ","
    << rss_post_load << "," << rss_post_build << "," << rss_post_solve << "," << img_w << "," << img_h << "\n";

  std::cout << "[repro_stereo] robots=" << num_robots << " ATE=" << m.ate << " ARE=" << m.are
            << " MRE=" << (n_incl ? sum_err / n_incl : -1) << " (incl " << n_incl << "/" << n_total << ") solve_ms=" << solve_ms
            << "\n";
  return 0;
}
