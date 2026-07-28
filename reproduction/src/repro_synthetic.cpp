/**
 * @file repro_synthetic.cpp
 * @brief Instrumented reproduction driver for the synthetic experiments
 *        (paper Figs. 1*, 5, 6, 7, 8, 9, 10*, 15-18 and Table I; * = substitute).
 *
 * Graph construction and solving replicate examples/cpp/apps/{prior,pgo}_experiment.cpp
 * and hyperparam_ablation_experiment.cpp EXACTLY (same builder calls, same
 * SYNCHRONOUS schedule, same iteration counts from the config file unless
 * explicitly overridden on the command line for a documented reason).
 * All added instrumentation is read-only between iterations.
 *
 * The optional --gn flag additionally runs an INDEPENDENTLY IMPLEMENTED batch
 * Gauss-Newton on the same total energy (Eq. 18). This is NOT the paper's GN+GP
 * baseline (that implementation is not public); results are labelled
 * "gn_gp_reimplementation".
 *
 * Usage:
 *  repro_synthetic --input in.g2o --config helix --params experiment_params.yaml
 *      --task prior|pgo|prior_nogp|ablation --out outdir
 *      [--gt gt_1hz.tum] [--gt100 gt_100hz.tum]
 *      [--iters N] [--qc-scale s] [--energy] [--interp] [--nees] [--gn] [--pure]
 */
#include <filesystem>
#include <iostream>
#include <map>
#include <omp.h>
#include <set>

#include <Eigen/Sparse>

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

// --------------------------------------------------------------------------
// Replicates the interpolation-timestamp generation loop of GPInterp()
// (gsolver/maths/gp_interpolation.h) so that exported samples carry stamps.
// --------------------------------------------------------------------------
static std::vector<double> interpTimestamps(const std::vector<std::shared_ptr<SE3PoseVel>>& nodes, int hertz) {
  std::vector<double> ts;
  const size_t N = nodes.size();
  double t_curr  = nodes[0]->timestamp_;
  for (size_t i = 0; i + 1 < N; ++i) {
    const double dt        = nodes[i + 1]->timestamp_ - nodes[i]->timestamp_;
    const double dt_interp = dt / hertz;
    for (int j = 0; j < hertz; ++j) {
      ts.push_back(t_curr);
      t_curr += dt_interp;
    }
  }
  return ts;
}

// --------------------------------------------------------------------------
// Independent Gauss-Newton on the same energy (supplementary; labelled).
// Right-multiplicative pose perturbation, additive velocity, numerical
// Jacobians (central differences), sparse normal equations, tiny fixed
// damping 1e-8 for the PGO gauge null-space.
// --------------------------------------------------------------------------
namespace gn {
  struct State {
    std::vector<double> ts;
    std::vector<Eigen::Isometry3d> T;
    std::vector<Vector6d> v;
  };

  template <typename ErrFn>
  static Eigen::MatrixXd numJac(const ErrFn& f, int err_dim, int n_params) {
    Eigen::MatrixXd J(err_dim, n_params);
    const double h = 1e-6;
    for (int k = 0; k < n_params; ++k) {
      Eigen::VectorXd d = Eigen::VectorXd::Zero(n_params);
      d(k)              = h;
      Eigen::VectorXd ep = f(d);
      d(k)               = -h;
      Eigen::VectorXd em = f(d);
      J.col(k)           = (ep - em) / (2 * h);
    }
    return J;
  }

  // returns per-iteration total energy (index 0 = initial state)
  static std::vector<double> run(State st,
                                 const std::vector<PriorMeas>& priors,
                                 const std::vector<OdometryMeas>& odom,
                                 const Vector6d& qc,
                                 int iters) {
    const int N = static_cast<int>(st.ts.size());
    std::map<double, int> idx;
    for (int i = 0; i < N; ++i)
      idx[st.ts[i]] = i;

    auto energy = [&]() {
      std::map<double, PoseVelState> states;
      for (int i = 0; i < N; ++i)
        states[st.ts[i]] = PoseVelState{st.T[i], st.v[i]};
      return evalEnergy(states, priors, odom, qc, true).total();
    };

    std::vector<double> E;
    E.push_back(energy());

    for (int it = 0; it < iters; ++it) {
      std::vector<Eigen::Triplet<double>> trips;
      Eigen::VectorXd b = Eigen::VectorXd::Zero(12 * N);

      auto addBlock = [&](const std::vector<int>& vars, const Eigen::MatrixXd& J, const Eigen::VectorXd& e,
                          const Eigen::MatrixXd& Sinv) {
        const Eigen::MatrixXd JtS = J.transpose() * Sinv;
        const Eigen::MatrixXd H   = JtS * J;
        const Eigen::VectorXd g   = JtS * e;
        for (size_t a = 0; a < vars.size(); ++a) {
          b.segment<12>(12 * vars[a]) -= g.segment<12>(12 * a);
          for (size_t c = 0; c < vars.size(); ++c)
            for (int r = 0; r < 12; ++r)
              for (int q = 0; q < 12; ++q)
                trips.emplace_back(12 * vars[a] + r, 12 * vars[c] + q, H(12 * a + r, 12 * c + q));
        }
      };

      for (const auto& pr : priors) {
        auto f = idx.find(pr.timestamp);
        if (f == idx.end())
          continue;
        const int i  = f->second;
        auto errOf   = [&](const Eigen::VectorXd& d) -> Eigen::VectorXd {
          const Eigen::Isometry3d Ti = st.T[i] * expMapSE3(d.segment<6>(0));
          return logMapSE3(pr.pose.inverse() * Ti);
        };
        const Eigen::VectorXd e0 = errOf(Eigen::VectorXd::Zero(12));
        const Eigen::MatrixXd J  = numJac(errOf, 6, 12);
        const Eigen::Matrix<double, 6, 6> S = pr.covariance + 1e-6 * Eigen::Matrix<double, 6, 6>::Identity();
        addBlock({i}, J, e0, S.inverse());
      }
      for (const auto& od : odom) {
        auto ff = idx.find(od.timestamp_from);
        auto ft = idx.find(od.timestamp_to);
        if (ff == idx.end() || ft == idx.end())
          continue;
        const int i = ff->second, j = ft->second;
        auto errOf = [&](const Eigen::VectorXd& d) -> Eigen::VectorXd {
          const Eigen::Isometry3d Ti = st.T[i] * expMapSE3(d.segment<6>(0));
          const Eigen::Isometry3d Tj = st.T[j] * expMapSE3(d.segment<6>(12));
          return logMapSE3(od.measurement.inverse() * (Ti.inverse() * Tj));
        };
        const Eigen::VectorXd e0 = errOf(Eigen::VectorXd::Zero(24));
        const Eigen::MatrixXd J  = numJac(errOf, 6, 24);
        const Eigen::Matrix<double, 6, 6> S = od.covariance + 1e-6 * Eigen::Matrix<double, 6, 6>::Identity();
        addBlock({i, j}, J, e0, S.inverse());
      }
      for (int i = 0; i + 1 < N; ++i) {
        const double dt = st.ts[i + 1] - st.ts[i];
        if (dt <= 0)
          continue;
        auto errOf = [&](const Eigen::VectorXd& d) -> Eigen::VectorXd {
          const Eigen::Isometry3d Ti = st.T[i] * expMapSE3(d.segment<6>(0));
          const Vector6d vi          = st.v[i] + d.segment<6>(6);
          const Eigen::Isometry3d Tj = st.T[i + 1] * expMapSE3(d.segment<6>(12));
          const Vector6d vj          = st.v[i + 1] + d.segment<6>(18);
          Eigen::VectorXd e(12);
          e.head<6>() = gpErrorPose(Ti, Tj, vi, dt);
          e.tail<6>() = gpErrorVel(Ti, Tj, vi, vj);
          return e;
        };
        const Eigen::VectorXd e0 = errOf(Eigen::VectorXd::Zero(24));
        const Eigen::MatrixXd J  = numJac(errOf, 12, 24);
        addBlock({i, i + 1}, J, e0, gpQd(qc, dt).inverse());
      }

      Eigen::SparseMatrix<double> H(12 * N, 12 * N);
      H.setFromTriplets(trips.begin(), trips.end());

      // Levenberg-Marquardt damping: accept only cost-decreasing steps.
      // (Reimplementation choice, documented; the paper's GN internals are unknown.)
      static double lambda = 1e-6;
      const double E_prev  = E.back();
      auto stBackup        = st;
      bool accepted        = false;
      for (int tries = 0; tries < 12 && !accepted; ++tries) {
        Eigen::SparseMatrix<double> Hd = H;
        for (int k = 0; k < 12 * N; ++k)
          Hd.coeffRef(k, k) += lambda * (1.0 + std::abs(H.coeff(k, k)));
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(Hd);
        if (solver.info() == Eigen::Success) {
          const Eigen::VectorXd dx = solver.solve(b);
          for (int i = 0; i < N; ++i) {
            st.T[i] = stBackup.T[i] * expMapSE3(dx.segment<6>(12 * i));
            st.v[i] = stBackup.v[i] + dx.segment<6>(12 * i + 6);
          }
          const double E_new = energy();
          if (std::isfinite(E_new) && E_new <= E_prev) {
            accepted = true;
            lambda   = std::max(1e-9, lambda * 0.3);
            E.push_back(E_new);
            break;
          }
        }
        st = stBackup;
        lambda *= 10.0;
      }
      if (!accepted)
        E.push_back(E_prev); // no acceptable step; report unchanged energy
    }
    return E;
  }
} // namespace gn

// --------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  std::map<std::string, std::string> opt;
  std::set<std::string> flags;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--", 0) == 0) {
      if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
        opt[a.substr(2)] = argv[++i];
      else
        flags.insert(a.substr(2));
    }
  }
  auto need = [&](const std::string& k) {
    if (!opt.count(k)) {
      std::cerr << "missing --" << k << "\n";
      exit(1);
    }
    return opt[k];
  };

  const std::string input_path = need("input");
  const std::string cfg_key    = need("config");
  const std::string params_path = need("params");
  const std::string task       = need("task"); // prior | pgo | prior_nogp | ablation
  const std::string out_dir    = need("out");
  const double qc_scale        = opt.count("qc-scale") ? std::stod(opt["qc-scale"]) : 1.0;
  const bool do_energy         = flags.count("energy");
  const bool do_interp         = flags.count("interp");
  const bool do_nees           = flags.count("nees");
  const bool do_gn             = flags.count("gn");
  const bool pure              = flags.count("pure"); // no per-iteration instrumentation
  std::filesystem::create_directories(out_dir);

  auto params = loadExperimentParams(params_path, cfg_key);
  const int iters = opt.count("iters") ? std::stoi(opt["iters"]) : params.num_iterations;
  const Vector6d qc_eff = params.qc_diag * qc_scale;

  omp_set_num_threads(omp_get_max_threads()); // as in the shipped apps
  const int threads = omp_get_max_threads();

  // ---------------- load ----------------
  auto t_load0 = Clock::now();
  std::vector<PoseInit> poses;
  std::vector<OdometryMeas> odometry;
  std::vector<PriorMeas> priors;
  std::vector<LandmarkMeas> landmarks;
  parseG2OFile(input_path, poses, odometry, priors, landmarks);
  const double load_ms = ms_since(t_load0);

  std::vector<TumPose> gt1, gt100;
  if (opt.count("gt"))
    gt1 = loadTum(opt["gt"]);
  if (opt.count("gt100"))
    gt100 = loadTum(opt["gt100"]);

  // ---------------- build (replicates the shipped apps) ----------------
  auto t_build0 = Clock::now();
  FactorGraph fg;
  if (task == "prior") {
    add_pose_variable_nodes(fg, poses);
    add_prior_factor_nodes(fg, priors);
    add_gp_prior_factor_nodes(fg, qc_eff);
  } else if (task == "prior_nogp") {
    add_pose_variable_nodes(fg, poses);
    add_prior_factor_nodes(fg, priors);
  } else if (task == "pgo") {
    add_pose_variable_nodes(fg, poses);
    add_odometry_factor_nodes(fg, odometry);
    add_gp_prior_factor_nodes(fg, qc_eff);
  } else if (task == "ablation") { // replicates hyperparam_ablation_experiment
    add_pose_variable_nodes(fg, poses);
    add_prior_factor_nodes(fg, priors);
    add_gp_prior_factor_nodes(fg, qc_eff);
    add_odometry_factor_nodes(fg, odometry);
  } else {
    std::cerr << "unknown task " << task << "\n";
    return 1;
  }
  const double build_ms = ms_since(t_build0);
  const bool with_gp    = (task != "prior_nogp");

  // initial guess dump
  {
    std::vector<TumPose> init;
    for (const auto& p : poses)
      init.push_back({p.timestamp, p.pose});
    writeTumFile(out_dir + "/knots_init.tum", init);
  }

  // ---------------- solve ----------------
  GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
  std::ofstream energy_csv;
  if (do_energy && !pure) {
    energy_csv.open(out_dir + "/energy.csv");
    energy_csv << "iteration,method,E_meas,E_gp,total_energy\n";
    auto E0 = evalEnergy(snapshotStates(fg), priors, odometry, qc_eff, with_gp);
    energy_csv << 0 << ",gsolver," << E0.meas << "," << E0.gp << "," << E0.total() << "\n";
  }

  std::vector<double> iter_ms;
  iter_ms.reserve(iters);
  auto t_solve0 = Clock::now();
  for (int i = 0; i < iters; ++i) {
    auto t0 = Clock::now();
    solver.performIteration(fg);
    iter_ms.push_back(ms_since(t0));
    if (do_energy && !pure) {
      auto E = evalEnergy(snapshotStates(fg), priors, odometry, qc_eff, with_gp);
      energy_csv << (i + 1) << ",gsolver," << E.meas << "," << E.gp << "," << E.total() << "\n";
    }
  }
  const double solve_ms = ms_since(t_solve0);

  // ---------------- outputs ----------------
  auto states = snapshotStates(fg);
  const std::string run_status = hasNaN(states) ? "diverged_nan" : "ok";

  std::vector<std::shared_ptr<SE3PoseVel>> nodes;
  for (auto& vn : fg.getPoseVariableNodes()) {
    auto n = std::dynamic_pointer_cast<SE3PoseVel>(vn);
    if (n)
      nodes.push_back(n);
  }

  std::vector<TumPose> est;
  for (auto& n : nodes)
    est.push_back({n->timestamp_, n->mu_.pose});
  writeTumFile(out_dir + "/knots_est.tum", est);

  auto t_metrics0 = Clock::now();
  AteAre knots_m;
  if (!gt1.empty())
    knots_m = computeAteAre(est, gt1);

  NeesResult nees;
  if (do_nees && !gt1.empty()) {
    std::vector<Eigen::Vector4d> per_pose;
    nees = computeNees(nodes, gt1, &per_pose);
    std::ofstream nf(out_dir + "/nees_per_pose.csv");
    nf << "timestamp,nees_local_est,nees_local_gt,nees_global\n";
    for (auto& r : per_pose)
      nf << r(0) << "," << r(1) << "," << r(2) << "," << r(3) << "\n";
  }
  const double metrics_ms = ms_since(t_metrics0);

  // interpolation 100x (official GPInterpAtTimestamps; stamps replicate GPInterp)
  double interp_ms = 0;
  AteAre interp_m;
  size_t n_interp = 0;
  if (do_interp && with_gp && nodes.size() >= 2) {
    auto t_i0     = Clock::now();
    auto ts       = interpTimestamps(nodes, 100);
    auto interp   = GPInterpAtTimestamps(nodes, ts, qc_eff);
    interp_ms     = ms_since(t_i0);
    n_interp      = interp.size();
    std::ofstream f(out_dir + "/interp.csv");
    f << "t,x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz,cov_xx,cov_yy,cov_zz,cov_xy,cov_xz,cov_yz\n";
    f << std::setprecision(12);
    std::vector<TumPose> ipose;
    for (size_t i = 0; i < interp.size(); ++i) {
      const auto& d = interp[i];
      Eigen::Quaterniond q(d.pose.linear());
      const auto& t = d.pose.translation();
      f << ts[i] << "," << t.x() << "," << t.y() << "," << t.z() << "," << q.x() << "," << q.y() << "," << q.z() << "," << q.w()
        << "," << d.vel(0) << "," << d.vel(1) << "," << d.vel(2) << "," << d.vel(3) << "," << d.vel(4) << "," << d.vel(5) << ","
        << d.covariance(0, 0) << "," << d.covariance(1, 1) << "," << d.covariance(2, 2) << "," << d.covariance(0, 1) << ","
        << d.covariance(0, 2) << "," << d.covariance(1, 2) << "\n";
      ipose.push_back({ts[i], d.pose});
    }
    if (!gt100.empty()) {
      const size_t n = std::min(ipose.size(), gt100.size());
      bool stamps_ok = true;
      for (size_t i = 0; i < n; i += 997)
        if (std::abs(ipose[i].t - gt100[i].t) > 1e-6)
          stamps_ok = false;
      if (stamps_ok)
        interp_m = computeAteAre(std::vector<TumPose>(ipose.begin(), ipose.begin() + n),
                                 std::vector<TumPose>(gt100.begin(), gt100.begin() + n));
      else
        std::cerr << "[warn] interp/gt100 stamps mismatch; interp ATE skipped\n";
    }
  }

  // GN+GP reimplementation (supplementary)
  if (do_gn) {
    gn::State st;
    for (const auto& p : poses)
      st.ts.push_back(p.timestamp);
    // same initial velocities as add_pose_variable_nodes (finite differences)
    for (size_t i = 0; i + 1 < poses.size(); ++i) {
      const double dt = poses[i + 1].timestamp - poses[i].timestamp;
      st.v.push_back(logMapSE3(poses[i].pose.inverse() * poses[i + 1].pose) / dt);
    }
    st.v.push_back(st.v.back());
    for (const auto& p : poses)
      st.T.push_back(p.pose);
    const int gn_iters = opt.count("gn-iters") ? std::stoi(opt["gn-iters"]) : iters;
    auto E = gn::run(st, priors, odometry, qc_eff, gn_iters);
    std::ofstream f(out_dir + "/energy_gn_reimplementation.csv");
    f << "iteration,method,total_energy\n";
    for (size_t i = 0; i < E.size(); ++i)
      f << i << ",gn_gp_reimplementation," << E[i] << "\n";
  }

  // per-iteration timing
  if (!pure) {
    std::ofstream f(out_dir + "/iteration_times.csv");
    f << "iteration,ms\n";
    for (size_t i = 0; i < iter_ms.size(); ++i)
      f << (i + 1) << "," << iter_ms[i] << "\n";
  }

  double avg_iter = 0;
  for (double m : iter_ms)
    avg_iter += m;
  avg_iter /= std::max<size_t>(1, iter_ms.size());

  std::ofstream s(out_dir + "/summary.csv");
  s << "task,config,input,qc_scale,iterations,threads,run_status,pure_timing,"
       "n_poses,n_priors,n_odometry,n_gp,"
       "ate_knots_m,are_knots_rad,ate_knots_aligned_m,are_knots_aligned_rad,"
       "ate_interp_m,are_interp_rad,ate_interp_aligned_m,are_interp_aligned_rad,n_interp,"
       "nees_local_est,nees_local_gt,nees_global,"
       "load_ms,build_ms,solve_ms,avg_iter_ms,interp_ms,metrics_ms,peak_rss_kb\n";
  s << std::setprecision(10);
  s << task << "," << cfg_key << "," << input_path << "," << qc_scale << "," << iters << "," << threads << "," << run_status
    << "," << (pure ? 1 : 0) << "," << poses.size() << "," << priors.size() << "," << odometry.size() << ","
    << (with_gp ? poses.size() - 1 : 0) << "," << knots_m.ate << "," << knots_m.are << "," << knots_m.ate_aligned << ","
    << knots_m.are_aligned << "," << interp_m.ate << "," << interp_m.are << "," << interp_m.ate_aligned << ","
    << interp_m.are_aligned << "," << n_interp << "," << nees.mean_local_est << "," << nees.mean_local_gt << ","
    << nees.mean_global << "," << load_ms << "," << build_ms << "," << solve_ms << "," << avg_iter << "," << interp_ms << ","
    << metrics_ms << "," << peakRSSKb() << "\n";

  std::cout << "[repro_synthetic] " << task << " " << cfg_key << " status=" << run_status << " ate_knots=" << knots_m.ate
            << " solve_ms=" << solve_ms << "\n";
  return 0;
}
