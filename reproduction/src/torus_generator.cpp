/**
 * @file torus_generator.cpp
 * @brief SUBSTITUTE torus dataset generator (clearly-labelled reproduction aid).
 *
 * The paper uses a torus motion sequence for Fig. 1 and Fig. 10, but the public
 * repository contains NO torus generator, dataset or hyperparameters (its
 * generator supports helix and sphere only). This tool creates a locally
 * defined torus trajectory so that a SUBSTITUTE reproduction can be produced.
 * Outputs from this tool must never be presented as the paper's original torus.
 *
 * Conventions follow examples/data/generators/ exactly: same g2o tokens, same
 * noise model (translation sigma [m], rotation sigma [rad], right-multiplied
 * expMapSE3 perturbation), std::mt19937 with fixed seed.
 *
 * Trajectory: (R + r cos(k*2pi*t)) * [cos(2pi*t), sin(2pi*t)], z = r sin(k*2pi*t),
 * yaw following the toroidal angle — same "yaw along path" style as the helix
 * generator. Defaults: R=10, r=3, k=8 poloidal loops, 600 poses at 1 Hz.
 *
 * Usage:
 *   torus_generator --out prefix --mode gt|prior|pgo [--hz 1] [--n 600]
 *     [--sigma-t 0.001] [--sigma-r 0.0001] [--ig-t 1.0] [--ig-r 0.1]
 *     [--seed 42] [--gap-start f] [--gap-end f] [--gap-keep m]
 *
 * --gap-*: for the Fig. 1 substitute, priors inside the trajectory fraction
 * [gap-start, gap-end) are kept only every m-th pose (sparse-observation region).
 */
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <vector>

#include <Eigen/Dense>
#include <gsolver/maths/geometry3d.h>

using namespace gsolver;
using Vector6d = Eigen::Matrix<double, 6, 1>;

static std::string upperTri(const Eigen::Matrix<double, 6, 6>& m) {
  std::ostringstream ss;
  for (int i = 0; i < 6; ++i)
    for (int j = i; j < 6; ++j)
      ss << m(i, j) << " ";
  return ss.str();
}

static Eigen::Isometry3d addNoise(const Eigen::Isometry3d& T, double st, double sr, std::mt19937& gen) {
  std::normal_distribution<double> nt(0.0, st), nr(0.0, sr);
  Vector6d n;
  n << nt(gen), nt(gen), nt(gen), nr(gen), nr(gen), nr(gen);
  return T * expMapSE3(n);
}

int main(int argc, char* argv[]) {
  std::map<std::string, std::string> opt;
  for (int i = 1; i + 1 < argc; i += 2)
    opt[std::string(argv[i]).substr(2)] = argv[i + 1];
  auto getd = [&](const std::string& k, double d) { return opt.count(k) ? std::stod(opt[k]) : d; };
  auto geti = [&](const std::string& k, int d) { return opt.count(k) ? std::stoi(opt[k]) : d; };

  const std::string out  = opt.count("out") ? opt["out"] : "torus";
  const std::string mode = opt.count("mode") ? opt["mode"] : "gt";
  const double hz        = getd("hz", 1.0);
  const int n_base       = geti("n", 600);
  const double R         = getd("R", 10.0);
  const double r         = getd("r", 3.0);
  const int k_loops      = geti("k", 8);
  const double sig_t     = getd("sigma-t", 0.0);
  const double sig_r     = getd("sigma-r", 0.0);
  const double ig_t      = getd("ig-t", 0.0);
  const double ig_r      = getd("ig-r", 0.0);
  const int seed         = geti("seed", 42);
  const double gap0      = getd("gap-start", -1.0);
  const double gap1      = getd("gap-end", -1.0);
  const int gap_keep     = geti("gap-keep", 1);
  const double lc_pct    = getd("loop-closures-pct", 50.0); // as generate_configs.py

  const int num_samples = static_cast<int>((n_base - 1) * hz);
  std::vector<double> ts;
  std::vector<Eigen::Isometry3d> traj;
  for (int i = 0; i <= num_samples; ++i) {
    const double t   = static_cast<double>(i) / (num_samples); // 0..1
    const double phi = 2.0 * M_PI * t;                          // toroidal
    const double u   = 2.0 * M_PI * k_loops * t;                // poloidal
    Eigen::Vector3d p((R + r * std::cos(u)) * std::cos(phi), (R + r * std::cos(u)) * std::sin(phi), r * std::sin(u));
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear()          = expMapSO3(Eigen::Vector3d(0, 0, phi));
    T.translation()     = p;
    traj.push_back(T);
    ts.push_back(static_cast<double>(i) / hz);
  }

  std::mt19937 gen(seed);

  if (mode == "gt") {
    std::ofstream f(out + ".tum");
    f << "#timestamp tx ty tz qx qy qz qw\n";
    for (size_t i = 0; i < traj.size(); ++i) {
      Eigen::Quaterniond q(traj[i].linear());
      const auto& t = traj[i].translation();
      f << ts[i] << " " << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
        << "\n";
    }
    std::cout << "[torus_generator SUBSTITUTE] wrote " << traj.size() << " GT poses to " << out << ".tum\n";
    return 0;
  }

  // vertices (initial guesses perturbed with ig noise), matching data_generator
  Eigen::Matrix<double, 6, 6> sig_ig = Eigen::Matrix<double, 6, 6>::Identity();
  sig_ig.block<3, 3>(0, 0) *= ig_t * ig_t;
  sig_ig.block<3, 3>(3, 3) *= ig_r * ig_r;

  std::ofstream f(out + ".g2o");
  for (size_t i = 0; i < traj.size(); ++i) {
    Eigen::Isometry3d T = traj[i];
    if (ig_t > 0 || ig_r > 0)
      T = addNoise(T, ig_t, ig_r, gen);
    Eigen::Quaterniond q(T.linear());
    const auto& t = T.translation();
    f << "VERTEX_SE3 " << ts[i] << " " << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << " " << q.y() << " " << q.z()
      << " " << q.w() << " " << upperTri(sig_ig) << "\n";
  }

  Eigen::Matrix<double, 6, 6> sig_m = Eigen::Matrix<double, 6, 6>::Identity();
  sig_m.block<3, 3>(0, 0) *= sig_t * sig_t;
  sig_m.block<3, 3>(3, 3) *= sig_r * sig_r;

  if (mode == "prior") {
    size_t kept = 0;
    for (size_t i = 0; i < traj.size(); ++i) {
      const double frac = static_cast<double>(i) / (traj.size() - 1);
      const bool in_gap = (gap0 >= 0 && frac >= gap0 && frac < gap1);
      if (in_gap && (i % std::max(1, gap_keep)) != 0)
        continue;
      Eigen::Isometry3d z = traj[i];
      if (sig_t > 0 || sig_r > 0)
        z = addNoise(z, sig_t, sig_r, gen);
      Eigen::Quaterniond q(z.linear());
      const auto& t = z.translation();
      f << "EDGE_PRIOR_SE3 " << ts[i] << " " << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << " " << q.y() << " "
        << q.z() << " " << q.w() << " " << upperTri(sig_m) << "\n";
      ++kept;
    }
    std::cout << "[torus_generator SUBSTITUTE] " << traj.size() << " vertices, " << kept << " priors -> " << out << ".g2o\n";
  } else if (mode == "pgo") {
    size_t cnt = 0;
    for (size_t i = 0; i + 1 < traj.size(); ++i) {
      Eigen::Isometry3d rel = traj[i].inverse() * traj[i + 1];
      if (sig_t > 0 || sig_r > 0)
        rel = addNoise(rel, sig_t, sig_r, gen);
      Eigen::Quaterniond q(rel.linear());
      const auto& t = rel.translation();
      f << "EDGE_POSE_POSE_SE3 " << ts[i] << " " << ts[i + 1] << " " << t.x() << " " << t.y() << " " << t.z() << " " << q.x()
        << " " << q.y() << " " << q.z() << " " << q.w() << " " << upperTri(sig_m) << "\n";
      ++cnt;
    }
    // random loop closures, same scheme as data_generator (50% of poses)
    const size_t n_lc = static_cast<size_t>(lc_pct * traj.size() / 100.0);
    std::uniform_int_distribution<size_t> dist(0, traj.size() - 1);
    for (size_t l = 0; l < n_lc; ++l) {
      size_t a, b;
      do {
        a = dist(gen);
        b = dist(gen);
      } while (a == b || std::abs(static_cast<int>(a) - static_cast<int>(b)) == 1);
      Eigen::Isometry3d rel = traj[a].inverse() * traj[b];
      if (sig_t > 0 || sig_r > 0)
        rel = addNoise(rel, sig_t, sig_r, gen);
      Eigen::Quaterniond q(rel.linear());
      const auto& t = rel.translation();
      f << "EDGE_POSE_POSE_SE3 " << ts[a] << " " << ts[b] << " " << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << " "
        << q.y() << " " << q.z() << " " << q.w() << " " << upperTri(sig_m) << "\n";
      ++cnt;
    }
    std::cout << "[torus_generator SUBSTITUTE] " << traj.size() << " vertices, " << cnt << " pose-pose edges -> " << out
              << ".g2o\n";
  }
  return 0;
}
