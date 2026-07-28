/**
 * @file repro_bench.cpp
 * @brief CPU latency microbenchmarks (paper Fig. 12) — local fallback protocol.
 *
 * The repository contains NO benchmark harness, so the paper's exact protocol
 * (graph, repetitions, warm-up) is unknown. This tool implements the fallback
 * documented in the reproduction protocol:
 *   - Release build, no console output inside timed regions, no visualisation,
 *   - >= 10 warm-up solver iterations to reach steady state,
 *   - >= 1000 measured repetitions per operation (default 5000),
 *   - std::chrono::steady_clock (monotonic), every raw sample saved.
 *
 * Operations timed (single calls on individual nodes, one thread):
 *   var_to_fac    : SE3PoseVel::sendMessage(edge_index)      (variable->factor)
 *   fac_to_var    : FactorNodeBase::sendMessage(variable_id) (factor->variable)
 *   factor_update : FactorNodeBase::updateFactor()
 *   var_update    : VariableNodeBase::updateMuAndSigma()
 *   interpolation : GPInterpAtTimestamps (single mid-interval query)
 *
 * Usage: repro_bench --input prior.g2o --config helix --params yaml --out dir
 *        [--samples 5000] [--warmup-iters 10]
 */
#include <filesystem>
#include <iostream>
#include <map>
#include <omp.h>

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

struct Stats {
  double mean = 0, med = 0, sd = 0, mn = 0, mx = 0, p5 = 0, p25 = 0, p75 = 0, p95 = 0;
  size_t n = 0;
};

static Stats stats(std::vector<double> v) {
  Stats s;
  s.n = v.size();
  if (v.empty())
    return s;
  std::sort(v.begin(), v.end());
  auto pct = [&](double p) { return v[std::min(v.size() - 1, static_cast<size_t>(p * v.size()))]; };
  for (double x : v)
    s.mean += x;
  s.mean /= v.size();
  for (double x : v)
    s.sd += (x - s.mean) * (x - s.mean);
  s.sd  = std::sqrt(s.sd / std::max<size_t>(1, v.size() - 1));
  s.med = pct(0.5);
  s.mn  = v.front();
  s.mx  = v.back();
  s.p5  = pct(0.05);
  s.p25 = pct(0.25);
  s.p75 = pct(0.75);
  s.p95 = pct(0.95);
  return s;
}

int main(int argc, char* argv[]) {
  std::map<std::string, std::string> opt;
  for (int i = 1; i + 1 < argc; i += 2)
    opt[std::string(argv[i]).substr(2)] = argv[i + 1];
  const std::string input_path  = opt.at("input");
  const std::string cfg_key     = opt.count("config") ? opt["config"] : "helix";
  const std::string params_path = opt.at("params");
  const std::string out_dir     = opt.at("out");
  const int n_samples           = opt.count("samples") ? std::stoi(opt["samples"]) : 5000;
  const int warmup_iters        = opt.count("warmup-iters") ? std::stoi(opt["warmup-iters"]) : 10;
  std::filesystem::create_directories(out_dir);

  auto params = loadExperimentParams(params_path, cfg_key);

  std::vector<PoseInit> poses;
  std::vector<OdometryMeas> odometry;
  std::vector<PriorMeas> priors;
  std::vector<LandmarkMeas> landmarks;
  parseG2OFile(input_path, poses, odometry, priors, landmarks);

  FactorGraph fg;
  add_pose_variable_nodes(fg, poses);
  add_prior_factor_nodes(fg, priors);
  add_odometry_factor_nodes(fg, odometry);
  add_gp_prior_factor_nodes(fg, params.qc_diag);

  // reach steady state (messages populated, linearisation settled)
  GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
  for (int i = 0; i < warmup_iters; ++i)
    solver.performIteration(fg);

  std::vector<std::shared_ptr<SE3PoseVel>> pose_nodes;
  for (auto& vn : fg.getPoseVariableNodes()) {
    auto n = std::dynamic_pointer_cast<SE3PoseVel>(vn);
    if (n)
      pose_nodes.push_back(n);
  }

  // factor -> connected variable ids, reconstructed from the deterministic
  // factor id scheme of factor_graph_builder.h
  auto varIdsOf = [](const std::string& fid) -> std::vector<std::string> {
    auto ts = [](const std::string& s) { return "POSE_" + s; };
    if (fid.rfind("PRIOR_ANCHOR_", 0) == 0)
      return {ts(fid.substr(13))};
    if (fid.rfind("PRIOR_", 0) == 0)
      return {ts(fid.substr(6))};
    if (fid.rfind("GP_", 0) == 0 || fid.rfind("ODOM_", 0) == 0) {
      const size_t p0  = fid.find('_');
      const size_t p1  = fid.find('_', p0 + 1);
      return {ts(fid.substr(p0 + 1, p1 - p0 - 1)), ts(fid.substr(p1 + 1))};
    }
    return {};
  };

  auto ns_of = [](const Clock::time_point& a, const Clock::time_point& b) {
    return std::chrono::duration<double, std::nano>(b - a).count();
  };

  std::map<std::string, std::vector<double>> samples;

  // --- var_to_fac: variables with >=2 edges (single-edge nodes skip sending) --
  {
    auto& out = samples["var_to_fac"];
    int warm  = 0;
    size_t vi = 0, ei = 0;
    while (static_cast<int>(out.size()) < n_samples) {
      auto& n = pose_nodes[vi % pose_nodes.size()];
      if (n->edges_.size() >= 2) {
        const size_t e = ei % n->edges_.size();
        auto t0        = Clock::now();
        n->sendMessage(e);
        auto t1 = Clock::now();
        if (warm < 10)
          ++warm;
        else
          out.push_back(ns_of(t0, t1));
        ++ei;
      }
      ++vi;
    }
  }

  // --- fac_to_var / factor_update ---
  {
    auto& out_msg = samples["fac_to_var"];
    auto& out_upd = samples["factor_update"];
    int warm = 0;
    size_t fi = 0;
    auto& factors = fg.factor_nodes_;
    while (static_cast<int>(out_msg.size()) < n_samples) {
      auto& f   = factors[fi % factors.size()];
      auto vids = varIdsOf(f->id_);
      if (!vids.empty()) {
        auto tu0 = Clock::now();
        f->updateFactor();
        auto tu1 = Clock::now();
        const std::string& vid = vids[out_msg.size() % vids.size()];
        auto tm0 = Clock::now();
        f->sendMessage(vid);
        auto tm1 = Clock::now();
        if (warm < 10)
          ++warm;
        else {
          out_upd.push_back(ns_of(tu0, tu1));
          out_msg.push_back(ns_of(tm0, tm1));
        }
      }
      ++fi;
    }
  }

  // --- var_update ---
  {
    auto& out = samples["var_update"];
    int warm  = 0;
    size_t vi = 0;
    while (static_cast<int>(out.size()) < n_samples) {
      auto& n = pose_nodes[vi % pose_nodes.size()];
      auto t0 = Clock::now();
      n->updateMuAndSigma();
      auto t1 = Clock::now();
      if (warm < 10)
        ++warm;
      else
        out.push_back(ns_of(t0, t1));
      ++vi;
    }
  }

  // --- interpolation: single mid-interval query (with covariance, as GPInterp) --
  {
    auto& out = samples["interpolation"];
    int warm  = 0;
    size_t qi = 0;
    while (static_cast<int>(out.size()) < n_samples) {
      const size_t i   = qi % (pose_nodes.size() - 1);
      const double tau = 0.5 * (pose_nodes[i]->timestamp_ + pose_nodes[i + 1]->timestamp_) + 1e-4 * ((qi % 7) - 3);
      std::vector<double> q{tau};
      auto t0 = Clock::now();
      auto res = GPInterpAtTimestamps(pose_nodes, q, params.qc_diag);
      auto t1 = Clock::now();
      if (res.empty() || !res[0].pose.matrix().allFinite())
        std::cerr << "bad interp\n";
      if (warm < 10)
        ++warm;
      else
        out.push_back(ns_of(t0, t1));
      ++qi;
    }
  }

  // ---- outputs: raw samples + summary ----
  std::ofstream sum(out_dir + "/summary.csv");
  sum << "operation,n,mean_us,median_us,std_us,min_us,max_us,p5_us,p25_us,p75_us,p95_us,graph,config,"
         "n_poses,n_factors,warmup_iters,threads\n";
  for (auto& [op, v] : samples) {
    std::ofstream raw(out_dir + "/raw_" + op + ".csv");
    raw << "operation,sample,ns\n";
    for (size_t i = 0; i < v.size(); ++i)
      raw << op << "," << i << "," << v[i] << "\n";
    auto st = stats(v);
    sum << op << "," << st.n << "," << st.mean / 1e3 << "," << st.med / 1e3 << "," << st.sd / 1e3 << "," << st.mn / 1e3 << ","
        << st.mx / 1e3 << "," << st.p5 / 1e3 << "," << st.p25 / 1e3 << "," << st.p75 / 1e3 << "," << st.p95 / 1e3 << ","
        << input_path << "," << cfg_key << "," << pose_nodes.size() << "," << fg.factor_nodes_.size() << "," << warmup_iters
        << "," << omp_get_max_threads() << "\n";
    std::cout << "[repro_bench] " << op << ": median " << st.med / 1e3 << " us, mean " << st.mean / 1e3 << " us (n=" << st.n
              << ")\n";
  }
  return 0;
}
