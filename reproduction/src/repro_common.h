/**
 * @file repro_common.h
 * @brief Shared read-only evaluation utilities for the paper reproduction.
 *
 * Everything here EVALUATES solver state; nothing mutates the factor graph,
 * the schedule, precision, factor definitions, priors, initialisation or
 * iteration counts. Error formulas replicate the library factor definitions
 * exactly (see docs/PAPER_REPRODUCTION_AUDIT.md §8).
 */
#pragma once

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "common/data_types.h"
#include <gsolver/graph/factor_graph.h>
#include <gsolver/graph/types/variables/se3_posevel.h>
#include <gsolver/maths/geometry3d.h>

namespace repro {

  using gsolver::expMapSE3;
  using gsolver::logMapSE3;
  using gsolver::logMapSO3;
  using Vector6d = Eigen::Matrix<double, 6, 1>;

  // ---------------------------------------------------------------- timing --
  using Clock = std::chrono::steady_clock;
  inline double ms_since(const Clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  }
  inline long peakRSSKb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss; // KiB on Linux
  }

  // ------------------------------------------------------------------- TUM --
  struct TumPose {
    double t;
    Eigen::Isometry3d T;
  };

  inline std::vector<TumPose> loadTum(const std::string& path) {
    std::vector<TumPose> out;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty() || line[0] == '#')
        continue;
      std::istringstream ss(line);
      double t, tx, ty, tz, qx, qy, qz, qw;
      if (!(ss >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw))
        continue;
      Eigen::Quaterniond q(qw, qx, qy, qz);
      q.normalize();
      TumPose p;
      p.t              = t;
      p.T              = Eigen::Isometry3d::Identity();
      p.T.translation() = Eigen::Vector3d(tx, ty, tz);
      p.T.linear()      = q.toRotationMatrix();
      out.push_back(p);
    }
    return out;
  }

  inline void writeTumFile(const std::string& path, const std::vector<TumPose>& poses) {
    std::ofstream f(path);
    f << std::fixed << std::setprecision(9) << "#timestamp tx ty tz qx qy qz qw\n";
    for (const auto& p : poses) {
      Eigen::Quaterniond q(p.T.linear());
      const auto& t = p.T.translation();
      f << p.t << " " << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
        << "\n";
    }
  }

  // -------------------------------------------------------------- ATE / ARE --
  struct AteAre {
    double ate = std::numeric_limits<double>::quiet_NaN();
    double are = std::numeric_limits<double>::quiet_NaN();
    double ate_aligned = std::numeric_limits<double>::quiet_NaN();
    double are_aligned = std::numeric_limits<double>::quiet_NaN();
    size_t matched = 0;
  };

  // est/gt matched by identical order (caller guarantees timestamp match).
  inline AteAre computeAteAre(const std::vector<TumPose>& est, const std::vector<TumPose>& gt) {
    AteAre r;
    const size_t n = std::min(est.size(), gt.size());
    if (n == 0)
      return r;
    double se_t = 0, se_r = 0;
    for (size_t i = 0; i < n; ++i) {
      se_t += (est[i].T.translation() - gt[i].T.translation()).squaredNorm();
      const Eigen::Matrix3d dR = gt[i].T.linear().transpose() * est[i].T.linear();
      se_r += logMapSO3(dR).squaredNorm();
    }
    r.ate     = std::sqrt(se_t / n);
    r.are     = std::sqrt(se_r / n);
    r.matched = n;

    // SE(3) alignment (Umeyama on positions, no scale), applied to est.
    Eigen::Matrix3Xd P(3, n), Q(3, n);
    for (size_t i = 0; i < n; ++i) {
      P.col(i) = est[i].T.translation();
      Q.col(i) = gt[i].T.translation();
    }
    const Eigen::Matrix4d A = Eigen::umeyama(P, Q, false);
    Eigen::Isometry3d Talign(A);
    double se_t2 = 0, se_r2 = 0;
    for (size_t i = 0; i < n; ++i) {
      const Eigen::Isometry3d Te = Talign * est[i].T;
      se_t2 += (Te.translation() - gt[i].T.translation()).squaredNorm();
      se_r2 += logMapSO3(gt[i].T.linear().transpose() * Te.linear()).squaredNorm();
    }
    r.ate_aligned = std::sqrt(se_t2 / n);
    r.are_aligned = std::sqrt(se_r2 / n);
    return r;
  }

  // ------------------------------------------------------------------ NEES --
  struct NeesResult {
    double mean_local_est = std::numeric_limits<double>::quiet_NaN(); // e = Log(T_est^-1 T_gt), Sigma in est tangent (primary)
    double mean_local_gt  = std::numeric_limits<double>::quiet_NaN(); // e = Log(T_gt^-1 T_est)
    double mean_global    = std::numeric_limits<double>::quiet_NaN(); // e = [t_e - t_g; Log(R_e R_g^T)]
    size_t count          = 0;
  };

  inline NeesResult computeNees(const std::vector<std::shared_ptr<gsolver::SE3PoseVel>>& nodes,
                                const std::vector<TumPose>& gt,
                                std::vector<Eigen::Vector4d>* per_pose /*t,v1,v2,v3*/ = nullptr) {
    NeesResult r;
    const size_t n = std::min(nodes.size(), gt.size());
    if (n == 0)
      return r;
    double s1 = 0, s2 = 0, s3 = 0;
    for (size_t i = 0; i < n; ++i) {
      const Eigen::Isometry3d& Te = nodes[i]->mu_.pose;
      const Eigen::Isometry3d& Tg = gt[i].T;
      const Eigen::Matrix<double, 6, 6> S = nodes[i]->Sigma_.topLeftCorner<6, 6>();
      const Eigen::Matrix<double, 6, 6> Sinv = S.inverse();
      const Vector6d e1 = logMapSE3(Te.inverse() * Tg);
      const Vector6d e2 = logMapSE3(Tg.inverse() * Te);
      Vector6d e3;
      e3.head<3>() = Te.translation() - Tg.translation();
      e3.tail<3>() = logMapSO3(Te.linear() * Tg.linear().transpose());
      const double v1 = e1.dot(Sinv * e1);
      const double v2 = e2.dot(Sinv * e2);
      const double v3 = e3.dot(Sinv * e3);
      s1 += v1;
      s2 += v2;
      s3 += v3;
      if (per_pose)
        per_pose->emplace_back(gt[i].t, v1, v2, v3);
    }
    r.mean_local_est = s1 / n;
    r.mean_local_gt  = s2 / n;
    r.mean_global    = s3 / n;
    r.count          = n;
    return r;
  }

  // ---------------------------------------------------------------- energy --
  // Replicates factor error definitions exactly (evaluated at BELIEF MEANS).
  struct PoseVelState {
    Eigen::Isometry3d T;
    Vector6d v;
  };

  inline Vector6d gpErrorPose(const Eigen::Isometry3d& Tf, const Eigen::Isometry3d& Tt, const Vector6d& vf, double dt) {
    return logMapSE3(expMapSE3(dt * vf).inverse() * Tf.inverse() * Tt);
  }
  inline Vector6d gpErrorVel(const Eigen::Isometry3d& Tf, const Eigen::Isometry3d& Tt, const Vector6d& vf, const Vector6d& vt) {
    const Eigen::Matrix3d R = Tf.linear().transpose() * Tt.linear();
    Eigen::Matrix<double, 6, 6> adj = Eigen::Matrix<double, 6, 6>::Zero();
    adj.block<3, 3>(0, 0) = R;
    adj.block<3, 3>(3, 3) = R;
    return (adj * vt) - vf;
  }

  // Q_i from factor_graph_builder.h (Qc clamped with cwiseMax(1e-4))
  inline Eigen::Matrix<double, 12, 12> gpQd(const Vector6d& Qc_diag, double dt) {
    Eigen::Matrix<double, 6, 6> Qc = Qc_diag.cwiseMax(1e-4).asDiagonal();
    Eigen::Matrix<double, 12, 12> Qd = Eigen::Matrix<double, 12, 12>::Zero();
    Qd.block<6, 6>(0, 0) = std::pow(dt, 3) / 3.0 * Qc;
    Qd.block<6, 6>(0, 6) = std::pow(dt, 2) / 2.0 * Qc;
    Qd.block<6, 6>(6, 0) = std::pow(dt, 2) / 2.0 * Qc;
    Qd.block<6, 6>(6, 6) = dt * Qc;
    return Qd;
  }

  struct EnergyBreakdown {
    double meas = 0; // measurement factors (priors and/or odometry)
    double gp   = 0; // GP prior factors
    double total() const { return meas + gp; }
  };

  // states: timestamp -> state at belief mean; measurement structs as parsed.
  inline EnergyBreakdown evalEnergy(const std::map<double, PoseVelState>& states,
                                    const std::vector<examples::PriorMeas>& priors,
                                    const std::vector<examples::OdometryMeas>& odom,
                                    const Vector6d& Qc_diag,
                                    bool with_gp) {
    EnergyBreakdown E;
    for (const auto& pr : priors) {
      auto it = states.find(pr.timestamp);
      if (it == states.end())
        continue;
      const Vector6d e = logMapSE3(pr.pose.inverse() * it->second.T);
      const Eigen::Matrix<double, 6, 6> S =
        pr.covariance + 1e-6 * Eigen::Matrix<double, 6, 6>::Identity(); // as in add_prior_factor_nodes
      E.meas += 0.5 * e.dot(S.inverse() * e);
    }
    for (const auto& od : odom) {
      auto itf = states.find(od.timestamp_from);
      auto itt = states.find(od.timestamp_to);
      if (itf == states.end() || itt == states.end())
        continue;
      const Vector6d e = logMapSE3(od.measurement.inverse() * (itf->second.T.inverse() * itt->second.T));
      const Eigen::Matrix<double, 6, 6> S = od.covariance + 1e-6 * Eigen::Matrix<double, 6, 6>::Identity();
      E.meas += 0.5 * e.dot(S.inverse() * e);
    }
    if (with_gp && states.size() >= 2) {
      auto it = states.begin();
      auto next = std::next(it);
      for (; next != states.end(); ++it, ++next) {
        const double dt = next->first - it->first;
        if (dt <= 0)
          continue;
        Eigen::Matrix<double, 12, 1> e;
        e.head<6>() = gpErrorPose(it->second.T, next->second.T, it->second.v, dt);
        e.tail<6>() = gpErrorVel(it->second.T, next->second.T, it->second.v, next->second.v);
        const Eigen::Matrix<double, 12, 12> Qd = gpQd(Qc_diag, dt);
        E.gp += 0.5 * e.dot(Qd.ldlt().solve(e));
      }
    }
    return E;
  }

  // Snapshot of pose-variable beliefs (sorted by timestamp).
  inline std::map<double, PoseVelState> snapshotStates(gsolver::FactorGraph& g) {
    std::map<double, PoseVelState> states;
    for (auto& vn : g.getPoseVariableNodes()) {
      auto n = std::dynamic_pointer_cast<gsolver::SE3PoseVel>(vn);
      if (!n)
        continue;
      states[n->timestamp_] = PoseVelState{n->mu_.pose, n->mu_.vel};
    }
    return states;
  }

  inline bool hasNaN(const std::map<double, PoseVelState>& states) {
    for (const auto& [t, s] : states) {
      if (!s.T.matrix().allFinite() || !s.v.allFinite())
        return true;
    }
    return false;
  }

} // namespace repro
