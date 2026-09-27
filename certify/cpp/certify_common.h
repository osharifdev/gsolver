#pragma once
/**
 * @file certify_common.h
 * @brief Phase A instrumentation shared by the certify_* drivers.
 *
 * Exports a G-solver factor graph as one linear model in fixed charts (PROOF_NOTE.md §0.1):
 * per-variable linearization point and tangent state, per-factor Jacobian J_f, error e_f (so r_f = -e_f)
 * and covariance Sigma_f (Lambda_f = Sigma_f^-1), the frozen single-edge messages, and the messages on
 * GP-prior edges. Frozen runs use GbpSolver::freeze() (certify/FREEZE_MODE.md). Unfrozen exports only
 * read the graph: they re-evaluate each factor at the belief means and restore the solver's own
 * message means before the next iteration, so the normal run is unchanged.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <memory>
#include <sstream>
#include <string>
#include <cstdio>
#include <sys/stat.h>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <functional>

#include <gsolver/core/flop_count.h>
#include <gsolver/core/frozen_mode.h>
#include <gsolver/core/gbp_solver.h>
#include <gsolver/graph/factor_graph.h>
#include <gsolver/graph/types/all_types.h>

namespace certify {

  using namespace gsolver;

  // ------------------------------------------------------------------------------------------ files

  inline void makeDirs(const std::string& path) {
    std::string cur;
    std::stringstream ss(path);
    std::string part;
    if (!path.empty() && path[0] == '/')
      cur = "/";
    while (std::getline(ss, part, '/')) {
      if (part.empty())
        continue;
      cur += part + "/";
      mkdir(cur.c_str(), 0755);
    }
  }

  // NPY v1.0 writer (little-endian, C order)
  inline void writeNpyRaw(const std::string& path,
                          const void* data,
                          size_t nbytes,
                          const std::string& descr,
                          const std::vector<size_t>& shape) {
    std::string shp = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
      shp += std::to_string(shape[i]);
      if (shape.size() == 1 || i + 1 < shape.size())
        shp += ",";
      if (i + 1 < shape.size())
        shp += " ";
    }
    shp += ")";
    std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shp + ", }";
    const size_t total = 10 + header.size() + 1;
    header += std::string((64 - total % 64) % 64, ' ') + "\n";
    std::ofstream f(path, std::ios::binary);
    if (!f)
      throw std::runtime_error("cannot write " + path);
    f.write("\x93NUMPY", 6);
    const char ver[2] = {1, 0};
    f.write(ver, 2);
    const uint16_t hl = static_cast<uint16_t>(header.size());
    f.write(reinterpret_cast<const char*>(&hl), 2);
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(nbytes));
  }
  inline void writeNpy(const std::string& path, const std::vector<double>& v, const std::vector<size_t>& shape) {
    writeNpyRaw(path, v.data(), v.size() * sizeof(double), "<f8", shape);
  }
  inline void writeNpy(const std::string& path, const std::vector<int64_t>& v, const std::vector<size_t>& shape) {
    writeNpyRaw(path, v.data(), v.size() * sizeof(int64_t), "<i8", shape);
  }

  inline std::string jsonStr(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
      if (c == '"' || c == '\\')
        o += '\\';
      o += c;
    }
    return o + "\"";
  }
  inline std::string jsonNum(double x) {
    if (!std::isfinite(x))
      return "null";
    char buf[64];
    snprintf(buf, sizeof(buf), "%.17g", x);
    return buf;
  }

  // Ordered JSON object built from already-encoded values
  struct Json {
    std::vector<std::pair<std::string, std::string>> kv;
    Json& set(const std::string& k, const std::string& encoded) {
      for (auto& p : kv)
        if (p.first == k) {
          p.second = encoded;
          return *this;
        }
      kv.emplace_back(k, encoded);
      return *this;
    }
    Json& str(const std::string& k, const std::string& v) { return set(k, jsonStr(v)); }
    Json& num(const std::string& k, double v) { return set(k, jsonNum(v)); }
    Json& integer(const std::string& k, long long v) { return set(k, std::to_string(v)); }
    Json& boolean(const std::string& k, bool v) { return set(k, v ? "true" : "false"); }
    std::string dump() const {
      std::string o = "{\n";
      for (size_t i = 0; i < kv.size(); ++i)
        o += "  " + jsonStr(kv[i].first) + ": " + kv[i].second + (i + 1 < kv.size() ? ",\n" : "\n");
      return o + "}";
    }
  };
  template <typename T>
  inline std::string jsonArray(const std::vector<T>& v) {
    std::string o = "[";
    for (size_t i = 0; i < v.size(); ++i) {
      if constexpr (std::is_floating_point_v<T>)
        o += jsonNum(v[i]);
      else
        o += std::to_string(v[i]);
      if (i + 1 < v.size())
        o += ", ";
    }
    return o + "]";
  }

  inline std::vector<int> parseIntList(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
      if (!tok.empty())
        out.push_back(std::stoi(tok));
    return out;
  }

  // ------------------------------------------------------------------------------ variable ordering

  struct VarRec {
    std::shared_ptr<VariableNodeBase> node;
    std::string id;
    std::string type; // SE3PoseVel | SE3Pose | SE3Point
    int dim        = 0;
    int offset     = 0;  // first row of the block in H
    int pose_index = -1; // time order among SE3PoseVel blocks
    double timestamp = std::nan("");
  };

  // Canonical block order: pose-velocity blocks by timestamp, then SE3Pose landmarks, then SE3Point
  // landmarks, each group in graph insertion order.
  inline std::vector<VarRec> orderVariables(FactorGraph& g) {
    std::vector<VarRec> poses, se3, pts;
    for (auto& v : g.variable_nodes_) {
      VarRec r;
      r.node = v;
      r.id   = v->id_;
      if (auto p = std::dynamic_pointer_cast<SE3PoseVel>(v)) {
        r.type      = "SE3PoseVel";
        r.dim       = 12;
        r.timestamp = p->timestamp_;
        poses.push_back(r);
      } else if (std::dynamic_pointer_cast<SE3Pose>(v)) {
        r.type = "SE3Pose";
        r.dim  = 6;
        se3.push_back(r);
      } else if (std::dynamic_pointer_cast<SE3Point>(v)) {
        r.type = "SE3Point";
        r.dim  = 3;
        pts.push_back(r);
      } else {
        throw std::runtime_error("unknown variable type for " + v->id_);
      }
    }
    std::stable_sort(poses.begin(), poses.end(), [](const VarRec& a, const VarRec& b) { return a.timestamp < b.timestamp; });
    std::vector<VarRec> all;
    int off = 0, pi = 0;
    for (auto* grp : {&poses, &se3, &pts})
      for (auto& r : *grp) {
        r.offset = off;
        off += r.dim;
        if (r.type == "SE3PoseVel")
          r.pose_index = pi++;
        all.push_back(r);
      }
    return all;
  }

  // Call fn(concrete_node) for the variable's concrete type
  template <typename Fn>
  inline void withVariable(const std::shared_ptr<VariableNodeBase>& v, Fn&& fn) {
    if (auto p = std::dynamic_pointer_cast<SE3PoseVel>(v))
      fn(p);
    else if (auto q = std::dynamic_pointer_cast<SE3Pose>(v))
      fn(q);
    else if (auto r = std::dynamic_pointer_cast<SE3Point>(v))
      fn(r);
    else
      throw std::runtime_error("unknown variable type");
  }

  // Call fn(concrete_factor, type_name) for the factor's concrete type
  template <typename Fn>
  inline void withFactor(const std::shared_ptr<FactorNodeBase>& f, Fn&& fn) {
    if (auto a = std::dynamic_pointer_cast<SE3PoseVelPrior>(f))
      fn(a, std::string("prior"));
    else if (auto b = std::dynamic_pointer_cast<SE3PoseVelGP>(f))
      fn(b, std::string("gp"));
    else if (auto c = std::dynamic_pointer_cast<SE3PoseVelPose>(f))
      fn(c, std::string("posevel_pose"));
    else if (auto d = std::dynamic_pointer_cast<SE3PoseVelStereoPoint>(f))
      fn(d, std::string("stereo"));
    else if (auto e = std::dynamic_pointer_cast<SE3PoseVelPoint>(f))
      fn(e, std::string("posevel_point"));
    else if (auto h = std::dynamic_pointer_cast<SE3PoseVelPoseVel>(f))
      fn(h, std::string("posevel_posevel"));
    else if (auto k = std::dynamic_pointer_cast<SE3PointEquality>(f))
      fn(k, std::string("point_equality"));
    else
      throw std::runtime_error("unknown factor type for " + f->id_);
  }

  template <int I, typename FN, typename Fn>
  inline void forEachEdge(FN& factor, Fn&& fn) {
    if constexpr (I < FN::NumberOfEdges_) {
      fn(I, factor.edges_.template at<I>());
      forEachEdge<I + 1>(factor, fn);
    }
  }

  // Flatten an Eigen matrix row-major
  template <typename M>
  inline void appendRowMajor(std::vector<double>& out, const M& m) {
    for (int r = 0; r < m.rows(); ++r)
      for (int c = 0; c < m.cols(); ++c)
        out.push_back(m(r, c));
  }

  inline void appendIsometry(std::vector<double>& out, const Eigen::Isometry3d& T) {
    appendRowMajor(out, T.matrix());
  }

  // ----------------------------------------------------------------------------------------- export

  struct FactorGroup {
    int err_dim = 0, var_dim = 0, nvars = 0;
    size_t n    = 0;
    std::vector<int64_t> vars;                // n x nvars block indices
    std::vector<double> J, e, Sigma, msg_tau; // n x m x d, n x m, n x m x m, n x d
    std::vector<double> dt;                   // gp only
    std::vector<int64_t> inlier;              // stereo only
    std::vector<std::string> ids;
  };

  enum class Charts { Frozen, Beliefs };

  /**
   * Write the linear model and the current state of `g` into `dir`.
   *  Charts::Frozen  : g is frozen; linearization points mu_lin_, state z_, J_lin_/error_lin_.
   *  Charts::Beliefs : g is not frozen; every factor is re-evaluated at the variables' belief means
   *                    (state z = 0 in those charts). The solver's own message means are restored.
   */
  inline void exportModel(FactorGraph& g, const std::string& dir, Charts charts, Json meta) {
    makeDirs(dir);
    auto vars = orderVariables(g);
    std::unordered_map<std::string, int> index;
    for (size_t i = 0; i < vars.size(); ++i)
      index[vars[i].id] = static_cast<int>(i);
    const int N = vars.empty() ? 0 : vars.back().offset + vars.back().dim;

    if (charts == Charts::Beliefs) {
      // Point every incoming message mean at the belief mean, evaluate each factor there, then restore. Works on
      // an unfrozen graph and, without changing the solver state, on a threshold-mode (frozen) one.
      for (auto& v : g.variable_nodes_)
        v->beginEvaluateAtBelief();
    }

    // ---- factors
    std::map<std::string, FactorGroup> groups;
    for (auto& f : g.factor_nodes_) {
      withFactor(f, [&](auto fp, const std::string& type) {
        using FT = typename std::decay_t<decltype(*fp)>;
        FactorGroup& G = groups[type];
        G.err_dim      = FT::MatrixErrorDim::RowsAtCompileTime;
        G.var_dim      = FT::VariablesDim_;
        G.nvars        = FT::NumberOfEdges_;
        typename FT::MatrixJacobianDim J;
        typename FT::VectorErrorDim e;
        if (charts == Charts::Frozen) {
          if (!fp->frozen_)
            throw std::runtime_error("factor " + fp->id_ + " is not frozen");
          J = fp->J_lin_;
          e = fp->error_lin_;
        } else {
          fp->updateErrorAndJacobian(); // at the belief means (see beginFreeze above)
          J = fp->J_;
          e = fp->error_;
        }
        appendRowMajor(G.J, J);
        appendRowMajor(G.e, e);
        appendRowMajor(G.Sigma, fp->Sigma_);
        G.ids.push_back(fp->id_);
        forEachEdge<0>(*fp, [&](int, auto& edge) {
          G.vars.push_back(index.at(edge->variable_id_));
          // Frozen: the incoming message mean in the fixed chart. Beliefs: filled in the second pass below.
          if (charts == Charts::Frozen)
            for (int k = 0; k < edge->variable_to_factor_message_.tau_.size(); ++k)
              G.msg_tau.push_back(edge->variable_to_factor_message_.tau_(k));
        });
        if constexpr (std::is_same_v<FT, SE3PoseVelGP>)
          G.dt.push_back(fp->dt_);
        if constexpr (std::is_same_v<FT, SE3PoseVelStereoPoint>)
          G.inlier.push_back(fp->inlier_ ? 1 : 0);
        G.n++;
      });
    }

    if (charts == Charts::Beliefs) {
      // Offsets of the solver's own linearization points (incoming message means) from the beliefs,
      // computed while the stash still holds them, then restore the solver state.
      std::unordered_map<const EdgeBase*, std::vector<double>> offset_of_edge;
      for (auto& v : g.variable_nodes_) {
        withVariable(v, [&](auto vp) {
          for (size_t i = 0; i < vp->edges_.size(); ++i) {
            auto tau = vp->cavityOffsetFromBelief(i);
            offset_of_edge[vp->edges_[i].get()] = std::vector<double>(tau.data(), tau.data() + tau.size());
          }
        });
      }
      for (auto& f : g.factor_nodes_) {
        withFactor(f, [&](auto fp, const std::string& type) {
          FactorGroup& G = groups[type];
          forEachEdge<0>(*fp, [&](int, auto& edge) {
            const auto& o = offset_of_edge.at(edge.get());
            G.msg_tau.insert(G.msg_tau.end(), o.begin(), o.end());
          });
        });
      }
      for (auto& v : g.variable_nodes_)
        v->restoreMessageMeans();
    }

    for (auto& [type, G] : groups) {
      const std::string p = dir + "/f_" + type + "_";
      writeNpy(p + "vars.npy", G.vars, {G.n, size_t(G.nvars)});
      writeNpy(p + "J.npy", G.J, {G.n, size_t(G.err_dim), size_t(G.var_dim)});
      writeNpy(p + "e.npy", G.e, {G.n, size_t(G.err_dim)});
      writeNpy(p + "Sigma.npy", G.Sigma, {G.n, size_t(G.err_dim), size_t(G.err_dim)});
      writeNpy(p + "msg_tau.npy", G.msg_tau, {G.n, size_t(G.var_dim)});
      if (!G.dt.empty())
        writeNpy(p + "dt.npy", G.dt, {G.n});
      if (!G.inlier.empty())
        writeNpy(p + "inlier.npy", G.inlier, {G.n});
      std::ofstream fid(p + "ids.txt");
      for (auto& s : G.ids)
        fid << s << "\n";
    }

    // ---- variables: linearization points, state, belief covariance, degree, frozen single-edge message
    std::vector<double> T_lin, v_lin, se3_lin, pt_lin, z, belief_Sigma;
    std::vector<int64_t> degree, se_idx;
    std::vector<double> se_tau, se_Lambda;
    for (size_t i = 0; i < vars.size(); ++i) {
      withVariable(vars[i].node, [&](auto vp) {
        using VT                      = typename std::decay_t<decltype(*vp)>;
        const bool frozen                   = (charts == Charts::Frozen);
        const typename VT::EstimateType lin = frozen ? vp->mu_lin_ : vp->beliefMean();
        if constexpr (std::is_same_v<VT, SE3PoseVel>) {
          appendIsometry(T_lin, lin.pose);
          for (int k = 0; k < 6; ++k)
            v_lin.push_back(lin.vel(k));
        } else if constexpr (std::is_same_v<VT, SE3Pose>) {
          appendIsometry(se3_lin, lin);
        } else {
          for (int k = 0; k < 3; ++k)
            pt_lin.push_back(lin(k));
        }
        for (int k = 0; k < VT::LieDim; ++k)
          z.push_back(frozen ? vp->z_(k) : 0.0);
        appendRowMajor(belief_Sigma, vp->Sigma_);
        degree.push_back(static_cast<int64_t>(vp->edges_.size()));
        if (vp->edges_.size() == 1) {
          // G-solver never updates the outgoing message of a single-edge variable (variable_node.cpp:26-28)
          const auto& m = vp->edges_[0]->variable_to_factor_message_;
          se_idx.push_back(static_cast<int64_t>(i));
          auto tau = frozen ? m.tau_ : (vp->frozen_ ? vp->offsetFromBelief(vp->boxplus(m.tau_)) : vp->boxminus(m.mu_));
          for (int k = 0; k < VT::LieDim; ++k)
            se_tau.push_back(tau(k));
          appendRowMajor(se_Lambda, m.Lambda_);
        }
      });
    }
    writeNpy(dir + "/pose_T_lin.npy", T_lin, {T_lin.size() / 16, 4, 4});
    writeNpy(dir + "/pose_v_lin.npy", v_lin, {v_lin.size() / 6, 6});
    writeNpy(dir + "/se3_T_lin.npy", se3_lin, {se3_lin.size() / 16, 4, 4});
    writeNpy(dir + "/point_p_lin.npy", pt_lin, {pt_lin.size() / 3, 3});
    writeNpy(dir + "/z.npy", z, {z.size()});
    writeNpy(dir + "/belief_Sigma_flat.npy", belief_Sigma, {belief_Sigma.size()});
    writeNpy(dir + "/degree.npy", degree, {degree.size()});
    writeNpy(dir + "/single_edge_index.npy", se_idx, {se_idx.size()});
    writeNpy(dir + "/single_edge_tau_flat.npy", se_tau, {se_tau.size()});
    writeNpy(dir + "/single_edge_Lambda_flat.npy", se_Lambda, {se_Lambda.size()});

    // ---- messages on GP-prior edges (fact F4)
    std::vector<double> gp_f2v_L, gp_v2f_L, gp_f2v_t, gp_v2f_t;
    for (auto& f : g.factor_nodes_) {
      auto gp = std::dynamic_pointer_cast<SE3PoseVelGP>(f);
      if (!gp)
        continue;
      forEachEdge<0>(*gp, [&](int, auto& edge) {
        appendRowMajor(gp_f2v_L, edge->factor_to_variable_message_.Lambda_);
        appendRowMajor(gp_v2f_L, edge->variable_to_factor_message_.Lambda_);
        for (int k = 0; k < 12; ++k) {
          gp_f2v_t.push_back(charts == Charts::Frozen ? edge->factor_to_variable_message_.tau_(k) : std::nan(""));
          gp_v2f_t.push_back(charts == Charts::Frozen ? edge->variable_to_factor_message_.tau_(k) : std::nan(""));
        }
      });
    }
    const size_t ngp = gp_f2v_t.size() / 24;
    writeNpy(dir + "/gp_f2v_Lambda.npy", gp_f2v_L, {ngp, 2, 12, 12});
    writeNpy(dir + "/gp_v2f_Lambda.npy", gp_v2f_L, {ngp, 2, 12, 12});
    writeNpy(dir + "/gp_f2v_tau.npy", gp_f2v_t, {ngp, 2, 12});
    writeNpy(dir + "/gp_v2f_tau.npy", gp_v2f_t, {ngp, 2, 12});

    // ---- structure
    {
      std::ofstream fv(dir + "/variables.tsv");
      fv << "index\tid\ttype\tdim\toffset\tpose_index\ttimestamp\n";
      for (size_t i = 0; i < vars.size(); ++i)
        fv << i << "\t" << vars[i].id << "\t" << vars[i].type << "\t" << vars[i].dim << "\t" << vars[i].offset << "\t"
           << vars[i].pose_index << "\t" << jsonNum(vars[i].timestamp) << "\n";
    }
    meta.str("charts", charts == Charts::Frozen ? "frozen" : "beliefs");
    meta.integer("n_variables", static_cast<long long>(vars.size()));
    meta.integer("n_dof", N);
    meta.integer("n_factors", static_cast<long long>(g.factor_nodes_.size()));
    meta.set("frozen_regularizers", "{\"belief\": " + jsonNum(frozen_eps_belief) + ", \"v2f\": " + jsonNum(frozen_eps_v2f) +
                                          ", \"marg\": " + jsonNum(frozen_eps_marg) + ", \"f2v\": " + jsonNum(frozen_eps_f2v) + "}");
    std::string groups_json = "{";
    for (auto it = groups.begin(); it != groups.end(); ++it) {
      groups_json += jsonStr(it->first) + ": " + std::to_string(it->second.n);
      if (std::next(it) != groups.end())
        groups_json += ", ";
    }
    meta.set("factor_counts", groups_json + "}");
    std::ofstream fm(dir + "/meta.json");
    fm << meta.dump() << "\n";
  }

  // Append the tangent state z of every block (canonical order) to a raw float64 stream
  inline void appendState(const std::vector<VarRec>& vars, std::ofstream& out) {
    std::vector<double> z;
    for (auto& r : vars)
      withVariable(r.node, [&](auto vp) {
        for (int k = 0; k < vp->z_.size(); ++k)
          z.push_back(vp->z_(k));
      });
    out.write(reinterpret_cast<const char*>(z.data()), static_cast<std::streamsize>(z.size() * sizeof(double)));
  }

  // ------------------------------------------------------------------------- normal-run step metric

  // Belief means before an iteration, to measure the step s_i = sqrt(d' Sigma_i^-1 d), d = mu_new [-] mu_old
  struct MeanSnapshot {
    std::vector<SE3PoseVelValue> pv;
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> se3;
    std::vector<Eigen::Vector3d> pt;
    void take(const std::vector<VarRec>& vars) {
      pv.clear();
      se3.clear();
      pt.clear();
      for (auto& r : vars)
        withVariable(r.node, [&](auto vp) {
          using VT = typename std::decay_t<decltype(*vp)>;
          if constexpr (std::is_same_v<VT, SE3PoseVel>)
            pv.push_back(vp->mu_);
          else if constexpr (std::is_same_v<VT, SE3Pose>)
            se3.push_back(vp->mu_);
          else
            pt.push_back(vp->mu_);
        });
    }
    // returns max and median of s_i over all blocks
    std::pair<double, double> steps(const std::vector<VarRec>& vars) const {
      std::vector<double> s;
      size_t a = 0, b = 0, c = 0;
      for (auto& r : vars)
        withVariable(r.node, [&](auto vp) {
          using VT = typename std::decay_t<decltype(*vp)>;
          typename VT::VectorLieDim d;
          if constexpr (std::is_same_v<VT, SE3PoseVel>)
            d = vp->boxminus(pv[a++]);
          else if constexpr (std::is_same_v<VT, SE3Pose>)
            d = vp->boxminus(se3[b++]);
          else
            d = vp->boxminus(pt[c++]);
          s.push_back(std::sqrt(std::max(0.0, (d.transpose() * vp->Sigma_.inverse() * d).value())));
        });
      if (s.empty())
        return {0.0, 0.0};
      std::vector<double> t = s;
      std::nth_element(t.begin(), t.begin() + t.size() / 2, t.end());
      return {*std::max_element(s.begin(), s.end()), t[t.size() / 2]};
    }
  };

  // --frozen-eps-sites "belief=E,v2f=E,marg=E,f2v=E": per-site override of the frozen regularizers
  inline void applySiteOverrides(const std::string& spec) {
    std::stringstream ss(spec);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      if (tok.empty())
        continue;
      const auto eq = tok.find('=');
      const std::string k = tok.substr(0, eq);
      const double v      = std::stod(tok.substr(eq + 1));
      if (k == "belief") frozen_eps_belief = v;
      else if (k == "v2f") frozen_eps_v2f = v;
      else if (k == "marg") frozen_eps_marg = v;
      else if (k == "f2v") frozen_eps_f2v = v;
      else throw std::runtime_error("unknown regularizer site " + k);
    }
  }

  inline double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }

  /**
   * Run `iters` frozen iterations. The state z^(t) is appended to <dir>/states_all.f64 (rows of n_dof
   * float64 values) for t = 0, every t <= dense_until, every stride-th t after that, and t = iters;
   * the dumped iteration numbers go to <dir>/states_iters.npy. Returns the wall time of the iterations.
   */
  inline double runFrozen(FactorGraph& g, GbpSolver& solver, int iters, const std::string& dir, int dense_until = 1000,
                          int stride = 1) {
    auto vars = orderVariables(g);
    std::ofstream out(dir + "/states_all.f64", std::ios::binary);
    std::vector<int64_t> dumped{0};
    appendState(vars, out);
    auto t0 = std::chrono::steady_clock::now();
    for (int t = 1; t <= iters; ++t) {
      solver.performIteration(g);
      if (t <= dense_until || t % stride == 0 || t == iters) {
        appendState(vars, out);
        dumped.push_back(t);
      }
    }
    const double secs = seconds_since(t0);
    writeNpy(dir + "/states_iters.npy", dumped, {dumped.size()});
    return secs;
  }

  // ---- Phase D, WP6a: flop classes (certify/FLOPS.md). Factor classes index the factor-side categories of
  // gsolver/core/flop_count.h (Linearize, FactorUpdate, F2V), variable classes the variable-side ones (Belief, V2F).
  inline const std::vector<std::string>& factorClassNames() {
    static const std::vector<std::string> n{"other", "gp", "stereo", "posevel_pose", "prior", "posevel_point", "posevel_posevel",
                                            "point_equality"};
    return n;
  }
  inline const std::vector<std::string>& variableClassNames() {
    static const std::vector<std::string> n{"other", "SE3PoseVel", "SE3Pose", "SE3Point", "", "", "", ""};
    return n;
  }
  inline int factorClassOf(const std::string& type) {
    const auto& n = factorClassNames();
    for (size_t k = 1; k < n.size(); ++k)
      if (n[k] == type)
        return static_cast<int>(k);
    return 0;
  }
  // Assign the flop class of every node added since the last call (nodes are only ever appended)
  inline void assignFlopClasses(FactorGraph& g, size_t& vars_done, size_t& factors_done) {
    for (; vars_done < g.variable_nodes_.size(); ++vars_done) {
      auto& v = g.variable_nodes_[vars_done];
      if (std::dynamic_pointer_cast<SE3PoseVel>(v))
        v->flop_class_ = 1;
      else if (std::dynamic_pointer_cast<SE3Pose>(v))
        v->flop_class_ = 2;
      else if (std::dynamic_pointer_cast<SE3Point>(v))
        v->flop_class_ = 3;
    }
    for (; factors_done < g.factor_nodes_.size(); ++factors_done)
      withFactor(g.factor_nodes_[factors_done], [&](auto fp, const std::string& type) { fp->flop_class_ = factorClassOf(type); });
  }

  // NPY v1/v2 reader for little-endian float64 / int64 arrays in C order
  struct NpyArray {
    std::vector<size_t> shape;
    std::vector<double> f;
    std::vector<int64_t> i;
    size_t size() const {
      size_t n = 1;
      for (size_t d : shape)
        n *= d;
      return n;
    }
  };
  inline NpyArray readNpy(const std::string& path) {
    std::ifstream fh(path, std::ios::binary);
    if (!fh)
      throw std::runtime_error("cannot read " + path);
    char magic[6];
    fh.read(magic, 6);
    if (std::string(magic, 6) != std::string("\x93NUMPY", 6))
      throw std::runtime_error("not an npy file: " + path);
    unsigned char ver[2];
    fh.read(reinterpret_cast<char*>(ver), 2);
    size_t hl = 0;
    if (ver[0] == 1) {
      uint16_t h;
      fh.read(reinterpret_cast<char*>(&h), 2);
      hl = h;
    } else {
      uint32_t h;
      fh.read(reinterpret_cast<char*>(&h), 4);
      hl = h;
    }
    std::string header(hl, ' ');
    fh.read(&header[0], static_cast<std::streamsize>(hl));
    if (header.find("'fortran_order': True") != std::string::npos)
      throw std::runtime_error("fortran order not supported: " + path);
    const bool f8 = header.find("'<f8'") != std::string::npos, i8 = header.find("'<i8'") != std::string::npos;
    if (!f8 && !i8)
      throw std::runtime_error("only <f8 and <i8 are supported: " + path);
    NpyArray a;
    const size_t p0 = header.find("'shape': (") + 10, p1 = header.find(')', p0);
    std::stringstream ss(header.substr(p0, p1 - p0));
    std::string tok;
    while (std::getline(ss, tok, ','))
      if (tok.find_first_not_of(' ') != std::string::npos)
        a.shape.push_back(static_cast<size_t>(std::stoll(tok)));
    const size_t n = a.size();
    if (f8) {
      a.f.resize(n);
      fh.read(reinterpret_cast<char*>(a.f.data()), static_cast<std::streamsize>(n * 8));
    } else {
      a.i.resize(n);
      fh.read(reinterpret_cast<char*>(a.i.data()), static_cast<std::streamsize>(n * 8));
    }
    if (!fh)
      throw std::runtime_error("short read: " + path);
    return a;
  }

  // ---- threshold relinearization (Phase D, WP1; certify/THRESHOLD_RELIN.md)

  inline bool isPoseId(const std::string& id) {
    return id.find("POSE_") != std::string::npos;
  }

  inline void enableThresholdRelinearization(GbpSolver& solver, double delta, int every, const std::string& metric) {
    solver.threshold_relinearization_ = true;
    solver.relinearize_threshold_     = delta;
    solver.relinearize_every_         = every;
    if (metric == "physical")
      solver.relinearize_metric_ = GbpSolver::RelinMetric::Physical;
    else if (metric == "D")
      solver.relinearize_metric_ = GbpSolver::RelinMetric::HBlock;
    else
      throw std::runtime_error("--relin-metric must be physical or D");
  }

  /** Per-round log of the threshold rule: re-frozen poses and other variables, relinearized factors, and the
   *  D-metric displacement d_v of poses (median, max) and of all variables (max) at the round's test. */
  struct RelinLog {
    std::ofstream f;
    explicit RelinLog(const std::string& path) : f(path) {
      f << "round,wall_s,relin_poses,relin_other,relin_factors,new_factors,d_pose_median,d_pose_max,d_all_max\n";
    }
    void row(const GbpSolver& solver, const FactorGraph& g, double wall_s) {
      const auto& st = solver.threshold_stats_;
      int np = 0, no = 0;
      for (int v : st.relinearized_vars)
        (isPoseId(g.variable_nodes_[v]->id_) ? np : no)++;
      std::vector<double> dp;
      double dmax = 0.0;
      for (size_t v = 0; v < st.displacement.size(); ++v) {
        dmax = std::max(dmax, st.displacement[v]);
        if (isPoseId(g.variable_nodes_[v]->id_))
          dp.push_back(st.displacement[v]);
      }
      double med = 0.0, pmax = 0.0;
      if (!dp.empty()) {
        pmax = *std::max_element(dp.begin(), dp.end());
        std::nth_element(dp.begin(), dp.begin() + dp.size() / 2, dp.end());
        med = dp[dp.size() / 2];
      }
      f << st.round << "," << wall_s << "," << np << "," << no << "," << st.relinearized_factors << "," << st.new_factors << "," << jsonNum(med)
        << "," << jsonNum(pmax) << "," << jsonNum(dmax) << "\n";
      f.flush();
    }
  };

  /**
   * Phase D, WP1b: rounds with a work account, curve exports and an optional stop point.
   * Work of a round = linearizations + message computations:
   *  - linearizations: every factor for G-solver's per-round schedule (each re-evaluates its error and Jacobian);
   *    in threshold mode the factors linearized for the first time plus those the rule relinearizes;
   *  - messages: every factor-to-variable message, and every variable-to-factor message of the variables with at
   *    least two edges (a single-edge variable never sends; variable_node.cpp).
   * A round is accounted before it runs (GbpSolver::planThresholdRound), so a budget W stops exactly after the
   * last round whose cumulative work is <= W: the state there is exported to <out>/at_budget (charts at the
   * beliefs) and the run continues. Every `curve_every` rounds the state is exported to <out>/curve/rNNNNN.
   */
  struct RoundRunner {
    GbpSolver& solver;
    FactorGraph& g;
    std::string out;
    long long budget = 0; // 0: none
    int curve_every  = 0; // 0: no curve exports
    Json meta;
    int rounds = 0, budget_round = -1;
    long long cumulative = 0, budget_work = 0;
    std::ofstream log;
    std::unique_ptr<RelinLog> relin_log;
    size_t nv_ = SIZE_MAX, nf_ = SIZE_MAX;
    long long messages_ = 0;

    // Phase D, WP6a: per-round GBP flops (gsolver/core/flop_count.h, certify/FLOPS.md)
    bool count_flops     = false;
    int flop_trace_round = -1; // record one operation of every kind in this round (hand-count validation)
    std::ofstream flop_log, flop_class_log;
    size_t classed_vars_ = 0, classed_factors_ = 0;

    void enableFlops(int trace_round) {
      count_flops      = true;
      flop_trace_round = trace_round;
      flops::enabled   = true;
      flops::collect(); // discard anything counted before
      flop_log.open(out + "/flops.csv");
      flop_log << "round,gbp_flops,factor_update,f2v,belief,v2f,linearizations,factor_updates,f2v_messages,belief_updates,"
                  "v2f_messages\n";
      flop_class_log.open(out + "/flops_by_class.csv");
      flop_class_log << "round,category,class,flops,ops\n";
    }
    void beforeRound() {
      if (!count_flops)
        return;
      assignFlopClasses(g, classed_vars_, classed_factors_);
      if (rounds + 1 == flop_trace_round)
        flops::trace_per_kind = 1;
    }
    void afterRound() {
      if (!count_flops)
        return;
      const flops::Tally t = flops::collect();
      uint64_t F[flops::NumCategories] = {}, N[flops::NumCategories] = {};
      static const char* cat[] = {"linearize", "factor_update", "f2v", "belief", "v2f"};
      for (int c = 0; c < flops::kClasses; ++c)
        for (int k = 0; k < flops::NumCategories; ++k) {
          F[k] += t.flops[c][k];
          N[k] += t.ops[c][k];
          if (t.ops[c][k])
            flop_class_log << rounds << "," << cat[k] << ","
                           << (k <= flops::F2V ? factorClassNames()[c] : variableClassNames()[c]) << "," << t.flops[c][k] << ","
                           << t.ops[c][k] << "\n";
        }
      const uint64_t total = F[flops::FactorUpdate] + F[flops::F2V] + F[flops::Belief] + F[flops::V2F];
      flop_log << rounds << "," << total << "," << F[flops::FactorUpdate] << "," << F[flops::F2V] << "," << F[flops::Belief] << ","
               << F[flops::V2F] << "," << N[flops::Linearize] << "," << N[flops::FactorUpdate] << "," << N[flops::F2V] << ","
               << N[flops::Belief] << "," << N[flops::V2F] << "\n";
      flop_log.flush();
      flop_class_log.flush();
      if (flops::trace_per_kind > 0) {
        flops::trace_per_kind = 0;
        std::ofstream tr(out + "/flops_trace.csv");
        tr << "round,category,class,a,b,degree,flops\n";
        for (const auto& r : flops::trace)
          tr << rounds << "," << cat[r.cat] << ","
             << (r.cat <= flops::F2V ? factorClassNames()[r.cls] : variableClassNames()[r.cls]) << "," << r.a << "," << r.b << ","
             << r.deg << "," << r.flops << "\n";
        flops::trace.clear();
      }
    }

    RoundRunner(GbpSolver& s, FactorGraph& graph, const std::string& dir, long long work_budget, int every, Json m)
        : solver(s), g(graph), out(dir), budget(work_budget), curve_every(every), meta(std::move(m)) {
      makeDirs(out);
      log.open(out + "/work.csv");
      log << "round,linearizations,messages,cumulative_work,wall_s,node_updates,asleep\n";
      if (solver.threshold_relinearization_)
        relin_log = std::make_unique<RelinLog>(out + "/relin_rounds.csv");
    }

    long long messageCount() {
      if (g.variable_nodes_.size() == nv_ && g.factor_nodes_.size() == nf_)
        return messages_;
      std::unordered_map<std::string, int> degree;
      long long f2v = 0, v2f = 0;
      for (auto& f : g.factor_nodes_) {
        const size_t k = f->numberOfEdges();
        f2v += static_cast<long long>(k);
        for (size_t i = 0; i < k; ++i)
          degree[f->edgeVariableId(i)]++;
      }
      for (auto& [id, d] : degree)
        if (d >= 2)
          v2f += d;
      nv_       = g.variable_nodes_.size();
      nf_       = g.factor_nodes_.size();
      messages_ = f2v + v2f;
      return messages_;
    }

    void exportBeliefs(const std::string& dir, const std::string& what) {
      Json m = meta;
      m.integer("round", rounds).num("cumulative_work", static_cast<double>(cumulative)).str("export", what);
      exportModel(g, dir, Charts::Beliefs, m);
    }

    // ---- Phase D, WP2: the sidecar protocol (certify/SIDECAR.md). Before the message passing of every round the
    // current linearization is exported to <sidecar>/model (frozen charts, state z), the per-pose round data to
    // <sidecar>/round_info.tsv, and <sidecar>/ready is written; the sidecar answers with <sidecar>/decision (the
    // round number, then the ids of the poses that sleep during this round's message passing).
    std::string sidecar;              // empty: off
    std::string persist;              // exports of the sampled rounds go to <persist>/export_rNNNNN
    std::vector<int> persist_rounds;  // e.g. 1, 10 and the last
    int last_round = -1;
    int pause_every = 0;              // checkpoint pause every N rounds (0: none)
    long long node_updates = 0;       // cumulative variable updates (awake variables with at least one edge)
    std::unordered_map<std::string, int> var_index_;

    void writeAtomic(const std::string& path, const std::string& text) {
      std::ofstream(path + ".tmp") << text;
      std::rename((path + ".tmp").c_str(), path.c_str());
    }

    // Messages and node updates of a round with the given sleep mask (sleeping variables neither send nor receive)
    std::pair<long long, long long> awakeWork() {
      if (var_index_.size() != g.variable_nodes_.size()) {
        var_index_.clear();
        for (size_t v = 0; v < g.variable_nodes_.size(); ++v)
          var_index_[g.variable_nodes_[v]->id_] = static_cast<int>(v);
      }
      const auto& asleep = solver.asleep_;
      auto sleeping = [&](int v) { return v < static_cast<int>(asleep.size()) && asleep[v]; };
      std::vector<int> degree(g.variable_nodes_.size(), 0);
      long long f2v = 0;
      for (auto& f : g.factor_nodes_)
        for (size_t i = 0; i < f->numberOfEdges(); ++i) {
          const int v = var_index_.at(f->edgeVariableId(i));
          degree[v]++;
          if (!sleeping(v))
            f2v++;
        }
      long long v2f = 0, updates = 0;
      for (size_t v = 0; v < degree.size(); ++v)
        if (!sleeping(static_cast<int>(v)) && degree[v] > 0) {
          updates++;
          if (degree[v] >= 2)
            v2f += degree[v];
        }
      return {f2v + v2f, updates};
    }

    void sidecarRound() {
      beforeRound();
      const long long lin = solver.planThresholdRound(g);
      solver.applyThresholdRelinearization(g);
      const int r = solver.threshold_stats_.round;
      exportModel(g, sidecar + "/model", Charts::Frozen, meta);
      {
        const auto& st = solver.threshold_stats_;
        std::vector<char> relin(g.variable_nodes_.size(), 0);
        for (int v : st.relinearized_vars)
          relin[v] = 1;
        std::ostringstream info;
        info << "round\t" << r << "\nlast\t" << (r == last_round ? 1 : 0) << "\nnew_factors\t" << st.new_factors
             << "\nrelinearized_factors\t" << st.relinearized_factors << "\n";
        info << "id\td\trelinearized\tasleep\n";
        for (size_t v = 0; v < g.variable_nodes_.size(); ++v)
          if (isPoseId(g.variable_nodes_[v]->id_))
            info << g.variable_nodes_[v]->id_ << "\t" << jsonNum(st.displacement[v]) << "\t" << int(relin[v]) << "\t"
                 << int(v < solver.asleep_.size() && solver.asleep_[v]) << "\n";
        writeAtomic(sidecar + "/round_info.tsv", info.str());
        writeAtomic(sidecar + "/ready", std::to_string(r) + "\n");
      }
      if (std::find(persist_rounds.begin(), persist_rounds.end(), r) != persist_rounds.end() || r == last_round) {
        char name[48];
        snprintf(name, sizeof(name), "/export_r%05d", r);
        Json m = meta;
        m.integer("round", r).str("export", "linearization and state before this round's message passing");
        exportModel(g, persist + name, Charts::Frozen, m);
      }
      // wait for the decision of this round
      std::set<std::string> sleep_ids;
      while (true) {
        std::ifstream f(sidecar + "/decision");
        int rr = -1;
        if (f && (f >> rr) && rr == r) {
          std::string id;
          while (f >> id)
            sleep_ids.insert(id);
          break;
        }
        struct stat stt;
        if (stat((sidecar + "/abort").c_str(), &stt) == 0)
          throw std::runtime_error("sidecar aborted at round " + std::to_string(r));
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
      for (size_t v = 0; v < g.variable_nodes_.size(); ++v)
        if (isPoseId(g.variable_nodes_[v]->id_))
          solver.setAsleep(g, v, sleep_ids.count(g.variable_nodes_[v]->id_) > 0);
      const auto [msg, updates] = awakeWork();
      auto t0 = std::chrono::steady_clock::now();
      solver.performIteration(g);
      const double wall = seconds_since(t0);
      ++rounds;
      cumulative += lin + msg;
      node_updates += updates;
      log << rounds << "," << lin << "," << msg << "," << cumulative << "," << wall << "," << updates << ","
          << sleep_ids.size() << "\n";
      log.flush();
      afterRound();
      if (relin_log)
        relin_log->row(solver, g, wall);
      if (pause_every > 0 && rounds % pause_every == 0 && rounds != last_round)
        if (!pause("r" + std::to_string(rounds)))
          throw std::runtime_error("aborted at the checkpoint after round " + std::to_string(rounds));
    }

    void round() {
      if (!sidecar.empty()) {
        sidecarRound();
        return;
      }
      beforeRound();
      const long long lin = solver.threshold_relinearization_ ? solver.planThresholdRound(g)
                                                              : static_cast<long long>(g.factor_nodes_.size());
      const long long msg = messageCount();
      if (budget > 0 && budget_round < 0 && cumulative + lin + msg > budget) {
        budget_round = rounds;
        budget_work  = cumulative;
        exportBeliefs(out + "/at_budget", "state after the last round within the work budget");
      }
      auto t0 = std::chrono::steady_clock::now();
      solver.performIteration(g);
      const double wall = seconds_since(t0);
      ++rounds;
      cumulative += lin + msg;
      log << rounds << "," << lin << "," << msg << "," << cumulative << "," << wall << ",,\n";
      log.flush();
      afterRound();
      if (relin_log)
        relin_log->row(solver, g, wall);
      if (curve_every > 0 && rounds % curve_every == 0) {
        char name[32];
        snprintf(name, sizeof(name), "/curve/r%05d", rounds);
        exportBeliefs(out + name, "curve point");
      }
    }

    // Stop point for a checkpointed run: export, then wait until <out>/continue (or <out>/abort) exists.
    bool pause(const std::string& tag) {
      exportBeliefs(out + "/checkpoint_" + tag, "checkpoint");
      std::cout << "checkpoint " << tag << " written after round " << rounds << "; waiting for " << out << "/continue"
                << std::endl;
      while (true) {
        struct stat st;
        if (stat((out + "/abort").c_str(), &st) == 0)
          return false;
        if (stat((out + "/continue").c_str(), &st) == 0) {
          std::remove((out + "/continue").c_str());
          return true;
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
      }
    }

    void finish() {
      std::ofstream(out + "/work_summary.json")
        << "{\"rounds\": " << rounds << ", \"cumulative_work\": " << cumulative << ", \"budget\": " << budget
        << ", \"budget_round\": " << budget_round << ", \"budget_work\": " << budget_work << "}\n";
    }
  };

  /** Re-freezes per variable over the whole run: id, pose flag, round of first linearization, count, and the
   *  rounds the run had (rounds per linearization = (rounds - created_round + 1) / (1 + relinearizations)). */
  inline void writeRelinCounts(const std::string& path, const GbpSolver& solver, const FactorGraph& g) {
    std::ofstream f(path);
    f << "id,is_pose,created_round,relinearizations,rounds\n";
    const auto& st = solver.threshold_stats_;
    for (size_t v = 0; v < st.relinearizations.size(); ++v)
      f << g.variable_nodes_[v]->id_ << "," << (isPoseId(g.variable_nodes_[v]->id_) ? 1 : 0) << ","
        << st.created_round[v] << "," << st.relinearizations[v] << "," << st.round << "\n";
  }

  // ---- Phase D, WP6b: the nonlinear estimating-equation residual at a stepped point (certify/NL_PROBE.md)

  /**
   * Load the linearization points and tangent states of a frozen-chart export into a freshly built graph (the same
   * driver construction, so the same variable ids and factor order), make every variable frozen at its point and
   * relinearize every factor there. Returns the largest differences between the recomputed J_lin/e_lin and the
   * export's, per factor type (they must agree to rounding: same code, same points).
   */
  inline Json loadFrozenState(FactorGraph& g, const std::string& dir) {
    auto vars = orderVariables(g);
    {
      std::ifstream fv(dir + "/variables.tsv");
      std::string line;
      std::getline(fv, line);
      size_t k = 0;
      while (std::getline(fv, line)) {
        if (line.empty())
          continue;
        std::stringstream ss(line);
        std::string idx, id;
        std::getline(ss, idx, '\t');
        std::getline(ss, id, '\t');
        if (k >= vars.size() || vars[k].id != id)
          throw std::runtime_error("export variable " + std::to_string(k) + " (" + id + ") does not match the rebuilt graph");
        ++k;
      }
      if (k != vars.size())
        throw std::runtime_error("export has " + std::to_string(k) + " variables, the rebuilt graph " + std::to_string(vars.size()));
    }
    const NpyArray T = readNpy(dir + "/pose_T_lin.npy"), V = readNpy(dir + "/pose_v_lin.npy"), S3 = readNpy(dir + "/se3_T_lin.npy"),
                   P = readNpy(dir + "/point_p_lin.npy"), Z = readNpy(dir + "/z.npy");
    size_t kp = 0, ks = 0, kq = 0, off = 0;
    auto iso = [](const std::vector<double>& a, size_t k) {
      Eigen::Matrix4d M;
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
          M(r, c) = a[16 * k + 4 * r + c];
      Eigen::Isometry3d X;
      X.matrix() = M;
      return X;
    };
    for (auto& r : vars)
      withVariable(r.node, [&](auto vp) {
        using VT = typename std::decay_t<decltype(*vp)>;
        typename VT::EstimateType lin;
        if constexpr (std::is_same_v<VT, SE3PoseVel>) {
          lin.pose = iso(T.f, kp);
          for (int k = 0; k < 6; ++k)
            lin.vel(k) = V.f[6 * kp + k];
          ++kp;
        } else if constexpr (std::is_same_v<VT, SE3Pose>) {
          lin = iso(S3.f, ks++);
        } else {
          lin = Eigen::Vector3d(P.f[3 * kq], P.f[3 * kq + 1], P.f[3 * kq + 2]);
          ++kq;
        }
        vp->mu_     = lin;
        vp->mu_lin_ = lin;
        for (int k = 0; k < VT::LieDim; ++k)
          vp->z_(k) = Z.f[off + k];
        off += VT::LieDim;
        vp->frozen_ = true;
        for (auto& e : vp->edges_) {
          e->variable_to_factor_message_.mu_ = lin;
          e->frozen_                         = true;
        }
      });
    const bool was = flops::enabled;
    flops::enabled = false;
    for (auto& f : g.factor_nodes_)
      f->freezeLinearization();
    flops::enabled = was;
    // compare with the export
    std::map<std::string, std::pair<NpyArray, NpyArray>> ref;
    std::map<std::string, size_t> pos;
    std::map<std::string, double> dJ, de, mJ, me;
    for (auto& f : g.factor_nodes_)
      withFactor(f, [&](auto fp, const std::string& type) {
        using FT = typename std::decay_t<decltype(*fp)>;
        if (!ref.count(type))
          ref[type] = {readNpy(dir + "/f_" + type + "_J.npy"), readNpy(dir + "/f_" + type + "_e.npy")};
        const size_t k = pos[type]++;
        const int m = FT::MatrixErrorDim::RowsAtCompileTime, n = FT::VariablesDim_;
        const auto& J = ref[type].first.f;
        const auto& e = ref[type].second.f;
        for (int a = 0; a < m; ++a) {
          for (int b = 0; b < n; ++b) {
            dJ[type] = std::max(dJ[type], std::abs(fp->J_lin_(a, b) - J[(k * m + a) * n + b]));
            mJ[type] = std::max(mJ[type], std::abs(J[(k * m + a) * n + b]));
          }
          de[type] = std::max(de[type], std::abs(fp->error_lin_(a) - e[k * m + a]));
          me[type] = std::max(me[type], std::abs(e[k * m + a]));
        }
      });
    Json out;
    out.integer("variables", static_cast<long long>(vars.size()));
    for (auto& [type, n] : pos) {
      if (ref[type].first.shape.empty() || ref[type].first.shape[0] != n)
        throw std::runtime_error("factor count mismatch for type " + type);
      out.set(type, "{\"factors\": " + std::to_string(n) + ", \"max_abs_dJ\": " + jsonNum(dJ[type]) + ", \"max_abs_J\": " +
                      jsonNum(mJ[type]) + ", \"max_abs_de\": " + jsonNum(de[type]) + ", \"max_abs_e\": " + jsonNum(me[type]) + "}");
    }
    return out;
  }

  /**
   * Probe server. A probe is a target set T of variables (canonical export order) with tangent offsets z1_v in their
   * frozen charts. Every factor touching T is evaluated by its own updateErrorAndJacobian() -- the code the solver
   * relinearizes with -- at x = x_lin ⊞ z, with z = z1 on T and the current state z_ elsewhere, and for v in T
   *   F_nl[c]_v  = sum over the factors f of class c touching v of [J~_f(x)' Sigma_f^-1 e_f(x)]_v,
   *   F_lin[c]_v = the same with the frozen linear model: J_lin' Sigma^-1 (e_lin + J_lin tau(z)),
   *   F_jac[c]_v = the new Jacobian with the linear model's error: J~_f(x)' Sigma^-1 (e_lin + J_lin tau(z)),
   *   F_true[c]_v = the same with the error's true linear model: J~_f(x)' Sigma^-1 (e_lin + J_lin,true tau(z)),
   * so F_jac - F_lin is the Gauss-Newton Jacobian-change term, F_true - F_jac the first-order mismatch of the solver's
   * linear model when its Jacobian is not the error's derivative (stereo, Theorem T0), F_nl - F_true the curvature.
   * The solver state is unchanged: message means are pointed back at the beliefs after each probe and restored at
   * the end of the session, and each factor's error_, J_ and inlier flag are restored after its evaluation.
   *
   * Protocol (files in `dir`): the client writes probe_targets.npy (int64), probe_offsets.npy (int64, n_probes + 1)
   * and probe_z.npy (float64, the offsets of every target in turn), then probe_request (a request number). The server
   * answers with probe_F.npy (float64, 4 x n_classes x total target dofs: F_nl, F_lin, F_jac, F_true), probe_stats.npy (int64,
   * n_probes x 2: factors evaluated, factors whose inlier status at x differs from the linearization point) and
   * probe_done (the request number). probe_classes.txt lists the factor classes. probe_quit ends serve().
   */
  struct ProbeServer {
    FactorGraph& g;
    std::string dir;
    std::vector<VarRec> vars;
    std::vector<int> dim;                          // per canonical variable
    std::vector<std::vector<int>> var_factors;     // canonical variable -> factor indices
    std::vector<std::vector<int>> factor_vars;     // factor -> canonical variables (edge order)
    std::vector<int> factor_class;                 // index into classes
    std::vector<std::string> classes;
    std::vector<Eigen::VectorXd> zev;              // tangent offset at which each variable is evaluated
    std::vector<std::function<void(Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd&, int&)>> eval;
    std::vector<int> stamp;
    int stamp_now = 0, last_request = 0;
    bool in_session = false;

    ProbeServer(FactorGraph& graph, const std::string& d) : g(graph), dir(d) {
      makeDirs(dir);
      vars = orderVariables(g);
      std::unordered_map<std::string, int> index;
      for (size_t i = 0; i < vars.size(); ++i) {
        index[vars[i].id] = static_cast<int>(i);
        dim.push_back(vars[i].dim);
      }
      var_factors.resize(vars.size());
      for (size_t k = 0; k < g.factor_nodes_.size(); ++k)
        withFactor(g.factor_nodes_[k], [&](auto fp, const std::string& type) {
          using FT = typename std::decay_t<decltype(*fp)>;
          int c = static_cast<int>(std::find(classes.begin(), classes.end(), type) - classes.begin());
          if (c == static_cast<int>(classes.size()))
            classes.push_back(type);
          factor_class.push_back(c);
          std::vector<int> vs;
          forEachEdge<0>(*fp, [&](int, auto& edge) { vs.push_back(index.at(edge->variable_id_)); });
          for (int v : vs)
            var_factors[v].push_back(static_cast<int>(k));
          factor_vars.push_back(vs);
          const std::vector<Eigen::VectorXd>* zp = &zev;
          eval.push_back([fp, vs, zp](Eigen::VectorXd& gnl, Eigen::VectorXd& glin, Eigen::VectorXd& gjac, Eigen::VectorXd& gtrue,
                                      int& flip) {
            const typename FT::VectorErrorDim e0    = fp->error_;
            const typename FT::MatrixJacobianDim J0 = fp->J_;
            bool in0                                = true;
            if constexpr (std::is_same_v<FT, SE3PoseVelStereoPoint>)
              in0 = fp->inlier_;
            fp->updateErrorAndJacobian();
            const typename FT::MatrixErrorDim W = fp->Sigma_.inverse();
            gnl                                 = fp->J_.transpose() * (W * fp->error_);
            typename FT::VectorVariablesDim tau;
            int o = 0;
            for (int v : vs) {
              const auto& zv                   = (*zp)[v];
              tau.segment(o, zv.size())        = zv;
              o += static_cast<int>(zv.size());
            }
            const typename FT::VectorErrorDim elin = fp->error_lin_ + fp->J_lin_ * tau;
            glin                                    = fp->J_lin_.transpose() * (W * elin);
            gjac                                    = fp->J_.transpose() * (W * elin); // new Jacobian, linear error
            // the true Jacobian at x_lin (Theorem T0: the stereo rotation columns without the factor 2 of
            // se3_posevel_stereo_point.cpp; every other type's Jacobian is used as the solver's)
            typename FT::MatrixJacobianDim Jt = fp->J_lin_;
            if constexpr (std::is_same_v<FT, SE3PoseVelStereoPoint>)
              Jt.template block<3, 3>(0, 3) *= 0.5;
            gtrue = fp->J_.transpose() * (W * (fp->error_lin_ + Jt * tau));
            flip = (fp->J_lin_.isZero(0.0) != fp->J_.isZero(0.0)) ? 1 : 0;
            if constexpr (std::is_same_v<FT, SE3PoseVelStereoPoint>)
              fp->inlier_ = in0;
            fp->error_ = e0;
            fp->J_     = J0;
          });
        });
      stamp.assign(g.factor_nodes_.size(), 0);
      std::ofstream fc(dir + "/probe_classes.txt");
      for (auto& c : classes)
        fc << c << "\n";
    }

    void setPoint(int v, const Eigen::VectorXd& z) {
      withVariable(vars[v].node, [&](auto vp) {
        using VT = typename std::decay_t<decltype(*vp)>;
        const typename VT::VectorLieDim tau = z;
        const auto point                     = vp->boxplus(tau); // at mu_ == mu_lin_ (frozen)
        for (auto& e : vp->edges_)
          e->variable_to_factor_message_.mu_ = point;
      });
      zev[v] = z;
    }

    void beginSession() {
      zev.assign(vars.size(), Eigen::VectorXd());
      for (size_t v = 0; v < vars.size(); ++v) {
        if (!vars[v].node->isFrozen())
          throw std::runtime_error("probe: variable " + vars[v].id + " is not frozen");
        vars[v].node->beginEvaluateAtBelief();
        zev[v] = vars[v].node->tangentState();
      }
      in_session = true;
    }
    void endSession() {
      if (!in_session)
        return;
      for (auto& r : vars)
        r.node->restoreMessageMeans();
      in_session = false;
    }

    void process(int request) {
      const NpyArray T = readNpy(dir + "/probe_targets.npy"), O = readNpy(dir + "/probe_offsets.npy"), Z = readNpy(dir + "/probe_z.npy");
      const size_t np = O.i.size() - 1, nc = classes.size();
      size_t L = 0;
      for (int64_t t : T.i)
        L += dim.at(t);
      if (Z.f.size() != L)
        throw std::runtime_error("probe_z has " + std::to_string(Z.f.size()) + " values, the targets need " + std::to_string(L));
      std::vector<double> F(4 * nc * L, 0.0);
      std::vector<int64_t> stats(2 * np, 0);
      std::vector<int> slot(vars.size(), -1); // target -> first dof in this probe's block of F
      size_t zoff = 0;
      for (size_t p = 0; p < np; ++p) {
        const size_t t0 = O.i[p], t1 = O.i[p + 1];
        for (size_t k = t0; k < t1; ++k) {
          const int v = static_cast<int>(T.i[k]);
          slot[v]     = static_cast<int>(zoff);
          setPoint(v, Eigen::Map<const Eigen::VectorXd>(Z.f.data() + zoff, dim[v]));
          zoff += dim[v];
        }
        ++stamp_now;
        std::vector<int> fl;
        for (size_t k = t0; k < t1; ++k)
          for (int f : var_factors[T.i[k]])
            if (stamp[f] != stamp_now) {
              stamp[f] = stamp_now;
              fl.push_back(f);
            }
        std::vector<Eigen::VectorXd> gnl(fl.size()), glin(fl.size()), gjac(fl.size()), gtrue(fl.size());
        std::vector<int> flip(fl.size(), 0);
#pragma omp parallel for schedule(dynamic, 256)
        for (size_t k = 0; k < fl.size(); ++k)
          eval[fl[k]](gnl[k], glin[k], gjac[k], gtrue[k], flip[k]);
        for (size_t k = 0; k < fl.size(); ++k) {
          const int f = fl[k], c = factor_class[f];
          int o = 0;
          for (int v : factor_vars[f]) {
            if (slot[v] >= 0)
              for (int a = 0; a < dim[v]; ++a) {
                F[(0 * nc + c) * L + slot[v] + a] += gnl[k](o + a);
                F[(1 * nc + c) * L + slot[v] + a] += glin[k](o + a);
                F[(2 * nc + c) * L + slot[v] + a] += gjac[k](o + a);
                F[(3 * nc + c) * L + slot[v] + a] += gtrue[k](o + a);
              }
            o += dim[v];
          }
          stats[2 * p + 1] += flip[k];
        }
        stats[2 * p] = static_cast<int64_t>(fl.size());
        for (size_t k = t0; k < t1; ++k) {
          const int v = static_cast<int>(T.i[k]);
          slot[v]     = -1;
          setPoint(v, vars[v].node->tangentState()); // back to the belief
        }
      }
      writeNpy(dir + "/probe_F.npy", F, {4, nc, L});
      writeNpy(dir + "/probe_stats.npy", stats, {np, 2});
      std::ofstream(dir + "/probe_done.tmp") << request << "\n";
      std::rename((dir + "/probe_done.tmp").c_str(), (dir + "/probe_done").c_str());
    }

    /**
     * Central-difference check of the factors' Jacobians at their linearization points: for the first n factors of
     * every class, J_fd[:, k] = (e(x_lin (+) h e_k) - e(x_lin (+) -h e_k)) / 2h along each variable's own chart, the other
     * variables at x_lin, compared with the solver's J_lin. Written to fd_check.tsv, one row per factor and tangent
     * column block of 3: max |J_lin - J_fd|, max |J_fd|, and the least-squares ratio <J_lin, J_fd> / <J_fd, J_fd>.
     */
    void fdCheck(int n_per_class, double h) {
      std::ofstream out(dir + "/fd_check.tsv");
      out << "class\tfactor\tslot\tcols\tmax_abs_diff\tmax_abs_fd\tratio\n";
      std::vector<int> seen(classes.size(), 0);
      std::vector<std::vector<double>> dump(classes.size());
      std::vector<std::vector<size_t>> shape(classes.size());
      for (size_t f = 0; f < g.factor_nodes_.size(); ++f) {
        const int c = factor_class[f];
        if (seen[c] >= n_per_class)
          continue;
        ++seen[c];
        withFactor(g.factor_nodes_[f], [&](auto fp, const std::string&) {
          using FT = typename std::decay_t<decltype(*fp)>;
          const auto e0 = fp->error_;
          const auto J0 = fp->J_;
          bool in0      = true;
          if constexpr (std::is_same_v<FT, SE3PoseVelStereoPoint>)
            in0 = fp->inlier_;
          const auto& vs = factor_vars[f];
          for (int v : vs)
            setPoint(v, Eigen::VectorXd::Zero(dim[v]));
          typename FT::MatrixJacobianDim Jfd;
          int col = 0;
          for (int v : vs) {
            for (int k = 0; k < dim[v]; ++k) {
              Eigen::VectorXd z = Eigen::VectorXd::Zero(dim[v]);
              z(k)              = h;
              setPoint(v, z);
              fp->updateErrorAndJacobian();
              const typename FT::VectorErrorDim ep = fp->error_;
              setPoint(v, -z);
              fp->updateErrorAndJacobian();
              Jfd.col(col + k) = (ep - fp->error_) / (2 * h);
              setPoint(v, Eigen::VectorXd::Zero(dim[v]));
            }
            col += dim[v];
          }
          appendRowMajor(dump[c], Jfd);
          shape[c] = {static_cast<size_t>(seen[c]), static_cast<size_t>(Jfd.rows()), static_cast<size_t>(Jfd.cols())};
          int o = 0;
          for (size_t slot = 0; slot < vs.size(); ++slot) {
            for (int b = 0; b < dim[vs[slot]]; b += 3) {
              const Eigen::MatrixXd A = fp->J_lin_.middleCols(o + b, 3), B = Jfd.middleCols(o + b, 3);
              const double bb = (B.array() * B.array()).sum();
              out << classes[c] << "\t" << fp->id_ << "\t" << slot << "\t" << b << "-" << (b + 2) << "\t" << (A - B).cwiseAbs().maxCoeff()
                  << "\t" << B.cwiseAbs().maxCoeff() << "\t" << (bb > 0 ? (A.array() * B.array()).sum() / bb : std::nan("")) << "\n";
            }
            o += dim[vs[slot]];
          }
          for (int v : vs)
            setPoint(v, vars[v].node->tangentState()); // back to the belief
          if constexpr (std::is_same_v<FT, SE3PoseVelStereoPoint>)
            fp->inlier_ = in0;
          fp->error_ = e0;
          fp->J_     = J0;
        });
      }
      out.close();
      // the central-difference Jacobians themselves (the first factors of each class, in graph order)
      for (size_t c = 0; c < classes.size(); ++c)
        if (!dump[c].empty())
          writeNpy(dir + "/fd_J_" + classes[c] + ".npy", dump[c], shape[c]);
      std::ofstream(dir + "/fd_done") << "1\n";
    }

    // Answer a pending request, if any; returns true if one was answered
    bool poll() {
      std::ifstream f(dir + "/probe_request");
      int r = -1;
      if (!(f && (f >> r)) || r <= last_request)
        return false;
      if (!in_session)
        beginSession();
      process(r);
      last_request = r;
      return true;
    }

    // Serve until probe_quit exists
    void serve() {
      while (true) {
        if (poll())
          continue;
        {
          std::ifstream fr(dir + "/fd_request");
          int n = 0;
          double h = 0.0;
          if (fr && (fr >> n >> h)) {
            fr.close();
            std::remove((dir + "/fd_request").c_str());
            if (!in_session)
              beginSession();
            fdCheck(n, h);
            continue;
          }
        }
        struct stat st;
        if (stat((dir + "/probe_quit").c_str(), &st) == 0)
          break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      endSession();
    }
  };

} // namespace certify
