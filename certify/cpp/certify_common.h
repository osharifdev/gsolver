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
#include <memory>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>

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
      // Point every incoming message mean at the belief mean, evaluate each factor there, then restore.
      for (auto& v : g.variable_nodes_)
        v->beginFreeze();
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
            auto tau = vp->boxminus(vp->mu_msg_stash_[i]);
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
        const bool frozen             = (charts == Charts::Frozen);
        const typename VT::EstimateType& lin = frozen ? vp->mu_lin_ : vp->mu_;
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
          auto tau = frozen ? m.tau_ : vp->boxminus(m.mu_);
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

} // namespace certify
