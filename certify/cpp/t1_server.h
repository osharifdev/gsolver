#pragma once
/**
 * Stage 2b (certify/STAGE2B.md): serve evaluations of the estimating equation's factors at the chart of a loaded state.
 *
 * `certify_stereo --t1-export EXPORT --t1-dir DIR [--true-jacobians]` loads EXPORT (a belief-chart export: the stop state
 * x_c, state 0) into the built graph and answers requests written to DIR:
 *   t1_z.npy (n_dof, x_c chart), [t1_rad.npy (n_dof coordinate radii)], t1_request = "<id> eval|enclose"
 *   eval    (double): t1_F_<type>.npy (n_f x d), t1_dF_<type>.npy (n_f x d x d), t1_flags_<type>.npy (n_f)
 *   enclose (intervals): F over the point z  -> t1_Flo_/t1_Fhi_<type>.npy,
 *                        F' over the box z +- rad -> t1_dFlo_/t1_dFhi_<type>.npy, flags over the box
 * then t1_done = "<id>". Types in graph order: "gp", "stereo", "prior" (Stage 2c's anchor) (the export's f_<type> order);
 * t1_types.txt lists the types present. t1_quit ends it.
 * The prior (Stage 2c) is served in eval mode only (no enclosures; interval work is Stage 2d).
 * Flags: 1 division by an interval containing 0, 2 a series outside its domain, 4 a stereo factor whose inlier conditions
 * (z > 0, disparity > 0, u - disparity >= 0) are not guaranteed.
 */

#include "t1_ad.h"

namespace certify {

  struct T1Server {
    FactorGraph& g;
    std::string dir;
    std::vector<VarRec> vars;
    std::unordered_map<std::string, int> index;
    struct SF {
      t1::StereoConst k;
      Eigen::Matrix3d Lam;
      int vp, vl;
    };
    struct GF {
      t1::GPConst k;
      Eigen::Matrix<double, 12, 12> Lam;
      int vi, vj;
    };
    struct PF {
      t1::PriorConst k;
      Eigen::Matrix<double, 6, 6> Lam;
      int vp;
    };
    std::vector<SF> stereo;
    std::vector<GF> gp;
    std::vector<PF> prior;

    T1Server(FactorGraph& graph, const std::string& d) : g(graph), dir(d) {
      makeDirs(dir);
      vars = orderVariables(g);
      for (size_t i = 0; i < vars.size(); ++i)
        index[vars[i].id] = static_cast<int>(i);
    }

    // constants at x_c = the variables' linearization points (loadFrozenState of a belief-chart export: the beliefs)
    void setup() {
      stereo.clear();
      gp.clear();
      prior.clear();
      for (auto& f : g.factor_nodes_)
        withFactor(f, [&](auto fp, const std::string& type) {
          using FT = typename std::decay_t<decltype(*fp)>;
          if constexpr (std::is_same_v<FT, SE3PoseVelStereoPoint>) {
            SF s;
            s.vp       = index.at(fp->edges_.template at<0>()->variable_id_);
            s.vl       = index.at(fp->edges_.template at<1>()->variable_id_);
            auto pose  = std::dynamic_pointer_cast<SE3PoseVel>(vars[s.vp].node)->mu_lin_;
            s.k.Rc     = pose.pose.linear();
            s.k.tc     = pose.pose.translation();
            s.k.vc     = pose.vel;
            s.k.pc     = std::dynamic_pointer_cast<SE3Point>(vars[s.vl].node)->mu_lin_;
            s.k.meas   = fp->z_;
            s.k.P      = fp->P_;
            s.k.baseline = fp->baseline_;
            s.Lam      = fp->Sigma_.inverse();
            stereo.push_back(s);
          } else if constexpr (std::is_same_v<FT, SE3PoseVelGP>) {
            GF q;
            q.vi   = index.at(fp->edges_.template at<0>()->variable_id_);
            q.vj   = index.at(fp->edges_.template at<1>()->variable_id_);
            auto a = std::dynamic_pointer_cast<SE3PoseVel>(vars[q.vi].node)->mu_lin_;
            auto b = std::dynamic_pointer_cast<SE3PoseVel>(vars[q.vj].node)->mu_lin_;
            q.k.Ri = a.pose.linear();
            q.k.ti = a.pose.translation();
            q.k.vi = a.vel;
            q.k.Rj = b.pose.linear();
            q.k.tj = b.pose.translation();
            q.k.vj = b.vel;
            q.k.dt = fp->dt_;
            q.Lam  = fp->Sigma_.inverse();
            gp.push_back(q);
          } else if constexpr (std::is_same_v<FT, SE3PoseVelPrior>) {
            PF q;
            q.vp   = index.at(fp->edges_.template at<0>()->variable_id_);
            auto a = std::dynamic_pointer_cast<SE3PoseVel>(vars[q.vp].node)->mu_lin_;
            q.k.Rc = a.pose.linear();
            q.k.tc = a.pose.translation();
            q.k.vc = a.vel;
            q.k.Rz = fp->z_.linear();
            q.k.tz = fp->z_.translation();
            q.Lam  = fp->Sigma_.inverse();
            prior.push_back(q);
          } else {
            throw std::runtime_error("t1: factor type " + type + " is not supported");
          }
        });
    }

    template <class T>
    void gather(const std::vector<T>& zall, int v, T* out) {
      const int off = vars[v].offset, d = vars[v].dim;
      for (int k = 0; k < d; ++k) out[k] = zall[off + k];
    }

    template <class T>
    void runPrior(const std::vector<T>& zall, std::vector<T>& Fp, std::vector<T>& dFp, std::vector<int64_t>& flp) {
      Fp.resize(prior.size() * 12);
      dFp.resize(prior.size() * 144);
      flp.assign(prior.size(), 0);
      for (size_t k = 0; k < prior.size(); ++k) {
        T z[12];
        gather(zall, prior[k].vp, z);
        flp[k] = t1::priorF<T>(prior[k].k, prior[k].Lam, z, &Fp[12 * k], &dFp[144 * k]);
      }
    }

    template <class T>
    void run(const std::vector<T>& zall, std::vector<T>& Fs, std::vector<T>& dFs, std::vector<int64_t>& fls, std::vector<T>& Fg,
             std::vector<T>& dFg, std::vector<int64_t>& flg) {
      Fs.resize(stereo.size() * 15);
      dFs.resize(stereo.size() * 225);
      fls.assign(stereo.size(), 0);
      Fg.resize(gp.size() * 24);
      dFg.resize(gp.size() * 576);
      flg.assign(gp.size(), 0);
#pragma omp parallel for schedule(dynamic, 64)
      for (size_t k = 0; k < stereo.size(); ++k) {
        T z[15];
        gather(zall, stereo[k].vp, z);
        gather(zall, stereo[k].vl, z + 12);
        fls[k] = t1::stereoF<T>(stereo[k].k, stereo[k].Lam, z, &Fs[15 * k], &dFs[225 * k]);
      }
#pragma omp parallel for schedule(dynamic, 1)
      for (size_t k = 0; k < gp.size(); ++k) {
        T z[24];
        gather(zall, gp[k].vi, z);
        gather(zall, gp[k].vj, z + 12);
        flg[k] = t1::gpF<T>(gp[k].k, gp[k].Lam, z, &Fg[24 * k], &dFg[576 * k]);
      }
    }

    static void split(const std::vector<t1::Iv>& a, std::vector<double>& lo, std::vector<double>& hi) {
      lo.resize(a.size());
      hi.resize(a.size());
      for (size_t i = 0; i < a.size(); ++i) {
        lo[i] = a[i].lo;
        hi[i] = a[i].hi;
      }
    }

    void process(int id, const std::string& mode) {
      const NpyArray Z = readNpy(dir + "/t1_z.npy");
      const size_t ns = stereo.size(), ng = gp.size();
      if (mode == "eval") {
        std::vector<double> Fs, dFs, Fg, dFg;
        std::vector<int64_t> fls, flg;
        run<double>(Z.f, Fs, dFs, fls, Fg, dFg, flg);
        writeNpy(dir + "/t1_F_stereo.npy", Fs, {ns, 15});
        writeNpy(dir + "/t1_dF_stereo.npy", dFs, {ns, 15, 15});
        writeNpy(dir + "/t1_flags_stereo.npy", fls, {ns});
        writeNpy(dir + "/t1_F_gp.npy", Fg, {ng, 24});
        writeNpy(dir + "/t1_dF_gp.npy", dFg, {ng, 24, 24});
        writeNpy(dir + "/t1_flags_gp.npy", flg, {ng});
        if (!prior.empty()) {
          std::vector<double> Fp, dFp;
          std::vector<int64_t> flp;
          runPrior<double>(Z.f, Fp, dFp, flp);
          writeNpy(dir + "/t1_F_prior.npy", Fp, {prior.size(), 12});
          writeNpy(dir + "/t1_dF_prior.npy", dFp, {prior.size(), 12, 12});
          writeNpy(dir + "/t1_flags_prior.npy", flp, {prior.size()});
        }
      } else if (mode == "enclose") {
        const NpyArray Rd = readNpy(dir + "/t1_rad.npy");
        std::vector<t1::Iv> zpt(Z.f.size()), zbox(Z.f.size());
        for (size_t i = 0; i < Z.f.size(); ++i) {
          zpt[i]  = {Z.f[i], Z.f[i]};
          zbox[i] = {t1::dn(Z.f[i] - Rd.f[i]), t1::upr(Z.f[i] + Rd.f[i])};
        }
        std::vector<t1::Iv> Fs, dFs, Fg, dFg, Fs2, dFs2, Fg2, dFg2;
        std::vector<int64_t> fls, flg, fls2, flg2;
        run<t1::Iv>(zpt, Fs, dFs, fls, Fg, dFg, flg);       // F (and F') at the point z
        run<t1::Iv>(zbox, Fs2, dFs2, fls2, Fg2, dFg2, flg2); // F' over the box
        std::vector<double> lo, hi;
        split(Fs, lo, hi);
        writeNpy(dir + "/t1_Flo_stereo.npy", lo, {ns, 15});
        writeNpy(dir + "/t1_Fhi_stereo.npy", hi, {ns, 15});
        split(dFs2, lo, hi);
        writeNpy(dir + "/t1_dFlo_stereo.npy", lo, {ns, 15, 15});
        writeNpy(dir + "/t1_dFhi_stereo.npy", hi, {ns, 15, 15});
        for (size_t k = 0; k < ns; ++k) fls2[k] |= fls[k];
        writeNpy(dir + "/t1_flags_stereo.npy", fls2, {ns});
        split(Fg, lo, hi);
        writeNpy(dir + "/t1_Flo_gp.npy", lo, {ng, 24});
        writeNpy(dir + "/t1_Fhi_gp.npy", hi, {ng, 24});
        split(dFg2, lo, hi);
        writeNpy(dir + "/t1_dFlo_gp.npy", lo, {ng, 24, 24});
        writeNpy(dir + "/t1_dFhi_gp.npy", hi, {ng, 24, 24});
        for (size_t k = 0; k < ng; ++k) flg2[k] |= flg[k];
        writeNpy(dir + "/t1_flags_gp.npy", flg2, {ng});
      } else {
        throw std::runtime_error("t1: unknown mode " + mode);
      }
      std::ofstream(dir + "/t1_done.tmp") << id << "\n";
      std::rename((dir + "/t1_done.tmp").c_str(), (dir + "/t1_done").c_str());
    }

    void serve() {
      int last = 0;
      while (true) {
        {
          std::ifstream f(dir + "/t1_request");
          int id = -1;
          std::string mode;
          if (f && (f >> id >> mode) && id > last) {
            f.close();
            process(id, mode);
            last = id;
            continue;
          }
        }
        struct stat st;
        if (stat((dir + "/t1_quit").c_str(), &st) == 0)
          return;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
  };

} // namespace certify
