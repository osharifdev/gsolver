#pragma once
/**
 * Stage 2b (certify/STAGE2B.md): the estimating equation's factors, differentiated by forward-mode automatic
 * differentiation, in double precision or in outward-rounded interval arithmetic.
 *
 * F(z) = sum_f J_f(x)' Lambda_f e_f(x) at x = x_c (+) z, with J_f the derivative of e_f along the local perturbation at x
 * (the corrected solver's Jacobian), is evaluated through nested duals: the outer level carries d/dz, the inner level
 * d/d(delta) at delta = 0 (the local perturbation). From e as a nested dual one reads e, J = de/d(delta), de/dz and
 * d2e/dz d(delta), hence F_f and its exact derivative F'_f = d/dz (J' Lambda e).
 *
 * Intervals (Iv): every operation is computed in round-to-nearest and its bounds are then moved outward by one ulp
 * (std::nextafter), which contains the exact result of +, -, *, / on the endpoints. The only non-rational maps are the
 * SO(3) exponential and logarithm; they are written as power series in s = |r|^2 (resp. q = |w|^2) whose truncation is
 * enclosed by a rigorous remainder bound on the value and its first two derivatives (geometric tail bounds), so no libm
 * function is needed. A map evaluated outside its series domain, a division by an interval containing 0, or a stereo
 * factor whose inlier conditions are not guaranteed on the box sets a flag instead of a result.
 */

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace t1 {

  // ------------------------------------------------------------------ intervals
  struct Iv {
    double lo = 0.0, hi = 0.0;
  };
  inline double dn(double x) { return std::nextafter(x, -std::numeric_limits<double>::infinity()); }
  inline double upr(double x) { return std::nextafter(x, std::numeric_limits<double>::infinity()); }
  inline Iv operator+(const Iv& a, const Iv& b) { return {dn(a.lo + b.lo), upr(a.hi + b.hi)}; }
  inline Iv operator-(const Iv& a, const Iv& b) { return {dn(a.lo - b.hi), upr(a.hi - b.lo)}; }
  inline Iv operator-(const Iv& a) { return {-a.hi, -a.lo}; }
  inline Iv operator*(const Iv& a, const Iv& b) {
    const double p1 = a.lo * b.lo, p2 = a.lo * b.hi, p3 = a.hi * b.lo, p4 = a.hi * b.hi;
    return {dn(std::min(std::min(p1, p2), std::min(p3, p4))), upr(std::max(std::max(p1, p2), std::max(p3, p4)))};
  }
  thread_local inline int flags = 0; // bit 1: division by an interval containing 0; bit 2: series domain; bit 4: inlier
  inline Iv operator/(const Iv& a, const Iv& b) {
    if (b.lo <= 0.0 && b.hi >= 0.0) {
      flags |= 1;
      const double inf = std::numeric_limits<double>::infinity();
      return {-inf, inf};
    }
    const double p1 = a.lo / b.lo, p2 = a.lo / b.hi, p3 = a.hi / b.lo, p4 = a.hi / b.hi;
    return {dn(std::min(std::min(p1, p2), std::min(p3, p4))), upr(std::max(std::max(p1, p2), std::max(p3, p4)))};
  }
  inline Iv& operator+=(Iv& a, const Iv& b) { return a = a + b; }

  // scalar traits: constants, sup of |value|, remainder terms
  template <class T> struct Traits;
  template <> struct Traits<double> {
    static double constant(double c) { return c; }
    static double supabs(const double& x) { return std::abs(x); }
    static double supval(const double& x) { return x; }
    static double infval(const double& x) { return x; }
  };
  template <> struct Traits<Iv> {
    static Iv constant(double c) { return {c, c}; }
    static double supabs(const Iv& x) { return std::max(std::abs(x.lo), std::abs(x.hi)); }
    static double supval(const Iv& x) { return x.hi; }
    static double infval(const Iv& x) { return x.lo; }
  };

  // ------------------------------------------------------------------ forward-mode duals
  template <class T, int N>
  struct D {
    T v;
    std::array<T, N> d;
  };
  template <class T, int N> struct Traits<D<T, N>> {
    static D<T, N> constant(double c) {
      D<T, N> r;
      r.v = Traits<T>::constant(c);
      r.d.fill(Traits<T>::constant(0.0));
      return r;
    }
    static double supabs(const D<T, N>& x) { return Traits<T>::supabs(x.v); }
    static double supval(const D<T, N>& x) { return Traits<T>::supval(x.v); }
    static double infval(const D<T, N>& x) { return Traits<T>::infval(x.v); }
  };
  template <class T, int N> inline D<T, N> operator+(const D<T, N>& a, const D<T, N>& b) {
    D<T, N> r;
    r.v = a.v + b.v;
    for (int i = 0; i < N; ++i) r.d[i] = a.d[i] + b.d[i];
    return r;
  }
  template <class T, int N> inline D<T, N> operator-(const D<T, N>& a, const D<T, N>& b) {
    D<T, N> r;
    r.v = a.v - b.v;
    for (int i = 0; i < N; ++i) r.d[i] = a.d[i] - b.d[i];
    return r;
  }
  template <class T, int N> inline D<T, N> operator-(const D<T, N>& a) {
    D<T, N> r;
    r.v = -a.v;
    for (int i = 0; i < N; ++i) r.d[i] = -a.d[i];
    return r;
  }
  template <class T, int N> inline D<T, N> operator*(const D<T, N>& a, const D<T, N>& b) {
    D<T, N> r;
    r.v = a.v * b.v;
    for (int i = 0; i < N; ++i) r.d[i] = a.d[i] * b.v + a.v * b.d[i];
    return r;
  }
  template <class T, int N> inline D<T, N> operator/(const D<T, N>& a, const D<T, N>& b) {
    D<T, N> r;
    const T inv = Traits<T>::constant(1.0) / b.v;
    r.v = a.v * inv;
    for (int i = 0; i < N; ++i) r.d[i] = (a.d[i] - r.v * b.d[i]) * inv;
    return r;
  }
  template <class T, int N> inline D<T, N>& operator+=(D<T, N>& a, const D<T, N>& b) { return a = a + b; }
  // with double constants
  template <class S> inline S cst(double c) { return Traits<S>::constant(c); }
  template <class T, int N> inline D<T, N> operator*(double c, const D<T, N>& a) { return cst<D<T, N>>(c) * a; }
  template <class T, int N> inline D<T, N> operator+(const D<T, N>& a, double c) { return a + cst<D<T, N>>(c); }
  inline Iv operator*(double c, const Iv& a) { return Iv{c, c} * a; }
  inline Iv operator+(const Iv& a, double c) { return a + Iv{c, c}; }

  // ------------------------------------------------------------------ series with rigorous remainders
  // An unknown function R with |R| <= t0, |R'| <= t1, |R''| <= t2 on the domain, evaluated at s (as the scalar type).
  template <class S> struct Rem;
  template <> struct Rem<double> {
    static double at(const double&, double, double, double) { return 0.0; } // floating mode: the remainder is < 1e-17
  };
  template <> struct Rem<Iv> {
    static Iv at(const Iv&, double t0, double, double) { return {-t0, t0}; }
  };
  template <int N> struct Rem<D<double, N>> {
    static D<double, N> at(const D<double, N>&, double, double, double) { return cst<D<double, N>>(0.0); }
  };
  template <int N, int M> struct Rem<D<D<double, N>, M>> {
    static D<D<double, N>, M> at(const D<D<double, N>, M>&, double, double, double) { return cst<D<D<double, N>, M>>(0.0); }
  };
  template <int N> struct Rem<D<Iv, N>> {
    static D<Iv, N> at(const D<Iv, N>& s, double t0, double t1, double) {
      D<Iv, N> r;
      r.v = Iv{-t0, t0};
      for (int i = 0; i < N; ++i) r.d[i] = Iv{-t1, t1} * s.d[i];
      return r;
    }
  };
  template <int N, int M> struct Rem<D<D<Iv, N>, M>> {
    static D<D<Iv, N>, M> at(const D<D<Iv, N>, M>& s, double t0, double t1, double t2) {
      D<D<Iv, N>, M> r;
      r.v.v = Iv{-t0, t0};
      for (int j = 0; j < N; ++j) r.v.d[j] = Iv{-t1, t1} * s.v.d[j];
      for (int i = 0; i < M; ++i) {
        r.d[i].v = Iv{-t1, t1} * s.d[i].v;
        for (int j = 0; j < N; ++j) r.d[i].d[j] = Iv{-t1, t1} * s.d[i].d[j] + Iv{-t2, t2} * s.d[i].v * s.v.d[j];
      }
      return r;
    }
  };

  // sum_{k<K} c_k s^k (Horner) plus the enclosed tail; tail bound for the m-th derivative: first omitted term of the
  // m-th derivative series at s_max divided by (1 - q_m), q_m = s_max * ratio_max * (K+1)/(K+1-m)
  template <class S>
  inline S series(const S& s, const double* c, int K, double cK_abs, double ratio_max, double s_domain) {
    // the series converge on |s| <= smax (an enclosure of s = |r|^2 may extend below 0: the bounds use |s|)
    const double smax = Traits<S>::supabs(s);
    if (smax > s_domain)
      flags |= 2;
    S y = cst<S>(c[K - 1]);
    for (int k = K - 2; k >= 0; --k)
      y = y * s + cst<S>(c[k]);
    double t[3];
    for (int m = 0; m < 3; ++m) {
      const double q = smax * ratio_max * (K + 1.0) / (K + 1.0 - m);
      double first = cK_abs;
      for (int i = 0; i < m; ++i) first *= (K - i);
      first *= std::pow(std::max(smax, 0.0), K - m);
      // (1 + 1e-12) covers the rounding of this bound (pow, products); + 1e-11 covers the rounding of the stored
      // coefficients c_k (relative error <= (k + 2) 2^-53 each, sum |c_k| k^m smax^(k-m) <= 1e3 on the domains used)
      t[m] = (q < 1.0) ? upr(upr(first) / dn(1.0 - q)) * (1.0 + 1e-12) + 1e-11 : std::numeric_limits<double>::infinity();
      if (!(q < 1.0))
        flags |= 2;
    }
    return y + Rem<S>::at(s, t[0], t[1], t[2]);
  }

  struct Coeffs {
    static constexpr int KA = 12, KL = 40;
    double A[KA], B[KA], G[KL];
    double A_K, B_K, G_K;
    Coeffs() {
      // A(s) = sin(t)/t, B(s) = (1 - cos t)/t^2, s = t^2; G(q) = asin(sqrt q)/sqrt q
      double f = 1.0; // (2k+1)!
      for (int k = 0; k <= KA; ++k) {
        if (k > 0) f *= (2.0 * k) * (2.0 * k + 1.0);
        const double a = ((k % 2) ? -1.0 : 1.0) / f;
        const double b = ((k % 2) ? -1.0 : 1.0) / (f * (2.0 * k + 2.0));
        if (k < KA) { A[k] = a; B[k] = b; } else { A_K = std::abs(a); B_K = std::abs(b); }
      }
      double c = 1.0; // (2k)!/(4^k (k!)^2)
      for (int k = 0; k <= KL; ++k) {
        if (k > 0) c *= (2.0 * k - 1.0) * (2.0 * k) / (4.0 * k * k);
        const double g = c / (2.0 * k + 1.0);
        if (k < KL) G[k] = g; else G_K = g;
      }
    }
  };
  inline const Coeffs& coeffs() {
    static const Coeffs c;
    return c;
  }
  constexpr double kRodDomain = 4.0;  // |r| <= 2 rad
  constexpr double kLogDomain = 0.25; // |w|^2 <= 1/4 (sin theta <= 1/2)
  template <class S> inline S seriesA(const S& s) {
    const auto& c = coeffs();
    const int K = Coeffs::KA;
    return series(s, c.A, K, c.A_K * 1.0000001, 1.0 / ((2.0 * K + 2) * (2.0 * K + 3)), kRodDomain);
  }
  template <class S> inline S seriesB(const S& s) {
    const auto& c = coeffs();
    const int K = Coeffs::KA;
    return series(s, c.B, K, c.B_K * 1.0000001, 1.0 / ((2.0 * K + 3) * (2.0 * K + 4)), kRodDomain);
  }
  template <class S> inline S seriesG(const S& q) {
    const auto& c = coeffs();
    return series(q, c.G, Coeffs::KL, c.G_K * 1.0000001, 1.0, kLogDomain);
  }

  // ------------------------------------------------------------------ small linear algebra
  template <class S> using V3 = std::array<S, 3>;
  template <class S> using M3 = std::array<S, 9>; // row major
  template <class S> inline M3<S> mconst(const Eigen::Matrix3d& R) {
    M3<S> m;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) m[3 * i + j] = cst<S>(R(i, j));
    return m;
  }
  template <class S> inline M3<S> mmul(const M3<S>& A, const M3<S>& B) {
    M3<S> C;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        C[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j];
    return C;
  }
  template <class S> inline M3<S> mtmul(const M3<S>& A, const M3<S>& B) { // A' B
    M3<S> C;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        C[3 * i + j] = A[i] * B[j] + A[3 + i] * B[3 + j] + A[6 + i] * B[6 + j];
    return C;
  }
  template <class S> inline V3<S> mv(const M3<S>& A, const V3<S>& x) {
    return {A[0] * x[0] + A[1] * x[1] + A[2] * x[2], A[3] * x[0] + A[4] * x[1] + A[5] * x[2], A[6] * x[0] + A[7] * x[1] + A[8] * x[2]};
  }
  template <class S> inline V3<S> mtv(const M3<S>& A, const V3<S>& x) { // A' x
    return {A[0] * x[0] + A[3] * x[1] + A[6] * x[2], A[1] * x[0] + A[4] * x[1] + A[7] * x[2], A[2] * x[0] + A[5] * x[1] + A[8] * x[2]};
  }
  template <class S> inline V3<S> vadd(const V3<S>& a, const V3<S>& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
  template <class S> inline V3<S> vsub(const V3<S>& a, const V3<S>& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
  // Rodrigues' formula R = I + A(s)[r]x + B(s)[r]x^2, s = |r|^2
  template <class S> inline M3<S> rod(const V3<S>& r) {
    const S s = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    const S a = seriesA(s), b = seriesB(s);
    const S one = cst<S>(1.0), zero = cst<S>(0.0);
    M3<S> W = {zero, -r[2], r[1], r[2], zero, -r[0], -r[1], r[0], zero};
    M3<S> W2 = mmul(W, W);
    M3<S> R;
    for (int k = 0; k < 9; ++k) R[k] = a * W[k] + b * W2[k];
    R[0] = R[0] + one;
    R[4] = R[4] + one;
    R[8] = R[8] + one;
    return R;
  }
  // I + [d]x: the local perturbation's rotation to first order (exact for derivatives at d = 0)
  template <class S> inline M3<S> rotFirstOrder(const V3<S>& d) {
    const S one = cst<S>(1.0), zero = cst<S>(0.0);
    return {one, -d[2], d[1], d[2], one, -d[0], -d[1], d[0], one};
  }
  // principal logarithm for rotations with cos(theta) > 0: phi = G(|w|^2) w, w = vee(R - R')/2
  template <class S> inline V3<S> logR(const M3<S>& R) {
    const S half = cst<S>(0.5);
    V3<S> w = {half * (R[7] - R[5]), half * (R[2] - R[6]), half * (R[3] - R[1])};
    const S q = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    const S tr = R[0] + R[4] + R[8];
    if (Traits<S>::infval(tr) <= 1.0) // theta < pi/2 not guaranteed
      flags |= 2;
    const S g = seriesG(q);
    return {g * w[0], g * w[1], g * w[2]};
  }

  // ------------------------------------------------------------------ pose charts at x_c
  // x_c (+) z (+) delta for SE3PoseVel: R = Rc Rod(z_r) (I + [delta_r]x), t = tc + Rc z_t + Rc Rod(z_r) delta_t,
  // v = vc + z_v + delta_v (the decoupled chart T Exp(tau), Exp(t, r) = (Rod(r), t), of geometry3d.h)
  template <class S> struct PoseAt {
    M3<S> R;
    V3<S> t;
    std::array<S, 6> v;
  };
  template <class S>
  inline PoseAt<S> poseAt(const Eigen::Matrix3d& Rc, const Eigen::Vector3d& tc, const Eigen::Matrix<double, 6, 1>& vc,
                          const S* z, const S* dl) { // z, dl: 12 each (t, r, v)
    PoseAt<S> P;
    const M3<S> Rz = mmul(mconst<S>(Rc), rod<S>({z[3], z[4], z[5]}));
    P.R = mmul(Rz, rotFirstOrder<S>({dl[3], dl[4], dl[5]}));
    const V3<S> tcS = {cst<S>(tc(0)), cst<S>(tc(1)), cst<S>(tc(2))};
    P.t = vadd(vadd(tcS, mv(mconst<S>(Rc), V3<S>{z[0], z[1], z[2]})), mv(Rz, V3<S>{dl[0], dl[1], dl[2]}));
    for (int k = 0; k < 6; ++k) P.v[k] = cst<S>(vc(k)) + z[6 + k] + dl[6 + k];
    return P;
  }

  // stereo error e = z_meas - h(p_cam), p_cam = R'(p - t), as se3_posevel_stereo_point.cpp: [xl; y; z; xr] = P_ext [p_cam; 1]
  // with P_ext = [P; P row 0 with -P(0,0) b in the last column], h = (xl/z, y/z, (xl - xr)/z)
  struct StereoConst {
    Eigen::Matrix3d Rc;
    Eigen::Vector3d tc, pc, meas;
    Eigen::Matrix<double, 6, 1> vc;
    Eigen::Matrix<double, 3, 4> P;
    double baseline;
  };
  template <class S>
  inline std::array<S, 3> stereoError(const StereoConst& k, const S* z, const S* dl, int& inlier_flags) {
    const PoseAt<S> P = poseAt<S>(k.Rc, k.tc, k.vc, z, dl);
    const V3<S> p = {cst<S>(k.pc(0)) + z[12] + dl[12], cst<S>(k.pc(1)) + z[13] + dl[13], cst<S>(k.pc(2)) + z[14] + dl[14]};
    const V3<S> pc = mtv(P.R, vsub(p, P.t));
    auto row = [&](int r, double last) {
      return cst<S>(k.P(r, 0)) * pc[0] + cst<S>(k.P(r, 1)) * pc[1] + cst<S>(k.P(r, 2)) * pc[2] + cst<S>(last);
    };
    const S xl = row(0, k.P(0, 3)), y = row(1, k.P(1, 3)), zc = row(2, k.P(2, 3)), xr = row(0, -k.P(0, 0) * k.baseline);
    const S invz = cst<S>(1.0) / zc;
    // xl - xr = P(0,3) + P(0,0) b is a constant: the disparity and the right-image column are written with a single
    // occurrence of the variables (identical values, tighter interval enclosures)
    const S u = xl * invz, v = y * invz, d = cst<S>(k.P(0, 3) + k.P(0, 0) * k.baseline) * invz, ur = xr * invz;
    // G-solver's inlier test: z > 0, disparity > 0 and u - disparity (the right-image column) >= 0; flag if not guaranteed
    if (!(Traits<S>::infval(zc) > 0.0) || !(Traits<S>::infval(d) > 0.0) || !(Traits<S>::infval(ur) >= 0.0))
      inlier_flags |= 4;
    return {cst<S>(k.meas(0)) - u, cst<S>(k.meas(1)) - v, cst<S>(k.meas(2)) - d};
  }

  // GP error e = [Log(Exp(dt v_i)^-1 T_i^-1 T_j); Adj(R_i' R_j) v_j - v_i], decoupled Exp/Log
  struct GPConst {
    Eigen::Matrix3d Ri, Rj;
    Eigen::Vector3d ti, tj;
    Eigen::Matrix<double, 6, 1> vi, vj;
    double dt;
  };
  template <class S>
  inline std::array<S, 12> gpError(const GPConst& k, const S* z, const S* dl) { // z, dl: 24 (pose i, then pose j)
    const PoseAt<S> A = poseAt<S>(k.Ri, k.ti, k.vi, z, dl);
    const PoseAt<S> B = poseAt<S>(k.Rj, k.tj, k.vj, z + 12, dl + 12);
    const M3<S> RM = mtmul(A.R, B.R);
    const V3<S> tM = mtv(A.R, vsub(B.t, A.t));
    const S dt = cst<S>(k.dt);
    const M3<S> Rv = rod<S>({dt * A.v[3], dt * A.v[4], dt * A.v[5]});
    const V3<S> p = mtv(Rv, vsub(tM, V3<S>{dt * A.v[0], dt * A.v[1], dt * A.v[2]}));
    const V3<S> phi = logR(mtmul(Rv, RM));
    const V3<S> nuj = mv(RM, V3<S>{B.v[0], B.v[1], B.v[2]});
    const V3<S> omj = mv(RM, V3<S>{B.v[3], B.v[4], B.v[5]});
    return {p[0], p[1], p[2], phi[0], phi[1], phi[2], nuj[0] - A.v[0], nuj[1] - A.v[1], nuj[2] - A.v[2], omj[0] - A.v[3],
            omj[1] - A.v[4], omj[2] - A.v[5]};
  }

  // Pose prior (Stage 2c's anchor, se3_posevel_prior.cpp): e = Log(Z^-1 T) = [Rz'(t - tz); Log(Rz' R)], decoupled Log
  struct PriorConst {
    Eigen::Matrix3d Rc, Rz;
    Eigen::Vector3d tc, tz;
    Eigen::Matrix<double, 6, 1> vc;
  };
  template <class S>
  inline std::array<S, 6> priorError(const PriorConst& k, const S* z, const S* dl) { // z, dl: 12 (the pose)
    const PoseAt<S> P = poseAt<S>(k.Rc, k.tc, k.vc, z, dl);
    const M3<S> Rz = mconst<S>(k.Rz);
    const V3<S> tz = {cst<S>(k.tz(0)), cst<S>(k.tz(1)), cst<S>(k.tz(2))};
    const V3<S> et = mtv(Rz, vsub(P.t, tz));
    const V3<S> er = logR(mtmul(Rz, P.R));
    return {et[0], et[1], et[2], er[0], er[1], er[2]};
  }

  // F_f = J' Lambda e and F'_f = d/dz (J' Lambda e) from e as a nested dual (outer z, inner delta); only the active
  // coordinates `act` of the factor's stacked variables carry derivatives
  template <class T, int NA, int M>
  inline void assemble(const std::array<D<D<T, NA>, NA>, M>& e, const Eigen::Matrix<double, M, M>& Lam, const std::array<int, NA>& act,
                       int n, T* F, T* dF) { // F: n, dF: n x n row major (row: F component, col: z coordinate)
    for (int k = 0; k < n; ++k) F[k] = cst<T>(0.0);
    for (int k = 0; k < n * n; ++k) dF[k] = cst<T>(0.0);
    std::array<T, M> Le; // Lambda e
    for (int a = 0; a < M; ++a) {
      T s = cst<T>(0.0);
      for (int b = 0; b < M; ++b) s += cst<T>(Lam(a, b)) * e[b].v.v;
      Le[a] = s;
    }
    for (int j = 0; j < NA; ++j) {
      T s = cst<T>(0.0);
      for (int a = 0; a < M; ++a) s += e[a].v.d[j] * Le[a];
      F[act[j]] = s;
    }
    for (int i = 0; i < NA; ++i) {
      std::array<T, M> Ldi; // Lambda de/dz_i
      for (int a = 0; a < M; ++a) {
        T s = cst<T>(0.0);
        for (int b = 0; b < M; ++b) s += cst<T>(Lam(a, b)) * e[b].d[i].v;
        Ldi[a] = s;
      }
      for (int j = 0; j < NA; ++j) {
        T s = cst<T>(0.0);
        for (int a = 0; a < M; ++a) s += e[a].d[i].d[j] * Le[a] + e[a].v.d[j] * Ldi[a];
        dF[act[j] * n + act[i]] = s;
      }
    }
  }

  // Evaluate one factor over z in [zlo, zhi] (a box; zlo == zhi for a point) in scalar type T (double or Iv)
  template <class T, int NA>
  inline void seedVars(const T* zval, const std::array<int, NA>& act, int n, std::vector<D<D<T, NA>, NA>>& z,
                       std::vector<D<D<T, NA>, NA>>& dl) {
    using S = D<D<T, NA>, NA>;
    z.assign(n, cst<S>(0.0));
    dl.assign(n, cst<S>(0.0));
    for (int k = 0; k < n; ++k) z[k].v.v = zval[k];
    for (int a = 0; a < NA; ++a) {
      z[act[a]].d[a].v = cst<T>(1.0);
      dl[act[a]].v.d[a] = cst<T>(1.0);
    }
  }

  template <class T>
  inline int stereoF(const StereoConst& k, const Eigen::Matrix3d& Lam, const T* zval, T* F, T* dF) {
    constexpr int NA = 9;
    const std::array<int, NA> act = {0, 1, 2, 3, 4, 5, 12, 13, 14};
    using S = D<D<T, NA>, NA>;
    std::vector<S> z, dl;
    seedVars<T, NA>(zval, act, 15, z, dl);
    flags = 0;
    int inl = 0;
    const std::array<S, 3> e = stereoError<S>(k, z.data(), dl.data(), inl);
    assemble<T, NA, 3>(e, Lam, act, 15, F, dF);
    return flags | inl;
  }
  template <class T>
  inline int gpF(const GPConst& k, const Eigen::Matrix<double, 12, 12>& Lam, const T* zval, T* F, T* dF) {
    constexpr int NA = 24;
    std::array<int, NA> act;
    for (int i = 0; i < NA; ++i) act[i] = i;
    using S = D<D<T, NA>, NA>;
    std::vector<S> z, dl;
    seedVars<T, NA>(zval, act, 24, z, dl);
    flags = 0;
    const std::array<S, 12> e = gpError<S>(k, z.data(), dl.data());
    assemble<T, NA, 12>(e, Lam, act, 24, F, dF);
    return flags;
  }

  template <class T>
  inline int priorF(const PriorConst& k, const Eigen::Matrix<double, 6, 6>& Lam, const T* zval, T* F, T* dF) {
    constexpr int NA = 6;
    const std::array<int, NA> act = {0, 1, 2, 3, 4, 5};
    using S = D<D<T, NA>, NA>;
    std::vector<S> z, dl;
    seedVars<T, NA>(zval, act, 12, z, dl);
    flags = 0;
    const std::array<S, 6> e = priorError<S>(k, z.data(), dl.data());
    assemble<T, NA, 6>(e, Lam, act, 12, F, dF);
    return flags;
  }

} // namespace t1
