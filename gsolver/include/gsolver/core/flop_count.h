#pragma once

#include <array>
#include <cstdint>
#include <mutex>
#include <vector>

namespace gsolver {

  /**
   * Phase D, WP6a: flop counts of the frozen (threshold-mode) GBP operations, certify/FLOPS.md.
   *
   * Every counted site adds the textbook count of the dense operations it executes, by block dimensions:
   * a product (m x k)(k x n) counts 2mkn, an explicit inverse of an n x n matrix 2n^3, and every elementwise
   * operation on a vector or matrix (sum, difference, negation, scaling, adding a regularizer to the diagonal)
   * one flop per entry it touches. Copies and block rearrangements count 0. Counts are kept per thread and summed
   * by collect(); they never change what the solver computes. Only the frozen code paths count (the threshold and
   * frozen schedules); G-solver's shipped per-round schedule, with its Lie-group maps per message, does not.
   */
  namespace flops {

    enum Category : int {
      Linearize    = 0, // a factor's updateErrorAndJacobian at a new linearization point (operations counted, flops not)
      FactorUpdate = 1, // updateFactor(): the linear model's error, information form and conditioned joint
      F2V          = 2, // factor-to-variable message (Schur-complement marginalization, the small inverses)
      Belief       = 3, // updateMuAndSigma(): the belief update
      V2F          = 4, // variable-to-factor messages of one variable
      V2FAlt       = 5, // the same messages formed as belief minus the one incoming message (a baseline; not executed)
      NumCategories
    };
    constexpr int kClasses = 8; // node classes, set by the driver (FactorNodeBase/VariableNodeBase::flop_class_)

    struct Tally {
      std::array<std::array<uint64_t, NumCategories>, kClasses> flops{};
      std::array<std::array<uint64_t, NumCategories>, kClasses> ops{};
      void operator+=(const Tally& o) {
        for (int c = 0; c < kClasses; ++c)
          for (int k = 0; k < NumCategories; ++k) {
            flops[c][k] += o.flops[c][k];
            ops[c][k] += o.ops[c][k];
          }
      }
    };

    // One traced operation (for the hand-count validation): class, category, dimensions and its count
    struct TraceRecord {
      int cls, cat, a, b, deg;
      uint64_t flops;
    };

    inline bool enabled = false;
    inline int trace_per_kind = 0; // > 0: record the first N operations of every (class, category, a, b, deg)
    inline std::mutex registry_mutex;
    inline std::vector<Tally*> registry;
    inline std::vector<TraceRecord> trace;

    struct Local {
      Tally t;
      Local() {
        std::lock_guard<std::mutex> lock(registry_mutex);
        registry.push_back(&t);
      }
    };
    inline Tally& local() {
      thread_local Local l;
      return l.t;
    }

    inline void add(int cls, Category cat, uint64_t f, int a = 0, int b = 0, int deg = 0) {
      if (!enabled)
        return;
      Tally& t = local();
      t.flops[cls][cat] += f;
      t.ops[cls][cat] += 1;
      if (trace_per_kind > 0) {
        std::lock_guard<std::mutex> lock(registry_mutex);
        int seen = 0;
        for (const auto& r : trace)
          seen += (r.cls == cls && r.cat == cat && r.a == a && r.b == b && r.deg == deg);
        if (seen < trace_per_kind)
          trace.push_back({cls, cat, a, b, deg, f});
      }
    }

    // Sum of all threads' tallies since the last collect(); resets them (call outside parallel regions)
    inline Tally collect() {
      std::lock_guard<std::mutex> lock(registry_mutex);
      Tally sum;
      for (Tally* t : registry) {
        sum += *t;
        *t = Tally{};
      }
      return sum;
    }

    // ---- textbook counts
    constexpr uint64_t prod(uint64_t m, uint64_t k, uint64_t n) {
      return 2 * m * k * n;
    }
    constexpr uint64_t inv(uint64_t n) {
      return 2 * n * n * n;
    }

    // updateFactor(), frozen branch; m = error dimension, n = stacked variable dimension, sq = sum of d_k^2 over edges
    constexpr uint64_t factorUpdate(uint64_t m, uint64_t n, uint64_t sq) {
      return prod(m, n, 1) + m                            // error = error_lin + J_lin tau
             + inv(m) + prod(n, m, m) + prod(n, m, 1) + m + n // eta = J' Sigma^-1 (-error) k
             + inv(m) + prod(n, m, m) + prod(n, m, n) + n * n // Lambda = J' Sigma^-1 J k
             + sq;                                         // conditioned joint: + incoming precisions
    }

    // computeMessage(), frozen; a = outgoing variable dimension, b = the rest (b = 0: unary factor)
    constexpr uint64_t f2v(uint64_t a, uint64_t b) {
      const uint64_t marg = (b == 0) ? 0
                                     : b                                              // bb += eps I
                                         + inv(b) + prod(a, b, b) + prod(a, b, 1) + a // eta_M = ea - ab bb^-1 eb
                                         + inv(b) + prod(a, b, b) + prod(a, b, a) + a * a; // Lambda_M = aa - ab bb^-1 ba
      return marg + a                // Lambda_M += eps I
             + inv(a) + prod(a, a, 1) // tau = Lambda_M^-1 eta_M
             + a;                    // tau_out = tau_in + tau
    }

    // updateMuAndSigma(), frozen; d = variable dimension, deg = number of edges
    constexpr uint64_t belief(uint64_t d, uint64_t deg) {
      return deg * (d + prod(d, d, 1) + d + d * d) // per incoming message: tau - z, Lambda * delta, eta +=, Lambda' +=
             + d                                    // + eps I
             + inv(d) + prod(d, d, 1)               // delta = Lambda'^-1 eta
             + d                                    // z += delta
             + inv(d);                              // Sigma = Lambda'^-1
    }

    // VariableNode::sendMessage(i), frozen, one outgoing message; the other deg - 1 incoming messages are summed afresh
    constexpr uint64_t v2f(uint64_t d, uint64_t deg) {
      return (deg - 1) * (d + prod(d, d, 1) + d + d * d) + d + inv(d) + prod(d, d, 1) + d;
    }

    // The usual GBP implementation of one outgoing message (not what G-solver executes): Lambda' = Lambda_b - Lambda_in
    // (d^2), eta' = eta_b - Lambda_in (tau_in - z) (d + 2d^2 + d), + eps I (d), delta = Lambda'^-1 eta' (2d^3 + 2d^2),
    // tau = z + delta (d); the belief itself is already counted in Belief
    constexpr uint64_t v2fAlt(uint64_t d) {
      return d * d + (d + 2 * d * d + d) + d + inv(d) + prod(d, d, 1) + d;
    }

  } // namespace flops

} // namespace gsolver
