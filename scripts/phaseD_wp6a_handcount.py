"""Phase D, WP6a: hand count of one GBP operation of every kind, against the solver's own counters.

Each operation is listed line by line from the frozen code path (conventions of certify/FLOPS.md: a product
(m x k)(k x n) counts 2mkn, an explicit n x n inverse 2n^3, an elementwise operation one flop per entry, "+ eps I" one
flop per diagonal entry, copies 0). The sums are compared with the counts the instrumented solver logged for the same
kind in one traced round (OUT/flops_trace.csv, --flop-trace-round).

Usage: python3 scripts/phaseD_wp6a_handcount.py <trace csv> <out json>
"""

import csv
import json
import sys

FN = "gsolver/src/graph/factor_node.cpp"
FH = "gsolver/include/gsolver/graph/helpers/factor_node_helper.h"
ET = "gsolver/include/gsolver/graph/helpers/factor_node_edge_tuple_helper.h"
VN = "gsolver/src/graph/variable_node.cpp"


def factor_update(m, n, dims):
    return [
        (FN + ":22", "error_ = error_lin_ + J_lin_ * tau_in: product (m x n)(n)", 2 * m * n),
        (FN + ":22", "  ... + error_lin_", m),
        (FN + ":34", "eta_: Sigma_.inverse()", 2 * m ** 3),
        (FN + ":34", "  J_' * Sigma^-1: (n x m)(m x m)", 2 * n * m * m),
        (FN + ":34", "  (J' Sigma^-1) * (-error_): (n x m)(m)", 2 * n * m),
        (FN + ":34", "  -error_", m),
        (FN + ":34", "  * k", n),
        (FN + ":35", "Lambdaprime_: Sigma_.inverse()", 2 * m ** 3),
        (FN + ":35", "  J_' * Sigma^-1: (n x m)(m x m)", 2 * n * m * m),
        (FN + ":35", "  (J' Sigma^-1) * J_: (n x m)(m x n)", 2 * n * m * n),
        (FN + ":35", "  * k", n * n),
    ] + [(ET + ":70/189", "conditioned joint += incoming precision (%d x %d)" % (d, d), d * d) for d in dims]


def f2v(a, b):
    ops = []
    if b:
        ops += [
            (FH + ":26", "bb += eps I", b),
            (FH + ":29", "eta_marg: bb.inverse()", 2 * b ** 3),
            (FH + ":29", "  ab * bb^-1: (a x b)(b x b)", 2 * a * b * b),
            (FH + ":29", "  (ab bb^-1) * eb: (a x b)(b)", 2 * a * b),
            (FH + ":29", "  ea - ...", a),
            (FH + ":30", "Lambdaprime_marg: bb.inverse()", 2 * b ** 3),
            (FH + ":30", "  ab * bb^-1", 2 * a * b * b),
            (FH + ":30", "  (ab bb^-1) * ba: (a x b)(b x a)", 2 * a * b * a),
            (FH + ":30", "  aa - ...", a * a),
        ]
    return ops + [
        (FH + ":74", "Lambdaprime_M += eps I", a),
        (FH + ":77", "Lambdaprime_M.inverse()", 2 * a ** 3),
        (FH + ":77", "  * eta_M", 2 * a * a),
        (FH + ":85", "tau_ = tau_in + tau", a),
    ]


def incoming(d, k, lines):
    return [
        (VN + ":%d" % lines[0], "%d x (tau_ - z_)" % k, k * d),
        (VN + ":%d" % lines[1], "%d x Lambda_in * delta" % k, k * 2 * d * d),
        (VN + ":%d" % lines[2], "%d x eta +=" % k, k * d),
        (VN + ":%d" % lines[3], "%d x Lambda' +=" % k, k * d * d),
    ]


def belief(d, deg):
    return incoming(d, deg, (178, 186, 188, 189)) + [
        (VN + ":192", "Lambda' += eps I", d),
        (VN + ":194", "Lambda'.inverse()", 2 * d ** 3),
        (VN + ":194", "  * eta", 2 * d * d),
        (VN + ":198", "z_ += delta", d),
        (VN + ":200", "Sigma_ = Lambda'.inverse()", 2 * d ** 3),
    ]


def v2f(d, deg):
    return incoming(d, deg - 1, (45, 53, 55, 56)) + [
        (VN + ":59", "Lambda' += eps I", d),
        (VN + ":61", "Lambda'.inverse()", 2 * d ** 3),
        (VN + ":61", "  * eta", 2 * d * d),
        (VN + ":66", "tau_ = z_ + delta", d),
    ]


# the variable dimensions of each factor class (edge order), for the conditioned joint
FACTOR_DIMS = {"gp": [12, 12], "stereo": [12, 3], "posevel_pose": [12, 6], "prior": [12], "posevel_point": [12, 3]}


def main(trace, out):
    rows = list(csv.DictReader(open(trace)))
    res = []
    for r in rows:
        cat, cls = r["category"], r["class"]
        a, b, deg, logged = int(r["a"]), int(r["b"]), int(r["degree"]), int(r["flops"])
        if cat == "linearize":
            continue
        if cat == "factor_update":
            ops = factor_update(a, b, FACTOR_DIMS[cls])
        elif cat == "f2v":
            ops = f2v(a, b)
        elif cat == "belief":
            ops = belief(a, deg)
        else:
            ops = v2f(a, deg)
        hand = sum(o[2] for o in ops)
        res.append(dict(category=cat, cls=cls, a=a, b=b, degree=deg, logged=logged, hand=hand, match=hand == logged,
                        items=[dict(line=o[0], op=o[1], flops=o[2]) for o in ops]))
    ok = all(x["match"] for x in res)
    json.dump(dict(trace=trace, kinds=len(res), all_match=ok, rows=res), open(out, "w"), indent=1)
    for x in res:
        print("%-14s %-13s a=%-2d b=%-2d deg=%-3d logged %8d hand %8d %s" % (x["category"], x["cls"], x["a"], x["b"], x["degree"],
                                                                       x["logged"], x["hand"], "ok" if x["match"] else "MISMATCH"))
    print("all match:", ok)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1], sys.argv[2]) else 1)
