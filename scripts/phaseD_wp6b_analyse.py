"""Phase D, WP6b: summary of the nonlinear residual probe (certify/NL_PROBE.md).

Subsets of pose-rounds (persisted WP2 rounds, re-certified with the growth rule decided after WP2):
  held  : eligible, eta margin > 0 and delta margin < 0 (the poses the delta condition holds back; recomputed margins)
  held_wp2 : the same selection by the WP2 log's margins (growth rule as run in WP2)
  oracle: the oracle stops them (exact error in place of E)
  eligible, all
For each: the distribution of r = s_NL / (eta S_lo), the fraction with r < 0.1 (criterion N) and r < 1 (the residual
term alone below the statistical threshold: necessary for T1 to release the pose), and for the worst decile the
attribution by factor class and by mechanism. With the GP-corrected yardstick (central-difference GP Jacobians,
scripts/phaseD_wp6b_gp_jacobian.py), r is rescaled per pose by S_exact(sidecar) / S_exact(GP corrected) on the rounds
where it was computed.

Usage: python3 scripts/phaseD_wp6b_analyse.py
"""

import csv
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
P = REPO / "results/phaseD/wp6/probe"


def fl(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", "nan", None) else np.nan


def dist(S, key="r"):
    x = np.array([s[key] for s in S])
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0)
    return dict(n=int(len(x)), median=float(np.median(x)), p90=float(np.percentile(x, 90)), p95=float(np.percentile(x, 95)),
                max=float(x.max()), min=float(x.min()), frac_below_0_1=float(np.mean(x < 0.1)), frac_below_1=float(np.mean(x < 1.0)))


def attribution(S):
    if not S:
        return {}
    S = sorted(S, key=lambda s: -s["r"])
    w = S[:max(1, int(np.ceil(0.1 * len(S))))]
    out = dict(n=len(w), r_range=[float(w[-1]["r"]), float(w[0]["r"])])
    for k in ("s_nl_stereo", "s_nl_gp", "s_nl_mismatch", "s_nl_jacobian", "s_nl_curvature"):
        out[k + "_over_s_nl_median"] = float(np.median([s[k] / s["s_nl"] for s in w]))
    out["frac_stereo_larger_than_gp"] = float(np.mean([s["s_nl_stereo"] > s["s_nl_gp"] for s in w]))
    return out


def main():
    res = {}
    for run in ("prefix", "kf40"):
        rows = []
        for f in sorted(glob.glob(str(P / run / "probe_r*.csv"))):
            rows += list(csv.DictReader(open(f)))
        gpj = P / f"gp_jacobian_{run}.json"
        corr = json.load(open(gpj)) if gpj.exists() else None
        corr_round = int(corr["export"].rsplit("_r", 1)[1]) if corr else None
        recs = []
        for r in rows:
            x = dict(round=int(r["round"]), pose=r["pose"], eligible=r["eligible"] == "1", d=fl(r, "d"), E=fl(r, "E"),
                     S_lo=fl(r, "S_lo"), m_eta=fl(r, "margin_eta"), m_delta=fl(r, "margin_delta"), oracle=r["oracle_stop"] == "1",
                     decision=r["decision"], wp2_m_eta=fl(r, "wp2_margin_eta"), wp2_m_delta=fl(r, "wp2_margin_delta"),
                     s_lin=fl(r, "s_lin"), s_nl=fl(r, "s_nl"), r=fl(r, "s_nl_over_etaSlo"), residue=fl(r, "s_lin_residue"),
                     flips=fl(r, "inlier_flips"), grow=r["grow_why"], wp2_decision=r["wp2_decision"])
            for k in ("s_nl_stereo", "s_nl_gp", "s_nl_mismatch", "s_nl_jacobian", "s_nl_curvature", "s_nl_state",
                      "s_nl_state_mismatch", "s_nl_state_jacobian", "s_nl_state_curvature", "step_elim_max", "step_elim_p99",
                      "elim_step_over_1m"):
                x[k] = fl(r, k)
            x["r_state"] = x["s_nl_state"] / (0.1 * x["S_lo"]) if np.isfinite(x["S_lo"]) and x["S_lo"] > 0 else np.nan
            if corr and x["round"] == corr_round:
                pp = corr["per_pose"][r["pose"]]
                x["r_gpfix"] = x["r"] * pp["S_exact_sidecar"] / pp["S_exact_gp_fd"]
            recs.append(x)
        out = dict(rounds=sorted(set(x["round"] for x in recs)))
        subsets = dict(
            held=[x for x in recs if x["eligible"] and x["m_eta"] > 0 and x["m_delta"] < 0],
            held_wp2=[x for x in recs if np.isfinite(x["wp2_m_eta"]) and x["wp2_m_eta"] > 0 and x["wp2_m_delta"] < 0],
            oracle=[x for x in recs if x["oracle"]],
            certified_stop=[x for x in recs if x["decision"] == "STOP"],
            eligible=[x for x in recs if x["eligible"]],
            all=recs)
        for name, S in subsets.items():
            out[name] = dict(dist=dist(S), attribution_worst_decile=attribution(S), dist_no_step=dist(S, "r_state"))
            if S:
                out[name]["no_step_share_median"] = {k: float(np.nanmedian([x[k] / x["s_nl_state"] if x["s_nl_state"] > 0 else np.nan for x in S] or [np.nan]))
                                                     for k in ("s_nl_state_mismatch", "s_nl_state_jacobian", "s_nl_state_curvature")}
                out[name]["s_nl_over_s_nl_state_median"] = float(np.nanmedian([x["s_nl"] / x["s_nl_state"] if x["s_nl_state"] > 0 else np.nan for x in S]))
                out[name]["landmark_step"] = dict(per_probe_max_median=float(np.nanmedian([x["step_elim_max"] for x in S])),
                                                  per_probe_p99_median=float(np.nanmedian([x["step_elim_p99"] for x in S])),
                                                  max=float(np.nanmax([x["step_elim_max"] for x in S])),
                                                  probes_with_step_over_1m=int(sum(x["elim_step_over_1m"] > 0 for x in S)))
            if corr:
                Sc = [x for x in S if "r_gpfix" in x]
                out[name]["dist_gp_corrected_yardstick"] = dist(Sc, "r_gpfix")
        by_round = {}
        for rnd in out["rounds"]:
            R = [x for x in recs if x["round"] == rnd]
            by_round[rnd] = dict(poses=len(R), eligible=sum(x["eligible"] for x in R),
                                 held=sum(x["eligible"] and x["m_eta"] > 0 and x["m_delta"] < 0 for x in R),
                                 held_wp2=sum(np.isfinite(x["wp2_m_eta"]) and x["wp2_m_eta"] > 0 and x["wp2_m_delta"] < 0 for x in R),
                                 oracle_stop=sum(x["oracle"] for x in R), certified_stop=sum(x["decision"] == "STOP" for x in R),
                                 wp2_certified_stop=sum(x["wp2_decision"] in ("STOP", "SLEEP_KEPT", "SLEEP_KEPT_REFACTORIZED") for x in R),
                                 grow_exits={k: sum(1 for x in R if x["eligible"] and x["grow"] == k) for k in ("stop", "continue", "tight", "cap", "")},
                                 all=dist(R))
        out["by_round"] = by_round
        el = [x for x in recs if x["eligible"] and np.isfinite(x["s_nl"])]
        if len(el) > 2:
            rho = spearmanr([x["d"] for x in el], [x["s_nl"] for x in el])
            out["spearman_d_vs_s_nl_eligible"] = dict(rho=float(rho.correlation), p=float(rho.pvalue), n=len(el))
        out["max_residue_over_s_lin"] = float(np.nanmax([x["residue"] / x["s_lin"] for x in recs if x["s_lin"] > 0]))
        out["inlier_flips_total"] = float(np.nansum([x["flips"] for x in recs]))
        out["mechanism_share_all_median"] = {k: float(np.nanmedian([x[k] / x["s_nl"] for x in recs])) for k in
                                             ("s_nl_mismatch", "s_nl_jacobian", "s_nl_curvature", "s_nl_stereo", "s_nl_gp")}
        worst = max((x for x in subsets["held"]), key=lambda x: x["r"], default=None)
        out["worst_held"] = worst and {k: worst[k] for k in ("round", "pose", "r", "s_nl", "S_lo", "E", "d", "s_nl_stereo", "s_nl_gp",
                                                              "s_nl_mismatch", "s_nl_jacobian", "s_nl_curvature")}
        best_oracle = min((x for x in subsets["oracle"]), key=lambda x: x["r"], default=None)
        out["smallest_r_oracle"] = best_oracle and {k: best_oracle[k] for k in ("round", "pose", "r", "s_nl", "S_lo", "d")}
        res[run] = out
    for f in ("checks_kf40_r100.json", "checks_prefix_r200.json", "gp_jacobian_kf40.json", "gp_jacobian_prefix.json"):
        if (P / f).exists():
            d = json.load(open(P / f))
            res.setdefault("checks", {})[f] = {k: v for k, v in d.items() if k not in ("per_pose", "load_check")}
    (P / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    for run in ("prefix", "kf40"):
        o = res[run]
        print("==", run, o["rounds"])
        for name in ("held", "held_wp2", "oracle", "certified_stop", "eligible", "all"):
            print("  %-15s %s" % (name, json.dumps(o[name]["dist"])))
            if "dist_gp_corrected_yardstick" in o[name]:
                print("  %-15s %s" % ("  gp-corrected", json.dumps(o[name]["dist_gp_corrected_yardstick"])))
        print("  worst decile held:", json.dumps(o["held"]["attribution_worst_decile"]))
        print("  mechanism (all, median share):", json.dumps(o["mechanism_share_all_median"]))
        print("  spearman d vs s_nl:", o.get("spearman_d_vs_s_nl_eligible"), " residue max", o["max_residue_over_s_lin"],
              " flips", o["inlier_flips_total"])
        for rnd, b in o["by_round"].items():
            print("  round", rnd, {k: v for k, v in b.items() if k != "all"})


if __name__ == "__main__":
    main()
