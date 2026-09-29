"""Stage 2c, Step 3 (BRIEF_stage2c.md §3): budgets, equal-guarantee and leave-one-out tables from the Step 2 logs.

All from results/stage2c/<dataset>/<variant>/ (stage1_analyse.py --stage stage2c); no further runs.

3.1 budgets.csv   B in {10, 50, 100, 200, 300}, each segment and variant (and the 161-keyframe prefix, Step 5), at the
                  evaluation after round B:
                  unfinished_current       fraction of poses with ||e_i||_{D_i} > eta S_lo,i (the oracle error)
                  unfinished_relinearized  fraction with the remaining Gauss-Newton move at the beliefs (solver's Jacobian,
                                           D_i = S_ii; column move_S) > eta S_lo,i
                  median and p95 remaining move (WP1's definition, column move), aligned keyframe ATE
3.2 equal_guarantee.csv (corrected variant)
                  rounds_needed   first evaluation round at which every pose passes the oracle test ||e_i||_{D_i} <= eta S_lo,i
                                  (resolution 10 rounds; column passed_oracle_lo)
                  common_budget   the largest rounds_needed over the four segments
                  per segment: the certified stop round; its total flops (GBP to that round plus the certificate at every
                  evaluation round up to it); the common budget's total flops (GBP only); the ratio; the same with the
                  belief-minus-incoming GBP baseline
3.3 leave_one_out.csv (corrected variant)
                  held_out_budget the largest rounds_needed over the other three segments; whether it finishes the held-out
                  segment (fraction unfinished at that round, both senses); the certified stop's total flops against the
                  held-out budget's (GBP only); the ratio
A quantity that needs a round the 300-round runs never reached is left empty, with the reason in `note`.
"""

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results/stage2c"
DS = ["kitti06", "kitti03", "kitti04", "kitti10"]
BUDGETS = (10, 50, 100, 200, 300)
ETA = 0.1


def rd(p):
    return list(csv.DictReader(open(p)))


class Run:
    def __init__(self, ds, var):
        d = R / ds / var
        self.rows = rd(d / "eval_rounds.csv")
        self.summ = {int(r["round"]): r for r in rd(d / "eval_summary.csv")}
        fl = rd(d / "flops.csv")
        self.gbp = np.array([float(r["gbp_flops"]) for r in fl])
        self.gbp_alt = np.array([float(r["gbp_flops_alt"]) for r in fl])
        self.cert = {k: float(v["certificate_flops"] or 0) for k, v in self.summ.items()}
        self.stops = {r["criterion"]: r for r in rd(d / "stops.csv")}
        self.by = {}
        for r in self.rows:
            self.by.setdefault(int(r["round"]), []).append(r)
        self.rounds_run = len(fl)

    def col(self, rnd, k):
        return np.array([float(r[k]) if r.get(k) not in (None, "") else np.nan for r in self.by[rnd]])

    def certifiable(self, rnd):
        return not any(r.get("decision") in ("A0_FAILS", "NONFINITE") for r in self.by[rnd])

    def unfinished(self, rnd):
        slo = self.col(rnd, "S_lo")
        return float(np.mean(self.col(rnd, "err") > ETA * slo)), float(np.mean(self.col(rnd, "move_S") > ETA * slo))

    def rounds_needed(self):
        return next((t for t in sorted(self.by) if all(int(r["passed_oracle_lo"]) for r in self.by[t])), None)

    def certified(self):
        c = self.stops["certified"]
        return int(c["round"]) if c["round"] else None

    def cert_total(self, t, alt=False):
        g = self.gbp_alt if alt else self.gbp
        return float(g[:t].sum()) + sum(v for k, v in self.cert.items() if k <= t)

    def gbp_to(self, t, alt=False):
        return float((self.gbp_alt if alt else self.gbp)[:t].sum()) if t is not None and t <= self.rounds_run else None


def ratio(a, b):
    return a / b if a is not None and b is not None else None


def main():
    runs = {(ds, v): Run(ds, v) for ds in DS + ["prefix"] for v in ("corrected", "shipped") if (R / ds / v / "stops.csv").exists()}
    # 3.1
    brows = []
    for (ds, v), run in runs.items():
        for b in BUDGETS:
            if b not in run.by:
                brows.append(dict(dataset=ds, variant=v, budget=b, note="no evaluation at this round"))
                continue
            if not run.certifiable(b):
                brows.append(dict(dataset=ds, variant=v, budget=b, poses=len(run.by[b]), unfinished_current=1.0,
                                  note="state non-finite (the run diverged): every pose counted unfinished; nothing measurable"))
                continue
            uc, ur = run.unfinished(b)
            mv = run.col(b, "move")
            brows.append(dict(dataset=ds, variant=v, budget=b, poses=len(run.by[b]), unfinished_current=uc, unfinished_relinearized=ur,
                              move_median=float(np.median(mv)), move_p95=float(np.percentile(mv, 95)),
                              move_true_median=float(np.median(run.col(b, "move_true"))),
                              ate_aligned=float(run.summ[b]["ate_aligned"] or "nan"), certified_pass=float(np.mean(run.col(b, "E") <= ETA * run.col(b, "S_lo")))))
    with open(R / "budgets.csv", "w", newline="") as fh:
        k = ["dataset", "variant", "budget", "poses", "unfinished_current", "unfinished_relinearized", "certified_pass", "move_median",
             "move_p95", "move_true_median", "ate_aligned", "note"]
        w = csv.DictWriter(fh, fieldnames=k, extrasaction="ignore")
        w.writeheader()
        w.writerows(brows)

    # 3.2
    cor = {ds: runs[(ds, "corrected")] for ds in DS if (ds, "corrected") in runs}
    need = {ds: r.rounds_needed() for ds, r in cor.items()}
    common = max(need.values()) if all(v is not None for v in need.values()) and len(need) == 4 else None
    erows = []
    for ds, run in cor.items():
        tc = run.certified()
        e = dict(dataset=ds, rounds_needed=need[ds], common_budget=common, certified_stop=tc,
                 certified_total_flops=run.cert_total(tc) if tc else None, certified_total_flops_alt=run.cert_total(tc, True) if tc else None,
                 budget_gbp_flops=run.gbp_to(common), budget_gbp_flops_alt=run.gbp_to(common, True))
        e["ratio"] = ratio(e["certified_total_flops"], e["budget_gbp_flops"])
        e["ratio_alt"] = ratio(e["certified_total_flops_alt"], e["budget_gbp_flops_alt"])
        notes = []
        if need[ds] is None:
            notes.append("oracle test not passed by every pose by round %d" % run.rounds_run)
        if common is None:
            notes.append("common budget undefined: a segment's rounds needed exceed the %d-round runs" % run.rounds_run)
        if tc is None:
            notes.append("no certified stop by round %d" % run.rounds_run)
        e["note"] = "; ".join(notes)
        erows.append(e)
    with open(R / "equal_guarantee.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(erows[0].keys()))
        w.writeheader()
        w.writerows(erows)

    # 3.3
    lrows = []
    for ds, run in cor.items():
        others = [need[o] for o in cor if o != ds]
        hb = max(others) if all(v is not None for v in others) and len(others) == 3 else None
        tc = run.certified()
        l = dict(dataset=ds, held_out_budget=hb, rounds_needed=need[ds], certified_stop=tc)
        if hb is not None and hb in run.by:
            uc, ur = run.unfinished(hb)
            l.update(unfinished_current_at_budget=uc, unfinished_relinearized_at_budget=ur, budget_finishes=bool(uc == 0.0))
        l.update(certified_total_flops=run.cert_total(tc) if tc else None, budget_gbp_flops=run.gbp_to(hb),
                 certified_total_flops_alt=run.cert_total(tc, True) if tc else None, budget_gbp_flops_alt=run.gbp_to(hb, True))
        l["ratio"] = ratio(l["certified_total_flops"], l["budget_gbp_flops"])
        l["ratio_alt"] = ratio(l["certified_total_flops_alt"], l["budget_gbp_flops_alt"])
        notes = []
        if hb is None:
            notes.append("held-out budget undefined: another segment's rounds needed exceed the %d-round runs" % run.rounds_run)
        if tc is None:
            notes.append("no certified stop by round %d" % run.rounds_run)
        l["note"] = "; ".join(notes)
        lrows.append(l)
    keys = ["dataset", "held_out_budget", "rounds_needed", "certified_stop", "unfinished_current_at_budget",
            "unfinished_relinearized_at_budget", "budget_finishes", "certified_total_flops", "budget_gbp_flops", "ratio",
            "certified_total_flops_alt", "budget_gbp_flops_alt", "ratio_alt", "note"]
    with open(R / "leave_one_out.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(lrows)
    print(json.dumps(dict(rounds_needed=need, common_budget=common), indent=1))
    for r in erows:
        print(r)
    for r in lrows:
        print(r)


if __name__ == "__main__":
    main()
