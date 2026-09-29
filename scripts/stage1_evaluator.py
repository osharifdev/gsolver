"""Stage 1: the evaluator of periodic certified termination (certify/STAGE1.md).

Waits for <eval_dir>/ready (written by certify_stereo / certify_charuco --eval-dir <eval_dir> --eval-every N after every
N-th round), reads <eval_dir>/model (the linearization the solver iterates on next, frozen charts) and <eval_dir>/beliefs
(the beliefs, belief charts), and logs
  <out>/eval_rounds.csv  per evaluation round and pose: E, L, S_lo, S_exact, ||e||, kappa, pass, patch sizes, growth exit,
                         validity margins, the patch lower bound against the exact marginal (sampled rounds), and the
                         remaining Gauss-Newton move (solver's Jacobian; true Jacobian)
  <out>/eval_summary.csv per evaluation round: fraction passing, all pass, violations, certificate operation counts and
                         flops, accuracy (median and p95 of the remaining GN move) and keyframe ATE, timings
The beliefs of the first evaluation round (>= --stop-from) at which every pose passes are kept in
<out>/certified_stop_beliefs, its linearization in <out>/certified_stop_model and every pose's final and base patch (and
E, L, S_lo) in <out>/certified_stop_patches.json (Stage 2a). With --keep-rounds, the linearization and beliefs of those
evaluation rounds are kept in <out>/model_rNNNNN and <out>/beliefs_rNNNNN (Stage 2c's anchor check). On an exception it
writes <eval_dir>/abort. Exits after the round marked last.

Usage: python3 scripts/stage1_evaluator.py <eval_dir> <out_dir> [--eta 0.1] [--p-L 1e-6] [--gt GT.tum]
                                           [--sampled 10,150,300] [--stop-from R] [--keep-rounds 100]
"""

import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from certify.io import Export  # noqa: E402
from certify.sidecar_model import PoseModelBuilder  # noqa: E402
from certify.stage1 import StatEngine, accuracy  # noqa: E402

POSE_FIELDS = ["round", "pose", "t", "E", "L", "err", "S_lo", "S_exact", "kappa", "passed", "passed_oracle_exact", "passed_oracle_lo",
               "patch", "base_patch", "grown_steps", "grow_why", "valid_upper", "valid_lower", "rel_upper_margin", "rel_lower_margin",
               "loewner_margin", "move", "move_S", "move_true", "move_true_S", "decision"]
SUMMARY_FIELDS = ["round", "poses", "a0_fails", "all_pass", "frac_pass", "frac_pass_oracle_exact", "rho_upper", "violations_upper",
                  "violations_lower", "worst_rel_upper", "worst_rel_lower", "loewner_min", "E_over_err_median", "grow_exits",
                  "factorizations", "factorizations_shared", "factor_flops", "solve_rhs", "solve_flops", "matvec_flops", "sweeps",
                  "sweep_flops", "yardstick_factorizations", "yardstick_flops", "certificate_flops", "move_median", "move_p95",
                  "move_max", "move_true_median", "move_true_p95", "ate_aligned", "ate_matched", "frac_move_gt_eta_S_lo",
                  "load_s", "certify_s", "accuracy_s", "sampled", "move_undefined", "nonfinite"]


def read_info(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        k, v = line.split("\t")
        out[k] = int(v)
    return out


class Evaluator:
    def __init__(self, edir, out, eta, p_L, gt, sampled, stop_from, keep_rounds=()):
        self.keep_rounds = set(keep_rounds)
        self.edir, self.out = Path(edir), Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.engine = StatEngine(eta)
        self.eta, self.p_L, self.gt, self.sampled, self.stop_from = eta, p_L, gt, set(sampled), stop_from
        self.builder = None
        self.stop_kept = False
        self.fr = open(self.out / "eval_rounds.csv", "w", newline="")
        self.fs = open(self.out / "eval_summary.csv", "w", newline="")
        self.wr = csv.DictWriter(self.fr, fieldnames=POSE_FIELDS, extrasaction="ignore")
        self.ws = csv.DictWriter(self.fs, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        self.wr.writeheader()
        self.ws.writeheader()

    def process(self, r):
        t0 = time.perf_counter()
        info = read_info(self.edir / "round_info.tsv")
        assert info["round"] == r, (info, r)
        ex, exB = Export(self.edir / "model"), Export(self.edir / "beliefs")
        if not all(np.isfinite(e.z).all() and np.isfinite(e.pose_T_lin).all() and np.isfinite(e.point_p_lin).all()
                   and all(np.isfinite(g["J"]).all() and np.isfinite(g["e"]).all() for g in e.factors.values()) for e in (ex, exB)):
            # Stage 2c: a diverged run (non-finite state); nothing can be certified or measured, the round is recorded as such
            pb = np.flatnonzero(ex.var_type == "SE3PoseVel")
            pb = pb[np.argsort(ex.timestamp[pb], kind="stable")]
            self.wr.writerows([dict(round=r, pose=str(ex.var_id[b]), t=float(ex.timestamp[b]), decision="NONFINITE", passed=0) for b in pb])
            self.ws.writerow(dict(round=r, poses=len(pb), a0_fails=1, all_pass=0, frac_pass=0.0, nonfinite=1, load_s=time.perf_counter() - t0))
            self.fr.flush()
            self.fs.flush()
            print("round %d: non-finite state (diverged)" % r, flush=True)
            return info["last"] == 1
        if self.builder is None:
            g2o = ex.meta.get("g2o") if self.p_L else None
            if g2o and not os.path.isabs(g2o):
                g2o = str(REPO / g2o)
            self.builder = PoseModelBuilder(p_L=self.p_L, g2o=g2o)
        model = self.builder.build(None, ex=ex)
        modelF = self.builder.build(None, true_jacobian=True, ex=ex)
        t1 = time.perf_counter()
        sampled = r in self.sampled or info["last"] == 1
        per_pose, summ = self.engine.evaluate(r, model, modelF, sampled=sampled)
        t2 = time.perf_counter()
        acc = accuracy(exB, self.builder, self.gt)
        t3 = time.perf_counter()
        pos = {p: k for k, p in enumerate(acc["pose_ids"])}
        for row in per_pose:
            k = pos.get(row["pose"])
            if k is not None:
                row.update(move=float(acc["move"][k]), move_S=float(acc["move_S"][k]), move_true=float(acc["move_true"][k]),
                           move_true_S=float(acc["move_true_S"][k]))
        slo = np.array([row.get("S_lo", np.nan) for row in per_pose], float)
        mvS = np.array([row.get("move_S", np.nan) for row in per_pose], float)
        summ.update(move_undefined=acc.get("move_undefined", 0))
        summ.update(move_median=float(np.median(acc["move"])), move_p95=float(np.percentile(acc["move"], 95)),
                    move_max=float(np.max(acc["move"])), move_true_median=float(np.median(acc["move_true"])),
                    move_true_p95=float(np.percentile(acc["move_true"], 95)), ate_aligned=acc.get("ate_aligned", np.nan),
                    ate_matched=acc.get("matched", ""), frac_move_gt_eta_S_lo=float(np.nanmean(mvS > self.eta * slo)),
                    load_s=t1 - t0, certify_s=t2 - t1, accuracy_s=t3 - t2, sampled=int(sampled))
        self.wr.writerows(per_pose)
        self.ws.writerow(summ)
        self.fr.flush()
        self.fs.flush()
        if r in self.keep_rounds:
            shutil.copytree(self.edir / "model", self.out / ("model_r%05d" % r), dirs_exist_ok=True)
            shutil.copytree(self.edir / "beliefs", self.out / ("beliefs_r%05d" % r), dirs_exist_ok=True)
        if summ.get("all_pass") and r >= self.stop_from and not self.stop_kept:
            shutil.copytree(self.edir / "beliefs", self.out / "certified_stop_beliefs", dirs_exist_ok=True)
            (self.out / "certified_stop_beliefs" / "ROUND").write_text("%d\n" % r)
            # Stage 2a: the linearization and every pose's final and base patch at the certified stop
            shutil.copytree(self.edir / "model", self.out / "certified_stop_model", dirs_exist_ok=True)
            ids = [str(q) for q in model["pose_ids"]]
            last = self.engine.last
            patches = {row["pose"]: dict(P=[ids[k] for k in last["rows"][ids.index(row["pose"])].P],
                                         base_P=list(self.engine.base[row["pose"]]["P"]), E=row["E"], L=row["L"], S_lo=row["S_lo"],
                                         S_exact=row["S_exact"], err=row["err"], kappa=row["kappa"], grow_why=row["grow_why"])
                       for row in per_pose}
            (self.out / "certified_stop_patches.json").write_text(json.dumps(dict(round=r, patches=patches), indent=1))
            self.stop_kept = True
        print("round %d: %d poses, pass %.3f (oracle %.3f), all %d, violations %s/%s, cert %.3g flops, move med %.4g, %.2fs" % (
            r, summ["poses"], summ["frac_pass"], summ.get("frac_pass_oracle_exact", np.nan), summ["all_pass"],
            summ.get("violations_upper"), summ.get("violations_lower"), summ.get("certificate_flops", 0), summ["move_median"],
            t3 - t0), flush=True)
        return info["last"] == 1

    def run(self):
        done = 0
        while True:
            try:
                r = int((self.edir / "ready").read_text().split()[0])
            except (FileNotFoundError, ValueError, IndexError):
                r = -1
            if r <= done:
                if (self.edir / "finished").exists() or (self.edir / "abort").exists():
                    return  # the driver finished without a further evaluation round, or stopped
                time.sleep(0.005)
                continue
            try:
                last = self.process(r)
            except Exception:
                (self.edir / "abort").write_text("evaluator failed after round %d\n" % r)
                raise
            (self.edir / "done.tmp").write_text("%d\n" % r)
            os.replace(self.edir / "done.tmp", self.edir / "done")
            done = r
            if last:
                return


if __name__ == "__main__":
    a = sys.argv[1:]

    def opt(name, default, conv=float):
        return conv(a[a.index(name) + 1]) if name in a else default

    p_L = a[a.index("--p-L") + 1] if "--p-L" in a else "1e-6"
    Evaluator(a[0], a[1], eta=opt("--eta", 0.1), p_L=None if p_L == "none" else float(p_L), gt=opt("--gt", None, str),
              sampled=[int(x) for x in opt("--sampled", "", str).split(",") if x], stop_from=opt("--stop-from", 0, int),
              keep_rounds=[int(x) for x in opt("--keep-rounds", "", str).split(",") if x]).run()
