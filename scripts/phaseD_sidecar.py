"""Phase D, WP2: the sidecar process (certify/SIDECAR.md).

Waits for <shm>/ready (written by certify_charuco / certify_stereo --sidecar <shm>), reads the round's export
(<shm>/model, the current linearization and state) and <shm>/round_info.tsv (per pose: displacement d_i, whether the
solver relinearized it this round, whether it is asleep), certifies every pose (certify/sidecar.py), writes the poses
that sleep during this round's message passing to <shm>/decision, and logs:
  <out>/rounds.csv   per round and pose (brief WP2 columns plus kappa, the two STOP margins, patch sizes, ...)
  <out>/summary.csv  per round: stop fractions, wakes, refactorizations, operation counts, timings, violations
On an exception it writes <shm>/abort (the driver then stops). Exits after the round marked last.

Usage: python3 scripts/phaseD_sidecar.py <shm_dir> <out_dir> [--delta 0.003] [--eta 0.1] [--p-L 1e-6] [--continue-rule L|s]
(--continue-rule s reproduces WP2's growth exit; the default L is the rule decided after WP2.)
"""

import csv
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from certify.io import Export  # noqa: E402
from certify.sidecar import Engine  # noqa: E402
from certify.sidecar_model import PoseModelBuilder  # noqa: E402

POSE_FIELDS = ["round", "pose", "t", "d", "relinearized", "eligible", "asleep_before", "E", "L", "err", "S_lo", "S_exact",
               "kappa", "margin_delta", "margin_eta", "decision", "asleep_after", "woke", "woke_reason", "refactorized",
               "patch", "base_patch", "grown_steps", "grow_why", "oracle_stop", "valid_upper", "valid_lower",
               "loewner_margin"]
SUMMARY_FIELDS = ["round", "poses", "a0_fails", "eligible", "asleep_before", "asleep_after", "certified_stop", "oracle_stop",
                  "woke_recheck", "woke_relinearization", "woke_new_factor", "refactorized", "rho_upper", "violations_upper",
                  "violations_lower", "loewner_min", "factorizations", "factorizations_shared", "factor_flops", "solve_rhs", "solve_flops",
                  "matvec_flops", "sweeps", "sweep_flops", "yardstick_factorizations", "yardstick_flops", "load_s",
                  "build_s", "certify_and_oracle_s", "new_factors", "relinearized_factors", "sampled"]


def read_info(path):
    lines = Path(path).read_text().splitlines()
    head = {}
    k = 0
    while not lines[k].startswith("id\t"):
        key, val = lines[k].split("\t")
        head[key] = int(val)
        k += 1
    d, relin, asleep = {}, {}, {}
    for line in lines[k + 1:]:
        if line:
            pid, dv, rl, sl = line.split("\t")
            d[pid], relin[pid], asleep[pid] = float(dv), rl == "1", sl == "1"
    return head, d, relin, asleep


class Sidecar:
    def __init__(self, shm, out, delta, eta, p_L, continue_rule="L"):
        self.shm, self.out = Path(shm), Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.engine = Engine(delta, eta, continue_rule)
        self.p_L = p_L
        self.builder = None
        self.fr = open(self.out / "rounds.csv", "w", newline="")
        self.fs = open(self.out / "summary.csv", "w", newline="")
        self.wr = csv.DictWriter(self.fr, fieldnames=POSE_FIELDS, extrasaction="raise")
        self.ws = csv.DictWriter(self.fs, fieldnames=SUMMARY_FIELDS, extrasaction="raise")
        self.wr.writeheader()
        self.ws.writeheader()

    def process(self, r):
        t0 = time.perf_counter()
        head, d, relin, asleep = read_info(self.shm / "round_info.tsv")
        assert head["round"] == r, (head["round"], r)
        ex = Export(self.shm / "model")
        if self.builder is None:
            g2o = ex.meta.get("g2o") if self.p_L else None
            if g2o and not os.path.isabs(g2o):
                g2o = str(REPO / g2o)
            self.builder = PoseModelBuilder(p_L=self.p_L, g2o=g2o)
        t1 = time.perf_counter()
        model = self.builder.build(None, ex=ex)
        modelF = self.builder.build(None, true_jacobian=True, ex=ex)
        t2 = time.perf_counter()
        sampled = r in (1, 10) or head["last"] == 1
        sleep_ids, per_pose, summ = self.engine.round(r, model, modelF, d, relin, asleep, sampled=sampled)
        t3 = time.perf_counter()
        tmp = self.shm / "decision.tmp"
        tmp.write_text("%d\n%s\n" % (r, "\n".join(sleep_ids)))
        os.replace(tmp, self.shm / "decision")
        summ.update(load_s=t1 - t0, build_s=t2 - t1, certify_and_oracle_s=t3 - t2, new_factors=head["new_factors"],
                    relinearized_factors=head["relinearized_factors"], sampled=int(sampled))
        self.wr.writerows(per_pose)
        self.ws.writerow(summ)
        self.fr.flush()
        self.fs.flush()
        print("round %d: %d poses, eligible %s, asleep %s -> %s, certified stop %.3f, oracle %s, violations %s/%s, refactorized %s, %.2fs" % (
            r, summ["poses"], summ.get("eligible"), summ.get("asleep_before"), summ.get("asleep_after"), summ["certified_stop"],
            summ.get("oracle_stop"), summ.get("violations_upper"), summ.get("violations_lower"), summ.get("refactorized"), t3 - t0), flush=True)
        return head["last"] == 1

    def run(self):
        done = 0
        while True:
            try:
                r = int((self.shm / "ready").read_text().split()[0])
            except (FileNotFoundError, ValueError, IndexError):
                r = -1
            if r <= done:
                time.sleep(0.002)
                continue
            try:
                last = self.process(r)
            except Exception:
                (self.shm / "abort").write_text("sidecar failed at round %d\n" % r)
                raise
            done = r
            if last:
                return


if __name__ == "__main__":
    a = sys.argv[1:]

    def opt(name, default):
        return float(a[a.index(name) + 1]) if name in a else default

    rule = a[a.index("--continue-rule") + 1] if "--continue-rule" in a else "L"
    Sidecar(a[0], a[1], delta=opt("--delta", 0.003), eta=opt("--eta", 0.1), p_L=opt("--p-L", None), continue_rule=rule).run()
