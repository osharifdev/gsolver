"""Phase D, WP6b: the nonlinear residual probe on the persisted WP2 rounds (certify/NL_PROBE.md).

For every persisted export of a WP2 stereo run (exports/phaseD/wp2/<run>/export_rNNNNN: the linearization and state
before that round's message passing), the round is re-certified with the growth rule decided after WP2 (CONTINUE only
on the certified lower bound), fed the solver's own d_i and relinearization flags from the WP2 log, and for every pose
s_i^NL (certify/nonlinear_probe.py) is computed with the factors re-evaluated in C++ (certify_stereo --probe-export).

Writes results/phaseD/wp6/probe/<run>/probe_rNNNNN.csv (per pose) and probe_checks_rNNNNN.json (model and load
checks, timings, certificate flops) for each round, as soon as the round is done.

Usage: python3 scripts/phaseD_wp6b_probe.py <run: prefix|kf40> [--rounds 1,10,200] [--batch 8]
"""

import csv
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from certify.io import Export  # noqa: E402
from certify.nonlinear_probe import FullModel, ProbeClient, Prober  # noqa: E402
from certify.sidecar import Engine  # noqa: E402
from certify.sidecar_model import PoseModelBuilder, _landmark_init  # noqa: E402

DELTA, ETA, P_L = 0.003, 0.1, 1e-6
BIN = REPO / "build-certify/certify/cpp/certify_stereo"
GT = REPO / "examples/data/datasets/stereo_visual/gt/06.tum"

FIELDS = ["run", "round", "pose", "t", "d", "relinearized", "eligible", "E", "L", "err", "S_lo", "S_exact", "kappa", "margin_delta",
          "margin_eta", "decision", "oracle_stop", "patch", "grow_why", "s_lin", "s_nl", "s_nl_jacobian", "s_nl_mismatch", "s_nl_curvature", "s_nl_over_etaSlo", "s_lin_residue",
          "s_nl_state", "s_nl_state_jacobian", "s_nl_state_mismatch", "s_nl_state_curvature", "factors_evaluated", "inlier_flips",
          "targets_pose", "targets_elim", "step_max_phys", "step_elim_median", "step_elim_p99", "step_elim_max", "elim_step_over_1m", "wp2_E", "wp2_margin_delta",
          "wp2_margin_eta", "wp2_decision", "wp2_oracle_stop", "wp2_patch", "wp2_grow_why"]


def wp2_rows(run):
    with gzip.open(REPO / f"results/phaseD/wp2/{run}/rounds.csv.gz", "rt") as fh:
        return list(csv.DictReader(fh))


def main(run, rounds=None, batch=8):
    exdir = REPO / f"exports/phaseD/wp2/{run}"
    out = REPO / f"results/phaseD/wp6/probe/{run}"
    out.mkdir(parents=True, exist_ok=True)
    log = wp2_rows(run)
    exports = sorted(exdir.glob("export_r*"))
    if rounds:
        exports = [e for e in exports if int(e.name.split("_r")[1]) in rounds]
    allrows, fields = [], list(FIELDS)
    checks = {}
    init = None
    for ed in exports:
        r = int(ed.name.split("_r")[1])
        t0 = time.perf_counter()
        ex = Export(ed)
        g2o = str(REPO / ex.meta["g2o"])
        init = init if init is not None else _landmark_init(g2o)
        B = PoseModelBuilder(p_L=P_L, g2o=g2o)
        model = B.build(None, ex=ex)
        modelF = B.build(None, true_jacobian=True, ex=ex)
        W = {x["pose"]: x for x in log if int(x["round"]) == r}
        ids = [str(p) for p in model["pose_ids"]]
        d = {p: float(W[p]["d"]) for p in ids}
        relin = {p: W[p]["relinearized"] == "1" for p in ids}
        eng = Engine(DELTA, ETA, continue_rule="L", yard_all=True)
        t1 = time.perf_counter()
        _, per_pose, summ = eng.round(r, model, modelF, d, relin, {}, sampled=False)
        t2 = time.perf_counter()
        fm = FullModel(ex, p_L=P_L, g2o=g2o, init=init)
        S_full, g_full = fm.reduced()
        S, g = model["S"], eng.last["g"]
        chk = dict(round=r, poses=len(ids), rel_S_full_vs_sidecar=float(np.abs(S_full - S).max() / np.abs(S).max()),
                   rel_g_full_vs_sidecar=float(np.abs(g_full - g).max() / max(np.abs(g).max(), 1e-300)),
                   certify_s=t2 - t1, certificate_flops=float(sum(v for k, v in summ.items() if k.endswith("flops"))),
                   violations_upper=summ["violations_upper"], violations_lower=summ["violations_lower"])
        pdir = f"/dev/shm/gbp_os717_probe_{run}"
        cmd = [str(BIN), "--g2o", g2o, "--gt", str(GT), "--keyframes", str(ex.meta["keyframes"]), "--config", ex.meta["config"],
               "--out", pdir + "/out", "--probe-export", str(ed), "--probe-dir", pdir]
        client = ProbeClient(cmd, pdir)
        chk["load_check"] = client.load_check
        prober = Prober(fm, eng.last, model["pose_ids"])
        res = {}
        t3 = time.perf_counter()
        order = list(range(len(ids)))
        for b0 in range(0, len(order), batch):
            prs = [prober.build(i) for i in order[b0:b0 + batch]]
            outs = client.evaluate([(p["targets"], p["z1"]) for p in prs] + [(p["targets"], p["z0"]) for p in prs])
            for k, p in enumerate(prs):
                Fnl, Flin, Fjac, Ftrue, st = outs[k]
                res[p["i"]] = prober.finish(p, Fnl, Flin, Fjac, Ftrue, st, client.classes)
                res[p["i"]].update(prober.finish_state(p, *outs[len(prs) + k][:4]))
        client.close()
        t4 = time.perf_counter()
        chk.update(probe_s=t4 - t3, classes=client.classes, total_s=t4 - t0)
        checks[r] = chk
        for c in client.classes:
            if "s_nl_" + c not in fields:
                fields.insert(fields.index("s_nl") + 1, "s_nl_" + c)
        for k, row in enumerate(per_pose):
            w = W[row["pose"]]
            x = dict(row)
            x.update(run=run, s_lin=float(eng.last["rows"][k].s), **res[k])
            slo = row["S_lo"]
            x["s_nl_over_etaSlo"] = x["s_nl"] / (ETA * slo) if np.isfinite(slo) and slo > 0 else np.nan
            x.update(wp2_E=w["E"], wp2_margin_delta=w["margin_delta"], wp2_margin_eta=w["margin_eta"], wp2_decision=w["decision"],
                     wp2_oracle_stop=w["oracle_stop"], wp2_patch=w["patch"], wp2_grow_why=w["grow_why"])
            allrows.append(x)
        with open(out / ("probe_r%05d.csv" % r), "w", newline="") as fh:  # one file per round (a checkpoint)
            wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            wr.writeheader()
            wr.writerows([x for x in allrows if x["round"] == r])
        (out / ("probe_checks_r%05d.json" % r)).write_text(json.dumps(chk, indent=1, default=float))
        print(json.dumps({k: v for k, v in chk.items() if k != "load_check"}), flush=True)
        print("  load check:", json.dumps(chk["load_check"]), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    rounds = [int(x) for x in a[a.index("--rounds") + 1].split(",")] if "--rounds" in a else None
    batch = int(a[a.index("--batch") + 1]) if "--batch" in a else 8
    main(a[0], rounds, batch)
