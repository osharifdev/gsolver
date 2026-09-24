"""Summarize Phase 0 experiment logs into results/phase0/summary.csv.

For each run: exit status, wall time and peak RSS (/usr/bin/time -v), CPU
utilisation, the graph-build and GBP phase durations parsed from the driver's
timestamped log lines, and ATE/ARE against KITTI ground truth where available.
"""

import csv
import re
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from certify.tools import traj_eval  # noqa: E402

OUT = REPO / "results" / "phase0"
GT = REPO / "examples/data/datasets/stereo_visual/gt/06.tum"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
STAMP = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3})\]")

RUNS = [
    # name, experiment, settings, has KITTI GT
    ("charuco_islam", "ChArUco incremental SLAM", "defaults (64 threads)", False),
    ("charuco_islam_omp16", "ChArUco incremental SLAM", "OMP_NUM_THREADS=16", False),
    ("charuco_ba_noise0", "ChArUco BA", "noise 0.0 (README default)", False),
    ("charuco_ba_noise0.1", "ChArUco BA", "noise 0.1 (paper setting)", False),
    ("stereo_ba_kitti06", "Stereo BA KITTI 06, 1 camera", "kitti_05 config (64 threads)", True),
    ("stereo_ba_kitti06_omp16", "Stereo BA KITTI 06, 1 camera", "kitti_05, OMP_NUM_THREADS=16", True),
    ("multicam8_kitti06", "Multi-camera KITTI 06", "8 cameras (README default)", True),
    ("rs_kitti06_0.1", "Rolling shutter KITTI 06", "readout 0.1 s (README default)", True),
    ("rs_kitti06_0.01", "Rolling shutter KITTI 06", "readout 0.01 s", True),
    ("rs_kitti06_0.001", "Rolling shutter KITTI 06", "readout 0.001 s", True),
    ("rs_kitti06_0.0001", "Rolling shutter KITTI 06", "readout 0.0001 s", True),
]


def parse_time_v(path):
    txt = path.read_text()
    wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", txt).group(1)
    parts = [float(p) for p in wall.split(":")]
    secs = sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
    rss_kb = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", txt).group(1))
    cpu = re.search(r"Percent of CPU this job got: (\d+)%", txt).group(1)
    status = re.search(r"Exit status: (\d+)", txt).group(1)
    return secs, rss_kb / 1024.0, int(cpu), int(status)


def parse_log(path):
    """Return (build_s, gbp_s, printed 'Optimization complete' ms, final graph line)."""
    lines = [ANSI.sub("", l) for l in path.read_text().replace("\r", "\n").splitlines()]
    t = {}
    opt_ms, graph = None, ""
    for l in lines:
        m = STAMP.match(l)
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
        if "[2/4] Building" in l or "[2/5] Initializing" in l or "[2/4] Building factor graph" in l:
            t.setdefault("build0", ts)
        if "[3/4] Running GBP" in l or "[3/5] Running incremental" in l or "[3/4] Running GBP optimization" in l:
            t.setdefault("gbp0", ts)
        mm = re.search(r"Optimization complete in (\d+)ms", l)
        if mm:
            opt_ms = int(mm.group(1))
            t["gbp1"] = ts
        mm = re.search(r"(Final graph|Graph ready): (.*)", l)
        if mm:
            graph = mm.group(2)
    build_s = (t["gbp0"] - t["build0"]).total_seconds() if "gbp0" in t and "build0" in t else float("nan")
    return build_s, opt_ms, graph


rows = []
for name, exp, settings, has_gt in RUNS:
    wall, rss, cpu, status = parse_time_v(OUT / f"{name}.stderr_time")
    build_s, opt_ms, graph = parse_log(OUT / f"{name}.stdout")
    r = dict(run=name, experiment=exp, settings=settings, exit_status=status, wall_s=round(wall, 2),
             peak_rss_MB=round(rss, 1), cpu_percent=cpu, graph_build_s=round(build_s, 2),
             solve_ms_printed=opt_ms, graph=graph)
    if has_gt:
        ev = traj_eval.evaluate(OUT / f"{name}.tum", GT)
        r.update({k: (round(v, 6) if isinstance(v, float) else v) for k, v in ev.items()})
    rows.append(r)

keys = []
for r in rows:
    for k in r:
        if k not in keys:
            keys.append(k)
with open(OUT / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print({k: r[k] for k in r if k not in ("experiment",)})
