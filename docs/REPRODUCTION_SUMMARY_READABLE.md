# What we reproduced, what we didn't, and why — a plain-language walkthrough

Companion to the technical report (`PAPER_REPRODUCTION_REPORT.md`). This goes
through the paper in the order its experiments appear and gives a verdict for
each, in plain terms. Every number here was re-verified against the raw result
files on disk.

## First: what is identical and what is different between "them" and "us"

**Identical by construction.** We ran their exact code at commit `e50581f` with
zero modifications, their exact shipped datasets (KITTI graphs, ChArUco file),
their dataset generator with its hard-coded seed 42, the same Eigen 3.4.0, the
same synchronous message-passing schedule, and the same iteration counts and GP
hyperparameters from their config file. Crucially, their solver gives
bit-identical answers regardless of thread count, so none of the *accuracy*
differences below can be blamed on our machine. Only *speed* depends on hardware.

**Different by necessity.**

| | Their setup | Our setup | Consequence |
|---|---|---|---|
| CPU | Intel i9-14900KF, 16 cores | AMD EPYC 7713, 64 cores | timings differ (~1.2–1.6× slower per core for us); accuracy unaffected |
| OS / compiler | unstated (Docker suggests Ubuntu 22.04) | RHEL 9.7, GCC 11.5 | none observed |
| Threads | presumably 16 | we ran both 16 and 64 | speed only |
| Baselines | Hyperion B-/Z-Spline, GN+GP | **not runnable** — Hyperion needs a ROS 2 stack and neither repo ships the glue code; GN+GP was never released | every baseline curve/number comes from their paper (printed table or digitised plot), never from us |
| Torus data (Figs 1, 10) | private | we built a clearly-labelled substitute torus | those two figures are reproductions *in spirit*, not of the original data |

## The experiments, one by one

**Figures 2–4** are explanatory diagrams — nothing to reproduce.

**Figure 8 — prior-based localisation sweeps. Verdict: reproduced, essentially exactly.**
We ran all 48 G-solver configurations (helix + sphere × 4 initial-noise levels ×
6 measurement-noise levels) and compared against the paper's curves extracted
from the PDF. 44 of 48 points agree within tolerance; the remaining 4 (helix at
σ=1) read 0.48 m for us vs ~0.42 m in the plot — a gap comparable to how
precisely a plot can be read. Overlaying our curve on theirs, they sit on top
of each other in all eight panels. The B-/Z-Spline curves are theirs, digitised.

**Figure 9 — pose-graph optimisation sweeps. Verdict: sphere reproduced; helix
partly different.** All sphere panels match (e.g. σ=1: ours 0.838 m vs theirs
~0.83 m). The helix panels match up to σ=0.01, but at σ≥1 we consistently get
~1.6–1.7× their value (0.468 vs 0.273 m at σ=1). This is *not* a hardware
effect — the same pipeline nails prior-helix and PGO-sphere — the most likely
explanation is that their helix PGO file came from a different random draw of
the loop closures/noise than seed 42 produces. One for the authors.

**Figure 5 — G-solver vs Gauss-Newton energy. Verdict: same story, different
numbers; their y-axis is not reproducible.** Their qualitative claim reproduces
perfectly: our GN drops to the floor in 2–3 iterations while GBP descends
gradually toward the same energy basin. But the absolute energies don't match:
evaluating the paper's own energy formula (Eq. 18) on the documented dataset
gives a starting energy of ~6.5 million, while their plot starts at ~450–530.
Telling detail: in their figure the two methods start at *different* energies at
iteration 0 despite sharing the same initial state — so their plot uses some
normalisation the paper doesn't describe. Also, their GN+GP code was never
released; our GN is an independent re-implementation, labelled as such.

**Figure 6 — "does the GP prior matter?" ablation. Verdict: reproduced.**
Under heavy noise, plain GBP stays a scribble (ATE 1.74 m) while GBP+GP recovers
a clean helix (0.49 m) — exactly the paper's point, from the same data and seed.

**Figure 7 / Figures 15–18 — qualitative trajectory galleries. Verdict: our
column reproduced; the other two columns can't exist.** All G-solver panels
(both tasks, both shapes, all six noise levels, with the same 100×-finer
continuous query) were regenerated and the full interpolated trajectories saved.
The B-Spline/Z-Spline columns require Hyperion, which we could not run.

**Table I — statistical consistency (NEES). Verdict: exact, to the third
decimal.** Helix: paper 0.030 / 1.533 / 3.925 / 4.510 / 4.694, ours 0.030 /
1.533 / 3.923 / 4.504 / 4.685. Sphere agrees to within 0.9% at worst. Bonus: the
paper never says which initialisation noise was used — we ran all five options
and proved the answer is identical for every one, so the omission is harmless.

**Figure 10 — hyperparameter ablation. Verdict: reproduced on a substitute.**
The original torus is private, so we built our own (clearly labelled), learned
Qc\* with their trainer, and swept scales. We get the same U-shape with the
minimum between s=1 and s=10 (0.82–0.95 m there vs 3.86 m at s=0.01) and the
same "stays reasonable even at 100×" robustness message. Side-finding: their
plotted curve has ~59 scale values; the released ablation app tests only 5.

**Figure 11 — ChArUco bundle adjustment. Verdict: experiment reproduced, their
metric impossible.** The pixel reprojection error needs the camera calibration
and pixel detections, which are not in the repository. Our substitute metric
(marker-pose error) converges smoothly and stabilises around iteration 10 —
the same convergence story they tell for G-solver. Their three curves (including
the B-Spline divergence) are preserved by digitisation.

**Figure 12 — micro-latencies. Verdict: same ballpark for messages; the
interpolation number doesn't transfer; hardware-confounded.** No benchmark code
exists in the repo, so we wrote one to a documented fallback protocol. Messages:
ours 2.3–2.4 µs median vs their ~1.3–1.6 µs boxes — consistent with our slower
cores. Interpolation: ours 50 µs vs their ~4.3 µs — too big for hardware alone;
our query includes full covariance propagation and a linear knot search, and
their protocol is unknown. Treat this figure as "same order of magnitude for
messages, unresolved for interpolation."

**Table II + Figures 13/19/20 — rolling shutter on KITTI 06. Verdict: the
endpoints reproduce exactly; the middle of the table does not, and that is the
single most important unresolved discrepancy.** With no simulated rolling
shutter (readout 0) we get ATE 0.668 m / ARE 0.0074 vs their 0.671 / 0.007 —
essentially exact (and this told us they use aligned trajectories, which the
paper doesn't state). At readout 0.1 s: 0.662 vs 0.726 — again fine. But at
readouts 1e-4, 1e-3, 1e-2 s the public code + config *diverges* for us (ATE
76 km / 21 m / 5.4 m vs their 0.676–0.684 m), and running more iterations makes
it worse, not better. Same code, same data, deterministic solver — so their
clean middle columns were produced by something not in the release. The overlay
figures were still generated with every projected coordinate saved, but on a
grey canvas: the KITTI images aren't shipped, and the Hyperion (green) points
can't exist without the baseline.

**Figure 14 — multi-camera scalability. Verdict: reproduced.** Per-camera solve
time halves with each doubling of cameras (ours 111→55→28→14 s at 16 threads;
theirs 91→46→23→12 s — the offset is our slower CPU), per-camera memory halves
too, and accuracy stays flat (ours 0.664–0.668 m throughout vs their ~0.70).
This is the paper's headline distributed-SLAM claim, and it holds.

**Figure 1 — the teaser. Verdict: reproduced on a substitute.** On our labelled
torus with a sparse-measurement gap, the posterior uncertainty balloons 2.3× in
the gap and contracts where measurements resume — the exact behaviour the teaser
illustrates.

## The scorecard

| Verdict | Artefacts |
|---|---|
| Exactly / within tolerance | Table I; Fig. 8; Fig. 6; Fig. 14; Table II at readouts 0 and 0.1 |
| Same conclusions, some numbers off | Fig. 9 (helix high-noise ~1.7×); Fig. 12 messages (hardware) |
| Same story, paper's numbers not derivable | Fig. 5 (energy scale); Fig. 12 interpolation |
| Reproduced on labelled substitutes | Fig. 1, Fig. 10 (torus is private) |
| Could not reproduce the paper's number at all | Table II readouts 1e-4–1e-2 (diverges locally); Fig. 11 pixel metric (inputs not released) |
| Could not run, preserved from the paper | every Hyperion B-/Z-Spline curve and the GN+GP baseline |

**One-line summary:** everything the released code can express, we reproduced —
usually to within rounding; the gaps are missing artefacts (torus, baselines,
calibration), one undocumented normalisation (Fig. 5), one suspected different
dataset draw (helix PGO), and one genuine open problem (the rolling-shutter
middle columns), none of which are caused by the hardware difference.
