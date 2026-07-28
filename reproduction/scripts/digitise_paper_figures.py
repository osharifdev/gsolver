#!/usr/bin/env python3
"""Digitise data points from the published PDF's vector figures.

ALL outputs of this script are labelled: "digitised approximation from
published figure". They are approximations reconstructed from the vector
drawing commands of the arXiv PDF (marker centroids, polyline vertices, bar
and box geometry), NOT original raw measurements.

Calibration methods per figure are documented inline. Known limitations:
- Fig. 8/9 y-axes: tick-label bboxes carry a small vertical offset; each
  panel is therefore zero-anchored at the Ours curve's sigma=1e-4 level
  (visually at the axis floor in the paper). Estimated uncertainty ~0.1 m.
- Values clipped by the axis frame are flagged clipped=1 (lower bounds).
"""
import csv
import math
import os
from collections import defaultdict

import fitz

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The arXiv PDF is NOT committed (see reproduction/.gitignore). Download
# arXiv:2606.06250v1 into docs/ to re-run digitisation.
PDF = os.environ.get("GSOLVER_PAPER_PDF", f"{ROOT}/docs/2606.06250v1 GBP CPU.pdf")
OUT = f"{ROOT}/reproduction/results/paper_digitised"
os.makedirs(OUT, exist_ok=True)

NOTE = "digitised approximation from published figure"

COLORS = {
    "gsolver": (0.12156862765550613, 0.46666666865348816, 0.7058823704719543),
    "bspline": (1.0, 0.4980390071868896, 0.054901998490095139),
    "zspline": (0.17254900932312012, 0.62745100259780884, 0.17254900932312012),
    "black": (0.0, 0.0, 0.0),
    "red": (0.83921599388122559, 0.15294100344181061, 0.15686300396919251),
}


def close(a, b, tol=0.03):
    return a is not None and b is not None and all(abs(x - y) < tol for x, y in zip(a, b))


def drawings_in(page, clip, slack=2.0):
    for d in page.get_drawings():
        r = d["rect"]
        if r.x0 >= clip[0] - slack and r.y0 >= clip[1] - slack and r.x1 <= clip[2] + slack and r.y1 <= clip[3] + slack:
            yield d


def markers(page, clip, color, max_size=6.0):
    out = []
    for d in drawings_in(page, clip):
        if d["type"] in ("f", "fs") and close(d.get("fill"), color):
            r = d["rect"]
            if r.width <= max_size and r.height <= max_size:
                out.append(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))
    return sorted(out)


def polyline_vertices(page, clip, color, x_window=True):
    """Vertices of stroked polylines whose bbox INTERSECTS the clip.
    matplotlib clip-masks curves at the axes but the PDF path keeps the true
    data coordinates, so out-of-axes values are recoverable."""
    pts = []
    cx0, cy0, cx1, cy1 = clip
    for d in page.get_drawings():
        r = d["rect"]
        if r.x1 < cx0 or r.x0 > cx1 or r.y1 < cy0 or r.y0 > cy1:
            continue
        if d["type"] == "s" and close(d.get("color"), color):
            for it in d["items"]:
                if it[0] == "l":
                    for p in (it[1], it[2]):
                        if not x_window or (cx0 - 2 <= p.x <= cx1 + 2):
                            pts.append((p.x, p.y))
    dedup = []
    for p in sorted(pts):
        if not dedup or abs(p[0] - dedup[-1][0]) > 0.4 or abs(p[1] - dedup[-1][1]) > 0.4:
            dedup.append(p)
    return dedup


def write_csv(name, header, rows):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as f:
        f.write(f"# {NOTE}\n")
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote", path, f"({len(rows)} rows)")


doc = fitz.open(PDF)

# ---------------------------------------------------------------- Figure 5 --
# x gridlines 353.2->0it, 541.8->25it; y gridlines 410.6->E=0, 282.8->E=500
page = doc[3]
def fig5():
    def x2it(xp): return (xp - 353.2) / (541.8 - 353.2) * 25.0
    def y2E(yp): return (410.6 - yp) / (410.6 - 282.8) * 500.0
    clip = (344, 260, 552, 412)  # y stops at the axis floor: excludes tick marks
    rows = []
    for method, ck in (("gsolver", "gsolver"), ("gn_gp", "black")):
        for (xp, yp) in markers(page, clip, COLORS[ck]):
            it = x2it(xp)
            E = y2E(yp)
            if -0.4 < it < 25.4 and E > -2:
                rows.append((round(it), method, round(E, 2)))
    write_csv("fig05_digitised.csv", ["iteration", "method", "total_energy"], sorted(set(rows)))
fig5()

# ------------------------------------------------------------- Figures 8/9 --
# 16 panels; x: 5 grey decade gridlines (1e-4..1e0) + sigma=1.5 at
# +log10(1.5)*decade; y: 13.0 pt per metre (from tick-label spacing),
# zero anchored at the Ours sigma=1e-4 level (documented approximation).
SIGMAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1.5]
def fig89(pageidx, name, row_y):
    page = doc[pageidx]
    col_x = [(74.2, 120.6), (129.9, 176.3), (185.5, 231.9), (241.2, 287.6)]
    etas = [1e-4, 1e-3, 1e-2, 1e-1]
    rows = []
    for (traj, (fy0, fy1)) in row_y:
        for ci, (fx0, fx1) in enumerate(col_x):
            clip = (fx0 - 1, fy0 - 1, fx1 + 1, fy1 + 1)
            grid = sorted({round(d["rect"].x0, 1) for d in drawings_in(page, clip)
                           if d["type"] == "s" and d.get("color") and abs(d["color"][0] - 0.69) < 0.02
                           and d["rect"].width < 0.5})
            if len(grid) != 5:
                print(f"warn {name} {traj} col{ci}: {len(grid)} gridlines")
                continue
            dec = (grid[-1] - grid[0]) / 4.0
            xpos = grid + [grid[-1] + dec * math.log10(1.5)]
            per_m = 13.0
            curves = {}
            for m in ("gsolver", "bspline", "zspline"):
                vs = polyline_vertices(page, clip, COLORS[m])
                vals = []
                for sx in xpos:
                    near = [v for v in vs if abs(v[0] - sx) < dec * 0.25]
                    vals.append(min(near, key=lambda v: abs(v[0] - sx))[1] if near else None)
                curves[m] = vals
            if curves["gsolver"][0] is None:
                print(f"warn {name} {traj} col{ci}: no gsolver anchor")
                continue
            y_zero = curves["gsolver"][0]  # ours @1e-4 ~ 0 (visual axis floor)
            for m, vals in curves.items():
                for si, yv in enumerate(vals):
                    if yv is None:
                        continue
                    # rendered_off_axis: true value recovered from unclipped
                    # path data even though the paper renders it clipped
                    clipped = 1 if yv <= fy0 + 0.3 else 0
                    rows.append((traj, etas[ci], SIGMAS[si], m, round((y_zero - yv) / per_m, 3), clipped))
    write_csv(f"{name}_digitised.csv",
              ["trajectory", "eta_ig", "sigma", "method", "ATE_m", "rendered_off_axis"], rows)

fig89(5, "fig08", [("helix", (66.7, 113.4)), ("sphere", (122.5, 169.2))])
fig89(5, "fig09", [("helix", (265.0, 311.7)), ("sphere", (320.8, 367.5))])

# ---------------------------------------------------------------- Figure 10 --
# dual axis: ATE green (left, 1.00@c103.35, 1.25@c82.15),
# ARE red (right, 0.04@c101.55, 0.05@c68.25); x log 10^0..10^2 at 71.5pt/dec
def fig10():
    page = doc[5]
    clip = (335, 55, 556, 118)
    x0, dec = None, 71.5
    # x tick centers from '100','101','102' labels
    labs = [(w[0] + (w[2] - w[0]) / 2, w[4]) for w in page.get_text("words")
            if 330 < w[0] < 560 and 114 < w[1] < 126 and w[4] in ("100", "101", "102")]
    labs.sort()
    x0 = labs[0][0]
    dec = (labs[2][0] - labs[0][0]) / 2.0
    def x2s(xp): return 10 ** ((xp - x0) / dec)
    def y2ate(yp): return 1.00 + (103.35 - yp) / (103.35 - 82.15) * 0.25
    def y2are(yp): return 0.04 + (101.55 - yp) / (101.55 - 68.25) * 0.01
    rows = []
    for m, conv, ck in (("ATE_m", y2ate, "zspline"), ("ARE_rad", y2are, "red")):
        # green in this figure = ATE curve colour (tab:green), red = ARE
        for (xp, yp) in polyline_vertices(page, clip, COLORS[ck]):
            s = x2s(xp)
            if 0.3 < s < 130:
                rows.append((round(s, 3), m, round(conv(yp), 4)))
    write_csv("fig10_digitised.csv", ["scale_s", "metric", "value"], rows)
fig10()

# ---------------------------------------------------------------- Figure 11 --
# y: 20@c314.65 .. 140@c220.65 (0.7833 pt/px); x ticks 0..25 below panel
def fig11():
    page = doc[5]
    # find x tick labels under the panel (y ~ 320-332)
    labs = [(w[0] + (w[2] - w[0]) / 2, w[4]) for w in page.get_text("words")
            if 330 < w[0] < 560 and 318 < w[1] < 334 and w[4].isdigit()]
    labs.sort()
    if len(labs) >= 2:
        (xa, va), (xb, vb) = labs[0], labs[-1]
        va, vb = float(va), float(vb)
    def x2it(xp): return va + (xp - xa) / (xb - xa) * (vb - va)
    per_px = (314.65 - 220.65) / 120.0
    def y2px(yp): return 20.0 + (314.65 - yp) / per_px
    clip = (340, 205, 560, 320)
    rows = []
    for m in ("gsolver", "bspline", "zspline"):
        for (xp, yp) in polyline_vertices(page, clip, COLORS[m]):
            it = x2it(xp)
            if -0.5 < it < 30.5:
                clipped = 1 if yp <= 207.5 else 0
                rows.append((m, round(it, 2), round(y2px(yp), 2), clipped))
    write_csv("fig11_digitised.csv",
              ["method", "iteration", "mean_reprojection_error_px", "clipped_at_axis_top"], rows)
fig11()

# ---------------------------------------------------------------- Figure 12 --
# box plots: y 1..5 us via labels (16.7pt/us); boxes = filled rects
def fig12():
    page = doc[6]
    # y calibration from labels '1'..'5' at x~68
    labs = [((w[1] + w[3]) / 2, float(w[4])) for w in page.get_text("words")
            if 66 < w[0] < 72 and w[4] in ("1", "2", "3", "4", "5")]
    labs.sort()
    (ya, va), (yb, vb) = labs[0], labs[-1]
    def y2us(yp): return va + (yp - ya) / (yb - ya) * (vb - va)
    clip = (75, 40, 295, 150)
    groups = {"var_to_fac": (85, 155), "fac_to_var": (150, 215), "interpolation": (215, 292)}
    rows = []
    for method, ck in (("gsolver", "gsolver"), ("hyperion", "bspline")):
        boxes, medians, whisk = [], [], []
        for d in drawings_in(page, clip):
            r = d["rect"]
            if d["type"] in ("f", "fs") and close(d.get("fill"), COLORS[ck]) and r.width > 4:
                boxes.append(r)
            if d["type"] == "s" and close(d.get("color"), (0.0, 0.0, 0.0)) and r.width < 0.6 and r.height > 0.4:
                whisk.append(r)  # vertical whisker lines
            if d["type"] == "s" and close(d.get("color"), (0.0, 0.0, 0.0)) and r.height < 0.6 and 4 < r.width < 20:
                medians.append(r)
        for gname, (gx0, gx1) in groups.items():
            bs = [r for r in boxes if gx0 <= (r.x0 + r.x1) / 2 <= gx1]
            if not bs:
                continue
            b = bs[0]
            med = [r for r in medians if b.x0 - 1 <= r.x0 and r.x1 <= b.x1 + 1 and b.y0 - 1 <= r.y0 <= b.y1 + 1]
            wh = [r for r in whisk if b.x0 - 2 <= (r.x0 + r.x1) / 2 <= b.x1 + 2]
            lo = max((r.y1 for r in wh), default=b.y1)
            hi = min((r.y0 for r in wh), default=b.y0)
            rows.append((gname, method,
                         round(y2us(b.y1), 3), round(y2us(b.y0), 3),
                         round(y2us(med[0].y0), 3) if med else "",
                         round(y2us(lo), 3), round(y2us(hi), 3)))
    write_csv("fig12_digitised.csv",
              ["operation", "method", "q1_us", "q3_us", "median_us", "whisker_low_us", "whisker_high_us"],
              rows)
fig12()

# ---------------------------------------------------------------- Figure 14 --
# grouped bars: left panel avg time (teal) 0..100 s / avg mem (purple) 0..1000;
# right panel ATE (red) 0..1.0 / ARE (green) 0..0.010; y0 at '0' label c~137.55
def fig14():
    page = doc[7]
    def centers(x0lo, x0hi, texts):
        out = {}
        for w in page.get_text("words"):
            if x0lo < w[0] < x0hi and w[4] in texts:
                out[w[4]] = (w[1] + w[3]) / 2
        return out
    rows = []
    # detect bar rects by colour in each panel
    def bars(clip, color):
        out = []
        for d in drawings_in(page, clip, slack=1.0):
            r = d["rect"]
            # bars are 4.0 pt wide and tall; legend patches are 9.8 x 3.4
            if d["type"] in ("f", "fs") and close(d.get("fill"), color, tol=0.12) and r.width < 6 and r.height >= 4.5:
                out.append(r)
        return sorted(out, key=lambda r: r.x0)
    # find the actual bar colors present in left panel
    y0 = 137.2  # bar baseline (bottom of every bar rect = axis zero)
    ytop = 64.4  # top tick label centre (100 s / 1000 MB / 1.0 m / 0.010 rad)
    cams = [1, 2, 4, 8]
    teal, purple = (0.0, 0.502, 0.502), (0.502, 0.0, 0.502)
    red, green = COLORS["red"], COLORS["zspline"]
    series = [
        ("avg_time_s", (70, 55, 160, 140), teal, 100.0),
        ("avg_mem_MB", (70, 55, 160, 140), purple, 1000.0),
        ("ATE_m", (190, 55, 292, 140), red, 1.0),      # legend: red = ATE [m]
        ("ARE_rad", (190, 55, 292, 140), green, 0.010),  # legend: green = ARE [rad]
    ]
    for metric, clip, colr, full in series:
        bs = bars(clip, colr)
        for i, r in enumerate(bs[:4]):
            rows.append((metric, cams[i] if i < 4 else "", round((y0 - r.y0) / (y0 - ytop) * full, 4)))
    write_csv("fig14_digitised.csv", ["metric", "num_cameras", "value"], rows)
fig14()
