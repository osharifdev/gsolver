"""Phase C figures (item C-i): E_i and the true error against iteration for the median and the worst block
of every dataset, one panel per block, from results/phaseC/<dataset>/curves.json."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "phaseC"


def main(names):
    figdir = OUT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for name in names:
        p = OUT / name / "curves.json"
        if not p.exists():
            continue
        cv = json.loads(p.read_text())
        summ = json.loads((OUT / name / "summary.json").read_text())
        floor = summ["gate"]["pose_floor"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
        for ax, lab in zip(axes, ("median", "worst")):
            d = cv[lab]
            it = np.array(d["iters"], float)
            ax.loglog(it, d["err"], "k-", lw=2.2, label="true error ||e_i||_D")
            if "err_clamped_w1" in d or any(k.startswith("err_clamped") for k in d):
                for k in d:
                    if k.startswith("err_clamped"):
                        ax.loglog(it, d[k], "k:", lw=1.2, label="hub-clamped error (%s)" % k.split("_")[-1])
            for k, style in (("E_C1", "C0-"), ("E_C2", "C1-")):
                v = np.array(d[k], float)
                if np.isfinite(v).any():
                    ax.loglog(it, v, style, label=k)
            for k in sorted(d):
                if k.startswith("E_C3"):
                    ax.loglog(it, d[k], "--", label=k)
                if k.startswith("E_C4") or k.startswith("E_hub"):
                    ax.loglog(it, d[k], "-.", label=k)
            ax.axhline(floor, color="grey", lw=0.8)
            ax.axhline(10 * floor, color="grey", lw=0.8, ls=":")
            ax.set_title("%s: %s block %s (%s)" % (name, lab, d["block"], d["kind"]), fontsize=9)
            ax.set_xlabel("frozen GBP iteration")
            ax.grid(True, which="both", alpha=0.3)
        axes[0].set_ylabel("D-metric error / certificate")
        axes[1].legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(figdir / ("curves_%s.png" % name), dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:] or [p.name for p in OUT.iterdir() if (p / "curves.json").exists()])
