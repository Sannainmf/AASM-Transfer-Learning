"""
Figure 7: Data Efficiency — Cp and Cf side by side.
% improvement of Transfer over Scratch, one line per architecture.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load(filename):
    with open(os.path.join(PROJECT_DIR, "Results", filename)) as f:
        return json.load(f)


mlp = load("final_results_mlp.json")
cnn = load("final_results_cnn.json")
fno = load("final_results_fno.json")

fracs   = ["10%", "20%", "40%", "60%", "80%", "100%"]
n_samps = [49,    99,    198,   298,   397,   497]


def extract(data, key):
    improv = [data[key][f]["improvement_pct"] for f in fracs]
    sig    = [data[key][f]["significant"]     for f in fracs]
    return np.array(improv), sig


# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "DejaVu Serif"],
    "font.size":            9,
    "axes.labelsize":       10,
    "legend.fontsize":      8.5,
    "xtick.labelsize":      8.5,
    "ytick.labelsize":      8.5,
    "axes.linewidth":       0.7,
    "xtick.major.width":    0.7,
    "ytick.major.width":    0.7,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "figure.dpi":           150,
})

COLORS  = {"MLP": "#0077BB", "CNN": "#EE7733", "FNO": "#009988"}
MARKERS = {"MLP": "o",       "CNN": "s",        "FNO": "^"}
LINES   = {"MLP": "-",       "CNN": "--",        "FNO": "-."}

x = np.array(n_samps)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)

def draw_panel(ax, key, ylabel):
    for arch, data in [("MLP", mlp), ("CNN", cnn), ("FNO", fno)]:
        imp, sig = extract(data, key)
        ax.plot(x, imp,
                color=COLORS[arch], marker=MARKERS[arch], ls=LINES[arch],
                ms=5, lw=1.6, label=arch, zorder=3, clip_on=False)
        for xi, yi, s in zip(x, imp, sig):
            if s:
                ax.plot(xi, yi, marker=MARKERS[arch], ms=6,
                        color=COLORS[arch], mec=COLORS[arch], mfc=COLORS[arch],
                        zorder=4, ls="none")
            else:
                ax.plot(xi, yi, marker=MARKERS[arch], ms=5,
                        color=COLORS[arch], mec=COLORS[arch], mfc="white",
                        zorder=4, ls="none")

    ax.axhline(0, color="0.50", lw=0.7, ls=":", zorder=1)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(fracs)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.grid(True, which="major", color="0.90", lw=0.5, zorder=0)
    ax.set_axisbelow(True)

draw_panel(axes[0], "cp_efficiency", r"RMSE Improvement over Scratch (%)")
draw_panel(axes[1], "cf_efficiency", r"RMSE Improvement over Scratch (%)")

# Panel titles above each subplot
axes[0].set_title(r"(a) $C_p$", fontsize=10, fontweight="bold", pad=4)
axes[1].set_title(r"(b) $C_f$", fontsize=10, fontweight="bold", pad=4)

# Shared x-axis label
for ax in axes:
    ax.set_xlabel("RANS Training Data Fraction")

plt.tight_layout(pad=0.6, w_pad=2.5)
plt.subplots_adjust(bottom=0.22)

# Legend centered below both panels — clearly shared
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3,
           frameon=True, framealpha=1.0, edgecolor="0.80",
           handlelength=2.0, handletextpad=0.5, borderpad=0.6,
           fontsize=8.5, bbox_to_anchor=(0.5, 0.01))

out = os.path.join(OUT_DIR, "fig7_budget_combined.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved {out}")
plt.show()