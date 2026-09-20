"""
Figure 7: Cp Data Efficiency Curve
% improvement of Transfer over Scratch, one line per architecture.
Publication-quality styling.
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


mlp = load("final_results_mlp.json")["cp_efficiency"]
cnn = load("final_results_cnn.json")["cp_efficiency"]
fno = load("final_results_fno.json")["cp_efficiency"]

fracs   = ["10%", "20%", "40%", "60%", "80%", "100%"]
n_samps = [49,    99,    198,   298,   397,   497]


def extract(data):
    improv = [data[f]["improvement_pct"] for f in fracs]
    sig    = [data[f]["significant"]     for f in fracs]
    return np.array(improv), sig


mlp_imp, mlp_sig = extract(mlp)
cnn_imp, cnn_sig = extract(cnn)
fno_imp, fno_sig = extract(fno)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "DejaVu Serif"],
    "font.size":            10,
    "axes.labelsize":       11,
    "legend.fontsize":      9,
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "axes.linewidth":       0.7,
    "xtick.major.width":    0.7,
    "ytick.major.width":    0.7,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "figure.dpi":           150,
})

# Colorblind-safe palette
COLORS  = {"MLP": "#0077BB", "CNN": "#EE7733", "FNO": "#009988"}
MARKERS = {"MLP": "o",       "CNN": "s",        "FNO": "^"}
LINES   = {"MLP": "-",       "CNN": "--",        "FNO": "-."}

x = np.array(n_samps)

fig, ax = plt.subplots(figsize=(5.5, 3.5))

for arch, imp, sig in [("MLP", mlp_imp, mlp_sig),
                        ("CNN", cnn_imp, cnn_sig),
                        ("FNO", fno_imp, fno_sig)]:
    ax.plot(x, imp,
            color=COLORS[arch], marker=MARKERS[arch], ls=LINES[arch],
            ms=5, lw=1.6, label=arch, zorder=3, clip_on=False)
    # Open marker → filled for significant points
    for xi, yi, s in zip(x, imp, sig):
        if s:
            ax.plot(xi, yi, marker=MARKERS[arch], ms=6,
                    color=COLORS[arch], mec=COLORS[arch], mfc=COLORS[arch],
                    zorder=4, ls="none")
        else:
            ax.plot(xi, yi, marker=MARKERS[arch], ms=5,
                    color=COLORS[arch], mec=COLORS[arch], mfc="white",
                    zorder=4, ls="none")

# Zero reference
ax.axhline(0, color="0.50", lw=0.7, ls=":", zorder=1)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlabel("RANS Training Data Fraction")
ax.set_ylabel(r"$C_p$ RMSE Improvement over Scratch (%)")
ax.set_xticks(x)
ax.set_xticklabels(fracs)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.yaxis.grid(True, which="major", color="0.90", lw=0.5, zorder=0)
ax.set_axisbelow(True)

# ── Legend ────────────────────────────────────────────────────────────────────
leg = ax.legend(loc="upper right", frameon=True,
                framealpha=1.0, edgecolor="0.80",
                handlelength=2.0, handletextpad=0.5,
                borderpad=0.6)
leg.get_frame().set_linewidth(0.5)

ax.text(0.02, 0.04,
        r"Filled markers: $p < 0.05$",
        transform=ax.transAxes, fontsize=7.5, color="0.50", va="bottom")

plt.tight_layout(pad=0.6)
out = os.path.join(OUT_DIR, "fig7_cp_budget.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved {out}")
plt.show()