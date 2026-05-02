"""
Figure: XFOIL vs RANS Cf Comparison
Runs XFOIL on the EXACT same CST/Mach/AoA/Re as each RANS test sample.
Cf axis is NOT inverted (positive up — standard convention).
"""

import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from math import comb

PROJECT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR     = Path(os.path.dirname(os.path.abspath(__file__)))
XFOIL_EXE   = r"C:\Users\areen\Downloads\XFOIL6.99\xfoil.exe"
TEMP_DIR    = PROJECT_DIR / "temp_xfoil"
TEMP_DIR.mkdir(exist_ok=True)

N_POINTS = 256
MAX_ITER = 200

# ── CST + XFOIL helpers ───────────────────────────────────────────────────────

def bernstein(n, k, x):
    from math import comb
    return comb(n, k) * (x ** k) * ((1 - x) ** (n - k))


def cst_airfoil(cst_params, n_points=300):
    cst_upper = cst_params[[0, 1, 2, 3, 4]]
    cst_lower = cst_params[[0, 5, 6, 7, 8]]
    beta = np.linspace(0, np.pi, n_points)
    x    = 0.5 * (1 - np.cos(beta))
    C    = np.sqrt(x) * (1 - x)
    def shape(c, x):
        n = len(c) - 1
        S = np.zeros_like(x)
        for k in range(n + 1):
            S += c[k] * bernstein(n, k, x)
        return S
    return x, C * shape(cst_upper, x), -C * shape(cst_lower, x)


def write_dat(path, x, y_upper, y_lower):
    with open(path, "w") as f:
        f.write("AIRFOIL\n")
        for i in range(len(x) - 1, -1, -1):
            f.write(f"  {x[i]:.10f}  {y_upper[i]:.10f}\n")
        for i in range(1, len(x)):
            f.write(f"  {x[i]:.10f}  {y_lower[i]:.10f}\n")


def run_xfoil_cf(foil_file, alpha, Re, Mach, timeout=45):
    """Run XFOIL and extract Cf via DUMP."""
    cp_out   = "cp_out.txt"
    dump_out = "bl_out.txt"
    for f in [cp_out, dump_out]:
        p = TEMP_DIR / f
        if p.exists():
            os.remove(p)

    cmds = (f"LOAD {foil_file}\nPANE\nOPER\n"
            f"VISC {Re:.0f}\nMACH {Mach:.4f}\nITER {MAX_ITER}\n"
            f"ALFA {alpha:.4f}\nCPWR {cp_out}\nDUMP {dump_out}\n\nQUIT\n")
    try:
        subprocess.run([XFOIL_EXE], input=cmds, capture_output=True,
                       text=True, timeout=timeout, cwd=str(TEMP_DIR))
    except Exception:
        return None

    # Parse DUMP: columns s, x, y, Ue, Dstar, Theta, Cf
    dump_path = TEMP_DIR / dump_out
    if not dump_path.exists():
        return None
    try:
        lines = (TEMP_DIR / dump_out).read_text().strip().split("\n")
        data  = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 7:
                try:
                    data.append((float(parts[1]), float(parts[6])))  # x, Cf
                except ValueError:
                    continue
        if len(data) < 20:
            return None
        arr = np.array(data)
        return arr[:, 0], arr[:, 1]   # x, Cf
    except Exception:
        return None


def interpolate_surface(x_raw, data_raw, n=N_POINTS):
    """Split at LE and interpolate upper/lower onto fixed grid."""
    min_idx  = np.argmin(x_raw)
    x_upper  = x_raw[:min_idx + 1][::-1]
    d_upper  = data_raw[:min_idx + 1][::-1]
    x_lower  = x_raw[min_idx:]
    d_lower  = data_raw[min_idx:]
    xg = np.linspace(0, 1, n)
    return np.interp(xg, x_upper, d_upper), np.interp(xg, x_lower, d_lower)


# ── Load RANS test data ───────────────────────────────────────────────────────
rans    = np.load(PROJECT_DIR / "Data" / "case3_test_v2.npz")
rans_X  = rans["inputs"]
rans_cf = rans["cf_distributions"]

TEST_INDICES = [51, 45, 25]
x_grid = np.linspace(0, 1, N_POINTS)

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

SUBLABELS = ['(a)', '(b)', '(c)']

fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.8))

for i, (ax, test_idx) in enumerate(zip(axes, TEST_INDICES)):
    inp  = rans_X[test_idx]
    cst  = inp[:9]
    mach = inp[9]; aoa = inp[10]; re = inp[11]

    print(f"\nRunning XFOIL: M={mach:.3f}, AoA={aoa:.2f}, Re={re:.0f}")

    x_foil, y_upper, y_lower = cst_airfoil(cst)
    dat_path = TEMP_DIR / "current.dat"
    write_dat(dat_path, x_foil, y_upper, y_lower)

    result = run_xfoil_cf("current.dat", aoa, re, mach)

    # RANS Cf
    cf_rans_upper = rans_cf[test_idx][:N_POINTS]
    cf_rans_lower = rans_cf[test_idx][N_POINTS:]

    ax.plot(x_grid, cf_rans_upper, color="black",  lw=1.2, ls="-",  label="RANS")
    ax.plot(x_grid, cf_rans_lower, color="black",  lw=1.2, ls="-")

    if result is not None:
        x_xf, cf_xf = result
        cf_xf_upper, cf_xf_lower = interpolate_surface(x_xf, cf_xf)
        ax.plot(x_grid, cf_xf_upper, color="#C0392B", lw=1.0, ls="--", label="XFOIL")
        ax.plot(x_grid, cf_xf_lower, color="#C0392B", lw=1.0, ls="--")
        print("  XFOIL converged.")
    else:
        ax.text(0.5, 0.5, "XFOIL did not converge",
                transform=ax.transAxes, ha="center", fontsize=8, color="0.5")
        print("  XFOIL did not converge.")

    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$C_f$")
    ax.yaxis.grid(True, which="major", color="0.92", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(
        f"{SUBLABELS[i]}  $M={mach:.2f}$,  $\\alpha={aoa:.1f}^\\circ$,  $Re={re/1e6:.1f}\\times10^6$",
        fontsize=8.5, pad=4)

plt.tight_layout(pad=0.6, w_pad=1.8)
plt.subplots_adjust(top=0.82)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98),
           ncol=2, frameon=True, framealpha=1.0, edgecolor="0.80",
           handlelength=2.0, fontsize=8.5)
out = str(OUT_DIR / "fig_cf_comparison.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print(f"\nSaved {out}")
plt.show()