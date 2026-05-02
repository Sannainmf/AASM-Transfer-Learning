"""
Generate LaTeX for Table 1: Input Parameter Bounds
"""

import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Parameter bounds (from xfoil_gen.py) ─────────────────────────────────────

rows = [
    ("Mach number",      "$M$",                       "0.2",       "0.7"),
    ("Angle of attack",  r"$\alpha$ (deg)",            "$-3.0$",    "5.0"),
    ("Reynolds number",  "$Re$",                      r"$1\times10^6$", r"$6.5\times10^6$"),
    ("CST1 (shared)",    r"$X_{u,1} = X_{l,1}$",      "0.0644",    "0.1932"),
    ("CST2 (upper)",     r"$X_{u,2}$",                "0.0688",    "0.2064"),
    ("CST3 (upper)",     r"$X_{u,3}$",                "0.0961",    "0.2883"),
    ("CST4 (upper)",     r"$X_{u,4}$",                "0.0961",    "0.2882"),
    ("CST5 (upper)",     r"$X_{u,5}$",                "0.1010",    "0.3030"),
    ("CST6 (lower)",     r"$X_{l,2}$",                "0.0680",    "0.2039"),
    ("CST7 (lower)",     r"$X_{l,3}$",                "0.1126",    "0.3377"),
    ("CST8 (lower)",     r"$X_{l,4}$",                "0.0381",    "0.1143"),
    ("CST9 (lower)",     r"$X_{l,5}$",                "$-0.0586$", "$-0.0195$"),
]

# ── Build LaTeX ───────────────────────────────────────────────────────────────

lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"  \centering")
lines.append(r"  \caption{Input parameter bounds for AASM Benchmark Case 3, "
             r"adapted from \citet{bekemeyer2025}.}")
lines.append(r"  \label{tab:param_bounds}")
lines.append(r"  \begin{tabular}{llcc}")
lines.append(r"    \hline")
lines.append(r"    \textbf{Parameter} & \textbf{Symbol} & \textbf{Lower Bound} & \textbf{Upper Bound} \\")
lines.append(r"    \hline")
for name, symbol, lo, hi in rows:
    lines.append(f"    {name} & {symbol} & {lo} & {hi} \\\\")
lines.append(r"    \hline")
lines.append(r"  \end{tabular}")
lines.append(r"  \begin{tablenotes}")
lines.append(r"    \small")
lines.append(r"    \item CST1 is shared between upper and lower surfaces.")
lines.append(r"    \item Designs are derived from the RAE 2822 baseline using CST perturbation.")
lines.append(r"    \item 597 total samples (497 train, 100 test) generated using a Sobol sequence.")
lines.append(r"  \end{tablenotes}")
lines.append(r"\end{table}")

latex = "\n".join(lines)

# ── Write to file ─────────────────────────────────────────────────────────────

out_path = os.path.join(OUTPUT_DIR, "table1_param_bounds.tex")
with open(out_path, "w") as f:
    f.write(latex + "\n")

print(f"Wrote {out_path}")
print()
print(latex)