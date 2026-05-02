"""
Paired t-test: Model B (Scratch) vs Model C (Transfer)
Tests whether transfer learning significantly improves over scratch
for both Cp and Cf across MLP, CNN, and FNO.
"""

import json
import os
import numpy as np
from scipy import stats

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(filename):
    with open(os.path.join(PROJECT_DIR, filename)) as f:
        return json.load(f)


def cohens_d(a, b):
    diff = np.array(a) - np.array(b)
    return diff.mean() / diff.std(ddof=1)


results = {
    "MLP": load("final_results_mlp.json"),
    "CNN": load("final_results_cnn.json"),
    "FNO": load("final_results_fno.json"),
}

print(f"{'Arch':<6} {'Output':<4} {'B RMSE (mean)':<16} {'C RMSE (mean)':<16} "
      f"{'Improvement':<13} {'p-value':<10} {'Cohens_d':<10} {'Sig?'}")
print("-" * 90)

for arch, data in results.items():
    for output in ("cp", "cf"):
        b_seeds = data[output]["model_b"]["rmse_seeds"]
        c_seeds = data[output]["model_c"]["rmse_seeds"]

        b_mean = np.mean(b_seeds)
        c_mean = np.mean(c_seeds)
        improvement = (b_mean - c_mean) / b_mean * 100

        t_stat, p_value = stats.ttest_rel(b_seeds, c_seeds)
        d = cohens_d(b_seeds, c_seeds)
        sig = "YES" if p_value < 0.05 else "no"

        print(f"{arch:<6} {output.upper():<4} {b_mean:<16.6f} {c_mean:<16.6f} "
              f"{improvement:>+.2f}%      {p_value:<10.4f} {d:<10.4f} {sig}")