# verify_cst.py
# Check that our CST formula correctly reconstructs airfoil shapes from the RANS data

import numpy as np
import glob
import os

DATA_PATH = r"C:\Users\areen\Documents\trainingData-20260406T015436Z-1-001\trainingData"


def cst_shape(x, coeffs):
    """Compute z-coordinates from CST coefficients at given x positions."""
    # class function: sqrt(x) * (1-x)
    C = np.sqrt(x) * (1 - x)
    
    # 4th order bernstein polynomials (5 coefficients)
    n = 4
    S = np.zeros_like(x)
    for i, a in enumerate(coeffs):
        # binomial(n, i) * x^i * (1-x)^(n-i)
        from math import comb
        bern = comb(n, i) * (x ** i) * ((1 - x) ** (n - i))
        S += a * bern
    
    return C * S


def check_sample(filepath):
    """Load one sample, reconstruct shape from CST, compare to actual coordinates."""
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
    
    sample_data = {}
    ct = 1
    with open(filepath, 'r') as f:
        for line in f:
            if ":" in line:
                key, value = line.strip().split(":", 1)
                sample_data[key.strip()] = float(value)
                ct += 1
    
    data = np.loadtxt(filepath, skiprows=ct + 1)
    x_raw = data[:, 0]
    z_raw = data[:, 1]
    
    # CST coefficients
    cst1 = sample_data['CST1']
    upper_coeffs = [cst1, sample_data['CST2'], sample_data['CST3'], sample_data['CST4'], sample_data['CST5']]
    lower_coeffs = [cst1, sample_data['CST6'], sample_data['CST7'], sample_data['CST8'], sample_data['CST9']]
    
    # split at leading edge
    min_idx = np.argmin(x_raw)
    x_lower = x_raw[:min_idx + 1][::-1]  # reverse to LE->TE
    z_lower = z_raw[:min_idx + 1][::-1]
    x_upper = x_raw[min_idx:]
    z_upper = z_raw[min_idx:]
    
    if x_upper[0] > x_upper[-1]:
        x_upper = x_upper[::-1]
        z_upper = z_upper[::-1]
    
    # skip x=0 and x=1 (class function is 0 there)
    mask_u = (x_upper > 1e-6) & (x_upper < 1 - 1e-6)
    mask_l = (x_lower > 1e-6) & (x_lower < 1 - 1e-6)
    
    z_upper_pred = cst_shape(x_upper[mask_u], upper_coeffs)
    z_lower_pred = -cst_shape(x_lower[mask_l], lower_coeffs)  # negative for lower surface
    
    rmse_upper = np.sqrt(np.mean((z_upper_pred - z_upper[mask_u])**2))
    rmse_lower = np.sqrt(np.mean((z_lower_pred - z_lower[mask_l])**2))
    
    return rmse_upper, rmse_lower


# check first 10 samples
files = sorted(glob.glob(os.path.join(DATA_PATH, "*")))
sample_files = [f for f in files if "Sample" in os.path.basename(f)][:10]

print("CST Verification:")
print(f"{'Sample':<20} {'Upper RMSE':<15} {'Lower RMSE':<15}")
print("-" * 50)

for f in sample_files:
    name = os.path.basename(f)
    try:
        ru, rl = check_sample(f)
        print(f"{name:<20} {ru:<15.10f} {rl:<15.10f}")
    except Exception as e:
        print(f"{name:<20} ERROR: {e}")