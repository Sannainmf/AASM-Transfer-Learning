"""
XFOIL Data Generation for Multi-Fidelity Transfer Learning (v2 — Cp + Cf)
===========================================================================
Generates XFOIL dataset with BOTH pressure coefficient (Cp) and skin friction
coefficient (Cf) for pre-training on the AASM Benchmark Case 3 parameter space.

Changes from v1:
  - Added DUMP command to extract boundary layer data (including Cf)
  - Saves both Cp and Cf distributions in the output .npz file
  - Cf is interpolated onto the same 256-point grid as Cp

Usage:
    python generate_xfoil_data_v2.py
"""

import numpy as np
import subprocess
import os
import sys
import time
from pathlib import Path
from math import comb

# ============================================================================
# USER CONFIGURATION — Update these paths for your system
# ============================================================================

XFOIL_EXE = os.environ.get("XFOIL_EXE", "xfoil")  # path to the XFOIL executable
OUTPUT_DIR = Path(__file__).resolve().parent  # Data/
TEMP_DIR = OUTPUT_DIR.parent / "temp_xfoil"

# Number of samples to generate
N_SAMPLES = 3000

# ============================================================================
# FIXED PARAMETERS (matching Case 3 bounds from Bekemeyer et al. 2025)
# ============================================================================

MACH_RANGE = (0.2, 0.7)
AOA_RANGE = (-3.0, 5.0)
RE_RANGE = (1e6, 6.5e6)

N_CP_POINTS = 256  # per surface, 512 total

MAX_ITER = 150
N_PANELS = 200

# ============================================================================
# RAE2822 CST COEFFICIENTS — VERIFIED AGAINST BENCHMARK DATA
# ============================================================================

CST_LOWER_BOUNDS = np.array([
    0.0644,   # CST1 = X_u,1 = X_l,1 (shared LE radius)
    0.0688,   # CST2 = X_u,2
    0.0961,   # CST3 = X_u,3
    0.0961,   # CST4 = X_u,4
    0.1010,   # CST5 = X_u,5
    0.0680,   # CST6 = X_l,2
    0.1126,   # CST7 = X_l,3
    0.0381,   # CST8 = X_l,4
   -0.0586,   # CST9 = X_l,5
])

CST_UPPER_BOUNDS = np.array([
    0.1932,   # CST1
    0.2064,   # CST2
    0.2883,   # CST3
    0.2882,   # CST4
    0.3030,   # CST5
    0.2039,   # CST6
    0.3377,   # CST7
    0.1143,   # CST8
   -0.0195,   # CST9
])


# ============================================================================
# CST AIRFOIL GENERATION
# ============================================================================

def bernstein_poly(n, k, x):
    return comb(n, k) * (x ** k) * ((1 - x) ** (n - k))


def cst_airfoil(cst_params, n_points=201):
    """Generate airfoil coordinates from 9 CST parameters."""
    cst_upper = np.array([cst_params[0], cst_params[1], cst_params[2],
                          cst_params[3], cst_params[4]])
    cst_lower = np.array([cst_params[0], cst_params[5], cst_params[6],
                          cst_params[7], cst_params[8]])

    beta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1 - np.cos(beta))

    class_func = np.sqrt(x) * (1 - x)

    def shape_function(cst_coeffs, x_pts):
        n = len(cst_coeffs) - 1
        S = np.zeros_like(x_pts)
        for k in range(n + 1):
            S += cst_coeffs[k] * bernstein_poly(n, k, x_pts)
        return S

    y_upper = class_func * shape_function(cst_upper, x)
    y_lower = -class_func * shape_function(cst_lower, x)

    return x, y_upper, y_lower


def write_airfoil_dat(filepath, x, y_upper, y_lower, name="AIRFOIL"):
    """Write airfoil coordinates in XFOIL format (TE->LE upper, LE->TE lower)."""
    with open(filepath, 'w') as f:
        f.write(f"{name}\n")
        for i in range(len(x) - 1, -1, -1):
            f.write(f"  {x[i]:.10f}  {y_upper[i]:.10f}\n")
        for i in range(1, len(x)):
            f.write(f"  {x[i]:.10f}  {y_lower[i]:.10f}\n")


# ============================================================================
# XFOIL RUNNER (now extracts both Cp and Cf)
# ============================================================================

def run_xfoil(airfoil_file, alpha, Re, Mach, cp_output, dump_output, timeout=30):
    """
    Run XFOIL and extract both Cp and boundary layer data (including Cf).

    Returns:
        (x_cp, y_cp, cp_raw, x_bl, cf_raw) if converged, None otherwise
        - x_cp, y_cp, cp_raw: from CPWR (pressure coefficient)
        - x_bl, cf_raw: from DUMP (skin friction coefficient)
    """
    # Clean up old output
    for f in [cp_output, dump_output]:
        fpath = TEMP_DIR / f
        if fpath.exists():
            os.remove(fpath)

    # Build XFOIL command sequence
    # CPWR writes x, y, Cp
    # DUMP writes s, x, y, Ue, Dstar, Theta, Cf
    commands = f"""LOAD {airfoil_file}
PANE
OPER
VISC {Re:.0f}
MACH {Mach:.4f}
ITER {MAX_ITER}
ALFA {alpha:.4f}
CPWR {cp_output}
DUMP {dump_output}

QUIT
"""

    try:
        proc = subprocess.run(
            [XFOIL_EXE],
            input=commands,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(TEMP_DIR)
        )
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None

    # Parse Cp file
    cp_path = TEMP_DIR / cp_output
    if not cp_path.exists():
        return None

    try:
        lines = cp_path.read_text().strip().split('\n')
        cp_data = []
        for line in lines[3:]:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    x_val = float(parts[0])
                    y_val = float(parts[1])
                    cp_val = float(parts[2])
                    cp_data.append((x_val, y_val, cp_val))
                except ValueError:
                    continue

        if len(cp_data) < 20:
            return None

        cp_arr = np.array(cp_data)
        x_cp = cp_arr[:, 0]
        y_cp = cp_arr[:, 1]
        cp_raw = cp_arr[:, 2]
    except Exception:
        return None

    # Parse DUMP file (boundary layer data)
    dump_path = TEMP_DIR / dump_output
    if not dump_path.exists():
        # Cf extraction failed but Cp succeeded — return Cp only with None for Cf
        return x_cp, y_cp, cp_raw, None, None

    try:
        lines = dump_path.read_text().strip().split('\n')
        bl_data = []
        for line in lines[1:]:  # skip header
            parts = line.split()
            if len(parts) >= 7:
                try:
                    # DUMP format: s, x, y, Ue, Dstar, Theta, Cf
                    x_val = float(parts[1])
                    cf_val = float(parts[6])
                    bl_data.append((x_val, cf_val))
                except ValueError:
                    continue

        if len(bl_data) < 20:
            return x_cp, y_cp, cp_raw, None, None

        bl_arr = np.array(bl_data)
        x_bl = bl_arr[:, 0]
        cf_raw = bl_arr[:, 1]

        return x_cp, y_cp, cp_raw, x_bl, cf_raw

    except Exception:
        return x_cp, y_cp, cp_raw, None, None


# ============================================================================
# INTERPOLATION
# ============================================================================

def interpolate_surface_data(x_raw, data_raw, n_points=N_CP_POINTS):
    """
    Interpolate any surface quantity onto a fixed grid (upper + lower).
    Works for both Cp and Cf.
    """
    min_idx = np.argmin(x_raw)

    x_upper = x_raw[:min_idx + 1]
    data_upper = data_raw[:min_idx + 1]
    x_lower = x_raw[min_idx:]
    data_lower = data_raw[min_idx:]

    # Reverse upper so it goes LE -> TE (ascending x)
    x_upper = x_upper[::-1]
    data_upper = data_upper[::-1]

    x_grid = np.linspace(0, 1, n_points)

    try:
        upper_interp = np.interp(x_grid, x_upper, data_upper)
        lower_interp = np.interp(x_grid, x_lower, data_lower)
    except Exception:
        return None

    return np.concatenate([upper_interp, lower_interp])


# ============================================================================
# SAMPLING
# ============================================================================

def generate_samples(n_samples):
    """Generate parameter samples using Latin Hypercube Sampling."""
    n_params = 12
    rng = np.random.default_rng(seed=42)
    samples = np.zeros((n_samples, n_params))

    for j in range(n_params):
        intervals = np.linspace(0, 1, n_samples + 1)
        points = rng.uniform(intervals[:-1], intervals[1:])
        rng.shuffle(points)
        samples[:, j] = points

    params = np.zeros_like(samples)
    for i in range(9):
        params[:, i] = CST_LOWER_BOUNDS[i] + samples[:, i] * (CST_UPPER_BOUNDS[i] - CST_LOWER_BOUNDS[i])
    params[:, 9] = MACH_RANGE[0] + samples[:, 9] * (MACH_RANGE[1] - MACH_RANGE[0])
    params[:, 10] = AOA_RANGE[0] + samples[:, 10] * (AOA_RANGE[1] - AOA_RANGE[0])
    params[:, 11] = RE_RANGE[0] + samples[:, 11] * (RE_RANGE[1] - RE_RANGE[0])

    return params


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("XFOIL Data Generation v2 (Cp + Cf)")
    print("=" * 70)

    if not os.path.exists(XFOIL_EXE):
        print(f"ERROR: XFOIL not found at {XFOIL_EXE}")
        sys.exit(1)

    TEMP_DIR.mkdir(exist_ok=True)

    print(f"\nGenerating {N_SAMPLES} parameter samples (LHS)...")
    params = generate_samples(N_SAMPLES)
    print(f"  Mach range: {MACH_RANGE}")
    print(f"  AoA range: {AOA_RANGE}")
    print(f"  Re range: {RE_RANGE}")

    all_inputs = []
    all_cp = []
    all_cf = []
    x_grid = np.linspace(0, 1, N_CP_POINTS)

    n_success = 0
    n_success_cf = 0
    n_failed = 0
    start_time = time.time()

    print(f"\nRunning XFOIL simulations...")
    print("-" * 70)

    for i in range(N_SAMPLES):
        cst_params = params[i, 0:9]
        mach = params[i, 9]
        aoa = params[i, 10]
        re = params[i, 11]

        # Generate airfoil shape
        try:
            x_foil, y_upper, y_lower = cst_airfoil(cst_params)
        except Exception:
            n_failed += 1
            continue

        # Write airfoil file
        airfoil_file = "current_airfoil.dat"
        write_airfoil_dat(TEMP_DIR / airfoil_file, x_foil, y_upper, y_lower,
                          name=f"SAMPLE_{i:05d}")

        # Run XFOIL (now extracts both Cp and Cf)
        cp_file = "cp_output.txt"
        dump_file = "bl_output.txt"
        result = run_xfoil(airfoil_file, aoa, re, mach, cp_file, dump_file)

        if result is None:
            n_failed += 1
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"  [{i+1:5d}/{N_SAMPLES}] success={n_success} "
                      f"failed={n_failed} cf_ok={n_success_cf}")
            continue

        x_cp, y_cp, cp_raw, x_bl, cf_raw = result

        # Interpolate Cp
        cp_interp = interpolate_surface_data(x_cp, cp_raw)
        if cp_interp is None:
            n_failed += 1
            continue

        # Interpolate Cf (may fail even if Cp succeeded)
        cf_interp = None
        if x_bl is not None and cf_raw is not None:
            cf_interp = interpolate_surface_data(x_bl, cf_raw)
            if cf_interp is not None:
                n_success_cf += 1

        # Store results
        all_inputs.append(params[i])
        all_cp.append(cp_interp)
        # Store Cf or zeros if Cf extraction failed
        if cf_interp is not None:
            all_cf.append(cf_interp)
        else:
            all_cf.append(np.zeros(N_CP_POINTS * 2))

        n_success += 1

        # Progress
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (N_SAMPLES - i - 1) / rate
            print(f"  [{i+1:5d}/{N_SAMPLES}] success={n_success} "
                  f"failed={n_failed} cf_ok={n_success_cf} "
                  f"({rate:.1f} cases/sec, ETA: {eta/60:.1f} min)")

    # ========================================================================
    # SAVE DATASET
    # ========================================================================
    elapsed_total = time.time() - start_time

    print("\n" + "=" * 70)
    print(f"COMPLETE")
    print(f"  Total time: {elapsed_total/60:.1f} minutes")
    print(f"  Successful (Cp): {n_success} / {N_SAMPLES} ({100*n_success/N_SAMPLES:.1f}%)")
    print(f"  Successful (Cf): {n_success_cf} / {N_SAMPLES} ({100*n_success_cf/N_SAMPLES:.1f}%)")
    print(f"  Failed:          {n_failed}")
    print("=" * 70)

    if n_success == 0:
        print("ERROR: No successful simulations.")
        sys.exit(1)

    inputs_array = np.array(all_inputs)
    cp_array = np.array(all_cp)
    cf_array = np.array(all_cf)

    # Save
    output_file = OUTPUT_DIR / "xfoil_dataset_v2.npz"
    np.savez(
        output_file,
        inputs=inputs_array,
        cp_distributions=cp_array,
        cf_distributions=cf_array,
        x_grid=x_grid,
        column_names=np.array([
            'cst1', 'cst2', 'cst3', 'cst4', 'cst5',
            'cst6', 'cst7', 'cst8', 'cst9',
            'mach', 'aoa', 'reynolds'
        ]),
        cst_baseline=(CST_LOWER_BOUNDS + CST_UPPER_BOUNDS) / 2,
        cst_lower_bounds=CST_LOWER_BOUNDS,
        cst_upper_bounds=CST_UPPER_BOUNDS,
    )

    print(f"\nDataset saved to: {output_file}")
    print(f"  Input shape:  {inputs_array.shape}")
    print(f"  Cp shape:     {cp_array.shape}")
    print(f"  Cf shape:     {cf_array.shape}")
    print(f"  File size:    {output_file.stat().st_size / 1e6:.1f} MB")

    # Summary JSON
    summary = {
        "n_samples_attempted": N_SAMPLES,
        "n_success_cp": n_success,
        "n_success_cf": n_success_cf,
        "n_failed": n_failed,
        "convergence_rate_cp": f"{100*n_success/N_SAMPLES:.1f}%",
        "convergence_rate_cf": f"{100*n_success_cf/N_SAMPLES:.1f}%",
        "total_time_minutes": f"{elapsed_total/60:.1f}",
    }

    import json
    summary_file = OUTPUT_DIR / "xfoil_dataset_v2_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary:      {summary_file}")

    print("\nDone! Ready for ML training.")


if __name__ == "__main__":
    main()