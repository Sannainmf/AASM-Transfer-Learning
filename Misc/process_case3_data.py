"""
Process AASM Case 3 RANS Data (v2 — Cp + Cf)
==============================================
Now also extracts signed skin friction coefficient from cfx, cfz components.
The tangent vector is computed from the surface normals (nx, nz).

Usage:
    python process_case3_data_v2.py
"""

import glob
import os
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

TRAIN_PATH = r"C:\Users\areen\Documents\trainingData-20260406T015436Z-1-001\trainingData"
TEST_PATH = r"C:\Users\areen\Documents\testData-20260406T015314Z-1-001\testData"
OUTPUT_DIR = r"C:\Users\areen\Documents\xfoil_transfer_learning"

N_CP_POINTS = 256  # per surface, 512 total


# ============================================================================
# LOAD SAMPLES
# ============================================================================

def load_samples(data_path):
    """Load all samples from a folder into a dictionary."""
    sample_files = list(glob.glob(os.path.join(data_path, "*")))
    sample_dict = {}

    for fname in sample_files:
        basename = os.path.basename(fname)
        if "Sample" not in basename:
            continue

        with open(fname, 'r') as file:
            first_line = file.readline().strip()
            sample_number = int(first_line.split("_")[0].split("Sample")[1])

        sample_dict[sample_number] = {}
        ct = 1

        with open(fname, 'r') as file:
            for line in file:
                if ":" in line:
                    key, value = line.strip().split(":", 1)
                    sample_dict[sample_number][key.strip()] = float(value)
                    ct += 1

        data = np.loadtxt(fname, skiprows=ct + 1)
        var_names = ["x", "z", "cp", "cfx", "cfz", "nx", "nz"]
        for i, var in enumerate(var_names):
            sample_dict[sample_number][var] = data[:, i]

    return sample_dict


# ============================================================================
# COMPUTE SIGNED CF FROM COMPONENTS
# ============================================================================

def compute_signed_cf(cfx, cfz, nx, nz, z):
    """
    z: the surface z-coordinates (positive = upper, negative = lower)
    """
    norm_mag = np.sqrt(nx**2 + nz**2)
    norm_mag[norm_mag < 1e-15] = 1.0

    nx_unit = nx / norm_mag
    nz_unit = nz / norm_mag

    tx = nz_unit
    tz = -nx_unit

    cf_signed = cfx * tx + cfz * tz

    # Flip sign for upper surface (z > 0) because tangent points upstream there
    cf_signed[z > 0] *= -1

    return cf_signed


# ============================================================================
# INTERPOLATION
# ============================================================================

def interpolate_surface_data(x_raw, data_raw, n_points=N_CP_POINTS):
    """Interpolate any surface quantity onto a fixed grid (upper + lower)."""
    min_idx = np.argmin(x_raw)

    x_lower = x_raw[:min_idx + 1]
    data_lower = data_raw[:min_idx + 1]
    x_upper = x_raw[min_idx:]
    data_upper = data_raw[min_idx:]

    # Lower goes TE->LE, reverse to LE->TE
    x_lower = x_lower[::-1]
    data_lower = data_lower[::-1]

    # Upper should go LE->TE
    if x_upper[0] > x_upper[-1]:
        x_upper = x_upper[::-1]
        data_upper = data_upper[::-1]

    x_grid = np.linspace(0, 1, n_points)

    upper_interp = np.interp(x_grid, x_upper, data_upper)
    lower_interp = np.interp(x_grid, x_lower, data_lower)

    return np.concatenate([upper_interp, lower_interp])


# ============================================================================
# PROCESS AND SAVE
# ============================================================================

def process_dataset(data_path, output_name):
    """Process all samples and save as .npz with both Cp and Cf."""
    print(f"Loading samples from: {data_path}")
    samples = load_samples(data_path)
    print(f"  Found {len(samples)} samples")

    sorted_keys = sorted(samples.keys())

    all_inputs = []
    all_cp = []
    all_cf = []
    n_success = 0
    n_failed = 0

    for key in sorted_keys:
        s = samples[key]

        # Extract inputs
        try:
            inputs = np.array([
                s['CST1'], s['CST2'], s['CST3'], s['CST4'], s['CST5'],
                s['CST6'], s['CST7'], s['CST8'], s['CST9'],
                s['Mach'], s['AoA'], s['Re'],
            ])
        except KeyError as e:
            print(f"  Sample {key}: missing key {e}, skipping")
            n_failed += 1
            continue

        try:
            # Interpolate Cp
            cp_interp = interpolate_surface_data(s['x'], s['cp'])

            # Compute signed Cf from components
            cf_signed = compute_signed_cf(s['cfx'], s['cfz'], s['nx'], s['nz'], s['z'])
            
            # Interpolate Cf
            cf_interp = interpolate_surface_data(s['x'], cf_signed)

        except Exception as e:
            print(f"  Sample {key}: processing failed ({e}), skipping")
            n_failed += 1
            continue

        all_inputs.append(inputs)
        all_cp.append(cp_interp)
        all_cf.append(cf_interp)
        n_success += 1

    inputs_array = np.array(all_inputs)
    cp_array = np.array(all_cp)
    cf_array = np.array(all_cf)
    x_grid = np.linspace(0, 1, N_CP_POINTS)

    output_file = os.path.join(OUTPUT_DIR, output_name)
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
    )

    print(f"  Successful: {n_success}")
    print(f"  Failed: {n_failed}")
    print(f"  Input shape: {inputs_array.shape}")
    print(f"  Cp shape: {cp_array.shape}")
    print(f"  Cf shape: {cf_array.shape}")
    print(f"  Saved to: {output_file}")

    return inputs_array, cp_array, cf_array


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Processing AASM Case 3 RANS Data (v2 — Cp + Cf)")
    print("=" * 60)

    print("\n--- Training Data ---")
    train_inputs, train_cp, train_cf = process_dataset(TRAIN_PATH, "case3_train_v2.npz")

    print("\n--- Test Data ---")
    test_inputs, test_cp, test_cf = process_dataset(TEST_PATH, "case3_test_v2.npz")

    # Sanity check: Cf statistics
    print("\n--- Cf Sanity Check ---")
    print(f"  Train Cf range: [{train_cf.min():.6f}, {train_cf.max():.6f}]")
    print(f"  Train Cf mean:  {train_cf.mean():.6f}")
    print(f"  Test Cf range:  [{test_cf.min():.6f}, {test_cf.max():.6f}]")
    print(f"  Test Cf mean:   {test_cf.mean():.6f}")

    # Check for negative Cf (separation)
    n_neg_train = np.sum(train_cf < 0) / train_cf.size * 100
    n_neg_test = np.sum(test_cf < 0) / test_cf.size * 100
    print(f"  Train: {n_neg_train:.1f}% of Cf values are negative (separation)")
    print(f"  Test:  {n_neg_test:.1f}% of Cf values are negative (separation)")

    print("\nDone!")