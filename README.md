# Multi-Fidelity Transfer Learning for Airfoil Surface Distribution Prediction

This repository contains the datasets, training scripts, results, and figures for the paper:

> "Multi-Fidelity Transfer Learning for Airfoil Surface Distribution Prediction on the AASM Benchmark" (accepted, AIAA SciTech Forum)

## Overview

We predict pressure coefficient (Cp) and skin friction coefficient (Cf) distributions over parametric airfoils from the AASM Benchmark Case 3. Low-fidelity XFOIL solutions are cheap to generate in bulk, while high-fidelity RANS samples are scarce. We pretrain neural surrogates (MLP, 1D-CNN, and FNO) on 2995 XFOIL samples, fine-tune them on 497 RANS samples, and compare against training from scratch. A data-budget study repeats the comparison at 10% to 100% of the RANS training set with paired significance tests, showing that transfer learning helps most when RANS data is limited.

## Repository Contents

```
Data/
├── xfoil_gen.py                   # Generates the XFOIL Cp/Cf dataset over the Case 3 parameter space
├── process_case3_data.py          # Parses the raw AASM Case 3 RANS samples into .npz files
├── xfoil_dataset_v2.npz           # XFOIL dataset (2995 samples)
├── case3_train_v2.npz             # RANS training set (497 samples)
└── case3_test_v2.npz              # RANS test set (100 samples)

Models/                            # One folder per architecture, each split into Pressure/ (Cp) and Friction/ (Cf)
├── MLP/
├── 1D-CNN/
└── FNO/
    ├── Pressure/
    │   ├── cp_model_a.py          # Model A: train on XFOIL only, evaluate on RANS
    │   ├── cp_model_b.py          # Model B: train on RANS from scratch
    │   ├── cp_model_c.py          # Model C: pretrain on XFOIL, fine-tune on RANS
    │   └── cp_budget_exp.py       # Data-budget study: transfer vs scratch at 10-100% of RANS data
    └── Friction/                  # Same four scripts for Cf

Results/
├── final_results_mlp.json         # Paper numbers per architecture (RMSE, R², per-seed values, budget curves)
├── final_results_cnn.json
├── final_results_fno.json
├── significance_test.py           # Paired t-tests, scratch vs transfer
├── CNN/                           # Per-run outputs
└── FNO/

Figures/
├── plot_cp_comparison.py          # XFOIL vs RANS Cp on the test cases
├── plot_cf_comparison.py          # XFOIL vs RANS Cf on the test cases
├── plot_cp_budget.py              # Cp data-efficiency curve
├── plot_cf_budget.py              # Cf data-efficiency curve
├── plot_budget_combined.py        # Cp and Cf data-efficiency side by side
├── plot_airfoil_shapes.py         # Airfoil shape variations in the dataset
├── generate_table1.py             # LaTeX table of input parameter bounds
└── *.png, *.pdf, *.tex            # Generated figures and table used in the paper
```

RANS data is from the AASM Benchmark Case 3 (Bekemeyer et al., 2025). Dependencies are listed in `requirements.txt`; the FNO scripts additionally require `neuraloperator`.
