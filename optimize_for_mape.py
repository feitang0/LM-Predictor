#!/usr/bin/env python3
"""
Find k and b that achieve target MAPE, showing the trade-off with R².
"""

import argparse
import re
import numpy as np
from scipy import stats
from scipy.optimize import minimize


def parse_real_run_file(filepath):
    """Parse real-run result file and extract average of rounds 1-4."""
    results = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()

    current_phase = None
    current_batch = None
    current_seqlen = None
    round_times = []

    for line in lines:
        line = line.strip()

        if 'Testing Prefill Phase' in line:
            current_phase = 'prefill'
            continue
        elif 'Testing Decode Phase' in line:
            if current_seqlen is not None and len(round_times) == 5:
                avg_time = np.mean(round_times[1:]) * 1000
                key = (current_phase, current_batch, current_seqlen)
                results[key] = avg_time
                round_times = []
                current_seqlen = None
            current_phase = 'decode'
            continue

        batch_match = re.match(r'Batch Size: (\d+)', line)
        if batch_match:
            if current_seqlen is not None and len(round_times) == 5:
                avg_time = np.mean(round_times[1:]) * 1000
                key = (current_phase, current_batch, current_seqlen)
                results[key] = avg_time
                round_times = []
                current_seqlen = None
            current_batch = int(batch_match.group(1))
            continue

        if current_phase == 'prefill':
            seqlen_match = re.match(r'Seq Length: (\d+)', line)
        else:
            seqlen_match = re.match(r'Cache Length: (\d+)', line)

        if seqlen_match:
            if current_seqlen is not None and len(round_times) == 5:
                avg_time = np.mean(round_times[1:]) * 1000
                key = (current_phase, current_batch, current_seqlen)
                results[key] = avg_time
            current_seqlen = int(seqlen_match.group(1))
            round_times = []
            continue

        round_match = re.match(r'Round (\d+): ([\d.]+) seconds', line)
        if round_match:
            time_sec = float(round_match.group(2))
            round_times.append(time_sec)
            continue

    if current_seqlen is not None and len(round_times) == 5:
        avg_time = np.mean(round_times[1:]) * 1000
        key = (current_phase, current_batch, current_seqlen)
        results[key] = avg_time

    return results


def parse_prediction_file(filepath):
    """Parse prediction result file."""
    results = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()

    current_phase = None
    current_batch = None
    current_seqlen = None

    for line in lines:
        line = line.strip()

        if 'Predicting Prefill Phase' in line:
            current_phase = 'prefill'
            continue
        elif 'Predicting Decode Phase' in line:
            current_phase = 'decode'
            continue

        batch_match = re.match(r'Batch Size: (\d+)', line)
        if batch_match:
            current_batch = int(batch_match.group(1))
            continue

        if current_phase == 'prefill':
            seqlen_match = re.match(r'Seq Length: (\d+)', line)
        else:
            seqlen_match = re.match(r'Cache Length: (\d+)', line)

        if seqlen_match:
            current_seqlen = int(seqlen_match.group(1))
            continue

        time_match = re.match(r'Total time: ([\d.]+)µs', line)
        if time_match:
            time_us = float(time_match.group(1))
            time_ms = time_us / 1000
            if current_batch is not None and current_seqlen is not None:
                key = (current_phase, current_batch, current_seqlen)
                results[key] = time_ms

    return results


def calculate_metrics(x, y, k, b):
    """Calculate R² and MAPE for given k, b"""
    y_pred = k * x + b
    residuals = y - y_pred

    # R²
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot

    # MAPE
    mape = np.mean(np.abs(residuals / y)) * 100

    # MAE
    mae = np.mean(np.abs(residuals))

    return r2, mape, mae


def fit_ols(x, y):
    """Standard OLS"""
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    return slope, intercept, r_value**2


def fit_weighted_ols(x, y):
    """Weighted OLS (minimizes relative errors)"""
    weights = 1.0 / x
    W = np.diag(weights)
    X = np.column_stack([x, np.ones_like(x)])
    Y = y
    coeffs = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ Y, rcond=None)[0]
    k, b = coeffs[0], coeffs[1]
    return k, b


def optimize_for_mape(x, y):
    """Directly optimize to minimize MAPE"""
    def objective(params):
        k, b = params
        y_pred = k * x + b
        mape = np.mean(np.abs((y - y_pred) / y))
        return mape

    # Initial guess from OLS
    k_init, b_init, _ = fit_ols(x, y)
    result = minimize(objective, [k_init, b_init], method='Nelder-Mead')
    k_opt, b_opt = result.x
    return k_opt, b_opt


def main():
    parser = argparse.ArgumentParser(
        description='Find k and b that minimize MAPE (target ≤ 30%)'
    )
    parser.add_argument('real_file', type=str, help='Real-run result file')
    parser.add_argument('pred_file', type=str, help='Prediction result file')
    parser.add_argument('--target-mape', type=float, default=30.0,
                        help='Target MAPE percentage (default: 30%%)')
    args = parser.parse_args()

    # Parse files
    print(f"Reading data files...")
    real_data = parse_real_run_file(args.real_file)
    pred_data = parse_prediction_file(args.pred_file)

    # Process each phase
    for phase_name in ['prefill', 'decode']:
        pred = []
        real = []

        for key in real_data:
            phase, batch, seqlen = key
            if phase == phase_name and key in pred_data:
                pred.append(pred_data[key])
                real.append(real_data[key])

        pred = np.array(pred)
        real = np.array(real)

        print("\n" + "="*80)
        print(f"{phase_name.upper()} PHASE - Finding Best k, b for MAPE ≤ {args.target_mape}%")
        print("="*80)

        # Method 1: OLS (minimizes squared errors)
        k_ols, b_ols, r2_ols_fit = fit_ols(pred, real)
        r2_ols, mape_ols, mae_ols = calculate_metrics(pred, real, k_ols, b_ols)

        print(f"\nMethod 1: OLS (Ordinary Least Squares)")
        print(f"  Formula:  y = {k_ols:.4f}x + {b_ols:.4f}")
        print(f"  R²:       {r2_ols:.6f}")
        print(f"  MAPE:     {mape_ols:.2f}%")
        print(f"  MAE:      {mae_ols:.4f} ms")
        if mape_ols <= args.target_mape:
            print(f"  ✓ Meets target (MAPE ≤ {args.target_mape}%)")
        else:
            print(f"  ✗ Does not meet target (MAPE > {args.target_mape}%)")

        # Method 2: Weighted OLS (emphasizes relative errors)
        k_wols, b_wols = fit_weighted_ols(pred, real)
        r2_wols, mape_wols, mae_wols = calculate_metrics(pred, real, k_wols, b_wols)

        print(f"\nMethod 2: Weighted OLS (weights = 1/x)")
        print(f"  Formula:  y = {k_wols:.4f}x + {b_wols:.4f}")
        print(f"  R²:       {r2_wols:.6f}")
        print(f"  MAPE:     {mape_wols:.2f}%")
        print(f"  MAE:      {mae_wols:.4f} ms")
        if mape_wols <= args.target_mape:
            print(f"  ✓ Meets target (MAPE ≤ {args.target_mape}%)")
        else:
            print(f"  ✗ Does not meet target (MAPE > {args.target_mape}%)")

        # Method 3: Direct MAPE optimization
        k_mape, b_mape = optimize_for_mape(pred, real)
        r2_mape, mape_mape_calc, mae_mape = calculate_metrics(pred, real, k_mape, b_mape)

        print(f"\nMethod 3: Direct MAPE Optimization")
        print(f"  Formula:  y = {k_mape:.4f}x + {b_mape:.4f}")
        print(f"  R²:       {r2_mape:.6f}")
        print(f"  MAPE:     {mape_mape_calc:.2f}%")
        print(f"  MAE:      {mae_mape:.4f} ms")
        if mape_mape_calc <= args.target_mape:
            print(f"  ✓ Meets target (MAPE ≤ {args.target_mape}%)")
        else:
            print(f"  ✗ Does not meet target (MAPE > {args.target_mape}%)")

        # Comparison table
        print(f"\n{'Method':<25} {'R²':<12} {'MAPE (%)':<12} {'MAE (ms)':<12} {'Meets Target':<15}")
        print("-"*80)
        print(f"{'OLS':<25} {r2_ols:<12.6f} {mape_ols:<12.2f} {mae_ols:<12.4f} "
              f"{'✓' if mape_ols <= args.target_mape else '✗':<15}")
        print(f"{'Weighted OLS':<25} {r2_wols:<12.6f} {mape_wols:<12.2f} {mae_wols:<12.4f} "
              f"{'✓' if mape_wols <= args.target_mape else '✗':<15}")
        print(f"{'Direct MAPE Optim':<25} {r2_mape:<12.6f} {mape_mape_calc:<12.2f} {mae_mape:<12.4f} "
              f"{'✓' if mape_mape_calc <= args.target_mape else '✗':<15}")

        # Recommendation
        print(f"\nRecommendation:")
        best_method = min(
            [('OLS', mape_ols, r2_ols, k_ols, b_ols),
             ('Weighted OLS', mape_wols, r2_wols, k_wols, b_wols),
             ('Direct MAPE', mape_mape_calc, r2_mape, k_mape, b_mape)],
            key=lambda x: x[1]  # Sort by MAPE
        )

        method_name, best_mape, best_r2, best_k, best_b = best_method

        if best_mape <= args.target_mape:
            print(f"  Use {method_name}: y = {best_k:.4f}x + {best_b:.4f}")
            print(f"    MAPE = {best_mape:.2f}% (target: ≤{args.target_mape}%)")
            print(f"    R² = {best_r2:.6f}")
            print(f"  ✓ Achieves target with best MAPE!")
        else:
            print(f"  Best available: {method_name} with MAPE = {best_mape:.2f}%")
            print(f"  ⚠ Cannot achieve target MAPE ≤ {args.target_mape}% with linear model")
            print(f"  Consider: (1) Using the best available, or (2) Non-linear calibration")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nFor MAPE ≤ {args.target_mape}%:")
    print("  • Weighted OLS typically provides the best balance")
    print("  • It reduces MAPE significantly while maintaining reasonable R²")
    print("  • Trade-off: Slightly lower R² but much better percentage accuracy")


if __name__ == '__main__':
    main()
