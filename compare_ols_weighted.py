#!/usr/bin/env python3
"""
Compare OLS vs Weighted OLS: Which gives better MAPE?
"""

import argparse
import re
import numpy as np
from scipy import stats


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


def fit_ols(x, y):
    """Standard OLS regression"""
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    return slope, intercept, r_value**2


def fit_weighted_ols(x, y):
    """Weighted OLS (weight by 1/x to emphasize relative errors)"""
    weights = 1.0 / x
    W = np.diag(weights)
    X = np.column_stack([x, np.ones_like(x)])
    Y = y
    # Solution: (X^T W X)^-1 X^T W Y
    coeffs = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ Y, rcond=None)[0]
    k, b = coeffs[0], coeffs[1]

    # Calculate R²
    y_pred = k * x + b
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot

    return k, b, r2


def calculate_metrics(x, y_real, k, b):
    """Calculate all metrics"""
    y_pred = k * x + b
    residuals = y_real - y_pred

    r2 = 1 - np.sum(residuals**2) / np.sum((y_real - np.mean(y_real))**2)
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    mape = np.mean(np.abs(residuals / y_real)) * 100

    return {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'y_pred': y_pred,
        'residuals': residuals
    }


def main():
    parser = argparse.ArgumentParser(description='Compare OLS vs Weighted OLS')
    parser.add_argument('real_file', type=str, help='Real-run result file')
    parser.add_argument('pred_file', type=str, help='Prediction result file')
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

        # Fit both methods
        k_ols, b_ols, r2_ols = fit_ols(pred, real)
        k_wols, b_wols, r2_wols = fit_weighted_ols(pred, real)

        # Calculate metrics for both
        metrics_ols = calculate_metrics(pred, real, k_ols, b_ols)
        metrics_wols = calculate_metrics(pred, real, k_wols, b_wols)

        # Print comparison
        print("\n" + "="*80)
        print(f"{phase_name.upper()} PHASE - OLS vs Weighted OLS Comparison")
        print("="*80)

        print(f"\nOrdinary Least Squares (OLS):")
        print(f"  Formula:  y = {k_ols:.4f}x + {b_ols:.4f}")
        print(f"  R²:       {metrics_ols['r2']:.6f}")
        print(f"  MAE:      {metrics_ols['mae']:.4f} ms")
        print(f"  RMSE:     {metrics_ols['rmse']:.4f} ms")
        print(f"  MAPE:     {metrics_ols['mape']:.2f}%")

        print(f"\nWeighted OLS (weights = 1/x):")
        print(f"  Formula:  y = {k_wols:.4f}x + {b_wols:.4f}")
        print(f"  R²:       {metrics_wols['r2']:.6f}")
        print(f"  MAE:      {metrics_wols['mae']:.4f} ms")
        print(f"  RMSE:     {metrics_wols['rmse']:.4f} ms")
        print(f"  MAPE:     {metrics_wols['mape']:.2f}%")

        print(f"\nComparison:")
        print(f"  {'Metric':<15} {'OLS':<15} {'Weighted OLS':<15} {'Better':<15}")
        print(f"  {'-'*60}")

        # R²
        better_r2 = "OLS" if metrics_ols['r2'] > metrics_wols['r2'] else "Weighted OLS"
        print(f"  {'R²':<15} {metrics_ols['r2']:<15.6f} {metrics_wols['r2']:<15.6f} {better_r2:<15}")

        # MAE
        better_mae = "OLS" if metrics_ols['mae'] < metrics_wols['mae'] else "Weighted OLS"
        print(f"  {'MAE (ms)':<15} {metrics_ols['mae']:<15.4f} {metrics_wols['mae']:<15.4f} {better_mae:<15}")

        # RMSE
        better_rmse = "OLS" if metrics_ols['rmse'] < metrics_wols['rmse'] else "Weighted OLS"
        print(f"  {'RMSE (ms)':<15} {metrics_ols['rmse']:<15.4f} {metrics_wols['rmse']:<15.4f} {better_rmse:<15}")

        # MAPE - lower is better!
        better_mape = "OLS" if metrics_ols['mape'] < metrics_wols['mape'] else "Weighted OLS"
        mape_diff = abs(metrics_ols['mape'] - metrics_wols['mape'])
        print(f"  {'MAPE (%)':<15} {metrics_ols['mape']:<15.2f} {metrics_wols['mape']:<15.2f} {better_mape:<15}")

        if metrics_wols['mape'] < metrics_ols['mape']:
            improvement = (metrics_ols['mape'] - metrics_wols['mape']) / metrics_ols['mape'] * 100
            print(f"\n  ✓ Weighted OLS improves MAPE by {improvement:.1f}%!")
        else:
            degradation = (metrics_wols['mape'] - metrics_ols['mape']) / metrics_ols['mape'] * 100
            print(f"\n  ✗ Weighted OLS worsens MAPE by {degradation:.1f}%")

        print(f"\nInterpretation:")
        if metrics_wols['mape'] < metrics_ols['mape']:
            print(f"  Weighted OLS reduces MAPE by emphasizing relative errors.")
            print(f"  Trade-off: Slightly lower R² and higher absolute errors (MAE, RMSE).")
            print(f"  Use Weighted OLS if you care more about percentage accuracy.")
        else:
            print(f"  OLS is better for this phase - already handles the data well.")
            print(f"  Weighting doesn't help because the data range is relatively narrow.")

    # Overall recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("\nThe relationship between R² and MAPE:")
    print("  • R² measures how well you capture the PATTERN")
    print("  • MAPE measures RELATIVE (percentage) accuracy")
    print("\nThey can disagree when data spans wide ranges:")
    print("  • High R² + High MAPE = Good pattern fit, but small values have large % errors")
    print("  • Weighted OLS trades some R² for better MAPE")
    print("\nChoose based on your priority:")
    print("  • Want best pattern fit & absolute errors → Use OLS")
    print("  • Want best percentage accuracy → Use Weighted OLS")


if __name__ == '__main__':
    main()
