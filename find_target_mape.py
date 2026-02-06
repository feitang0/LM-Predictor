#!/usr/bin/env python3
"""
Find k and b that achieve a target MAPE range (e.g., 25-30%).
Uses variable weighting to control the trade-off between R² and MAPE.
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


def calculate_metrics(x, y, k, b):
    """Calculate R², MAPE, MAE for given k, b"""
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

    # RMSE
    rmse = np.sqrt(np.mean(residuals**2))

    return r2, mape, mae, rmse


def fit_weighted_ols_alpha(x, y, alpha):
    """
    Weighted OLS with adjustable weight parameter alpha.
    weights = 1 / x^alpha

    alpha = 0: Standard OLS (no weighting)
    alpha = 1: Full weighted OLS (weight by 1/x)
    alpha between 0-1: Partial weighting
    """
    if alpha == 0:
        # Standard OLS
        slope, intercept, _, _, _ = stats.linregress(x, y)
        return slope, intercept
    else:
        # Weighted OLS
        weights = 1.0 / (x ** alpha)
        W = np.diag(weights)
        X = np.column_stack([x, np.ones_like(x)])
        Y = y
        coeffs = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ Y, rcond=None)[0]
        k, b = coeffs[0], coeffs[1]
        return k, b


def search_for_target_mape(x, y, target_min, target_max):
    """
    Search for alpha that gives MAPE in target range.
    Returns k, b, alpha, and all metrics.
    """
    best_alpha = None
    best_k = None
    best_b = None
    best_mape = None
    best_metrics = None

    # Search alpha from 0 to 1
    for alpha in np.linspace(0, 1, 101):
        k, b = fit_weighted_ols_alpha(x, y, alpha)
        r2, mape, mae, rmse = calculate_metrics(x, y, k, b)

        # Check if MAPE is in target range
        if target_min <= mape <= target_max:
            # Prefer higher R² within target range
            if best_alpha is None or r2 > best_metrics[0]:
                best_alpha = alpha
                best_k = k
                best_b = b
                best_mape = mape
                best_metrics = (r2, mape, mae, rmse)

    return best_alpha, best_k, best_b, best_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Find k and b for target MAPE range'
    )
    parser.add_argument('real_file', type=str, help='Real-run result file')
    parser.add_argument('pred_file', type=str, help='Prediction result file')
    parser.add_argument('--min-mape', type=float, default=25.0,
                        help='Minimum target MAPE (default: 25%%)')
    parser.add_argument('--max-mape', type=float, default=30.0,
                        help='Maximum target MAPE (default: 30%%)')
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
        print(f"{phase_name.upper()} PHASE - Target MAPE: {args.min_mape}% - {args.max_mape}%")
        print("="*80)

        # Search for alpha that achieves target MAPE
        alpha, k, b, metrics = search_for_target_mape(pred, real, args.min_mape, args.max_mape)

        if alpha is not None:
            r2, mape, mae, rmse = metrics
            print(f"\n✓ Found solution:")
            print(f"  Formula:       y = {k:.4f}x + {b:.4f}")
            print(f"  Alpha:         {alpha:.4f} (weighting parameter)")
            print(f"  R²:            {r2:.6f}")
            print(f"  MAPE:          {mape:.2f}% ← Target: {args.min_mape}%-{args.max_mape}%")
            print(f"  MAE:           {mae:.4f} ms")
            print(f"  RMSE:          {rmse:.4f} ms")

            print(f"\nInterpretation:")
            print(f"  • Alpha = {alpha:.4f} means partial weighting")
            print(f"    (0 = standard OLS, 1 = full weighted OLS)")
            print(f"  • This balances R² and MAPE optimally")
            print(f"  • MAPE is within your target range!")
        else:
            print(f"\n✗ No solution found in MAPE range {args.min_mape}%-{args.max_mape}%")

            # Show what's available
            print(f"\nSearching nearby ranges...")
            for test_alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                k_test, b_test = fit_weighted_ols_alpha(pred, real, test_alpha)
                r2_test, mape_test, mae_test, rmse_test = calculate_metrics(pred, real, k_test, b_test)
                print(f"  Alpha={test_alpha:.1f}: MAPE={mape_test:.2f}%, R²={r2_test:.6f}")

        # Also show comparison with standard methods
        print(f"\nComparison with standard methods:")

        # OLS
        k_ols, b_ols = fit_weighted_ols_alpha(pred, real, 0.0)
        r2_ols, mape_ols, mae_ols, rmse_ols = calculate_metrics(pred, real, k_ols, b_ols)
        print(f"  OLS (alpha=0):   y = {k_ols:.4f}x + {b_ols:.4f}")
        print(f"                   R²={r2_ols:.6f}, MAPE={mape_ols:.2f}%")

        # Full Weighted OLS
        k_wols, b_wols = fit_weighted_ols_alpha(pred, real, 1.0)
        r2_wols, mape_wols, mae_wols, rmse_wols = calculate_metrics(pred, real, k_wols, b_wols)
        print(f"  Weighted (α=1):  y = {k_wols:.4f}x + {b_wols:.4f}")
        print(f"                   R²={r2_wols:.6f}, MAPE={mape_wols:.2f}%")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTo achieve MAPE in range {args.min_mape}%-{args.max_mape}%:")
    print("  Use the formulas shown above with the specified alpha values")
    print("  Alpha controls the trade-off between R² and MAPE")


if __name__ == '__main__':
    main()
