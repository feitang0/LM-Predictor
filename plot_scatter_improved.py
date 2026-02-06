#!/usr/bin/env python3
"""
Scatter plot: Prediction (x-axis) vs Real-run (y-axis) with different regression methods.
"""

import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit


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


def fit_linear_ols(x, y):
    """Method 1: Ordinary Least Squares on original data"""
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    return slope, intercept, r_value**2


def fit_linear_log(x, y):
    """Method 2: OLS on log-transformed data (for log-log plot)"""
    log_x = np.log(x)
    log_y = np.log(y)
    slope_log, intercept_log, r_value, _, _ = stats.linregress(log_x, log_y)
    # Transform back: log(y) = slope*log(x) + intercept => y = exp(intercept) * x^slope
    # For linear form y = kx + b, this doesn't directly apply, so we fit y = a*x^p instead
    return slope_log, intercept_log, r_value**2


def fit_weighted_ols(x, y):
    """Method 3: Weighted OLS (weight by 1/x to emphasize relative errors)"""
    weights = 1.0 / x
    # Weighted least squares: minimize sum(w_i * (y_i - (k*x_i + b))^2)
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


def fit_log_linear(x, y):
    """Method 4: Fit y = a * x^p (power law) then approximate as linear"""
    def power_law(x, a, p):
        return a * x**p

    popt, _ = curve_fit(power_law, x, y, p0=[1, 1])
    a, p = popt

    # For comparison, convert to linear at midpoint
    x_mid = np.median(x)
    # At x_mid: y = a * x_mid^p
    # Derivative: dy/dx = a * p * x_mid^(p-1)
    k = a * p * x_mid**(p-1)
    b = a * x_mid**p - k * x_mid

    # Calculate R²
    y_pred = a * x**p
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot

    return k, b, r2, a, p


def plot_scatter(real_data, pred_data, output_file, method='ols'):
    """Create scatter plots with specified regression method"""

    # Prepare data for prefill
    prefill_pred = []
    prefill_real = []
    prefill_batch = []

    for key in real_data:
        phase, batch, seqlen = key
        if phase == 'prefill' and key in pred_data:
            prefill_pred.append(pred_data[key])
            prefill_real.append(real_data[key])
            prefill_batch.append(batch)

    prefill_pred = np.array(prefill_pred)
    prefill_real = np.array(prefill_real)
    prefill_batch = np.array(prefill_batch)

    # Prepare data for decode
    decode_pred = []
    decode_real = []
    decode_batch = []

    for key in real_data:
        phase, batch, seqlen = key
        if phase == 'decode' and key in pred_data:
            decode_pred.append(pred_data[key])
            decode_real.append(real_data[key])
            decode_batch.append(batch)

    decode_pred = np.array(decode_pred)
    decode_real = np.array(decode_real)
    decode_batch = np.array(decode_batch)

    # Get unique batch sizes for coloring
    unique_batches = sorted(set(prefill_batch))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(unique_batches)))
    batch_to_color = {batch: colors[i] for i, batch in enumerate(unique_batches)}

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # ===== PREFILL SCATTER PLOT =====
    for batch in unique_batches:
        mask = prefill_batch == batch
        ax1.scatter(prefill_pred[mask], prefill_real[mask],
                   c=[batch_to_color[batch]], s=100, alpha=0.7,
                   label=f'BS {batch}', edgecolors='black', linewidth=0.5)

    # Fit using selected method
    if method == 'ols':
        k_p, b_p, r2_p = fit_linear_ols(prefill_pred, prefill_real)
        method_name = 'OLS (Original Data)'
    elif method == 'weighted':
        k_p, b_p, r2_p = fit_weighted_ols(prefill_pred, prefill_real)
        method_name = 'Weighted OLS'
    elif method == 'power':
        k_p, b_p, r2_p, a_p, p_p = fit_log_linear(prefill_pred, prefill_real)
        method_name = f'Power Law (y={a_p:.2f}x^{p_p:.3f})'

    # Plot fit line
    x_range = np.logspace(np.log10(prefill_pred.min()), np.log10(prefill_pred.max()), 100)
    if method == 'power':
        y_fit = a_p * x_range**p_p
        label_text = f'{method_name}\nLinear approx: y={k_p:.2f}x+{b_p:.2f}\nR²={r2_p:.4f}'
    else:
        y_fit = k_p * x_range + b_p
        label_text = f'{method_name}\ny={k_p:.2f}x+{b_p:.2f}\nR²={r2_p:.4f}'

    ax1.plot(x_range, y_fit, 'b-', linewidth=3, label=label_text, zorder=9, alpha=0.8)

    ax1.set_xlabel('Prediction Time (ms)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Real-Run Time (ms)', fontsize=14, fontweight='bold')
    ax1.set_title('PREFILL Phase', fontsize=16, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(loc='upper left', fontsize=10)

    # ===== DECODE SCATTER PLOT =====
    colors_decode = plt.cm.plasma(np.linspace(0.1, 0.9, len(unique_batches)))
    batch_to_color_decode = {batch: colors_decode[i] for i, batch in enumerate(unique_batches)}

    for batch in unique_batches:
        mask = decode_batch == batch
        ax2.scatter(decode_pred[mask], decode_real[mask],
                   c=[batch_to_color_decode[batch]], s=100, alpha=0.7,
                   label=f'BS {batch}', edgecolors='black', linewidth=0.5)

    # Fit using selected method
    if method == 'ols':
        k_d, b_d, r2_d = fit_linear_ols(decode_pred, decode_real)
    elif method == 'weighted':
        k_d, b_d, r2_d = fit_weighted_ols(decode_pred, decode_real)
    elif method == 'power':
        k_d, b_d, r2_d, a_d, p_d = fit_log_linear(decode_pred, decode_real)

    # Plot fit line
    x_range_dec = np.logspace(np.log10(decode_pred.min()), np.log10(decode_pred.max()), 100)
    if method == 'power':
        y_fit_dec = a_d * x_range_dec**p_d
        label_text_dec = f'{method_name}\nLinear approx: y={k_d:.2f}x+{b_d:.2f}\nR²={r2_d:.4f}'
    else:
        y_fit_dec = k_d * x_range_dec + b_d
        label_text_dec = f'{method_name}\ny={k_d:.2f}x+{b_d:.2f}\nR²={r2_d:.4f}'

    ax2.plot(x_range_dec, y_fit_dec, 'b-', linewidth=3, label=label_text_dec, zorder=9, alpha=0.8)

    ax2.set_xlabel('Prediction Time (ms)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Real-Run Time (ms)', fontsize=14, fontweight='bold')
    ax2.set_title('DECODE Phase', fontsize=16, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(loc='upper left', fontsize=10)

    # Overall title
    fig.suptitle(f'Prediction vs Real-Run Time ({method_name})',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")

    return k_p, b_p, r2_p, k_d, b_d, r2_d


def main():
    parser = argparse.ArgumentParser(
        description='Create scatter plots with different regression methods'
    )
    parser.add_argument('real_file', type=str, help='Real-run result file')
    parser.add_argument('pred_file', type=str, help='Prediction result file')
    parser.add_argument('--method', '-m', type=str, default='all',
                        choices=['ols', 'weighted', 'power', 'all'],
                        help='Regression method: ols (default), weighted, power, or all')
    parser.add_argument('--output-prefix', '-o', type=str, default='scatter',
                        help='Output filename prefix')

    args = parser.parse_args()

    # Parse files
    print(f"Reading real-run results from: {args.real_file}")
    real_data = parse_real_run_file(args.real_file)
    print(f"  → Found {len(real_data)} data points")

    print(f"Reading predictions from: {args.pred_file}")
    pred_data = parse_prediction_file(args.pred_file)
    print(f"  → Found {len(pred_data)} data points")

    # Generate plots
    methods = ['ols', 'weighted', 'power'] if args.method == 'all' else [args.method]

    print("\n" + "="*80)
    print("REGRESSION METHOD COMPARISON")
    print("="*80)

    results = {}
    for method in methods:
        print(f"\nMethod: {method.upper()}")
        output = f"{args.output_prefix}_{method}.png"
        k_p, b_p, r2_p, k_d, b_d, r2_d = plot_scatter(real_data, pred_data, output, method)
        results[method] = {
            'prefill': (k_p, b_p, r2_p),
            'decode': (k_d, b_d, r2_d)
        }

        print(f"  PREFILL: y = {k_p:.4f}x + {b_p:.4f} (R² = {r2_p:.6f})")
        print(f"  DECODE:  y = {k_d:.4f}x + {b_d:.4f} (R² = {r2_d:.6f})")

    # Print comparison
    if len(methods) > 1:
        print("\n" + "="*80)
        print("METHOD COMPARISON SUMMARY")
        print("="*80)
        print("\n1. OLS (Ordinary Least Squares): Minimizes sum of squared errors")
        print("2. Weighted OLS: Emphasizes relative errors (better for log-scale data)")
        print("3. Power Law: Fits y = a*x^p (best for log-log relationships)")
        print("\nRecommendation: Choose method with highest R² value")

        for phase in ['prefill', 'decode']:
            print(f"\n{phase.upper()} - Best method:")
            best_method = max(results.keys(), key=lambda m: results[m][phase][2])
            k, b, r2 = results[best_method][phase]
            print(f"  {best_method.upper()}: y = {k:.4f}x + {b:.4f} (R² = {r2:.6f})")


if __name__ == '__main__':
    main()
