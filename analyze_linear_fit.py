#!/usr/bin/env python3
"""
Linear regression analysis to calibrate theoretical predictions with actual measurements.
Model: y = k*x + b
where:
  x = theoretical prediction
  y = actual measured time
  k = efficiency scaling factor
  b = base overhead
"""

import argparse
import re
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path


def parse_real_run_file(filepath):
    """
    Parse real-run result file and extract average of rounds 1-4.
    Returns dict: {('prefill'|'decode', batch_size, seq_len): avg_time_ms}
    """
    results = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    current_phase = None
    current_batch = None
    current_seqlen = None
    round_times = []

    for line in lines:
        line = line.strip()

        # Detect phase
        if 'Testing Prefill Phase' in line:
            current_phase = 'prefill'
            continue
        elif 'Testing Decode Phase' in line:
            # Save pending data before phase change
            if current_seqlen is not None and len(round_times) == 5:
                avg_time = np.mean(round_times[1:]) * 1000
                key = (current_phase, current_batch, current_seqlen)
                results[key] = avg_time
                round_times = []
                current_seqlen = None
            current_phase = 'decode'
            continue

        # Parse batch size
        batch_match = re.match(r'Batch Size: (\d+)', line)
        if batch_match:
            # Save previous data before changing batch size
            if current_seqlen is not None and len(round_times) == 5:
                avg_time = np.mean(round_times[1:]) * 1000
                key = (current_phase, current_batch, current_seqlen)
                results[key] = avg_time
                round_times = []
                current_seqlen = None
            current_batch = int(batch_match.group(1))
            continue

        # Parse sequence/cache length
        if current_phase == 'prefill':
            seqlen_match = re.match(r'Seq Length: (\d+)', line)
        else:  # decode
            seqlen_match = re.match(r'Cache Length: (\d+)', line)

        if seqlen_match:
            # Save previous data if exists
            if current_seqlen is not None and len(round_times) == 5:
                # Drop round 0, average rounds 1-4
                avg_time = np.mean(round_times[1:]) * 1000  # convert to ms
                key = (current_phase, current_batch, current_seqlen)
                results[key] = avg_time

            current_seqlen = int(seqlen_match.group(1))
            round_times = []
            continue

        # Parse round times
        round_match = re.match(r'Round (\d+): ([\d.]+) seconds', line)
        if round_match:
            round_num = int(round_match.group(1))
            time_sec = float(round_match.group(2))
            round_times.append(time_sec)
            continue

    # Handle last entry
    if current_seqlen is not None and len(round_times) == 5:
        avg_time = np.mean(round_times[1:]) * 1000
        key = (current_phase, current_batch, current_seqlen)
        results[key] = avg_time

    return results


def parse_prediction_file(filepath):
    """
    Parse prediction result file.
    Returns dict: {('prefill'|'decode', batch_size, seq_len): prediction_time_ms}
    """
    results = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    current_phase = None
    current_batch = None
    current_seqlen = None

    for line in lines:
        line = line.strip()

        # Detect phase
        if 'Predicting Prefill Phase' in line:
            current_phase = 'prefill'
            continue
        elif 'Predicting Decode Phase' in line:
            current_phase = 'decode'
            continue

        # Parse batch size
        batch_match = re.match(r'Batch Size: (\d+)', line)
        if batch_match:
            current_batch = int(batch_match.group(1))
            continue

        # Parse sequence/cache length
        if current_phase == 'prefill':
            seqlen_match = re.match(r'Seq Length: (\d+)', line)
        else:  # decode
            seqlen_match = re.match(r'Cache Length: (\d+)', line)

        if seqlen_match:
            current_seqlen = int(seqlen_match.group(1))
            continue

        # Parse total time
        time_match = re.match(r'Total time: ([\d.]+)µs', line)
        if time_match:
            time_us = float(time_match.group(1))
            time_ms = time_us / 1000
            # Store result using current context
            if current_batch is not None and current_seqlen is not None:
                key = (current_phase, current_batch, current_seqlen)
                results[key] = time_ms

    return results


def linear_regression(x, y):
    """Perform linear regression: y = k*x + b"""
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, intercept, r_value**2


def analyze_data(real_data, pred_data, phase_filter=None):
    """
    Perform linear regression analysis.

    Args:
        real_data: dict of actual measurements
        pred_data: dict of predictions
        phase_filter: 'prefill', 'decode', or None for both

    Returns:
        k, b, r2, x, y, y_pred
    """
    x_list = []
    y_list = []
    configs = []

    for key in real_data:
        phase, batch, seqlen = key

        # Apply phase filter
        if phase_filter and phase != phase_filter:
            continue

        if key in pred_data:
            x_list.append(pred_data[key])  # prediction (ms)
            y_list.append(real_data[key])  # actual (ms)
            configs.append(key)

    x = np.array(x_list)
    y = np.array(y_list)

    # Perform linear regression
    k, b, r2 = linear_regression(x, y)

    # Calculate predictions
    y_pred = k * x + b

    return k, b, r2, x, y, y_pred, configs


def print_analysis(phase_name, k, b, r2, x, y, y_pred, configs):
    """Print detailed analysis results"""
    residuals = y - y_pred
    mae = np.mean(np.abs(residuals))
    mape = np.mean(np.abs(residuals / y)) * 100
    rmse = np.sqrt(np.mean(residuals**2))

    print(f"\n{'='*70}")
    print(f"{phase_name} - Linear Model Fit")
    print(f"{'='*70}")
    print(f"Model: y = k*x + b")
    print(f"  k (efficiency factor) = {k:.4f}")
    print(f"  b (base overhead)     = {b:.4f} ms")
    print(f"\nFit Quality:")
    print(f"  R² (goodness of fit)  = {r2:.6f}")
    print(f"  MAE (mean abs error)  = {mae:.4f} ms")
    print(f"  MAPE (mean % error)   = {mape:.2f}%")
    print(f"  RMSE (root mean sq)   = {rmse:.4f} ms")

    print(f"\nInterpretation:")
    print(f"  • Reality is {k:.2f}× slower than theoretical prediction")
    print(f"  • Fixed overhead per operation: {b:.4f} ms")
    print(f"  • Model explains {r2*100:.2f}% of variance in measurements")

    # Show sample predictions
    print(f"\nSample Predictions vs Actual:")
    print(f"{'Config':<20} {'Theory(ms)':<12} {'Actual(ms)':<12} {'Predicted(ms)':<15} {'Error(%)':<10}")
    print("-" * 75)

    # Show every 5th sample or key samples
    sample_indices = list(range(0, len(configs), max(1, len(configs) // 10)))
    for i in sample_indices[:10]:
        phase, batch, seqlen = configs[i]
        config_name = f"{phase[:4].upper()}/BS{batch}/SL{seqlen}"
        theory = x[i]
        actual = y[i]
        pred = y_pred[i]
        error = abs(actual - pred) / actual * 100
        print(f"{config_name:<20} {theory:<12.4f} {actual:<12.4f} {pred:<15.4f} {error:<10.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and calibrate model predictions with actual measurements'
    )
    parser.add_argument('real_file', type=str, help='Real-run result file')
    parser.add_argument('pred_file', type=str, help='Prediction result file')
    parser.add_argument('--output', '-o', type=str, default='linear_fit_analysis.png',
                        help='Output plot filename')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')

    args = parser.parse_args()

    # Parse files
    print(f"Reading real-run results from: {args.real_file}")
    real_data = parse_real_run_file(args.real_file)
    print(f"  → Found {len(real_data)} data points")

    print(f"Reading predictions from: {args.pred_file}")
    pred_data = parse_prediction_file(args.pred_file)
    print(f"  → Found {len(pred_data)} data points")

    # Analyze prefill phase
    k_prefill, b_prefill, r2_prefill, x_prefill, y_prefill, pred_prefill, configs_prefill = \
        analyze_data(real_data, pred_data, phase_filter='prefill')
    print_analysis("PREFILL PHASE", k_prefill, b_prefill, r2_prefill,
                   x_prefill, y_prefill, pred_prefill, configs_prefill)

    # Analyze decode phase
    k_decode, b_decode, r2_decode, x_decode, y_decode, pred_decode, configs_decode = \
        analyze_data(real_data, pred_data, phase_filter='decode')
    print_analysis("DECODE PHASE", k_decode, b_decode, r2_decode,
                   x_decode, y_decode, pred_decode, configs_decode)

    # Combined analysis
    k_combined, b_combined, r2_combined, x_combined, y_combined, pred_combined, configs_combined = \
        analyze_data(real_data, pred_data, phase_filter=None)
    print_analysis("COMBINED (BOTH PHASES)", k_combined, b_combined, r2_combined,
                   x_combined, y_combined, pred_combined, configs_combined)

    # Summary
    print(f"\n{'='*70}")
    print(f"CALIBRATION FORMULAS")
    print(f"{'='*70}")
    print(f"Use these formulas to calibrate future predictions:")
    print(f"  Prefill: actual_time = {k_prefill:.4f} × theoretical + {b_prefill:.4f} ms")
    print(f"  Decode:  actual_time = {k_decode:.4f} × theoretical + {b_decode:.4f} ms")
    print(f"  Combined: actual_time = {k_combined:.4f} × theoretical + {b_combined:.4f} ms")

    # Create visualization
    if not args.no_plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Prefill
        axes[0].scatter(x_prefill, y_prefill, alpha=0.6, s=50, label='Actual')
        axes[0].plot(x_prefill, pred_prefill, 'r-', linewidth=2,
                     label=f'y={k_prefill:.2f}x+{b_prefill:.2f}')
        axes[0].set_xlabel('Theoretical Prediction (ms)', fontsize=12)
        axes[0].set_ylabel('Actual Measured Time (ms)', fontsize=12)
        axes[0].set_title(f'Prefill (R²={r2_prefill:.4f})', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Decode
        axes[1].scatter(x_decode, y_decode, alpha=0.6, s=50, label='Actual', color='green')
        axes[1].plot(x_decode, pred_decode, 'r-', linewidth=2,
                     label=f'y={k_decode:.2f}x+{b_decode:.2f}')
        axes[1].set_xlabel('Theoretical Prediction (ms)', fontsize=12)
        axes[1].set_ylabel('Actual Measured Time (ms)', fontsize=12)
        axes[1].set_title(f'Decode (R²={r2_decode:.4f})', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Combined
        axes[2].scatter(x_prefill, y_prefill, alpha=0.6, s=50, label='Prefill', color='blue')
        axes[2].scatter(x_decode, y_decode, alpha=0.6, s=50, label='Decode', color='green')
        x_sorted = np.sort(x_combined)
        y_fit = k_combined * x_sorted + b_combined
        axes[2].plot(x_sorted, y_fit, 'r-', linewidth=2,
                     label=f'y={k_combined:.2f}x+{b_combined:.2f}')
        axes[2].set_xlabel('Theoretical Prediction (ms)', fontsize=12)
        axes[2].set_ylabel('Actual Measured Time (ms)', fontsize=12)
        axes[2].set_title(f'Combined (R²={r2_combined:.4f})', fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {args.output}")


if __name__ == '__main__':
    main()
