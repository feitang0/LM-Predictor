#!/usr/bin/env python3
"""
Scatter plot: Prediction (x-axis) vs Real-run (y-axis) for prefill and decode phases.
"""

import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


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


def plot_scatter(real_data, pred_data, output_file):
    """Create scatter plots with prediction on x-axis and real on y-axis"""

    # Create figure with 1x2 layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

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

    # ===== PREFILL SCATTER PLOT =====
    for batch in unique_batches:
        mask = prefill_batch == batch
        ax1.scatter(prefill_pred[mask], prefill_real[mask],
                   c=[batch_to_color[batch]], s=100, alpha=0.7,
                   label=f'BS {batch}', edgecolors='black', linewidth=0.5)

    # Add perfect prediction line (y=x)
    max_val = max(prefill_pred.max(), prefill_real.max())
    min_val = min(prefill_pred.min(), prefill_real.min())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
            label='Perfect Prediction (y=x)', zorder=10)

    # Add linear regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(prefill_pred, prefill_real)
    x_line = np.array([min_val, max_val])
    y_line = slope * x_line + intercept
    ax1.plot(x_line, y_line, 'b-', linewidth=2,
            label=f'Linear Fit: y={slope:.2f}x+{intercept:.2f}\n(R²={r_value**2:.4f})',
            zorder=9)

    ax1.set_xlabel('Prediction Time (ms)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Real-Run Time (ms)', fontsize=14, fontweight='bold')
    ax1.set_title('PREFILL Phase: Prediction vs Reality', fontsize=16, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(loc='upper left', fontsize=10)

    # Add statistics text
    mean_ratio = np.mean(prefill_real / prefill_pred)
    stats_text = f'Mean Ratio: {mean_ratio:.2f}×\n'
    stats_text += f'Points: {len(prefill_pred)}'
    ax1.text(0.98, 0.02, stats_text,
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # ===== DECODE SCATTER PLOT =====
    colors_decode = plt.cm.plasma(np.linspace(0.1, 0.9, len(unique_batches)))
    batch_to_color_decode = {batch: colors_decode[i] for i, batch in enumerate(unique_batches)}

    for batch in unique_batches:
        mask = decode_batch == batch
        ax2.scatter(decode_pred[mask], decode_real[mask],
                   c=[batch_to_color_decode[batch]], s=100, alpha=0.7,
                   label=f'BS {batch}', edgecolors='black', linewidth=0.5)

    # Add perfect prediction line (y=x)
    max_val_dec = max(decode_pred.max(), decode_real.max())
    min_val_dec = min(decode_pred.min(), decode_real.min())
    ax2.plot([min_val_dec, max_val_dec], [min_val_dec, max_val_dec], 'r--', linewidth=2,
            label='Perfect Prediction (y=x)', zorder=10)

    # Add linear regression line
    slope_dec, intercept_dec, r_value_dec, p_value_dec, std_err_dec = stats.linregress(decode_pred, decode_real)
    x_line_dec = np.array([min_val_dec, max_val_dec])
    y_line_dec = slope_dec * x_line_dec + intercept_dec
    ax2.plot(x_line_dec, y_line_dec, 'b-', linewidth=2,
            label=f'Linear Fit: y={slope_dec:.2f}x+{intercept_dec:.2f}\n(R²={r_value_dec**2:.4f})',
            zorder=9)

    ax2.set_xlabel('Prediction Time (ms)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Real-Run Time (ms)', fontsize=14, fontweight='bold')
    ax2.set_title('DECODE Phase: Prediction vs Reality', fontsize=16, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(loc='upper left', fontsize=10)

    # Add statistics text
    mean_ratio_dec = np.mean(decode_real / decode_pred)
    stats_text_dec = f'Mean Ratio: {mean_ratio_dec:.2f}×\n'
    stats_text_dec += f'Points: {len(decode_pred)}'
    ax2.text(0.98, 0.02, stats_text_dec,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # Overall title
    fig.suptitle('Prediction vs Real-Run Time Scatter Analysis',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")

    # Print detailed statistics
    print("\n" + "="*80)
    print("SCATTER PLOT ANALYSIS")
    print("="*80)

    print(f"\nPREFILL Phase:")
    print(f"  Data points:        {len(prefill_pred)}")
    print(f"  Linear fit:         y = {slope:.4f}x + {intercept:.4f}")
    print(f"  R² (fit quality):   {r_value**2:.6f}")
    print(f"  Mean ratio:         {mean_ratio:.2f}× (Real/Pred)")
    print(f"  Ratio range:        {(prefill_real/prefill_pred).min():.2f}× - {(prefill_real/prefill_pred).max():.2f}×")

    print(f"\nDECODE Phase:")
    print(f"  Data points:        {len(decode_pred)}")
    print(f"  Linear fit:         y = {slope_dec:.4f}x + {intercept_dec:.4f}")
    print(f"  R² (fit quality):   {r_value_dec**2:.6f}")
    print(f"  Mean ratio:         {mean_ratio_dec:.2f}× (Real/Pred)")
    print(f"  Ratio range:        {(decode_real/decode_pred).min():.2f}× - {(decode_real/decode_pred).max():.2f}×")

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("• Perfect prediction: all points on red dashed line (y=x)")
    print("• Above the line: Real-run is slower than predicted (underprediction)")
    print("• Below the line: Real-run is faster than predicted (overprediction)")
    print("• Linear fit (blue): calibration formula to convert prediction → reality")
    print("• R² close to 1: good linear relationship, calibration will work well")
    print("• R² close to 0: poor linear relationship, calibration won't help much")


def main():
    parser = argparse.ArgumentParser(
        description='Create scatter plots of prediction vs real-run time'
    )
    parser.add_argument('real_file', type=str, help='Real-run result file')
    parser.add_argument('pred_file', type=str, help='Prediction result file')
    parser.add_argument('--output', '-o', type=str, default='scatter_pred_vs_real.png',
                        help='Output plot filename')

    args = parser.parse_args()

    # Parse files
    print(f"Reading real-run results from: {args.real_file}")
    real_data = parse_real_run_file(args.real_file)
    print(f"  → Found {len(real_data)} data points")

    print(f"Reading predictions from: {args.pred_file}")
    pred_data = parse_prediction_file(args.pred_file)
    print(f"  → Found {len(pred_data)} data points")

    # Create plots
    print("\nGenerating scatter plots...")
    plot_scatter(real_data, pred_data, args.output)


if __name__ == '__main__':
    main()
