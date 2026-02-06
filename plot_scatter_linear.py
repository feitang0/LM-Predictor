#!/usr/bin/env python3
"""
Scatter plot with LINEAR scale: Prediction (x-axis) vs Real-run (y-axis).
Shows y = kx + b relationship naturally.
"""

import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
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


def plot_scatter_linear(real_data, pred_data, output_file):
    """Create scatter plots with LINEAR scale"""

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

    # Create figure with 2x2 layout
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # ===== PREFILL - Full Range =====
    ax1 = fig.add_subplot(gs[0, 0])

    for batch in unique_batches:
        mask = prefill_batch == batch
        ax1.scatter(prefill_pred[mask], prefill_real[mask],
                   c=[batch_to_color[batch]], s=100, alpha=0.7,
                   label=f'BS {batch}', edgecolors='black', linewidth=0.5)

    # Linear regression
    k_p, b_p, r_value_p, _, _ = stats.linregress(prefill_pred, prefill_real)
    r2_p = r_value_p**2

    # Plot fit line
    x_range = np.linspace(prefill_pred.min(), prefill_pred.max(), 100)
    y_fit = k_p * x_range + b_p
    ax1.plot(x_range, y_fit, 'b-', linewidth=3,
            label=f'y = {k_p:.4f}x + {b_p:.4f}\nR² = {r2_p:.6f}',
            zorder=10, alpha=0.8)

    ax1.set_xlabel('Prediction Time (ms)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Real-Run Time (ms)', fontsize=14, fontweight='bold')
    ax1.set_title('PREFILL Phase - Full Range', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)

    # ===== PREFILL - Zoomed to Small Values =====
    ax2 = fig.add_subplot(gs[0, 1])

    # Only show points where prediction < 50ms
    mask_small = prefill_pred < 50
    for batch in unique_batches:
        mask = (prefill_batch == batch) & mask_small
        if mask.any():
            ax2.scatter(prefill_pred[mask], prefill_real[mask],
                       c=[batch_to_color[batch]], s=100, alpha=0.7,
                       label=f'BS {batch}', edgecolors='black', linewidth=0.5)

    # Plot same fit line
    x_range_zoom = np.linspace(0, 50, 100)
    y_fit_zoom = k_p * x_range_zoom + b_p
    ax2.plot(x_range_zoom, y_fit_zoom, 'b-', linewidth=3,
            label=f'y = {k_p:.4f}x + {b_p:.4f}', zorder=10, alpha=0.8)

    ax2.set_xlabel('Prediction Time (ms)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Real-Run Time (ms)', fontsize=14, fontweight='bold')
    ax2.set_title('PREFILL Phase - Zoomed (Pred < 50ms)', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=10)

    # ===== DECODE - Full Range =====
    ax3 = fig.add_subplot(gs[1, 0])

    colors_decode = plt.cm.plasma(np.linspace(0.1, 0.9, len(unique_batches)))
    batch_to_color_decode = {batch: colors_decode[i] for i, batch in enumerate(unique_batches)}

    for batch in unique_batches:
        mask = decode_batch == batch
        ax3.scatter(decode_pred[mask], decode_real[mask],
                   c=[batch_to_color_decode[batch]], s=100, alpha=0.7,
                   label=f'BS {batch}', edgecolors='black', linewidth=0.5)

    # Linear regression
    k_d, b_d, r_value_d, _, _ = stats.linregress(decode_pred, decode_real)
    r2_d = r_value_d**2

    # Plot fit line
    x_range_dec = np.linspace(decode_pred.min(), decode_pred.max(), 100)
    y_fit_dec = k_d * x_range_dec + b_d
    ax3.plot(x_range_dec, y_fit_dec, 'b-', linewidth=3,
            label=f'y = {k_d:.4f}x + {b_d:.4f}\nR² = {r2_d:.6f}',
            zorder=10, alpha=0.8)

    ax3.set_xlabel('Prediction Time (ms)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Real-Run Time (ms)', fontsize=14, fontweight='bold')
    ax3.set_title('DECODE Phase - Full Range', fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left', fontsize=10)

    # ===== DECODE - Zoomed to Small Values =====
    ax4 = fig.add_subplot(gs[1, 1])

    # Only show points where prediction < 2ms
    mask_small_dec = decode_pred < 2
    for batch in unique_batches:
        mask = (decode_batch == batch) & mask_small_dec
        if mask.any():
            ax4.scatter(decode_pred[mask], decode_real[mask],
                       c=[batch_to_color_decode[batch]], s=100, alpha=0.7,
                       label=f'BS {batch}', edgecolors='black', linewidth=0.5)

    # Plot same fit line
    x_range_zoom_dec = np.linspace(0, 2, 100)
    y_fit_zoom_dec = k_d * x_range_zoom_dec + b_d
    ax4.plot(x_range_zoom_dec, y_fit_zoom_dec, 'b-', linewidth=3,
            label=f'y = {k_d:.4f}x + {b_d:.4f}', zorder=10, alpha=0.8)

    ax4.set_xlabel('Prediction Time (ms)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Real-Run Time (ms)', fontsize=14, fontweight='bold')
    ax4.set_title('DECODE Phase - Zoomed (Pred < 2ms)', fontsize=16, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left', fontsize=10)

    # Overall title
    fig.suptitle('Linear Calibration: y = kx + b (Linear Scale)',
                 fontsize=18, fontweight='bold', y=0.995)

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")

    # Print statistics
    print("\n" + "="*80)
    print("LINEAR SCALE ANALYSIS (y = kx + b)")
    print("="*80)

    print(f"\nPREFILL Phase:")
    print(f"  Linear fit:         y = {k_p:.4f}x + {b_p:.4f}")
    print(f"  R² (goodness):      {r2_p:.6f}")
    print(f"  Interpretation:")
    print(f"    - Slope k={k_p:.2f}: Reality is {k_p:.2f}× the prediction on average")
    print(f"    - Intercept b={b_p:.2f}ms: Base overhead of {b_p:.2f}ms")
    print(f"  Prediction range:   {prefill_pred.min():.2f} - {prefill_pred.max():.2f} ms")
    print(f"  Real range:         {prefill_real.min():.2f} - {prefill_real.max():.2f} ms")

    print(f"\nDECODE Phase:")
    print(f"  Linear fit:         y = {k_d:.4f}x + {b_d:.4f}")
    print(f"  R² (goodness):      {r2_d:.6f}")
    print(f"  Interpretation:")
    print(f"    - Slope k={k_d:.2f}: Reality is {k_d:.2f}× the prediction on average")
    print(f"    - Intercept b={b_d:.2f}ms: Base overhead of {b_d:.2f}ms")
    print(f"  Prediction range:   {decode_pred.min():.3f} - {decode_pred.max():.2f} ms")
    print(f"  Real range:         {decode_real.min():.2f} - {decode_real.max():.2f} ms")

    print("\n" + "="*80)
    print("USAGE")
    print("="*80)
    print(f"\nTo calibrate your predictions:")
    print(f"  real_prefill_time = {k_p:.4f} × predicted_prefill_time + {b_p:.4f}")
    print(f"  real_decode_time  = {k_d:.4f} × predicted_decode_time  + {b_d:.4f}")

    # Calculate residuals
    prefill_residuals = prefill_real - (k_p * prefill_pred + b_p)
    decode_residuals = decode_real - (k_d * decode_pred + b_d)

    print(f"\nResidual Analysis (how well does the line fit?):")
    print(f"  PREFILL - Mean Absolute Error: {np.mean(np.abs(prefill_residuals)):.2f} ms")
    print(f"  PREFILL - Max Error:            {np.max(np.abs(prefill_residuals)):.2f} ms")
    print(f"  DECODE  - Mean Absolute Error: {np.mean(np.abs(decode_residuals)):.2f} ms")
    print(f"  DECODE  - Max Error:            {np.max(np.abs(decode_residuals)):.2f} ms")


def main():
    parser = argparse.ArgumentParser(
        description='Create scatter plots with LINEAR scale'
    )
    parser.add_argument('real_file', type=str, help='Real-run result file')
    parser.add_argument('pred_file', type=str, help='Prediction result file')
    parser.add_argument('--output', '-o', type=str, default='scatter_linear.png',
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
    print("\nGenerating linear scale scatter plots...")
    plot_scatter_linear(real_data, pred_data, args.output)


if __name__ == '__main__':
    main()
