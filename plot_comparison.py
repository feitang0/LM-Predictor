#!/usr/bin/env python3
"""
Plot prediction vs real-run comparison for both prefill and decode phases.
"""

import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


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
                avg_time = np.mean(round_times[1:]) * 1000
                key = (current_phase, current_batch, current_seqlen)
                results[key] = avg_time

            current_seqlen = int(seqlen_match.group(1))
            round_times = []
            continue

        # Parse round times
        round_match = re.match(r'Round (\d+): ([\d.]+) seconds', line)
        if round_match:
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


def organize_by_batch(data, phase):
    """
    Organize data by batch size for plotting.
    Returns: {batch_size: ([seq_lens], [times])}
    """
    organized = {}

    for (ph, batch, seqlen), time in data.items():
        if ph != phase:
            continue

        if batch not in organized:
            organized[batch] = ([], [])

        organized[batch][0].append(seqlen)
        organized[batch][1].append(time)

    # Sort by sequence length
    for batch in organized:
        seqlens, times = organized[batch]
        sorted_pairs = sorted(zip(seqlens, times))
        organized[batch] = ([s for s, t in sorted_pairs], [t for s, t in sorted_pairs])

    return organized


def plot_comparison(real_data, pred_data, output_file):
    """Create comprehensive comparison plots"""

    # Create figure with 2x2 layout
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Organize data
    real_prefill = organize_by_batch(real_data, 'prefill')
    pred_prefill = organize_by_batch(pred_data, 'prefill')
    real_decode = organize_by_batch(real_data, 'decode')
    pred_decode = organize_by_batch(pred_data, 'decode')

    batch_sizes = sorted(real_prefill.keys())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(batch_sizes)))

    # ===== PREFILL PHASE =====
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Prefill - Real vs Pred Overlay
    for i, batch in enumerate(batch_sizes):
        color = colors[i]

        # Real data
        if batch in real_prefill:
            seqlens, times = real_prefill[batch]
            ax1.plot(seqlens, times, 'o-', color=color, linewidth=2.5,
                    markersize=8, label=f'BS{batch} Real', alpha=0.8)

        # Prediction data
        if batch in pred_prefill:
            seqlens, times = pred_prefill[batch]
            ax1.plot(seqlens, times, 's--', color=color, linewidth=1.5,
                    markersize=6, alpha=0.5)

    ax1.set_xlabel('Sequence Length', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('PREFILL: Prediction (dashed) vs Real (solid)',
                  fontsize=14, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(loc='upper left', fontsize=9, ncol=2)

    # Prefill - Error Analysis
    for i, batch in enumerate(batch_sizes):
        if batch in real_prefill and batch in pred_prefill:
            real_seqlens, real_times = real_prefill[batch]
            pred_seqlens, pred_times = pred_prefill[batch]

            # Calculate error ratios
            ratios = [r/p for r, p in zip(real_times, pred_times)]

            ax2.plot(real_seqlens, ratios, 'o-', color=colors[i],
                    linewidth=2, markersize=8, label=f'BS{batch}', alpha=0.8)

    ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Perfect Match')
    ax2.set_xlabel('Sequence Length', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Real / Prediction Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('PREFILL: Error Ratio (Real/Pred)',
                  fontsize=14, fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(loc='best', fontsize=9)

    # ===== DECODE PHASE =====
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    colors_decode = plt.cm.plasma(np.linspace(0.1, 0.9, len(batch_sizes)))

    # Decode - Real vs Pred Overlay (Linear scale to show flatness)
    for i, batch in enumerate(batch_sizes):
        color = colors_decode[i]

        # Real data
        if batch in real_decode:
            cachelens, times = real_decode[batch]
            ax3.plot(cachelens, times, 'o-', color=color, linewidth=2.5,
                    markersize=8, label=f'BS{batch} Real', alpha=0.8)

        # Prediction data
        if batch in pred_decode:
            cachelens, times = pred_decode[batch]
            ax3.plot(cachelens, times, 's--', color=color, linewidth=1.5,
                    markersize=6, alpha=0.5)

    ax3.set_xlabel('Cache Length', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax3.set_title('DECODE: Prediction (dashed) vs Real (solid)',
                  fontsize=14, fontweight='bold')
    ax3.set_xscale('log', base=2)
    ax3.grid(True, alpha=0.3, which='both')
    ax3.legend(loc='upper left', fontsize=9, ncol=2)

    # Decode - Error Analysis
    for i, batch in enumerate(batch_sizes):
        if batch in real_decode and batch in pred_decode:
            real_cachelens, real_times = real_decode[batch]
            pred_cachelens, pred_times = pred_decode[batch]

            # Calculate error ratios
            ratios = [r/p for r, p in zip(real_times, pred_times)]

            ax4.plot(real_cachelens, ratios, 'o-', color=colors_decode[i],
                    linewidth=2, markersize=8, label=f'BS{batch}', alpha=0.8)

    ax4.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Perfect Match')
    ax4.set_xlabel('Cache Length', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Real / Prediction Ratio', fontsize=12, fontweight='bold')
    ax4.set_title('DECODE: Error Ratio (Real/Pred)',
                  fontsize=14, fontweight='bold')
    ax4.set_xscale('log', base=2)
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3, which='both')
    ax4.legend(loc='best', fontsize=9)

    # Add overall title
    fig.suptitle('OPT-125M: Prediction vs Real-Run Comparison',
                 fontsize=18, fontweight='bold', y=0.995)

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")


def print_statistics(real_data, pred_data):
    """Print comparison statistics"""

    print("\n" + "="*80)
    print("PREDICTION ACCURACY STATISTICS")
    print("="*80)

    for phase in ['prefill', 'decode']:
        real_phase = {k: v for k, v in real_data.items() if k[0] == phase}
        pred_phase = {k: v for k, v in pred_data.items() if k[0] == phase}

        # Calculate ratios
        ratios = []
        errors = []

        for key in real_phase:
            if key in pred_phase:
                ratio = real_phase[key] / pred_phase[key]
                error = abs(real_phase[key] - pred_phase[key]) / real_phase[key] * 100
                ratios.append(ratio)
                errors.append(error)

        if ratios:
            print(f"\n{phase.upper()} Phase:")
            print(f"  Mean Ratio (Real/Pred):     {np.mean(ratios):.2f}×")
            print(f"  Ratio Range:                {np.min(ratios):.2f}× - {np.max(ratios):.2f}×")
            print(f"  Mean Absolute Error:        {np.mean(errors):.1f}%")
            print(f"  Median Absolute Error:      {np.median(errors):.1f}%")

            # Group by batch size
            batch_ratios = {}
            for (ph, batch, seqlen), real_time in real_phase.items():
                if (ph, batch, seqlen) in pred_phase:
                    pred_time = pred_phase[(ph, batch, seqlen)]
                    ratio = real_time / pred_time

                    if batch not in batch_ratios:
                        batch_ratios[batch] = []
                    batch_ratios[batch].append(ratio)

            print(f"\n  Ratio by Batch Size:")
            for batch in sorted(batch_ratios.keys()):
                ratios_arr = np.array(batch_ratios[batch])
                print(f"    BS {batch:>2}: mean={ratios_arr.mean():6.2f}×, "
                      f"std={ratios_arr.std():5.2f}, "
                      f"range={ratios_arr.min():.2f}-{ratios_arr.max():.2f}×")


def main():
    parser = argparse.ArgumentParser(
        description='Plot prediction vs real-run comparison'
    )
    parser.add_argument('real_file', type=str, help='Real-run result file')
    parser.add_argument('pred_file', type=str, help='Prediction result file')
    parser.add_argument('--output', '-o', type=str, default='prediction_vs_real.png',
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
    print("\nGenerating visualization...")
    plot_comparison(real_data, pred_data, args.output)

    # Print statistics
    print_statistics(real_data, pred_data)


if __name__ == '__main__':
    main()
