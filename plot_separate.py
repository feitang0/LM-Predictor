#!/usr/bin/env python3
"""
Plot real-run and prediction results separately to understand their behavior.
"""

import argparse
import re
import numpy as np
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


def plot_phase_comparison(real_data, pred_data, phase, ax):
    """Plot comparison for a specific phase"""
    real_by_batch = organize_by_batch(real_data, phase)
    pred_by_batch = organize_by_batch(pred_data, phase)

    batch_sizes = sorted(set(list(real_by_batch.keys()) + list(pred_by_batch.keys())))

    colors = plt.cm.tab10(np.linspace(0, 1, len(batch_sizes)))

    for i, batch in enumerate(batch_sizes):
        color = colors[i]

        # Plot real data
        if batch in real_by_batch:
            seqlens, times = real_by_batch[batch]
            ax.plot(seqlens, times, 'o-', color=color, linewidth=2,
                   markersize=8, label=f'BS{batch} (Real)', alpha=0.8)

        # Plot prediction data
        if batch in pred_by_batch:
            seqlens, times = pred_by_batch[batch]
            ax.plot(seqlens, times, 's--', color=color, linewidth=1.5,
                   markersize=6, label=f'BS{batch} (Pred)', alpha=0.5)

    ax.set_xlabel('Sequence/Cache Length', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title(f'{phase.upper()} Phase', fontsize=14, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)


def plot_separate_views(real_data, pred_data, output_prefix):
    """Create separate visualizations for different views"""

    # Figure 1: Prefill and Decode side by side
    fig1, axes = plt.subplots(1, 2, figsize=(20, 8))
    plot_phase_comparison(real_data, pred_data, 'prefill', axes[0])
    plot_phase_comparison(real_data, pred_data, 'decode', axes[1])
    plt.tight_layout()
    fig1.savefig(f'{output_prefix}_phase_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_prefix}_phase_comparison.png")

    # Figure 2: Real vs Prediction - Prefill only
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Real prefill
    real_prefill = organize_by_batch(real_data, 'prefill')
    for i, batch in enumerate(sorted(real_prefill.keys())):
        seqlens, times = real_prefill[batch]
        ax1.plot(seqlens, times, 'o-', linewidth=2, markersize=8,
                label=f'BS{batch}', alpha=0.8)
    ax1.set_xlabel('Sequence Length', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('PREFILL - Real Run', fontsize=14, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend()

    # Prediction prefill
    pred_prefill = organize_by_batch(pred_data, 'prefill')
    for i, batch in enumerate(sorted(pred_prefill.keys())):
        seqlens, times = pred_prefill[batch]
        ax2.plot(seqlens, times, 's--', linewidth=2, markersize=6,
                label=f'BS{batch}', alpha=0.7)
    ax2.set_xlabel('Sequence Length', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('PREFILL - Prediction', fontsize=14, fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend()

    plt.tight_layout()
    fig2.savefig(f'{output_prefix}_prefill_separate.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_prefix}_prefill_separate.png")

    # Figure 3: Real vs Prediction - Decode only
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Real decode
    real_decode = organize_by_batch(real_data, 'decode')
    for i, batch in enumerate(sorted(real_decode.keys())):
        seqlens, times = real_decode[batch]
        ax1.plot(seqlens, times, 'o-', linewidth=2, markersize=8,
                label=f'BS{batch}', alpha=0.8)
    ax1.set_xlabel('Cache Length', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('DECODE - Real Run', fontsize=14, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend()

    # Prediction decode
    pred_decode = organize_by_batch(pred_data, 'decode')
    for i, batch in enumerate(sorted(pred_decode.keys())):
        seqlens, times = pred_decode[batch]
        ax2.plot(seqlens, times, 's--', linewidth=2, markersize=6,
                label=f'BS{batch}', alpha=0.7)
    ax2.set_xlabel('Cache Length', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('DECODE - Prediction', fontsize=14, fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend()

    plt.tight_layout()
    fig3.savefig(f'{output_prefix}_decode_separate.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_prefix}_decode_separate.png")


def main():
    parser = argparse.ArgumentParser(
        description='Plot real-run and prediction results separately'
    )
    parser.add_argument('real_file', type=str, help='Real-run result file')
    parser.add_argument('pred_file', type=str, help='Prediction result file')
    parser.add_argument('--output-prefix', '-o', type=str, default='analysis',
                        help='Output filename prefix (default: analysis)')

    args = parser.parse_args()

    # Parse files
    print(f"Reading real-run results from: {args.real_file}")
    real_data = parse_real_run_file(args.real_file)
    print(f"  → Found {len(real_data)} data points")

    print(f"Reading predictions from: {args.pred_file}")
    pred_data = parse_prediction_file(args.pred_file)
    print(f"  → Found {len(pred_data)} data points")

    # Create plots
    print("\nGenerating visualizations...")
    plot_separate_views(real_data, pred_data, args.output_prefix)

    # Print statistics
    print("\n" + "="*70)
    print("DATA STATISTICS")
    print("="*70)

    real_prefill = [v for (p, b, s), v in real_data.items() if p == 'prefill']
    real_decode = [v for (p, b, s), v in real_data.items() if p == 'decode']
    pred_prefill = [v for (p, b, s), v in pred_data.items() if p == 'prefill']
    pred_decode = [v for (p, b, s), v in pred_data.items() if p == 'decode']

    print("\nPREFILL Phase:")
    print(f"  Real:       min={min(real_prefill):.3f}ms, max={max(real_prefill):.3f}ms, "
          f"mean={np.mean(real_prefill):.3f}ms")
    print(f"  Prediction: min={min(pred_prefill):.3f}ms, max={max(pred_prefill):.3f}ms, "
          f"mean={np.mean(pred_prefill):.3f}ms")
    print(f"  Ratio (Real/Pred): {np.mean(real_prefill)/np.mean(pred_prefill):.2f}×")

    print("\nDECODE Phase:")
    print(f"  Real:       min={min(real_decode):.3f}ms, max={max(real_decode):.3f}ms, "
          f"mean={np.mean(real_decode):.3f}ms")
    print(f"  Prediction: min={min(pred_decode):.3f}ms, max={max(pred_decode):.3f}ms, "
          f"mean={np.mean(pred_decode):.3f}ms")
    print(f"  Ratio (Real/Pred): {np.mean(real_decode)/np.mean(pred_decode):.2f}×")


if __name__ == '__main__':
    main()
