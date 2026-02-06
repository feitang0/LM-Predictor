#!/usr/bin/env python3
"""
Plot real-run decode phase results.
"""

import argparse
import re
import numpy as np
import matplotlib.pyplot as plt


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


def plot_real_decode(data, output_file):
    """Create a detailed plot of real decode phase results"""

    decode_data = organize_by_batch(data, 'decode')
    batch_sizes = sorted(decode_data.keys())

    # Create figure with better styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Use a nice color palette
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(batch_sizes)))

    # Plot 1: Linear scale to show the flat behavior
    for i, batch in enumerate(batch_sizes):
        cachelens, times = decode_data[batch]
        ax1.plot(cachelens, times, 'o-',
                color=colors[i],
                linewidth=2.5,
                markersize=10,
                label=f'Batch Size {batch}',
                alpha=0.85)

    ax1.set_xlabel('Cache Length', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Execution Time (ms)', fontsize=14, fontweight='bold')
    ax1.set_title('OPT-125M Decode Phase - Linear Scale',
                 fontsize=16, fontweight='bold', pad=20)
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(cachelens)
    ax1.set_xticklabels([str(s) for s in cachelens])
    ax1.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.95)

    # Plot 2: Log-log scale for comparison
    for i, batch in enumerate(batch_sizes):
        cachelens, times = decode_data[batch]
        ax2.plot(cachelens, times, 's-',
                color=colors[i],
                linewidth=2.5,
                markersize=10,
                label=f'Batch Size {batch}',
                alpha=0.85)

    ax2.set_xlabel('Cache Length', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Execution Time (ms)', fontsize=14, fontweight='bold')
    ax2.set_title('OPT-125M Decode Phase - Log-Log Scale',
                 fontsize=16, fontweight='bold', pad=20)
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.set_xticks(cachelens)
    ax2.set_xticklabels([str(s) for s in cachelens])
    ax2.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
    ax2.grid(True, alpha=0.2, which='minor', linestyle=':', linewidth=0.3)
    ax2.legend(loc='upper left', fontsize=11, framealpha=0.95)

    # Add statistics annotation to first plot
    stats_text = f"Data points: {sum(len(v[0]) for v in decode_data.values())}\n"
    stats_text += f"Batch sizes: {min(batch_sizes)} - {max(batch_sizes)}\n"
    stats_text += f"Cache lengths: {min(cachelens)} - {max(cachelens)}\n"
    all_times = [t for cachelens, times in decode_data.values() for t in times]
    stats_text += f"Time range: {min(all_times):.2f} - {max(all_times):.2f} ms"

    ax1.text(0.98, 0.98, stats_text,
            transform=ax1.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")

    # Print detailed statistics
    print("\n" + "="*70)
    print("DECODE PHASE - DETAILED STATISTICS")
    print("="*70)
    print(f"\n{'Batch':<8} {'CacheLen':<10} {'Time(ms)':<12} {'Tokens/Iter':<15} {'Throughput':<15}")
    print("-"*80)

    for batch in batch_sizes:
        cachelens, times = decode_data[batch]
        for cachelen, time in zip(cachelens, times):
            # In decode, we generate 1 new token per iteration
            tokens_per_iter = batch  # batch size new tokens
            throughput = tokens_per_iter / (time / 1000)  # tokens per second
            print(f"{batch:<8} {cachelen:<10} {time:<12.4f} {tokens_per_iter:<15} {throughput:<15.1f} tok/s")

    # Summary statistics by batch size
    print("\n" + "="*70)
    print("SUMMARY BY BATCH SIZE")
    print("="*70)
    print(f"{'Batch':<8} {'Min(ms)':<12} {'Max(ms)':<12} {'Mean(ms)':<12} {'Std(ms)':<12} {'Variation':<12}")
    print("-"*80)

    for batch in batch_sizes:
        cachelens, times = decode_data[batch]
        times_array = np.array(times)
        variation = (times_array.max() - times_array.min()) / times_array.mean() * 100
        print(f"{batch:<8} {times_array.min():<12.4f} {times_array.max():<12.4f} "
              f"{times_array.mean():<12.4f} {times_array.std():<12.4f} {variation:<12.1f}%")

    # Analyze overhead dominance
    print("\n" + "="*70)
    print("OVERHEAD ANALYSIS")
    print("="*70)
    print("Checking if decode time varies significantly with cache length:\n")

    for batch in batch_sizes:
        cachelens, times = decode_data[batch]
        times_array = np.array(times)

        # Calculate if time is essentially constant (coefficient of variation < 20%)
        cv = (times_array.std() / times_array.mean()) * 100

        if cv < 20:
            print(f"BS {batch:>2}: OVERHEAD-DOMINATED (CV={cv:.1f}%, mean={times_array.mean():.2f}ms)")
        else:
            print(f"BS {batch:>2}: COMPUTE-INFLUENCED (CV={cv:.1f}%, range={times_array.min():.2f}-{times_array.max():.2f}ms)")


def main():
    parser = argparse.ArgumentParser(
        description='Plot real-run decode phase results'
    )
    parser.add_argument('real_file', type=str, help='Real-run result file')
    parser.add_argument('--output', '-o', type=str, default='real_decode.png',
                        help='Output plot filename')

    args = parser.parse_args()

    # Parse file
    print(f"Reading real-run results from: {args.real_file}")
    data = parse_real_run_file(args.real_file)
    decode_count = sum(1 for (phase, _, _) in data.keys() if phase == 'decode')
    print(f"  → Found {decode_count} decode data points")

    # Create plot
    print("\nGenerating visualization...")
    plot_real_decode(data, args.output)


if __name__ == '__main__':
    main()
