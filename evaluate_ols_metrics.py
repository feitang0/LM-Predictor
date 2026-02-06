#!/usr/bin/env python3
"""
Comprehensive evaluation metrics for OLS regression y = kx + b.
Shows how well the linear model fits the real-run data.
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


def calculate_metrics(x, y_real, k, b):
    """Calculate comprehensive evaluation metrics"""

    # Predicted values using y = kx + b
    y_pred = k * x + b

    # Residuals (errors)
    residuals = y_real - y_pred

    # 1. R² (Coefficient of Determination)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_real - np.mean(y_real))**2)
    r2 = 1 - (ss_res / ss_tot)

    # 2. MSE (Mean Squared Error)
    mse = np.mean(residuals**2)

    # 3. RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mse)

    # 4. MAE (Mean Absolute Error)
    mae = np.mean(np.abs(residuals))

    # 5. MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs(residuals / y_real)) * 100

    # 6. Max Absolute Error
    max_error = np.max(np.abs(residuals))

    # 7. Standard deviation of residuals
    std_residuals = np.std(residuals)

    # 8. Median Absolute Error
    median_ae = np.median(np.abs(residuals))

    metrics = {
        'r2': r2,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'max_error': max_error,
        'std_residuals': std_residuals,
        'median_ae': median_ae,
        'residuals': residuals,
        'y_pred': y_pred
    }

    return metrics


def plot_metrics(x, y_real, k, b, metrics, phase_name, output_file):
    """Create comprehensive metric visualization"""

    y_pred = metrics['y_pred']
    residuals = metrics['residuals']

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    # 1. Scatter plot with fit line
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.scatter(x, y_real, alpha=0.6, s=60, label='Real Data', color='blue')
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = k * x_line + b
    ax1.plot(x_line, y_line, 'r-', linewidth=2, label=f'Fit: y={k:.4f}x+{b:.4f}')
    ax1.set_xlabel('Prediction Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Real-Run Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{phase_name} - Data vs Fitted Line', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 2. Residual plot (residuals vs predicted)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(y_pred, residuals, alpha=0.6, s=60, color='green')
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Fitted Values (ms)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Residuals (ms)', fontsize=11, fontweight='bold')
    ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Histogram of residuals
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(residuals, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Residuals (ms)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Q-Q plot (check normality of residuals)
    ax4 = fig.add_subplot(gs[1, 1])
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. Prediction vs Actual
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(y_real, y_pred, alpha=0.6, s=60, color='orange')
    min_val = min(y_real.min(), y_pred.min())
    max_val = max(y_real.max(), y_pred.max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
    ax5.set_xlabel('Actual Values (ms)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Predicted Values (ms)', fontsize=11, fontweight='bold')
    ax5.set_title('Predicted vs Actual', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # 6. Absolute Error by Prediction Value
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.scatter(x, np.abs(residuals), alpha=0.6, s=60, color='red')
    ax6.set_xlabel('Prediction Time (ms)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Absolute Error (ms)', fontsize=11, fontweight='bold')
    ax6.set_title('Absolute Error Distribution', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # 7. Percentage Error by Prediction Value
    ax7 = fig.add_subplot(gs[2, 1])
    pct_errors = np.abs(residuals / y_real) * 100
    ax7.scatter(x, pct_errors, alpha=0.6, s=60, color='brown')
    ax7.set_xlabel('Prediction Time (ms)', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Absolute % Error', fontsize=11, fontweight='bold')
    ax7.set_title('Percentage Error Distribution', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)

    # 8. Metrics Summary Text Box
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')

    metrics_text = f"""
    REGRESSION METRICS SUMMARY
    {'='*40}

    Goodness of Fit:
      R² = {metrics['r2']:.6f}

    Absolute Errors (ms):
      MAE     = {metrics['mae']:.4f}
      Median  = {metrics['median_ae']:.4f}
      RMSE    = {metrics['rmse']:.4f}
      Max     = {metrics['max_error']:.4f}
      StdDev  = {metrics['std_residuals']:.4f}

    Relative Error:
      MAPE    = {metrics['mape']:.2f}%

    Sample Size:
      N = {len(x)} data points
    """

    ax8.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.suptitle(f'{phase_name} Phase - OLS Evaluation Metrics',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")


def print_detailed_metrics(metrics, phase_name):
    """Print detailed explanation of each metric"""

    print("\n" + "="*80)
    print(f"{phase_name} PHASE - DETAILED METRICS EXPLANATION")
    print("="*80)

    print(f"\n1. R² (Coefficient of Determination) = {metrics['r2']:.6f}")
    print(f"   Interpretation: {metrics['r2']*100:.2f}% of variance in real-run time is explained by prediction")
    if metrics['r2'] > 0.99:
        print("   ✓ EXCELLENT: Nearly perfect linear relationship")
    elif metrics['r2'] > 0.95:
        print("   ✓ VERY GOOD: Strong linear relationship")
    elif metrics['r2'] > 0.90:
        print("   ✓ GOOD: Reliable linear relationship")
    elif metrics['r2'] > 0.80:
        print("   ⚠ MODERATE: Some non-linearity present")
    else:
        print("   ✗ POOR: Linear model may not be appropriate")

    print(f"\n2. MAE (Mean Absolute Error) = {metrics['mae']:.4f} ms")
    print(f"   Interpretation: On average, predictions are off by ±{metrics['mae']:.2f}ms")
    print(f"   This is the typical error you can expect")

    print(f"\n3. RMSE (Root Mean Squared Error) = {metrics['rmse']:.4f} ms")
    print(f"   Interpretation: Standard deviation-like measure of error")
    print(f"   Penalizes large errors more than MAE")
    if metrics['rmse'] / metrics['mae'] > 1.5:
        print("   ⚠ Warning: Some large outliers present (RMSE >> MAE)")

    print(f"\n4. Median Absolute Error = {metrics['median_ae']:.4f} ms")
    print(f"   Interpretation: Half of predictions are within ±{metrics['median_ae']:.2f}ms")
    print(f"   Robust to outliers (compare with MAE)")

    print(f"\n5. Max Absolute Error = {metrics['max_error']:.4f} ms")
    print(f"   Interpretation: Worst-case error is {metrics['max_error']:.2f}ms")
    print(f"   This is the largest mistake the model makes")

    print(f"\n6. MAPE (Mean Absolute Percentage Error) = {metrics['mape']:.2f}%")
    print(f"   Interpretation: Predictions are off by {metrics['mape']:.1f}% on average")
    if metrics['mape'] < 5:
        print("   ✓ EXCELLENT: <5% error")
    elif metrics['mape'] < 10:
        print("   ✓ VERY GOOD: <10% error")
    elif metrics['mape'] < 20:
        print("   ✓ GOOD: <20% error")
    else:
        print("   ⚠ MODERATE: Consider improving the model")

    print(f"\n7. Standard Deviation of Residuals = {metrics['std_residuals']:.4f} ms")
    print(f"   Interpretation: Typical spread of errors around the fit line")
    print(f"   ~68% of points within ±{metrics['std_residuals']:.2f}ms of fit")
    print(f"   ~95% of points within ±{2*metrics['std_residuals']:.2f}ms of fit")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate OLS regression metrics comprehensively'
    )
    parser.add_argument('real_file', type=str, help='Real-run result file')
    parser.add_argument('pred_file', type=str, help='Prediction result file')
    parser.add_argument('--output-prefix', '-o', type=str, default='metrics',
                        help='Output filename prefix')

    args = parser.parse_args()

    # Parse files
    print(f"Reading real-run results from: {args.real_file}")
    real_data = parse_real_run_file(args.real_file)
    print(f"  → Found {len(real_data)} data points")

    print(f"Reading predictions from: {args.pred_file}")
    pred_data = parse_prediction_file(args.pred_file)
    print(f"  → Found {len(pred_data)} data points")

    # Process prefill phase
    prefill_pred = []
    prefill_real = []
    for key in real_data:
        phase, batch, seqlen = key
        if phase == 'prefill' and key in pred_data:
            prefill_pred.append(pred_data[key])
            prefill_real.append(real_data[key])

    prefill_pred = np.array(prefill_pred)
    prefill_real = np.array(prefill_real)

    # Fit OLS
    k_p, b_p, _, _, _ = stats.linregress(prefill_pred, prefill_real)

    # Calculate metrics
    metrics_prefill = calculate_metrics(prefill_pred, prefill_real, k_p, b_p)

    # Process decode phase
    decode_pred = []
    decode_real = []
    for key in real_data:
        phase, batch, seqlen = key
        if phase == 'decode' and key in pred_data:
            decode_pred.append(pred_data[key])
            decode_real.append(real_data[key])

    decode_pred = np.array(decode_pred)
    decode_real = np.array(decode_real)

    # Fit OLS
    k_d, b_d, _, _, _ = stats.linregress(decode_pred, decode_real)

    # Calculate metrics
    metrics_decode = calculate_metrics(decode_pred, decode_real, k_d, b_d)

    # Print detailed metrics
    print_detailed_metrics(metrics_prefill, "PREFILL")
    print_detailed_metrics(metrics_decode, "DECODE")

    # Create visualizations
    print("\nGenerating visualizations...")
    plot_metrics(prefill_pred, prefill_real, k_p, b_p, metrics_prefill,
                'PREFILL', f'{args.output_prefix}_prefill.png')
    plot_metrics(decode_pred, decode_real, k_d, b_d, metrics_decode,
                'DECODE', f'{args.output_prefix}_decode.png')

    # Summary comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Metric':<30} {'PREFILL':<20} {'DECODE':<20}")
    print("-"*80)
    print(f"{'R²':<30} {metrics_prefill['r2']:<20.6f} {metrics_decode['r2']:<20.6f}")
    print(f"{'MAE (ms)':<30} {metrics_prefill['mae']:<20.4f} {metrics_decode['mae']:<20.4f}")
    print(f"{'RMSE (ms)':<30} {metrics_prefill['rmse']:<20.4f} {metrics_decode['rmse']:<20.4f}")
    print(f"{'MAPE (%)':<30} {metrics_prefill['mape']:<20.2f} {metrics_decode['mape']:<20.2f}")
    print(f"{'Max Error (ms)':<30} {metrics_prefill['max_error']:<20.4f} {metrics_decode['max_error']:<20.4f}")


if __name__ == '__main__':
    main()
