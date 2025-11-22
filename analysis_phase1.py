"""
Analysis script for Phase 1 experiment results
Loads results and creates convergence plots
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

from config import config


def load_results(output_dir: str) -> pd.DataFrame:
    """
    Load experiment results from CSV.
    
    Args:
        output_dir: Directory containing results
    
    Returns:
        DataFrame with results
    """
    csv_path = os.path.join(output_dir, "convergence_results.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics for each sample size N.
    
    Args:
        df: Results DataFrame
    
    Returns:
        DataFrame with statistics grouped by N
    """
    stats = df.groupby('N').agg({
        'T_hat_ab': ['mean', 'std', 'min', 'max'],
        'T_hat_ac': ['mean', 'std', 'min', 'max'],
        'diff_T_hat': ['mean', 'std', 'min', 'max'],
        'true_temperature': 'first'
    }).reset_index()
    
    # Flatten column names
    stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in stats.columns.values]
    
    return stats


def plot_convergence(df: pd.DataFrame, output_dir: str):
    """
    Create convergence plots showing estimator behavior vs sample size.
    
    Args:
        df: Results DataFrame
        output_dir: Output directory for saving plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 1: Estimator Convergence Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: T_hat_ab and T_hat_ac vs N
    ax1 = axes[0, 0]
    for N in df['N'].unique():
        subset = df[df['N'] == N]
        ax1.scatter([N] * len(subset), subset['T_hat_ab'], alpha=0.6, color='blue', s=50, label='T_hat (a,b)' if N == df['N'].unique()[0] else '')
        ax1.scatter([N] * len(subset), subset['T_hat_ac'], alpha=0.6, color='red', s=50, label='T_hat (a,c)' if N == df['N'].unique()[0] else '')
    
    # Add mean lines
    stats = compute_statistics(df)
    ax1.plot(stats['N'], stats['T_hat_ab_mean'], 'b-', linewidth=2, label='Mean T_hat (a,b)')
    ax1.plot(stats['N'], stats['T_hat_ac_mean'], 'r-', linewidth=2, label='Mean T_hat (a,c)')
    
    # Add true temperature line
    true_temp = df['true_temperature'].iloc[0]
    ax1.axhline(y=true_temp, color='green', linestyle='--', linewidth=2, label=f'True T = {true_temp}')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Sample Size N (log scale)', fontsize=12)
    ax1.set_ylabel('Estimated Temperature', fontsize=12)
    ax1.set_title('Temperature Estimates vs Sample Size', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Difference |T_hat_ab - T_hat_ac| vs N
    ax2 = axes[0, 1]
    for N in df['N'].unique():
        subset = df[df['N'] == N]
        ax2.scatter([N] * len(subset), subset['diff_T_hat'], alpha=0.6, color='purple', s=50)
    
    # Add mean line
    ax2.plot(stats['N'], stats['diff_T_hat_mean'], 'purple', linewidth=2, label='Mean |T_hat_ab - T_hat_ac|')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Sample Size N (log scale)', fontsize=12)
    ax2.set_ylabel('|T_hat_ab - T_hat_ac| (log scale)', fontsize=12)
    ax2.set_title('Convergence of Estimator Difference', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error from true temperature
    ax3 = axes[1, 0]
    df['error_ab'] = abs(df['T_hat_ab'] - df['true_temperature'])
    df['error_ac'] = abs(df['T_hat_ac'] - df['true_temperature'])
    
    for N in df['N'].unique():
        subset = df[df['N'] == N]
        ax3.scatter([N] * len(subset), subset['error_ab'], alpha=0.6, color='blue', s=50, label='|T_hat (a,b) - T_true|' if N == df['N'].unique()[0] else '')
        ax3.scatter([N] * len(subset), subset['error_ac'], alpha=0.6, color='red', s=50, label='|T_hat (a,c) - T_true|' if N == df['N'].unique()[0] else '')
    
    # Add mean lines
    error_stats = df.groupby('N').agg({
        'error_ab': 'mean',
        'error_ac': 'mean'
    }).reset_index()
    
    ax3.plot(error_stats['N'], error_stats['error_ab'], 'b-', linewidth=2, label='Mean error (a,b)')
    ax3.plot(error_stats['N'], error_stats['error_ac'], 'r-', linewidth=2, label='Mean error (a,c)')
    
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('Sample Size N (log scale)', fontsize=12)
    ax3.set_ylabel('Absolute Error (log scale)', fontsize=12)
    ax3.set_title('Estimation Error vs Sample Size', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Sample counts visualization
    ax4 = axes[1, 1]
    sample_counts = df[['N', 'n_a_C1', 'n_b_C1', 'n_c_C1', 'n_a_C2', 'n_b_C2', 'n_c_C2']].iloc[-1]  # Last row
    N_final = sample_counts['N']
    
    categories = ['a (C1)', 'b (C1)', 'c (C1)', 'a (C2)', 'b (C2)', 'c (C2)']
    counts = [
        sample_counts['n_a_C1'], sample_counts['n_b_C1'], sample_counts['n_c_C1'],
        sample_counts['n_a_C2'], sample_counts['n_b_C2'], sample_counts['n_c_C2']
    ]
    
    colors = ['blue', 'orange', 'green', 'blue', 'orange', 'green']
    bars = ax4.bar(categories, counts, color=colors, alpha=0.7)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title(f'Sample Counts for N = {int(N_final)}', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(output_dir, "convergence_plots.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plots saved to {plot_path}")
    
    plt.show()


def print_summary(df: pd.DataFrame):
    """
    Print summary statistics.
    
    Args:
        df: Results DataFrame
    """
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print()
    
    stats = compute_statistics(df)
    true_temp = df['true_temperature'].iloc[0]
    
    print(f"True Temperature: {true_temp:.4f}")
    print()
    print("Estimator Statistics by Sample Size:")
    print("-" * 70)
    print(f"{'N':<10} {'T_hat_ab (mean±std)':<25} {'T_hat_ac (mean±std)':<25} {'|diff| (mean)':<15}")
    print("-" * 70)
    
    for _, row in stats.iterrows():
        N = int(row['N'])
        t_ab_mean = row['T_hat_ab_mean']
        t_ab_std = row['T_hat_ab_std']
        t_ac_mean = row['T_hat_ac_mean']
        t_ac_std = row['T_hat_ac_std']
        diff_mean = row['diff_T_hat_mean']
        
        print(f"{N:<10} {t_ab_mean:>7.4f}±{t_ab_std:<7.4f}   {t_ac_mean:>7.4f}±{t_ac_std:<7.4f}   {diff_mean:>10.6f}")
    
    print()
    
    # Final convergence check
    final_stats = stats.iloc[-1]
    print("Final Convergence (largest N):")
    print(f"  T_hat (a,b): {final_stats['T_hat_ab_mean']:.4f} ± {final_stats['T_hat_ab_std']:.4f}")
    print(f"  T_hat (a,c): {final_stats['T_hat_ac_mean']:.4f} ± {final_stats['T_hat_ac_std']:.4f}")
    print(f"  True T: {true_temp:.4f}")
    print(f"  |T_hat_ab - T_hat_ac|: {final_stats['diff_T_hat_mean']:.6f}")
    print()


def main():
    """Main analysis execution"""
    print("=" * 70)
    print("Phase 1: Results Analysis")
    print("=" * 70)
    
    # Load results
    try:
        df = load_results(config.OUTPUT_DIR)
        print(f"\nLoaded {len(df)} result rows from {config.OUTPUT_DIR}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run experiment_phase1.py first to generate results.")
        return
    
    # Print summary
    print_summary(df)
    
    # Create plots
    print("Generating plots...")
    plot_convergence(df, config.OUTPUT_DIR)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

