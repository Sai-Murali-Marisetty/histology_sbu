#!/usr/bin/env python3
"""
08_compare_segmenters.py - Compare Cellpose vs StarDist segmentation

Compares detection rates, feature distributions, and spatial agreement.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
import seaborn as sns


def load_results(cellpose_csv, stardist_csv):
    """Load segmentation results from both methods"""
    print("📊 Loading results...")
    cp = pd.read_csv(cellpose_csv)
    sd = pd.read_csv(stardist_csv)
    print(f"  Cellpose: {len(cp):,} nuclei")
    print(f"  StarDist: {len(sd):,} nuclei")
    return cp, sd


def compare_counts(cp_df, sd_df):
    """Compare detection counts"""
    print(f"\n{'='*80}")
    print("DETECTION COUNT COMPARISON")
    print(f"{'='*80}")
    print(f"Cellpose:  {len(cp_df):,} nuclei")
    print(f"StarDist:  {len(sd_df):,} nuclei")
    
    diff = abs(len(cp_df) - len(sd_df))
    pct_diff = diff / max(len(cp_df), len(sd_df)) * 100
    
    print(f"Difference: {diff:,} nuclei ({pct_diff:.1f}%)")
    
    if len(cp_df) > len(sd_df):
        print(f"→ Cellpose detected {diff:,} more nuclei")
    elif len(sd_df) > len(cp_df):
        print(f"→ StarDist detected {diff:,} more nuclei")
    else:
        print("→ Same detection count!")


def spatial_agreement(cp_df, sd_df, distance_threshold=10):
    """
    Compute spatial agreement between segmenters.
    
    For each Cellpose detection, find nearest StarDist detection.
    Agreement = fraction within distance_threshold pixels.
    """
    print(f"\n{'='*80}")
    print("SPATIAL AGREEMENT ANALYSIS")
    print(f"{'='*80}")
    print(f"Distance threshold: {distance_threshold} pixels")
    
    cp_coords = cp_df[['x', 'y']].values
    sd_coords = sd_df[['x', 'y']].values
    
    # Build tree for StarDist
    tree = cKDTree(sd_coords)
    
    # For each Cellpose nucleus, find nearest StarDist
    distances, _ = tree.query(cp_coords)
    
    matches = (distances < distance_threshold).sum()
    agreement = matches / len(cp_coords) * 100
    
    print(f"\nCellpose → StarDist matching:")
    print(f"  Matched: {matches:,} / {len(cp_coords):,} ({agreement:.1f}%)")
    print(f"  Mean distance: {distances.mean():.2f} px")
    print(f"  Median distance: {np.median(distances):.2f} px")
    
    # Reverse: StarDist → Cellpose
    tree_cp = cKDTree(cp_coords)
    distances_rev, _ = tree_cp.query(sd_coords)
    matches_rev = (distances_rev < distance_threshold).sum()
    agreement_rev = matches_rev / len(sd_coords) * 100
    
    print(f"\nStarDist → Cellpose matching:")
    print(f"  Matched: {matches_rev:,} / {len(sd_coords):,} ({agreement_rev:.1f}%)")
    
    return agreement, agreement_rev


def compare_features(cp_df, sd_df, features=None):
    """Compare feature distributions"""
    
    if features is None:
        features = ['area_px', 'aspect_ratio', 'circularity', 'eccentricity']
    
    # Filter to common features
    common_features = [f for f in features if f in cp_df.columns and f in sd_df.columns]
    
    if not common_features:
        print("Warning: No common features to compare")
        return None
    
    print(f"\n{'='*80}")
    print("FEATURE DISTRIBUTION COMPARISON")
    print(f"{'='*80}")
    
    fig, axes = plt.subplots(2, len(common_features), figsize=(5*len(common_features), 8))
    if len(common_features) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, feat in enumerate(common_features):
        # Histograms
        ax = axes[0, i]
        ax.hist(cp_df[feat], bins=50, alpha=0.5, label='Cellpose', density=True, color='blue')
        ax.hist(sd_df[feat], bins=50, alpha=0.5, label='StarDist', density=True, color='red')
        ax.set_xlabel(feat.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_title(f'{feat}\nCP: {cp_df[feat].mean():.2f} | SD: {sd_df[feat].mean():.2f}')
        
        # Box plots
        ax = axes[1, i]
        data = [cp_df[feat], sd_df[feat]]
        bp = ax.boxplot(data, labels=['Cellpose', 'StarDist'], patch_artist=True)
        bp['boxes'][0].set_facecolor('blue')
        bp['boxes'][1].set_facecolor('red')
        ax.set_ylabel(feat.replace('_', ' ').title())
        
        # Print stats
        print(f"\n{feat}:")
        print(f"  Cellpose: {cp_df[feat].mean():.3f} ± {cp_df[feat].std():.3f}")
        print(f"  StarDist: {sd_df[feat].mean():.3f} ± {sd_df[feat].std():.3f}")
        
        # T-test
        from scipy.stats import ttest_ind
        t_stat, p_val = ttest_ind(cp_df[feat], sd_df[feat])
        print(f"  t-test: t={t_stat:.3f}, p={p_val:.3e}")
    
    plt.tight_layout()
    return fig


def plot_spatial_distribution(cp_df, sd_df, out_path):
    """Plot spatial distribution of detections"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Cellpose
    ax1.scatter(cp_df['x'], cp_df['y'], s=1, alpha=0.3, c='blue')
    ax1.set_title(f'Cellpose ({len(cp_df):,} nuclei)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.invert_yaxis()
    
    # StarDist
    ax2.scatter(sd_df['x'], sd_df['y'], s=1, alpha=0.3, c='red')
    ax2.set_title(f'StarDist ({len(sd_df):,} nuclei)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (pixels)')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved spatial distribution plot: {out_path}")


def create_comparison_report(cp_df, sd_df, agreement_cp, agreement_sd, out_path):
    """Create text summary report"""
    
    with open(out_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CELLPOSE VS STARDIST COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("DETECTION COUNTS\n")
        f.write("-"*80 + "\n")
        f.write(f"Cellpose: {len(cp_df):,} nuclei\n")
        f.write(f"StarDist: {len(sd_df):,} nuclei\n")
        diff = abs(len(cp_df) - len(sd_df))
        f.write(f"Difference: {diff:,} ({diff/max(len(cp_df), len(sd_df))*100:.1f}%)\n\n")
        
        f.write("SPATIAL AGREEMENT\n")
        f.write("-"*80 + "\n")
        f.write(f"Cellpose → StarDist: {agreement_cp:.1f}%\n")
        f.write(f"StarDist → Cellpose: {agreement_sd:.1f}%\n\n")
        
        f.write("FEATURE COMPARISON\n")
        f.write("-"*80 + "\n")
        
        features = ['area_px', 'aspect_ratio', 'circularity', 'eccentricity']
        for feat in features:
            if feat in cp_df.columns and feat in sd_df.columns:
                f.write(f"\n{feat}:\n")
                f.write(f"  Cellpose: {cp_df[feat].mean():.3f} ± {cp_df[feat].std():.3f}\n")
                f.write(f"  StarDist: {sd_df[feat].mean():.3f} ± {sd_df[feat].std():.3f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Saved comparison report: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Compare Cellpose and StarDist segmentation")
    ap.add_argument('--cellpose_csv', required=True, help="Cellpose results CSV")
    ap.add_argument('--stardist_csv', required=True, help="StarDist results CSV")
    ap.add_argument('--out_dir', required=True, help="Output directory")
    ap.add_argument('--slide_name', required=True, help="Slide name for labeling")
    ap.add_argument('--distance_threshold', type=float, default=10, 
                   help="Distance threshold for spatial matching (pixels)")
    args = ap.parse_args()
    
    print(f"\n{'='*80}")
    print(f"SEGMENTATION COMPARISON: {args.slide_name}")
    print(f"{'='*80}\n")
    
    # Create output directory
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    cp_df, sd_df = load_results(args.cellpose_csv, args.stardist_csv)
    
    # Compare counts
    compare_counts(cp_df, sd_df)
    
    # Spatial agreement
    agreement_cp, agreement_sd = spatial_agreement(cp_df, sd_df, args.distance_threshold)
    
    # Feature comparison
    fig_features = compare_features(cp_df, sd_df)
    if fig_features:
        fig_features.savefig(Path(args.out_dir) / f'{args.slide_name}_feature_comparison.png', 
                            dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Saved feature comparison plot")
    
    # Spatial distribution
    plot_spatial_distribution(cp_df, sd_df, 
                             Path(args.out_dir) / f'{args.slide_name}_spatial_distribution.png')
    
    # Create summary report
    create_comparison_report(cp_df, sd_df, agreement_cp, agreement_sd,
                           Path(args.out_dir) / f'{args.slide_name}_comparison_report.txt')
    
    # Save summary CSV
    summary = {
        'slide': args.slide_name,
        'cellpose_count': len(cp_df),
        'stardist_count': len(sd_df),
        'count_difference': abs(len(cp_df) - len(sd_df)),
        'spatial_agreement_cp_to_sd': agreement_cp,
        'spatial_agreement_sd_to_cp': agreement_sd,
        'cellpose_mean_area': cp_df['area_px'].mean() if 'area_px' in cp_df.columns else np.nan,
        'stardist_mean_area': sd_df['area_px'].mean() if 'area_px' in sd_df.columns else np.nan,
    }
    
    pd.DataFrame([summary]).to_csv(
        Path(args.out_dir) / f'{args.slide_name}_comparison_summary.csv', 
        index=False
    )
    print(f"✓ Saved summary CSV")
    
    print(f"\n{'='*80}")
    print("✅ COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"📂 Results saved to: {args.out_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
