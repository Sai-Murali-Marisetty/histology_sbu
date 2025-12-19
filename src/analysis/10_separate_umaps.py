#!/usr/bin/env python3
"""
Generate separate UMAP analyses for each stain type.
Combines all slides of the same stain type into one UMAP.
"""

import pandas as pd
import numpy as np
import umap
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

import matplotlib as mpl
from matplotlib.lines import Line2D
from collections import Counter

# Global style (tweak as you like)
MPL_RC = {
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11
}
mpl.rcParams.update(MPL_RC)

def _format_axes(ax, title=None, xlabel='UMAP 1', ylabel='UMAP 2'):
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, weight='bold')

def _annotate_cluster_centroids(ax, df, xcol='umap_1', ycol='umap_2',
                                cluster_col='umap_cluster',
                                text_size=10, facecolor='white'):
    """Put cluster id at cluster centroid."""
    if cluster_col not in df.columns:
        return
    for cid, g in df.groupby(cluster_col):
        if len(g) == 0:
            continue
        x = g[xcol].mean()
        y = g[ycol].mean()
        # Outline box for readability
        ax.text(x, y, str(cid),
                ha='center', va='center',
                fontsize=text_size, weight='bold',
                color='black',
                bbox=dict(boxstyle='round,pad=0.22',
                          facecolor=facecolor, edgecolor='black', linewidth=0.6, alpha=0.8))

def _nice_colorbar(cb, label=None):
    if label:
        cb.set_label(label)
    cb.ax.tick_params(labelsize=11)


import re

# ---- Stain rules -----------------------------------------------------------
STAIN_PREFIXES = {
    "H&E":     [r"^HE-?", r"^H&E-?", r"^HandE-?"],
    "CD3":     [r"^CD3-"],
    "GFAP":    [r"^GFAP-"],
    "Iba1":    [r"^Iba1-", r"^Ibal-"],   # tolerate Ibal typo
    "NF":      [r"^NF-"],
    # PGP9.5 variants: PGP9-5-, PGP9.5-, PGP9_5-
    "PGP9.5":  [r"^PGP9[._-]?5-"],
    # NEW: Treat PID & BID (and BIDS typo) as ONE stain type called "PID"
    "PID":     [r"^PID", r"^BID", r"^BIDS"],
}

def normalize_slide_basename(name: str) -> str:
    """Fix common typos while preserving families."""
    # Normalize mistaken 'BIDS' family into PID family logically
    if re.match(r"^BIDS", name, flags=re.IGNORECASE):
        return re.sub(r"^BIDS", "PID", name, flags=re.IGNORECASE)
    return name

def matches_any(patterns, text) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)

def is_stain_slide(slide_id: str, stain_type: str) -> bool:
    """Positive inclusion by explicit prefixes only."""
    key = {"PGP9-5":"PGP9.5","PGP9_5":"PGP9.5"}.get(stain_type, stain_type)
    pats = STAIN_PREFIXES.get(key, [])
    return matches_any(pats, slide_id)

def load_all_slides_by_type(results_dir: Path, stain_type: str) -> pd.DataFrame:
    """
    Load all slides of a specific stain type using explicit prefix rules.
    - H&E only: HE-/H&E-/HandE-
    - PID: PID*/BID*/BIDS* (BIDS normalized to PID logically)
    - Other markers: CD3-/GFAP-/Iba1-/NF-/PGP9.5- variants
    """
    all_data = []

    # for PGP legend names
    stain_type_key = {"PGP9-5":"PGP9.5","PGP9_5":"PGP9.5"}.get(stain_type, stain_type)

    for slide_dir in results_dir.iterdir():
        if not slide_dir.is_dir():
            continue

        raw_id   = slide_dir.name          # directory name on disk
        norm_id  = normalize_slide_basename(raw_id)  # logical label

        # Include only slides that match the requested stain
        if not is_stain_slide(norm_id, stain_type_key):
            continue

        # Locate a features CSV. Historically different pipelines created
        # files like *_final.csv, *_nuclei_features.csv, or *_nuclei_features_enriched.csv
        # Try a few common candidates, then fall back to scanning the features folder.
        features_dir = slide_dir / 'features'
        if not features_dir.exists():
            continue

        csv_path = None
        candidates = [
            features_dir / f"{raw_id}_nuclei_features_enriched.csv",
            features_dir / f"{raw_id}_nuclei_features.csv",
            features_dir / f"{norm_id}_nuclei_features_enriched.csv",
            features_dir / f"{norm_id}_nuclei_features.csv",
            features_dir / f"{raw_id}_final.csv",
            features_dir / f"{norm_id}_final.csv",
        ]
        for p in candidates:
            if p.exists():
                csv_path = p
                break

        # If still not found, scan for any reasonable CSV (prefer nuclei_features)
        if csv_path is None:
            all_csvs = list(features_dir.glob('*.csv'))
            for p in all_csvs:
                if 'nuclei_features' in p.name:
                    csv_path = p
                    break
            if csv_path is None and all_csvs:
                # last resort: pick the first CSV in features dir
                csv_path = all_csvs[0]
        if csv_path is None:
            continue

        df = pd.read_csv(csv_path)
        df['slide_id'] = norm_id     # label used in plots
        # Optional: collapse PID/BID labels to a single family ID
        if stain_type_key == "PID":
            # e.g., PID30B17.svs, BID… -> keep their specific IDs,
            # but you could also add a family tag if helpful:
            df['family'] = 'PID'     # unified family column
        all_data.append(df)
        print(f"✅ Loaded {norm_id}: {len(df):,} nuclei")

    if not all_data:
        raise ValueError(f"No slides found for stain type: {stain_type}")

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Total for {stain_type}: {len(combined):,} nuclei from {len(all_data)} slides\n")
    return combined


def select_features(df: pd.DataFrame, stain_type: str) -> tuple:
    """Select features based on stain type."""
    
    # Base morphological features
    features = [
        'area_px', 'perimeter_px', 'circularity', 'aspect_ratio',
        'eccentricity', 'solidity'
    ]
    
    # Spatial features
    features.extend([
        'density_um2_r50.0', 'density_um2_r100.0', 'density_um2_r150.0',
        'corrected_density_um2_r50.0', 'corrected_density_um2_r100.0', 
        'corrected_density_um2_r150.0'
    ])
    
    # Coherency
    features.extend([
        'coherency_50um', 'coherency_100um', 'coherency_150um'
    ])
    
    # Color
    features.extend(['r', 'g', 'b', 'gray_mean'])
    
    # Variance features
    variance_cols = [c for c in df.columns if 'local_variance' in c or 'local_cv' in c]
    features.extend(variance_cols[:10])  # Limit to avoid too many features
    
    # IHC-specific
    if stain_type != "H&E" and 'brown_intensity' in df.columns:
        features.append('brown_intensity')
        features.extend([c for c in df.columns if 'brown_density' in c])
    
    # Filter existing columns
    features = [f for f in features if f in df.columns]
    
    print(f"Selected {len(features)} features for {stain_type}")
    return df[features].values, features

def generate_umap(df: pd.DataFrame, stain_type: str, output_dir: Path):
    """Generate UMAP for stain type."""
    
    print(f"\n{'='*60}")
    print(f"Generating UMAP: {stain_type}")
    print(f"{'='*60}\n")
    
    # Get features
    X, feature_names = select_features(df, stain_type)
    X = np.nan_to_num(X, nan=0.0)
    
    # Standardize
    print("Standardizing...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # UMAP
    print("Running UMAP...")
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        n_components=2,
        random_state=42,
        verbose=True
    )
    embedding = reducer.fit_transform(X_scaled)
    
    # Cluster
    print("Clustering...")
    n_clusters = 20 if stain_type == "H&E" else 15
    clusterer = Birch(n_clusters=n_clusters, threshold=0.5)
    clusters = clusterer.fit_predict(embedding)
    
    # Add to dataframe
    df['umap_1'] = embedding[:, 0]
    df['umap_2'] = embedding[:, 1]
    df['umap_cluster'] = clusters
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f'{stain_type.replace("&", "and")}_combined.csv'
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved: {output_csv}")
    
    # Visualize
    visualize_umap(df, stain_type, output_dir, n_clusters)
    
    return df

def visualize_umap(df: pd.DataFrame, stain_type: str, output_dir: Path, n_clusters: int,
                   max_points=150_000, point_size=3, dpi=300):
    """
    Create readable 2x2 UMAP panels with big labels and cluster annotations.
    Saves both PNG (hi-dpi) and SVG (vector).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Sample for speed/clarity
    if len(df) > max_points:
        df_viz = df.sample(n=max_points, random_state=42)
    else:
        df_viz = df.copy()

    # ---- Figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), facecolor='white')
    (axC, axS), (axA, axD) = axes

    # =========================
    # (1) Clusters panel
    # =========================
    sc = axC.scatter(df_viz['umap_1'], df_viz['umap_2'],
                     c=df_viz['umap_cluster'], cmap='tab20',
                     s=point_size, alpha=0.7, rasterized=True)
    _format_axes(axC, title=f'{stain_type} — Clusters (k={n_clusters})')
    cb = plt.colorbar(sc, ax=axC, fraction=0.046, pad=0.04)
    _nice_colorbar(cb, 'Cluster ID')
    _annotate_cluster_centroids(axC, df_viz)

    # =========================
    # (2) By Slide panel (legend that’s actually readable)
    # =========================
    # Map each slide to an index
    slides = df_viz['slide_id'].astype(str).values
    uniq_slides = list(dict.fromkeys(slides))  # preserve order
    slide_index = {s: i for i, s in enumerate(uniq_slides)}
    sc2 = axS.scatter(df_viz['umap_1'], df_viz['umap_2'],
                      c=[slide_index[s] for s in slides],
                      cmap='tab20', s=point_size, alpha=0.7, rasterized=True)
    _format_axes(axS, title=f'{stain_type} — By Slide ({len(uniq_slides)} slides)')

    # Build a compact legend (collapse if too many)
    max_legend = 12
    handles = []
    shown = uniq_slides[:max_legend]
    for s in shown:
        color = plt.get_cmap('tab20')(slide_index[s] % 20)
        handles.append(Line2D([0], [0], marker='o', linestyle='',
                              markersize=7, markerfacecolor=color, markeredgecolor='none', label=s))
    if len(uniq_slides) > max_legend:
        handles.append(Line2D([0], [0], linestyle='', label=f'+ {len(uniq_slides)-max_legend} more…'))
    axS.legend(handles=handles, loc='upper right', frameon=True, ncol=1)

    # =========================
    # (3) Circularity panel
    # =========================
    if 'circularity' in df_viz.columns:
        sc3 = axA.scatter(df_viz['umap_1'], df_viz['umap_2'],
                          c=df_viz['circularity'], cmap='viridis',
                          s=point_size, alpha=0.7, rasterized=True, vmin=0.5, vmax=1.0)
        _format_axes(axA, title=f'{stain_type} — Circularity')
        cb3 = plt.colorbar(sc3, ax=axA, fraction=0.046, pad=0.04)
        _nice_colorbar(cb3, 'Circularity')
    else:
        _format_axes(axA, title=f'{stain_type} — Circularity (missing)')

    # =========================
    # (4) Marker or Density panel
    # =========================
    if stain_type != "H&E" and 'brown_intensity' in df_viz.columns:
        sc4 = axD.scatter(df_viz['umap_1'], df_viz['umap_2'],
                          c=df_viz['brown_intensity'], cmap='YlOrBr',
                          s=point_size, alpha=0.7, rasterized=True)
        _format_axes(axD, title=f'{stain_type} — Marker Intensity')
        cb4 = plt.colorbar(sc4, ax=axD, fraction=0.046, pad=0.04)
        _nice_colorbar(cb4, 'Brown intensity')
    else:
        dens_col = None
        for cand in ('density_um2_r150.0', 'corrected_density_um2_r150.0'):
            if cand in df_viz.columns:
                dens_col = cand
                break
        if dens_col:
            sc4 = axD.scatter(df_viz['umap_1'], df_viz['umap_2'],
                              c=df_viz[dens_col], cmap='plasma',
                              s=point_size, alpha=0.7, rasterized=True)
            _format_axes(axD, title=f'{stain_type} — Density (150µm)')
            cb4 = plt.colorbar(sc4, ax=axD, fraction=0.046, pad=0.04)
            _nice_colorbar(cb4, 'Cells / µm²')
        else:
            _format_axes(axD, title=f'{stain_type} — Density (missing)')

    # Tight & save (PNG + SVG)
    plt.tight_layout()
    base = output_dir / f'{stain_type.replace("&", "and")}_umap'
    fig.savefig(f'{base}.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{base}.svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ Saved: {base}.png  and  {base}.svg")
    
    # Generate additional comprehensive visualizations
    generate_cluster_stats(df_viz, stain_type, output_dir, dpi)


def generate_cluster_stats(df, stain_type, output_dir, dpi=300):
    """
    Generate cluster characterization figure with statistics.
    """
    if 'cluster' not in df.columns:
        return
    
    import seaborn as sns
    
    print(f"  Creating cluster statistics for {stain_type}...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Cluster size distribution
    ax1 = fig.add_subplot(gs[0, 0])
    cluster_counts = df['cluster'].value_counts().sort_index()
    cluster_counts.plot(kind='bar', ax=ax1, color='steelblue')
    ax1.set_xlabel('Cluster', fontsize=10)
    ax1.set_ylabel('Number of Cells', fontsize=10)
    ax1.set_title(f'{stain_type} — Cluster Sizes', fontsize=12, fontweight='bold')
    ax1.tick_params(labelsize=9)
    
    # 2. Feature means per cluster (if morphology available)
    ax2 = fig.add_subplot(gs[0, 1:])
    feature_cols = [c for c in df.columns if c in ['area', 'eccentricity', 'solidity', 
                                                     'aspect_ratio', 'perimeter', 'circularity']]
    if feature_cols:
        cluster_means = df.groupby('cluster')[feature_cols].mean()
        sns.heatmap(cluster_means.T, annot=True, fmt='.2f', cmap='viridis', 
                   ax=ax2, cbar_kws={'label': 'Feature Value'})
        ax2.set_xlabel('Cluster', fontsize=10)
        ax2.set_ylabel('Feature', fontsize=10)
        ax2.set_title('Morphological Features by Cluster', fontsize=12, fontweight='bold')
    
    # 3. Marker intensity distribution (if IHC)
    if stain_type != "H&E" and 'brown_intensity' in df.columns:
        ax3 = fig.add_subplot(gs[1, :])
        clusters = sorted(df['cluster'].unique())
        data_to_plot = [df[df['cluster']==c]['brown_intensity'].values for c in clusters]
        bp = ax3.boxplot(data_to_plot, labels=clusters, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax3.set_xlabel('Cluster', fontsize=10)
        ax3.set_ylabel('Brown Intensity', fontsize=10)
        ax3.set_title(f'{stain_type} Marker Intensity Distribution', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
    
    # 4. Spatial distribution (if coordinates available)
    if 'centroid_x' in df.columns and 'centroid_y' in df.columns:
        ax4 = fig.add_subplot(gs[2, :])
        for cluster in sorted(df['cluster'].unique()):
            cluster_df = df[df['cluster'] == cluster]
            ax4.scatter(cluster_df['centroid_x'], cluster_df['centroid_y'], 
                       s=1, alpha=0.5, label=f'C{cluster}')
        ax4.set_xlabel('X Position', fontsize=10)
        ax4.set_ylabel('Y Position', fontsize=10)
        ax4.set_title('Spatial Distribution of Clusters', fontsize=12, fontweight='bold')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax4.set_aspect('equal')
    
    # Save
    base = output_dir / f'{stain_type.replace("&", "and")}_cluster_stats'
    fig.savefig(f'{base}.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ Saved cluster stats: {base}.png")
    
    # Generate violin plots for feature distributions
    generate_violin_plots(df, stain_type, output_dir, dpi)


def generate_violin_plots(df, stain_type, output_dir, dpi=300):
    """
    Generate violin plots showing feature distributions per cluster.
    Inspired by HistoVision's detailed distribution visualizations.
    """
    if 'cluster' not in df.columns:
        return
    
    import seaborn as sns
    
    print(f"  Creating violin plots for {stain_type}...")
    
    # Select key features for visualization
    feature_cols = []
    for feat in ['area_px', 'circularity', 'eccentricity', 'aspect_ratio']:
        if feat in df.columns:
            feature_cols.append(feat)
    
    # Add brown intensity for IHC
    if stain_type != "H&E" and 'brown_intensity' in df.columns:
        feature_cols.append('brown_intensity')
    
    if not feature_cols:
        return
    
    # Create violin plot figure
    n_features = len(feature_cols)
    fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 5))
    if n_features == 1:
        axes = [axes]
    
    for ax, feature in zip(axes, feature_cols):
        # Get clusters sorted by count
        cluster_order = df['cluster'].value_counts().index.tolist()
        
        # Create violin plot
        sns.violinplot(data=df, x='cluster', y=feature, order=cluster_order,
                      ax=ax, palette='Set2', scale='width', inner='quartile')
        
        ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_title(f'{feature.replace("_", " ").title()} Distribution', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    base = output_dir / f'{stain_type.replace("&", "and")}_violin_plots'
    fig.savefig(f'{base}.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ Saved violin plots: {base}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--stain_types', nargs='+', 
                       default=['H&E', 'CD3', 'GFAP', 'Iba1', 'NF', 'PGP9.5',  'PID'])
    parser.add_argument('--max_points', type=int, default=150_000)
    parser.add_argument('--point_size', type=float, default=3.0)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    print(f"\n{'='*70}")
    print("SEPARATE UMAP ANALYSIS BY STAIN TYPE")
    print(f"{'='*70}\n")
    
    for stain_type in args.stain_types:
        try:
            df = load_all_slides_by_type(results_dir, stain_type)
            generate_umap(df, stain_type, output_dir / stain_type.replace('&', 'and'))
            print(f"\n✅ Completed {stain_type}\n")
        except Exception as e:
            print(f"\n❌ Error with {stain_type}: {e}\n")
    
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
