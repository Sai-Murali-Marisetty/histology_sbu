#!/usr/bin/env python3
"""
09_umap_clustering.py - UMAP dimensionality reduction + clustering

Creates low-dimensional embeddings of nuclear features and identifies
cell populations using BIRCH/KMeans clustering.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from umap import UMAP
from sklearn.cluster import Birch, KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import sys
import pickle
import warnings

# Increase PIL decompression bomb limit for large whole slide images
Image.MAX_IMAGE_PIXELS = None

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import SlideConfig


def select_features(df, slide_type, config=None):
    """
    Select appropriate features for clustering based on slide type.
    
    Args:
        df: DataFrame with all features
        slide_type: Type of slide (H&E, IHC_CD3, etc.)
        config: SlideConfig object
    
    Returns:
        feature_matrix, feature_names
    """
    if config is None:
        config = SlideConfig()
    
    # Get configured features
    clust_params = config.get_clustering_params(slide_type)
    feature_list = clust_params.get('features', [])
    
    # Filter to available columns
    available = [f for f in feature_list if f in df.columns]
    
    if not available:
        # Fallback to basic features
        print("Warning: No configured features found, using basic features")
        available = [c for c in df.columns if c in [
            'area_px', 'aspect_ratio', 'circularity', 'eccentricity',
            'r', 'g', 'b', 'density_um2_r100.0', 'coherency_150um'
        ]]
    
    print(f"Using {len(available)} features for clustering:")
    for f in available:
        print(f"  - {f}")
    
    return df[available].fillna(0), available


def run_umap(features, n_components=2, n_neighbors=30, min_dist=0.1, random_state=42, use_pca=True, n_pca_components=10):
    """
    Run UMAP dimensionality reduction with optional PCA preprocessing.
    
    Args:
        features: Input feature matrix
        n_components: UMAP output dimensions
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        random_state: Random seed
        use_pca: Whether to apply PCA before UMAP (recommended for >10 features)
        n_pca_components: Number of PCA components (if use_pca=True)
    
    Returns:
        embedding: UMAP embedding
    """
    from sklearn.decomposition import PCA
    
    input_features = features
    
    # Apply PCA preprocessing if requested (HistoVision approach)
    if use_pca and features.shape[1] > n_pca_components:
        print(f"\n🔄 Pre-processing with PCA ({features.shape[1]} → {n_pca_components} components)...")
        pca = PCA(n_components=n_pca_components)
        input_features = pca.fit_transform(features)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"   Explained variance: {explained_var:.1%}")
    
    print(f"\n🔄 Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        verbose=False
    )
    
    embedding = reducer.fit_transform(input_features)
    print("✓ UMAP complete")
    
    return embedding


def cluster_cells(embedding, n_clusters=20, method='birch', min_cluster_size=50):
    """
    Cluster cells using BIRCH, KMeans, or HDBSCAN.
    
    Args:
        embedding: UMAP embedding (n_samples, 2)
        n_clusters: Number of clusters (ignored for HDBSCAN)
        method: 'birch', 'kmeans', or 'hdbscan'
        min_cluster_size: Minimum cluster size for HDBSCAN
    
    Returns:
        cluster labels
    """
    print(f"\n🔗 Clustering with {method.upper()}...")
    
    if method.lower() == 'hdbscan':
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=10,
                cluster_selection_epsilon=0.0,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            labels = clusterer.fit_predict(embedding)
            
            # Count clusters (excluding noise label -1)
            unique_labels = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            print(f"✓ Found {unique_labels} clusters + {n_noise:,} noise points")
            
        except ImportError:
            print("⚠️ HDBSCAN not installed, falling back to BIRCH")
            clusterer = Birch(n_clusters=n_clusters, threshold=0.5)
            labels = clusterer.fit_predict(embedding)
            unique_labels = len(np.unique(labels))
            print(f"✓ Found {unique_labels} clusters")
    
    elif method.lower() == 'birch':
        clusterer = Birch(n_clusters=n_clusters, threshold=0.5)
        labels = clusterer.fit_predict(embedding)
        unique_labels = len(np.unique(labels))
        print(f"✓ Found {unique_labels} clusters")
    
    else:  # kmeans
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(embedding)
        unique_labels = len(np.unique(labels))
        print(f"✓ Found {unique_labels} clusters")
    
    return labels


def plot_umap_panels(df, embedding, labels, out_path, feature_cols=None):
    """Create multi-panel UMAP visualization"""
    
    if feature_cols is None:
        feature_cols = ['circularity', 'density_um2_r100.0', 'coherency_150um']
    
    # Filter to available features
    feature_cols = [f for f in feature_cols if f in df.columns]
    
    n_panels = 1 + len(feature_cols)  # Clusters + features
    fig, axes = plt.subplots(1, n_panels, figsize=(6*n_panels, 5))
    
    if n_panels == 1:
        axes = [axes]
    
    # Panel 1: Clusters
    scatter = axes[0].scatter(embedding[:, 0], embedding[:, 1], 
                             c=labels, cmap='tab20', s=1, alpha=0.5)
    axes[0].set_title(f'UMAP Clusters (n={len(np.unique(labels))})', 
                     fontsize=14, fontweight='bold')
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')
    plt.colorbar(scatter, ax=axes[0], label='Cluster')
    
    # Additional panels: Features
    for i, feat in enumerate(feature_cols, start=1):
        vmin, vmax = df[feat].quantile([0.02, 0.98])
        scatter = axes[i].scatter(embedding[:, 0], embedding[:, 1], 
                                 c=df[feat], cmap='viridis', s=1, alpha=0.5,
                                 vmin=vmin, vmax=vmax)
        axes[i].set_title(feat.replace('_', ' ').title(), fontsize=12)
        axes[i].set_xlabel('UMAP 1')
        plt.colorbar(scatter, ax=axes[i], label=feat)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved UMAP panels: {out_path}")


def plot_cluster_features(df, labels, out_path, features=None):
    """Plot feature distributions per cluster"""
    
    if features is None:
        features = ['area_px', 'aspect_ratio', 'circularity']
    
    # Filter to available
    features = [f for f in features if f in df.columns]
    
    if not features:
        print("Warning: No features available for cluster visualization")
        return
    
    n_clusters = len(np.unique(labels))
    top_clusters = sorted(pd.Series(labels).value_counts().head(10).index)
    
    fig, axes = plt.subplots(1, len(features), figsize=(5*len(features), 4))
    if len(features) == 1:
        axes = [axes]
    
    df_with_labels = df.copy()
    df_with_labels['cluster'] = labels
    
    for i, feat in enumerate(features):
        for cluster_id in top_clusters:
            cluster_data = df_with_labels[df_with_labels['cluster'] == cluster_id][feat]
            axes[i].hist(cluster_data, bins=30, alpha=0.5, label=f'C{cluster_id}')
        
        axes[i].set_xlabel(feat.replace('_', ' ').title())
        axes[i].set_ylabel('Count')
        axes[i].legend(fontsize=8, ncol=2)
        axes[i].set_title(f'{feat}')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved cluster features: {out_path}")


def plot_cluster_spatial(df, labels, thumb_path, out_path):
    """Overlay clusters on tissue image"""
    
    if not Path(thumb_path).exists():
        print(f"Warning: Thumbnail not found: {thumb_path}")
        return
    
    img = np.array(Image.open(thumb_path).convert('RGB'))
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img)
    
    # Use tab20 colormap for clusters
    n_clusters = len(np.unique(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    
    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        ax.scatter(df.loc[mask, 'x'], df.loc[mask, 'y'], 
                  c=[colors[cluster_id]], s=3, alpha=0.6, 
                  label=f'C{cluster_id}')
    
    ax.set_title('Cluster Spatial Distribution', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Only show legend if reasonable number of clusters
    if n_clusters <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                 markerscale=3, fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved cluster spatial map: {out_path}")


def compute_cluster_stats(df, labels, feature_cols):
    """Compute statistics per cluster"""
    
    df_with_labels = df.copy()
    df_with_labels['cluster'] = labels
    
    stats = []
    for cluster_id in sorted(np.unique(labels)):
        cluster_df = df_with_labels[df_with_labels['cluster'] == cluster_id]
        
        stat = {
            'cluster': int(cluster_id),
            'n_cells': len(cluster_df),
            'pct_total': len(cluster_df) / len(df) * 100
        }
        
        for feat in feature_cols:
            if feat in cluster_df.columns:
                stat[f'{feat}_mean'] = cluster_df[feat].mean()
                stat[f'{feat}_std'] = cluster_df[feat].std()
        
        stats.append(stat)
    
    return pd.DataFrame(stats)


def create_hierarchical_labels(df, labels, n_subclusters=4):
    """
    Create hierarchical cluster labels with sub-clusters.
    Inspired by HistoVision's approach: 0a, 0b, 0c, 0d, 1a, 1b, etc.
    
    Args:
        df: DataFrame with features
        labels: Primary cluster labels
        n_subclusters: Number of sub-clusters per main cluster
    
    Returns:
        hierarchical_labels: List of labels like '0a', '0b', '1a', '1b', etc.
    """
    from sklearn.cluster import KMeans
    
    hierarchical_labels = []
    
    for cluster_id in sorted(np.unique(labels)):
        # Get nuclei in this cluster
        mask = labels == cluster_id
        cluster_indices = np.where(mask)[0]
        
        if len(cluster_indices) < n_subclusters:
            # Too few points for sub-clustering, just label them all the same
            for _ in cluster_indices:
                hierarchical_labels.append(f"{cluster_id}a")
            continue
        
        # Get embedding for this cluster
        cluster_embedding = df.loc[mask, ['umap_1', 'umap_2']].values
        
        # Sub-cluster within this main cluster
        sub_clusterer = KMeans(n_clusters=min(n_subclusters, len(cluster_indices)), 
                              random_state=42, n_init=10)
        sub_labels = sub_clusterer.fit_predict(cluster_embedding)
        
        # Create hierarchical labels (0a, 0b, 0c, 0d, etc.)
        letter_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        for idx, sub_label in zip(cluster_indices, sub_labels):
            letter = letter_labels[sub_label] if sub_label < len(letter_labels) else chr(97 + sub_label)
            hierarchical_labels.append(f"{cluster_id}{letter}")
    
    return np.array(hierarchical_labels)


def save_umap_model(umap_model, pca_model, output_dir, prefix='umap'):
    """
    Save trained UMAP and PCA models for reuse.
    Inspired by HistoVision's model serialization approach.
    
    Args:
        umap_model: Trained UMAP object
        pca_model: Trained PCA object (or None)
        output_dir: Directory to save models
        prefix: Filename prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save UMAP model
    umap_path = output_dir / f"{prefix}_model.pkl"
    with open(umap_path, 'wb') as f:
        pickle.dump(umap_model, f)
    print(f"✓ Saved UMAP model to {umap_path}")
    
    # Save PCA model if provided
    if pca_model is not None:
        pca_path = output_dir / f"{prefix}_pca_model.pkl"
        with open(pca_path, 'wb') as f:
            pickle.dump(pca_model, f)
        print(f"✓ Saved PCA model to {pca_path}")


def load_umap_model(model_dir, prefix='umap'):
    """
    Load trained UMAP and PCA models.
    
    Args:
        model_dir: Directory containing models
        prefix: Filename prefix
    
    Returns:
        umap_model, pca_model (or None if PCA not found)
    """
    model_dir = Path(model_dir)
    
    # Load UMAP model
    umap_path = model_dir / f"{prefix}_model.pkl"
    if not umap_path.exists():
        raise FileNotFoundError(f"UMAP model not found: {umap_path}")
    
    with open(umap_path, 'rb') as f:
        umap_model = pickle.load(f)
    print(f"✓ Loaded UMAP model from {umap_path}")
    
    # Load PCA model if available
    pca_path = model_dir / f"{prefix}_pca_model.pkl"
    pca_model = None
    if pca_path.exists():
        with open(pca_path, 'rb') as f:
            pca_model = pickle.load(f)
        print(f"✓ Loaded PCA model from {pca_path}")
    
    return umap_model, pca_model


def filter_nuclei_by_size(df, min_area=None, max_area=None):
    """
    Filter nuclei by area thresholds.
    Inspired by HistoVision's size_thresh.py approach.
    
    Args:
        df: DataFrame with 'area_px' column
        min_area: Minimum area threshold (pixels)
        max_area: Maximum area threshold (pixels)
    
    Returns:
        filtered_df: DataFrame with nuclei within size range
        removed_count: Number of nuclei removed
    """
    original_count = len(df)
    
    filtered_df = df.copy()
    
    # Check which area column exists
    area_col = None
    for col in ['area_px', 'area', 'Area']:
        if col in filtered_df.columns:
            area_col = col
            break
    
    if area_col is None:
        print("⚠️  Warning: No area column found. Skipping size filtering.")
        return filtered_df, 0
    
    if min_area is not None:
        filtered_df = filtered_df[filtered_df[area_col] >= min_area]
    
    if max_area is not None:
        filtered_df = filtered_df[filtered_df[area_col] <= max_area]
    
    removed_count = original_count - len(filtered_df)
    
    if removed_count > 0:
        pct_removed = removed_count / original_count * 100
        print(f"✓ Size filtering: removed {removed_count:,} nuclei ({pct_removed:.1f}%)")
        if min_area:
            print(f"  Min area: {min_area} px²")
        if max_area:
            print(f"  Max area: {max_area} px²")
    
    return filtered_df, removed_count





def main():
    ap = argparse.ArgumentParser(description="UMAP clustering analysis")
    ap.add_argument('--input_csv', required=True, help="Input features CSV")
    ap.add_argument('--output_csv', required=True, help="Output CSV with clusters")
    ap.add_argument('--out_dir', required=True, help="Output directory")
    ap.add_argument('--thumb', help="Slide thumbnail for spatial visualization")
    ap.add_argument('--slide_type', default='H&E', help="Slide type")
    ap.add_argument('--config', help="Config file path")
    ap.add_argument('--n_clusters', type=int, help="Override number of clusters")
    ap.add_argument('--n_neighbors', type=int, help="Override UMAP n_neighbors")
    ap.add_argument('--method', default='birch', choices=['birch', 'kmeans', 'hdbscan'], help="Clustering method")
    ap.add_argument('--use_pca', action='store_true', help="Apply PCA preprocessing before UMAP")
    ap.add_argument('--n_pca', type=int, default=10, help="Number of PCA components")
    ap.add_argument('--min_cluster_size', type=int, default=50, help="Min cluster size for HDBSCAN")
    ap.add_argument('--hierarchical', action='store_true', help="Create hierarchical sub-cluster labels")
    ap.add_argument('--n_subclusters', type=int, default=4, help="Number of sub-clusters per main cluster")
    ap.add_argument('--save_model', action='store_true', help="Save trained UMAP/PCA models")
    ap.add_argument('--load_model', help="Load trained models from directory")
    ap.add_argument('--min_area', type=float, help="Minimum nucleus area (pixels)")
    ap.add_argument('--max_area', type=float, help="Maximum nucleus area (pixels)")
    args = ap.parse_args()
    
    # Load config
    cfg = SlideConfig(args.config) if args.config else SlideConfig()
    clust_params = cfg.get_clustering_params(args.slide_type)
    
    n_clusters = args.n_clusters or clust_params.get('n_clusters', 20)
    n_neighbors = args.n_neighbors or clust_params.get('n_neighbors', 30)
    
    print(f"\n{'='*80}")
    print(f"UMAP CLUSTERING ANALYSIS - {args.slide_type}")
    print(f"{'='*80}")
    print(f"Clusters: {n_clusters if args.method != 'hdbscan' else 'auto (HDBSCAN)'}")
    print(f"UMAP neighbors: {n_neighbors}")
    print(f"Method: {args.method}")
    print(f"PCA preprocessing: {args.use_pca}")
    if args.use_pca:
        print(f"PCA components: {args.n_pca}")
    print(f"Hierarchical labeling: {args.hierarchical}")
    if args.hierarchical:
        print(f"Sub-clusters per main: {args.n_subclusters}")
    print(f"Size filtering: min={args.min_area}, max={args.max_area}")
    
    # Load data
    print(f"\n📊 Loading features...")
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df):,} nuclei with {len(df.columns)} features")
    
    # Apply size filtering if requested
    if args.min_area or args.max_area:
        df, removed = filter_nuclei_by_size(df, args.min_area, args.max_area)
    
    # Select features
    features, feature_names = select_features(df, args.slide_type, cfg)
    
    # Normalize
    print(f"\n⚖️  Normalizing features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # UMAP with optional PCA preprocessing
    embedding = run_umap(features_scaled, n_neighbors=n_neighbors, 
                        use_pca=args.use_pca, n_pca_components=args.n_pca)
    df['umap_1'] = embedding[:, 0]
    df['umap_2'] = embedding[:, 1]
    
    # Clustering
    labels = cluster_cells(embedding, n_clusters=n_clusters, method=args.method,
                          min_cluster_size=args.min_cluster_size)
    df['cluster'] = labels
    
    # Create hierarchical labels if requested
    if args.hierarchical:
        print(f"\n🔀 Creating hierarchical sub-cluster labels...")
        hierarchical_labels = create_hierarchical_labels(df, labels, args.n_subclusters)
        df['cluster_hierarchical'] = hierarchical_labels
        print(f"Created {len(np.unique(hierarchical_labels))} sub-clusters")
    
    # Save models if requested
    if args.save_model:
        print(f"\n💾 Saving models...")
        # Note: We'd need to refactor run_umap to return the models
        # For now, just indicate where they would be saved
        print(f"Models would be saved to {args.out_dir}/models/")
    
    # Save enriched CSV
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\n💾 Saved: {args.output_csv}")
    
    # Visualizations
    print(f"\n🎨 Creating visualizations...")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    plot_umap_panels(df, embedding, labels, 
                    Path(args.out_dir) / 'umap_clusters.png',
                    feature_names[:3])
    
    plot_cluster_features(df, labels,
                         Path(args.out_dir) / 'cluster_features.png',
                         feature_names[:3])
    
    if args.thumb:
        from PIL import Image
        plot_cluster_spatial(df, labels, args.thumb,
                           Path(args.out_dir) / 'cluster_spatial.png')
    
    # Cluster statistics
    stats_df = compute_cluster_stats(df, labels, feature_names)
    stats_df.to_csv(Path(args.out_dir) / 'cluster_statistics.csv', index=False)
    print(f"✓ Saved cluster statistics")
    
    # Print summary
    print(f"\n{'='*80}")
    print("CLUSTER SUMMARY")
    print(f"{'='*80}")
    print(f"\nCluster sizes:")
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        pct = count / len(labels) * 100
        print(f"  Cluster {cluster_id}: {count:,} cells ({pct:.1f}%)")
    
    print(f"\n✅ Clustering complete!")
    print(f"📂 Results saved to: {args.out_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
