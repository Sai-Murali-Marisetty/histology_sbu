#!/usr/bin/env python3
"""
07_ihc_brown_stain.py - IHC brown (DAB) stain quantification

Detects and quantifies DAB (brown) staining in immunohistochemistry slides.
Uses color deconvolution to separate hematoxylin and DAB channels.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import sys

# Increase PIL decompression bomb limit for large whole slide images
Image.MAX_IMAGE_PIXELS = None

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import SlideConfig

# Color deconvolution matrix for H-DAB (Ruifrok & Johnston, 2001)
# Optical density (OD) from RGB
RGB_TO_OD = np.array([
    [1.88, 0.07, 0.00],  # Hematoxylin (blue/purple)
    [0.00, 1.00, 0.57],  # DAB (brown)
    [0.00, 0.00, 1.00],  # Residual
])


def rgb_to_od(rgb):
    """Convert RGB to optical density"""
    rgb = np.maximum(rgb, 1e-6) / 255.0  # Normalize and avoid log(0)
    return -np.log10(rgb)


def separate_stains(img):
    """
    Separate H-DAB stains using color deconvolution.
    
    Args:
        img: RGB image (H x W x 3)
    
    Returns:
        hematoxylin_channel, dab_channel (both H x W)
    """
    h, w = img.shape[:2]
    
    # Convert to OD
    od = rgb_to_od(img)
    
    # Reshape for matrix multiplication
    od_flat = od.reshape(-1, 3)
    
    # Deconvolve
    stains = np.dot(od_flat, RGB_TO_OD.T)
    stains = stains.reshape(h, w, 3)
    
    # Return H and DAB channels
    return stains[:,:,0], stains[:,:,1]


def detect_brown_per_nucleus(df, tiles_dir, threshold=0.15, radius_multiplier=1.5):
    """
    Detect brown stain for each nucleus.
    
    Args:
        df: DataFrame with nucleus features (must have 'tile', 'cx', 'cy', 'major_axis_length')
        tiles_dir: Path to directory with tile images
        threshold: DAB intensity threshold for positive detection
        radius_multiplier: How much to expand beyond nucleus (1.5 = 150% of major axis)
    
    Returns:
        DataFrame with 'nucleus_id', 'has_brown', 'brown_intensity'
    """
    tiles_dir = Path(tiles_dir)
    results = []
    
    # Process tile by tile
    for tile_name in df['tile'].unique():
        tile_path = tiles_dir / tile_name
        if not tile_path.exists():
            print(f"Warning: Tile not found: {tile_name}")
            continue
        
        # Load tile and separate stains
        img = np.array(Image.open(tile_path).convert('RGB'))
        _, dab = separate_stains(img)
        
        # Get nuclei in this tile
        tile_df = df[df['tile'] == tile_name]
        
        for idx, nucleus in tile_df.iterrows():
            cx = int(nucleus.get('cx', nucleus.get('x_px', 0)))
            cy = int(nucleus.get('cy', nucleus.get('y_px', 0)))
            radius = int(nucleus['major_axis_length'] * radius_multiplier)
            
            # Extract neighborhood
            x0 = max(0, cx - radius)
            x1 = min(img.shape[1], cx + radius)
            y0 = max(0, cy - radius)
            y1 = min(img.shape[0], cy + radius)
            
            dab_roi = dab[y0:y1, x0:x1]
            
            if dab_roi.size == 0:
                brown_mean = 0.0
                has_brown = 0
            else:
                brown_mean = float(dab_roi.mean())
                has_brown = 1 if brown_mean > threshold else 0
            
            results.append({
                'nucleus_id': nucleus.get('nucleus_id', idx),
                'has_brown': has_brown,
                'brown_intensity': brown_mean
            })
    
    return pd.DataFrame(results)


def compute_brown_density(df, radius_um):
    """
    Compute density of brown-positive nuclei in local neighborhoods.
    
    Args:
        df: DataFrame with 'x_um', 'y_um', 'has_brown'
        radius_um: Neighborhood radius in micrometers
    
    Returns:
        np.array: Brown-positive density for each nucleus
    """
    coords = np.vstack([df['x_um'], df['y_um']]).T
    tree = cKDTree(coords)
    has_brown = df['has_brown'].to_numpy()
    
    brown_density = []
    for i, p in enumerate(coords):
        idx = tree.query_ball_point(p, radius_um)
        if len(idx) > 0:
            brown_count = has_brown[idx].sum()
            total_count = len(idx)
            density = brown_count / total_count  # Fraction
        else:
            density = 0.0
        brown_density.append(density)
    
    return np.array(brown_density)


def visualize_brown_overlay(df, thumb_path, out_path):
    """Create visualization showing brown-positive nuclei"""
    if not Path(thumb_path).exists():
        print(f"Warning: Thumbnail not found: {thumb_path}")
        return
    
    img = np.array(Image.open(thumb_path).convert('RGB'))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Panel 1: Binary brown/not-brown
    ax1.imshow(img)
    brown_pos = df[df['has_brown'] == 1]
    brown_neg = df[df['has_brown'] == 0]
    
    ax1.scatter(brown_neg['x'], brown_neg['y'], c='blue', s=2, alpha=0.3, label='Brown-')
    ax1.scatter(brown_pos['x'], brown_pos['y'], c='red', s=5, alpha=0.7, label='Brown+')
    ax1.legend(loc='upper right', markerscale=3)
    ax1.set_title(f'Brown Stain Detection\n{len(brown_pos):,} / {len(df):,} positive ({len(brown_pos)/len(df)*100:.1f}%)')
    ax1.axis('off')
    
    # Panel 2: Brown intensity heatmap
    ax2.imshow(img)
    sc = ax2.scatter(df['x'], df['y'], c=df['brown_intensity'], 
                     cmap='YlOrBr', s=5, alpha=0.8, 
                     vmin=0, vmax=df['brown_intensity'].quantile(0.98))
    ax2.set_title('Brown Intensity')
    ax2.axis('off')
    plt.colorbar(sc, ax=ax2, shrink=0.5, label='DAB Intensity')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved brown stain overlay: {out_path}")


def visualize_brown_density(df, thumb_path, out_path, radius_um=100):
    """Visualize brown-positive density"""
    if not Path(thumb_path).exists():
        return
    
    img = np.array(Image.open(thumb_path).convert('RGB'))
    fig, ax = plt.subplots(figsize=(12, 12))
    
    ax.imshow(img)
    density_col = f'brown_density_{int(radius_um)}um'
    if density_col in df.columns:
        sc = ax.scatter(df['x'], df['y'], c=df[density_col], 
                       cmap='RdYlBu_r', s=5, alpha=0.7, vmin=0, vmax=1)
        ax.set_title(f'Brown+ Density ({int(radius_um)}µm neighborhoods)')
        plt.colorbar(sc, ax=ax, shrink=0.5, label='Fraction Brown+')
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved brown density map: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="IHC brown stain quantification")
    ap.add_argument('--input_csv', required=True, help="Input nuclei features CSV")
    ap.add_argument('--tiles_dir', required=True, help="Directory with tile images")
    ap.add_argument('--output_csv', required=True, help="Output CSV with brown features")
    ap.add_argument('--thumb', required=True, help="Slide thumbnail for visualization")
    ap.add_argument('--out_dir', required=True, help="Output directory for visualizations")
    ap.add_argument('--slide_type', default='IHC_CD3', help="Slide type for config")
    ap.add_argument('--config', help="Path to config file (optional)")
    ap.add_argument('--radii_um', nargs='+', type=float, help="Override density radii")
    args = ap.parse_args()
    
    # Load configuration
    cfg = SlideConfig(args.config) if args.config else SlideConfig()
    brown_params = cfg.get_brown_params(args.slide_type)
    
    threshold = brown_params.get('threshold', 0.15)
    radius_mult = brown_params.get('neighborhood_radius', 1.5)
    radii_um = args.radii_um if args.radii_um else cfg.get_density_radii(args.slide_type)
    
    print(f"\n{'='*80}")
    print(f"IHC BROWN STAIN ANALYSIS - {args.slide_type}")
    print(f"{'='*80}")
    print(f"Threshold: {threshold}")
    print(f"Neighborhood radius: {radius_mult}x nucleus size")
    print(f"Density radii: {radii_um} µm")
    
    # Load data
    print(f"\n📊 Loading nuclei features...")
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df):,} nuclei")
    
    # Detect brown stain
    print(f"\n🔬 Detecting brown stain...")
    brown_df = detect_brown_per_nucleus(df, args.tiles_dir, threshold, radius_mult)
    
    # Merge with original dataframe
    df = df.merge(brown_df, on='nucleus_id', how='left')
    
    # Fill NaN values (nuclei where detection failed)
    df['has_brown'].fillna(0, inplace=True)
    df['brown_intensity'].fillna(0.0, inplace=True)
    
    n_brown = df['has_brown'].sum()
    pct_brown = n_brown / len(df) * 100
    print(f"✓ Brown+ nuclei: {int(n_brown):,} ({pct_brown:.1f}%)")
    print(f"✓ Mean brown intensity: {df['brown_intensity'].mean():.3f}")
    
    # Compute brown density at multiple radii
    print(f"\n📈 Computing brown density...")
    for radius in radii_um:
        density_col = f'brown_density_{int(radius)}um'
        df[density_col] = compute_brown_density(df, radius)
        mean_density = df[density_col].mean()
        print(f"  {radius}µm: mean={mean_density:.3f}")
    
    # Save enriched CSV
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\n💾 Saved enriched CSV: {args.output_csv}")
    
    # Visualizations
    print(f"\n🎨 Creating visualizations...")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    visualize_brown_overlay(df, args.thumb, 
                           Path(args.out_dir) / 'brown_stain_overlay.jpg')
    
    for radius in radii_um:
        visualize_brown_density(df, args.thumb,
                               Path(args.out_dir) / f'brown_density_{int(radius)}um.jpg',
                               radius)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total nuclei: {len(df):,}")
    print(f"Brown+ nuclei: {int(n_brown):,} ({pct_brown:.1f}%)")
    print(f"Brown intensity: {df['brown_intensity'].mean():.3f} ± {df['brown_intensity'].std():.3f}")
    
    for radius in radii_um:
        density_col = f'brown_density_{int(radius)}um'
        print(f"\nBrown density ({int(radius)}µm):")
        print(f"  Mean: {df[density_col].mean():.3f}")
        print(f"  Median: {df[density_col].median():.3f}")
        print(f"  P10-P90: {df[density_col].quantile(0.1):.3f} - {df[density_col].quantile(0.9):.3f}")
    
    print(f"\n✅ Brown stain analysis complete!")
    print(f"📂 Results saved to: {args.out_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
