#!/usr/bin/env python3
"""
generate_feature_maps.py - Generate Feature Color Maps with Validation

Creates three-panel visualizations for each feature:
1. Color map alone (feature values)
2. Raw H&E image
3. Color map overlaid on H&E (semi-transparent)

This allows visual validation of computed features against tissue morphology.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from PIL import Image
from pathlib import Path
import openslide
from tqdm import tqdm

# Increase PIL decompression bomb limit for large whole slide images
Image.MAX_IMAGE_PIXELS = None


def load_slide_thumbnail(slide_path, target_size=2000):
    """Load H&E thumbnail from slide"""
    print(f"Loading slide: {slide_path}")
    slide = openslide.OpenSlide(str(slide_path))
    
    # Get thumbnail
    thumb = slide.get_thumbnail((target_size, target_size))
    thumb_array = np.array(thumb.convert('RGB'))
    
    # Get dimensions for coordinate scaling
    level0_dims = slide.level_dimensions[0]
    thumb_dims = thumb.size
    
    scale_x = thumb_dims[0] / level0_dims[0]
    scale_y = thumb_dims[1] / level0_dims[1]
    
    slide.close()
    
    return thumb_array, scale_x, scale_y


def create_feature_map(df, feature, thumb_shape, scale_x, scale_y, 
                       colormap='viridis', vmin=None, vmax=None,
                       percentile_clip=(2, 98)):
    """
    Create a feature color map
    
    Args:
        df: DataFrame with x, y coordinates and feature values
        feature: Feature column name
        thumb_shape: Shape of thumbnail (H, W, C)
        scale_x, scale_y: Coordinate scaling factors
        colormap: Matplotlib colormap name
        vmin, vmax: Value range (if None, use percentile_clip)
        percentile_clip: Percentile range for clipping outliers
    """
    # Get feature values
    values = df[feature].values
    
    # Handle NaN values
    valid_mask = ~np.isnan(values)
    if valid_mask.sum() == 0:
        print(f"  ⚠️  All NaN values for {feature}, skipping")
        return None, None, None
    
    # Clip outliers using percentiles if vmin/vmax not provided
    if vmin is None:
        vmin = np.percentile(values[valid_mask], percentile_clip[0])
    if vmax is None:
        vmax = np.percentile(values[valid_mask], percentile_clip[1])
    
    # Normalize values
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(colormap)
    
    # Create empty map
    map_shape = (thumb_shape[0], thumb_shape[1], 4)  # RGBA
    feature_map = np.zeros(map_shape, dtype=np.float32)
    
    # Convert coordinates to thumbnail space
    x_coords = (df['x'].values * scale_x).astype(int)
    y_coords = (df['y'].values * scale_y).astype(int)
    
    # Clip to image bounds
    valid_coords = (
        (x_coords >= 0) & (x_coords < thumb_shape[1]) &
        (y_coords >= 0) & (y_coords < thumb_shape[0]) &
        valid_mask
    )
    
    # Map feature values to colors
    for i in np.where(valid_coords)[0]:
        x, y = x_coords[i], y_coords[i]
        val = values[i]
        color = cmap(norm(val))
        
        # Draw a small circle (3x3 pixels) for visibility
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = y + dy, x + dx
                if 0 <= ny < thumb_shape[0] and 0 <= nx < thumb_shape[1]:
                    feature_map[ny, nx] = color
    
    return feature_map, vmin, vmax


def create_three_panel_figure(thumb, feature_map, feature_name, 
                               vmin, vmax, colormap, output_path):
    """
    Create three-panel visualization:
    1. Color map alone
    2. Raw H&E
    3. Overlay (50% transparent)
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # Panel 1: Feature map alone
    ax = axes[0]
    # Show only non-zero pixels
    mask = feature_map[:, :, 3] > 0
    feature_img = feature_map.copy()
    feature_img[~mask] = [1, 1, 1, 1]  # White background
    ax.imshow(feature_img)
    ax.set_title(f'{feature_name}\n(Color Map Only)', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Panel 2: Raw H&E
    ax = axes[1]
    ax.imshow(thumb)
    ax.set_title('H&E Image\n(Original Tissue)', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Panel 3: Overlay
    ax = axes[2]
    ax.imshow(thumb)
    # Overlay with 50% transparency
    overlay = feature_map.copy()
    overlay[:, :, 3] = overlay[:, :, 3] * 0.5  # 50% transparent
    ax.imshow(overlay)
    ax.set_title(f'{feature_name}\n(Overlay on H&E)', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=colormap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes, orientation='horizontal', 
                       fraction=0.05, pad=0.05, aspect=40)
    cbar.set_label(feature_name, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate feature color maps with validation (3-panel visualizations)"
    )
    parser.add_argument('--input_csv', required=True, help='Path to enriched features CSV')
    parser.add_argument('--slide', required=True, help='Path to .svs slide file')
    parser.add_argument('--output_dir', required=True, help='Output directory for maps')
    parser.add_argument('--thumb_size', type=int, default=2000, 
                       help='Thumbnail size (default: 2000)')
    parser.add_argument('--features', nargs='+', 
                       help='Specific features to plot (default: auto-select best features)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("FEATURE COLOR MAP GENERATION WITH VALIDATION")
    print("=" * 80)
    print()
    
    # Load data
    print("📊 Loading data...")
    df = pd.read_csv(args.input_csv)
    print(f"  Loaded {len(df):,} nuclei")
    print(f"  Available features: {len(df.columns)} columns")
    print()
    
    # Load slide thumbnail
    print("🖼️  Loading H&E thumbnail...")
    thumb, scale_x, scale_y = load_slide_thumbnail(args.slide, args.thumb_size)
    print(f"  Thumbnail size: {thumb.shape[1]}×{thumb.shape[0]} pixels")
    print(f"  Scale factors: x={scale_x:.6f}, y={scale_y:.6f}")
    print()
    
    # Define feature sets with their colormaps
    if args.features:
        feature_configs = [
            (feat, 'viridis', (2, 98)) for feat in args.features
        ]
    else:
        # Auto-select best features for visualization
        feature_configs = [
            # Nuclear morphology
            ('area_px', 'plasma', (2, 98)),
            ('circularity', 'viridis', (2, 98)),
            ('aspect_ratio', 'coolwarm', (2, 98)),
            ('eccentricity', 'RdYlBu_r', (2, 98)),
            
            # Texture/intensity
            ('gray_mean', 'gray', (2, 98)),
            ('r', 'Reds', (2, 98)),
            ('g', 'Greens', (2, 98)),
            ('b', 'Blues', (2, 98)),
            
            # Density metrics (multiple radii)
            ('density_um2_r50.0', 'YlOrRd', (2, 98)),
            ('density_um2_r100.0', 'YlOrRd', (2, 98)),
            ('density_um2_r150.0', 'YlOrRd', (2, 98)),
            
            # Coherency (alignment)
            ('coherency_50um', 'twilight', (2, 98)),
            ('coherency_100um', 'twilight', (2, 98)),
            ('coherency_150um', 'twilight', (2, 98)),
            
            # Local variance metrics
            ('circularity_local_variance_50um', 'magma', (2, 98)),
            ('circularity_local_variance_100um', 'magma', (2, 98)),
            ('circularity_local_variance_150um', 'magma', (2, 98)),
            
            ('area_px_local_variance_50um', 'inferno', (2, 98)),
            ('area_px_local_variance_100um', 'inferno', (2, 98)),
            ('area_px_local_variance_150um', 'inferno', (2, 98)),
            
            # Local coefficient of variation
            ('circularity_local_cv_50um', 'cividis', (2, 98)),
            ('circularity_local_cv_100um', 'cividis', (2, 98)),
            ('circularity_local_cv_150um', 'cividis', (2, 98)),
            
            # Local mean/median metrics
            ('gray_mean_local_mean_50um', 'Greys', (2, 98)),
            ('gray_mean_local_mean_100um', 'Greys', (2, 98)),
            ('gray_mean_local_mean_150um', 'Greys', (2, 98)),
        ]
    
    # Filter to features that exist in the dataframe
    available_features = []
    for feat, cmap, clip in feature_configs:
        if feat in df.columns:
            available_features.append((feat, cmap, clip))
        else:
            print(f"  ⚠️  Feature '{feat}' not found in CSV, skipping")
    
    print(f"📈 Generating maps for {len(available_features)} features...")
    print()
    
    # Generate maps
    for feature, colormap, percentile_clip in tqdm(available_features, desc="Features"):
        print(f"Processing: {feature}")
        
        # Create feature map
        feature_map, vmin, vmax = create_feature_map(
            df, feature, thumb.shape, scale_x, scale_y,
            colormap=colormap, percentile_clip=percentile_clip
        )
        
        if feature_map is None:
            continue
        
        # Create three-panel visualization
        output_path = output_dir / f"{feature}_validation.png"
        create_three_panel_figure(
            thumb, feature_map, feature, vmin, vmax, colormap, output_path
        )
        print()
    
    print("=" * 80)
    print("✅ COMPLETE!")
    print("=" * 80)
    print(f"Generated {len(available_features)} feature maps")
    print(f"Output directory: {output_dir}")
    print()
    print("Each feature has a 3-panel visualization:")
    print("  • Panel 1: Color map alone (feature values)")
    print("  • Panel 2: Raw H&E image")
    print("  • Panel 3: Overlay (semi-transparent, for validation)")
    print()


if __name__ == '__main__':
    main()
