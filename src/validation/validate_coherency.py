#!/usr/bin/env python3
"""
Validate coherency by visualizing orientation vectors
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Increase PIL decompression bomb limit for large whole slide images
Image.MAX_IMAGE_PIXELS = None

def plot_orientation_field(df, thumb_path, out_path, 
                           region=None, subsample=20, 
                           arrow_scale=30):
    """
    Draw orientation vectors on tissue thumbnail
    
    region: (x_min, x_max, y_min, y_max) to zoom
    subsample: show every Nth cell
    arrow_scale: arrow length multiplier
    """
    img = np.array(Image.open(thumb_path).convert('RGB'))
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(img)
    
    # Subsample to avoid clutter
    df_sub = df.iloc[::subsample].copy()
    
    # Crop to region if specified
    if region:
        x_min, x_max, y_min, y_max = region
        df_sub = df_sub[
            (df_sub['x'] >= x_min) & (df_sub['x'] <= x_max) &
            (df_sub['y'] >= y_min) & (df_sub['y'] <= y_max)
        ]
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)  # Inverted y
    
    # Calculate angles from major/minor axis
    angles = np.arctan2(df_sub['minor_axis_length'], 
                       df_sub['major_axis_length'])
    
    # Draw arrows
    for idx, row in df_sub.iterrows():
        angle = angles[idx]
        length = row['major_axis_length'] / 2 * arrow_scale
        
        dx = length * np.cos(angle)
        dy = length * np.sin(angle)
        
        ax.arrow(row['x'], row['y'], dx, dy,
                head_width=5, head_length=8,
                fc='red', ec='red', alpha=0.7, linewidth=1)
    
    ax.set_title(f'Orientation Vectors (every {subsample}th nucleus)', 
                fontsize=16)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_local_average_angle(df, thumb_path, out_path, radius_um=150):
    """
    Show local average orientation as heatmap
    """
    from scipy.spatial import cKDTree
    
    coords = np.vstack([df['x_um'], df['y_um']]).T
    tree = cKDTree(coords)
    angles = np.arctan2(df['minor_axis_length'], df['major_axis_length'])
    
    local_avg_angles = []
    for i, p in enumerate(coords):
        idx = tree.query_ball_point(p, radius_um)
        if len(idx) > 3:
            # Circular mean for angles
            local_angles = angles.iloc[idx] if hasattr(angles, 'iloc') else angles[idx]
            avg_angle = np.arctan2(
                np.mean(np.sin(local_angles)),
                np.mean(np.cos(local_angles))
            )
        else:
            avg_angle = angles.iloc[i] if hasattr(angles, 'iloc') else angles[i]
        local_avg_angles.append(avg_angle)
    
    df['local_avg_angle'] = local_avg_angles
    
    # Plot
    img = np.array(Image.open(thumb_path).convert('RGB'))
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img)
    
    sc = ax.scatter(df['x'], df['y'], 
                   c=df['local_avg_angle'], 
                   cmap='hsv', s=4, alpha=0.7,
                   vmin=-np.pi, vmax=np.pi)
    ax.set_title(f'Local Average Angle ({radius_um}µm)', fontsize=16)
    ax.axis('off')
    plt.colorbar(sc, label='Angle (radians)', shrink=0.5)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--thumb", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--slide_name", required=True)
    args = ap.parse_args()
    
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    
    # Full slide orientation field
    plot_orientation_field(
        df, args.thumb,
        Path(args.out_dir) / f"{args.slide_name}_orientation_vectors_full.png",
        subsample=50
    )
    
    # Zoomed regions (pick 3 interesting areas)
    # You'll need to identify good regions - tumor boundary, uniform tissue, etc.
    
    # Local average angle heatmap
    df = plot_local_average_angle(
        df, args.thumb,
        Path(args.out_dir) / f"{args.slide_name}_local_avg_angle.png"
    )
    
    # Save updated CSV with local avg angle
    df.to_csv(args.csv.replace('.csv', '_with_angles.csv'), index=False)
