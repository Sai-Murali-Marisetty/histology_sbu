#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from pathlib import Path

def create_multi_zoom_panel(csv_path, thumb_path, out_path):
    """Create panel with 4 zoomed regions showing different areas"""
    df = pd.read_csv(csv_path)
    img = np.array(Image.open(thumb_path).convert('RGB'))
    img_h, img_w = img.shape[:2]
    
    print(f"Image size: {img_w} x {img_h}")
    print(f"Nuclei range: x=[{df['x'].min():.0f}, {df['x'].max():.0f}], y=[{df['y'].min():.0f}, {df['y'].max():.0f}]")
    
    # Find 4 interesting regions - ensure they're within image bounds
    zoom_size = 300
    margin = zoom_size + 50
    
    # Filter to nuclei well within image
    df_safe = df[
        (df['x'] > margin) & (df['x'] < img_w - margin) &
        (df['y'] > margin) & (df['y'] < img_h - margin)
    ].copy()
    
    if len(df_safe) < 100:
        print("Not enough nuclei in safe region, using all nuclei")
        df_safe = df.copy()
    
    # 1. Highest coherency
    high_coh = df_safe.nlargest(500, 'coherency_150um')
    region1 = (high_coh['x'].median(), high_coh['y'].median())
    
    # 2. Lowest coherency  
    low_coh = df_safe.nsmallest(500, 'coherency_150um')
    region2 = (low_coh['x'].median(), low_coh['y'].median())
    
    # 3. Highest density
    high_dens = df_safe.nlargest(500, 'corrected_density_um2_r50.0')
    region3 = (high_dens['x'].median(), high_dens['y'].median())
    
    # 4. Center region
    region4 = (df['x'].median(), df['y'].median())
    
    regions = [
        (region1, "Highest Coherency"),
        (region2, "Lowest Coherency"),
        (region3, "High Density"),
        (region4, "Center Region")
    ]
    
    # Create 2x2 panel
    fig = plt.figure(figsize=(16, 16))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    subsample = 3
    arrow_scale = 15
    
    for plot_idx, ((cx, cy), title) in enumerate(regions):
        print(f"\nProcessing region {plot_idx+1}: {title} at ({cx:.0f}, {cy:.0f})")
        
        row = plot_idx // 2
        col = plot_idx % 2
        
        ax = fig.add_subplot(gs[row, col])
        
        # Define region with bounds checking
        x_min = int(max(0, cx - zoom_size))
        x_max = int(min(img_w, cx + zoom_size))
        y_min = int(max(0, cy - zoom_size))
        y_max = int(min(img_h, cy + zoom_size))
        
        print(f"  Bounds: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
        
        # Skip if region is too small
        if (x_max - x_min) < 100 or (y_max - y_min) < 100:
            print(f"  Skipping - region too small")
            continue
        
        # Get nuclei in region
        df_region = df[
            (df['x'] >= x_min) & (df['x'] <= x_max) &
            (df['y'] >= y_min) & (df['y'] <= y_max)
        ].copy()
        
        print(f"  Nuclei in region: {len(df_region)}")
        
        if len(df_region) < 10:
            print(f"  Skipping - too few nuclei")
            continue
            
        # Subsample for visualization
        df_sub = df_region.iloc[::subsample]
        angles = np.arctan2(df_sub['minor_axis_length'], 
                           df_sub['major_axis_length'])
        
        # Crop image
        img_crop = img[y_min:y_max, x_min:x_max]
        
        if img_crop.size == 0:
            print(f"  Skipping - empty crop")
            continue
        
        ax.imshow(img_crop, extent=[x_min, x_max, y_max, y_min])
        
        # Draw arrows
        for idx_row, nucleus in df_sub.iterrows():
            angle = angles[idx_row]
            length = nucleus['major_axis_length'] / 2 * arrow_scale
            
            dx = length * np.cos(angle)
            dy = length * np.sin(angle)
            
            ax.arrow(nucleus['x'], nucleus['y'], dx, dy,
                    head_width=4, head_length=6,
                    fc='yellow', ec='red', alpha=0.9, linewidth=2)
        
        # Stats
        coh_mean = df_region['coherency_150um'].mean()
        coh_std = df_region['coherency_150um'].std()
        n_cells = len(df_region)
        
        ax.set_title(f'{title}\nCells: {n_cells}, Coh: {coh_mean:.3f}±{coh_std:.3f}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)
        ax.axis('off')
    
    plt.suptitle('Coherency Validation - Zoomed Regions', 
                 fontsize=16, fontweight='bold')
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--thumb", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    
    create_multi_zoom_panel(args.csv, args.thumb, args.out)
