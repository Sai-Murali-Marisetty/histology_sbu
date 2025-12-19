#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path

def create_demo_panel(slide_name, results_dir):
    base = Path(results_dir) / slide_name
    
    if not base.exists():
        print(f"Error: {base} not found")
        return
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Load images
    thumb = Image.open(base / "preview" / f"{slide_name}_thumb.jpg")
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(thumb)
    ax1.set_title("A. H&E Slide", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Try to load other images
    # Around line 25, change the img_paths list:
    
    img_paths = [
        (gs[0, 1], "cellpose/qc_panel.jpg", "B. Segmentation"),
        (gs[0, 2], "features/density_overlay_r50.0.jpg", "C. Density 50µm"),  # Note: r50.0 not r50
        (gs[0, 3], "features/density_overlay_r150.0.jpg", "D. Density 150µm NEW"),  # r150.0
        (gs[1, 0], "viz/overlay_coherency_150um.jpg", "E. Coherency NEW"),
        (gs[1, 1], "viz/overlay_area_px_local_variance_150um.jpg", "F. Variance NEW"),
    ]
    
    for grid_pos, img_path, title in img_paths:
        full_path = base / img_path
        if full_path.exists():
            ax = fig.add_subplot(grid_pos)
            ax.imshow(Image.open(full_path))
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.axis('off')
    
    # Stats panel
    ax7 = fig.add_subplot(gs[1, 2:])
    ax7.axis('off')
    
    csv_path = base / "features" / f"{slide_name}_nuclei_features_enriched.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        has_coh = 'coherency_150um' in df.columns
        has_var = 'area_px_local_variance_150um' in df.columns
        
        # Simple text without f-string issues
        lines = [
            f"PROGRESS UPDATE - {slide_name}",
            "",
            "COMPLETED:",
            "- Coherency metric",
            "- Variance features", 
            "- 150um radius",
            "",
            "RESULTS:",
            f"- Total nuclei: {len(df):,}",
        ]
        
        if has_coh:
            lines.append(f"- Coherency mean: {df['coherency_150um'].mean():.3f}")
        if has_var:
            lines.append(f"- Variance mean: {df['area_px_local_variance_150um'].mean():.1f}")
        
        lines.extend([
            f"- Total features: {len(df.columns)}",
            "",
            "NEXT WEEK:",
            "- Cellpose vs StarDist",
            "- IHC brown stain",
            "- Enhanced clustering",
        ])
        
        text = '\n'.join(lines)
        ax7.text(0.1, 0.5, text, fontsize=12, family='monospace', va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f"Pipeline Progress - {slide_name}", fontsize=18, fontweight='bold')
    
    output_path = base / f"DEMO_PANEL_{slide_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}\n")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide_name", required=True)
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    create_demo_panel(args.slide_name, args.results_dir)

if __name__ == "__main__":
    main()
