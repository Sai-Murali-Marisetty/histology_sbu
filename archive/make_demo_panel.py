#!/usr/bin/env python3
"""
Generate demonstration panel for meeting showing new features.
Usage: python make_demo_panel.py --slide_name CD3-S25 --results_dir results
"""

import argparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path


def create_demo_panel(slide_name, results_dir):
    """Create comprehensive demo panel showing old + new features"""
    
    base = Path(results_dir) / slide_name
    
    # Check if results exist
    if not base.exists():
        print(f"Error: Results directory not found: {base}")
        print(f"Make sure you've run the pipeline on {slide_name}")
        return
    
    # Load images
    try:
        thumb = Image.open(base / "preview" / f"{slide_name}_thumb.jpg")
        seg_qc = Image.open(base / "cellpose" / "qc_panel.jpg")
        
        # Try to load density maps
        density_50 = base / "features" / "density_overlay_r50.jpg"
        density_150 = base / "features" / "density_overlay_r150.jpg"
        
        # New features
        coherency = base / "viz" / "overlay_coherency_150um.jpg"
        variance = base / "viz" / "overlay_area_px_local_variance_150um.jpg"
        
        # Load CSV for statistics
        csv_path = base / "features" / f"{slide_name}_nuclei_features_enriched.csv"
        
    except FileNotFoundError as e:
        print(f"Error: Missing file - {e}")
        print("Run the complete pipeline first:")
        print(f"  ./run_one_slide.sh /path/to/{slide_name}.svs results")
        return
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Row 1: Original results
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(thumb)
    ax1.set_title("A. H&E Slide", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(seg_qc)
    ax2.set_title("B. Segmentation QC", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    if density_50.exists():
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(Image.open(density_50))
        ax3.set_title("C. Density (50µm)", fontsize=14, fontweight='bold')
        ax3.axis('off')
    
    if density_150.exists():
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(Image.open(density_150))
        ax4.set_title("D. Density (150µm) - NEW", fontsize=14, fontweight='bold', color='red')
        ax4.axis('off')
    
    # Row 2: New features
    if coherency.exists():
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.imshow(Image.open(coherency))
        ax5.set_title("E. Coherency - NEW ✓", fontsize=14, fontweight='bold', color='red')
        ax5.axis('off')
    
    if variance.exists():
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.imshow(Image.open(variance))
        ax6.set_title("F. Area Variance - NEW ✓", fontsize=14, fontweight='bold', color='red')
        ax6.axis('off')
    
    # Statistics panel
    ax7 = fig.add_subplot(gs[1, 2:])
    ax7.axis('off')
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        
        # Check which new columns exist
        has_coherency = any('coherency' in col for col in df.columns)
        has_variance = any('variance' in col for col in df.columns)
        
        stats_text = f"""PROGRESS UPDATE - {slide_name}
        
        COMPLETED:
        - Coherency metric 
        - Variance features
        - 150um radius added
        
        RESULTS:
        - Nuclei: {len(df):,}
        - Coherency: {df['coherency_150um'].mean():.3f if has_coherency else 'N/A'}
        - Variance: {df['area_px_local_variance_150um'].mean():.1f if has_variance else 'N/A'}
        - Features: {len(df.columns)} columns
        
        NEXT WEEK:
        - Cellpose vs StarDist
        - IHC brown stain
        - Enhanced clustering
        """
        
        ax7.text(0.05, 0.5, stats_text, 
                fontsize=11, 
                family='monospace', 
                va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f"HistoVision Pipeline Progress - {slide_name}", 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Save
    output_path = base / f"DEMO_PANEL_for_meeting_{slide_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n{'='*60}")
    print(f"✓ Saved demo panel: {output_path}")
    print(f"{'='*60}\n")
    
    # Also save as PDF for higher quality
    pdf_path = base / f"DEMO_PANEL_for_meeting_{slide_name}.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Also saved PDF: {pdf_path}\n")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate demo panel for meeting")
    parser.add_argument("--slide_name", required=True, 
                       help="Slide name (e.g., CD3-S25)")
    parser.add_argument("--results_dir", default="results", 
                       help="Results directory (default: results)")
    args = parser.parse_args()
    
    create_demo_panel(args.slide_name, args.results_dir)


if __name__ == "__main__":
    main()
