#!/usr/bin/env python3
"""
08_nfb_filament_analysis.py - NFB Filament Segmentation and Analysis

Neurofilament (NFB) staining shows FILAMENTOUS structures (axons), not nuclear staining.
This is a SEPARATE analysis from the nuclear pipeline.

Pipeline:
1. Segment dark brown DAB filaments (not nuclei!)
2. Skeletonize to trace individual filaments
3. Measure filament properties:
   - Number of filaments
   - Total length, mean length, length distribution
   - Connectivity (branching points, endpoints)
   - Alignment/coherency (orientation distribution)
   - Intensity distribution along filaments
   - Spatial density maps

This provides axonal architecture analysis complementary to nucleus-based features.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2
from scipy import ndimage
from skimage import morphology, measure, filters
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

# Increase PIL decompression bomb limit
Image.MAX_IMAGE_PIXELS = None


def detect_dab_filaments(rgb_image, threshold=30, min_size=50):
    """
    Segment brown DAB-stained filaments using color deconvolution.
    
    Args:
        rgb_image: RGB image (H, W, 3) with values 0-255
        threshold: Intensity threshold for DAB detection (0-255)
        min_size: Minimum object size in pixels
    
    Returns:
        binary_mask: Binary mask of filaments (H, W)
        dab_intensity: DAB intensity map (H, W)
    """
    # Convert to float
    rgb = rgb_image.astype(np.float32) / 255.0
    rgb = np.maximum(rgb, 1e-6)  # Avoid log(0)
    
    # Optical density
    od = -np.log(rgb)
    
    # DAB staining vector (brown)
    dab_vector = np.array([0.27, 0.57, 0.78])
    
    # Project OD onto DAB vector
    dab_intensity = np.dot(od, dab_vector)
    
    # Normalize to 0-255
    dab_intensity = np.clip(dab_intensity, 0, None)
    if dab_intensity.max() > 0:
        dab_intensity = (dab_intensity / dab_intensity.max() * 255).astype(np.uint8)
    else:
        dab_intensity = np.zeros_like(dab_intensity, dtype=np.uint8)
    
    # Threshold to get binary mask
    binary_mask = (dab_intensity > threshold).astype(np.uint8)
    
    # Remove small objects (noise)
    binary_mask = morphology.remove_small_objects(
        binary_mask.astype(bool), 
        min_size=min_size
    ).astype(np.uint8)
    
    # Morphological closing to connect nearby filaments
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return binary_mask, dab_intensity


def skeletonize_filaments(binary_mask):
    """
    Skeletonize filament mask to get centerlines.
    """
    distance_map = ndimage.distance_transform_edt(binary_mask)
    skeleton = skeletonize(binary_mask > 0).astype(np.uint8)
    return skeleton, distance_map


def analyze_filament_topology(skeleton):
    """
    Analyze filament network topology: endpoints and branch points.
    """
    kernel = np.array([[1, 1, 1],
                      [1, 10, 1],
                      [1, 1, 1]], dtype=np.uint8)
    
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    
    # Endpoints: 1 neighbor
    endpoints = (neighbor_count == 11) & (skeleton > 0)
    
    # Branch points: 3+ neighbors
    branches = (neighbor_count >= 13) & (skeleton > 0)
    
    endpoint_coords = np.argwhere(endpoints)
    branch_coords = np.argwhere(branches)
    
    return {
        'n_endpoints': len(endpoint_coords),
        'n_branches': len(branch_coords),
        'endpoints': endpoint_coords,
        'branches': branch_coords
    }


def measure_filament_lengths(skeleton, min_length=10):
    """
    Measure individual filament lengths.
    """
    labeled_skeleton = label(skeleton, connectivity=2)
    
    filaments = []
    for region in regionprops(labeled_skeleton):
        length_px = region.area
        
        if length_px < min_length:
            continue
        
        min_row, min_col, max_row, max_col = region.bbox
        width = max_col - min_col
        height = max_row - min_row
        orientation = region.orientation
        cy, cx = region.centroid
        
        filaments.append({
            'filament_id': region.label,
            'length_px': length_px,
            'centroid_x': cx,
            'centroid_y': cy,
            'orientation_rad': orientation,
            'orientation_deg': np.degrees(orientation),
            'bbox_width': width,
            'bbox_height': height,
            'aspect_ratio': max(width, height) / (min(width, height) + 1e-6)
        })
    
    return pd.DataFrame(filaments)


def visualize_filament_segmentation(rgb_image, binary_mask, skeleton, output_path):
    """Create 3-panel visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    axes[0].imshow(rgb_image)
    axes[0].set_title('Original NFB Staining', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(rgb_image)
    mask_overlay = np.ma.masked_where(binary_mask == 0, binary_mask)
    axes[1].imshow(mask_overlay, cmap='Reds', alpha=0.5)
    axes[1].set_title('Filament Segmentation', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(rgb_image)
    skeleton_overlay = np.ma.masked_where(skeleton == 0, skeleton)
    axes[2].imshow(skeleton_overlay, cmap='spring', alpha=0.8)
    axes[2].set_title('Skeletonized Filaments', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def visualize_filament_topology(rgb_image, skeleton, topology, output_path):
    """Visualize endpoints and branch points"""
    fig, ax = plt.subplots(figsize=(12, 12))
    
    ax.imshow(rgb_image)
    skeleton_overlay = np.ma.masked_where(skeleton == 0, skeleton)
    ax.imshow(skeleton_overlay, cmap='gray', alpha=0.3)
    
    if len(topology['endpoints']) > 0:
        ax.scatter(topology['endpoints'][:, 1], topology['endpoints'][:, 0], 
                  c='red', s=20, marker='o', alpha=0.7, 
                  label=f"Endpoints ({topology['n_endpoints']})")
    
    if len(topology['branches']) > 0:
        ax.scatter(topology['branches'][:, 1], topology['branches'][:, 0], 
                  c='blue', s=30, marker='^', alpha=0.7, 
                  label=f"Branches ({topology['n_branches']})")
    
    ax.set_title('Filament Network Topology', fontsize=14)
    ax.legend(loc='upper right')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def visualize_filament_orientations(rgb_image, filaments_df, output_path):
    """Visualize filament orientations"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    ax1.imshow(rgb_image)
    if len(filaments_df) > 0:
        scatter = ax1.scatter(filaments_df['centroid_x'], filaments_df['centroid_y'],
                            c=filaments_df['orientation_deg'], cmap='hsv',
                            s=50, alpha=0.7, vmin=-90, vmax=90)
        plt.colorbar(scatter, ax=ax1, label='Orientation (degrees)')
    ax1.set_title(f'Filament Orientations (n={len(filaments_df)})', fontsize=14)
    ax1.axis('off')
    
    if len(filaments_df) > 0:
        ax2.hist(filaments_df['orientation_deg'], bins=36, range=(-90, 90),
                color='steelblue', edgecolor='black', alpha=0.7)
        ax2.axvline(filaments_df['orientation_deg'].median(), color='red', 
                   linestyle='--', linewidth=2, 
                   label=f"Median: {filaments_df['orientation_deg'].median():.1f}°")
        ax2.set_xlabel('Orientation (degrees)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Orientation Distribution', fontsize=14)
        ax2.legend()
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No filaments detected', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="NFB Filament Analysis")
    parser.add_argument("--slide", required=True, help="Path to WSI")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--level", type=int, default=1, help="Pyramid level (default: 1)")
    parser.add_argument("--threshold", type=int, default=30, help="DAB threshold (default: 30)")
    parser.add_argument("--min_filament_length", type=int, default=20, help="Min length (default: 20)")
    parser.add_argument("--mpp", type=float, default=None, help="Microns per pixel")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("NFB FILAMENT ANALYSIS")
    print("=" * 80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading slide: {args.slide}")
    import openslide
    slide = openslide.OpenSlide(str(args.slide))
    
    if args.mpp is None:
        try:
            mpp = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25))
        except:
            mpp = 0.25
    else:
        mpp = args.mpp
    
    level_downsample = slide.level_downsamples[args.level]
    level_mpp = mpp * level_downsample
    
    print(f"MPP (level 0): {mpp:.4f}")
    print(f"Using level {args.level} (MPP: {level_mpp:.4f})")
    
    w, h = slide.level_dimensions[args.level]
    print(f"Image size: {w} × {h} pixels")
    
    region = slide.read_region((0, 0), args.level, (w, h))
    rgb_image = np.array(region.convert('RGB'))
    
    print(f"\n🔬 Segmenting filaments (threshold={args.threshold})...")
    binary_mask, dab_intensity = detect_dab_filaments(rgb_image, threshold=args.threshold)
    
    filament_area_px = binary_mask.sum()
    filament_fraction = filament_area_px / (w * h)
    print(f"✓ Filament area: {filament_area_px:,} px ({filament_fraction*100:.2f}%)")
    
    print(f"\n🦴 Skeletonizing...")
    skeleton, distance_map = skeletonize_filaments(binary_mask)
    
    skeleton_length_px = skeleton.sum()
    skeleton_length_um = skeleton_length_px * level_mpp
    print(f"✓ Skeleton length: {skeleton_length_px:,} px ({skeleton_length_um/1000:.2f} mm)")
    
    print(f"\n🕸️  Analyzing topology...")
    topology = analyze_filament_topology(skeleton)
    print(f"✓ Endpoints: {topology['n_endpoints']:,}")
    print(f"✓ Branches: {topology['n_branches']:,}")
    
    print(f"\n📏 Measuring filaments...")
    filaments_df = measure_filament_lengths(skeleton, min_length=args.min_filament_length)
    print(f"✓ Detected {len(filaments_df):,} filaments")
    
    if len(filaments_df) > 0:
        filaments_df['length_um'] = filaments_df['length_px'] * level_mpp
        print(f"  Mean: {filaments_df['length_um'].mean():.1f} µm")
        print(f"  Median: {filaments_df['length_um'].median():.1f} µm")
    
    filaments_csv = output_dir / 'filaments.csv'
    filaments_df.to_csv(filaments_csv, index=False)
    print(f"\n💾 Saved: {filaments_csv}")
    
    summary = {
        'slide': str(args.slide),
        'level': args.level,
        'level_mpp': level_mpp,
        'filament_area_px': int(filament_area_px),
        'filament_fraction': float(filament_fraction),
        'total_skeleton_length_um': float(skeleton_length_um),
        'n_endpoints': int(topology['n_endpoints']),
        'n_branches': int(topology['n_branches']),
        'n_filaments': len(filaments_df),
        'mean_filament_length_um': float(filaments_df['length_um'].mean()) if len(filaments_df) > 0 else 0,
    }
    
    summary_json = output_dir / 'filament_summary.json'
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Saved: {summary_json}")
    
    print(f"\n🎨 Creating visualizations...")
    
    visualize_filament_segmentation(rgb_image, binary_mask, skeleton,
                                   output_dir / 'filament_segmentation.png')
    
    visualize_filament_topology(rgb_image, skeleton, topology,
                               output_dir / 'filament_topology.png')
    
    visualize_filament_orientations(rgb_image, filaments_df,
                                   output_dir / 'filament_orientations.png')
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Filament area: {filament_fraction*100:.2f}%")
    print(f"Total length: {skeleton_length_um/1000:.2f} mm")
    print(f"Filaments: {len(filaments_df):,}")
    print(f"Endpoints: {topology['n_endpoints']:,}")
    print(f"Branches: {topology['n_branches']:,}")
    print(f"\n📂 Results: {output_dir}")
    print("=" * 80)
    print("\n✅ Complete!")
    
    slide.close()


if __name__ == "__main__":
    main()
