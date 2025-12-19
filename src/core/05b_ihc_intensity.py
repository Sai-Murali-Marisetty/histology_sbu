#!/usr/bin/env python3
"""
IHC Intensity Measurement Module

Measures marker intensity in perinuclear regions for IHC slides.
For each segmented nucleus:
1. Expand the region by 10-20% to capture cytoplasmic/membrane staining
2. Measure intensity of the marker (brown DAB staining)
3. Calculate perinuclear intensity (expanded region - nucleus)
4. Add measurements to nucleus features for downstream analysis

This is CRITICAL for IHC markers like NFB and PGP9.5 which localize
to cytoplasm/axons, not nucleus.
"""

import os
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import openslide
from skimage import morphology, measure
from scipy import ndimage
from tqdm import tqdm

# Increase PIL decompression bomb limit for whole slide images
Image.MAX_IMAGE_PIXELS = None


def detect_brown_dab(rgb_image):
    """
    Detect brown DAB staining in IHC slides using color deconvolution approach.
    
    Args:
        rgb_image: RGB image as numpy array (H, W, 3) with values 0-255
    
    Returns:
        intensity_map: Grayscale intensity map where higher values = stronger brown staining
    """
    # Convert to float
    rgb = rgb_image.astype(np.float32) / 255.0
    
    # Avoid log(0) by adding small epsilon
    rgb = np.maximum(rgb, 1e-6)
    
    # Optical density
    od = -np.log(rgb)
    
    # DAB staining vector (brown) - approximation
    # Typical DAB has higher absorption in blue, some in green, less in red
    dab_vector = np.array([0.27, 0.57, 0.78])  # R, G, B weights
    
    # Project OD onto DAB vector
    dab_intensity = np.dot(od, dab_vector)
    
    # Normalize to 0-255 range
    dab_intensity = np.clip(dab_intensity, 0, None)
    if dab_intensity.max() > 0:
        dab_intensity = (dab_intensity / dab_intensity.max() * 255).astype(np.uint8)
    else:
        dab_intensity = np.zeros_like(dab_intensity, dtype=np.uint8)
    
    return dab_intensity


def measure_nucleus_ihc_intensity(
    slide_path,
    nuclei_csv,
    output_csv,
    expansion_factor=0.15,
    level=0,
    mpp=None,
    min_positive_threshold=30
):
    """
    Measure IHC marker intensity in perinuclear regions.
    
    Args:
        slide_path: Path to whole slide image
        nuclei_csv: Path to CSV with nucleus coordinates and properties
        output_csv: Path to save enriched CSV with intensity measurements
        expansion_factor: Fraction to expand nucleus region (0.15 = 15%)
        level: Pyramid level to use for measurement (0 = highest resolution)
        mpp: Microns per pixel at level 0 (auto-detected if None)
        min_positive_threshold: Intensity threshold for positive staining (0-255)
    """
    print("=" * 60)
    print("IHC Intensity Measurement")
    print("=" * 60)
    
    # Load slide
    print(f"\nLoading slide: {slide_path}")
    slide = openslide.OpenSlide(str(slide_path))
    
    # Get microns per pixel
    if mpp is None:
        try:
            mpp = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25))
        except:
            mpp = 0.25
            print(f"Warning: Could not auto-detect MPP, using default: {mpp}")
    
    print(f"Microns per pixel (level 0): {mpp:.4f}")
    
    # Get level downsampling
    level_downsample = slide.level_downsamples[level]
    level_mpp = mpp * level_downsample
    print(f"Using level {level} (downsample: {level_downsample:.2f}x, MPP: {level_mpp:.4f})")
    
    # Load nuclei data
    print(f"\nLoading nuclei from: {nuclei_csv}")
    df = pd.read_csv(nuclei_csv)
    n_nuclei = len(df)
    print(f"Found {n_nuclei:,} nuclei")
    
    if n_nuclei == 0:
        print("No nuclei to process!")
        df.to_csv(output_csv, index=False)
        return
    
    # Initialize intensity columns
    df['marker_intensity_mean'] = 0.0
    df['marker_intensity_max'] = 0.0
    df['marker_intensity_std'] = 0.0
    df['marker_positive_area_fraction'] = 0.0
    df['perinuclear_intensity_mean'] = 0.0
    df['perinuclear_positive_fraction'] = 0.0
    
    # Process nuclei in batches to manage memory
    batch_size = 100
    print(f"\nProcessing nuclei in batches of {batch_size}...")
    
    for batch_start in tqdm(range(0, n_nuclei, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, n_nuclei)
        batch_df = df.iloc[batch_start:batch_end]
        
        for idx, row in batch_df.iterrows():
            try:
                # Get nucleus properties
                x_um = row['x_um']
                y_um = row['y_um']
                area_px = row['area_px']
                
                # Estimate nucleus radius in microns
                radius_um = np.sqrt(area_px * mpp * mpp / np.pi)
                
                # Calculate expanded radius
                expanded_radius_um = radius_um * (1 + expansion_factor)
                
                # Convert to pixel coordinates at the working level
                x_px_level0 = int(x_um / mpp)
                y_px_level0 = int(y_um / mpp)
                
                # Calculate region size at working level
                region_radius_px = int(expanded_radius_um / level_mpp)
                region_size = region_radius_px * 2
                
                # Ensure minimum region size
                if region_size < 20:
                    region_size = 20
                    region_radius_px = 10
                
                # Calculate top-left corner for region extraction
                x0 = max(0, int(x_px_level0 / level_downsample - region_radius_px))
                y0 = max(0, int(y_px_level0 / level_downsample - region_radius_px))
                
                # Extract region from slide
                location = (int(x0 * level_downsample), int(y0 * level_downsample))
                size = (region_size, region_size)
                
                try:
                    region = slide.read_region(location, level, size)
                    region_rgb = np.array(region.convert('RGB'))
                except:
                    # Skip if region extraction fails (edge of slide, etc.)
                    continue
                
                # Detect brown DAB staining
                dab_intensity = detect_brown_dab(region_rgb)
                
                # Create circular masks
                center = (region_size // 2, region_size // 2)
                y_grid, x_grid = np.ogrid[:region_size, :region_size]
                
                # Nucleus mask (original radius)
                nucleus_radius_px = int(radius_um / level_mpp)
                nucleus_mask = ((x_grid - center[1])**2 + (y_grid - center[0])**2) <= nucleus_radius_px**2
                
                # Expanded mask (expanded radius)
                expanded_mask = ((x_grid - center[1])**2 + (y_grid - center[0])**2) <= region_radius_px**2
                
                # Perinuclear mask (expanded - nucleus)
                perinuclear_mask = expanded_mask & ~nucleus_mask
                
                # Measure intensity in expanded region
                if expanded_mask.sum() > 0:
                    expanded_intensities = dab_intensity[expanded_mask]
                    df.at[idx, 'marker_intensity_mean'] = float(expanded_intensities.mean())
                    df.at[idx, 'marker_intensity_max'] = float(expanded_intensities.max())
                    df.at[idx, 'marker_intensity_std'] = float(expanded_intensities.std())
                    
                    # Calculate positive area fraction
                    positive_pixels = (expanded_intensities > min_positive_threshold).sum()
                    df.at[idx, 'marker_positive_area_fraction'] = float(positive_pixels / len(expanded_intensities))
                
                # Measure intensity in perinuclear region only
                if perinuclear_mask.sum() > 0:
                    perinuclear_intensities = dab_intensity[perinuclear_mask]
                    df.at[idx, 'perinuclear_intensity_mean'] = float(perinuclear_intensities.mean())
                    
                    # Calculate positive fraction in perinuclear region
                    positive_peri = (perinuclear_intensities > min_positive_threshold).sum()
                    df.at[idx, 'perinuclear_positive_fraction'] = float(positive_peri / len(perinuclear_intensities))
            
            except Exception as e:
                # If any error occurs, skip this nucleus and continue
                continue
    
    # Save enriched data
    print(f"\nSaving enriched data to: {output_csv}")
    df.to_csv(output_csv, index=False)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Intensity Measurement Summary")
    print("=" * 60)
    print(f"Nuclei processed: {n_nuclei:,}")
    print(f"\nMarker intensity (expanded region):")
    print(f"  Mean: {df['marker_intensity_mean'].mean():.2f} ± {df['marker_intensity_mean'].std():.2f}")
    print(f"  Range: {df['marker_intensity_mean'].min():.2f} - {df['marker_intensity_mean'].max():.2f}")
    print(f"\nPerinuclear intensity:")
    print(f"  Mean: {df['perinuclear_intensity_mean'].mean():.2f} ± {df['perinuclear_intensity_mean'].std():.2f}")
    print(f"  Range: {df['perinuclear_intensity_mean'].min():.2f} - {df['perinuclear_intensity_mean'].max():.2f}")
    print(f"\nPositive staining:")
    print(f"  Marker positive fraction: {df['marker_positive_area_fraction'].mean():.3f} ± {df['marker_positive_area_fraction'].std():.3f}")
    print(f"  Perinuclear positive fraction: {df['perinuclear_positive_fraction'].mean():.3f} ± {df['perinuclear_positive_fraction'].std():.3f}")
    
    # Count highly positive nuclei (mean intensity > threshold)
    highly_positive = (df['marker_intensity_mean'] > min_positive_threshold).sum()
    print(f"\nHighly positive nuclei (>{min_positive_threshold}): {highly_positive:,} ({highly_positive/n_nuclei*100:.1f}%)")
    
    slide.close()
    print("\n" + "=" * 60)
    print("✅ IHC intensity measurement complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Measure IHC marker intensity in perinuclear regions"
    )
    parser.add_argument("--slide", required=True, help="Path to whole slide image")
    parser.add_argument("--nuclei_csv", required=True, help="CSV with nucleus coordinates")
    parser.add_argument("--output_csv", required=True, help="Output CSV with intensity measurements")
    parser.add_argument("--expansion", type=float, default=0.15, 
                       help="Fraction to expand nucleus region (default: 0.15 = 15%%)")
    parser.add_argument("--level", type=int, default=0,
                       help="Pyramid level for measurement (default: 0 = highest resolution)")
    parser.add_argument("--mpp", type=float, default=None,
                       help="Microns per pixel at level 0 (auto-detected if not provided)")
    parser.add_argument("--threshold", type=int, default=30,
                       help="Minimum intensity for positive staining (0-255, default: 30)")
    
    args = parser.parse_args()
    
    measure_nucleus_ihc_intensity(
        slide_path=args.slide,
        nuclei_csv=args.nuclei_csv,
        output_csv=args.output_csv,
        expansion_factor=args.expansion,
        level=args.level,
        mpp=args.mpp,
        min_positive_threshold=args.threshold
    )


if __name__ == "__main__":
    main()
