#!/usr/bin/env python3
"""
Generate combined multi-modal UMAP analysis.
Registers all stain types from the same tissue sample and extracts
multi-modal features for integrated immunophenotyping.

Strategy:
1. Group slides by sample ID (B17, B27, S25, etc.)
2. Register all IHC slides to H&E reference (affine + B-spline)
3. For each nucleus in H&E, extract features from all registered images
4. Generate combined UMAP with full multi-modal feature set
"""

import pandas as pd
import numpy as np
import SimpleITK as sitk
import umap
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
import re
from collections import defaultdict
import cv2
from tqdm import tqdm
from scipy.spatial import cKDTree

import matplotlib as mpl
from matplotlib.lines import Line2D

# Global style
MPL_RC = {
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10
}
mpl.rcParams.update(MPL_RC)

# Stain type definitions
STAIN_MARKERS = ['CD3', 'GFAP', 'Iba1', 'NF', 'PGP9.5', 'PID']

def parse_slide_name(slide_name: str) -> Tuple[str, str]:
    """
    Extract stain type and sample ID from slide name.
    
    Examples:
        HE-B17 → (H&E, B17)
        CD3-S25 → (CD3, S25)
        PGP9-5-B27 → (PGP9.5, B27)
        BIDS19 → (PID, S19)  # Handle typo
    """
    # Normalize BIDS → PID-S
    if slide_name.startswith('BIDS'):
        slide_name = re.sub(r'^BIDS', 'PID-S', slide_name, flags=re.IGNORECASE)
    
    # PID special cases
    if re.match(r'^PID\d+', slide_name, flags=re.IGNORECASE):
        # PID30B17 → PID, B17
        match = re.match(r'^PID\d+([BS]\d+)', slide_name, flags=re.IGNORECASE)
        if match:
            return 'PID', match.group(1).upper()
    
    # Standard format: STAIN-SAMPLE
    parts = slide_name.split('-')
    
    # Handle PGP9-5-SAMPLE (3 parts)
    if len(parts) == 3 and parts[0] == 'PGP9' and parts[1] == '5':
        stain = 'PGP9.5'
        sample = parts[2].upper()
        return stain, sample
    
    if len(parts) >= 2:
        stain = parts[0]
        sample = parts[1].upper()
        
        # Normalize stain names to match STAIN_MARKERS
        stain_upper = stain.upper()
        
        # Handle HE variants
        if stain_upper in ['HE', 'H&E', 'HE=']:
            stain = 'H&E'
        # Handle PGP9.5 variants (PGP9-5 → PGP9.5)
        elif 'PGP' in stain_upper:
            stain = 'PGP9.5'
        # Handle Iba1 (keep mixed case)
        elif stain_upper == 'IBA1':
            stain = 'Iba1'
        # Handle other markers (preserve case from STAIN_MARKERS)
        elif stain_upper in ['CD3', 'GFAP', 'NF', 'PID']:
            stain = stain_upper
        else:
            stain = stain  # Keep as-is
            
        return stain, sample
    
    return None, None


def group_slides_by_sample(results_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    Group slide directories by sample ID.
    
    Returns:
        {
            'B17': {
                'H&E': Path('HE-B17'),
                'CD3': Path('CD3-B17'),
                ...
            },
            'B27': {...},
            ...
        }
    """
    samples = defaultdict(dict)
    
    for slide_dir in results_dir.iterdir():
        if not slide_dir.is_dir():
            continue
        
        stain, sample = parse_slide_name(slide_dir.name)
        if stain and sample:
            samples[sample][stain] = slide_dir
    
    # Filter to complete samples (must have H&E + at least 4 markers)
    complete_samples = {}
    for sample, stains in samples.items():
        if 'H&E' in stains and len(stains) >= 5:  # H&E + 4+ markers
            complete_samples[sample] = stains
            print(f"✅ Sample {sample}: {len(stains)} stains - {list(stains.keys())}")
        else:
            print(f"⚠️  Sample {sample}: Incomplete ({len(stains)} stains) - skipping")
    
    return complete_samples


def load_tissue_mask(slide_dir: Path) -> np.ndarray:
    """Load tissue mask for registration."""
    # Try new location first (preview/tissue_mask.png)
    mask_path = slide_dir / 'preview' / 'tissue_mask.png'
    
    # Try old location (masks/SLIDENAME_tissue_mask.png)
    if not mask_path.exists():
        slide_name = slide_dir.name
        mask_path = slide_dir / 'masks' / f'{slide_name}_tissue_mask.png'
    
    if not mask_path.exists():
        raise FileNotFoundError(f"Tissue mask not found in preview/ or masks/: {slide_dir}")
    
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    return mask


def load_thumbnail(slide_dir: Path) -> np.ndarray:
    """Load thumbnail image for registration."""
    # Try new naming convention first (thumbnail.jpg)
    thumb_path = slide_dir / 'preview' / 'thumbnail.jpg'
    
    # Try old naming convention (SLIDENAME_thumb.jpg)
    if not thumb_path.exists():
        slide_name = slide_dir.name
        thumb_path = slide_dir / 'preview' / f'{slide_name}_thumb.jpg'
    
    if not thumb_path.exists():
        raise FileNotFoundError(f"Thumbnail not found in preview/: {slide_dir}")
    
    img = cv2.imread(str(thumb_path))
    # Convert to grayscale for registration
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def find_best_orientation(fixed_img: np.ndarray, moving_img: np.ndarray) -> np.ndarray:
    """
    Try different orientations of moving image and return the one with best overlap to fixed.
    Tests: original, flip vertical, flip horizontal, rotate 180°
    Uses normalized cross-correlation as similarity metric.
    
    NOTE: Resizes moving to match fixed for comparison, then returns oriented version in original size.
    """
    from skimage.transform import resize
    
    # Get target size from fixed
    target_shape = fixed_img.shape
    
    # Resize moving to match fixed for comparison
    moving_resized = resize(moving_img, target_shape, preserve_range=True, anti_aliasing=True)
    
    # Normalize fixed for comparison
    fixed_norm = (fixed_img - fixed_img.mean()) / (fixed_img.std() + 1e-8)
    
    orientations = {
        'original': moving_img,
        'flip_vertical': np.flipud(moving_img),
        'flip_horizontal': np.fliplr(moving_img),
        'rotate_180': np.rot90(moving_img, 2),
    }
    
    orientations_resized = {
        'original': moving_resized,
        'flip_vertical': np.flipud(moving_resized),
        'flip_horizontal': np.fliplr(moving_resized),
        'rotate_180': np.rot90(moving_resized, 2),
    }
    
    best_score = -np.inf
    best_orientation = moving_img
    best_name = 'original'
    
    for name in orientations.keys():
        # Compare resized versions
        oriented_norm = (orientations_resized[name] - orientations_resized[name].mean()) / \
                       (orientations_resized[name].std() + 1e-8)
        
        # Compute normalized cross-correlation
        correlation = np.corrcoef(fixed_norm.flatten(), oriented_norm.flatten())[0, 1]
        
        if correlation > best_score:
            best_score = correlation
            best_orientation = orientations[name]  # Return original size
            best_name = name
    
    if best_name != 'original':
        print(f"    → Adjusted orientation: {best_name} (correlation: {best_score:.3f})")
    
    return best_orientation


def initialize_with_center_of_mass(fixed: sitk.Image, moving: sitk.Image) -> sitk.TranslationTransform:
    """
    Initialize transform by aligning centers of mass using NumPy.
    Works better than geometric centers for irregularly shaped/sparse tissues.
    Critical for CD3 (sparse immune cells) and GFAP (glial cells only).
    
    IMPORTANT: Works in PIXEL space, not physical space, since thumbnails don't have
    proper spacing metadata.
    """
    # Convert to numpy arrays
    fixed_array = sitk.GetArrayFromImage(fixed)
    moving_array = sitk.GetArrayFromImage(moving)
    
    # Create binary masks (non-zero pixels)
    fixed_mask = fixed_array > 0.01
    moving_mask = moving_array > 0.01
    
    # Compute center of mass manually using NumPy
    # Center of mass = average position of all tissue pixels
    fixed_coords = np.array(np.where(fixed_mask))  # Shape: (2, N_pixels) - (row, col)
    moving_coords = np.array(np.where(moving_mask))
    
    if fixed_coords.shape[1] == 0 or moving_coords.shape[1] == 0:
        print(f"    ⚠️  Empty image detected, skipping center alignment")
        return sitk.TranslationTransform(fixed.GetDimension())
    
    # Compute center of mass in pixel coordinates (row, col)
    fixed_com_pixels = fixed_coords.mean(axis=1)  
    moving_com_pixels = moving_coords.mean(axis=1)
    
    # Translation in pixel coordinates (row, col)
    # SimpleITK expects offset in (x, y) = (col, row) order
    translation_pixels = fixed_com_pixels - moving_com_pixels
    translation_xy = np.array([translation_pixels[1], translation_pixels[0]])  # Swap to (x, y)
    
    # Create translation transform
    # Since images don't have spacing metadata, SimpleITK treats pixels as mm
    # So we can use pixel offsets directly
    initial_transform = sitk.TranslationTransform(fixed.GetDimension())
    initial_transform.SetOffset(translation_xy.tolist())
    
    print(f"    → Center-of-mass alignment: translation={translation_xy} pixels (row={translation_pixels[0]:.1f}, col={translation_pixels[1]:.1f})")
    return initial_transform


def register_affine_bspline(fixed_img: np.ndarray,
                            moving_img: np.ndarray,
                            output_dir: Optional[Path] = None,
                            fixed_name: str = 'fixed',
                            moving_name: str = 'moving') -> Tuple[np.ndarray, sitk.CompositeTransform]:
    """
    Hierarchical registration: Affine followed by B-spline deformable.
    
    This is the GOLD STANDARD approach for histology registration:
    - Center-of-mass initialization handles sparse/offset tissues
    - Affine handles global rotation, translation, scaling, shearing
    - B-spline handles local tissue deformation (wrinkles, folds)
    - Mutual Information metric works across different stains
    
    Args:
        fixed_img: Reference image (H&E)
        moving_img: Image to align (IHC marker)
        output_dir: Optional directory to save registration QC images
        
    Returns:
        registered_img: Aligned moving image
        composite_transform: Combined translation + affine + B-spline transform
    """
    
    print(f"  Registering {moving_name} → {fixed_name}...")
    
    # CRITICAL: Find best initial orientation
    moving_img = find_best_orientation(fixed_img, moving_img)
    
    # Convert to SimpleITK
    fixed = sitk.GetImageFromArray(fixed_img.astype(np.float32))
    moving = sitk.GetImageFromArray(moving_img.astype(np.float32))
    
    # ==========================================
    # PHASE 0: Center-of-Mass Initialization
    # ==========================================
    # CRITICAL FIX: Align centers before affine registration
    # This solves: "All samples map outside moving image buffer"
    center_transform = initialize_with_center_of_mass(fixed, moving)
    
    # Apply center alignment to moving image
    moving_centered = sitk.Resample(moving, fixed, center_transform, sitk.sitkLinear, 0.0)
    
    # ==========================================
    # PHASE 1: Affine Registration (Global)
    # ==========================================
    print(f"    Phase 1: Affine (global alignment)...")
    
    affine_reg = sitk.ImageRegistrationMethod()
    
    # Metric: Mutual Information (best for multi-modal)
    # Increased sampling for sparse tissues like CD3
    affine_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    affine_reg.SetMetricSamplingStrategy(affine_reg.RANDOM)
    affine_reg.SetMetricSamplingPercentage(0.1)  # Increased from 0.01 for sparse images
    
    # Optimizer: Gradient Descent with line search
    affine_reg.SetOptimizerAsGradientDescentLineSearch(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    affine_reg.SetOptimizerScalesFromPhysicalShift()
    
    # Transform: Affine (12 DOF in 2D: rotation, translation, scale, shear)
    # Start from identity since we already centered
    initial_affine = sitk.AffineTransform(2)
    affine_reg.SetInitialTransform(initial_affine, inPlace=False)
    
    # Multi-resolution pyramid (coarse to fine)
    affine_reg.SetShrinkFactorsPerLevel([4, 2, 1])
    affine_reg.SetSmoothingSigmasPerLevel([4, 2, 0])
    affine_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Interpolator
    affine_reg.SetInterpolator(sitk.sitkLinear)
    
    # Execute affine registration on centered image
    affine_transform = affine_reg.Execute(fixed, moving_centered)
    print(f"      Affine optimizer stop: {affine_reg.GetOptimizerStopConditionDescription()}")
    print(f"      Affine iterations: {affine_reg.GetOptimizerIteration()}")
    print(f"      Affine metric: {affine_reg.GetMetricValue():.4f}")
    
    # Apply affine transform
    moving_affine = sitk.Resample(moving_centered, fixed, affine_transform, sitk.sitkLinear, 0.0)
    
    # ==========================================
    # PHASE 2: B-spline Registration (Local Deformation)
    # ==========================================
    print(f"    Phase 2: B-spline (local deformation)...")
    
    bspline_reg = sitk.ImageRegistrationMethod()
    
    # Metric: Same as affine
    bspline_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    bspline_reg.SetMetricSamplingStrategy(bspline_reg.RANDOM)
    bspline_reg.SetMetricSamplingPercentage(0.01)
    
    # Optimizer: LBFGSB (better for deformable)
    bspline_reg.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=100,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1000
    )
    
    # Transform: B-spline with control point grid
    transform_domain_mesh_size = [8, 8]  # Control points (more = more flexible)
    bspline_transform = sitk.BSplineTransformInitializer(
        fixed,
        transform_domain_mesh_size
    )
    bspline_reg.SetInitialTransform(bspline_transform, inPlace=True)
    
    # Multi-resolution for B-spline
    bspline_reg.SetShrinkFactorsPerLevel([2, 1])
    bspline_reg.SetSmoothingSigmasPerLevel([2, 0])
    bspline_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Interpolator
    bspline_reg.SetInterpolator(sitk.sitkLinear)
    
    # Execute B-spline registration (starting from affine result)
    final_transform = bspline_reg.Execute(fixed, moving_affine)
    print(f"      B-spline optimizer stop: {bspline_reg.GetOptimizerStopConditionDescription()}")
    print(f"      B-spline iterations: {bspline_reg.GetOptimizerIteration()}")
    print(f"      B-spline metric: {bspline_reg.GetMetricValue():.4f}")
    
    # ==========================================
    # Composite Transform (Center + Affine + B-spline)
    # ==========================================
    composite = sitk.CompositeTransform(2)
    composite.AddTransform(center_transform)  # CRITICAL: Include center alignment
    composite.AddTransform(affine_transform)
    composite.AddTransform(final_transform)
    
    # Apply composite transform to original moving image
    registered = sitk.Resample(moving, fixed, composite, sitk.sitkLinear, 0.0)
    registered_array = sitk.GetArrayFromImage(registered)
    
    # ==========================================
    # Quality Check Visualization (Optional)
    # ==========================================
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original images
        axes[0, 0].imshow(fixed_img, cmap='gray')
        axes[0, 0].set_title(f'Reference: {fixed_name}')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(moving_img, cmap='gray')
        axes[0, 1].set_title(f'Moving: {moving_name}')
        axes[0, 1].axis('off')
        
        # After affine
        axes[0, 2].imshow(sitk.GetArrayFromImage(moving_affine), cmap='gray')
        axes[0, 2].set_title('After Affine')
        axes[0, 2].axis('off')
        
        # After B-spline
        axes[1, 0].imshow(registered_array, cmap='gray')
        axes[1, 0].set_title('After Affine + B-spline')
        axes[1, 0].axis('off')
        
        # Checkerboard overlay
        checker = create_checkerboard(fixed_img, registered_array)
        axes[1, 1].imshow(checker, cmap='gray')
        axes[1, 1].set_title('Checkerboard Overlay')
        axes[1, 1].axis('off')
        
        # Difference image
        diff = np.abs(fixed_img.astype(float) - registered_array.astype(float))
        axes[1, 2].imshow(diff, cmap='hot')
        axes[1, 2].set_title('Difference (lower = better)')
        axes[1, 2].axis('off')
        
        plt.suptitle(f'Registration QC: {moving_name} → {fixed_name}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        qc_path = output_dir / f'registration_{moving_name}_to_{fixed_name}.png'
        plt.savefig(qc_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      ✅ QC saved: {qc_path}")
    
    print(f"  ✅ Registration complete")
    return registered_array, composite


def create_checkerboard(img1: np.ndarray, img2: np.ndarray, 
                        square_size: int = 50) -> np.ndarray:
    """Create checkerboard overlay of two images for QC."""
    checker = img1.copy()
    h, w = img1.shape
    
    for i in range(0, h, square_size):
        for j in range(0, w, square_size):
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                checker[i:i+square_size, j:j+square_size] = \
                    img2[i:i+square_size, j:j+square_size]
    
    return checker


def extract_multimodal_features(sample_id: str,
                                sample_dirs: Dict[str, Path],
                                registered_images: Dict[str, np.ndarray],
                                transforms: Dict[str, sitk.Transform]) -> pd.DataFrame:
    """
    Extract multi-modal features for each nucleus in H&E.
    
    For each nucleus:
    - Load its morphology from H&E features CSV
    - Query registered IHC images at nucleus location
    - Extract local IHC density around nucleus
    - Combine into comprehensive feature vector
    
    Returns:
        DataFrame with columns:
        - All morphology features from H&E
        - CD3_intensity, CD3_local_density, CD3_proximity
        - GFAP_intensity, GFAP_local_density, GFAP_proximity
        - [same for all other markers]
        - sample_id
    """
    
    print(f"\n  Extracting multi-modal features for {sample_id}...")
    
    # Load H&E nuclei features
    he_dir = sample_dirs['H&E']
    he_csv = he_dir / 'features' / f'{he_dir.name}_final.csv'
    
    if not he_csv.exists():
        raise FileNotFoundError(f"H&E features not found: {he_csv}")
    
    df_he = pd.read_csv(he_csv)
    print(f"    Loaded {len(df_he):,} nuclei from H&E")
    
    # Add sample ID
    df_he['sample_id'] = sample_id
    
    # Get H&E image dimensions for coordinate scaling
    he_thumb = registered_images['H&E']
    thumb_height, thumb_width = he_thumb.shape
    
    # Need to get original WSI dimensions to scale coordinates
    # Read from preview metadata if available, otherwise estimate
    preview_dir = he_dir / 'preview'
    
    # For each IHC marker, add intensity and local density features
    for marker in STAIN_MARKERS:
        if marker not in sample_dirs:
            print(f"    ⚠️  {marker} not available - filling with zeros")
            df_he[f'{marker}_intensity'] = 0.0
            df_he[f'{marker}_local_density'] = 0.0
            df_he[f'{marker}_nearby_positive'] = 0
            continue
        
        print(f"    Processing {marker}...")
        
        # Load marker features
        marker_dir = sample_dirs[marker]
        marker_csv = marker_dir / 'features' / f'{marker_dir.name}_final.csv'
        
        if not marker_csv.exists():
            print(f"      ⚠️  Features CSV not found - skipping")
            df_he[f'{marker}_intensity'] = 0.0
            df_he[f'{marker}_local_density'] = 0.0
            df_he[f'{marker}_nearby_positive'] = 0
            continue
        
        df_marker = pd.read_csv(marker_csv)
        
        # Extract features at each H&E nucleus location
        he_intensities = []
        he_local_densities = []
        he_nearby_positives = []
        
        # Since we registered at thumbnail resolution, we need to:
        # 1. Scale H&E nucleus coords to thumbnail space
        # 2. Use spatial matching in registered space
        
        # Get scale factor from full res to thumbnail
        # (Assuming centroids are in full resolution pixel coordinates)
        # Check for different column naming conventions
        he_x_col = 'centroid_x' if 'centroid_x' in df_he.columns else 'x'
        he_y_col = 'centroid_y' if 'centroid_y' in df_he.columns else 'y'
        
        if he_x_col in df_he.columns and he_y_col in df_he.columns:
            # Estimate scale from coordinate ranges
            he_x_max = df_he[he_x_col].max()
            he_y_max = df_he[he_y_col].max()
            
            scale_x = thumb_width / he_x_max if he_x_max > 0 else 1.0
            scale_y = thumb_height / he_y_max if he_y_max > 0 else 1.0
            
            # Scale H&E coordinates to thumbnail space
            he_coords_thumb = np.column_stack([
                df_he[he_x_col].values * scale_x,
                df_he[he_y_col].values * scale_y
            ])
            
            # Scale marker coordinates to thumbnail space (same ratio)
            marker_x_col = 'centroid_x' if 'centroid_x' in df_marker.columns else 'x'
            marker_y_col = 'centroid_y' if 'centroid_y' in df_marker.columns else 'y'
            
            if marker_x_col in df_marker.columns:
                marker_x_max = df_marker[marker_x_col].max()
                marker_y_max = df_marker[marker_y_col].max()
                
                marker_scale_x = thumb_width / marker_x_max if marker_x_max > 0 else scale_x
                marker_scale_y = thumb_height / marker_y_max if marker_y_max > 0 else scale_y
                
                marker_coords_thumb = np.column_stack([
                    df_marker[marker_x_col].values * marker_scale_x,
                    df_marker[marker_y_col].values * marker_scale_y
                ])
                
                # For each H&E nucleus, find nearby marker nuclei
                # Use spatial KDTree for efficient nearest neighbor search
                from scipy.spatial import cKDTree
                
                # Build tree of marker nuclei (in registered space)
                marker_tree = cKDTree(marker_coords_thumb)
                
                # Get brown intensity if available
                if 'brown_intensity' in df_marker.columns:
                    marker_intensities = df_marker['brown_intensity'].values
                else:
                    marker_intensities = np.zeros(len(df_marker))
                
                # For each H&E nucleus
                print(f"      Extracting {marker} features for {len(df_he):,} nuclei...")
                for he_coord in tqdm(he_coords_thumb, desc=f"      {marker}", 
                                    leave=False, disable=len(he_coords_thumb)<1000):
                    
                    # Find nearest marker nucleus (within reasonable distance)
                    dist, idx = marker_tree.query(he_coord, k=1, distance_upper_bound=50)
                    
                    if dist < 50:  # Within 50 pixels in thumbnail space
                        # Use nearest marker's intensity
                        intensity = marker_intensities[idx]
                    else:
                        # Too far - likely no marker at this location
                        intensity = 0.0
                    
                    # Count positive markers nearby (within 100 pixels)
                    nearby_indices = marker_tree.query_ball_point(he_coord, r=100)
                    nearby_intensities = marker_intensities[nearby_indices]
                    
                    # Positive threshold (common for DAB)
                    positive_count = np.sum(nearby_intensities > 0.15)
                    local_density = len(nearby_indices) / (np.pi * 100**2) if nearby_indices else 0
                    
                    he_intensities.append(intensity)
                    he_local_densities.append(local_density)
                    he_nearby_positives.append(positive_count)
            
            else:
                # No coordinate data in marker
                print(f"      ⚠️  No coordinate data in {marker}")
                he_intensities = [0.0] * len(df_he)
                he_local_densities = [0.0] * len(df_he)
                he_nearby_positives = [0] * len(df_he)
        
        else:
            # No coordinate data in H&E
            print(f"      ⚠️  No coordinate data in H&E")
            he_intensities = [0.0] * len(df_he)
            he_local_densities = [0.0] * len(df_he)
            he_nearby_positives = [0] * len(df_he)
        
        # Add to dataframe
        df_he[f'{marker}_intensity'] = he_intensities
        df_he[f'{marker}_local_density'] = he_local_densities
        df_he[f'{marker}_nearby_positive'] = he_nearby_positives
        
        # Statistics
        mean_int = np.mean(he_intensities)
        pct_pos = np.sum(np.array(he_intensities) > 0.15) / len(he_intensities) * 100
        print(f"      ✅ {marker}: mean={mean_int:.3f}, {pct_pos:.1f}% positive")
    
    print(f"  ✅ Features extracted: {df_he.shape}")
    return df_he


def generate_combined_umap(df: pd.DataFrame, output_dir: Path):
    """
    Generate combined UMAP from multi-modal features.
    
    Uses all morphology + IHC features for dimensionality reduction.
    """
    
    print(f"\n{'='*70}")
    print(f"Generating Combined UMAP")
    print(f"{'='*70}\n")
    
    # Select features
    feature_cols = []
    
    # Morphology
    morphology = ['area_px', 'perimeter_px', 'circularity', 'aspect_ratio',
                  'eccentricity', 'solidity']
    feature_cols.extend([c for c in morphology if c in df.columns])
    
    # Density
    density = [c for c in df.columns if 'density_um2' in c]
    feature_cols.extend(density[:6])
    
    # Coherency
    coherency = [c for c in df.columns if 'coherency' in c]
    feature_cols.extend(coherency)
    
    # IHC markers
    for marker in STAIN_MARKERS:
        for suffix in ['_intensity', '_local_density', '_nearby_positive']:
            col = f'{marker}{suffix}'
            if col in df.columns:
                feature_cols.append(col)
    
    print(f"Selected {len(feature_cols)} features")
    print(f"  Morphology: {len([c for c in feature_cols if any(m in c for m in morphology)])}")
    print(f"  Spatial: {len([c for c in feature_cols if 'density' in c or 'coherency' in c])}")
    print(f"  IHC: {len([c for c in feature_cols if any(m in c for m in STAIN_MARKERS)])}")
    
    # Extract feature matrix
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    
    # Standardize
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # UMAP
    print("Running UMAP (this may take a while with many features)...")
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=42,
        verbose=True
    )
    embedding = reducer.fit_transform(X_scaled)
    
    # Cluster
    print("Clustering...")
    n_clusters = 25  # More clusters for multi-modal data
    clusterer = Birch(n_clusters=n_clusters, threshold=0.5)
    clusters = clusterer.fit_predict(embedding)
    
    # Add to dataframe
    df['umap_1'] = embedding[:, 0]
    df['umap_2'] = embedding[:, 1]
    df['combined_cluster'] = clusters
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / 'combined_multimodal.csv'
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved: {output_csv}")
    
    # Visualize
    visualize_combined_umap(df, output_dir, n_clusters)
    
    return df


def visualize_combined_umap(df: pd.DataFrame, output_dir: Path, 
                           n_clusters: int, max_points: int = 150000):
    """Create comprehensive visualizations of combined UMAP."""
    
    print(f"\nCreating comprehensive visualizations...")
    
    # Sample for plotting
    if len(df) > max_points:
        df_viz = df.sample(n=max_points, random_state=42)
    else:
        df_viz = df.copy()
    
    print(f"  Visualizing {len(df_viz):,} nuclei ({len(df):,} total)")
    
    # ==========================================
    # Figure 1: Overview (4x3 grid)
    # ==========================================
    print("  Creating overview figure...")
    fig1 = plt.figure(figsize=(30, 24))
    gs1 = fig1.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    
    # Row 1: Clusters and Sample
    ax00 = fig1.add_subplot(gs1[0, 0])
    sc = ax00.scatter(df_viz['umap_1'], df_viz['umap_2'],
                     c=df_viz['combined_cluster'], cmap='tab20',
                     s=3, alpha=0.7, rasterized=True)
    ax00.set_title(f'Combined Multi-Modal Clustering\n{n_clusters} clusters identified',
                   fontsize=18, fontweight='bold', pad=15)
    ax00.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax00.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    ax00.grid(True, alpha=0.3)
    cb = plt.colorbar(sc, ax=ax00, fraction=0.046, pad=0.04)
    cb.set_label('Cluster ID', fontsize=12, fontweight='bold')
    
    # Add text box
    textstr = f'Multi-modal immunophenotyping\n'
    textstr += f'Features: Morphology + 6 IHC markers\n'
    textstr += f'Total nuclei: {len(df):,}\n'
    textstr += f'Clusters: {n_clusters}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2)
    ax00.text(0.02, 0.98, textstr, transform=ax00.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    
    # By Sample
    ax01 = fig1.add_subplot(gs1[0, 1])
    samples = df_viz['sample_id'].values
    unique_samples = sorted(list(set(samples)))
    sample_to_idx = {s: i for i, s in enumerate(unique_samples)}
    sample_colors = [sample_to_idx[s] for s in samples]
    sc2 = ax01.scatter(df_viz['umap_1'], df_viz['umap_2'],
                      c=sample_colors, cmap='Set1',
                      s=3, alpha=0.7, rasterized=True)
    ax01.set_title('By Tissue Sample', fontsize=18, fontweight='bold', pad=15)
    ax01.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax01.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    ax01.grid(True, alpha=0.3)
    
    # Legend
    handles = []
    for s in unique_samples:
        color = plt.get_cmap('Set1')(sample_to_idx[s] % 9)
        n_nuclei = np.sum(samples == s)
        handles.append(Line2D([0], [0], marker='o', linestyle='',
                             markersize=8, markerfacecolor=color, 
                             markeredgecolor='none',
                             label=f'{s} ({n_nuclei:,})'))
    ax01.legend(handles=handles, loc='upper right', frameon=True, 
               title='Sample', title_fontsize=12, fontsize=11)
    
    # Circularity
    ax02 = fig1.add_subplot(gs1[0, 2])
    if 'circularity' in df_viz.columns:
        sc3 = ax02.scatter(df_viz['umap_1'], df_viz['umap_2'],
                          c=df_viz['circularity'], cmap='viridis',
                          s=3, alpha=0.7, rasterized=True, vmin=0.5, vmax=1.0)
        ax02.set_title('Nuclear Circularity', fontsize=18, fontweight='bold', pad=15)
        ax02.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
        ax02.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
        ax02.grid(True, alpha=0.3)
        cb3 = plt.colorbar(sc3, ax=ax02, fraction=0.046, pad=0.04)
        cb3.set_label('Circularity (0-1)', fontsize=12, fontweight='bold')
    
    # Row 2-4: IHC Markers (6 markers = 2 rows × 3 cols)
    marker_axes = [
        fig1.add_subplot(gs1[1, 0]), fig1.add_subplot(gs1[1, 1]), fig1.add_subplot(gs1[1, 2]),
        fig1.add_subplot(gs1[2, 0]), fig1.add_subplot(gs1[2, 1]), fig1.add_subplot(gs1[2, 2]),
    ]
    
    for idx, marker in enumerate(STAIN_MARKERS):
        if idx >= len(marker_axes):
            break
        
        ax = marker_axes[idx]
        col = f'{marker}_intensity'
        
        if col in df_viz.columns:
            sc = ax.scatter(df_viz['umap_1'], df_viz['umap_2'],
                           c=df_viz[col], cmap='YlOrBr',
                           s=3, alpha=0.7, rasterized=True, vmin=0, vmax=0.5)
            ax.set_title(f'{marker} Marker Intensity', fontsize=16, fontweight='bold', pad=12)
            ax.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
            ax.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label('DAB Intensity', fontsize=11, fontweight='bold')
            
            # Stats
            mean_val = df_viz[col].mean()
            pct_pos = (df_viz[col] > 0.15).sum() / len(df_viz) * 100
            textstr = f'Mean: {mean_val:.3f}\nPositive: {pct_pos:.1f}%'
            props = dict(boxstyle='round', facecolor='white', alpha=0.9, 
                        edgecolor='black', linewidth=1.5)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
        else:
            ax.text(0.5, 0.5, f'{marker}\nNot Available', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Row 4: Density features
    ax30 = fig1.add_subplot(gs1[3, 0])
    if 'density_um2_r150.0' in df_viz.columns:
        sc = ax30.scatter(df_viz['umap_1'], df_viz['umap_2'],
                         c=df_viz['density_um2_r150.0'], cmap='plasma',
                         s=3, alpha=0.7, rasterized=True)
        ax30.set_title('Nuclear Density (150µm)', fontsize=16, fontweight='bold', pad=12)
        ax30.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
        ax30.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
        ax30.grid(True, alpha=0.3)
        cb = plt.colorbar(sc, ax=ax30, fraction=0.046, pad=0.04)
        cb.set_label('Nuclei/µm²', fontsize=11, fontweight='bold')
    
    ax31 = fig1.add_subplot(gs1[3, 1])
    if 'coherency_150um' in df_viz.columns:
        sc = ax31.scatter(df_viz['umap_1'], df_viz['umap_2'],
                         c=df_viz['coherency_150um'], cmap='coolwarm',
                         s=3, alpha=0.7, rasterized=True, vmin=0, vmax=1)
        ax31.set_title('Nuclear Coherency (150µm)', fontsize=16, fontweight='bold', pad=12)
        ax31.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
        ax31.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
        ax31.grid(True, alpha=0.3)
        cb = plt.colorbar(sc, ax=ax31, fraction=0.046, pad=0.04)
        cb.set_label('Coherency (0-1)', fontsize=11, fontweight='bold')
    
    ax32 = fig1.add_subplot(gs1[3, 2])
    if 'area_px' in df_viz.columns:
        sc = ax32.scatter(df_viz['umap_1'], df_viz['umap_2'],
                         c=np.log10(df_viz['area_px'] + 1), cmap='viridis',
                         s=3, alpha=0.7, rasterized=True)
        ax32.set_title('Nuclear Area (log scale)', fontsize=16, fontweight='bold', pad=12)
        ax32.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
        ax32.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
        ax32.grid(True, alpha=0.3)
        cb = plt.colorbar(sc, ax=ax32, fraction=0.046, pad=0.04)
        cb.set_label('log₁₀(Area px)', fontsize=11, fontweight='bold')
    
    plt.suptitle('Combined Multi-Modal UMAP: Immunophenotyping Across All Markers',
                fontsize=24, fontweight='bold', y=0.995)
    
    fig1_path = output_dir / 'combined_umap_overview.png'
    fig1.savefig(fig1_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig1_svg = output_dir / 'combined_umap_overview.svg'
    fig1.savefig(fig1_svg, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    print(f"  ✅ Saved: {fig1_path}")
    print(f"  ✅ Saved: {fig1_svg}")
    
    # ==========================================
    # Figure 2: Cluster Characterization
    # ==========================================
    print("  Creating cluster characterization figure...")
    
    fig2, axes2 = plt.subplots(3, 2, figsize=(20, 24))
    
    # Get cluster statistics
    cluster_stats = []
    for cluster_id in range(n_clusters):
        cluster_mask = df['combined_cluster'] == cluster_id
        cluster_df = df[cluster_mask]
        
        if len(cluster_df) == 0:
            continue
        
        stats = {
            'cluster': cluster_id,
            'count': len(cluster_df),
            'pct': len(cluster_df) / len(df) * 100
        }
        
        # Morphology
        if 'circularity' in cluster_df.columns:
            stats['circularity'] = cluster_df['circularity'].mean()
        if 'area_px' in cluster_df.columns:
            stats['area'] = cluster_df['area_px'].mean()
        
        # Markers
        for marker in STAIN_MARKERS:
            col = f'{marker}_intensity'
            if col in cluster_df.columns:
                stats[f'{marker}_mean'] = cluster_df[col].mean()
                stats[f'{marker}_pct_pos'] = (cluster_df[col] > 0.15).sum() / len(cluster_df) * 100
        
        cluster_stats.append(stats)
    
    df_stats = pd.DataFrame(cluster_stats)
    
    # Plot 1: Cluster sizes
    axes2[0, 0].bar(df_stats['cluster'], df_stats['count'], color='steelblue', edgecolor='black')
    axes2[0, 0].set_xlabel('Cluster ID', fontsize=14, fontweight='bold')
    axes2[0, 0].set_ylabel('Number of Nuclei', fontsize=14, fontweight='bold')
    axes2[0, 0].set_title('Cluster Sizes', fontsize=16, fontweight='bold', pad=15)
    axes2[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Marker expression heatmap
    marker_cols = [f'{m}_mean' for m in STAIN_MARKERS]
    marker_cols = [c for c in marker_cols if c in df_stats.columns]
    
    if marker_cols:
        marker_matrix = df_stats[marker_cols].values.T
        im = axes2[0, 1].imshow(marker_matrix, cmap='YlOrBr', aspect='auto', vmin=0, vmax=0.3)
        axes2[0, 1].set_xlabel('Cluster ID', fontsize=14, fontweight='bold')
        axes2[0, 1].set_yticks(range(len(marker_cols)))
        axes2[0, 1].set_yticklabels([c.replace('_mean', '') for c in marker_cols], fontsize=12)
        axes2[0, 1].set_title('Mean Marker Intensity by Cluster', fontsize=16, fontweight='bold', pad=15)
        plt.colorbar(im, ax=axes2[0, 1], label='Mean Intensity')
    
    # Plots 3-6: Top markers per cluster (bar charts)
    for plot_idx in range(4):
        row = 1 + plot_idx // 2
        col = plot_idx % 2
        ax = axes2[row, col]
        
        if plot_idx < len(STAIN_MARKERS):
            marker = STAIN_MARKERS[plot_idx]
            pct_col = f'{marker}_pct_pos'
            
            if pct_col in df_stats.columns:
                bars = ax.bar(df_stats['cluster'], df_stats[pct_col], 
                            color='coral', edgecolor='black', alpha=0.8)
                
                # Color bars by percentage
                for i, bar in enumerate(bars):
                    pct = df_stats[pct_col].iloc[i]
                    if pct > 50:
                        bar.set_color('darkred')
                    elif pct > 25:
                        bar.set_color('orangered')
                    elif pct > 10:
                        bar.set_color('orange')
                    else:
                        bar.set_color('lightcoral')
                
                ax.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
                ax.set_ylabel('% Positive', fontsize=12, fontweight='bold')
                ax.set_title(f'{marker}⁺ Cells by Cluster', fontsize=14, fontweight='bold', pad=12)
                ax.grid(True, alpha=0.3, axis='y')
                ax.axhline(25, color='gray', linestyle='--', linewidth=1, alpha=0.5)
                ax.set_ylim([0, 100])
    
    plt.suptitle('Cluster Characterization: Marker Expression Profiles',
                fontsize=22, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    fig2_path = output_dir / 'combined_umap_cluster_stats.png'
    fig2.savefig(fig2_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f"  ✅ Saved: {fig2_path}")
    
    # ==========================================
    # Figure 3: Marker Co-expression
    # ==========================================
    print("  Creating marker co-expression figure...")
    
    n_markers = len(STAIN_MARKERS)
    fig3, axes3 = plt.subplots(n_markers, n_markers, figsize=(24, 24))
    
    for i, marker1 in enumerate(STAIN_MARKERS):
        for j, marker2 in enumerate(STAIN_MARKERS):
            ax = axes3[i, j]
            
            col1 = f'{marker1}_intensity'
            col2 = f'{marker2}_intensity'
            
            if i == j:
                # Diagonal: histogram
                if col1 in df_viz.columns:
                    ax.hist(df_viz[col1], bins=50, color='steelblue', 
                           edgecolor='black', alpha=0.7)
                    ax.set_ylabel('Count', fontsize=10)
                    ax.set_title(marker1, fontsize=12, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                           ha='center', va='center', fontsize=12)
                    ax.set_xticks([])
                    ax.set_yticks([])
            
            elif col1 in df_viz.columns and col2 in df_viz.columns:
                # Off-diagonal: scatter
                sample_size = min(10000, len(df_viz))
                df_sample = df_viz.sample(n=sample_size, random_state=42)
                
                ax.scatter(df_sample[col2], df_sample[col1],
                          s=1, alpha=0.3, c='steelblue', rasterized=True)
                
                # Quadrant lines for positive/negative
                ax.axhline(0.15, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
                ax.axvline(0.15, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
                
                if i == n_markers - 1:
                    ax.set_xlabel(marker2, fontsize=10)
                if j == 0:
                    ax.set_ylabel(marker1, fontsize=10)
            
            else:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                       ha='center', va='center', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
    
    plt.suptitle('Marker Co-expression Analysis',
                fontsize=24, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    fig3_path = output_dir / 'combined_umap_coexpression.png'
    fig3.savefig(fig3_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig3)
    print(f"  ✅ Saved: {fig3_path}")
    
    print(f"\n✅ All visualizations complete!")


def main():
    parser = argparse.ArgumentParser(description='Combined multi-modal UMAP analysis')
    parser.add_argument('--results_dir', required=True, help='Directory with slide results')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--samples', nargs='+', default=None,
                       help='Specific samples to process (default: all complete samples)')
    parser.add_argument('--save_qc', action='store_true',
                       help='Save registration QC images')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    print(f"\n{'='*70}")
    print("COMBINED MULTI-MODAL UMAP ANALYSIS")
    print(f"{'='*70}\n")
    
    # Group slides by sample
    samples = group_slides_by_sample(results_dir)
    
    if not samples:
        print("❌ No complete samples found!")
        return
    
    # Filter to requested samples
    if args.samples:
        samples = {k: v for k, v in samples.items() if k in args.samples}
    
    print(f"\nProcessing {len(samples)} samples: {list(samples.keys())}\n")
    
    # Process each sample
    all_sample_data = []
    
    for sample_id, sample_dirs in samples.items():
        print(f"\n{'='*70}")
        print(f"SAMPLE: {sample_id}")
        print(f"{'='*70}\n")
        
        sample_output = output_dir / sample_id
        sample_output.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load H&E reference
            print("Loading H&E reference...")
            he_mask = load_tissue_mask(sample_dirs['H&E'])
            he_thumb = load_thumbnail(sample_dirs['H&E'])
            
            # Register all markers to H&E
            registered_images = {'H&E': he_thumb}
            transforms = {}
            
            qc_dir = sample_output / 'registration_qc' if args.save_qc else None
            
            for marker in STAIN_MARKERS:
                if marker not in sample_dirs:
                    print(f"  ⚠️  {marker} not available - skipping")
                    continue
                
                try:
                    marker_thumb = load_thumbnail(sample_dirs[marker])
                    
                    registered, transform = register_affine_bspline(
                        he_thumb, marker_thumb,
                        output_dir=qc_dir,
                        fixed_name='HE',
                        moving_name=marker
                    )
                    
                    registered_images[marker] = registered
                    transforms[marker] = transform
                    
                except Exception as e:
                    print(f"  ❌ Error registering {marker}: {e}")
            
            # Extract multi-modal features
            df_sample = extract_multimodal_features(
                sample_id, sample_dirs, registered_images, transforms
            )
            
            all_sample_data.append(df_sample)
            
            print(f"\n✅ Completed {sample_id}")
            
        except Exception as e:
            print(f"\n❌ Error processing {sample_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Combine all samples
    if not all_sample_data:
        print("\n❌ No samples successfully processed!")
        return
    
    print(f"\n{'='*70}")
    print("COMBINING ALL SAMPLES")
    print(f"{'='*70}\n")
    
    df_combined = pd.concat(all_sample_data, ignore_index=True)
    print(f"Total nuclei: {len(df_combined):,} from {len(all_sample_data)} samples")
    
    # Generate combined UMAP
    generate_combined_umap(df_combined, output_dir)
    
    print(f"\n{'='*70}")
    print("✅ COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
