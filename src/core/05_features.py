import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import os
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from PIL import Image

# Increase PIL decompression bomb limit for large whole slide images
Image.MAX_IMAGE_PIXELS = None


def compute_coherency(df, radius_um):
    """
    Compute coherency (nuclear alignment) in local neighborhoods.
    
    Coherency measures how aligned nuclear orientations are:
    - 0 = random orientations
    - 1 = perfectly aligned
    
    Uses structure tensor approach similar to OrientationJ.
    """
    coords = np.vstack([df["x_um"], df["y_um"]]).T
    tree = cKDTree(coords)
    
    # Calculate orientation angle from major/minor axis
    # Angle of major axis relative to horizontal
    # angles = np.arctan2(df["minor_axis_length"], df["major_axis_length"])
    angles = df['orientation']
    
    coherency = []
    for i, p in enumerate(coords):
        idx = tree.query_ball_point(p, radius_um)
        
        if len(idx) < 3:  # Need minimum neighbors for meaningful coherency
            coherency.append(0.0)
            continue
        
        # Get local orientations
        local_angles = angles.iloc[idx].values if hasattr(angles, 'iloc') else angles[idx]
        
        # Compute structure tensor components
        Jxx = np.mean(np.cos(local_angles)**2)
        Jyy = np.mean(np.sin(local_angles)**2)
        Jxy = np.mean(np.cos(local_angles) * np.sin(local_angles))
        
        # Coherency from eigenvalues of structure tensor
        trace = Jxx + Jyy
        det = Jxx * Jyy - Jxy**2
        
        # Avoid numerical errors
        discriminant = trace**2 - 4*det
        if discriminant < 0:
            discriminant = 0
        
        lambda1 = 0.5 * (trace + np.sqrt(discriminant))
        lambda2 = 0.5 * (trace - np.sqrt(discriminant))
        
        if lambda1 + lambda2 > 1e-10:
            coh = (lambda1 - lambda2) / (lambda1 + lambda2)
        else:
            coh = 0.0
        
        coherency.append(coh)
    
    return np.array(coherency)


def compute_local_stats(df, feature, radius_um):
    """
    Compute mean, variance, and coefficient of variation for local neighborhoods.
    
    Returns:
        local_mean: average value in neighborhood
        local_var: variance in neighborhood
        local_cv: coefficient of variation (std/mean)
    """
    coords = np.vstack([df["x_um"], df["y_um"]]).T
    tree = cKDTree(coords)
    values = df[feature].to_numpy()
    
    local_mean = []
    local_var = []
    local_cv = []
    
    for i, p in enumerate(coords):
        idx = tree.query_ball_point(p, radius_um)
        local_vals = values[idx]
        
        if len(local_vals) == 0:
            local_mean.append(values[i])
            local_var.append(0.0)
            local_cv.append(0.0)
            continue
        
        mean_val = np.mean(local_vals)
        var_val = np.var(local_vals)
        std_val = np.std(local_vals)
        cv_val = std_val / mean_val if mean_val > 1e-10 else 0.0
        
        local_mean.append(mean_val)
        local_var.append(var_val)
        local_cv.append(cv_val)
    
    return np.array(local_mean), np.array(local_var), np.array(local_cv)


def compute_local_feature(df, feature, radius_um):
    """Original function - kept for backward compatibility"""
    coords = np.vstack([df["x_um"], df["y_um"]]).T
    tree = cKDTree(coords)
    values = df[feature].to_numpy()
    smoothed = []
    for i, p in enumerate(coords):
        idx = tree.query_ball_point(p, radius_um)
        local_vals = values[idx]
        smoothed.append(np.median(local_vals))
    return smoothed


def overlay_feature(df, feature, thumb_path, out_path, title, cmap="plasma", vmin=None, vmax=None):
    """Generate overlay visualization of a feature on the slide thumbnail"""
    if not Path(thumb_path).exists():
        print(f"Warning: Missing thumbnail: {thumb_path}")
        return
    
    img = np.array(Image.open(thumb_path).convert("RGB"))
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img)
    
    # Use percentile clipping if not specified
    if vmin is None:
        vmin = np.percentile(df[feature], 2)
    if vmax is None:
        vmax = np.percentile(df[feature], 98)
    
    sc = ax.scatter(df["x"], df["y"], c=df[feature], cmap=cmap,
                    s=4, edgecolor="none", vmin=vmin, vmax=vmax, alpha=0.7)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis("off")
    plt.colorbar(sc, ax=ax, shrink=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved overlay: {out_path}")


def make_qc_panel(qc_dir, slide_id, overlay_files):
    """Create QC panel from multiple overlay images"""
    images = [Image.open(f) for f in overlay_files if Path(f).exists()]
    if not images:
        print("Warning: No overlay images found for QC panel")
        return
    
    widths, heights = zip(*(img.size for img in images))
    panel_width = sum(widths)
    panel_height = max(heights)
    
    panel = Image.new("RGB", (panel_width, panel_height), color=(255, 255, 255))
    x_offset = 0
    for img in images:
        panel.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    
    panel_path = os.path.join(qc_dir, f"panel_overlay_{slide_id}.jpg")
    panel.save(panel_path, quality=90)
    print(f"Saved QC panel: {panel_path}")


def plot_orientation_vectors(df, thumb_path, out_path, subsample=50, arrow_scale=20):
    """
    Visualize nucleus orientations as arrows overlaid on tissue.
    
    Args:
        df: DataFrame with 'x', 'y', 'orientation' columns
        thumb_path: Path to tissue thumbnail
        out_path: Output path for visualization
        subsample: Show every Nth nucleus (default: 50)
        arrow_scale: Arrow length multiplier (default: 20)
    """
    img = np.array(Image.open(thumb_path).convert('RGB'))
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(img)
    
    # Subsample to avoid clutter
    df_sub = df.iloc[::subsample].copy()
    
    # Use orientation column directly (radians)
    if 'orientation' not in df_sub.columns:
        print("Warning: 'orientation' column not found, skipping vector visualization")
        return
    
    angles = df_sub['orientation'].values
    
    # Draw arrows
    for i, (idx, row) in enumerate(df_sub.iterrows()):
        angle = angles[i]  # Use position in subsampled array, not original index
        
        # Arrow length based on major axis if available, otherwise fixed
        if 'major_axis_length' in row:
            length = row['major_axis_length'] / 2 * arrow_scale
        else:
            length = 10 * arrow_scale
        
        dx = length * np.cos(angle)
        dy = length * np.sin(angle)
        
        ax.arrow(row['x'], row['y'], dx, dy,
                head_width=5, head_length=8,
                fc='cyan', ec='blue', alpha=0.7, linewidth=1.5)
    
    ax.set_title(f'Nucleus Orientation Vectors (every {subsample}th nucleus)', 
                fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved orientation vectors: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Enrich nucleus features with coherency and local statistics")
    parser.add_argument("--input_csv", required=True, help="Input CSV with nucleus features")
    parser.add_argument("--out_csv", required=True, help="Output CSV with enriched features")
    parser.add_argument("--thumb", required=True, help="Path to thumbnail image")
    parser.add_argument("--out_dir", required=True, help="Output directory for visualizations")
    parser.add_argument("--radii_um", nargs="+", type=float, default=[50, 100, 150],
                       help="Radii for local feature computation (micrometers)")
    parser.add_argument("--features", nargs="+", default=["circularity", "gray_mean"],
                       help="Features to compute local statistics for")
    
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df):,} nuclei")

    # Calculate circularity if not present
    if "circularity" not in df.columns:
        print("Computing circularity...")
        df["circularity"] = np.clip(4 * np.pi * df["area_px"] / np.maximum(df["perimeter_px"] ** 2, 1e-6), 0.0, 1.0)

    # Calculate gray_mean from RGB if available
    has_rgb = {"r", "g", "b"}.issubset(df.columns)
    if has_rgb and "gray_mean" not in df.columns:
        print("Computing gray_mean from RGB channels...")
        df["gray_mean"] = (df["r"] + df["g"] + df["b"]) / 3.0
    elif not has_rgb and "gray_mean" in args.features:
        print("Warning: RGB columns not found, skipping gray_mean")
        args.features = [f for f in args.features if f != "gray_mean"]

    # Ensure output directory exists
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    overlay_files = []

    print(f"\nProcessing {len(args.radii_um)} radii: {args.radii_um}")
    
    # === NEW: Compute coherency for each radius ===
    for radius in args.radii_um:
        print(f"\nComputing coherency at {radius} um...")
        coherency_col = f"coherency_{int(radius)}um"
        df[coherency_col] = compute_coherency(df, radius)
        
        # Visualize coherency
        out_img = os.path.join(args.out_dir, f"overlay_{coherency_col}.jpg")
        overlay_feature(df, coherency_col, args.thumb, out_img, 
                       title=f"Nuclear Coherency ({int(radius)}µm)", 
                       cmap="coolwarm", vmin=0, vmax=1)
        overlay_files.append(out_img)

    # === NEW: Compute variance and CV for each feature ===
    print("\nComputing local statistics (mean, variance, CV)...")
    for feature in args.features:
        if feature not in df.columns:
            print(f"Warning: Feature '{feature}' not found in data, skipping")
            continue
        
        for radius in args.radii_um:
            print(f"  {feature} at {radius} um...")
            mean, var, cv = compute_local_stats(df, feature, radius)
            
            mean_col = f"{feature}_local_mean_{int(radius)}um"
            var_col = f"{feature}_local_variance_{int(radius)}um"
            cv_col = f"{feature}_local_cv_{int(radius)}um"
            
            df[mean_col] = mean
            df[var_col] = var
            df[cv_col] = cv
            
            # Visualize variance (most informative)
            out_img = os.path.join(args.out_dir, f"overlay_{var_col}.jpg")
            overlay_feature(df, var_col, args.thumb, out_img, 
                           title=f"{feature.replace('_', ' ').title()} Variance ({int(radius)}µm)")
            overlay_files.append(out_img)

    # === ORIGINAL: Local median features (kept for compatibility) ===
    print("\nComputing local median features (original method)...")
    for feature in args.features:
        if feature not in df.columns:
            continue
        for r in args.radii_um:
            smoothed = compute_local_feature(df, feature, r)
            new_col = f"{feature}_local_median_{int(r)}um"
            df[new_col] = smoothed

    # === Per-channel overlays if RGB available ===
    if has_rgb:
        print("\nGenerating per-channel overlays...")
        cmap_map = {"r": "Reds", "g": "Greens", "b": "Blues"}
        for channel in ["r", "g", "b"]:
            out_img = os.path.join(args.out_dir, f"overlay_{channel}.jpg")
            overlay_feature(df, channel, args.thumb, out_img, 
                           title=f"{channel.upper()} Channel Intensity", 
                           cmap=cmap_map[channel])
            overlay_files.append(out_img)

    # Save enriched CSV
    print(f"\nSaving enriched features to {args.out_csv}...")
    df.to_csv(args.out_csv, index=False)
    print(f"Total features: {len(df.columns)} columns")

    # Generate orientation vector visualization
    print("\nGenerating orientation vector visualization...")
    orientation_out = os.path.join(args.out_dir, "orientation_vectors.png")
    plot_orientation_vectors(df, args.thumb, orientation_out, subsample=50, arrow_scale=20)

    # Generate QC panel
    print("\nGenerating QC panel...")
    make_qc_panel(args.out_dir, Path(args.input_csv).stem, overlay_files[:6])  # First 6 images

    print("\nDone!")
    print(f"Output CSV: {args.out_csv}")
    print(f"Visualizations: {args.out_dir}")


if __name__ == "__main__":
    main()
