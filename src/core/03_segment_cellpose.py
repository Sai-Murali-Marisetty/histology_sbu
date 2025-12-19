import os
import json
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from cellpose.models import CellposeModel
from skimage.measure import regionprops
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
from matplotlib import cm

# Increase PIL decompression bomb limit for large whole slide images
Image.MAX_IMAGE_PIXELS = None


def _ensure_uint16(arr):
    if arr.dtype != np.uint16:
        if arr.max() > np.iinfo(np.uint16).max:
            raise ValueError("Label IDs exceed uint16 range.")
        return arr.astype(np.uint16)
    return arr


def _props_from_labelmask(label_img):
            for p in regionprops(label_img):
                yield {
                    "label": int(p.label),
                    "cy": float(p.centroid[0]),
                    "cx": float(p.centroid[1]),
                    "area_px": float(p.area),
                    "perimeter_px": float(getattr(p, "perimeter", 0.0)),
                    "eccentricity": float(getattr(p, "eccentricity", 0.0)),
                    "solidity": float(getattr(p, "solidity", 0.0)),
                    "major_axis_length": float(getattr(p, "major_axis_length", 0.0)),
                    "minor_axis_length": float(getattr(p, "minor_axis_length", 0.0)),
                    "orientation": float(getattr(p, "orientation", 0.0)),  # ADD THIS LINE
                }


def stable_id(slide, x_um, y_um, ndigits=1):
    key = f"{slide}:{round(x_um, ndigits)}:{round(y_um, ndigits)}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:16]


def dedup_by_radius(df, radius_um, prefer_col="area_px"):
    """Basic radius-based deduplication (keeps larger nuclei)"""
    sort_idx = np.argsort(df[prefer_col].to_numpy(np.float64))[::-1]
    dfs = df.iloc[sort_idx].reset_index(drop=False).rename(columns={"index": "_orig_idx"})

    coords = dfs[["x_um", "y_um"]].to_numpy(np.float32)
    tree = cKDTree(coords)
    keep = np.ones(len(dfs), dtype=bool)

    for i in range(len(dfs)):
        if not keep[i]:
            continue
        nbrs = tree.query_ball_point(coords[i], r=radius_um)
        for j in nbrs:
            if j == i:
                continue
            keep[j] = False

    kept = dfs.loc[keep].copy()
    kept = kept.sort_values("_orig_idx").drop(columns=["_orig_idx"]).reset_index(drop=True)
    return kept


def dedup_across_tiles_advanced(df, distance_threshold_um=3.5, area_diff_threshold=0.2):
    """
    Advanced tile boundary deduplication using centroid distance and area comparison.
    Inspired by HistoVision's trimNuclei2() method.
    
    Args:
        df: DataFrame with x_um, y_um, area_px columns
        distance_threshold_um: Max distance (µm) between centroids to consider as duplicate
        area_diff_threshold: Relative area difference threshold (keep if areas differ by >20%)
    
    Returns:
        Deduplicated DataFrame
    """
    if len(df) == 0:
        return df
    
    coords = df[['x_um', 'y_um']].values
    areas = df['area_px'].values
    
    # Build spatial index for fast neighbor search
    tree = cKDTree(coords)
    
    to_remove = set()
    
    # Check each nucleus against nearby nuclei
    for i in range(len(df)):
        if i in to_remove:
            continue
        
        # Find nearby nuclei within distance threshold
        nearby_indices = tree.query_ball_point(coords[i], r=distance_threshold_um)
        
        for j in nearby_indices:
            if j <= i or j in to_remove:  # Skip self and already processed
                continue
            
            # Calculate actual distance
            dist = np.sqrt(np.sum((coords[i] - coords[j])**2))
            
            if dist < distance_threshold_um:
                # Calculate area difference
                area_ratio = min(areas[i], areas[j]) / max(areas[i], areas[j])
                
                # If areas are very similar (likely same nucleus detected twice)
                if area_ratio > (1 - area_diff_threshold):
                    # Remove the smaller one
                    if areas[i] < areas[j]:
                        to_remove.add(i)
                    else:
                        to_remove.add(j)
    
    # Remove duplicates
    keep_indices = [i for i in range(len(df)) if i not in to_remove]
    result = df.iloc[keep_indices].reset_index(drop=True)
    
    n_removed = len(df) - len(result)
    if n_removed > 0:
        print(f"  🔧 Tile boundary dedup removed {n_removed:,} duplicate nuclei")
    
    return result


def process_batch(model, batch_imgs, batch_infos, diam_px, masks_dir, tiles_dir, viz_dir):
    masks_out, *_ = model.eval(
        batch_imgs,
        diameter=diam_px,
        channels=[0, 0],
        batch_size=len(batch_imgs)
    )

    if isinstance(masks_out, np.ndarray):
        masks_out = [masks_out] if masks_out.ndim == 2 else list(masks_out)

    rows = []
    for (x, y, tile_name), label_img in zip(batch_infos, masks_out):
        if label_img is None or label_img.max() == 0:
            continue

        # === Save 16-bit mask ===
        mask_path = os.path.join(masks_dir, f"mask_{x}_{y}.png")
        Image.fromarray(_ensure_uint16(label_img), mode="I;16").save(mask_path)

        # === Save overlay JPEG with boundaries ===
        try:
            from scipy import ndimage
            from skimage.segmentation import find_boundaries
            
            tile_path = os.path.join(tiles_dir, tile_name)
            tile_img = np.array(Image.open(tile_path).convert("RGB"))

            # Create colored overlay
            label_overlay = cm.nipy_spectral(label_img / (label_img.max() + 1e-5))[:, :, :3]
            label_overlay = (label_overlay * 255).astype(np.uint8)

            # Find boundaries for better visualization
            boundaries = find_boundaries(label_img, mode='outer')
            
            # Create overlay: tissue + colored masks + white boundaries
            overlay = (0.6 * tile_img + 0.4 * label_overlay).astype(np.uint8)
            overlay[boundaries] = [255, 255, 255]  # White boundaries

            viz_path = os.path.join(viz_dir, f"preview_{x}_{y}.jpg")
            Image.fromarray(overlay).save(viz_path, quality=95)
        except Exception as e:
            print(f"⚠️ Could not save overlay for {tile_name}: {e}")
            tile_img = None  # fallback: no color features

        # === Extract features ===
        if tile_img is None:
            continue

        for P in _props_from_labelmask(label_img):
            major = P["major_axis_length"]
            minor = P["minor_axis_length"]
            aspect = (major / minor) if minor > 0 else 0.0

            # Extract RGB mean intensities
            mask = (label_img == P["label"])
            if mask.sum() > 0:
                r_mean = float(tile_img[:, :, 0][mask].mean())
                g_mean = float(tile_img[:, :, 1][mask].mean())
                b_mean = float(tile_img[:, :, 2][mask].mean())
            else:
                r_mean = g_mean = b_mean = 0.0

            rows.append({
                "x": x + P["cx"],
                "y": y + P["cy"],
                "area_px": P["area_px"],
                "perimeter_px": P["perimeter_px"],
                "eccentricity": P["eccentricity"],
                "solidity": P["solidity"],
                "major_axis_length": major,
                "minor_axis_length": minor,
                "aspect_ratio": aspect,
                "orientation": P["orientation"],  # ADD THIS LINE
                "r": r_mean,
                "g": g_mean,
                "b": b_mean,
                "tile_x": x,
                "tile_y": y,
                "tile_id": f"{x}_{y}",
                "tile": tile_name,
                "label": P["label"],
            })
    return rows


def make_qc_panel(viz_dir, out_path, max_images=5, df=None):
    """Create comprehensive QC panel with overlays and statistics"""
    previews = sorted(Path(viz_dir).glob("preview_*.jpg"))
    if not previews:
        print("⚠️ No overlay previews found.")
        return
    
    # Load sample images
    images = [Image.open(p) for p in previews[:max_images]]
    widths, heights = zip(*(img.size for img in images))
    
    # Create panel with space for stats
    panel_height = max(heights)
    panel_width = sum(widths)
    
    # If we have dataframe, add statistics panel below
    if df is not None and len(df) > 0:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        # Create figure for statistics
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 1. Nucleus size distribution
        axes[0].hist(df['area_px'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(df['area_px'].median(), color='red', linestyle='--', linewidth=2, label=f'Median: {df["area_px"].median():.0f}')
        axes[0].set_xlabel('Nucleus Area (px²)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Size Distribution (n={len(df):,})', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # 2. Shape metrics
        if 'eccentricity' in df.columns and 'solidity' in df.columns:
            axes[1].scatter(df['eccentricity'], df['solidity'], s=1, alpha=0.3, c='darkgreen')
            axes[1].set_xlabel('Eccentricity', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Solidity', fontsize=12, fontweight='bold')
            axes[1].set_title('Shape Quality', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].axhline(0.9, color='red', linestyle='--', alpha=0.5, label='Quality threshold')
            axes[1].legend()
        
        # 3. Summary statistics
        axes[2].axis('off')
        stats_text = f"""
SEGMENTATION SUMMARY
{'='*30}

Total Nuclei: {len(df):,}

Area Statistics:
  Mean:   {df['area_px'].mean():.1f} px²
  Median: {df['area_px'].median():.1f} px²
  Std:    {df['area_px'].std():.1f} px²
  Range:  {df['area_px'].min():.0f} - {df['area_px'].max():.0f}

Shape Quality:
"""
        if 'eccentricity' in df.columns:
            stats_text += f"  Eccentricity: {df['eccentricity'].mean():.3f} ± {df['eccentricity'].std():.3f}\n"
        if 'solidity' in df.columns:
            stats_text += f"  Solidity:     {df['solidity'].mean():.3f} ± {df['solidity'].std():.3f}\n"
        if 'aspect_ratio' in df.columns:
            stats_text += f"  Aspect Ratio: {df['aspect_ratio'].mean():.2f} ± {df['aspect_ratio'].std():.2f}\n"
        
        axes[2].text(0.1, 0.5, stats_text, fontsize=11, family='monospace', 
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        # Save stats figure
        stats_path = str(out_path).replace('.jpg', '_stats.png')
        plt.savefig(stats_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"📊 Statistics panel saved: {stats_path}")
    
    # Create overlay montage
    panel = Image.new("RGB", (panel_width, panel_height))
    x_offset = 0
    for img in images:
        panel.paste(img, (x_offset, 0))
        x_offset += img.width
    panel.save(out_path)
    print(f"🖼️ QC panel saved: {out_path}")


def segment_and_merge(tiles_dir, tiles_json, masks_dir, out_csv,
                      slide_id, mpp, diam_um, batch_size, gpu, dedup_radius_um):

    with open(tiles_json, "r") as f:
        meta = json.load(f)
    tiles = meta["tiles"]
    print(f"📦 Loaded {len(tiles)} tiles from {tiles_json}")

    Path(masks_dir).mkdir(parents=True, exist_ok=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    viz_dir = Path(masks_dir).parent / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    diam_px = float(diam_um) / float(mpp)
    print(f"📏 Nucleus diameter: {diam_um} µm → {diam_px:.1f} px at {mpp:.4f} µm/px")

    model = CellposeModel(gpu=gpu, model_type="nuclei")
    print(f"🧠 Cellpose model loaded (GPU={gpu}, model=nuclei)")

    all_rows = []
    batch_imgs, batch_infos = [], []

    for t in tqdm(tiles, desc="🔍 Segmenting tiles"):
        x, y, w, h = t["x"], t["y"], t["w"], t["h"]
        tile_name = f"tile_{x}_{y}.png"
        tile_path = os.path.join(tiles_dir, tile_name)
        if not os.path.exists(tile_path):
            print(f"⚠️ Missing tile: {tile_name}")
            continue

        img = np.array(Image.open(tile_path).convert("RGB"))
        batch_imgs.append(img)
        batch_infos.append((x, y, tile_name))

        if len(batch_imgs) >= batch_size:
            rows = process_batch(model, batch_imgs, batch_infos, diam_px, masks_dir, tiles_dir, viz_dir)
            all_rows.extend(rows)
            batch_imgs, batch_infos = [], []

    if batch_imgs:
        rows = process_batch(model, batch_imgs, batch_infos, diam_px, masks_dir, tiles_dir, viz_dir)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    print(f"🔬 Nuclei before dedup: {len(df):,}")

    df["x_um"] = df["x"] * mpp
    df["y_um"] = df["y"] * mpp

    before = len(df)
    
    # First pass: Remove tile boundary duplicates (close centroids, similar areas)
    df = dedup_across_tiles_advanced(df, distance_threshold_um=3.5, area_diff_threshold=0.2)
    
    # Second pass: Standard radius-based dedup
    df = dedup_by_radius(df, dedup_radius_um)
    print(f"✅ After deduplication: {len(df):,} nuclei (from {before:,}; removed {before-len(df):,})")


    df["slide_id"] = slide_id
    df["nucleus_id"] = [
        stable_id(slide_id, x, y) for x, y in zip(df["x_um"], df["y_um"])
    ]

    key_cols = ["slide_id", "nucleus_id", "x", "y", "x_um", "y_um", "area_px"]
    df = df[key_cols + [c for c in df.columns if c not in key_cols]]

    df.to_csv(out_csv, index=False)
    print(f"📄 Saved features CSV: {out_csv}")

    # Generate comprehensive QC panel with statistics
    qc_dir = Path(masks_dir).parent / "cellpose"
    qc_dir.mkdir(parents=True, exist_ok=True)
    make_qc_panel(viz_dir, qc_dir / "qc_panel.jpg", df=df)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles_dir", required=True)
    ap.add_argument("--tiles_json", required=True)
    ap.add_argument("--masks_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--slide_id", required=True)
    ap.add_argument("--mpp", type=float, required=True)
    ap.add_argument("--diam_um", type=float, default=10.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--dedup_radius_um", type=float, default=6.0)
    args = ap.parse_args()

    segment_and_merge(
        tiles_dir=args.tiles_dir,
        tiles_json=args.tiles_json,
        masks_dir=args.masks_dir,
        out_csv=args.out_csv,
        slide_id=args.slide_id,
        mpp=args.mpp,
        diam_um=args.diam_um,
        batch_size=args.batch_size,
        gpu=args.gpu,
        dedup_radius_um=args.dedup_radius_um
    )
