# src/03_segment_and_merge_stardist.py
import os, json, argparse, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.measure import regionprops
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

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
            "orientation": float(getattr(p, "orientation", 0.0)),
        }

def stable_id(slide, x_um, y_um, ndigits=1):
    key = f"{slide}:{round(x_um, ndigits)}:{round(y_um, ndigits)}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:16]

def dedup_by_radius(df, radius_um, prefer_col="area_px"):
    """Basic radius-based deduplication (keeps larger nuclei)"""
    sort_idx = np.argsort(df[prefer_col].to_numpy(np.float64))[::-1]
    dfs = df.iloc[sort_idx].reset_index(drop=True)
    coords = dfs[["x_um", "y_um"]].to_numpy(np.float32)
    tree = cKDTree(coords)
    keep = np.ones(len(dfs), dtype=bool)
    for i in range(len(dfs)):
        if not keep[i]: continue
        for j in tree.query_ball_point(coords[i], r=radius_um):
            if j != i: keep[j] = False
    return dfs.loc[keep].reset_index(drop=True)


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

def process_batch(model, batch_imgs, batch_infos, diam_px, masks_dir, tiles_dir, viz_dir, prob_thresh, nms_thresh):
    import cv2
    rows = []
    for (x, y, tile_name), img in zip(batch_infos, batch_imgs):
        img_rgb = np.asarray(img)
        
        # Use RGB directly - 2D_versatile_he expects 3 channels
        if img_rgb.ndim != 3:
            print(f"Warning: Expected RGB image, got shape {img_rgb.shape}, skipping")
            continue
        
        # Normalize RGB image
        img_norm = normalize(img_rgb, 1, 99.8, axis=(0,1))
        
        # Final check: must be 3D (H, W, C)
        if img_norm.ndim != 3 or img_norm.shape[2] != 3:
            print(f"Error: Image shape is {img_norm.shape}, expected (H, W, 3), skipping tile {tile_name}")
            continue
        
        # StarDist prediction with RGB
        try:
            labels, _ = model.predict_instances(img_norm, axes='YXC', prob_thresh=prob_thresh, nms_thresh=nms_thresh)
        except Exception as e:
            print(f"Warning: StarDist failed on tile {tile_name}: {e}")
            continue
            
        if labels is None or labels.max() == 0: 
            continue
        label_img = _ensure_uint16(labels)

        # save mask
        mask_path = os.path.join(masks_dir, f"mask_{x}_{y}.png")
        Image.fromarray(label_img).save(mask_path)

        # overlay preview
        preview = img_rgb.copy()
        contours, _ = cv2.findContours((label_img > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview, contours, -1, (255, 0, 0), 2)
        Image.fromarray(preview).save(os.path.join(viz_dir, f"preview_{x}_{y}.jpg"))

        for P in _props_from_labelmask(label_img):
            lbl = P["label"]
            m = (label_img == lbl)
            r = float(img_rgb[...,0][m].mean()) if m.any() else 0.0
            g = float(img_rgb[...,1][m].mean()) if m.any() else 0.0
            b = float(img_rgb[...,2][m].mean()) if m.any() else 0.0
            major = P["major_axis_length"]; minor = P["minor_axis_length"]
            aspect = float(major/minor) if minor > 0 else 0.0
            rows.append({
                "tile": tile_name, "tile_x": x, "tile_y": y, "label": lbl,
                "cx": P["cx"], "cy": P["cy"], "x_px": P["cx"], "y_px": P["cy"],
                "area_px": P["area_px"], "perimeter_px": P["perimeter_px"],
                "eccentricity": P["eccentricity"], "solidity": P["solidity"],
                "major_axis_length": major, "minor_axis_length": minor, "aspect_ratio": aspect,
                "orientation": P["orientation"],
                "r": r, "g": g, "b": b, "tile_id": f"{x}_{y}",
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
    
    # If we have dataframe, add statistics panel
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
                      slide_id, mpp, diam_um, batch_size, gpu, dedup_radius_um,
                      stardist_model, prob_thresh, nms_thresh):
    with open(tiles_json, "r") as f: meta = json.load(f)
    tiles = meta["tiles"]; print(f"📦 Loaded {len(tiles)} tiles from {tiles_json}")

    Path(masks_dir).mkdir(parents=True, exist_ok=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    viz_dir = Path(masks_dir).parent / "viz"; viz_dir.mkdir(parents=True, exist_ok=True)

    # Load model from local path to avoid disk quota issues
    local_model_dir = "/gpfs/scratch/smarisetty/histology/stardist_models"
    
    if stardist_model.lower() in {"2d_versatile_he", "versatile_he"}:
        print(f"🧠 Loading StarDist 2D_versatile_he from local path: {local_model_dir}")
        from stardist.models import StarDist2D
        model = StarDist2D(None, name='2D_versatile_he', basedir=local_model_dir)
    elif stardist_model.lower() in {"2d_versatile_fluo", "versatile_fluo"}:
        model = StarDist2D.from_pretrained("2D_versatile_fluo")
    else:
        model = StarDist2D(None, name=Path(stardist_model).name, basedir=str(Path(stardist_model).parent))

    diam_px = float(diam_um) / float(mpp)
    print(f"📏 Target nucleus diameter: {diam_um} µm → ~{diam_px:.1f} px at {mpp:.4f} µm/px (StarDist ignores diameter)")

    all_rows, batch_imgs, batch_infos = [], [], []
    for t in tqdm(tiles, desc="Segmenting (StarDist)"):
        x, y = t["x"], t["y"]
        tile_name = f"tile_{x}_{y}.png"
        tile_path = os.path.join(tiles_dir, tile_name)
        if not os.path.exists(tile_path):
            print(f"⚠️ Missing tile: {tile_name}"); continue
        img = Image.open(tile_path).convert("RGB")
        batch_imgs.append(img); batch_infos.append((x, y, tile_name))
        if len(batch_imgs) >= batch_size:
            all_rows += process_batch(model, batch_imgs, batch_infos, diam_px, masks_dir, tiles_dir, viz_dir, prob_thresh, nms_thresh)
            batch_imgs, batch_infos = [], []
    if batch_imgs:
        all_rows += process_batch(model, batch_imgs, batch_infos, diam_px, masks_dir, tiles_dir, viz_dir, prob_thresh, nms_thresh)

    if not all_rows:
        print("⚠️ No nuclei detected."); pd.DataFrame(columns=["slide_id","nucleus_id"]).to_csv(out_csv, index=False); return

    df = pd.DataFrame(all_rows)

    # map to absolute coords - FIXED coordinate mapping
    tile_size = meta.get("tile_size", 1024)
    level = meta.get("level", 0)
    downsample = meta.get("downsample", 1.0)
    
    abs_x_px, abs_y_px = [], []
    for _, row in df.iterrows():
        tx, ty = map(int, row["tile_id"].split("_"))
        
        # Get tile coordinates from metadata
        tile_meta = next((tt for tt in tiles if tt["x"]==tx and tt["y"]==ty), None)
        
        if tile_meta and "x_lvl0" in tile_meta and "y_lvl0" in tile_meta:
            # Use level-0 coordinates from tile metadata (from 02_tile.py fix)
            ox = tile_meta["x_lvl0"]
            oy = tile_meta["y_lvl0"]
        else:
            # Fallback: compute from tile grid
            ox = tx * tile_size * downsample
            oy = ty * tile_size * downsample
        
        # Add nucleus offset
        abs_x_px.append(ox + row["x_px"])
        abs_y_px.append(oy + row["y_px"])    
    df["x"] = np.array(abs_x_px, dtype=np.float32)
    df["y"] = np.array(abs_y_px, dtype=np.float32)
    df["x_um"] = df["x"] * float(mpp); df["y_um"] = df["y"] * float(mpp)

    before = len(df)
    
    # First pass: Remove tile boundary duplicates (close centroids, similar areas)
    df = dedup_across_tiles_advanced(df, distance_threshold_um=3.5, area_diff_threshold=0.2)
    
    # Second pass: Standard radius-based dedup
    df = dedup_by_radius(df, radius_um=dedup_radius_um, prefer_col="area_px")
    print(f"✅ After deduplication: {len(df):,} nuclei (from {before:,}; removed {before-len(df):,})")


    df["slide_id"] = slide_id
    df["nucleus_id"] = [stable_id(slide_id, x, y) for x, y in zip(df["x_um"], df["y_um"])]
    key_cols = ["slide_id", "nucleus_id", "x", "y", "x_um", "y_um", "area_px"]
    df = df[key_cols + [c for c in df.columns if c not in key_cols]]
    df.to_csv(out_csv, index=False); print(f"📄 Saved features CSV: {out_csv}")

    # Generate comprehensive QC panel with statistics
    qc_dir = Path(masks_dir).parent / "stardist"
    qc_dir.mkdir(parents=True, exist_ok=True)
    make_qc_panel(viz_dir, qc_dir / "qc_panel.jpg", df=df)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles_dir", required=True)
    ap.add_argument("--tiles_json", required=True)
    ap.add_argument("--masks_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--slide_id", required=True)
    ap.add_argument("--mpp", type=float, required=True)
    ap.add_argument("--diam_um", type=float, default=10.0)
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size for GPU inference (default 32 for A100)")
    ap.add_argument("--gpu", action="store_true", help="(unused; TF handles this)")
    ap.add_argument("--dedup_radius_um", type=float, default=6.0)
    ap.add_argument("--stardist_model", default="2D_versatile_he", help='{"2D_versatile_he","2D_versatile_fluo"} or path to custom model dir')
    ap.add_argument("--prob_thresh", type=float, default=0.5)
    ap.add_argument("--nms_thresh", type=float, default=0.4)
    args = ap.parse_args()

    segment_and_merge(
        tiles_dir=args.tiles_dir, tiles_json=args.tiles_json, masks_dir=args.masks_dir, out_csv=args.out_csv,
        slide_id=args.slide_id, mpp=args.mpp, diam_um=args.diam_um, batch_size=args.batch_size, gpu=args.gpu,
        dedup_radius_um=args.dedup_radius_um, stardist_model=args.stardist_model,
        prob_thresh=args.prob_thresh, nms_thresh=args.nms_thresh
    )

if __name__ == "__main__":
    main()
