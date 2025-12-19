#!/usr/bin/env python3
# 01_tissue_mask.py — Improved Tissue Detection Mask Generator
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import sys

# Increase PIL decompression bomb limit for large whole slide images
Image.MAX_IMAGE_PIXELS = None  # Remove limit for trusted medical images

try:
    import openslide
except Exception:
    sys.stderr.write(
        "ERROR: openslide-python is required. Install it via pip and ensure OpenSlide libs are installed.\n"
    )
    raise


def generate_tissue_mask(slide_path: str, out_path: str, level: int = 0):
    """
    Generate tissue mask from whole slide image.
    
    Args:
        slide_path: Path to whole slide image
        out_path: Output path for mask
        level: Pyramid level to use (default 0 for full resolution with A100)
               Use level 0 for highest resolution (recommended with GPU)
               Use level 1 for good balance
               Use level 2 for fastest (may miss small tissue regions)
    """
    slide = openslide.OpenSlide(slide_path)
    if level >= slide.level_count:
        raise ValueError(f"Slide has only {slide.level_count} levels; got level={level}.")

    w, h = slide.level_dimensions[level]
    print(f"ℹ️  Reading slide at level {level}: {w}×{h} pixels")
    img = slide.read_region((0, 0), level, (w, h)).convert("RGB")
    img_np = np.array(img)

    # --- HSV threshold (looser to include more tissue) ---
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)
    mask = ((S > 10) & (V > 20)).astype(np.uint8) * 255

    # --- Morphological cleanup (scale kernel size with level) ---
    # Larger kernels for lower resolution levels
    close_size = 25 if level >= 2 else 15 if level == 1 else 10
    open_size = 5 if level >= 2 else 3
    
    # Close gaps inside tissue
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8), iterations=2)
    # Remove small specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8), iterations=1)

    # --- Keep all significant tissue components (NOT just largest) ---
    # This preserves multiple tissue pieces (e.g., upper right regions)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        # Calculate minimum area threshold adaptively but conservatively small
        # We keep any component larger than either 5,000 px or 0.001% of image area
        # (0.00001 fraction) whichever is larger. This keeps legitimate small
        # tissue shards while still removing specks.
        total_area = mask.shape[0] * mask.shape[1]
        min_area = max(int(total_area * 0.00001), 5000)

        # Keep all components above threshold
        final_mask = np.zeros_like(mask)
        kept_components = 0
        for label_id in range(1, num_labels):  # Skip background (0)
            component_area = stats[label_id, cv2.CC_STAT_AREA]
            if component_area >= min_area:
                final_mask[labels == label_id] = 255
                kept_components += 1

        mask = final_mask
        print(f"ℹ️  Kept {kept_components} tissue components (min area: {min_area:,} pixels)")

    # --- Save result ---
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save binary mask
    Image.fromarray(mask).save(out_path)
    print(f"✅ Saved tissue mask at level {level}: {out_path}  (size={w}×{h})")
    
    # --- Create visual overlay for inspection ---
    # Save a colored version overlaid on the original image for easy visual inspection
    visual_path = out_path.parent / f"{out_path.stem}_visual.png"

    # Create red overlay on original image
    overlay = img_np.copy()
    mask_bool = mask > 0
    overlay[mask_bool] = (overlay[mask_bool].astype(np.uint16) * 0.5 +
                         np.array([255, 0, 0], dtype=np.uint16) * 0.5).astype(np.uint8)

    # Add red border around mask regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = cv2.drawContours(overlay, contours, -1, (255, 0, 0), thickness=3)

    Image.fromarray(overlay).save(visual_path)
    print(f"ℹ️  Saved visual overlay: {visual_path}")

    # --- Save downsampled thumbnails for quick human inspection ---
    thumb_max = 2048
    try:
        # Create a thumbnail of the binary mask (nearest neighbor to preserve shape)
        mask_img = Image.fromarray(mask)
        if max(w, h) > thumb_max:
            scale = thumb_max / max(w, h)
            thumb_size = (int(w * scale), int(h * scale))
            mask_thumb = mask_img.resize(thumb_size, resample=Image.NEAREST)
        else:
            mask_thumb = mask_img
        thumb_path = out_path.parent / f"{out_path.stem}_mask_thumb.png"
        mask_thumb.save(thumb_path)

        # And a thumbnail of the overlay
        overlay_img = Image.fromarray(overlay)
        if max(w, h) > thumb_max:
            overlay_thumb = overlay_img.resize(thumb_size)
        else:
            overlay_thumb = overlay_img
        overlay_thumb_path = out_path.parent / f"{out_path.stem}_visual_thumb.png"
        overlay_thumb.save(overlay_thumb_path)
        print(f"ℹ️  Saved thumbnails: {thumb_path}, {overlay_thumb_path}")
    except Exception:
        # Non-critical: if thumbnail generation fails, keep going
        pass


def main():
    ap = argparse.ArgumentParser(description="Generate a binary tissue mask from a WSI using HSV thresholds and morphology.")
    ap.add_argument("--raw", required=True, help="Path to .svs slide")
    ap.add_argument("--out", required=True, help="Output path for mask (.png/.tif)")
    ap.add_argument("--level", type=int, default=0, help="OpenSlide level to use (default 0 for full resolution)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite out file if it exists")
    args = ap.parse_args()

    if Path(args.out).exists() and not args.overwrite:
        print(f"⚠️ Output exists, skipping (use --overwrite to force): {args.out}")
        return
    generate_tissue_mask(args.raw, args.out, level=args.level)


if __name__ == "__main__":
    main()
