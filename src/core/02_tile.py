# src/02_tile_and_stain_norm.py

import os
import argparse
import json
import openslide
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from tqdm import tqdm

# Increase PIL decompression bomb limit for large whole slide images
Image.MAX_IMAGE_PIXELS = None  # Remove limit entirely (safe for trusted medical images)


def extract_tiles(slide_path, mask_path, out_dir, tile_size=1024, overlap=128, level=0):
    slide = openslide.OpenSlide(slide_path)

    # Load tissue mask and get its dimensions
    tissue_mask_img = Image.open(mask_path).convert("L")
    tissue_mask = np.array(tissue_mask_img) > 0
    mask_h, mask_w = tissue_mask.shape

    # Get slide dimensions at specified level
    slide_w, slide_h = slide.level_dimensions[level]
    
    # CRITICAL FIX: Get the correct downsample factor for this level
    downsample = slide.level_downsamples[level]

    # Scale between slide coordinates at level 0 and tissue mask
    # Mask is created from a downsampled level, need to map correctly
    scale_x = mask_w / slide.level_dimensions[0][0]
    scale_y = mask_h / slide.level_dimensions[0][1]

    step = tile_size - overlap
    tiles = []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tile_count = 0
    skipped = 0

    print(f"Slide dimensions at level {level}: {slide_w} x {slide_h}")
    print(f"Downsample factor: {downsample}")
    print(f"Tissue mask size: {mask_w} x {mask_h}")
    print(f"Tile size: {tile_size}, Overlap: {overlap}, Step: {step}")

    for y in tqdm(range(0, slide_h - tile_size + 1, step), desc="Tiling"):
        for x in range(0, slide_w - tile_size + 1, step):
            # CRITICAL FIX: Map tile coordinates to level 0 coordinates
            # Tiles are in level coordinates, need to convert to level 0 for mask mapping
            x_lvl0 = int(x * downsample)
            y_lvl0 = int(y * downsample)

            # Map level-0 coordinates to tissue mask coordinates
            x_mask = int(x_lvl0 * scale_x)
            y_mask = int(y_lvl0 * scale_y)
            
            # Size of tile in mask coordinates
            mask_tile_size = int(tile_size * downsample * scale_x)

            # Crop mask area and check if it's mostly tissue
            mask_crop = tissue_mask[
                y_mask:y_mask + mask_tile_size,
                x_mask:x_mask + mask_tile_size
            ]

            # Skip if mask crop is too small (edge of image)
            if mask_crop.shape[0] < mask_tile_size * 0.5 or mask_crop.shape[1] < mask_tile_size * 0.5:
                skipped += 1
                continue

            # Skip if less than 50% tissue
            if np.mean(mask_crop) < 0.5:
                skipped += 1
                continue

            # Read tile from WSI at target level
            # Note: read_region uses level-0 coordinates even when reading at other levels
            tile_img = slide.read_region((x_lvl0, y_lvl0), level, (tile_size, tile_size)).convert("RGB")
            
            tile_fname = out_dir / f"tile_{x}_{y}.png"
            tile_img.save(tile_fname)

            tiles.append({
                "x": x, 
                "y": y, 
                "w": tile_size, 
                "h": tile_size,
                "level": level,
                "x_lvl0": x_lvl0,
                "y_lvl0": y_lvl0
            })
            tile_count += 1

    # Save tile metadata
    metadata = {
        "tiles": tiles,
        "level": level,
        "downsample": float(downsample),
        "level_dimensions": list(slide.level_dimensions[level]),
        "tile_size": tile_size,
        "overlap": overlap,
        "level_mpp": float(slide.properties.get("openslide.mpp-x", -1))
    }
    
    with open(out_dir / "tiles.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone: {tile_count} tiles saved | {skipped} skipped (background or edge)")
    print(f"Output dir: {out_dir}")
    print(f"Metadata saved: {out_dir / 'tiles.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract tiles from WSI using tissue mask with coordinate mapping fix"
    )
    parser.add_argument("--raw", required=True, help="Path to .svs slide")
    parser.add_argument("--tissue_mask", required=True, help="Path to tissue mask")
    parser.add_argument("--out_dir", required=True, help="Output tile directory")
    parser.add_argument("--tile_size", type=int, default=1024, help="Tile size in pixels")
    parser.add_argument("--overlap", type=int, default=128, help="Overlap between tiles")
    parser.add_argument("--level", type=int, default=0, help="OpenSlide level for tiling (default=0)")
    args = parser.parse_args()

    extract_tiles(args.raw, args.tissue_mask, args.out_dir, 
                  args.tile_size, args.overlap, args.level)
