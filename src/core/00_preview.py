#!/usr/bin/env python3
# 00_quick_preview_enhanced.py — Smart Slide Preview Generator + Panel

import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import openslide

# Increase PIL decompression bomb limit for large whole slide images
Image.MAX_IMAGE_PIXELS = None


def add_watermark(image, text, position=(10, 10)):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    x, y = position
    for dx, dy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        draw.text((x+dx, y+dy), text, font=font, fill=(0,0,0))
    draw.text((x, y), text, font=font, fill=(255,255,255))
    return image


def detect_tissue_centroid(slide):
    """Return ((cx, cy), thumb_img, tissue_mask_img, contour_list) in level-0 coords"""
    level0_w, level0_h = slide.dimensions
    max_dim = 1024
    scale = max(level0_w, level0_h) / max_dim if max(level0_w, level0_h) > max_dim else 1.0
    thumb_w = int(round(level0_w / scale))
    thumb_h = int(round(level0_h / scale))
    thumb = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
    thumb_np = np.array(thumb)

    hsv = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)
    tissue = ((S > 20) & (V > 40)).astype(np.uint8) * 255
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)

    cnts, _ = cv2.findContours(tissue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, thumb, Image.fromarray(tissue), []

    largest = max(cnts, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, thumb, Image.fromarray(tissue), cnts

    cx_thumb = int(M["m10"] / M["m00"])
    cy_thumb = int(M["m01"] / M["m00"])
    cx = int(round(cx_thumb * scale))
    cy = int(round(cy_thumb * scale))
    return (cx, cy), thumb, Image.fromarray(tissue), cnts


def combine_panel(thumb, tissue_mask, crop, out_path):
    thumb = thumb.resize((512, 512))
    mask = tissue_mask.resize((512, 512))
    crop = crop.resize((512, 512))

    panel = Image.new("RGB", (3 * 512, 512))
    panel.paste(thumb, (0, 0))
    panel.paste(mask.convert("RGB"), (512, 0))
    panel.paste(crop, (1024, 0))
    panel.save(out_path, quality=95)
    print(f"🧩 Saved preview panel: {out_path}")


def quick_preview(slide_path, out_dir, crop_size=1024):
    slide = openslide.OpenSlide(str(slide_path))
    slide_id = Path(slide_path).stem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Thumbnail
    w0, h0 = slide.dimensions
    max_dim = 2048
    scale = max(w0, h0) / max_dim if max(w0, h0) > max_dim else 1.0
    tw, th = int(round(w0 / scale)), int(round(h0 / scale))
    thumb = slide.get_thumbnail((tw, th)).convert("RGB")

    # 2. Tissue detection
    detected, thumb, tissue_mask, contours = detect_tissue_centroid(slide)
    thumb_np = np.array(thumb)
    if contours:
        cv2.drawContours(thumb_np, contours, -1, (255, 0, 0), 3)
    thumb = Image.fromarray(thumb_np)
    thumb = add_watermark(thumb, slide_id)
    thumb.save(out_dir / f"{slide_id}_thumb.jpg", quality=95)
    print(f"🖼️ Saved thumbnail: {out_dir / f'{slide_id}_thumb.jpg'}")

    # 3. Tissue mask
    tissue_mask.save(out_dir / f"{slide_id}_tissue_mask.png")
    print(f"🧫 Saved tissue mask: {out_dir / f'{slide_id}_tissue_mask.png'}")

    # 4. Crop
    cx, cy = detected if detected else (w0 // 2, h0 // 2)
    half = crop_size // 2
    x0 = max(0, min(w0 - crop_size, cx - half))
    y0 = max(0, min(h0 - crop_size, cy - half))

    crop = slide.read_region((x0, y0), 0, (crop_size, crop_size)).convert("RGB")
    crop = add_watermark(crop, slide_id)
    crop.save(out_dir / f"{slide_id}_crop.png")
    print(f"🧪 Saved crop: {out_dir / f'{slide_id}_crop.png'}")

    # 5. Combined panel
    combine_panel(thumb, tissue_mask, crop, out_dir / f"panel_{slide_id}_preview.png")


def main():
    ap = argparse.ArgumentParser(description="Generate preview images from a WSI.")
    ap.add_argument("--raw", required=True, help="Path to slide")
    ap.add_argument("--out_dir", required=True, help="Output folder")
    ap.add_argument("--crop_size", type=int, default=1024)
    args = ap.parse_args()
    quick_preview(args.raw, args.out_dir, args.crop_size)

if __name__ == "__main__":
    main()
