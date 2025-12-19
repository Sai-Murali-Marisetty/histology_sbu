# src/00_inspect_wsi.py

import os
import argparse
import openslide
from pathlib import Path

def inspect_slide(slide_path):
    slide = openslide.OpenSlide(str(slide_path))
    properties = slide.properties
    dimensions = slide.dimensions

    try:
        mpp_x = float(properties.get("openslide.mpp-x", -1))
        mpp_y = float(properties.get("openslide.mpp-y", -1))
        mpp = (mpp_x + mpp_y) / 2 if mpp_x > 0 and mpp_y > 0 else "Not found"
    except:
        mpp = "Not found"

    levels = slide.level_count
    downsample_levels = [slide.level_downsamples[i] for i in range(levels)]

    return {
        "filename": slide_path.name,
        "width": dimensions[0],
        "height": dimensions[1],
        "mpp": mpp,
        "levels": levels,
        "downsamples": downsample_levels
    }

def main(raw_dir, out_path):
    raw_dir = Path(raw_dir)
    out_path = Path(out_path)

    # ✅ Ensure parent directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    slides = sorted(raw_dir.glob("*.svs"))

    rows = []
    for slide_path in slides:
        info = inspect_slide(slide_path)
        rows.append(info)

    with open(out_path, "w") as f:
        f.write("| Slide | Width | Height | MPP (µm/px) | Levels | Downsamples |\n")
        f.write("|-------|--------|--------|--------------|--------|-------------|\n")
        for row in rows:
            f.write(f"| {row['filename']} | {row['width']} | {row['height']} | {row['mpp']} | {row['levels']} | {row['downsamples']} |\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True, help="Directory with raw .svs slides")
    parser.add_argument("--out", required=True, help="Markdown output summary")
    args = parser.parse_args()
    main(args.raw_dir, args.out)
