import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.io import imread
from PIL import Image
import matplotlib.pyplot as plt
import json
import gc

# Increase PIL decompression bomb limit for large whole slide images
Image.MAX_IMAGE_PIXELS = None  # Remove limit for trusted medical images


def compute_density(x_um, y_um, radii_um):
    from scipy.spatial import cKDTree

    coords = np.vstack([x_um, y_um]).T
    tree = cKDTree(coords)

    density_dict = {}
    for r in radii_um:
        r_sq = np.pi * (r ** 2)
        counts = np.array([len(tree.query_ball_point(p, r)) - 1 for p in coords])
        density = counts / r_sq  # per µm²
        density_dict[f"density_um2_r{r}"] = density
    return density_dict


def mask_correction(x_px, y_px, mask_img, radius_px):
    mask = mask_img > 0
    h, w = mask.shape
    mask_area = np.zeros(len(x_px))

    for i, (x, y) in enumerate(zip(x_px, y_px)):
        x, y = int(round(x)), int(round(y))
        x0 = max(x - radius_px, 0)
        x1 = min(x + radius_px, w)
        y0 = max(y - radius_px, 0)
        y1 = min(y + radius_px, h)

        roi = mask[y0:y1, x0:x1]
        mask_area[i] = roi.sum()
    return mask_area


def scatter_overlay(x, y, vals, thumb_path, out_path, title="", vmin=None, vmax=None):
    if not Path(thumb_path).exists():
        print(f"⚠️ Thumbnail not found: {thumb_path}")
        return

    img = np.array(Image.open(thumb_path).convert("RGB"))
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img)

    vmin = vmin or np.percentile(vals, 2)
    vmax = vmax or np.percentile(vals, 98)

    sc = ax.scatter(x, y, c=vals, cmap="plasma", s=10, edgecolor="none", vmin=vmin, vmax=vmax)
    ax.axis("off")
    plt.title(title, fontsize=14)
    plt.colorbar(sc, shrink=0.5)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"🖼️ Saved: {out_path}")


def scatter_panel(images, out_path):
    if not images:
        return
    loaded = [Image.open(img) for img in images if Path(img).exists()]
    if not loaded:
        return

    widths, heights = zip(*(img.size for img in loaded))
    panel = Image.new("RGB", (sum(widths), max(heights)))
    x_offset = 0
    for img in loaded:
        panel.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    panel.save(out_path)
    print(f"📊 QC panel saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--mpp", type=float, required=True)
    ap.add_argument("--radii_um", nargs="+", type=float, default=[50, 100])
    ap.add_argument("--tissue_mask", required=True)
    ap.add_argument("--mask_level", type=int, default=0, help="Mask pyramid level (default 0 for full resolution)")
    ap.add_argument("--downsamples", nargs="+", type=float, required=True)
    ap.add_argument("--thumb", required=True)
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--percentile_clip", nargs=2, type=float, default=[2, 98])
    args = ap.parse_args()

    # === Load data
    df = pd.read_csv(args.input_csv)
    x_um = df["x_um"].to_numpy()
    y_um = df["y_um"].to_numpy()

    # === Compute densities
    print("📈 Computing densities...")
    density_dict = compute_density(x_um, y_um, args.radii_um)
    for k, v in density_dict.items():
        df[k] = v

    # === Load tissue mask
    print("🧬 Loading tissue mask...")
    mask_img = imread(args.tissue_mask)
    down = args.downsamples[args.mask_level]
    x_px = (df["x"] / down).to_numpy()
    y_px = (df["y"] / down).to_numpy()

    # === Apply tissue correction
    for r in args.radii_um:
        r_px = r / (args.mpp * down)
        mask_area = mask_correction(x_px, y_px, mask_img, radius_px=int(round(r_px)))
        key = f"corrected_density_um2_r{r}"
        df[key] = df[f"density_um2_r{r}"] * (np.pi * r_px ** 2) / np.maximum(mask_area, 1e-6)

    # === Save enriched CSV
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"💾 Saved: {args.output_csv}")

    # === Save visual overlays
    base_dir = Path(args.output_csv).parent
    preview_images = []
    for r in args.radii_um:
        col = f"corrected_density_um2_r{r}"
        out_img = base_dir / f"density_overlay_r{r}.jpg"
        scatter_overlay(
            df["x"], df["y"], df[col],
            thumb_path=args.thumb,
            out_path=out_img,
            title=f"Corrected Density r={r} µm",
            vmin=np.percentile(df[col], args.percentile_clip[0]),
            vmax=np.percentile(df[col], args.percentile_clip[1])
        )
        preview_images.append(out_img)

    # === QC Panel
    scatter_panel(preview_images, base_dir / f"panel_density_{Path(args.input_csv).stem}.jpg")

    # === Summary JSON
    summary = {
        "total_cells": int(len(df)),
        "mean_x": float(df["x_um"].mean()),
        "mean_y": float(df["y_um"].mean()),
        "radii_um": args.radii_um
    }
    for r in args.radii_um:
        key = f"corrected_density_um2_r{r}"
        vals = df[key]
        summary[f"{key}_P10"] = float(np.percentile(vals, 10))
        summary[f"{key}_median"] = float(np.percentile(vals, 50))
        summary[f"{key}_P90"] = float(np.percentile(vals, 90))

    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"📄 Saved summary: {args.summary_json}")

    # === Cleanup
    del df
    gc.collect()


if __name__ == "__main__":
    main()
