import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
import numpy as np

# Increase PIL decompression bomb limit for large whole slide images
Image.MAX_IMAGE_PIXELS = None

def draw_overlay(df, col, out_path, background, scale=1.0, title=None):
    if not Path(background).exists():
        print(f"⚠️ Background not found: {background}")
        return
    img = np.array(Image.open(background).convert("RGB"))
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img)
    vmin = df[col].quantile(0.02)
    vmax = df[col].quantile(0.98)
    sc = ax.scatter(df["x"] / scale, df["y"] / scale, c=df[col],
                 cmap="plasma", s=4, edgecolor="none",
                 vmin=vmin, vmax=vmax)
    ax.set_title(title or col)
    ax.axis("off")
    plt.colorbar(sc, ax=ax, shrink=0.5)
    
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"🖼️ Saved overlay: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--thumb", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    summary = {"slide": Path(args.input_csv).stem, "n_nuclei": int(len(df))}
    for col in df.columns:
        if col.startswith("corrected_density") or "local" in col:
            out_img = Path(args.output_dir) / f"{col}_overlay.jpg"
            draw_overlay(df, col, out_img, args.thumb, scale=args.scale)
            summary[f"{col}_median"] = float(df[col].median())
            summary[f"{col}_mean"] = float(df[col].mean())

    with open(Path(args.output_dir) / "qc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"📄 Saved summary: {args.output_dir}/qc_summary.json")


if __name__ == "__main__":
    main()
