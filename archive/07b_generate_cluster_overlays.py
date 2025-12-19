"""
07b_generate_cluster_overlays.py

Overlay nuclei cluster labels on each tile and save color-coded visualization.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from skimage.io import imread
import seaborn as sns

# === Paths ===
CSV_PATH = Path("results/CD3-S25/comparison/stardist_clustered_features.csv")
TILE_DIR = Path("results/CD3-S25/tiles")
OUT_DIR = Path("results/CD3-S25/comparison/overlays")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === Load Data ===
df = pd.read_csv(CSV_PATH)

# === Cluster colormap ===
clusters = sorted(df["cluster"].unique())
palette = sns.color_palette("tab20", len(clusters))
cluster_color_map = {c: palette[i % len(palette)] for i, c in enumerate(clusters)}

# === Generate overlay per tile ===
tiles = df["tile"].unique()
for tile_name in tiles:
    tile_path = TILE_DIR / tile_name
    if not tile_path.exists():
        print(f"⚠️ Tile missing: {tile_path}")
        continue

    img = imread(tile_path)
    tile_df = df[df["tile"] == tile_name]

    plt.figure(figsize=(8, 8))
    plt.imshow(img)

    for cluster_id in clusters:
        cluster_df = tile_df[tile_df["cluster"] == cluster_id]
        plt.scatter(
            cluster_df["x"],
            cluster_df["y"],
            s=20,
            color=cluster_color_map[cluster_id],
            label=str(cluster_id),
            alpha=0.7,
            edgecolor='black',
            linewidth=0.3
        )

    # Legend
    handles = [mpatches.Patch(color=cluster_color_map[c], label=f"Cluster {c}") for c in clusters]
    plt.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)
    plt.title(f"Cluster Overlay - {tile_name}")
    plt.axis("off")

    out_path = OUT_DIR / tile_name.replace(".png", "_overlay.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved overlay: {out_path}")
