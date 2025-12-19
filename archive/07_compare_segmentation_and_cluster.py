"""
07_compare_segmentation_and_cluster.py

Full pipeline:
- Runs Stardist (on GPU if available) on all tiles
- Extracts nuclear features (shape, color, density)
- Runs UMAP + BIRCH clustering
- Saves per-nucleus feature CSV + UMAP plot
"""

import numpy as np
import pandas as pd
from pathlib import Path
from skimage.io import imread
from skimage.measure import regionprops
from scipy.spatial import cKDTree
from sklearn.cluster import Birch
import matplotlib.pyplot as plt
import tensorflow as tf

from stardist.models import StarDist2D
from csbdeep.utils import normalize
from umap import UMAP

# === Config ===
TILE_DIR = Path("results/CD3-S25/tiles")
OUT_DIR = Path("results/CD3-S25/comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === Prefer GPU if available ===
device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
print(f"🚀 Using device: {device}")

# === Load Stardist model ===
print("📦 Loading Stardist model...")
with tf.device(device):
    model = StarDist2D.from_pretrained('2D_versatile_he')

# === Load all tiles ===
tile_paths = sorted(TILE_DIR.glob("tile_*.png"))
print(f"🖼️ Processing {len(tile_paths)} tiles")

all_rows = []

# === Process each tile ===
for tile_path in tile_paths:
    print(f"🔍 Segmenting {tile_path.name}")
    img = imread(tile_path)
    img_norm = normalize(img)

    with tf.device(device):
        labels, _ = model.predict_instances(img_norm)

    for prop in regionprops(labels):
        y, x = prop.centroid
        area = prop.area
        perimeter = prop.perimeter
        circ = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
        ecc = prop.eccentricity
        major = prop.major_axis_length
        minor = prop.minor_axis_length
        aspect = major / minor if minor > 0 else 0

        mask = labels == prop.label
        r = img[:, :, 0][mask].mean()
        g = img[:, :, 1][mask].mean()
        b = img[:, :, 2][mask].mean()

        all_rows.append({
            "tile": tile_path.name,
            "x": x,
            "y": y,
            "area": area,
            "perimeter": perimeter,
            "circularity": circ,
            "aspect_ratio": aspect,
            "eccentricity": ecc,
            "r": r,
            "g": g,
            "b": b
        })

# === Assemble DataFrame ===
df = pd.DataFrame(all_rows)
print(f"🧬 Extracted {len(df)} nuclei")

# === Compute local density ===
coords = df[["x", "y"]].values
tree = cKDTree(coords)
df["density_r50"] = [len(tree.query_ball_point(p, 50)) - 1 for p in coords]

# === UMAP + BIRCH Clustering ===
features = df[["area", "aspect_ratio", "circularity", "density_r50", "r", "g", "b"]]
print("📉 Running UMAP...")
embedding = UMAP(n_components=2, random_state=42).fit_transform(features)
df["umap_x"] = embedding[:, 0]
df["umap_y"] = embedding[:, 1]

print("🔗 Running BIRCH clustering...")
#birch = Birch(n_clusters=None, threshold=0.5)
#birch = Birch(n_clusters=30)  # ✅ updated from None
birch = Birch(n_clusters=30, threshold=0.5)


df["cluster"] = birch.fit_predict(embedding)

# === Save results ===
df_out_path = OUT_DIR / "stardist_clustered_features.csv"
df.to_csv(df_out_path, index=False)
print(f"✅ Saved CSV: {df_out_path}")

# === Plot UMAP ===
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df["umap_x"], df["umap_y"], c=df["cluster"], cmap="tab10", s=10)
plt.colorbar(scatter, label="Cluster")
plt.title("Stardist UMAP + BIRCH Clustering")
plot_path = OUT_DIR / "stardist_umap_clusters.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"🖼️ Saved plot: {plot_path}")
