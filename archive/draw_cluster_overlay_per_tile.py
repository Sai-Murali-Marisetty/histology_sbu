import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image

# === CONFIG ===
CSV_PATH = Path("results/CD3-S25/comparison/stardist_clustered_features.csv")
TILE_DIR = Path("results/CD3-S25/tiles")
OUT_DIR = Path("results/CD3-S25/comparison/overlays")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === Load CSV ===
df = pd.read_csv(CSV_PATH)

# === Get unique tiles ===
for tile_name in df["tile"].unique():
    tile_path = TILE_DIR / tile_name
    if not tile_path.exists():
        print(f"❌ Missing tile: {tile_name}")
        continue

    # Load image
    img = np.array(Image.open(tile_path).convert("RGB"))

    # Subset of nuclei from this tile
    sub_df = df[df["tile"] == tile_name]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    scatter = ax.scatter(
        sub_df["x"],
        sub_df["y"],
        c=sub_df["cluster"],
        cmap="tab20",
        s=10,
        edgecolor="none"
    )
    ax.set_title(f"Cluster Overlay - {tile_name}")
    ax.axis("off")
    plt.colorbar(scatter, ax=ax, shrink=0.5, label="Cluster")

    # Save
    out_path = OUT_DIR / tile_name.replace(".png", "_overlay.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved overlay: {out_path}")
