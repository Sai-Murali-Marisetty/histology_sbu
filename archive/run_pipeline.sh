#!/bin/bash
# Usage: ./run_pipeline.sh CD3-S25

set -e  # stop on error

if [ -z "$1" ]; then
  echo "❌ Please provide a slide ID (e.g., ./run_pipeline.sh CD3-S25)"
  exit 1
fi

SLIDE="$1"
RAW="data/raw/${SLIDE}.svs"
RESULTS="results/$SLIDE"

MMP=0.262697
MASK_LEVEL=2
DOWNSAMPLES="1.0 4.0 16.0 64.0"
RADII="50 100"
FEATURES="circularity gray_mean"
CROP_SIZE=1024

echo "🔁 Running full pipeline 00 → 06 for $SLIDE..."

# Step 00 — Quick preview
python3 src/00_quick_preview_enhanced.py \
  --raw "$RAW" \
  --out_dir "$RESULTS/00_preview" \
  --crop_size $CROP_SIZE

# Step 01 — Tissue mask
python3 src/01_tissue_mask.py \
  --raw "$RAW" \
  --out "$RESULTS/01_mask/tissue_mask.png" \
  --level $MASK_LEVEL

# Step 02 — Tile and stain normalization
python3 src/02_tile_and_stain_norm.py \
  --raw "$RAW" \
  --tissue_mask "$RESULTS/01_mask/tissue_mask.png" \
  --out_dir "$RESULTS/02_tiles" \
  --tile_size $CROP_SIZE \
  --overlap 128 \
  --level 0

# Step 03 — Segment with Cellpose
python3 src/03_segment_and_merge_cellpose.py \
  --tiles_dir "$RESULTS/02_tiles" \
  --tiles_json "$RESULTS/02_tiles/tiles.json" \
  --masks_dir "$RESULTS/03_segment/masks" \
  --out_csv "$RESULTS/03_segment/nuclei_features.csv" \
  --slide_id "$SLIDE" \
  --mpp $MMP \
  --diam_um 10 \
  --dedup_radius_um 6 \
  --gpu \
  --batch_size 16

# Step 04 — Density analysis
python3 src/04_density.py \
  --input_csv "$RESULTS/03_segment/nuclei_features.csv" \
  --output_csv "$RESULTS/04_density/nuclei_with_density.csv" \
  --mpp $MMP \
  --radii_um $RADII \
  --tissue_mask "$RESULTS/01_mask/tissue_mask.png" \
  --mask_level $MASK_LEVEL \
  --downsamples $DOWNSAMPLES \
  --thumb "$RESULTS/00_preview/${SLIDE}_thumb.jpg" \
  --summary_json "$RESULTS/04_density/density_summary.json"

# Step 05 — Feature enrichment
python3 src/05_enrich_and_visualize_features.py \
  --input_csv "$RESULTS/04_density/nuclei_with_density.csv" \
  --out_csv "$RESULTS/05_features/features_enriched.csv" \
  --out_dir "$RESULTS/05_features" \
  --thumb "$RESULTS/00_preview/${SLIDE}_thumb.jpg" \
  --radii_um $RADII \
  --features $FEATURES

# Step 06 — QC Panel
python3 src/06_qc_panel_and_summary.py \
  --input_csv "$RESULTS/05_features/features_enriched.csv" \
  --output_dir "$RESULTS/06_qc" \
  --thumb "$RESULTS/00_preview/${SLIDE}_thumb.jpg" \
  --scale 1.0

echo "🎉 Completed pipeline 00 → 06 for $SLIDE"
