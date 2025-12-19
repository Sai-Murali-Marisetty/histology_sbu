#!/bin/bash
set -e

RAW=data/raw
RESULTS=results

# Fixed parameters
MMP=0.262697
MASK_LEVEL=2
DOWNSAMPLES="1.0 4.0 16.0 64.0"
RADII="50 100"
FEATURES="circularity gray_mean"
CROP_SIZE=1024

for SLIDE_PATH in ${RAW}/*.svs; do
  SLIDE_FILE=$(basename "$SLIDE_PATH")
  SLIDE="${SLIDE_FILE%.*}"
  echo "🔁 Processing $SLIDE..."

  OUT_DIR="${RESULTS}/${SLIDE}"
  PREVIEW_DIR="${OUT_DIR}/00_preview"
  MASK_DIR="${OUT_DIR}/01_mask"
  TILE_DIR="${OUT_DIR}/02_tiles"
  SEG_DIR="${OUT_DIR}/03_segment"
  DENSITY_DIR="${OUT_DIR}/04_density"
  FEATURES_DIR="${OUT_DIR}/05_features"
  QC_DIR="${OUT_DIR}/06_qc"

  ### Step 00 — Quick Preview
  python3 src/00_quick_preview_enhanced.py \
    --raw "$SLIDE_PATH" \
    --out_dir "$PREVIEW_DIR" \
    --crop_size "$CROP_SIZE"

  ### Step 01 — Tissue Mask
  python3 src/01_tissue_mask.py \
      --raw "$SLIDE_PATH" \
      --out "$MASK_DIR/tissue_mask.png" \
      --level "$MASK_LEVEL"
    

  ### Step 02 — Tile + Stain Norm
  python3 src/02_tile_and_stain_norm.py \
    --input "$SLIDE_PATH" \
    --mask "$MASK_DIR/tissue_mask.png" \
    --out_dir "$TILE_DIR" \
    --level "$MASK_LEVEL" \
    --slide_id "$SLIDE"

  ### Step 03 — Segment + Merge
  python3 src/03_segment_and_merge_cellpose.py \
    --tile_dir "$TILE_DIR" \
    --out_dir "$SEG_DIR" \
    --diameter 30 \
    --mpp "$MMP" \
    --tile_json "$TILE_DIR/tiles.json" \
    --model "cyto2"

  ### Step 04 — Density Estimation
  python3 src/04_density.py \
    --input_csv "$SEG_DIR/nuclei_features.csv" \
    --output_csv "$DENSITY_DIR/nuclei_with_density.csv" \
    --mpp "$MMP" \
    --radii_um $RADII \
    --tissue_mask "$MASK_DIR/tissue_mask.png" \
    --mask_level "$MASK_LEVEL" \
    --downsamples $DOWNSAMPLES \
    --thumb "$PREVIEW_DIR/${SLIDE}_thumb.jpg" \
    --summary_json "$DENSITY_DIR/density_summary.json"

  ### Step 05 — Feature Enrichment + Overlays
  python3 src/05_enrich_and_visualize_features.py \
    --input_csv "$DENSITY_DIR/nuclei_with_density.csv" \
    --out_csv "$FEATURES_DIR/features_enriched.csv" \
    --out_dir "$FEATURES_DIR" \
    --thumb "$PREVIEW_DIR/${SLIDE}_thumb.jpg" \
    --radii_um $RADII \
    --features $FEATURES

  ### Step 06 — QC Panel + Summary
  python3 src/06_qc_panel_and_summary.py \
    --input_csv "$FEATURES_DIR/features_enriched.csv" \
    --output_dir "$QC_DIR" \
    --thumb "$PREVIEW_DIR/${SLIDE}_thumb.jpg" \
    --scale 1.0

  echo "✅ Done with $SLIDE"
done

echo "🎉 ALL SLIDES COMPLETED!"
