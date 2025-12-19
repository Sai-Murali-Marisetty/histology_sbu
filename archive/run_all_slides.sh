#!/bin/bash
set -e  # Exit on error

# === CONFIGURATION ===
RAW_DIR="data/raw"
RESULTS_DIR="results"
RADIUS_UM=100

# Get list of .svs slides
slides=($(ls $RAW_DIR/*.svs))

for slide_path in "${slides[@]}"; do
    SLIDE=$(basename "$slide_path")
    SLIDE_NAME="${SLIDE%.*}"
    SLIDE_OUT="$RESULTS_DIR/$SLIDE_NAME"

    mkdir -p "$SLIDE_OUT/inspect"

    echo "🚀 Processing $SLIDE_NAME"

    # Step 00: Inspect slides and save summary
    python3 src/00_inspect_wsi.py \
        --raw_dir "$RAW_DIR" \
        --out "$SLIDE_OUT/inspect/slide_summary.md"

    # Extract MPP value for this slide
    MPP=$(awk -F'|' -v s="$SLIDE" '$2 ~ s {gsub(/ /,"",$5); print $5}' "$SLIDE_OUT/inspect/slide_summary.md")

    # Step 00b: Generate quick preview (thumbnail + crop)
    python3 src/00_quick_preview_enhanced.py \
        --raw "$slide_path" \
        --out_dir "$SLIDE_OUT/preview" \
        --crop_size 1024

    # Step 01: Tissue Mask
    python3 src/01_tissue_mask.py \
        --raw "$slide_path" \
        --out "$SLIDE_OUT/mask/mask_level2.png" \
        --level 2 --overwrite

    # Step 02: Tile and Normalize
    python3 src/02_tile_and_stain_norm.py \
        --raw "$slide_path" \
        --tissue_mask "$SLIDE_OUT/mask/mask_level2.png" \
        --out_dir "$SLIDE_OUT/tiles" \
        --tile_size 1024 \
        --overlap 128 \
        --level 0

    # Step 03: Segment nuclei
    python3 src/03_segment_and_merge_cellpose.py \
        --tiles_dir "$SLIDE_OUT/tiles" \
        --tiles_json "$SLIDE_OUT/tiles/tiles.json" \
        --masks_dir "$SLIDE_OUT/seg" \
        --out_csv "$SLIDE_OUT/features/${SLIDE_NAME}_nuclei_features.csv" \
        --slide_id "$SLIDE_NAME" \
        --mpp "$MPP" \
        --diam_um 10 \
        --batch_size 4 \
        --gpu \
        --dedup_radius_um 6

    # Step 04: Density profiling
    python3 src/04_density.py \
        --input_csv "$SLIDE_OUT/features/${SLIDE_NAME}_nuclei_features.csv" \
        --output_csv "$SLIDE_OUT/features/${SLIDE_NAME}_nuclei_features_with_density.csv" \
        --mpp "$MPP" \
        --radii_um "$RADIUS_UM"

    # Step 05: Feature enrichment & visualization
    python3 src/05_enrich_and_visualize_features.py \
        --input_csv "$SLIDE_OUT/features/${SLIDE_NAME}_nuclei_features_with_density.csv" \
        --output_csv "$SLIDE_OUT/features/${SLIDE_NAME}_nuclei_features_enriched.csv" \
        --mpp "$MPP" \
        --features area_px circularity mean_gray aspect_ratio \
        --radii_um "$RADIUS_UM" \
        --thumb "$SLIDE_OUT/preview/${SLIDE_NAME}_thumb.jpg" \
        --out_dir "$SLIDE_OUT/viz/features"

    # Step 06: QC overlays + per-slide summary
    python3 src/06_qc_panel_and_summary.py \
        --mode per_slide \
        --csv "$SLIDE_OUT/features/${SLIDE_NAME}_nuclei_features_enriched.csv" \
        --thumb "$SLIDE_OUT/preview/${SLIDE_NAME}_thumb.jpg" \
        --out_dir "$SLIDE_OUT/qc" \
        --mpp "$MPP" \
        --slide_id "$SLIDE_NAME" \
        --features circularity aspect_ratio mean_gray \
        --radii_um "$RADIUS_UM"

    # 🧹 Free caches between slides to avoid OOM
    sync; echo 3 > /proc/sys/vm/drop_caches || true
    sleep 10
done

# Step 07: Merge QC summaries across all slides
python3 src/06_qc_panel_and_summary.py \
    --mode batch_merge \
    --root_dir "$RESULTS_DIR" \
    --out_csv "$RESULTS_DIR/qc_summary.csv"
