#!/bin/bash

set -e  # stop on first error
set -o pipefail

RAW_DIR="data/raw"
RESULTS_DIR="results"
MMP=0.262697
TILE_SIZE=1024
OVERLAP=128
LEVEL=0
DIAM_UM=10.0
DEDUP_RADIUS=6.0
USE_GPU="--gpu"  # remove this if no GPU

echo "🔁 Starting batch run 00 → 03..."

for SLIDE_PATH in ${RAW_DIR}/*.svs; do
    SLIDE_FILE=$(basename "$SLIDE_PATH")
    SLIDE_ID="${SLIDE_FILE%.*}"

    echo "----------------------------------------"
    echo "🧪 Processing: $SLIDE_ID"
    echo "----------------------------------------"

    OUT_PREVIEW="${RESULTS_DIR}/${SLIDE_ID}/00_preview"
    OUT_MASK="${RESULTS_DIR}/${SLIDE_ID}/01_mask"
    OUT_TILES="${RESULTS_DIR}/${SLIDE_ID}/02_tiles"
    OUT_SEG="${RESULTS_DIR}/${SLIDE_ID}/03_segment"

    #### 00: Quick preview
    python3 src/core/00_quick_preview_enhanced.py \
        --raw "$SLIDE_PATH" \
        --out_dir "$OUT_PREVIEW" \
        --crop_size $TILE_SIZE

    #### 01: Tissue mask (low-res)
    mkdir -p "$OUT_MASK"
    python3 src/core/01_tissue_mask.py \
        --raw "$SLIDE_PATH" \
        --out "${OUT_MASK}/tissue_mask.png" \
        --level 2

    #### 02: Tile based on mask
    python3 src/core/02_tile_and_stain_norm.py \
        --raw "$SLIDE_PATH" \
        --tissue_mask "${OUT_MASK}/tissue_mask.png" \
        --out_dir "$OUT_TILES" \
        --tile_size $TILE_SIZE \
        --overlap $OVERLAP \
        --level $LEVEL

    #### 03: Segment with Cellpose and deduplicate
    python3 src/core/03_segment_and_merge_cellpose.py \
        --tiles_dir "$OUT_TILES" \
        --tiles_json "$OUT_TILES/tiles.json" \
        --masks_dir "$OUT_SEG/masks" \
        --out_csv "$OUT_SEG/nuclei_features.csv" \
        --slide_id "$SLIDE_ID" \
        --mpp $MMP \
        --diam_um $DIAM_UM \
        --dedup_radius_um $DEDUP_RADIUS \
        $USE_GPU

    echo "✅ Done: $SLIDE_ID"
done

echo "🎉 All slides processed."
