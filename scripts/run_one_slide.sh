#!/bin/bash
set -euo pipefail

# Usage:
#   ./run_one_slide.sh path/to/slide.svs [RESULTS_DIR=results] [NUC_DIAM_UM=10] [DENSITY_RADII_UM="50 100 150"]
# Notes:
#   - Assumes python scripts are in ./src/ (relative to this file). Override via SRC_DIR=/abs/path ./run_one_slide.sh ...

RAW_SLIDE="${1:-}"
RESULTS_DIR="${2:-results}"
NUC_DIAM_UM="${3:-10}"
DENSITY_RADII_UM="${4:-50 100 150}"

if [[ -z "$RAW_SLIDE" ]]; then
  echo "❌ Usage: $0 path/to/slide.svs [results_dir] [nucleus_diameter_um] [\"r1 r2 ...\"]"
  exit 1
fi
if [[ ! -f "$RAW_SLIDE" ]]; then
  echo "❌ Slide not found: $RAW_SLIDE"
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SRC_DIR="${SRC_DIR:-src}"   # scripts live in src/ already

mkdir -p logs results

# Output layout
SLIDE_BASENAME="$(basename "$RAW_SLIDE")"
SLIDE_NAME="${SLIDE_BASENAME%.*}"
SLIDE_OUT="$RESULTS_DIR/$SLIDE_NAME"

PREVIEW_DIR="$SLIDE_OUT/preview"
MASKS_DIR="$SLIDE_OUT/masks"
TILES_DIR="$SLIDE_OUT/tiles"
CELLPOSE_DIR="$SLIDE_OUT/cellpose"
FEATURES_DIR="$SLIDE_OUT/features"
VIZ_DIR="$SLIDE_OUT/viz"
QC_DIR="$SLIDE_OUT/qc"
mkdir -p "$PREVIEW_DIR" "$MASKS_DIR" "$TILES_DIR" "$CELLPOSE_DIR" "$FEATURES_DIR" "$VIZ_DIR" "$QC_DIR"

# ---- Read MPP and downsamples safely (no argv issues) ----
SLIDE_PATH="$RAW_SLIDE"
read -r MPP DOWNSAMPLES <<< "$(
SLIDE_PATH="$SLIDE_PATH" python3 - <<'PY'
import os, openslide
p = os.environ['SLIDE_PATH']
sl = openslide.OpenSlide(p)
props = sl.properties
mpp = None
for k in ("openslide.mpp-x", "openslide.mpp-y"):
    v = props.get(k)
    try:
        if v is not None:
            mpp = float(v); break
    except Exception:
        pass
if mpp is None:
    mpp = 0.25
downs = [float(d) for d in sl.level_downsamples]
print(mpp, " ".join(str(d) for d in downs))
PY
)"
echo "ℹ️  MPP: $MPP"
echo "ℹ️  Level downsamples: $DOWNSAMPLES"

# Array forms
read -r -a RADII_ARR <<< "$DENSITY_RADII_UM"
read -r -a DOWNS_ARR <<< "$DOWNSAMPLES"

# Common paths
THUMB_PATH="$PREVIEW_DIR/${SLIDE_NAME}_thumb.jpg"
TISSUE_MASK_PATH="$MASKS_DIR/${SLIDE_NAME}_tissue_mask.png"
TILES_JSON="$TILES_DIR/tiles.json"
CELLPOSE_MASKS_DIR="$CELLPOSE_DIR/masks"
RAW_FEATURES_CSV="$FEATURES_DIR/${SLIDE_NAME}_nuclei_features.csv"
DENSITY_CSV="$FEATURES_DIR/${SLIDE_NAME}_nuclei_features_density.csv"
SUMMARY_JSON="$FEATURES_DIR/${SLIDE_NAME}_summary.json"
ENRICHED_FEATURES_CSV="$FEATURES_DIR/${SLIDE_NAME}_nuclei_features_enriched.csv"

# 00 preview
python3 "$SRC_DIR/core/00_preview.py" \
  --raw "$RAW_SLIDE" --out_dir "$PREVIEW_DIR" --crop_size 1024

# 01 tissue mask
python3 "$SRC_DIR/core/01_tissue_mask.py" \
  --raw "$RAW_SLIDE" --out "$TISSUE_MASK_PATH" --level 2 --overwrite

# 02 tiling (+ optional stain norm inside)
python3 "$SRC_DIR/core/02_tile.py" \
  --raw "$RAW_SLIDE" --tissue_mask "$TISSUE_MASK_PATH" \
  --out_dir "$TILES_DIR" --tile_size 1024 --overlap 128 --level 0

# 03 cellpose seg + merge
mkdir -p "$CELLPOSE_MASKS_DIR"
SEG_ARGS=(
  --tiles_dir "$TILES_DIR"
  --tiles_json "$TILES_JSON"
  --masks_dir "$CELLPOSE_MASKS_DIR"
  --out_csv "$RAW_FEATURES_CSV"
  --slide_id "$SLIDE_NAME"
  --mpp "$MPP"
  --diam_um "$NUC_DIAM_UM"
  --batch_size 8
  --gpu
)
python3 "$SRC_DIR/core/03_segment_cellpose.py" "${SEG_ARGS[@]}"

# 04 density + summary
python3 "$SRC_DIR/core/04_density.py" \
  --input_csv "$RAW_FEATURES_CSV" --output_csv "$DENSITY_CSV" \
  --mpp "$MPP" --radii_um "${RADII_ARR[@]}" \
  --tissue_mask "$TISSUE_MASK_PATH" --mask_level 2 \
  --downsamples "${DOWNS_ARR[@]}" \
  --thumb "$THUMB_PATH" --summary_json "$SUMMARY_JSON" \
  --percentile_clip 2 98

# 05 enrich + visualizations
python3 "$SRC_DIR/core/05_features.py" \
  --input_csv "$DENSITY_CSV" --out_csv "$ENRICHED_FEATURES_CSV" \
  --out_dir "$VIZ_DIR" --thumb "$THUMB_PATH" \
  --radii_um "${RADII_ARR[@]}" --features circularity gray_mean

# 06 QC
python3 "$SRC_DIR/core/06_qc.py" \
  --input_csv "$ENRICHED_FEATURES_CSV" --output_dir "$QC_DIR" \
  --thumb "$THUMB_PATH" --scale 1.0

echo "✅ Finished $SLIDE_NAME"
echo "📂 Outputs in: $SLIDE_OUT"
