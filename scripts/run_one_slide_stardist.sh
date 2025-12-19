#!/bin/bash
set -euo pipefail

# Usage:
#   src/run_one_slide_stardist.sh path/to/slide.svs [RESULTS_DIR=results] [NUC_DIAM_UM=10] ["50 100 150"]
# Notes:
#   - Mirrors your existing run_one_slide.sh, but step 03 uses StarDist.
#   - Expects Python scripts already in src/: 00_quick_preview_enhanced.py, 01_tissue_mask.py,
#     02_tile_and_stain_norm.py, 03_segment_and_merge_stardist.py, 04_density.py, 05_enrich_and_visualize_features.py,
#     06_qc_panel_and_summary.py (and optional 07*, 07b* if you have them).
#
# StarDist tunables (override via env):
#   STARDIST_MODEL (default 2D_versatile_he)  # or a path to your custom model dir
#   STARDIST_PROB   (default 0.5)
#   STARDIST_NMS    (default 0.4)
#   BATCH_SIZE      (default 32 for A100)
#   DEDUP_RADIUS_UM (default 6.0)
#
# Example:
#   src/run_one_slide_stardist.sh raw_slides/PGP9-5-B27.svs results_pgp_stardist 10 "50 100 150"

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

# ---- StarDist settings (env-overridable) ----
STARDIST_MODEL="${STARDIST_MODEL:-2D_versatile_he}"
STARDIST_PROB="${STARDIST_PROB:-0.5}"
STARDIST_NMS="${STARDIST_NMS:-0.4}"
BATCH_SIZE="${BATCH_SIZE:-32}"  # Increased for A100 GPU
DEDUP_RADIUS_UM="${DEDUP_RADIUS_UM:-6.0}"

# Add scratch Python packages to PYTHONPATH (for numpy installed in scratch due to home quota)
export PYTHONPATH="/gpfs/scratch/smarisetty/python_packages:${PYTHONPATH:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SRC_DIR="${SRC_DIR:-src}"   # scripts live in src/ already

mkdir -p logs results

# Output layout
SLIDE_BASENAME="$(basename "$RAW_SLIDE")"
SLIDE_NAME="${SLIDE_BASENAME%.*}"
SLIDE_OUT="$RESULTS_DIR/$SLIDE_NAME"

# Detect slide type from filename
if [[ "$SLIDE_NAME" =~ ^HE- ]] || [[ "$SLIDE_NAME" =~ ^HE= ]]; then
    SLIDE_TYPE="H&E"
elif [[ "$SLIDE_NAME" =~ ^CD3- ]]; then
    SLIDE_TYPE="IHC_CD3"
elif [[ "$SLIDE_NAME" =~ ^GFAP- ]]; then
    SLIDE_TYPE="IHC_GFAP"
elif [[ "$SLIDE_NAME" =~ ^Iba1- ]] || [[ "$SLIDE_NAME" =~ ^IBA1- ]]; then
    SLIDE_TYPE="IHC_IBA1"
elif [[ "$SLIDE_NAME" =~ ^NF- ]]; then
    SLIDE_TYPE="IHC_NF"
elif [[ "$SLIDE_NAME" =~ ^PGP ]] || [[ "$SLIDE_NAME" =~ ^PID30 ]]; then
    SLIDE_TYPE="IHC_PGP95"
elif [[ "$SLIDE_NAME" =~ ^HCC ]]; then
    SLIDE_TYPE="H&E"
else
    # Default to H&E if pattern not recognized
    echo "⚠️  WARNING: Could not detect slide type from filename '$SLIDE_NAME', defaulting to H&E"
    SLIDE_TYPE="H&E"
fi
echo "Detected slide type: $SLIDE_TYPE"

PREVIEW_DIR="$SLIDE_OUT/preview"
MASKS_DIR="$SLIDE_OUT/masks"
TILES_DIR="$SLIDE_OUT/tiles"
STARDIST_DIR="$SLIDE_OUT/stardist"
FEATURES_DIR="$SLIDE_OUT/features"
VIZ_DIR="$SLIDE_OUT/viz"
QC_DIR="$SLIDE_OUT/qc"
mkdir -p "$PREVIEW_DIR" "$MASKS_DIR" "$TILES_DIR" "$STARDIST_DIR" "$FEATURES_DIR" "$VIZ_DIR" "$QC_DIR"

# ---- Read MPP and downsamples, with graceful fallback if openslide is missing ----
SLIDE_PATH="$RAW_SLIDE"
read -r MPP DOWNSAMPLES <<< "$(
SLIDE_PATH="$SLIDE_PATH" python3 - <<'PY' || true
import os, sys
mpp_default = 0.25
downs_default = "1.0"
try:
    import openslide
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
        mpp = mpp_default
    downs = [float(d) for d in sl.level_downsamples]
    print(mpp, " ".join(str(d) for d in downs))
except Exception as e:
    # No openslide or failure: fallback
    print(mpp_default, downs_default)
PY
)"
echo "ℹ️  MPP: $MPP"
echo "ℹ️  Level downsamples: ${DOWNSAMPLES:-1.0}"

# Arrays
read -r -a RADII_ARR <<< "$DENSITY_RADII_UM"
read -r -a DOWNS_ARR <<< "${DOWNSAMPLES:-1.0}"

# Common paths
THUMB_PATH="$PREVIEW_DIR/${SLIDE_NAME}_thumb.jpg"
TISSUE_MASK_PATH="$MASKS_DIR/${SLIDE_NAME}_tissue_mask.png"
TILES_JSON="$TILES_DIR/tiles.json"
STARDIST_MASKS_DIR="$STARDIST_DIR/masks"
mkdir -p "$STARDIST_MASKS_DIR"

RAW_FEATURES_CSV="$FEATURES_DIR/${SLIDE_NAME}_nuclei_features.csv"
DENSITY_CSV="$FEATURES_DIR/${SLIDE_NAME}_nuclei_features_density.csv"
SUMMARY_JSON="$FEATURES_DIR/${SLIDE_NAME}_summary.json"
ENRICHED_FEATURES_CSV="$FEATURES_DIR/${SLIDE_NAME}_nuclei_features_enriched.csv"

# --------------------------- Pipeline ---------------------------

# 00 preview
echo "==> [00] Quick preview"
python3 "$SRC_DIR/core/00_preview.py" \
  --raw "$RAW_SLIDE" --out_dir "$PREVIEW_DIR" --crop_size 1024

# 01 tissue mask
echo "==> [01] Tissue mask"
python3 "$SRC_DIR/core/01_tissue_mask.py" \
  --raw "$RAW_SLIDE" --out "$TISSUE_MASK_PATH" --level 0 --overwrite

# 02 tiling (+ optional stain norm inside)
# 02 make tiles + optional stain normalization
echo "==> [02] Make tiles"
python3 "$SRC_DIR/core/02_tile.py" \
  --raw "$RAW_SLIDE" --tissue_mask "$TISSUE_MASK_PATH" \
  --out_dir "$TILES_DIR" --tile_size 1024 --overlap 128 --level 0

# 02b NFB Filament Analysis (PARALLEL - only for NFB slides)
if [[ "$SLIDE_TYPE" == "IHC_NF" ]]; then
  echo "==> [02b] NFB Filament Analysis (filament-centric, not nucleus-based)"
  FILAMENT_DIR="$SLIDE_OUT/filaments"
  mkdir -p "$FILAMENT_DIR"
  python3 "$SRC_DIR/analysis/08_nfb_filament_analysis.py" \
    --slide "$RAW_SLIDE" \
    --output_dir "$FILAMENT_DIR" \
    --level 1 \
    --threshold 30 \
    --min_filament_length 20 &
  FILAMENT_PID=$!
  echo "  Filament analysis running in background (PID: $FILAMENT_PID)"
else
  echo "==> [02b] NFB Filament Analysis - SKIPPED (not NFB slide)"
fi

# 03 stardist seg

# 03 StarDist segmentation + merge  (REPLACES Cellpose)
echo "==> [03] StarDist segmentation + merge"
SEG_ARGS=(
  --tiles_dir "$TILES_DIR"
  --tiles_json "$TILES_JSON"
  --masks_dir "$STARDIST_MASKS_DIR"
  --out_csv "$RAW_FEATURES_CSV"
  --slide_id "$SLIDE_NAME"
  --mpp "$MPP"
  --diam_um "$NUC_DIAM_UM"
  --batch_size "$BATCH_SIZE"
  --dedup_radius_um "$DEDUP_RADIUS_UM"
  --stardist_model "$STARDIST_MODEL"
  --prob_thresh "$STARDIST_PROB"
  --nms_thresh "$STARDIST_NMS"
)
python3 "$SRC_DIR/core/03_segment_stardist.py" "${SEG_ARGS[@]}"

# 04 density + summary
echo "==> [04] Density + summary"
python3 "$SRC_DIR/core/04_density.py" \
  --input_csv "$RAW_FEATURES_CSV" --output_csv "$DENSITY_CSV" \
  --mpp "$MPP" --radii_um "${RADII_ARR[@]}" \
  --tissue_mask "$TISSUE_MASK_PATH" --mask_level 0 \
  --downsamples "${DOWNS_ARR[@]}" \
  --thumb "$THUMB_PATH" --summary_json "$SUMMARY_JSON" \
  --percentile_clip 2 98

# 05 enrich + visualizations (unchanged)
echo "==> [05] Enrich + visualize"
python3 "$SRC_DIR/core/05_features.py" \
  --input_csv "$DENSITY_CSV" --out_csv "$ENRICHED_FEATURES_CSV" \
  --out_dir "$VIZ_DIR" --thumb "$THUMB_PATH" \
  --radii_um "${RADII_ARR[@]}" --features circularity gray_mean

# 05b IHC intensity measurement (conditional - only for IHC slides)
if [[ "$SLIDE_TYPE" != "H&E" ]]; then
  echo "==> [05b] IHC intensity measurement (slide type: $SLIDE_TYPE)"
  python3 "$SRC_DIR/core/05b_ihc_intensity.py" \
    --slide "$RAW_SLIDE" \
    --nuclei_csv "$ENRICHED_FEATURES_CSV" \
    --output_csv "$ENRICHED_FEATURES_CSV" \
    --expansion 0.15 \
    --level 0 \
    --threshold 30
else
  echo "==> [05b] IHC intensity measurement - SKIPPED (H&E slide)"
fi

# 06 QC (unchanged)
echo "==> [06] QC"
python3 "$SRC_DIR/core/06_qc.py" \
  --input_csv "$ENRICHED_FEATURES_CSV" --output_dir "$QC_DIR" \
  --thumb "$THUMB_PATH" --scale 1.0

# 07 Feature Maps - Visual validation of all features overlaid on tissue
echo "==> [07] Feature Maps (validation visualizations)"
FEATURE_MAPS_DIR="$SLIDE_OUT/feature_maps"
mkdir -p "$FEATURE_MAPS_DIR"
python3 "$SRC_DIR/validation/generate_feature_maps.py" \
  --input_csv "$ENRICHED_FEATURES_CSV" \
  --slide "$RAW_SLIDE" \
  --output_dir "$FEATURE_MAPS_DIR"

# Wait for filament analysis if it's running
if [[ "$SLIDE_TYPE" == "IHC_NF" ]] && [[ -n "$FILAMENT_PID" ]]; then
  echo "==> Waiting for filament analysis to complete..."
  wait $FILAMENT_PID
  echo "✓ Filament analysis complete"
fi

echo "✅ Finished $SLIDE_NAME"
echo "📂 Outputs in: $SLIDE_OUT"
