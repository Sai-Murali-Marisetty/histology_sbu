#!/bin/bash
# run_adaptive_pipeline.sh - Smart pipeline that auto-detects slide type

set -euo pipefail

# Set threading limits globally to avoid OpenBLAS/UMAP segfaults
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

SLIDE_PATH="$1"
RESULTS_DIR="${2:-results}"

if [[ -z "$SLIDE_PATH" ]]; then
    echo "Usage: $0 <slide_path> [results_dir]"
    echo ""
    echo "Example:"
    echo "  $0 raw_slides/CD3-S25.svs results"
    exit 1
fi

if [[ ! -f "$SLIDE_PATH" ]]; then
    echo "❌ Error: Slide not found: $SLIDE_PATH"
    exit 1
fi

SLIDE_NAME=$(basename "$SLIDE_PATH" .svs)
SLIDE_OUT="$RESULTS_DIR/$SLIDE_NAME"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                    ADAPTIVE HISTOLOGY PIPELINE                             ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Slide: $SLIDE_NAME"
echo "📂 Output: $SLIDE_OUT"
echo ""

# ============================================================================
# STEP 0: Auto-detect slide type
# ============================================================================
echo "🔍 Step 0: Detecting slide type..."

SLIDE_TYPE=$(python3 -c "
import sys
sys.path.insert(0, 'src')
from utils.slide_detector import detect_slide_type
print(detect_slide_type('$SLIDE_NAME'))
")

echo "   → Detected: $SLIDE_TYPE"

# Get marker info if IHC
if [[ "$SLIDE_TYPE" == IHC_* ]]; then
    MARKER_INFO=$(python3 -c "
import sys
sys.path.insert(0, 'src')
from utils.slide_detector import get_marker, get_marker_info
marker = get_marker('$SLIDE_NAME')
if marker:
    print(get_marker_info(marker))
else:
    print('Unknown marker')
" || echo "Unknown marker")
    echo "   → Marker: $MARKER_INFO"
fi

echo ""

# ============================================================================
# STEP 1-6: Core pipeline (same for all slides)
# ============================================================================
echo "🔬 Steps 1-6: Running core pipeline..."
echo "────────────────────────────────────────────────────────────────────────────"

./scripts/run_one_slide.sh "$SLIDE_PATH" "$RESULTS_DIR"

ENRICHED_CSV="$SLIDE_OUT/features/${SLIDE_NAME}_nuclei_features_enriched.csv"

if [[ ! -f "$ENRICHED_CSV" ]]; then
    echo "❌ Error: Core pipeline failed - $ENRICHED_CSV not found"
    exit 1
fi

echo ""
echo "✅ Core pipeline complete"
echo ""

# ============================================================================
# STEP 7: Type-specific analysis (IHC brown stain)
# ============================================================================
FINAL_CSV="$ENRICHED_CSV"

if [[ "$SLIDE_TYPE" == IHC_* ]]; then
    echo "💉 Step 7: IHC brown stain analysis..."
    echo "────────────────────────────────────────────────────────────────────────────"
    
    python3 src/analysis/07_ihc_brown_stain.py \
        --input_csv "$ENRICHED_CSV" \
        --tiles_dir "$SLIDE_OUT/tiles" \
        --output_csv "$SLIDE_OUT/features/${SLIDE_NAME}_with_brown.csv" \
        --thumb "$SLIDE_OUT/preview/${SLIDE_NAME}_thumb.jpg" \
        --out_dir "$SLIDE_OUT/brown_stain" \
        --slide_type "$SLIDE_TYPE" \
        --config "configs/slide_config.yaml" || echo "WARNING: Step 7 failed (continuing anyway)"
    
    FINAL_CSV="$SLIDE_OUT/features/${SLIDE_NAME}_with_brown.csv"
    echo ""
    echo "✅ Brown stain analysis complete"
    echo ""
else
    echo "ℹ️  Step 7: Skipped (H&E slide - no brown stain)"
    echo ""
fi

# ============================================================================
# STEP 9: UMAP clustering
# ============================================================================
echo "🧬 Step 9: UMAP clustering..."
echo "────────────────────────────────────────────────────────────────────────────"

python3 src/analysis/09_umap_clustering.py \
    --input_csv "$FINAL_CSV" \
    --output_csv "$SLIDE_OUT/features/${SLIDE_NAME}_final.csv" \
    --out_dir "$SLIDE_OUT/clustering" \
    --thumb "$SLIDE_OUT/preview/${SLIDE_NAME}_thumb.jpg" \
    --slide_type "$SLIDE_TYPE" \
    --config "configs/slide_config.yaml" || echo "WARNING: Step 7 failed (continuing anyway)"

echo ""
echo "✅ Clustering complete"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                           PIPELINE COMPLETE                                 ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Slide: $SLIDE_NAME ($SLIDE_TYPE)"
echo "📂 Results: $SLIDE_OUT"
echo ""
echo "Key outputs:"
echo "  📄 Final CSV: $SLIDE_OUT/features/${SLIDE_NAME}_final.csv"
echo "  🖼️  Preview: $SLIDE_OUT/preview/panel_${SLIDE_NAME}_preview.png"
echo "  🔬 QC: $SLIDE_OUT/qc/"
echo "  📊 Clustering: $SLIDE_OUT/clustering/"

if [[ "$SLIDE_TYPE" == IHC_* ]]; then
    echo "  💉 Brown stain: $SLIDE_OUT/brown_stain/"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
