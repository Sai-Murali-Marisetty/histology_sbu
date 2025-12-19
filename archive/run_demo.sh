#!/bin/bash
# Quick demo script to test new features on one slide
# Usage: ./run_demo.sh /path/to/slide.svs

set -euo pipefail

SLIDE_PATH="${1:-}"

if [[ -z "$SLIDE_PATH" ]]; then
    echo "Usage: $0 /path/to/slide.svs"
    echo ""
    echo "This runs the pipeline with NEW features:"
    echo "  • Coherency metric"
    echo "  • Variance statistics"
    echo "  • 150µm radius (matching HistoVision paper)"
    echo "  • Coordinate mapping fix"
    exit 1
fi

if [[ ! -f "$SLIDE_PATH" ]]; then
    echo "Error: Slide not found: $SLIDE_PATH"
    exit 1
fi

# Get slide name
SLIDE_NAME=$(basename "$SLIDE_PATH" .svs)
echo "Processing: $SLIDE_NAME"

# Output directories
RESULTS_DIR="results_demo"
SLIDE_OUT="$RESULTS_DIR/$SLIDE_NAME"

# Run main pipeline with NEW parameters
echo ""
echo "Running pipeline with NEW features enabled..."
echo "  - Density radii: 50, 100, 150 µm"
echo "  - Features: coherency, variance, CV"
echo ""

./run_one_slide.sh "$SLIDE_PATH" "$RESULTS_DIR" 10 "50 100 150"

# Generate demo panel
echo ""
echo "Generating demo panel for meeting..."
python make_demo_panel.py --slide_name "$SLIDE_NAME" --results_dir "$RESULTS_DIR"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ DEMO COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Results in: $SLIDE_OUT"
echo ""
echo "Key outputs:"
echo "  • Features CSV: $SLIDE_OUT/features/${SLIDE_NAME}_nuclei_features_enriched.csv"
echo "  • Demo panel: $SLIDE_OUT/DEMO_PANEL_for_meeting_${SLIDE_NAME}.png"
echo "  • Coherency map: $SLIDE_OUT/viz/overlay_coherency_150um.jpg"
echo "  • Variance maps: $SLIDE_OUT/viz/overlay_*_variance_*.jpg"
echo ""
echo "Open the demo panel to see everything in one view!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
