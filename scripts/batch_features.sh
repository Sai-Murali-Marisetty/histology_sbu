#!/bin/bash
set -e

MMP=0.262697
MASK_LEVEL=2
DOWNSAMPLES="1.0 4.0 16.0 64.0"
RADII="50 100"
FEATURES="circularity gray_mean"

RAW=data/raw
RESULTS=results

for CSV in $(find $RESULTS -name nuclei_features.csv); do
    SLIDE=$(basename $(dirname $(dirname "$CSV")))
    echo "🔁 Processing $SLIDE..."

    ### Step 04 - Density Analysis
    python3 src/core/04_density.py \
      --input_csv "$CSV" \
      --output_csv "$RESULTS/$SLIDE/04_density/nuclei_with_density.csv" \
      --mpp $MMP \
      --radii_um $RADII \
      --tissue_mask "$RESULTS/$SLIDE/01_mask/tissue_mask.png" \
      --mask_level $MASK_LEVEL \
      --downsamples $DOWNSAMPLES \
      --thumb "$RESULTS/$SLIDE/00_preview/${SLIDE}_thumb.jpg" \
      --summary_json "$RESULTS/$SLIDE/04_density/density_summary.json"

    ### Step 05 - Feature Enrichment & Overlays
    python3 src/core/05_enrich_and_visualize_features.py \
      --input_csv "$RESULTS/$SLIDE/04_density/nuclei_with_density.csv" \
      --out_csv "$RESULTS/$SLIDE/05_features/features_enriched.csv" \
      --out_dir "$RESULTS/$SLIDE/05_features" \
      --thumb "$RESULTS/$SLIDE/00_preview/${SLIDE}_thumb.jpg" \
      --radii_um $RADII \
      --features $FEATURES

    ### Step 06 - QC Panel + Summary
    python3 src/core/06_qc_panel_and_summary.py \
      --input_csv "$RESULTS/$SLIDE/05_features/features_enriched.csv" \
      --output_dir "$RESULTS/$SLIDE/06_qc" \
      --thumb "$RESULTS/$SLIDE/00_preview/${SLIDE}_thumb.jpg" \
      --scale 1.0

    echo "✅ Finished $SLIDE"
done

echo "🎉 All done!"
