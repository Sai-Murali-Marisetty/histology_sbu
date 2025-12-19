#!/bin/bash

SLIDES_DIR="data/raw"
OUT_DIR="outputs"
MPP=0.262697
DIAM=10
RADII="50 100"

for slide in $SLIDES_DIR/*.svs; do
    SLIDE_NAME=$(basename "$slide" .svs)
    echo "🚀 Processing $SLIDE_NAME"

    python src/run_full_pipeline.py \
      --slide "$slide" \
      --out_base "$OUT_DIR" \
      --mpp $MPP \
      --diam_um $DIAM \
      --radii_um $RADII \
      --overwrite
done
