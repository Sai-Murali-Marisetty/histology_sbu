#!/bin/bash
# run_all_by_type.sh - Process all slides, grouped by type

set -e

RAW_DIR="${1:-raw_slides}"
RESULTS_DIR="${2:-results_all}"

if [[ ! -d "$RAW_DIR" ]]; then
    echo "❌ Error: Slide directory not found: $RAW_DIR"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                    BATCH PROCESSING ALL SLIDES                             ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Input directory: $RAW_DIR"
echo "📂 Output directory: $RESULTS_DIR"
echo ""

# Count slides
N_SLIDES=$(ls "$RAW_DIR"/*.svs 2>/dev/null | wc -l)
if [[ $N_SLIDES -eq 0 ]]; then
    echo "❌ No .svs files found in $RAW_DIR"
    exit 1
fi

echo "Found $N_SLIDES slides"
echo ""

# Classify all slides first
echo "🔍 Classifying slides by type..."
python3 src/utils/slide_detector.py --batch "$RAW_DIR"

echo ""
read -p "Proceed with processing all $N_SLIDES slides? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy](es)?$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Process each slide
PROCESSED=0
FAILED=0

for SLIDE_PATH in "$RAW_DIR"/*.svs; do
    SLIDE_NAME=$(basename "$SLIDE_PATH" .svs)
    
    echo ""
    echo "┌────────────────────────────────────────────────────────────────────────┐"
    echo "│  Processing: $SLIDE_NAME"
    echo "│  Progress: $((PROCESSED + 1)) / $N_SLIDES"
    echo "└────────────────────────────────────────────────────────────────────────┘"
    echo ""
    
    # Run adaptive pipeline
    if ./scripts/run_adaptive_pipeline.sh "$SLIDE_PATH" "$RESULTS_DIR"; then
        PROCESSED=$((PROCESSED + 1))
        echo "✅ Success: $SLIDE_NAME"
    else
        FAILED=$((FAILED + 1))
        echo "❌ Failed: $SLIDE_NAME"
        
        # Log failure
        echo "$SLIDE_NAME" >> "$RESULTS_DIR/failed_slides.txt"
    fi
    
    echo ""
    echo "────────────────────────────────────────────────────────────────────────────"
    
    # Brief pause to avoid overheating
    sleep 2
done

# ============================================================================
# GENERATE COMPARISON REPORT
# ============================================================================
echo ""
echo "📊 Generating comparison report across all slides..."

python3 -c "
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, 'src')
from utils.slide_detector import detect_slide_type

results_dir = Path('$RESULTS_DIR')
summaries = []

for slide_dir in results_dir.iterdir():
    if not slide_dir.is_dir() or slide_dir.name == 'failed_slides.txt':
        continue
    
    final_csv = slide_dir / 'features' / f'{slide_dir.name}_final.csv'
    
    if not final_csv.exists():
        continue
    
    df = pd.read_csv(final_csv)
    slide_type = detect_slide_type(slide_dir.name)
    
    summary = {
        'slide': slide_dir.name,
        'type': slide_type,
        'n_nuclei': len(df),
        'mean_area': df['area_px'].mean() if 'area_px' in df.columns else None,
        'mean_density_100um': df['density_um2_r100.0'].mean() if 'density_um2_r100.0' in df.columns else None,
        'mean_coherency': df['coherency_150um'].mean() if 'coherency_150um' in df.columns else None,
        'n_clusters': df['cluster'].nunique() if 'cluster' in df.columns else None,
    }
    
    # Add brown stain stats if IHC
    if 'has_brown' in df.columns:
        summary['pct_brown_positive'] = df['has_brown'].mean() * 100
        summary['mean_brown_intensity'] = df['brown_intensity'].mean()
    
    summaries.append(summary)

if summaries:
    df_summary = pd.DataFrame(summaries)
    df_summary.to_csv(results_dir / 'SUMMARY_all_slides.csv', index=False)
    
    print(f'\n✅ Saved summary: {results_dir}/SUMMARY_all_slides.csv')
    print(f'\nProcessed {len(summaries)} slides:')
    
    # Print by type
    for slide_type in sorted(df_summary['type'].unique()):
        subset = df_summary[df_summary['type'] == slide_type]
        print(f'\n{slide_type} ({len(subset)} slides):')
        for _, row in subset.iterrows():
            print(f'  - {row[\"slide\"]}: {int(row[\"n_nuclei\"]):,} nuclei')
else:
    print('⚠️  No results to summarize')
"

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                         BATCH PROCESSING COMPLETE                          ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Results:"
echo "  ✅ Processed: $PROCESSED / $N_SLIDES"
echo "  ❌ Failed: $FAILED / $N_SLIDES"
echo ""
echo "📂 All results: $RESULTS_DIR"
echo "📄 Summary: $RESULTS_DIR/SUMMARY_all_slides.csv"

if [[ $FAILED -gt 0 ]]; then
    echo "⚠️  Failed slides: $RESULTS_DIR/failed_slides.txt"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
