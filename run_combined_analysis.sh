#!/bin/bash
#
# Post-Processing: Combined UMAP Analysis
#
# Runs AFTER all individual slides are processed.
# Combines slides by stain type and generates:
# - Combined UMAP embeddings per stain type
# - Violin plots comparing slides within each stain
# - Feature distributions across slides
#
# Run this ONCE after all production jobs complete.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Usage
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <results_directory>"
    echo ""
    echo "Example:"
    echo "  $0 results_production_20251102_120000"
    echo ""
    echo "This script combines all slides by stain type and generates:"
    echo "  - Combined UMAP embeddings"
    echo "  - Violin plots comparing slides"
    echo "  - Feature distributions"
    exit 1
fi

RESULTS_DIR="$1"
OUTPUT_DIR="${RESULTS_DIR}/combined_analysis"
LOGS_DIR="logs_combined_analysis"

mkdir -p "$LOGS_DIR"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "COMBINED UMAP ANALYSIS - Post-Processing"
echo "=========================================="
echo "Results directory: $RESULTS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: Results directory not found: $RESULTS_DIR"
    exit 1
fi

# Count completed slides
N_SLIDES=$(find "$RESULTS_DIR" -maxdepth 1 -type d -name "*-*" | wc -l)
echo "Found $N_SLIDES slides in results directory"
echo ""

if [ "$N_SLIDES" -eq 0 ]; then
    echo "ERROR: No slide results found in $RESULTS_DIR"
    echo "Make sure individual slide processing is complete first."
    exit 1
fi

# Check that slides have enriched features
ENRICHED_COUNT=$(find "$RESULTS_DIR" -name "*_nuclei_features_enriched.csv" | wc -l)
echo "Slides with enriched features: $ENRICHED_COUNT / $N_SLIDES"
echo ""

if [ "$ENRICHED_COUNT" -eq 0 ]; then
    echo "ERROR: No enriched feature files found."
    echo "Make sure individual slide processing completed successfully."
    exit 1
fi

# Create SLURM job for combined analysis
JOB_SCRIPT="$LOGS_DIR/combined_umap_job.sh"

cat > "$JOB_SCRIPT" << 'EOF'
#!/bin/bash
#SBATCH --job-name=combined_umap
#SBATCH --output=logs_combined_analysis/combined_umap_%j.log
#SBATCH --error=logs_combined_analysis/combined_umap_%j.err
#SBATCH --partition=short-40core
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00

echo "=========================================="
echo "COMBINED UMAP ANALYSIS"
echo "=========================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# Activate environment
module purge
module load anaconda/3
eval "$(conda shell.bash hook)"
conda activate histology

# Add scratch Python packages
export PYTHONPATH="/gpfs/scratch/smarisetty/python_packages:$PYTHONPATH"

# Set threading
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

RESULTS_DIR="$1"
OUTPUT_DIR="$2"

echo "Results directory: $RESULTS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run combined UMAP analysis
echo "=========================================="
echo "Generating combined UMAPs by stain type..."
echo "=========================================="

python src/analysis/10_separate_umaps.py \
    --results_dir "$RESULTS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --stain_types "H&E" "CD3" "GFAP" "IBA1" "NF" "PGP9.5" \
    --max_points 200000 \
    --point_size 2.0 \
    --dpi 300

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS: Combined UMAP analysis complete"
    echo ""
    echo "Generated outputs:"
    echo "  - Per-stain UMAP embeddings"
    echo "  - Violin plots comparing slides"
    echo "  - Feature distributions"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
else
    echo "❌ FAILED: Combined UMAP analysis (exit code: $EXIT_CODE)"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
EOF

chmod +x "$JOB_SCRIPT"

# Submit job
echo "Submitting combined UMAP analysis job..."
echo ""

JOB_ID=$(sbatch "$JOB_SCRIPT" "$RESULTS_DIR" "$OUTPUT_DIR" | awk '{print $NF}')

echo "Job submitted: $JOB_ID"
echo ""
echo "Monitor progress:"
echo "  squeue -j $JOB_ID"
echo "  tail -f $LOGS_DIR/combined_umap_${JOB_ID}.log"
echo ""
echo "=========================================="
echo ""
echo "This will generate:"
echo ""
echo "For each stain type (H&E, CD3, GFAP, etc.):"
echo "  - <stain>_umap.png           # UMAP embedding with clusters"
echo "  - <stain>_umap.svg           # High-res vector version"
echo "  - <stain>_combined.csv       # All nuclei with UMAP coordinates"
echo "  - <stain>_violin_plots.png   # Feature comparisons across slides"
echo ""
echo "Expected completion: ~30-60 minutes depending on total nuclei count"
echo ""
echo "=========================================="
