#!/bin/bash
#
# Production Pipeline - HCC_raw_slides (12 slides)
#
# Processes all slides in /gpfs/scratch/smarisetty/histology/HCC_raw_slides/
# These are hepatocellular carcinoma slides - assume H&E staining
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$SCRIPT_DIR/HCC_raw_slides"
RESULTS_DIR="$SCRIPT_DIR/results_production_HCC_$(date +%Y%m%d_%H%M%S)"
LOGS_DIR="$SCRIPT_DIR/logs_production_HCC"

mkdir -p "$LOGS_DIR"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "PRODUCTION PIPELINE - HCC_raw_slides"
echo "=========================================="
echo "Input directory: $RAW_DIR"
echo "Output directory: $RESULTS_DIR"
echo "Logs directory: $LOGS_DIR"
echo ""

# Count slides
N_SLIDES=$(ls "$RAW_DIR"/*.svs 2>/dev/null | wc -l)
echo "Found $N_SLIDES HCC slides to process"
echo ""

# Job tracking
declare -A JOB_IDS

# Submit each slide
for SLIDE_PATH in "$RAW_DIR"/*.svs; do
    SLIDE_FILE=$(basename "$SLIDE_PATH")
    SLIDE_NAME="${SLIDE_FILE%.*}"
    
    echo "Submitting: $SLIDE_NAME"
    
    # HCC slides - assume H&E staining
    PARTITION="a100"
    TIME="01:00:00"
    MEM="64G"
    GPU="--gres=gpu:1"
    SLIDE_TYPE="H&E"
    
    # Create SLURM submission script
    JOB_SCRIPT="$LOGS_DIR/${SLIDE_NAME}_job.sh"
    
    cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=HCC_${SLIDE_NAME}
#SBATCH --output=${LOGS_DIR}/${SLIDE_NAME}_%j.log
#SBATCH --error=${LOGS_DIR}/${SLIDE_NAME}_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=${MEM}
#SBATCH ${GPU}
#SBATCH --time=${TIME}

echo "=========================================="
echo "Processing HCC Slide: ${SLIDE_NAME}"
echo "Slide type: ${SLIDE_TYPE}"
echo "=========================================="
echo "Start time: \$(date)"
echo "Node: \$(hostname)"
echo "Job ID: \$SLURM_JOB_ID"
echo ""

# Activate environment
module purge
module load anaconda/3
eval "\$(conda shell.bash hook)"
conda activate histology

# Add scratch Python packages to PYTHONPATH
export PYTHONPATH="/gpfs/scratch/smarisetty/python_packages:\$PYTHONPATH"

# Set threading limits
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

# Set cache directories
export KERAS_HOME=/gpfs/scratch/smarisetty/.keras
export TFHUB_CACHE_DIR=/gpfs/scratch/smarisetty/.tfhub_cache

# Run pipeline
bash ${SCRIPT_DIR}/scripts/run_one_slide_stardist.sh \\
    "$SLIDE_PATH" \\
    "$RESULTS_DIR" \\
    10 \\
    "50 100 150"

EXIT_CODE=\$?

echo ""
echo "=========================================="
if [ \$EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS: ${SLIDE_NAME}"
else
    echo "❌ FAILED: ${SLIDE_NAME} (exit code: \$EXIT_CODE)"
fi
echo "End time: \$(date)"
echo "=========================================="

exit \$EXIT_CODE
EOF

    chmod +x "$JOB_SCRIPT"
    
    # Submit job
    JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $NF}')
    JOB_IDS["$SLIDE_NAME"]=$JOB_ID
    
    echo "  Job ID: $JOB_ID (HCC, ${SLIDE_TYPE}, ${PARTITION}, ${TIME})"
    
    # Small delay to avoid overwhelming scheduler
    sleep 0.5
done

echo ""
echo "=========================================="
echo "SUBMISSION COMPLETE"
echo "=========================================="
echo "Submitted $N_SLIDES HCC jobs"
echo ""
echo "Job IDs:"
for SLIDE in "${!JOB_IDS[@]}"; do
    echo "  $SLIDE: ${JOB_IDS[$SLIDE]}"
done
echo ""
echo "Monitor jobs:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs:"
echo "  tail -f $LOGS_DIR/<slide_name>_*.log"
echo ""
echo "Results will be saved to:"
echo "  $RESULTS_DIR"
echo ""
echo "=========================================="
