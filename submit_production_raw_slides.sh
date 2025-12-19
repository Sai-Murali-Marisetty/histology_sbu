#!/bin/bash
#
# Production Pipeline - raw_slides (28 slides)
#
# Processes all slides in /gpfs/scratch/smarisetty/histology/raw_slides/
# Each slide submitted as separate SLURM job with appropriate resources
#
# Slide types:
# - H&E: Standard nuclear analysis
# - IHC (CD3, GFAP, IBA1, PGP9-5): Nuclear + perinuclear intensity
# - IHC_NF: Nuclear + perinuclear intensity + filament analysis
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$SCRIPT_DIR/raw_slides"
RESULTS_DIR="$SCRIPT_DIR/results_production_$(date +%Y%m%d_%H%M%S)"
LOGS_DIR="$SCRIPT_DIR/logs_production"

mkdir -p "$LOGS_DIR"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "PRODUCTION PIPELINE - raw_slides"
echo "=========================================="
echo "Input directory: $RAW_DIR"
echo "Output directory: $RESULTS_DIR"
echo "Logs directory: $LOGS_DIR"
echo ""

# Count slides
N_SLIDES=$(ls "$RAW_DIR"/*.svs 2>/dev/null | wc -l)
echo "Found $N_SLIDES slides to process"
echo ""

# Job tracking
declare -A JOB_IDS

# Submit each slide
for SLIDE_PATH in "$RAW_DIR"/*.svs; do
    SLIDE_FILE=$(basename "$SLIDE_PATH")
    SLIDE_NAME="${SLIDE_FILE%.*}"
    
    echo "Submitting: $SLIDE_NAME"
    
    # Detect slide type for resource allocation
    if [[ "$SLIDE_NAME" =~ ^HE- ]]; then
        PARTITION="a100"
        TIME="01:00:00"
        MEM="64G"
        GPU="--gres=gpu:1"
        SLIDE_TYPE="H&E"
    elif [[ "$SLIDE_NAME" =~ ^NF- ]]; then
        PARTITION="a100"
        TIME="02:00:00"  # Extra time for filament analysis
        MEM="80G"
        GPU="--gres=gpu:1"
        SLIDE_TYPE="IHC_NF"
    elif [[ "$SLIDE_NAME" =~ ^(CD3|GFAP|IBA1|PGP) ]]; then
        PARTITION="a100"
        TIME="01:30:00"  # Extra time for IHC intensity
        MEM="64G"
        GPU="--gres=gpu:1"
        if [[ "$SLIDE_NAME" =~ ^PGP ]]; then
            SLIDE_TYPE="IHC_PGP95"
        elif [[ "$SLIDE_NAME" =~ ^CD3 ]]; then
            SLIDE_TYPE="IHC_CD3"
        elif [[ "$SLIDE_NAME" =~ ^GFAP ]]; then
            SLIDE_TYPE="IHC_GFAP"
        elif [[ "$SLIDE_NAME" =~ ^IBA1 ]]; then
            SLIDE_TYPE="IHC_IBA1"
        else
            SLIDE_TYPE="IHC"
        fi
    else
        # Default to H&E settings
        PARTITION="a100"
        TIME="01:00:00"
        MEM="64G"
        GPU="--gres=gpu:1"
        SLIDE_TYPE="H&E"
    fi
    
    # Create SLURM submission script
    JOB_SCRIPT="$LOGS_DIR/${SLIDE_NAME}_job.sh"
    
    cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${SLIDE_NAME}
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
echo "Processing: ${SLIDE_NAME}"
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
    
    echo "  Job ID: $JOB_ID (${SLIDE_TYPE}, ${PARTITION}, ${TIME})"
    
    # Small delay to avoid overwhelming scheduler
    sleep 0.5
done

echo ""
echo "=========================================="
echo "SUBMISSION COMPLETE"
echo "=========================================="
echo "Submitted $N_SLIDES jobs"
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
