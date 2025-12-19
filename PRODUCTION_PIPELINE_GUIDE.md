# Production Pipeline - Complete Guide

## Fixes Applied (Nov 2, 2025)

Two critical bugs were identified and fixed:

1. **Missing `#SBATCH` prefix for GPU directive**: The `${GPU}` variable in job scripts was missing the `#SBATCH` prefix, causing SLURM to interpret it as a shell command instead of a directive.
   - **Error**: `/cm/local/apps/slurm/var/spool/job1424577/slurm_script: line 10: --gres=gpu:1: command not found`
   - **Fix**: Changed `${GPU}` to `#SBATCH ${GPU}` in both submission scripts

2. **Incorrect Python script name**: Pipeline referenced `02_tile_and_stain_norm.py` but actual file is `02_tile.py`
   - **Error**: `python3: can't open file '/gpfs/scratch/smarisetty/histology/src/core/02_tile_and_stain_norm.py': [Errno 2] No such file or directory`
   - **Fix**: Updated `run_one_slide_stardist.sh` to use correct filename `02_tile.py`

**All jobs cancelled and cleaned up. Ready to resubmit.**

---



Complete histology image analysis pipeline with:
- ✅ Nuclear segmentation (StarDist)
- ✅ Morphology, density, coherency features
- ✅ IHC intensity measurement (perinuclear)
- ✅ NFB filament analysis (axonal tracing)
- ✅ Orientation vector visualization
- ✅ UMAP clustering with spatial overlay
- ✅ Comprehensive feature validation maps

## Datasets

### 1. Original Slides (`raw_slides/`) - 28 slides
- **H&E** (4 slides): HE-B17, HE-B27, HE-S19, HE-S25
- **CD3** (4 slides): CD3-B17, CD3-B27, CD3-S19, CD3-S25
- **GFAP** (4 slides): GFAP-B17, GFAP-B27, GFAP-S19, GFAP-S25
- **IBA1** (4 slides): IBA1-B17, IBA1-B27, IBA1-S19, IBA1-S25
- **Neurofilament** (4 slides): NF-B17, NF-B27, NF-S19, NF-S25
- **PGP9.5** (4 slides): PGP9-5-B17, PGP9-5-B27, PGP9-5-S19, PGP9-5-S25
- **Other IHC** (4 slides): Various markers

### 2. HCC Slides (`HCC_raw_slides/`) - 12 slides
- Hepatocellular carcinoma specimens
- Assumed H&E staining

**Total: 40 slides**

## Pipeline Features by Slide Type

### H&E Slides
- Tissue mask (multi-component)
- StarDist nuclear segmentation
- Morphology features (area, circularity, eccentricity, etc.)
- Density maps (50, 100, 150 µm radii)
- Coherency (nuclear alignment)
- Local statistics (variance, CV, mean)
- Orientation vector visualization ⭐ **NEW**
- UMAP clustering with spatial overlay
- Feature validation maps

### IHC Slides (CD3, GFAP, IBA1, PGP9.5)
- **All H&E features PLUS:**
- Perinuclear intensity measurement (15% expansion)
  - `marker_intensity_mean`, `marker_intensity_max`, `marker_intensity_std`
  - `marker_positive_area_fraction`
  - `perinuclear_intensity_mean`, `perinuclear_positive_fraction`

### NFB Slides (Neurofilament)
- **All IHC features PLUS:**
- **Filament-centric analysis** (parallel, independent) ⭐ **NEW**
  - DAB filament segmentation
  - Skeletonization for centerline tracing
  - Filament count, length distribution
  - Network topology (endpoints, branch points)
  - Orientation distribution
  - Spatial density maps
  - Connectivity analysis

## Resource Allocation

### H&E Slides
- Partition: `a100`
- Time: 1 hour
- Memory: 64 GB
- GPU: 1 A100

### IHC Slides (CD3, GFAP, IBA1, PGP9.5)
- Partition: `a100`
- Time: 1.5 hours (extra for intensity measurement)
- Memory: 64 GB
- GPU: 1 A100

### NFB Slides
- Partition: `a100`
- Time: 2 hours (extra for filament analysis)
- Memory: 80 GB
- GPU: 1 A100

## Running Production Pipeline

### Step 1: Submit Original Slides (28 slides)
```bash
cd /gpfs/scratch/smarisetty/histology
bash submit_production_raw_slides.sh
```

This will:
- Auto-detect slide type from filename
- Submit 28 independent SLURM jobs
- Allocate appropriate resources per slide type
- Run NFB filament analysis automatically for NF- slides
- Log all output to `logs_production/`
- Save results to `results_production_YYYYMMDD_HHMMSS/`

**Note**: Each slide runs **per-slide analysis only** at this stage. Combined UMAP comes later.

### Step 2: Submit HCC Slides (12 slides)
```bash
bash submit_production_HCC_slides.sh
```

This will:
- Process 12 HCC slides as H&E
- Submit 12 independent SLURM jobs
- Log to `logs_production_HCC/`
- Save results to `results_production_HCC_YYYYMMDD_HHMMSS/`

### Step 3: Monitor Progress
```bash
# Check job status
squeue -u $USER

# Watch specific slide log
tail -f logs_production/HE-B17_*.log

# Count completed jobs
grep "✅ SUCCESS" logs_production/*.log | wc -l

# Check for failures
grep "❌ FAILED" logs_production/*.log
```

### Step 4: Combined UMAP Analysis (AFTER all slides complete)

**IMPORTANT**: This step combines slides by stain type. Run ONLY after all individual slides are processed.

```bash
# For original slides
bash run_combined_analysis.sh results_production_YYYYMMDD_HHMMSS

# For HCC slides (optional - all H&E)
bash run_combined_analysis.sh results_production_HCC_YYYYMMDD_HHMMSS
```

This generates:
- **Per-stain UMAP embeddings** (H&E, CD3, GFAP, NF, PGP9.5, etc.)
- **Violin plots** comparing feature distributions across slides within each stain type
- **Combined CSV files** with all nuclei from that stain type

Expected time: 30-60 minutes depending on total nuclei count.

## Output Structure

For each slide, results are organized as:

```
results_production_YYYYMMDD_HHMMSS/
├── <SLIDE_NAME>/               # Per-slide results
│   ├── preview/
│   ├── masks/
│   ├── tiles/
│   ├── stardist/
│   ├── features/
│   │   └── <slide>_nuclei_features_enriched.csv  # Per-slide data
│   ├── viz/                    # Coherency & local stats overlays
│   ├── feature_maps/           # ⭐ All features validated on tissue
│   │   ├── area_px_validation.png
│   │   ├── circularity_validation.png
│   │   ├── coherency_50um_validation.png
│   │   ├── coherency_100um_validation.png
│   │   ├── coherency_150um_validation.png
│   │   ├── density_um2_r50.0_validation.png
│   │   ├── density_um2_r100.0_validation.png
│   │   ├── density_um2_r150.0_validation.png
│   │   └── ... (all features as 3-panel visualizations)
│   ├── qc/
│   └── filaments/              # NFB only
└── combined_analysis/          # Generated AFTER all slides (Step 4)
    ├── HandE/
    │   ├── HandE_umap.png      # Combined UMAP for all H&E slides
    │   ├── HandE_umap.svg      # Vector version
    │   ├── HandE_combined.csv  # ALL H&E nuclei from all slides
    │   └── HandE_violin_plots.png  # Compare slides
    ├── CD3/
    │   ├── CD3_umap.png
    │   ├── CD3_combined.csv
    │   └── CD3_violin_plots.png
    ├── GFAP/
    │   └── ...
    ├── NF/
    │   └── ...
    └── PGP9.5/
        └── ...
```

## Key Output Files

### Nuclear Analysis
- **`clustered.csv`**: Complete nucleus-level data
  - Coordinates, morphology, density, coherency
  - Local statistics, cluster labels
  - IHC intensity (if applicable)
  - ~45,000-50,000 nuclei per slide

### Visualizations
- **`cluster_spatial.png`**: Clusters overlaid on tissue
- **`orientation_vectors.png`**: Nucleus orientations as arrows ⭐ **NEW**
- **`overlay_coherency_*.jpg`**: Alignment color maps
- **Feature validation maps**: 23 different features

### NFB Filaments (NF slides only)
- **`filaments.csv`**: Per-filament data
  - Length, orientation, centroid
  - ~500-1000 filaments per slide
- **`filament_summary.json`**: Network statistics
  - Total length, endpoint count, branching

## Analysis Duration Estimates

- **H&E slide**: ~15-20 minutes
- **IHC slide**: ~25-30 minutes (+ intensity measurement)
- **NFB slide**: ~30-40 minutes (+ filament analysis running in parallel)

**Total estimated time:**
- 28 original slides: ~12-15 hours (parallel execution)
- 12 HCC slides: ~3-4 hours (parallel execution)
- **Combined: ~15-19 hours** (if all resources available)

With 10 A100 GPUs available, actual wall time will be shorter.

## Validation Checklist

For each completed slide, verify:

1. ✅ **Tissue mask captures all regions** (check `tissue_mask_visual_thumb.png`)
2. ✅ **Nuclei count reasonable** (~40,000-60,000 typical)
3. ✅ **Cluster spatial distribution makes sense** (not random noise)
4. ✅ **Coherency maps show structure** (high along vessels/boundaries)
5. ✅ **Orientation vectors aligned** (in structured regions)
6. ✅ **IHC intensity non-zero** (for IHC slides)
7. ✅ **Filaments detected** (for NFB slides, ~500-1000)

## Troubleshooting

### Job Failed
```bash
# Check error log
cat logs_production/<slide>_*.err

# Common issues:
# - OOM (out of memory): Increase --mem
# - Timeout: Increase --time
# - Numpy error: Check PYTHONPATH is set
```

### Resubmit Single Slide
```bash
# Edit the generated job script
vim logs_production/<SLIDE_NAME>_job.sh

# Resubmit
sbatch logs_production/<SLIDE_NAME>_job.sh
```

### Missing Output
```bash
# Check if job completed
grep "SUCCESS\|FAILED" logs_production/<slide>_*.log

# Check disk space
df -h /gpfs/scratch/smarisetty
```

## Post-Processing

After all jobs complete:

1. **Aggregate statistics**
```bash
python scripts/aggregate_results.py \
    --results_dir results_production_YYYYMMDD_HHMMSS \
    --output summary_statistics.csv
```

2. **Compare slides**
```bash
python scripts/compare_slides.py \
    --results_dir results_production_YYYYMMDD_HHMMSS \
    --output comparison_plots/
```

3. **Archive results**
```bash
tar -czf production_results_$(date +%Y%m%d).tar.gz results_production_*
```

## Updates in This Pipeline

### ⭐ New Features Added
1. **Orientation Vector Visualization** - Arrows showing nucleus direction
2. **NFB Filament Analysis** - Separate axonal tracing pipeline
3. **IHC Intensity Measurement** - Perinuclear marker quantification
4. **Multi-component Tissue Mask** - No longer drops small tissue pieces

### 🔧 Fixes Applied
1. Tissue mask keeps ALL components (not just largest)
2. StarDist batch size optimized for A100 (32)
3. Numpy installed to scratch (home quota issue)
4. Shell argument quoting fixed (H&E → "H&E")

## Support

For issues or questions, check:
- `PIPELINE_TEST_RESULTS.md` - Validation run summary
- `FEATURE_MAPS_GUIDE.md` - Feature interpretation guide
- `GETTING_STARTED.md` - Original setup guide

---

**Ready to run!** Execute the submission scripts and monitor progress.
