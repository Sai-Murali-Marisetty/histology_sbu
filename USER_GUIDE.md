# User Guide: Histology Image Analysis Pipeline

**For End Users** - A step-by-step guide to analyze your histology slides

**Last Updated**: December 2024
**Version**: 1.0

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Running the Pipeline](#running-the-pipeline)
4. [Understanding Your Results](#understanding-your-results)
5. [Common Issues & Solutions](#common-issues--solutions)
6. [Getting Help](#getting-help)

---

## 🚀 Quick Start

**Goal**: Analyze a whole slide image (WSI) and get nuclear features, density maps, and clusters.

**Time Required**:
- H&E slides: ~15-30 minutes
- IHC slides: ~30-45 minutes
- Neurofilament slides: ~45-60 minutes

**What You Need**:
- Whole slide image file (.svs format)
- Access to a computer with GPU (recommended) or CPU
- Pipeline already installed (see GETTING_STARTED.md if not)

---

## 📦 Prerequisites

### 1. Environment Setup (One-Time)

If this is your first time using the pipeline:

```bash
# Navigate to the pipeline directory
cd /path/to/histology

# Activate the conda environment
conda activate histology-pipeline

# Verify everything is installed
./scripts/test_setup.sh
```

You should see: `ALL TESTS PASSED ✅`

### 2. Prepare Your Data

**Organize your slides**:
```
data/
  └── raw_slides/
      ├── HE-S1.svs          # H&E stained slide
      ├── CD3-S25.svs        # CD3 IHC marker
      ├── GFAP-S17.svs       # GFAP IHC marker
      └── ...
```

**Naming Convention** (Important!):
- Use prefix to indicate stain type: `HE-`, `H&E-`, `CD3-`, `GFAP-`, `IBA1-`, `NF-`, `PGP9-5-`
- Example: `HE-S1.svs`, `CD3-tumor-section25.svs`
- The pipeline auto-detects slide type from filename

---

## 🔬 Running the Pipeline

### Option 1: Single Slide (Recommended for First-Time Users)

**Easiest way** - Let the pipeline auto-detect everything:

```bash
./scripts/run_adaptive_pipeline.sh data/raw_slides/HE-S1.svs
```

This script will:
1. ✅ Detect slide type automatically (H&E vs IHC)
2. ✅ Choose the best segmentation method (StarDist by default)
3. ✅ Run all appropriate analysis steps
4. ✅ Generate all visualizations
5. ✅ Create final CSV with results

**Watch for completion**:
- You'll see progress messages for each step
- Takes 15-60 minutes depending on slide size and type
- Results appear in `results/<slide_name>/`

---

### Option 2: Batch Processing Multiple Slides

**Process all slides in a folder**:

```bash
./scripts/run_all_by_type.sh data/raw_slides results
```

This will:
- Process all `.svs` files in `data/raw_slides/`
- Auto-detect each slide type
- Generate a summary report with statistics
- Create combined UMAP visualizations grouped by stain type

**Expected output**:
```
Processing 10 slides...
  [1/10] HE-S1.svs - COMPLETE ✓
  [2/10] CD3-S25.svs - COMPLETE ✓
  ...
  [10/10] NF-S19.svs - COMPLETE ✓

Summary saved to: results/summary_report.csv
```

---

### Option 3: HPC/SLURM Cluster (For Large Batches)

**If you have access to an HPC cluster**:

```bash
# Edit the script to list your slide paths
nano submit_production_raw_slides.sh

# Submit all jobs
./submit_production_raw_slides.sh
```

This creates SLURM job scripts for each slide and submits them in parallel.

**Monitor progress**:
```bash
# Check job status
squeue -u $USER

# View logs
tail -f logs_production/HE-S1_*.out
```

---

## 📊 Understanding Your Results

### Output Directory Structure

After processing `HE-S1.svs`, you'll find:

```
results/HE-S1/
├── preview/
│   ├── HE-S1_thumb.jpg                     # Quick thumbnail
│   └── panel_HE-S1_preview.png             # 3-panel preview
│
├── masks/
│   └── HE-S1_tissue_mask.png               # Detected tissue regions
│
├── stardist/  (or cellpose/)
│   └── masks/                              # Nuclear segmentation masks
│
├── features/
│   ├── HE-S1_nuclei_features.csv           # Raw features
│   ├── HE-S1_nuclei_features_density.csv   # + density
│   ├── HE-S1_nuclei_features_enriched.csv  # + coherency
│   └── HE-S1_final.csv                     # ⭐ FINAL OUTPUT
│
├── clustering/
│   ├── umap_clusters.png                   # ⭐ UMAP visualization
│   ├── cluster_spatial.png                 # Clusters overlaid on tissue
│   ├── cluster_features.png                # Feature distributions
│   └── cluster_statistics.csv              # Per-cluster metrics
│
├── feature_maps/
│   ├── area_px_validation.png              # 3-panel: map + H&E + overlay
│   ├── circularity_validation.png
│   ├── density_r100.0_validation.png
│   └── ...                                 # One per feature
│
└── qc/
    └── qc_summary.json                     # Quality metrics
```

**For IHC slides**, you'll also get:
```
results/CD3-S25/
├── brown_stain/
│   ├── brown_stain_overlay.jpg             # DAB staining visualization
│   └── brown_density_r100.jpg              # Brown stain density map
└── ...
```

**For Neurofilament slides**, you'll also get:
```
results/NF-S19/
├── filaments/
│   ├── filaments.csv                       # Per-filament measurements
│   ├── filament_summary.json               # Statistics
│   └── filament_visualizations/            # Traced filaments
└── ...
```

---

### Key Output Files

#### 1. **Final CSV** - `results/<slide>/features/<slide>_final.csv`

**This is your main result file.** Each row = one nucleus.

**Essential Columns**:
- `nucleus_id` - Unique identifier
- `x`, `y` - Pixel coordinates
- `x_um`, `y_um` - Micron coordinates
- `area_px` - Nucleus area (pixels)
- `circularity` - Shape metric (0-1, 1=perfect circle)
- `eccentricity` - Elongation (0-1, 0=circle, 1=line)
- `density_um2_r50.0` - Nuclei per µm² within 50µm radius
- `density_um2_r100.0` - Nuclei per µm² within 100µm radius
- `density_um2_r150.0` - Nuclei per µm² within 150µm radius
- `coherency_150um` - Nuclear alignment metric (0-1, higher=more aligned)
- `umap_1`, `umap_2` - UMAP embedding coordinates
- `cluster` - Cluster assignment (0 to n_clusters-1)

**For IHC slides, additional columns**:
- `has_brown` - Boolean (0 or 1)
- `brown_intensity` - DAB staining intensity (0-1)
- `brown_density_100um` - Brown-positive nuclei density
- `marker_intensity_mean` - Average marker intensity in perinuclear region

**Use this file for**:
- Further statistical analysis
- Plotting custom visualizations
- Comparing diseased vs healthy samples
- Exporting to other tools (R, MATLAB, etc.)

---

#### 2. **UMAP Clusters** - `results/<slide>/clustering/umap_clusters.png`

**Visual Summary** showing:
- Each point = one nucleus
- Colors = different clusters
- Spatial proximity in UMAP = similar nuclear features
- Helps identify distinct cell populations

**How to Interpret**:
- **Tight clusters**: Homogeneous cell population
- **Spread-out points**: Heterogeneous features
- **Multiple distinct clusters**: Multiple cell types or tissue regions
- **Outliers**: Unusual nuclei (artifacts, dividing cells, etc.)

---

#### 3. **Feature Maps** - `results/<slide>/feature_maps/*_validation.png`

**3-Panel Visualizations** for each feature:
- **Left**: Feature colormap (heatmap showing feature values)
- **Middle**: Original H&E image
- **Right**: Overlay (feature map on top of H&E)

**Use these to**:
- Verify features make biological sense
- Identify artifacts or processing errors
- Understand spatial patterns (e.g., density gradients)
- Quality control validation

---

#### 4. **Cluster Spatial Map** - `results/<slide>/clustering/cluster_spatial.png`

**Shows clusters overlaid on tissue image**:
- Each color = one cluster
- Helps understand spatial organization
- Identifies tissue architecture patterns

**Look for**:
- Spatial segregation of clusters
- Boundary regions between tissue types
- Artifacts concentrated in specific clusters

---

## 🔧 Common Issues & Solutions

### Issue 1: "No tissue detected"

**Symptom**: Empty tissue mask, no nuclei found

**Solutions**:
1. Check slide orientation (is tissue visible in preview?)
2. Adjust tissue detection parameters in `src/core/01_tissue_mask.py`
3. Verify slide isn't corrupted: `openslide-show-properties slide.svs`

---

### Issue 2: "Out of memory" error

**Symptom**: Process crashes during segmentation or tiling

**Solutions**:
1. Reduce batch size in `configs/slide_config.yaml`:
   ```yaml
   segmentation:
     batch_size: 16  # Reduce from 32
   ```
2. Use CPU instead of GPU (slower but more memory):
   ```bash
   export CUDA_VISIBLE_DEVICES=""
   ./scripts/run_adaptive_pipeline.sh slide.svs
   ```
3. Process on HPC with more RAM

---

### Issue 3: "Slide type not recognized"

**Symptom**: Pipeline defaults to H&E when it shouldn't

**Solutions**:
1. Rename file with proper prefix: `mv slide.svs CD3-slide.svs`
2. Manually specify type in `configs/slide_config.yaml`
3. Check `src/utils/slide_detector.py` for pattern matching rules

---

### Issue 4: Too many/too few nuclei detected

**Symptom**: Segmentation looks wrong

**Solutions**:
1. Try different segmenter:
   - StarDist (default): Better for densely packed nuclei
   - Cellpose: Better for irregular shapes
   ```bash
   # Use Cellpose instead
   ./scripts/run_one_slide.sh slide.svs
   ```
2. Adjust nucleus diameter in `configs/slide_config.yaml`:
   ```yaml
   segmentation:
     diameter_um: 8.0  # Increase for larger nuclei, decrease for smaller
   ```
3. Compare both segmenters:
   ```bash
   python src/analysis/08_compare_segmenters.py slide_name
   ```

---

### Issue 5: UMAP clustering doesn't separate tissues

**Symptom**: All nuclei in one cluster, or random clustering

**Solutions**:
1. Increase number of clusters in `configs/slide_config.yaml`:
   ```yaml
   clustering:
     n_clusters: 20  # Try 30-40 for complex tissues
   ```
2. Add more discriminative features:
   ```yaml
   clustering:
     features: [area_px, circularity, eccentricity, density_um2_r100.0,
                coherency_150um, r, g, b]  # Add color features
   ```
3. Enable PCA preprocessing for high-dimensional data:
   ```python
   # In 09_umap_clustering.py
   USE_PCA = True
   ```

---

## 🎯 Typical Workflows

### Workflow 1: Analyze New Patient Samples

**Scenario**: You have 5 new H&E slides from patients to analyze

```bash
# 1. Copy slides to data folder
cp /path/to/patient_slides/*.svs data/raw_slides/

# 2. Run batch processing
./scripts/run_all_by_type.sh data/raw_slides results

# 3. Review results
# - Check feature_maps/ for any obvious artifacts
# - Look at clustering/umap_clusters.png for separation
# - Open final CSVs in Excel/R for statistical analysis

# 4. Generate summary statistics
python -c "
import pandas as pd
import glob

# Load all final CSVs
csv_files = glob.glob('results/*/features/*_final.csv')
for f in csv_files:
    df = pd.read_csv(f)
    print(f'\n{f}:')
    print(df.describe())
"
```

---

### Workflow 2: Compare Diseased vs Healthy Tissue

**Scenario**: You have matched diseased and healthy slides

```bash
# 1. Process all slides
./scripts/run_all_by_type.sh data/raw_slides results

# 2. Compare UMAP results
# - Open results/healthy_HE-S1/clustering/umap_clusters.png
# - Open results/diseased_HE-S2/clustering/umap_clusters.png
# - Look for different cluster patterns

# 3. Statistical comparison
python -c "
import pandas as pd
import scipy.stats as stats

healthy = pd.read_csv('results/healthy_HE-S1/features/healthy_HE-S1_final.csv')
diseased = pd.read_csv('results/diseased_HE-S2/features/diseased_HE-S2_final.csv')

# Compare mean density
print('Density comparison:')
print(f'Healthy: {healthy.density_um2_r100.mean():.3f}')
print(f'Diseased: {diseased.density_um2_r100.mean():.3f}')
t_stat, p_val = stats.ttest_ind(healthy.density_um2_r100, diseased.density_um2_r100)
print(f'p-value: {p_val:.4e}')

# Compare coherency
print('\nCoherency comparison:')
print(f'Healthy: {healthy.coherency_150um.mean():.3f}')
print(f'Diseased: {diseased.coherency_150um.mean():.3f}')
t_stat, p_val = stats.ttest_ind(healthy.coherency_150um, diseased.coherency_150um)
print(f'p-value: {p_val:.4e}')
"
```

---

### Workflow 3: IHC Marker Analysis

**Scenario**: Quantify CD3+ T cells in tumor sections

```bash
# 1. Process CD3-stained slides
./scripts/run_adaptive_pipeline.sh data/raw_slides/CD3-tumor-S25.svs

# 2. Check brown stain visualization
open results/CD3-tumor-S25/brown_stain/brown_stain_overlay.jpg

# 3. Count positive cells
python -c "
import pandas as pd

df = pd.read_csv('results/CD3-tumor-S25/features/CD3-tumor-S25_final.csv')

total_nuclei = len(df)
positive_nuclei = df.has_brown.sum()
percent_positive = 100 * positive_nuclei / total_nuclei

print(f'Total nuclei: {total_nuclei}')
print(f'CD3+ nuclei: {positive_nuclei}')
print(f'Percent positive: {percent_positive:.1f}%')

# Get spatial distribution
print(f'\nMean brown intensity: {df[df.has_brown==1].brown_intensity.mean():.3f}')
print(f'Mean brown density: {df.brown_density_100um.mean():.3f}')
"
```

---

### Workflow 4: Quality Control Check

**Scenario**: Verify pipeline is working correctly on new dataset

```bash
# 1. Run on test slide
./scripts/run_adaptive_pipeline.sh data/raw_slides/test_HE-S1.svs

# 2. Check validation visualizations
# Open all files in results/test_HE-S1/feature_maps/
# Verify each feature makes biological sense:

# - area_px_validation.png: Larger nuclei should be visible as brighter
# - circularity_validation.png: Round nuclei = yellow/red, elongated = blue
# - density_*_validation.png: Dense regions should be bright
# - coherency_150um_validation.png: Aligned structures = bright

# 3. Check for artifacts in clusters
# - Open clustering/cluster_spatial.png
# - Artifacts should be in separate cluster
# - Different tissue regions should have different clusters

# 4. Verify nuclei count is reasonable
python -c "
import pandas as pd
df = pd.read_csv('results/test_HE-S1/features/test_HE-S1_final.csv')
print(f'Total nuclei: {len(df)}')
print(f'Nuclei per cluster:')
print(df.cluster.value_counts().sort_index())
"
```

---

## 📞 Getting Help

### If you encounter issues:

1. **Check the logs**:
   ```bash
   # For local runs
   cat results/<slide_name>/processing.log

   # For SLURM jobs
   tail -50 logs_production/<slide_name>_*.out
   ```

2. **Verify setup**:
   ```bash
   ./scripts/test_setup.sh
   ```

3. **Review documentation**:
   - Installation: `GETTING_STARTED.md`
   - HPC usage: `PRODUCTION_PIPELINE_GUIDE.md`
   - Code details: `DEVELOPER_GUIDE.md`
   - Troubleshooting: `PRODUCTION_PIPELINE_GUIDE.md` (Section 4)

4. **Contact**:
   - Developer documentation: See `DEVELOPER_GUIDE.md`
   - Code repository: [Will be on GitHub - check with Principal Investigator]
   - Lab contact: Principal Investigator

---

## 📚 Next Steps

### For Basic Users:
- Start with one slide using `run_adaptive_pipeline.sh`
- Review output visualizations for quality
- Explore final CSV in Excel or Python/R
- Try batch processing when comfortable

### For Advanced Users:
- Read `DEVELOPER_GUIDE.md` to understand pipeline architecture
- Modify parameters in `configs/slide_config.yaml`
- Add custom features or analysis steps
- Contribute improvements back to the codebase

### For Developers Taking Over:
- Start with `HANDOFF_NOTES.md` for current project status
- Read `DEVELOPER_GUIDE.md` for code architecture
- Review `VALIDATION_STATUS.md` for completed validations
- Check `DELIVERY_CHECKLIST.md` for remaining tasks

---

**Happy analyzing!** 🔬
