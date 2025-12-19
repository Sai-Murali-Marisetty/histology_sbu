# Complete File Guide - Histology Pipeline

**Comprehensive guide explaining every file in the repository**
**Last Updated**: December 2024
**Version**: 1.0

---

## 📑 Table of Contents

1. [Root Directory Files](#root-directory-files)
2. [Documentation Files](#documentation-files)
3. [Source Code - Core Pipeline](#source-code---core-pipeline)
4. [Source Code - Analysis](#source-code---analysis)
5. [Source Code - Utilities](#source-code---utilities)
6. [Source Code - Validation](#source-code---validation)
7. [Configuration Files](#configuration-files)
8. [Shell Scripts](#shell-scripts)
9. [Production Scripts](#production-scripts)
10. [Archive](#archive)

---

## 🏠 Root Directory Files

### Environment & Dependencies

#### `environment.yml`
**Purpose**: Conda environment specification
**What it does**: Defines all Python packages and dependencies needed for the pipeline
**Key contents**:
- Python 3.10 base
- Scientific stack: NumPy, Pandas, SciPy, scikit-image, scikit-learn
- Deep learning: PyTorch 2.0.1, TensorFlow 2.13.0
- GPU support: CUDA 11.8, cuDNN
- Segmentation: Cellpose 3.0.1, StarDist 0.8.5
- Analysis: UMAP 0.5.3, SimpleITK >=2.2.0
- Visualization: Matplotlib, seaborn

**How to use**:
```bash
conda env create -f environment.yml
conda activate histology-pipeline
```

#### `requirements.txt`
**Purpose**: Pip-based dependency list
**What it does**: Alternative to conda for installing Python packages
**When to use**: If you prefer pip over conda, or for Docker/CI environments

**How to use**:
```bash
pip install -r requirements.txt
```

### License & Attribution

#### `LICENSE`
**Purpose**: Software license (if applicable)
**What it does**: Defines usage rights and restrictions
**Note**: Check with Principal Investigator for specific license choice

---

## 📚 Documentation Files

### Main Documentation

#### `README.md` (12 KB)
**Purpose**: Primary project documentation and entry point
**What it covers**:
- Project overview and capabilities
- **Dual segmentation** (Cellpose vs StarDist) - prominently featured
- Quick start guide
- Directory structure
- Output files explanation
- Supported slide types (H&E, CD3, GFAP, IBA1, NF, PGP9.5)
- Configuration overview
- Performance metrics
- Links to all other documentation

**Who should read**: Everyone (first stop for new users)

**Key sections**:
- Lines 43-67: Dual segmentation comparison table
- Lines 176-186: Output file format (final CSV columns)
- Lines 315-339: Quick reference commands

---

#### `USER_GUIDE.md` (16 KB)
**Purpose**: Comprehensive guide for end users
**What it covers**:
- Step-by-step workflows
- How to run pipeline on single slides
- Batch processing instructions
- Understanding output files
- Common issues & solutions
- Typical workflows (analyze patient samples, compare diseased vs healthy, IHC quantification)
- Quality control checks

**Who should read**: Lab members who want to analyze slides without coding

**Key sections**:
- Lines 72-112: Quick Start
- Lines 117-218: Output directory structure
- Lines 220-293: Understanding key output files (final CSV, UMAP, feature maps)
- Lines 297-397: Common issues & troubleshooting
- Lines 401-535: Typical workflows with code examples

**Example workflow covered**:
```python
# Workflow 3: IHC Marker Analysis (lines 459-489)
# Shows how to quantify CD3+ T cells step-by-step
```

---

#### `DEVELOPER_GUIDE.md` (34 KB)
**Purpose**: Technical architecture and development guide
**What it covers**:
- Complete system architecture
- Code organization principles
- Pipeline data flow (detailed diagram)
- Key design patterns (stateless processing, configuration-driven, etc.)
- How to add new features (with complete examples)
- Configuration system deep dive
- Testing & validation framework
- Performance optimization strategies
- Common development tasks

**Who should read**: Developers extending or modifying the pipeline

**Key sections**:
- Lines 40-88: Architecture overview and technology stack
- Lines 170-290: Complete pipeline flow diagram
- Lines 292-379: Key design patterns explained
- Lines 381-477: Adding new features (complete example: local entropy)
- Lines 479-592: Configuration system
- Lines 777-920: Common development tasks

**Example covered** (lines 381-470):
- Complete walkthrough of adding "Local Entropy" feature
- Shows code placement, configuration updates, validation

---

#### `HANDOFF_NOTES.md` (20 KB)
**Purpose**: Project status and transition guide for new developer
**What it covers**:
- Executive summary (what's done, what's pending)
- Complete feature list with status
- Current validation status (H&E complete, IHC pending)
- Pending work with priorities
- Known issues and limitations
- Key design decisions with rationale
- Quick start guide for new developer (Day 1, Week 1 tasks)
- Resources and contacts

**Who should read**: New developer taking over the project

**Key sections**:
- Lines 11-222: What's complete (comprehensive list)
- Lines 226-320: Pending work (IHC validation, liver testing, HistoVision comparison)
- Lines 324-357: Known issues (all minor, none blocking)
- Lines 361-439: Key design decisions explained
- Lines 715-780: Quick start for new developer (day-by-day plan)

**Critical info**:
- Lines 226-262: Priority 1 - Complete IHC validation (immediate task)
- Lines 264-294: Priority 2 - Liver sample testing
- Lines 296-320: Priority 3 - HistoVision comparison

---

#### `VALIDATION_STATUS.md` (19 KB)
**Purpose**: Detailed validation tracking for all metrics and slide types
**What it covers**:
- Validation completion status matrix (by slide type × metric)
- Detailed validation protocols for each metric
- H&E validation results (complete)
- Pending validations (IHC, liver, HistoVision)
- Acceptance criteria for each validation
- Validation log

**Who should read**: Person conducting validation, Principal Investigator

**Key sections**:
- Lines 13-28: Summary table showing overall progress
- Lines 34-132: Pipeline execution validation (per slide type)
- Lines 136-202: Segmentation quality validation
- Lines 206-284: Morphology features validation
- Lines 288-362: Density features validation
- Lines 366-432: Coherency features validation
- Lines 436-527: IHC brown stain features validation (pending)
- Lines 531-590: Filament analysis validation (pending)
- Lines 594-669: UMAP clustering validation
- Lines 673-731: HistoVision comparison protocol

**Example validation protocol** (lines 206-250):
```
H&E Morphology - COMPLETE ✅
- Feature maps generated
- Visual validation passed
- Statistical checks passed
- Outlier filtering implemented
```

---

#### `GETTING_STARTED.md` (8 KB)
**Purpose**: Installation and environment setup guide
**What it covers**:
- Prerequisites
- Conda environment setup
- Dependency installation
- GPU configuration
- First-time setup verification
- Testing the installation

**Who should read**: Anyone setting up the pipeline for the first time

---

#### `PRODUCTION_PIPELINE_GUIDE.md` (11 KB)
**Purpose**: HPC/SLURM deployment and batch processing guide
**What it covers**:
- SLURM job submission
- Resource allocation per slide type
- Batch processing strategies
- Monitoring jobs
- Troubleshooting HPC issues

**Who should read**: Users running on HPC clusters

---

#### `DELIVERY_CHECKLIST.md` (10 KB)
**Purpose**: Pre-deployment verification checklist
**What it covers**:
- Complete feature list with checkboxes
- Pre-deployment tasks
- Deliverables checklist
- Success criteria
- Known limitations
- Post-delivery maintenance

**Who should read**: Person responsible for final deployment

---

#### `DELIVERY_PACKAGE_README.md` (9 KB)
**Purpose**: Overview of the delivery package for Principal Investigator
**What it covers**:
- Package contents summary
- Quick start for reviewers
- Current status overview
- Next steps for new developer
- Quality assurance summary

**Who should read**: Principal Investigator, stakeholders receiving the package

---

#### `CODE_AUDIT_REPORT.md` (11 KB)
**Purpose**: Final code quality verification report
**What it covers**:
- Python code verification (all 31 scripts)
- Shell script verification (all 16 scripts)
- Configuration validation
- Code completeness check (zero TODO/FIXME)
- Import verification
- Documentation verification
- Testing status
- Quality assurance summary

**Who should read**: Quality assurance, project reviewers

**Key findings**:
- Lines 11-108: All Python scripts verified ✅
- Lines 112-149: All shell scripts verified ✅
- Lines 153-170: Configuration verified ✅
- Lines 174-185: Zero TODO/FIXME markers ✅
- Lines 435-472: Final verdict: PRODUCTION READY ✅

---

### Supporting Documentation

#### `MEETING_SUMMARY.txt`
**Purpose**: Notes from October 2024 meeting
**What it contains**: Historical project decisions and requirements
**Relevance**: Context for why certain design choices were made

---

## 💻 Source Code - Core Pipeline

All core pipeline scripts located in `src/core/`

### Step 00: Preview Generation

#### `00_preview.py` (4.4 KB, ~140 lines)
**Purpose**: Generate slide thumbnail and preview panel
**Inputs**: Whole slide image (.svs file)
**Outputs**:
- `{slide}_thumb.jpg` - Low-resolution thumbnail
- `panel_{slide}_preview.png` - 3-panel preview (overview + tissue + zoom)

**How it works**:
1. Opens slide with OpenSlide
2. Detects tissue vs background using thresholding
3. Generates thumbnail at lowest resolution
4. Creates 3-panel figure with matplotlib
5. Saves outputs to `results/{slide}/preview/`

**Key functions**:
- `generate_preview()` - Main entry point
- Lines 50-80: Tissue detection logic
- Lines 85-120: Panel creation

**When to run**: First step of pipeline (always)

**Example usage**:
```bash
python src/core/00_preview.py slide_name
```

---

### Step 01: Tissue Segmentation

#### `01_tissue_mask.py` (6.2 KB, ~210 lines)
**Purpose**: Create binary mask separating tissue from background
**Inputs**: Whole slide image
**Outputs**: `{slide}_tissue_mask.png` - Binary mask (white=tissue, black=background)

**How it works**:
1. Load slide at low resolution (level 2-3)
2. Convert RGB to HSV color space
3. Apply adaptive thresholding based on saturation channel
4. Morphological operations (erosion → dilation) to clean up
5. **Multi-component preservation** - keeps ALL tissue pieces (not just largest)
6. Save mask to `results/{slide}/masks/`

**Key functions**:
- `create_tissue_mask()` - Main segmentation logic
- Lines 45-78: HSV-based thresholding
- Lines 82-105: Morphological cleanup
- Lines 110-145: Multi-component handling

**Algorithm details**:
- HSV saturation channel used (better than RGB for histology)
- Adaptive threshold: `mean(saturation) - 0.1 * std(saturation)`
- Morphological kernel size: 11×11 pixels (adjustable in code)

**When to run**: Step 2 of pipeline (after preview)

**Example usage**:
```bash
python src/core/01_tissue_mask.py slide_name
```

---

### Step 02: Tiling

#### `02_tile.py` (4.8 KB, ~165 lines)
**Purpose**: Extract tiles from slide for processing
**Inputs**: Whole slide image + tissue mask
**Outputs**:
- `results/{slide}/tiles/tile_*.png` - Individual tile images
- `results/{slide}/tiles/tiles.json` - Metadata (coordinates, level, dimensions)

**How it works**:
1. Determine optimal level (based on slide dimensions)
2. Create 1024×1024 pixel tiles with 128px overlap
3. Filter tiles using tissue mask (skip background-only tiles)
4. Extract tiles with OpenSlide `read_region()`
5. Save tiles as PNG images
6. Save metadata JSON for reassembly

**Key parameters**:
- Tile size: 1024×1024 pixels
- Overlap: 128 pixels (12.5%)
- Tissue threshold: >10% tissue to keep tile

**Key functions**:
- `tile_slide()` - Main tiling logic
- Lines 55-90: Level selection and grid calculation
- Lines 95-140: Tile extraction loop
- Lines 145-160: Metadata generation

**Why overlapping tiles**:
- Prevents nuclei at tile boundaries from being cut off
- Allows deduplication in segmentation step

**When to run**: Step 3 of pipeline

**Example usage**:
```bash
python src/core/02_tile.py slide_name
```

---

### Step 03: Nuclear Segmentation

#### `03_segment_cellpose.py` (15.3 KB, ~485 lines)
**Purpose**: Segment nuclei using Cellpose model
**Inputs**: Tiles from step 02
**Outputs**:
- `results/{slide}/cellpose/masks/` - Segmentation masks per tile
- `results/{slide}/features/{slide}_nuclei_features.csv` - Nuclear features

**How it works**:
1. Load Cellpose model (cyto model, diameter=8µm)
2. Process tiles in batches
3. For each tile:
   - Run Cellpose segmentation
   - Extract nuclear properties (regionprops)
   - Calculate features: area, centroid, bounding box, orientation
4. **Deduplication**: Remove duplicates at tile boundaries
   - Distance-based matching (within 5µm)
   - Area comparison (within 20%)
   - Inspired by HistoVision approach
5. Merge features from all tiles
6. Save to CSV

**Key features extracted** (per nucleus):
- `nucleus_id` - Unique identifier
- `tile_id` - Source tile
- `x`, `y` - Pixel coordinates
- `x_um`, `y_um` - Micron coordinates
- `area_px` - Area in pixels
- `area_um2` - Area in µm²
- `perimeter_px` - Perimeter
- `major_axis_length`, `minor_axis_length` - Shape descriptors
- `orientation` - Nucleus rotation angle
- `eccentricity` - How elongated (0=circle, 1=line)
- `solidity` - Convexity measure

**Key functions**:
- `segment_with_cellpose()` - Main segmentation
- `deduplicate_nuclei()` - Remove boundary duplicates (lines 250-310)
- `extract_features()` - Feature calculation (lines 315-380)

**Deduplication algorithm** (lines 250-310):
```python
# For each nucleus pair:
# 1. Check if distance < 5µm
# 2. If yes, check if area difference < 20%
# 3. If both true, mark as duplicate, keep one
```

**When to run**: Step 4 (choose Cellpose OR StarDist)

**Example usage**:
```bash
python src/core/03_segment_cellpose.py slide_name
```

---

#### `03_segment_stardist.py` (16.6 KB, ~520 lines)
**Purpose**: Segment nuclei using StarDist model
**Inputs**: Tiles from step 02
**Outputs**:
- `results/{slide}/stardist/masks/` - Segmentation masks
- `results/{slide}/features/{slide}_nuclei_features.csv` - Nuclear features

**How it works**:
Same as Cellpose but using StarDist 2D_versatile_he model

**Key differences from Cellpose**:
- **Model**: StarDist 2D_versatile_he (specialized for nuclei)
- **Batch processing**: Optimized for A100 GPU (batch_size=32)
- **Speed**: ~2x faster than Cellpose on GPU
- **Accuracy**: Better for dense, round nuclei (typical in histology)

**GPU optimization**:
- Batch size: 32 (A100 with 40-80GB VRAM)
- Adjust in config for other GPUs:
  - V100 (32GB): batch_size=24
  - RTX 3090 (24GB): batch_size=16
  - RTX 2080 Ti (11GB): batch_size=8

**When to run**: Step 4 (choose Cellpose OR StarDist)
**Recommended**: Use StarDist for most histology slides

**Example usage**:
```bash
python src/core/03_segment_stardist.py slide_name
```

---

### Step 04: Density Profiling

#### `04_density.py` (5.7 KB, ~190 lines)
**Purpose**: Calculate nuclear density at multiple radii
**Inputs**: `{slide}_nuclei_features.csv` from step 03
**Outputs**: `{slide}_nuclei_features_density.csv` - CSV with added density columns

**How it works**:
1. Load nuclear coordinates
2. For each nucleus at (x, y):
   - Count neighbors within radius r (default: 50, 100, 150 µm)
   - Correct for mask boundaries (avoid edge artifacts)
   - Calculate density = count / area
3. Add density columns to CSV:
   - `density_um2_r50.0`
   - `density_um2_r100.0`
   - `density_um2_r150.0`

**Key functions**:
- `calculate_density()` - Main density calculation (lines 45-125)
- `mask_correction()` - Edge artifact correction (lines 130-165)

**Algorithm** (lines 60-95):
```python
# For each nucleus i at (xi, yi):
#   For each radius r in [50, 100, 150]:
#     neighbors = count nuclei within distance r
#     area = π × r²
#     density[i, r] = neighbors / area
```

**Mask boundary correction** (lines 130-165):
- If nucleus near mask edge, reduce effective area
- Prevents artificially low density at tissue boundaries

**When to run**: Step 5 (after segmentation)

**Example usage**:
```bash
python src/core/04_density.py slide_name
```

---

### Step 05: Feature Enrichment

#### `05_features.py` (12.6 KB, ~410 lines)
**Purpose**: Add advanced features (coherency, local variance, RGB)
**Inputs**: `{slide}_nuclei_features_density.csv`
**Outputs**:
- `{slide}_nuclei_features_enriched.csv` - CSV with all features
- `results/{slide}/viz/overlay_*.jpg` - Visualization overlays

**How it works**:
1. **Coherency calculation** (nuclear alignment):
   - For each nucleus, extract local neighborhood (150µm radius)
   - Compute structure tensor from gradient images
   - Calculate eigenvalues λ1, λ2
   - Coherency = (λ1 - λ2)² / (λ1 + λ2)²
   - Range: 0 (random) to 1 (highly aligned)

2. **Local variance** (spatial heterogeneity):
   - For each feature (area, circularity, etc.)
   - Calculate standard deviation in 150µm neighborhood
   - Add as `{feature}_local_variance_150um`

3. **RGB color features**:
   - Sample RGB values at nucleus centroid
   - Add columns: `r`, `g`, `b`

4. **Coefficient of variation**:
   - CV = std / mean for each feature
   - Measures relative variability

**Key features added**:
- `coherency_150um` - Nuclear alignment (0-1)
- `area_px_local_variance_150um` - Area heterogeneity
- `circularity_local_variance_150um` - Shape heterogeneity
- `r`, `g`, `b` - RGB color values
- `area_px_cv_150um` - Coefficient of variation

**Key functions**:
- `calculate_coherency()` - Structure tensor method (lines 65-145)
- `calculate_local_variance()` - Spatial statistics (lines 150-210)
- `extract_rgb_features()` - Color sampling (lines 215-265)

**Coherency algorithm** (lines 65-145):
```python
# Structure tensor approach:
# 1. Compute image gradients (Sobel)
# 2. Calculate structure tensor: [Ixx Ixy; Ixy Iyy]
# 3. Find eigenvalues λ1, λ2
# 4. Coherency = (λ1 - λ2)² / (λ1 + λ2)²
```

**When to run**: Step 6 (after density)

**Example usage**:
```bash
python src/core/05_features.py slide_name
```

---

#### `05b_ihc_intensity.py` (11.5 KB, ~370 lines)
**Purpose**: Measure IHC marker intensity (for IHC slides only)
**Inputs**: `{slide}_nuclei_features_enriched.csv`
**Outputs**: `{slide}_nuclei_features_enriched.csv` - Updated with intensity columns

**How it works**:
1. For each nucleus:
   - Expand nucleus mask by 15% (perinuclear region)
   - Sample pixel intensities in expanded region
   - Calculate statistics: mean, max, percentiles
2. Add columns:
   - `marker_intensity_mean` - Average intensity
   - `marker_intensity_max` - Maximum intensity
   - `marker_intensity_median` - Median intensity
   - `perinuclear_intensity_mean` - Intensity in perinuclear ring

**Perinuclear expansion**:
- Rationale: IHC staining often strongest in cytoplasm around nucleus
- Expansion factor: 1.15 (15% larger radius)
- Configurable in `configs/slide_config.yaml`

**Key functions**:
- `measure_ihc_intensity()` - Main measurement (lines 55-180)
- `expand_mask()` - Perinuclear dilation (lines 185-220)

**When to run**: Step 6b (only for IHC slides, after step 05)

**Example usage**:
```bash
python src/core/05b_ihc_intensity.py slide_name
```

---

### Step 06: Quality Control

#### `06_qc.py` (2.0 KB, ~65 lines)
**Purpose**: Generate QC summary and statistics
**Inputs**: `{slide}_nuclei_features_enriched.csv`
**Outputs**:
- `results/{slide}/qc/qc_summary.json` - Statistics and metrics
- Console output with summary stats

**How it works**:
1. Load final feature CSV
2. Calculate summary statistics:
   - Total nuclei count
   - Mean/median/std for key features
   - Outlier counts
   - Density distribution
3. Save to JSON
4. Print summary to console

**QC metrics**:
- Nucleus count (should be >1000 for good coverage)
- Mean area (should be ~250 pixels for 8µm nuclei)
- Density range (check for reasonable values)
- Outlier percentage (should be <5%)

**When to run**: Step 7 (final core pipeline step)

**Example usage**:
```bash
python src/core/06_qc.py slide_name
```

---

## 🔬 Source Code - Analysis

All analysis scripts located in `src/analysis/`

### Step 07: IHC Brown Stain Quantification

#### `07_ihc_brown_stain.py` (10.9 KB, ~350 lines)
**Purpose**: Detect and quantify DAB brown staining (IHC markers)
**Inputs**:
- Whole slide image
- `{slide}_nuclei_features_enriched.csv`
**Outputs**:
- `{slide}_with_brown.csv` - CSV with brown stain columns
- `results/{slide}/brown_stain/brown_stain_overlay.jpg` - Visualization
- `results/{slide}/brown_stain/brown_density_*.jpg` - Density maps

**How it works**:
1. **Color deconvolution** (H-DAB separation):
   - Use Ruifrok & Johnston (2001) deconvolution matrix
   - Separate Hematoxylin (blue) and DAB (brown) channels
   - Extract DAB intensity image

2. **Brown pixel detection**:
   - Apply threshold to DAB channel
   - Threshold varies by marker (configured in slide_config.yaml):
     - CD3: 0.08
     - GFAP: 0.12 (more stringent)
     - IBA1: 0.08

3. **Per-nucleus quantification**:
   - For each nucleus:
     - Count brown pixels in perinuclear region
     - Calculate `has_brown` (boolean)
     - Calculate `brown_intensity` (mean DAB value)
     - Calculate `brown_density_100um`, `brown_density_150um`

**Features added**:
- `has_brown` - Boolean (0 or 1)
- `brown_intensity` - DAB intensity (0-1)
- `brown_density_100um` - Fraction of neighbors that are brown
- `brown_density_150um` - Same at larger radius

**Key functions**:
- `color_deconvolution()` - H-DAB separation (lines 50-110)
- `detect_brown_pixels()` - Thresholding (lines 115-155)
- `quantify_per_nucleus()` - Feature calculation (lines 160-270)

**Color deconvolution matrix** (lines 60-75):
```python
# Ruifrok & Johnston (2001) H-DAB matrix
HDab_from_RGB = np.array([
    [0.650, 0.704, 0.286],  # Hematoxylin
    [0.268, 0.570, 0.776],  # DAB
    [0.000, 0.000, 0.000]   # Background
])
```

**When to run**: Step 8 (only for IHC slides)

**Example usage**:
```bash
python src/analysis/07_ihc_brown_stain.py slide_name
```

---

### Step 08: Specialized Analysis

#### `08_compare_segmenters.py` (9.7 KB, ~310 lines)
**Purpose**: Compare Cellpose vs StarDist segmentation side-by-side
**Inputs**:
- Results from both segmenters (must run both first)
- `results/{slide}/cellpose/` and `results/{slide}/stardist/`
**Outputs**:
- `results/{slide}/comparison/comparison_report.txt` - Metrics
- `results/{slide}/comparison/side_by_side.png` - Visual comparison
- Console output with statistics

**How it works**:
1. Load results from both segmenters
2. Compare metrics:
   - Nucleus count
   - Mean/median area
   - Circularity distribution
   - Processing time
   - Agreement rate (overlapping detections)
3. Generate side-by-side visualization
4. Print recommendation

**Metrics compared**:
- **Count agreement**: |count1 - count2| / mean(count1, count2)
- **Area correlation**: Pearson R between matched nuclei
- **Shape correlation**: Circularity, eccentricity comparison
- **Speed**: Processing time per tile

**Recommendation logic** (lines 270-300):
- If dense nuclei (>500/mm²): Recommend StarDist
- If irregular shapes: Recommend Cellpose
- If similar performance: Recommend StarDist (faster)

**When to run**: After running both segmenters (optional, for validation)

**Example usage**:
```bash
# First run both segmenters
python src/core/03_segment_cellpose.py slide_name
python src/core/03_segment_stardist.py slide_name

# Then compare
python src/analysis/08_compare_segmenters.py slide_name
```

---

#### `08_nfb_filament_analysis.py` (13.0 KB, ~420 lines)
**Purpose**: Analyze neurofilament filament architecture
**Inputs**: Whole slide image (neurofilament-stained)
**Outputs**:
- `results/{slide}/filaments/filaments.csv` - Per-filament data
- `results/{slide}/filaments/filament_summary.json` - Statistics
- `results/{slide}/filaments/filament_visualizations/` - Traced filaments

**How it works**:
1. **Preprocessing**:
   - Convert to grayscale
   - Enhance contrast
   - Apply median filter (noise reduction)

2. **Filament detection**:
   - Adaptive thresholding
   - Morphological closing (connect breaks)
   - Skeletonization (reduce to 1-pixel centerlines)

3. **Filament tracing**:
   - Find connected components in skeleton
   - Trace each filament path
   - Measure length, orientation, curvature

4. **Branch point detection**:
   - Find pixels with >2 neighbors in skeleton
   - Count branches per filament

**Features extracted per filament**:
- `filament_id` - Unique ID
- `length_px`, `length_um` - Filament length
- `orientation` - Mean angle (0-180°)
- `n_branch_points` - Number of branches
- `curvature` - How straight (0=straight, 1=very curved)
- `intensity_mean` - Average pixel intensity along filament

**Key functions**:
- `detect_filaments()` - Thresholding + morphology (lines 60-130)
- `skeletonize()` - Reduce to centerlines (lines 135-165)
- `trace_filaments()` - Path extraction (lines 170-260)
- `detect_branches()` - Branch point finding (lines 265-295)

**Note**: This runs in **parallel** with nuclear analysis (separate from Steps 00-06)

**When to run**: For neurofilament-stained slides only

**Example usage**:
```bash
python src/analysis/08_nfb_filament_analysis.py slide_name
```

---

### Step 09: UMAP Clustering

#### `09_umap_clustering.py` (21.3 KB, ~680 lines)
**Purpose**: Dimensionality reduction and clustering
**Inputs**:
- `{slide}_final.csv` (or `{slide}_with_brown.csv` for IHC)
- Feature list from `configs/slide_config.yaml`
**Outputs**:
- `{slide}_final.csv` - CSV with UMAP and cluster columns
- `results/{slide}/clustering/umap_clusters.png` - UMAP scatter plot
- `results/{slide}/clustering/cluster_spatial.png` - Spatial overlay
- `results/{slide}/clustering/cluster_features.png` - Feature distributions
- `results/{slide}/clustering/cluster_statistics.csv` - Per-cluster stats
- `results/{slide}/clustering/umap_clusters.pkl` - Saved UMAP object

**How it works**:
1. **Feature selection**:
   - Load feature names from config
   - For H&E: morphology, density, coherency
   - For IHC: + brown stain features

2. **Preprocessing** (optional):
   - PCA if >10 features (recommended)
   - StandardScaler normalization

3. **UMAP embedding**:
   - n_neighbors: 15 (typical)
   - min_dist: 0.1 (allows tight clusters)
   - metric: euclidean
   - n_components: 2 (for visualization)

4. **Clustering**:
   - BIRCH algorithm (scalable)
   - n_clusters from config (default: 20)
   - Assign cluster labels

5. **Visualization**:
   - Scatter plot with cluster colors
   - Spatial overlay on tissue image
   - Feature distributions per cluster

**Features used** (configurable):
- H&E: area, circularity, eccentricity, density, coherency, RGB
- IHC: + brown_intensity, brown_density

**Key functions**:
- `load_and_prepare_features()` - Feature selection (lines 70-130)
- `run_umap()` - UMAP embedding (lines 135-180)
- `cluster_birch()` - Clustering (lines 185-220)
- `visualize_umap()` - Plotting (lines 225-340)
- `spatial_overlay()` - Overlay on tissue (lines 345-420)

**UMAP parameters** (lines 135-165):
```python
umap_model = umap.UMAP(
    n_neighbors=15,      # Local structure preservation
    min_dist=0.1,        # Minimum distance between points
    n_components=2,      # 2D for visualization
    metric='euclidean',  # Distance metric
    random_state=42      # Reproducibility
)
```

**When to run**: Step 10 (after all feature extraction)

**Example usage**:
```bash
python src/analysis/09_umap_clustering.py slide_name
```

---

### Step 10: Combined UMAP by Stain Type

#### `10_separate_umaps.py` (19.8 KB, ~630 lines)
**Purpose**: Generate combined UMAP for multiple slides of same stain type
**Inputs**: Multiple final CSVs (all H&E slides, or all CD3 slides, etc.)
**Outputs**:
- `combined_results/{stain_type}_combined_umap.png` - Multi-slide UMAP
- `combined_results/{stain_type}_violin_plots.png` - Feature distributions
- `combined_results/{stain_type}_summary.csv` - Cross-slide statistics

**How it works**:
1. **Group slides by type**:
   - Detect slide type for each CSV
   - Group: all H&E together, all CD3 together, etc.

2. **Concatenate features**:
   - Merge all slides of same type
   - Add slide_id column to track origin

3. **Combined UMAP**:
   - Run UMAP on concatenated data
   - Color by slide_id to see if slides cluster together

4. **Statistical comparison**:
   - Violin plots per slide
   - ANOVA to test for slide differences
   - Summary statistics per slide

**Use cases**:
- Compare batch effects across slides
- Identify slide-specific patterns
- Validate consistent processing

**Key functions**:
- `group_by_stain_type()` - Slide grouping (lines 50-95)
- `concatenate_features()` - Merge CSVs (lines 100-145)
- `combined_umap()` - Multi-slide UMAP (lines 150-230)
- `violin_plots()` - Statistical comparison (lines 235-315)

**When to run**: After all individual slides processed (batch post-processing)

**Example usage**:
```bash
python src/analysis/10_separate_umaps.py
# Processes all slides in results/ directory
```

---

### Step 11: Multi-Modal Registration

#### `11_combined_umap.py` (47.8 KB, ~1520 lines)
**Purpose**: Cross-stain spatial registration and multi-modal analysis
**Inputs**:
- H&E slide features
- IHC slide features (same tissue section)
**Outputs**:
- `combined_results/multimodal/registered_nuclei.csv` - Matched nuclei
- `combined_results/multimodal/registration_overlay.png` - Visual alignment
- `combined_results/multimodal/cross_stain_umap.png` - Combined UMAP

**How it works**:
1. **Image registration** (SimpleITK):
   - Load H&E and IHC images
   - Detect landmarks/features
   - Compute transformation (affine or deformable)
   - Apply transformation to align slides

2. **Nuclear matching**:
   - For each H&E nucleus at (x, y)
   - Find nearest IHC nucleus in registered space
   - Within distance threshold (5µm)
   - Match if distance + area agreement

3. **Feature fusion**:
   - Combine H&E features with IHC features
   - Create multi-modal feature vector
   - Run UMAP on combined features

4. **Cross-stain analysis**:
   - Compare morphology vs marker expression
   - Identify marker+ cell populations
   - Spatial co-localization analysis

**Key functions**:
- `register_images()` - SimpleITK registration (lines 80-210)
- `match_nuclei()` - Spatial matching (lines 215-295)
- `fuse_features()` - Feature combination (lines 300-365)
- `cross_stain_umap()` - Multi-modal UMAP (lines 370-480)

**Requirements**:
- SimpleITK library (now in requirements.txt)
- Paired H&E + IHC slides from same tissue section

**When to run**: Advanced analysis (optional, for multi-modal studies)

**Example usage**:
```bash
python src/analysis/11_combined_umap.py HE_slide_name IHC_slide_name
```

---

## 🛠️ Source Code - Utilities

All utility scripts located in `src/utils/`

#### `slide_detector.py` (4.5 KB, ~145 lines)
**Purpose**: Automatically detect slide type from filename
**Inputs**: Slide filename (string)
**Outputs**: Slide type string (e.g., "H&E", "IHC_CD3")

**How it works**:
1. Parse filename
2. Pattern matching with regex:
   - `CD3` → "IHC_CD3"
   - `GFAP` → "IHC_GFAP"
   - `IBA1` or `Iba1` → "IHC_IBA1"
   - `NF` → "IHC_NF"
   - `PGP9-5` or `PGP9.5` → "IHC_PGP95"
   - `H&E` or `HE-` → "H&E"
3. Return detected type (default: "H&E" if no match)

**Patterns** (lines 35-85):
```python
if re.search(r'CD3', filename, re.IGNORECASE):
    return 'IHC_CD3'
elif re.search(r'GFAP', filename, re.IGNORECASE):
    return 'IHC_GFAP'
# ... etc
```

**Used by**:
- `run_adaptive_pipeline.sh` - Choose processing steps
- `config_loader.py` - Load appropriate config section

**Example usage**:
```python
from src.utils.slide_detector import detect_slide_type
slide_type = detect_slide_type('CD3-tumor-S25.svs')
# Returns: 'IHC_CD3'
```

---

#### `config_loader.py` (3.4 KB, ~110 lines)
**Purpose**: Load slide-type-specific configuration
**Inputs**: Slide type string
**Outputs**: Dictionary with configuration parameters

**How it works**:
1. Read `configs/slide_config.yaml`
2. Parse YAML to dictionary
3. Return config section for requested slide type
4. Fallback to "default" or "H&E" if type not found

**Configuration structure**:
```yaml
H&E:
  segmentation: {...}
  density_radii_um: [...]
  clustering: {...}

IHC_CD3:
  segmentation: {...}
  brown_detection: {...}
  clustering: {...}
```

**Key functions**:
- `load_config(slide_type)` - Main loader (lines 25-65)
- `get_parameter(config, key, default)` - Safe parameter access (lines 70-90)

**Error handling**:
- If file not found: Print warning, return default config
- If slide type not found: Print warning, fallback to H&E

**Example usage**:
```python
from src.utils.config_loader import load_config
config = load_config('IHC_CD3')
diameter = config['segmentation']['diameter_um']  # 8.0
threshold = config['brown_detection']['threshold']  # 0.08
```

---

## ✅ Source Code - Validation

All validation scripts located in `src/validation/`

#### `generate_feature_maps.py` (10.3 KB, ~330 lines)
**Purpose**: Create 3-panel validation visualizations for all features
**Inputs**:
- Slide name
- `{slide}_final.csv` with all features
**Outputs**: `results/{slide}/feature_maps/*_validation.png` - One per feature

**How it works**:
1. For each feature in CSV:
   - **Left panel**: Feature colormap (heatmap)
   - **Middle panel**: Original H&E image
   - **Right panel**: Overlay (feature map on H&E)
2. Save as PNG

**3-panel layout**:
```
[Colormap]  [H&E Image]  [Overlay]
```

**Features visualized**:
- All morphology features (area, circularity, eccentricity, etc.)
- All density features (density_r50, density_r100, density_r150)
- Coherency
- Local variance features
- Brown stain features (if applicable)

**Visual QC use**:
- Does area map show larger nuclei as expected? ✓
- Does density map show crowded regions? ✓
- Does coherency map show aligned structures (vessels, tracts)? ✓
- Are there obvious artifacts? ✗

**Key functions**:
- `create_feature_map()` - Colormap generation (lines 50-125)
- `overlay_on_he()` - Transparent overlay (lines 130-180)
- `generate_all_maps()` - Batch processing (lines 185-280)

**Example usage**:
```bash
python src/validation/generate_feature_maps.py slide_name
# Creates ~50 validation images in results/slide_name/feature_maps/
```

---

#### `test_coherency_synthetic.py` (6.1 KB, ~195 lines)
**Purpose**: Validate coherency metric on synthetic data
**Inputs**: None (creates synthetic images)
**Outputs**:
- Console output with coherency values
- `validation/test_coherency_*.png` - Synthetic test images

**How it works**:
1. **Create synthetic patterns**:
   - **Aligned pattern**: Horizontal lines (expected coherency ≈ 1.0)
   - **Random pattern**: Random dots (expected coherency ≈ 0.0)
   - **Partially aligned**: Two groups at different angles

2. **Calculate coherency**: Run same algorithm as production code

3. **Verify results**:
   - Aligned: coherency > 0.9 ✓
   - Random: coherency < 0.1 ✓
   - Partial: 0.4 < coherency < 0.6 ✓

**Test patterns** (lines 35-100):
```python
# Aligned (horizontal lines)
image[::10, :] = 255  # Every 10th row

# Random (noise)
image = np.random.randint(0, 255, size=(512, 512))

# Two groups (45° and 135°)
# ... (lines 75-95)
```

**Acceptance criteria**:
- Aligned must be > 0.9
- Random must be < 0.1
- Two-group must be moderate (0.4-0.6)

**Example usage**:
```bash
python src/validation/test_coherency_synthetic.py
# Output:
# Aligned pattern: coherency = 0.97 ✓
# Random pattern: coherency = 0.03 ✓
# Two groups: coherency = 0.52 ✓
```

---

#### `test_features.py` (6.1 KB, ~195 lines)
**Purpose**: Unit tests for feature extraction functions
**Inputs**: Synthetic test data
**Outputs**: Console output (pass/fail for each test)

**Tests included**:
1. **Area calculation**: Known shapes with known areas
2. **Circularity**: Circle should be 1.0, square should be ~0.785
3. **Eccentricity**: Circle=0, line=1
4. **Density calculation**: Known grid patterns
5. **RGB extraction**: Known color patches

**Example tests** (lines 40-140):
```python
def test_circularity():
    # Circle with radius 10
    circle = create_circle(radius=10)
    circ = calculate_circularity(circle)
    assert abs(circ - 1.0) < 0.01  # Should be 1.0

def test_density():
    # 10×10 grid with 10µm spacing
    expected_density = 0.01  # nuclei/µm²
    calculated = calculate_density(grid)
    assert abs(calculated - expected) < 0.001
```

**Example usage**:
```bash
python src/validation/test_features.py
# Output:
# test_area: PASS ✓
# test_circularity: PASS ✓
# test_eccentricity: PASS ✓
# test_density: PASS ✓
# test_rgb: PASS ✓
```

---

#### `validate_coherency.py` (4.5 KB, ~145 lines)
**Purpose**: Validate coherency on real histology images
**Inputs**: Slide image
**Outputs**: Console output + annotated images

**How it works**:
1. Load real histology image
2. Select regions with known properties:
   - White matter tract (high coherency expected)
   - Gray matter (low coherency expected)
   - Blood vessel (high coherency expected)
3. Calculate coherency in each region
4. Verify matches expectations

**Expected values** (lines 80-115):
- White matter: coherency > 0.7
- Gray matter: coherency < 0.3
- Blood vessel: coherency > 0.8

**Example usage**:
```bash
python src/validation/validate_coherency.py slide_name
# Output:
# White matter region: coherency = 0.85 ✓
# Gray matter region: coherency = 0.15 ✓
# Blood vessel: coherency = 0.92 ✓
```

---

## ⚙️ Configuration Files

### Main Configuration

#### `configs/slide_config.yaml` (4.7 KB with comments)
**Purpose**: Centralized configuration for all slide types
**What it contains**:

**For each slide type** (H&E, IHC_CD3, IHC_GFAP, etc.):

1. **Segmentation parameters**:
   ```yaml
   segmentation:
     diameter_um: 8.0        # Expected nucleus diameter
     model: stardist         # cellpose or stardist
     batch_size: 32          # GPU batch size
     dedup_radius_um: 5.0    # Deduplication threshold
   ```

2. **Density parameters**:
   ```yaml
   density_radii_um: [50, 100, 150]  # Radii for density calculation
   ```

3. **Brown stain detection** (IHC only):
   ```yaml
   brown_detection:
     threshold: 0.08               # DAB intensity threshold
     neighborhood_radius: 1.5      # Perinuclear expansion factor
   ```

4. **Clustering parameters**:
   ```yaml
   clustering:
     n_clusters: 20
     features: [area_px, circularity, density_um2_r100.0, ...]
   ```

**Slide types configured**:
- H&E: Standard histology
- IHC_CD3: CD3+ T cells
- IHC_GFAP: GFAP+ astrocytes
- IHC_IBA1: IBA1+ microglia
- IHC_NF: Neurofilament
- IHC_PGP95: PGP9.5 neurons
- default: Fallback configuration

**How to modify**:
1. Open `configs/slide_config.yaml`
2. Find slide type section
3. Modify parameters
4. Save file
5. No code changes needed!

**Example**:
```yaml
# To change nucleus diameter for CD3:
IHC_CD3:
  segmentation:
    diameter_um: 7.5  # Changed from 8.0
```

---

## 📜 Shell Scripts

All orchestration scripts located in `scripts/`

### Main Entry Points

#### `scripts/run_adaptive_pipeline.sh` (6.4 KB)
**Purpose**: Main entry point - auto-detects slide type and runs appropriate steps
**Inputs**: Path to .svs file
**Outputs**: All results in `results/{slide_name}/`

**What it does**:
1. Detect slide type (H&E vs IHC)
2. Run steps 00-06 (core pipeline)
3. If IHC: Also run 05b, 07 (brown stain)
4. If NF: Also run 08_nfb_filament_analysis
5. Run step 09 (UMAP clustering)
6. Generate validation visualizations

**Usage**:
```bash
./scripts/run_adaptive_pipeline.sh data/raw_slides/CD3-S25.svs
```

**Key logic** (lines 35-95):
```bash
# Detect slide type
SLIDE_TYPE=$(python -c "from src.utils.slide_detector import detect_slide_type; print(detect_slide_type('$SLIDE_PATH'))")

# Run core steps (00-06)
python src/core/00_preview.py "$SLIDE_NAME"
python src/core/01_tissue_mask.py "$SLIDE_NAME"
# ... etc

# If IHC, add IHC steps
if [[ "$SLIDE_TYPE" == IHC* ]]; then
    python src/core/05b_ihc_intensity.py "$SLIDE_NAME"
    python src/analysis/07_ihc_brown_stain.py "$SLIDE_NAME"
fi
```

---

#### `scripts/run_one_slide_stardist.sh` (8.3 KB)
**Purpose**: Run complete pipeline using StarDist segmentation
**Inputs**: Path to .svs file
**Outputs**: All results in `results/{slide_name}/`

**What it does**:
Same as adaptive pipeline but:
- Always uses StarDist (not Cellpose)
- Runs all steps including IHC analysis (if applicable)
- More comprehensive than `run_one_slide.sh`

**Usage**:
```bash
./scripts/run_one_slide_stardist.sh data/raw_slides/slide.svs
```

**Recommended**: Use this for most histology slides (StarDist is faster and more accurate)

---

#### `scripts/run_one_slide.sh` (4.0 KB)
**Purpose**: Run core pipeline using Cellpose segmentation
**Inputs**: Path to .svs file
**Outputs**: Basic results (no IHC analysis)

**What it does**:
- Runs steps 00-06 only
- Uses Cellpose for segmentation
- No brown stain or advanced analysis

**Usage**:
```bash
./scripts/run_one_slide.sh data/raw_slides/slide.svs
```

**When to use**: Testing Cellpose, or for irregular cell types

---

#### `scripts/run_all_by_type.sh` (6.5 KB)
**Purpose**: Batch process all slides in a directory
**Inputs**: Directory with .svs files
**Outputs**: Results for all slides + summary report

**What it does**:
1. Find all .svs files in input directory
2. Detect slide type for each
3. Group by type (H&E, CD3, GFAP, etc.)
4. Process each slide with appropriate pipeline
5. Generate summary statistics
6. Create combined UMAP per stain type

**Usage**:
```bash
./scripts/run_all_by_type.sh data/raw_slides results
```

**Outputs**:
- `results/{slide}/` for each slide
- `results/summary_report.csv` - Aggregate statistics
- `results/batch_processing.log` - Processing log

**Key logic** (lines 50-140):
```bash
# Group slides by type
for slide in *.svs; do
    type=$(detect_slide_type "$slide")
    group_slides[$type]+=" $slide"
done

# Process each group
for type in "${!group_slides[@]}"; do
    for slide in ${group_slides[$type]}; do
        ./run_adaptive_pipeline.sh "$slide"
    done
done
```

---

#### `scripts/test_setup.sh` (4.8 KB)
**Purpose**: Validate environment and installation
**Inputs**: None
**Outputs**: Console output (pass/fail for each test)

**9 test categories**:
1. Directory structure
2. Core pipeline scripts exist
3. Analysis scripts exist
4. Utility modules exist
5. Configuration files exist
6. Shell scripts executable
7. Python imports work
8. Slide type detection works
9. Data availability

**Usage**:
```bash
./scripts/test_setup.sh
# Should output: ALL TESTS PASSED ✅
```

**Example output**:
```
✓ Test 1: Directory structure - PASS
✓ Test 2: Core pipeline scripts - PASS
✓ Test 3: Analysis scripts - PASS
✓ Test 4: Utility modules - PASS
✓ Test 5: Configuration - PASS
✓ Test 6: Shell scripts - PASS
✓ Test 7: Python imports - PASS
✓ Test 8: Slide type detection - PASS
✓ Test 9: Slide availability - PASS (with warning)

ALL TESTS PASSED ✅
```

---

### Batch Processing Scripts

#### `scripts/batch_cellpose.sh` (1.9 KB)
**Purpose**: Run Cellpose on multiple slides
**Usage**: `./scripts/batch_cellpose.sh slide1.svs slide2.svs ...`

#### `scripts/batch_features.sh` (1.6 KB)
**Purpose**: Run feature extraction on multiple slides
**Usage**: `./scripts/batch_features.sh slide1 slide2 ...`

---

## 🚀 Production Scripts

### SLURM Submission

#### `submit_production_raw_slides.sh` (4.7 KB)
**Purpose**: Generate and submit SLURM jobs for 28 raw slides
**What it does**:
1. List all slides to process (hardcoded paths)
2. Detect slide type for each
3. Allocate resources based on type:
   - H&E: 1h, 64GB RAM, 1× A100
   - IHC: 1.5h, 64GB RAM, 1× A100
   - NF: 2h, 80GB RAM, 1× A100 (filament analysis overhead)
4. Create SLURM job script for each slide
5. Submit jobs to queue

**Usage**:
```bash
./submit_production_raw_slides.sh
# Creates jobs in logs_production/
# Submits to SLURM queue
```

**SLURM template** (lines 80-125):
```bash
#!/bin/bash
#SBATCH --job-name=histology_{slide}
#SBATCH --time={time_limit}
#SBATCH --mem={memory}
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs_production/{slide}_%j.out

module load anaconda3
conda activate histology-pipeline
./scripts/run_adaptive_pipeline.sh {slide_path}
```

---

#### `submit_production_HCC_slides.sh` (3.6 KB)
**Purpose**: Generate and submit SLURM jobs for 12 HCC slides
**Similar to above but for HCC dataset** (assumed H&E)

---

#### `run_combined_analysis.sh` (4.9 KB)
**Purpose**: Post-processing - run combined analysis after all slides complete
**What it does**:
1. Check all individual slides processed
2. Group slides by stain type
3. Run `10_separate_umaps.py` for each stain
4. Generate summary reports
5. Create combined visualizations

**Usage**:
```bash
# After all individual slides complete
./run_combined_analysis.sh
```

**Outputs**:
- `combined_results/H&E_combined_umap.png`
- `combined_results/CD3_combined_umap.png`
- ... (one per stain type)
- `combined_results/summary_statistics.csv`

---

### Monitoring & Cleanup

#### `monitor_production.sh` (3.3 KB)
**Purpose**: Monitor SLURM job status and progress
**Usage**: `./monitor_production.sh`

**Output**:
- Job queue status
- Completed jobs count
- Failed jobs (if any)
- Estimated time remaining

---

#### `cleanup_workspace.sh` (8.1 KB)
**Purpose**: Clean up temporary files and intermediate results
**Usage**: `./cleanup_workspace.sh [--dry-run]`

**What it removes**:
- Temporary tile images (keeps only results)
- Intermediate CSVs (keeps only final)
- Cache files
- Log files older than 30 days

---

#### `quick_cleanup.sh` (1.4 KB)
**Purpose**: Quick cleanup (removes only cache files)
**Usage**: `./quick_cleanup.sh`

---

### Utility Scripts

#### `test_single_slide_full.sh` (2.7 KB)
**Purpose**: Full pipeline test on single slide with verbose output
**Usage**: `./test_single_slide_full.sh slide.svs`

---

#### `create_review_package.sh` (9.3 KB)
**Purpose**: Create package for code review
**Usage**: `./create_review_package.sh`

---

#### `fix_script_paths.sh` (1.9 KB)
**Purpose**: Fix any broken path references in scripts
**Usage**: `./fix_script_paths.sh`

---

## 📦 Archive

#### `archive/` directory
**Contents**: Old/deprecated code (7 scripts)
**Purpose**: Preserve old implementations for reference
**Status**: Not used in production

**Files**:
- Old visualization scripts
- Deprecated batch processing scripts
- Early prototypes

**Note**: Safe to ignore for production use

---

## 🎯 Key Takeaways

### Most Important Files to Understand

**For End Users**:
1. `README.md` - Start here
2. `USER_GUIDE.md` - How to use
3. `scripts/run_adaptive_pipeline.sh` - Main command

**For Developers**:
1. `DEVELOPER_GUIDE.md` - Architecture
2. `src/core/03_segment_stardist.py` - Segmentation
3. `src/core/05_features.py` - Feature extraction
4. `configs/slide_config.yaml` - Configuration

**For Handoff**:
1. `HANDOFF_NOTES.md` - Current status
2. `VALIDATION_STATUS.md` - What's done/pending
3. `CODE_AUDIT_REPORT.md` - Quality verification

---

## 📝 File Count Summary

- **Python scripts**: 31 (core: 9, analysis: 6, utils: 2, validation: 4, archive: 7, backup: 3)
- **Shell scripts**: 16 (main: 5, batch: 2, production: 3, utility: 6)
- **Documentation**: 11 markdown files (159KB total)
- **Configuration**: 1 YAML file
- **Total lines of code**: ~6,400 (Python only)

---

**This guide provides complete documentation of every file in the repository. Use it as a reference for understanding code organization and functionality.**

**For questions about specific files, see the relevant documentation guide (USER, DEVELOPER, or HANDOFF).**
