# Developer Guide: Histology Image Analysis Pipeline

**For Developers** - Technical architecture, code organization, and extension guide

**Last Updated**: December 2024
**Version**: 1.0

---

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Code Organization](#code-organization)
3. [Pipeline Flow](#pipeline-flow)
4. [Key Design Patterns](#key-design-patterns)
5. [Adding New Features](#adding-new-features)
6. [Configuration System](#configuration-system)
7. [Testing & Validation](#testing--validation)
8. [Performance Optimization](#performance-optimization)
9. [Common Development Tasks](#common-development-tasks)
10. [Contributing Guidelines](#contributing-guidelines)

---

## 🏗️ Architecture Overview

### System Design Principles

1. **Modular Pipeline**: Each step is independent and can be run standalone
2. **Type-Aware Processing**: Automatic adaptation based on slide type (H&E, IHC)
3. **Configuration-Driven**: All parameters externalized to YAML
4. **Reproducible**: Same input → same output
5. **GPU-Optimized**: Batch processing for deep learning models
6. **HPC-Ready**: SLURM integration for large-scale processing

### Technology Stack

**Core Scientific Computing**:
- NumPy 1.24.3 - Array operations
- SciPy 1.11.1 - Scientific algorithms
- Pandas 2.0.3 - Data manipulation
- scikit-image 0.21.0 - Image processing
- scikit-learn 1.3.0 - Machine learning

**Deep Learning**:
- PyTorch 2.0.1 - Cellpose backend
- TensorFlow 2.13.0 - StarDist backend
- CUDA 11.8 - GPU acceleration

**Whole Slide Imaging**:
- OpenSlide 1.3.1 - WSI reading
- Pillow 10.0.0 - Image I/O

**Segmentation Models**:
- Cellpose 3.0.1 - Generalist segmentation
- StarDist 0.8.5 - Nuclear segmentation
- csbdeep 0.7.4 - StarDist dependency

**Visualization**:
- Matplotlib 3.7.2 - Plotting
- seaborn 0.12.2 - Statistical plots

**Dimensionality Reduction**:
- UMAP 0.5.3 - Manifold learning
- SimpleITK ≥2.2.0 - Image registration

---

## 📁 Code Organization

### Directory Structure

```
histology/
│
├── src/                          # Main source code
│   ├── core/                     # Core pipeline (00-06)
│   │   ├── 00_preview.py         # Thumbnail generation
│   │   ├── 01_tissue_mask.py     # Tissue segmentation
│   │   ├── 02_tile.py            # Slide tiling
│   │   ├── 03_segment_cellpose.py    # Cellpose segmentation
│   │   ├── 03_segment_stardist.py    # StarDist segmentation
│   │   ├── 04_density.py         # Density profiling
│   │   ├── 05_features.py        # Feature extraction
│   │   ├── 05b_ihc_intensity.py  # IHC intensity
│   │   ├── 06_qc.py              # Quality control
│   │   └── README.md
│   │
│   ├── analysis/                 # Advanced analysis (07-11)
│   │   ├── 07_ihc_brown_stain.py     # DAB quantification
│   │   ├── 08_compare_segmenters.py  # Segmenter comparison
│   │   ├── 08_nfb_filament_analysis.py   # Filament tracing
│   │   ├── 09_umap_clustering.py     # UMAP + clustering
│   │   ├── 10_separate_umaps.py      # Per-stain UMAP
│   │   ├── 11_combined_umap.py       # Multi-modal registration
│   │   └── README.md
│   │
│   ├── utils/                    # Utility modules
│   │   ├── slide_detector.py     # Auto slide type detection
│   │   ├── config_loader.py      # YAML config loading
│   │   └── README.md
│   │
│   └── validation/               # Testing & validation
│       ├── generate_feature_maps.py      # 3-panel visualizations
│       ├── test_coherency_synthetic.py   # Coherency validation
│       ├── test_features.py              # Feature tests
│       └── validate_coherency.py         # Coherency metric validation
│
├── scripts/                      # Shell orchestration
│   ├── run_adaptive_pipeline.sh  # Main entry point
│   ├── run_one_slide.sh          # Cellpose pipeline
│   ├── run_one_slide_stardist.sh # StarDist pipeline
│   ├── run_all_by_type.sh        # Batch processing
│   ├── test_setup.sh             # Environment validation
│   ├── batch_cellpose.sh         # Cellpose batching
│   └── batch_features.sh         # Feature batching
│
├── configs/                      # Configuration files
│   └── slide_config.yaml         # Per-slide-type parameters
│
├── data/                         # Data directory (user-created)
│   └── raw_slides/               # Input WSI files
│
├── results/                      # Output directory
│   └── <slide_name>/             # Per-slide results
│
├── archive/                      # Deprecated code
│
├── docs/                         # Documentation
│
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment
│
├── README.md                     # Project overview
├── GETTING_STARTED.md            # Setup guide
├── PRODUCTION_PIPELINE_GUIDE.md  # HPC guide
├── USER_GUIDE.md                 # End-user instructions
├── DEVELOPER_GUIDE.md            # This file
├── HANDOFF_NOTES.md              # Project status
├── VALIDATION_STATUS.md          # Validation documentation
└── DELIVERY_CHECKLIST.md         # Deployment checklist
```

---

## 🔄 Pipeline Flow

### Data Flow Diagram

```
Input: slide.svs
    ↓
┌─────────────────────────────────────────────────────┐
│ Step 00: Preview Generation                         │
│   - Load slide at low resolution                   │
│   - Detect tissue vs background                    │
│   - Generate thumbnail + 3-panel preview           │
│   Output: thumb.jpg, preview.png                   │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ Step 01: Tissue Mask                                │
│   - HSV color space conversion                     │
│   - Adaptive thresholding                          │
│   - Morphological cleanup (erosion + dilation)    │
│   - Multi-component preservation                   │
│   Output: tissue_mask.png                          │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ Step 02: Tiling                                     │
│   - Extract 1024×1024 tiles with 128px overlap     │
│   - Apply tissue mask filter                       │
│   - Downsample if slide is high-res               │
│   Output: tiles/*.png, tiles.json                  │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ Step 03: Nuclear Segmentation                       │
│   ┌───────────────┐      ┌──────────────────┐      │
│   │  Cellpose     │  OR  │   StarDist       │      │
│   │  - Diameter=8 │      │   - Batch=32     │      │
│   │  - Model: cyto│      │   - 2D_versatile │      │
│   └───────────────┘      └──────────────────┘      │
│                                                      │
│   - Batch process tiles                            │
│   - Extract nuclear masks + boundaries            │
│   - Deduplicate cross-tile nuclei                 │
│   Output: masks/, features CSV                     │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ Step 04: Density Profiling                          │
│   - For each nucleus at (x,y):                     │
│     - Count neighbors in radii: 50, 100, 150 µm   │
│     - Correct for mask boundaries                  │
│   - Calculate density (nuclei/µm²)                │
│   Output: *_density.csv                            │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ Step 05: Feature Enrichment                         │
│   - Coherency (structure tensor)                   │
│   - Local variance & CV                            │
│   - RGB color features                             │
│   - Morphology (area, circularity, etc.)          │
│   Output: *_enriched.csv, viz/overlays             │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ Step 05b: IHC Intensity (IHC slides only)           │
│   - Perinuclear expansion (15% radius)            │
│   - Marker intensity measurement                   │
│   - Color deconvolution preparation                │
│   Output: *_enriched.csv (updated)                 │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ Step 06: Quality Control                            │
│   - Generate QC summary panels                     │
│   - Statistics (mean, std, outliers)              │
│   - Validation metrics                             │
│   Output: qc/summary.json                          │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ Step 07: Brown Stain (IHC only)                     │
│   - H-DAB color deconvolution                      │
│   - Brown pixel detection                          │
│   - Per-nucleus brown intensity                    │
│   - Brown density calculation                      │
│   Output: *_with_brown.csv, brown_stain/           │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ Step 08: NFB Filaments (NF slides only)             │
│   - Separate from nuclear analysis                 │
│   - Skeletonization                                │
│   - Filament tracing                               │
│   - Branch point detection                         │
│   Output: filaments/filaments.csv                  │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ Step 09: UMAP Clustering                            │
│   - Load features from config                      │
│   - Optional PCA (if >10 features)                 │
│   - UMAP embedding (n_neighbors=15, min_dist=0.1) │
│   - BIRCH clustering                               │
│   - Generate visualizations                        │
│   Output: *_final.csv, clustering/                 │
└─────────────────────────────────────────────────────┘
    ↓
Final Output: results/<slide>/features/<slide>_final.csv
```

---

## 🎯 Key Design Patterns

### 1. **Stateless Processing**

Each script is stateless and reads its input from disk:

```python
# Example: 05_features.py
def main():
    # Read input from previous step
    df = pd.read_csv(f'results/{slide_name}/features/{slide_name}_nuclei_features_density.csv')

    # Process
    df = add_coherency_features(df, slide)
    df = add_local_variance(df)

    # Write output for next step
    df.to_csv(f'results/{slide_name}/features/{slide_name}_nuclei_features_enriched.csv')
```

**Why**: Enables restarting from any step, parallel processing, and debugging.

---

### 2. **Configuration Over Code**

All parameters live in `configs/slide_config.yaml`:

```yaml
H&E:
  segmentation:
    diameter_um: 8.0
    model: stardist
    batch_size: 32
  density_radii_um: [50, 100, 150]
  clustering:
    n_clusters: 20
    features: [area_px, circularity, ...]
```

Loading:
```python
from src.utils.config_loader import load_config

config = load_config('H&E')
diameter = config['segmentation']['diameter_um']
```

**Why**: Non-programmers can tune parameters, easier A/B testing.

---

### 3. **Type-Aware Processing**

Automatic slide type detection:

```python
# src/utils/slide_detector.py
def detect_slide_type(slide_path: str) -> str:
    filename = os.path.basename(slide_path)

    if re.search(r'CD3', filename, re.IGNORECASE):
        return 'IHC_CD3'
    elif re.search(r'H&E|HE[-_]', filename, re.IGNORECASE):
        return 'H&E'
    # ... more patterns

    return 'H&E'  # Default
```

Used in orchestration:
```bash
# scripts/run_adaptive_pipeline.sh
SLIDE_TYPE=$(python -c "from src.utils.slide_detector import detect_slide_type; print(detect_slide_type('$SLIDE_PATH'))")

if [[ "$SLIDE_TYPE" == *"IHC"* ]]; then
    # Run IHC-specific steps
    python src/core/05b_ihc_intensity.py "$SLIDE_NAME"
    python src/analysis/07_ihc_brown_stain.py "$SLIDE_NAME"
fi
```

**Why**: Single entry point for all slide types, reduces user error.

---

### 4. **Deduplication Strategy**

Cross-tile boundary nuclei are deduplicated using spatial + area matching:

```python
# Inspired by HistoVision
def deduplicate_nuclei(nuclei_list, dedup_radius_um=5.0):
    """
    For each nucleus pair:
    1. Check if distance < dedup_radius_um
    2. If yes, check if area difference < 20%
    3. If both true, merge (keep one)
    """
    # Convert to spatial index for O(n log n) instead of O(n²)
    from scipy.spatial import cKDTree

    tree = cKDTree(nuclei_coords)
    duplicates = tree.query_pairs(r=dedup_radius_um)

    for i, j in duplicates:
        area_i, area_j = nuclei_list[i].area, nuclei_list[j].area
        if abs(area_i - area_j) / max(area_i, area_j) < 0.2:
            # Mark j for removal
            nuclei_list[j].duplicate = True

    return [n for n in nuclei_list if not n.duplicate]
```

**Location**: `src/core/03_segment_*.py` (both Cellpose and StarDist)

**Why**: Ensures accurate counts despite overlapping tiles.

---

### 5. **Progressive Feature Addition**

Features are added incrementally:

```
nuclei_features.csv
    → nuclei_features_density.csv       (+ density columns)
    → nuclei_features_enriched.csv      (+ coherency, variance)
    → *_with_brown.csv                  (+ brown stain, IHC only)
    → *_final.csv                       (+ UMAP, cluster)
```

**Why**: Easier debugging (can inspect intermediate outputs), supports partial re-runs.

---

## ➕ Adding New Features

### Example: Add "Local Entropy" Feature

**Step 1**: Add function to `src/core/05_features.py`

```python
from skimage.filters.rank import entropy
from skimage.morphology import disk

def calculate_local_entropy(slide, nuclei_df, radius_um=100):
    """
    Calculate local entropy around each nucleus.

    Parameters:
    -----------
    slide : OpenSlide
        Whole slide image object
    nuclei_df : pd.DataFrame
        Nuclear features with x, y coordinates
    radius_um : float
        Radius for entropy calculation (µm)

    Returns:
    --------
    np.array : Local entropy values
    """
    # Get pixel-per-micron conversion
    mpp = float(slide.properties.get('openslide.mpp-x', 0.25))
    radius_px = int(radius_um / mpp)

    # Load grayscale image at appropriate level
    level = slide.get_best_level_for_downsample(2.0)
    img = slide.read_region((0, 0), level, slide.level_dimensions[level])
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    # Calculate entropy image
    entropy_img = entropy(gray, disk(radius_px))

    # Sample at nucleus locations
    entropy_values = []
    for _, row in nuclei_df.iterrows():
        x, y = int(row['x'] / (2**level)), int(row['y'] / (2**level))
        entropy_values.append(entropy_img[y, x])

    return np.array(entropy_values)
```

**Step 2**: Call it in the main processing function

```python
def main():
    # ... existing code ...

    # Add local entropy
    print("Calculating local entropy...")
    df[f'entropy_{radius}um'] = calculate_local_entropy(slide, df, radius=100)

    # ... rest of code ...
```

**Step 3**: Add to configuration

```yaml
# configs/slide_config.yaml
H&E:
  clustering:
    features: [area_px, circularity, coherency_150um, entropy_100um]  # Add here
```

**Step 4**: Update validation

```python
# src/validation/generate_feature_maps.py
# Add entropy to the list of features to visualize
features_to_validate = [
    'area_px', 'circularity', 'coherency_150um',
    'entropy_100um'  # Add here
]
```

**Step 5**: Test

```bash
# Run on test slide
./scripts/run_adaptive_pipeline.sh data/raw_slides/test_slide.svs

# Check output
python -c "import pandas as pd; df = pd.read_csv('results/test_slide/features/test_slide_final.csv'); print(df['entropy_100um'].describe())"

# View visualization
open results/test_slide/feature_maps/entropy_100um_validation.png
```

---

## ⚙️ Configuration System

### Structure of `slide_config.yaml`

```yaml
# Each slide type has its own section
H&E:
  description: "Standard H&E stained tissue"

  # Segmentation parameters
  segmentation:
    diameter_um: 8.0              # Expected nucleus diameter
    model: stardist               # cellpose or stardist
    batch_size: 32                # GPU batch size (A100 optimized)
    dedup_radius_um: 5.0          # Deduplication distance threshold

  # Density calculation
  density_radii_um: [50, 100, 150]  # Radii for density profiling

  # Clustering parameters
  clustering:
    n_clusters: 20
    features: [area_px, aspect_ratio, circularity, eccentricity,
               r, g, b, density_um2_r50.0, density_um2_r100.0,
               density_um2_r150.0, coherency_150um]

IHC_CD3:
  description: "CD3+ T-cell marker"

  segmentation:
    diameter_um: 8.0
    model: stardist
    batch_size: 32
    dedup_radius_um: 5.0

  density_radii_um: [50, 100, 150]

  # IHC-specific parameters
  brown_detection:
    threshold: 0.08               # DAB intensity threshold
    neighborhood_radius: 1.5      # Perinuclear expansion multiplier

  clustering:
    n_clusters: 20
    # Note: includes brown features
    features: [area_px, circularity, density_um2_r100.0,
               brown_intensity, brown_density_100um]
```

### Loading Configuration

```python
# src/utils/config_loader.py
import yaml

def load_config(slide_type: str) -> dict:
    """Load configuration for specific slide type."""
    with open('configs/slide_config.yaml', 'r') as f:
        all_configs = yaml.safe_load(f)

    if slide_type in all_configs:
        return all_configs[slide_type]
    else:
        print(f"Warning: Config for '{slide_type}' not found, using H&E default")
        return all_configs['H&E']
```

### Adding a New Slide Type

1. **Add to `configs/slide_config.yaml`**:

```yaml
IHC_PD1:
  description: "PD-1 immune checkpoint marker"
  segmentation:
    diameter_um: 7.5              # Smaller lymphocytes
    model: stardist
    batch_size: 32
    dedup_radius_um: 4.0
  density_radii_um: [50, 100, 150]
  brown_detection:
    threshold: 0.10               # Higher threshold (less sensitive)
    neighborhood_radius: 1.3
  clustering:
    n_clusters: 25                # More clusters expected
    features: [area_px, circularity, brown_intensity, brown_density_150um]
```

2. **Add detection pattern to `src/utils/slide_detector.py`**:

```python
def detect_slide_type(slide_path: str) -> str:
    filename = os.path.basename(slide_path)

    # ... existing patterns ...

    # Add new pattern
    if re.search(r'PD-?1', filename, re.IGNORECASE):
        return 'IHC_PD1'

    return 'H&E'
```

3. **Test**:

```bash
# Name your file with the pattern
cp slide.svs data/raw_slides/PD1-S25.svs

# Run - should auto-detect as IHC_PD1
./scripts/run_adaptive_pipeline.sh data/raw_slides/PD1-S25.svs
```

---

## 🧪 Testing & Validation

### Test Suite

**Location**: `scripts/test_setup.sh`

**9 Test Categories**:
1. Directory structure
2. Core pipeline scripts existence
3. Analysis scripts existence
4. Utility modules existence
5. Configuration files
6. Shell script permissions
7. Python imports
8. Slide type detection
9. Data availability

**Run tests**:
```bash
./scripts/test_setup.sh
# Should output: ALL TESTS PASSED ✅
```

---

### Validation Scripts

**1. Feature Map Validation** (`src/validation/generate_feature_maps.py`):

Generates 3-panel visualizations for visual QC:
- Left: Feature heatmap
- Middle: H&E image
- Right: Overlay

```bash
python src/validation/generate_feature_maps.py <slide_name>
# Output: results/<slide>/feature_maps/*_validation.png
```

**2. Coherency Validation** (`src/validation/test_coherency_synthetic.py`):

Creates synthetic images with known alignment patterns:
```python
# Generate horizontal lines (coherency ≈ 1.0)
# Generate random dots (coherency ≈ 0.0)
# Verify metric responds correctly
```

**3. Segmentation Comparison** (`src/analysis/08_compare_segmenters.py`):

Compares Cellpose vs StarDist side-by-side:
- Nucleus count
- Mean/median area
- Circularity distribution
- Processing time
- Visual overlay comparison

```bash
# Run both segmenters on same slide
./scripts/run_one_slide.sh slide.svs              # Cellpose
./scripts/run_one_slide_stardist.sh slide.svs    # StarDist

# Compare
python src/analysis/08_compare_segmenters.py slide_name
```

---

### Adding New Tests

**Example: Test density calculation accuracy**

Create `src/validation/test_density.py`:

```python
#!/usr/bin/env python
"""Test density calculation on synthetic data."""

import numpy as np
import pandas as pd
from src.core.04_density import calculate_density

def test_known_density():
    """Test with known uniform grid."""
    # Create 10×10 grid with 10µm spacing
    x = np.repeat(np.arange(10) * 10, 10)  # 0, 0, ..., 90, 90, ...
    y = np.tile(np.arange(10) * 10, 10)    # 0, 10, ..., 90, 0, 10, ...

    df = pd.DataFrame({'x_um': x, 'y_um': y})

    # Calculate density at 50µm radius
    # Expected: π × 50² = 7,854 µm²
    # Grid has 10µm spacing, so ~49 nuclei in circle
    # Density = 49 / 7854 ≈ 0.0062 nuclei/µm²

    densities = calculate_density(df, radius_um=50)
    expected = 0.0062

    # Allow 10% tolerance
    assert np.abs(densities.mean() - expected) / expected < 0.1
    print("✓ Density test passed")

if __name__ == '__main__':
    test_known_density()
```

Add to test suite:
```bash
# scripts/test_setup.sh
echo "✓ Test 10: Density accuracy"
python src/validation/test_density.py
```

---

## ⚡ Performance Optimization

### GPU Optimization

**Current settings** (optimized for NVIDIA A100):

```yaml
# configs/slide_config.yaml
segmentation:
  batch_size: 32  # A100 with 40-80GB VRAM
```

**For other GPUs**:
- **V100 (32GB)**: `batch_size: 24`
- **RTX 3090 (24GB)**: `batch_size: 16`
- **RTX 2080 Ti (11GB)**: `batch_size: 8`

**Check GPU usage**:
```bash
watch -n 1 nvidia-smi
# Monitor "GPU-Util" and "Memory-Usage"
```

---

### Memory Profiling

**Profile memory usage**:

```python
import tracemalloc

tracemalloc.start()

# Your code here
result = process_slide(slide_path)

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory: {current / 1024**3:.2f} GB")
print(f"Peak memory: {peak / 1024**3:.2f} GB")
tracemalloc.stop()
```

**Reduce memory usage**:
1. Process tiles in smaller batches
2. Use lower resolution for feature extraction
3. Clear intermediate results: `del large_array; gc.collect()`

---

### Parallelization

**Tile-level parallelization** (already implemented):

```python
# src/core/03_segment_stardist.py
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(segment_tile, tile) for tile in tiles]
    results = [f.result() for f in futures]
```

**Slide-level parallelization** (SLURM):

```bash
# submit_production_raw_slides.sh creates job per slide
for slide in *.svs; do
    sbatch job_${slide}.sh
done
```

---

## 🛠️ Common Development Tasks

### Task 1: Debug a Processing Step

**Scenario**: Step 05 (feature extraction) is failing

```bash
# 1. Check if previous step completed
ls results/slide_name/features/*_density.csv

# 2. Run step manually with verbose output
python -u src/core/05_features.py slide_name 2>&1 | tee debug.log

# 3. Inspect intermediate outputs
python -c "import pandas as pd; df = pd.read_csv('results/slide_name/features/slide_name_nuclei_features_density.csv'); print(df.info()); print(df.describe())"

# 4. Check for NaN or inf values
python -c "import pandas as pd; import numpy as np; df = pd.read_csv('results/slide_name/features/slide_name_nuclei_features_density.csv'); print('NaNs:', df.isna().sum().sum()); print('Infs:', np.isinf(df.select_dtypes(include=[np.number])).sum().sum())"
```

---

### Task 2: Add a New Segmentation Model

**Example**: Integrate Mask R-CNN

**Step 1**: Create `src/core/03_segment_maskrcnn.py`

```python
#!/usr/bin/env python
"""Nuclear segmentation using Mask R-CNN."""

import sys
import pandas as pd
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

def segment_with_maskrcnn(tile_path, config):
    """Segment nuclei using Mask R-CNN."""
    # Load model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = "path/to/trained_weights.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)

    # Load tile
    img = cv2.imread(tile_path)

    # Predict
    outputs = predictor(img)
    masks = outputs["instances"].pred_masks.cpu().numpy()

    # Extract features (same as other segmenters)
    nuclei_features = []
    for mask in masks:
        props = regionprops(mask.astype(int))[0]
        nuclei_features.append({
            'area_px': props.area,
            'centroid_x': props.centroid[1],
            'centroid_y': props.centroid[0],
            # ... more features
        })

    return pd.DataFrame(nuclei_features)

def main():
    slide_name = sys.argv[1]
    # ... implementation following same pattern as 03_segment_stardist.py
```

**Step 2**: Add to config

```yaml
H&E:
  segmentation:
    model: maskrcnn  # Add option
```

**Step 3**: Update orchestration

```bash
# scripts/run_adaptive_pipeline.sh
if [[ "$SEGMENTER" == "maskrcnn" ]]; then
    python src/core/03_segment_maskrcnn.py "$SLIDE_NAME"
elif [[ "$SEGMENTER" == "stardist" ]]; then
    python src/core/03_segment_stardist.py "$SLIDE_NAME"
else
    python src/core/03_segment_cellpose.py "$SLIDE_NAME"
fi
```

---

### Task 3: Optimize for New HPC Cluster

**Scenario**: Moving from SLURM to PBS

**Step 1**: Create PBS submission template

```bash
# templates/pbs_job_template.sh
#!/bin/bash
#PBS -N histology_{{SLIDE_NAME}}
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=02:00:00
#PBS -o logs/{{SLIDE_NAME}}.out
#PBS -e logs/{{SLIDE_NAME}}.err

cd $PBS_O_WORKDIR
module load anaconda3
source activate histology-pipeline

./scripts/run_adaptive_pipeline.sh {{SLIDE_PATH}}
```

**Step 2**: Create PBS submission script

```bash
# submit_production_pbs.sh
#!/bin/bash

for slide in data/raw_slides/*.svs; do
    slide_name=$(basename "$slide" .svs)

    # Generate job script from template
    sed "s|{{SLIDE_NAME}}|$slide_name|g; s|{{SLIDE_PATH}}|$slide|g" \
        templates/pbs_job_template.sh > jobs/${slide_name}.pbs

    # Submit
    qsub jobs/${slide_name}.pbs
done
```

---

### Task 4: Create Custom Visualization

**Example**: Plot density gradient across tissue

```python
# custom_viz/density_gradient.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_density_gradient(slide_name):
    """Plot spatial density gradient."""
    df = pd.read_csv(f'results/{slide_name}/features/{slide_name}_final.csv')

    # Create 2D histogram
    fig, ax = plt.subplots(figsize=(12, 10))

    # Bin nuclei by position
    x_bins = np.linspace(df.x_um.min(), df.x_um.max(), 50)
    y_bins = np.linspace(df.y_um.min(), df.y_um.max(), 50)

    # Calculate mean density per bin
    density_grid = np.zeros((len(y_bins)-1, len(x_bins)-1))
    for i in range(len(y_bins)-1):
        for j in range(len(x_bins)-1):
            mask = (df.x_um >= x_bins[j]) & (df.x_um < x_bins[j+1]) & \
                   (df.y_um >= y_bins[i]) & (df.y_um < y_bins[i+1])
            if mask.sum() > 0:
                density_grid[i, j] = df.loc[mask, 'density_um2_r100.0'].mean()

    # Plot
    im = ax.imshow(density_grid, cmap='viridis', aspect='auto',
                   extent=[df.x_um.min(), df.x_um.max(),
                          df.y_um.min(), df.y_um.max()])
    plt.colorbar(im, label='Mean Density (nuclei/µm²)')
    ax.set_xlabel('X position (µm)')
    ax.set_ylabel('Y position (µm)')
    ax.set_title(f'{slide_name} - Density Gradient')

    plt.savefig(f'results/{slide_name}/custom_viz/density_gradient.png', dpi=300)
    print(f"Saved to results/{slide_name}/custom_viz/density_gradient.png")

if __name__ == '__main__':
    import sys
    plot_density_gradient(sys.argv[1])
```

---

## 📝 Contributing Guidelines

### Code Style

**Follow PEP 8**:
```bash
# Install linter
pip install flake8

# Check code
flake8 src/

# Auto-format
pip install black
black src/
```

**Docstring format**:
```python
def calculate_feature(df, param1, param2=10):
    """
    Calculate custom feature for nuclei.

    Parameters
    ----------
    df : pd.DataFrame
        Nuclear features with x, y coordinates
    param1 : float
        Description of param1
    param2 : int, optional
        Description of param2 (default: 10)

    Returns
    -------
    np.array
        Feature values for each nucleus

    Examples
    --------
    >>> df = pd.DataFrame({'x': [0, 1], 'y': [0, 1]})
    >>> features = calculate_feature(df, 5.0)
    >>> len(features)
    2
    """
```

---

### Git Workflow

**Branching**:
```bash
# Create feature branch

# Make changes, commit frequently
git add src/core/05_features.py
git commit -m "Add local entropy feature calculation"

# Push to remote
git push origin feature/add-entropy-feature
```

**Commit messages**:
```
Short summary (50 chars or less)

More detailed explanation if needed (wrap at 72 chars).
Explain what changed and why, not how.

- Bullet points are fine
- Use present tense: "Add feature" not "Added feature"
```

---

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update

## Testing
- [ ] Tested on H&E slide
- [ ] Tested on IHC slide
- [ ] All validation tests pass
- [ ] No performance regression

## Checklist
- [ ] Code follows style guidelines
- [ ] Added docstrings
- [ ] Updated documentation
- [ ] Added tests if applicable
```

---

## 📚 Additional Resources

### Understanding the Methods

**Coherency Calculation**:
- Based on structure tensor eigenvalue analysis
- Reference: Rezakhaniha et al. (2012) Biomech Model Mechanobiol
- Implementation: `src/core/05_features.py:calculate_coherency()`

**Color Deconvolution**:
- H-DAB separation using Ruifrok & Johnston matrix
- Reference: Ruifrok & Johnston (2001) Anal Quant Cytol Histol
- Implementation: `src/analysis/07_ihc_brown_stain.py`

**UMAP**:
- McInnes et al. (2018) arXiv:1802.03426
- Parameters: n_neighbors=15, min_dist=0.1
- Implementation: `src/analysis/09_umap_clustering.py`

---

### External Documentation

- **OpenSlide**: https://openslide.org/api/python/
- **Cellpose**: https://cellpose.readthedocs.io/
- **StarDist**: https://github.com/stardist/stardist
- **UMAP**: https://umap-learn.readthedocs.io/
- **scikit-image**: https://scikit-image.org/

---

**For additional help, see**:
- **USER_GUIDE.md** - For end-user instructions
- **HANDOFF_NOTES.md** - For current project status
- **VALIDATION_STATUS.md** - For validation documentation

---

**Happy developing!** 💻
