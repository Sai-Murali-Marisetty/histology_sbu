# Getting Started Guide

## 📋 What You Have Now

Your pipeline is **fully organized and ready to use**! Here's what's been created:

```
histology/
├── src/
│   ├── core/                    ✅ Core pipeline (00-06)
│   ├── analysis/                ✅ NEW analysis scripts (07-09)
│   ├── utils/                   ✅ NEW utilities
│   └── validation/              ✅ Testing scripts
├── scripts/                     ✅ Batch processing
├── configs/                     ✅ Configuration files
└── archive/                     ✅ Old files (backed up)
```

---

## 🚀 Quick Start

### **Step 1: Test Your Setup**

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Test everything is working
./scripts/test_setup.sh
```

Expected output: `ALL TESTS PASSED ✅`

---

### **Step 2: Process One Slide**

```bash
# Process one H&E slide
./scripts/run_adaptive_pipeline.sh raw_slides/HE-S25.svs results

# Process one IHC slide (with brown stain analysis)
./scripts/run_adaptive_pipeline.sh raw_slides/CD3-S25.svs results
```

**What it does:**
- ✅ Auto-detects slide type (H&E or IHC)
- ✅ Segments nuclei (Cellpose)
- ✅ Computes density, coherency, variance
- ✅ **IHC only:** Quantifies brown stain
- ✅ UMAP clustering
- ✅ Creates visualizations

---

### **Step 3: Check Results**

```bash
cd results/HE-S25

# View structure
tree -L 2

# Key files:
# - features/HE-S25_final.csv         → All features + clusters
# - clustering/umap_clusters.png      → UMAP visualization
# - qc/                               → Quality control
# - brown_stain/                      → IHC only
```

---

### **Step 4: Process All Slides**

```bash
# Process everything at once
./scripts/run_all_by_type.sh raw_slides results_all

# This will:
# 1. Classify all slides by type
# 2. Run adaptive pipeline on each
# 3. Generate comparison report
```

---

## 📊 What Each Slide Type Gets

### **H&E Slides**
- Nuclear segmentation
- Shape features (area, circularity, aspect ratio)
- RGB color features
- Density profiles (50, 100, 150µm)
- Coherency (nuclear alignment)
- Local variance
- UMAP clustering

### **IHC Slides (CD3, GFAP, Iba1, etc.)**
- Everything from H&E, **PLUS:**
- **Brown (DAB) stain detection**
- **Brown intensity per nucleus**
- **Brown-positive density**
- **Marker-specific clustering**

---

## 🔧 Advanced Usage

### **Compare Cellpose vs StarDist**

```bash
# Run both segmenters on same slide
./scripts/run_one_slide.sh raw_slides/HE-S25.svs results_cellpose
./scripts/run_one_slide_stardist.sh raw_slides/HE-S25.svs results_stardist

# Compare results
python3 src/analysis/08_compare_segmenters.py \
    --cellpose_csv results_cellpose/HE-S25/features/HE-S25_nuclei_features.csv \
    --stardist_csv results_stardist/HE-S25/features/HE-S25_nuclei_features.csv \
    --out_dir comparisons/HE-S25 \
    --slide_name HE-S25
```

### **Customize Configuration**

Edit `configs/slide_config.yaml` to change:
- Segmentation parameters (diameter, model)
- Brown stain thresholds (IHC)
- Density radii
- Clustering parameters
- Features to include

Example:
```yaml
IHC_CD3:
  segmentation:
    diameter_um: 12.0     # Change nucleus size
  brown_detection:
    threshold: 0.20       # Stricter brown detection
  clustering:
    n_clusters: 25        # More clusters
```

---

## 📁 Understanding the Output

### **Per-Slide Results Structure**

```
results/CD3-S25/
├── preview/
│   ├── CD3-S25_thumb.jpg          # Slide thumbnail
│   └── panel_CD3-S25_preview.png  # Preview panel
├── masks/
│   └── CD3-S25_tissue_mask.png    # Tissue segmentation
├── tiles/
│   ├── tile_0_0.png               # Image tiles
│   └── tiles.json                 # Tile metadata
├── cellpose/
│   ├── masks/                     # Segmentation masks
│   └── viz/                       # Segmentation overlays
├── features/
│   ├── CD3-S25_nuclei_features.csv           # Raw features
│   ├── CD3-S25_nuclei_features_enriched.csv  # + density, coherency
│   ├── CD3-S25_with_brown.csv                # + brown stain (IHC only)
│   └── CD3-S25_final.csv                     # + clusters
├── brown_stain/                   # IHC only
│   ├── brown_stain_overlay.jpg
│   └── brown_density_100um.jpg
├── clustering/
│   ├── umap_clusters.png
│   ├── cluster_features.png
│   ├── cluster_spatial.png
│   └── cluster_statistics.csv
├── viz/
│   ├── overlay_coherency_150um.jpg
│   └── overlay_area_px_local_variance_150um.jpg
└── qc/
    └── qc_summary.json
```

### **Key CSV Columns**

The final CSV (`*_final.csv`) contains:

**Basic:**
- `nucleus_id`, `slide_id`, `x`, `y`, `x_um`, `y_um`

**Shape:**
- `area_px`, `perimeter_px`, `circularity`, `aspect_ratio`, `eccentricity`
- `major_axis_length`, `minor_axis_length`, `orientation`

**Color:**
- `r`, `g`, `b` (mean RGB values)

**Spatial:**
- `density_um2_r50.0`, `density_um2_r100.0`, `density_um2_r150.0`
- `coherency_150um`

**Statistics:**
- `area_px_local_variance_150um`
- `circularity_local_variance_150um`
- (etc. for all features at all radii)

**IHC only:**
- `has_brown` (0/1)
- `brown_intensity` (DAB value)
- `brown_density_100um`, `brown_density_150um`

**Clustering:**
- `umap_1`, `umap_2`
- `cluster` (0, 1, 2, ...)

---


### ✅ **Delivered Features**

| Requirement | Status | Location |
|-------------|--------|----------|
| Shape factors | ✅ Complete | All CSVs |
| Density profiles | ✅ Complete | All CSVs (3 radii) |
| Variance statistics | ✅ Complete | All CSVs |
| Coherency metric | ✅ Complete | All CSVs |
| IHC brown stain | ✅ Complete | IHC slides only |
| UMAP clustering | ✅ Complete | `clustering/` |
| Cellpose vs StarDist | ✅ Available | Run comparison script |

---

## 🐛 Troubleshooting

### **Issue: "Module not found" error**

```bash
# Make sure you're in the right directory
cd histology

# Check Python path
python3 -c "import sys; print(sys.path)"
```

### **Issue: Script not executable**

```bash
chmod +x scripts/*.sh
```

### **Issue: YAML import error**

```bash
pip install pyyaml
```

### **Issue: UMAP import error**

```bash
pip install umap-learn
```

### **Issue: Segmentation fails**

Check GPU availability:
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## 📞 Next Steps for Your Meeting

### **1. Test on Sample Slides**

```bash
# H&E slide
./scripts/run_adaptive_pipeline.sh raw_slides/HE-S25.svs test_results

# IHC slide
./scripts/run_adaptive_pipeline.sh raw_slides/CD3-S25.svs test_results
```

### **2. Verify All Features**

```bash
python3 -c "
import pandas as pd

df = pd.read_csv('test_results/HE-S25/features/HE-S25_final.csv')

required = [
    'circularity', 'aspect_ratio',
    'density_um2_r50.0', 'density_um2_r100.0', 'density_um2_r150.0',
    'coherency_150um',
    'area_px_local_variance_150um',
    'cluster', 'umap_1', 'umap_2'
]

for col in required:
    assert col in df.columns, f'Missing: {col}'
    
print('✅ All required features present!')
print(f'Total features: {len(df.columns)}')
print(f'Total nuclei: {len(df):,}')
"
```

### **3. Create Demo Slides**

- Slide thumbnail with segmentation overlay
- Density heatmaps
- Coherency visualization
- Brown stain analysis (IHC)
- UMAP clusters

All these are auto-generated in `results/[slide]/clustering/` and `viz/`

---

## 📚 Documentation

- **Pipeline overview:** `src/core/README.md`
- **Analysis methods:** `src/analysis/README.md`
- **Configuration guide:** `configs/slide_config.yaml` (has comments)
- **Utility functions:** `src/utils/README.md`

---

## ✨ Summary

You now have a **complete, production-ready pipeline** that:

1. ✅ Auto-detects slide types (H&E vs IHC)
2. ✅ Segments nuclei (Cellpose or StarDist)
3. ✅ Extracts 50+ features per nucleus
4. ✅ Computes density, coherency, variance
5. ✅ Quantifies IHC brown stain
6. ✅ Performs UMAP clustering
7. ✅ Generates comprehensive visualizations
8. ✅ Processes batches efficiently

