# Histology Image Analysis Pipeline

**Production-ready pipeline for comprehensive nuclear segmentation and feature extraction from whole-slide histology images (H&E and IHC).**

**Version**: 1.0 (Production Release)
**Last Updated**: December 2024
**Status**: ✅ Ready for deployment

---

## 🎯 Overview

This pipeline performs automated analysis of whole slide images (WSI) to extract nuclear features, quantify IHC markers, and identify cell populations through unsupervised clustering. Designed for both end-users (plug-and-play analysis) and developers (extensible architecture).

**Key Capabilities**:
- 🔬 **Dual Segmentation**: Both Cellpose AND StarDist supported (choose based on your data)
- 📊 **50+ Features**: Morphology, density, alignment, color, IHC intensity
- 🎨 **Multiple Stains**: H&E, CD3, GFAP, IBA1, Neurofilament, PGP9.5
- 🚀 **HPC-Ready**: SLURM integration for batch processing
- 🔄 **Type-Aware**: Automatically adapts processing based on slide type
- ✅ **Validated**: Comprehensive testing framework included

---

## ✨ Key Features

### Core Pipeline (Steps 00-06)
✅ **Tissue Detection** - HSV-based segmentation with multi-component preservation
✅ **Nuclear Segmentation** - **Dual support: Cellpose OR StarDist** (A100-optimized)
✅ **Density Profiling** - Multi-radius analysis (50, 100, 150 µm)
✅ **Coherency Analysis** - Nuclear alignment via structure tensor
✅ **Feature Extraction** - Morphology, RGB, local variance (50+ features)
✅ **IHC Intensity** - Perinuclear marker intensity measurement
✅ **Quality Control** - Automated QC visualizations

### Advanced Analysis (Steps 07-11)
✅ **Brown Stain Quantification** - DAB color deconvolution for IHC
✅ **Filament Analysis** - Neurofilament tracing and architecture
✅ **UMAP Clustering** - Unsupervised cell population identification
✅ **Combined Analysis** - Multi-slide per-stain comparisons
✅ **Multi-Modal Registration** - Cross-stain spatial alignment

### Dual Segmentation Support

**Why two segmentation methods?**

This pipeline supports **both** Cellpose and StarDist, allowing you to choose the best method for your data:

| Method | Best For | Speed | Accuracy on Dense Nuclei |
|--------|----------|-------|-------------------------|
| **StarDist** ⭐ | Dense, round nuclei (H&E, IHC) | Fast (GPU-optimized, batch=32) | Excellent |
| **Cellpose** | Irregular cells, overlapping structures | Moderate | Good |

**Usage**:
```bash
# StarDist (recommended for most histology)
./scripts/run_one_slide_stardist.sh slide.svs

# Cellpose (alternative method)
./scripts/run_one_slide.sh slide.svs

# Auto-detect & choose best (StarDist default)
./scripts/run_adaptive_pipeline.sh slide.svs

# Compare both methods
python src/analysis/08_compare_segmenters.py slide_name
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
conda env create -f environment.yml
conda activate histology-pipeline

# Verify installation
./scripts/test_setup.sh
# Should output: ALL TESTS PASSED ✅
```

**See [GETTING_STARTED.md](GETTING_STARTED.md) for detailed setup instructions.**

---

### Process Your First Slide

```bash
# Single slide (auto-detects type, uses StarDist)
./scripts/run_adaptive_pipeline.sh data/raw_slides/HE-S25.svs

# Results appear in:
# results/HE-S25/features/HE-S25_final.csv          (main output)
# results/HE-S25/clustering/umap_clusters.png       (visualization)
```

**See [USER_GUIDE.md](USER_GUIDE.md) for complete usage examples.**

---

### Batch Processing

```bash
# Process all slides in a directory
./scripts/run_all_by_type.sh data/raw_slides results

# HPC/SLURM submission (for large batches)
./submit_production_raw_slides.sh
```

---

## 📁 Directory Structure

```
histology/
├── src/                          # Source code
│   ├── core/                     # Core pipeline (00-06)
│   │   ├── 00_preview.py         # Thumbnail generation
│   │   ├── 01_tissue_mask.py     # Tissue segmentation
│   │   ├── 02_tile.py            # Slide tiling
│   │   ├── 03_segment_cellpose.py    # Cellpose segmentation
│   │   ├── 03_segment_stardist.py    # StarDist segmentation ⭐
│   │   ├── 04_density.py         # Density profiling
│   │   ├── 05_features.py        # Feature extraction
│   │   ├── 05b_ihc_intensity.py  # IHC marker intensity
│   │   └── 06_qc.py              # Quality control
│   ├── analysis/                 # Advanced analysis (07-11)
│   │   ├── 07_ihc_brown_stain.py     # DAB quantification
│   │   ├── 08_compare_segmenters.py  # Cellpose vs StarDist
│   │   ├── 08_nfb_filament_analysis.py   # Filament tracing
│   │   ├── 09_umap_clustering.py     # UMAP clustering
│   │   ├── 10_separate_umaps.py      # Per-stain combined UMAP
│   │   └── 11_combined_umap.py       # Multi-modal registration
│   ├── utils/                    # Utilities
│   │   ├── slide_detector.py     # Auto slide type detection
│   │   └── config_loader.py      # Configuration loading
│   └── validation/               # Testing & validation
│       ├── generate_feature_maps.py      # 3-panel visualizations
│       └── test_*.py             # Validation scripts
├── scripts/                      # Shell scripts
│   ├── run_adaptive_pipeline.sh  # 🌟 Main entry point
│   ├── run_one_slide_stardist.sh # StarDist pipeline
│   ├── run_one_slide.sh          # Cellpose pipeline
│   ├── run_all_by_type.sh        # Batch processing
│   └── test_setup.sh             # Environment validation
├── configs/                      # Configuration
│   └── slide_config.yaml         # Per-slide-type parameters
├── data/                         # Input data (user-created)
│   └── raw_slides/               # Place .svs files here
├── results/                      # Output directory
│   └── <slide_name>/             # Per-slide results
│       ├── features/<slide>_final.csv    # Main output ⭐
│       ├── clustering/umap_clusters.png  # Visualization ⭐
│       └── ...
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment
│
├── README.md                     # This file (overview)
├── GETTING_STARTED.md            # Setup & installation guide
├── USER_GUIDE.md                 # 🌟 End-user instructions
├── DEVELOPER_GUIDE.md            # 🌟 Architecture & extension guide
├── HANDOFF_NOTES.md              # Project status & handoff info
├── PRODUCTION_PIPELINE_GUIDE.md  # HPC deployment guide
├── DELIVERY_CHECKLIST.md         # Pre-deployment verification
└── VALIDATION_STATUS.md          # Validation documentation
```

---

## 📊 Output Files

### Main Output: `results/<slide>/features/<slide>_final.csv`

Each row = one nucleus with 50+ features:

**Identity**: `nucleus_id`, `x`, `y`, `x_um`, `y_um`
**Morphology**: `area_px`, `circularity`, `eccentricity`, `aspect_ratio`
**Color**: `r`, `g`, `b`
**Density**: `density_um2_r50.0`, `density_um2_r100.0`, `density_um2_r150.0`
**Alignment**: `coherency_150um`
**Clustering**: `umap_1`, `umap_2`, `cluster`
**IHC (if applicable)**: `brown_intensity`, `brown_density_*`, `marker_intensity_*`

### Key Visualizations

- `clustering/umap_clusters.png` - Cell population structure
- `clustering/cluster_spatial.png` - Spatial distribution of clusters
- `feature_maps/*_validation.png` - 3-panel QC for each feature
- `brown_stain/brown_stain_overlay.jpg` - IHC marker visualization (IHC only)

---

## 📚 Documentation

**For End Users** (just want to analyze slides):
- 🌟 **[USER_GUIDE.md](USER_GUIDE.md)** - Step-by-step instructions, workflows, troubleshooting
- [GETTING_STARTED.md](GETTING_STARTED.md) - Installation and setup
- [PRODUCTION_PIPELINE_GUIDE.md](PRODUCTION_PIPELINE_GUIDE.md) - HPC/SLURM usage

**For Developers** (want to extend or modify):
- 🌟 **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** - Architecture, code organization, adding features
- [src/core/README.md](src/core/README.md) - Core pipeline details
- [src/analysis/README.md](src/analysis/README.md) - Analysis module details

**For Project Handoff**:
- 🌟 **[HANDOFF_NOTES.md](HANDOFF_NOTES.md)** - Current status, pending work, known issues
- [VALIDATION_STATUS.md](VALIDATION_STATUS.md) - Completed validations
- [DELIVERY_CHECKLIST.md](DELIVERY_CHECKLIST.md) - Pre-deployment verification

---

## 🧪 Testing & Validation

```bash
# Run all tests (9 categories)
./scripts/test_setup.sh

# Generate validation visualizations
python src/validation/generate_feature_maps.py <slide_name>

# Compare segmentation methods
python src/analysis/08_compare_segmenters.py <slide_name>
```

**All tests passing ✅** (as of December 2024)

---

## 🎓 Supported Slide Types

**Auto-detected from filename**:

| Stain Type | Filename Pattern | Example |
|------------|------------------|---------|
| H&E | `HE-`, `H&E-` | `HE-S25.svs` |
| CD3 (T cells) | `CD3-` | `CD3-tumor-S1.svs` |
| GFAP (Astrocytes) | `GFAP-` | `GFAP-S17.svs` |
| IBA1 (Microglia) | `IBA1-`, `Iba1-` | `IBA1-S9.svs` |
| Neurofilament | `NF-` | `NF-S19.svs` |
| PGP9.5 (Neurons) | `PGP9-5-`, `PGP9.5-` | `PGP9-5-B27.svs` |

Configuration for each type in `configs/slide_config.yaml`

---

## 🔧 Configuration

All parameters externalized to `configs/slide_config.yaml`:

```yaml
H&E:
  segmentation:
    diameter_um: 8.0        # Expected nucleus diameter
    model: stardist         # cellpose or stardist
    batch_size: 32          # GPU batch size
  density_radii_um: [50, 100, 150]
  clustering:
    n_clusters: 20
    features: [area_px, circularity, density_um2_r100.0, ...]
```

**Modify parameters without changing code!**

---

## ⚡ Performance

**Processing Time** (typical slide, NVIDIA A100):
- H&E slide: ~15-30 minutes
- IHC slide: ~30-45 minutes
- Neurofilament slide: ~45-60 minutes

**GPU Optimization**:
- StarDist batch size optimized for A100 (batch_size=32)
- Adjust in config for other GPUs (V100: 24, RTX 3090: 16)

---

## 🤝 Contributing

See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for:
- Code organization and architecture
- How to add new features
- Development best practices
- Extension examples

---

## 📞 Support & Contact

**Documentation**:
- Installation: [GETTING_STARTED.md](GETTING_STARTED.md)
- Usage: [USER_GUIDE.md](USER_GUIDE.md)
- Development: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
- Handoff: [HANDOFF_NOTES.md](HANDOFF_NOTES.md)

**Principal Investigator**:
- Principal Investigator 

**Original Developer**:
- Development Team

---


---

## 🎯 Quick Reference Commands

```bash
# Setup
conda env create -f environment.yml
conda activate histology-pipeline
./scripts/test_setup.sh

# Single slide processing
./scripts/run_adaptive_pipeline.sh data/raw_slides/slide.svs

# Batch processing
./scripts/run_all_by_type.sh data/raw_slides results

# Compare segmentation methods
./scripts/run_one_slide.sh slide.svs              # Cellpose
./scripts/run_one_slide_stardist.sh slide.svs    # StarDist
python src/analysis/08_compare_segmenters.py slide_name

# HPC submission
./submit_production_raw_slides.sh

# Validation
python src/validation/generate_feature_maps.py slide_name
```

---

**🌟 For detailed instructions, start with [USER_GUIDE.md](USER_GUIDE.md) or [GETTING_STARTED.md](GETTING_STARTED.md)**
