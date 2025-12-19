# Analysis Scripts (Steps 07-11)

Advanced analysis and visualization scripts for downstream processing of segmented nuclei.

## Scripts

### Step 07: IHC Brown Stain Analysis
- **`07_ihc_brown_stain.py`** - IHC brown stain quantification using DAB color deconvolution
  - H-DAB color deconvolution matrix (Ruifrok & Johnston, 2001)
  - Marker-specific thresholds (CD3: 0.08, GFAP: 0.12, IBA1: 0.08)
  - Perinuclear intensity measurement with configurable expansion radius
  - Outputs: brown_intensity, brown_density features per nucleus

### Step 08: Specialized Analysis
- **`08_compare_segmenters.py`** - Cellpose vs StarDist comparison
  - Side-by-side segmentation metrics
  - Performance evaluation and validation

- **`08_nfb_filament_analysis.py`** - Neurofilament filament tracing
  - Separate filament-centric analysis (not nucleus-based)
  - Skeletonization for centerline extraction
  - Measures: filament count, length distribution, branch points, orientation
  - Runs in parallel with main pipeline

### Step 09: UMAP Clustering
- **`09_umap_clustering.py`** - UMAP dimensionality reduction & clustering
  - Slide-type aware feature selection from config
  - Optional PCA preprocessing (recommended for >10 features)
  - BIRCH clustering with configurable n_clusters
  - Generates spatial overlays + cluster statistics
  - Outputs: umap_1, umap_2, cluster assignments

### Step 10: Combined Analysis
- **`10_separate_umaps.py`** - Per-stain-type combined UMAPs
  - Groups slides by stain type (H&E, CD3, GFAP, Iba1, NF, PGP9.5)
  - Generates multi-slide embeddings for cross-slide comparison
  - Creates violin plots and statistical summaries

### Step 11: Multi-Modal Registration
- **`11_combined_umap.py`** - Multi-modal spatial registration
  - Advanced cross-stain nuclear matching using SimpleITK
  - Spatial alignment of H&E with IHC slides
  - Requires SimpleITK for image registration

## Usage

All scripts follow the standard pattern:
```bash
python src/analysis/<script>.py <slide_name>
```

Or use the orchestration scripts:
```bash
./scripts/run_adaptive_pipeline.sh <slide_path>
./scripts/run_all_by_type.sh
```

## Requirements

- All scripts require completed core pipeline (steps 00-06)
- Step 07: Requires IHC slides with DAB staining
- Step 08 (NFB): Requires neurofilament-stained slides
- Step 11: Requires SimpleITK for registration
