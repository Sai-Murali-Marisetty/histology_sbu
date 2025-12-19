# Core Pipeline Scripts (Steps 00-06)

These are the main pipeline scripts that run in sequence.

## Scripts

- `00_preview.py` - Generate slide thumbnails and previews
- `01_tissue_mask.py` - Create tissue mask
- `02_tile.py` - Tile extraction with overlap
- `03_segment_cellpose.py` - Nuclear segmentation (Cellpose)
- `03_segment_stardist.py` - Nuclear segmentation (StarDist)
- `04_density.py` - Density profiling
- `05_features.py` - Feature extraction & enrichment
- `06_qc.py` - Quality control visualizations

## Usage

Use `scripts/run_one_slide.sh` to run the full pipeline.
