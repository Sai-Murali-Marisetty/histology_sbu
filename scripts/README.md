# Batch Processing Scripts

Shell scripts for running the pipeline.

## Main Scripts

- `run_one_slide.sh` - Process single slide (Cellpose)
- `run_one_slide_stardist.sh` - Process single slide (StarDist)
- `run_adaptive_pipeline.sh` - Auto-detect type & process
- `run_all_by_type.sh` - Process all slides

## Usage

```bash
# Single slide
./scripts/run_one_slide.sh raw_slides/HE-S25.svs results

# All slides
./scripts/run_all_by_type.sh
```
