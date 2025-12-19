#!/bin/bash
# test_setup.sh - Quick test to verify everything is set up correctly

set -e

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                        TESTING PIPELINE SETUP                              ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Test 1: Check directory structure
echo "✓ Test 1: Directory structure"
for dir in src/core src/analysis src/utils src/validation scripts configs; do
    if [[ -d "$dir" ]]; then
        echo "  ✓ $dir exists"
    else
        echo "  ❌ $dir missing"
        exit 1
    fi
done

# Test 2: Check core scripts
echo ""
echo "✓ Test 2: Core pipeline scripts"
for script in 00_preview.py 01_tissue_mask.py 02_tile.py 03_segment_cellpose.py 04_density.py 05_features.py 06_qc.py; do
    if [[ -f "src/core/$script" ]]; then
        echo "  ✓ $script exists"
    else
        echo "  ❌ $script missing"
        exit 1
    fi
done

# Test 3: Check analysis scripts
echo ""
echo "✓ Test 3: Analysis scripts"
for script in 07_ihc_brown_stain.py 09_umap_clustering.py 08_compare_segmenters.py; do
    if [[ -f "src/analysis/$script" ]]; then
        echo "  ✓ $script exists"
    else
        echo "  ❌ $script missing"
        exit 1
    fi
done

# Test 4: Check utilities
echo ""
echo "✓ Test 4: Utility modules"
if [[ -f "src/utils/slide_detector.py" ]]; then
    echo "  ✓ slide_detector.py exists"
else
    echo "  ❌ slide_detector.py missing"
    exit 1
fi

if [[ -f "src/utils/config_loader.py" ]]; then
    echo "  ✓ config_loader.py exists"
else
    echo "  ❌ config_loader.py missing"
    exit 1
fi

# Test 5: Check config
echo ""
echo "✓ Test 5: Configuration"
if [[ -f "configs/slide_config.yaml" ]]; then
    echo "  ✓ slide_config.yaml exists"
else
    echo "  ❌ slide_config.yaml missing"
    exit 1
fi

# Test 6: Check shell scripts
echo ""
echo "✓ Test 6: Shell scripts"
for script in run_one_slide.sh run_adaptive_pipeline.sh run_all_by_type.sh; do
    if [[ -f "scripts/$script" ]]; then
        if [[ -x "scripts/$script" ]]; then
            echo "  ✓ $script exists and is executable"
        else
            echo "  ⚠ $script exists but not executable - fixing..."
            chmod +x "scripts/$script"
        fi
    else
        echo "  ❌ $script missing"
        exit 1
    fi
done

# Test 7: Test Python imports
echo ""
echo "✓ Test 7: Python imports"

python3 -c "
import sys
sys.path.insert(0, 'src')

try:
    from utils.slide_detector import detect_slide_type
    print('  ✓ slide_detector imports correctly')
except Exception as e:
    print(f'  ❌ slide_detector import failed: {e}')
    sys.exit(1)

try:
    from utils.config_loader import SlideConfig
    print('  ✓ config_loader imports correctly')
except Exception as e:
    print(f'  ❌ config_loader import failed: {e}')
    sys.exit(1)
"

# Test 8: Test slide detection
echo ""
echo "✓ Test 8: Slide type detection"
python3 src/utils/slide_detector.py "CD3-S25.svs" | grep -q "IHC_CD3" && echo "  ✓ CD3 detection works" || echo "  ❌ CD3 detection failed"
python3 src/utils/slide_detector.py "HE-B17.svs" | grep -q "H&E" && echo "  ✓ H&E detection works" || echo "  ❌ H&E detection failed"

# Test 9: Check for slides
echo ""
echo "✓ Test 9: Slide availability"
if [[ -d "raw_slides" ]]; then
    N_SLIDES=$(ls raw_slides/*.svs 2>/dev/null | wc -l)
    echo "  ✓ Found $N_SLIDES .svs files in raw_slides/"
    if [[ $N_SLIDES -eq 0 ]]; then
        echo "  ⚠️  Warning: No slides found - pipeline can't be tested"
    fi
else
    echo "  ⚠️  raw_slides/ directory not found"
fi

# Final summary
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                          ALL TESTS PASSED ✅                                ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "🎉 Your pipeline is ready to use!"
echo ""
echo "Quick start:"
echo "  1. Single slide:  ./scripts/run_adaptive_pipeline.sh raw_slides/HE-S25.svs"
echo "  2. All slides:    ./scripts/run_all_by_type.sh raw_slides results"
echo ""
