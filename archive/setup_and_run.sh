#!/bin/bash

# === Configuration ===
ENV_NAME="venv"
REQUIREMENTS_FILE="requirements.txt"
SLIDE="CD3-S25"
RAW_SLIDE="data/raw/${SLIDE}.svs"
PREVIEW_DIR="results/${SLIDE}/00_preview"
SCRIPT_PATH="src/00_quick_preview_enhanced.py"

echo "🚀 Setting up virtual environment..."

# === Create virtual environment ===
python3 -m venv $ENV_NAME
source $ENV_NAME/bin/activate

echo "✅ Virtual environment activated: $ENV_NAME"

# === Install requirements ===
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "📦 Installing from $REQUIREMENTS_FILE..."
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "⚠️ No requirements.txt found — using inline packages..."
    pip install numpy pandas matplotlib tqdm opencv-python-headless Pillow openslide-python scikit-image scipy cellpose
fi

# === Verify raw slide exists ===
if [ ! -f "$RAW_SLIDE" ]; then
    echo "❌ Slide file not found: $RAW_SLIDE"
    echo "Please make sure the slide is available before running."
    exit 1
fi

# === Run preview generation ===
echo "🖼️ Generating preview for: $SLIDE"
python3 "$SCRIPT_PATH" \
  --raw "$RAW_SLIDE" \
  --out_dir "$PREVIEW_DIR" \
  --crop_size 1024

echo "✅ Preview complete. Check output in: $PREVIEW_DIR"
