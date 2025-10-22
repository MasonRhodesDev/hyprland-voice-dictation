#!/bin/bash
set -e

BASE_URL="https://alphacephei.com/vosk/models"

download() {
    local model=$1
    
    [ -d "$model" ] && { echo "✓ $model exists"; return; }
    
    if [ ! -f "$model.zip" ]; then
        echo "Downloading $model..."
        wget -q --show-progress "$BASE_URL/$model.zip" || \
        curl -# -L -O "$BASE_URL/$model.zip"
    fi
    
    echo "Extracting $model..."
    unzip -q "$model.zip"
    echo "✓ $model ready"
    echo ""
}

echo "Downloading Vosk models (2GB total)..."
echo ""
download "vosk-model-small-en-us-0.15"
download "vosk-model-en-us-0.22"
echo "✓ All models downloaded"
