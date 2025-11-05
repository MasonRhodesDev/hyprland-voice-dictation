#!/bin/bash
#
# Download Whisper models for voice dictation
#
# This script downloads GGML-format Whisper models from Hugging Face
# and installs them to ~/.config/voice-dictation/models/whisper/
#
# Models:
# - ggml-base.en.bin (142MB) - Fast, good for preview
# - ggml-small.en.bin (466MB) - Accurate, recommended for final pass
#

set -e

MODELS_DIR="${HOME}/.config/voice-dictation/models/whisper"
BASE_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Whisper Model Downloader${NC}"
echo "Target directory: $MODELS_DIR"
echo

# Create directory
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

# Download base model (for preview, optional)
if [ -f "ggml-base.en.bin" ]; then
    echo -e "${GREEN}✓${NC} ggml-base.en.bin already exists (142MB)"
else
    echo "Downloading ggml-base.en.bin (142MB)..."
    wget -q --show-progress "${BASE_URL}/ggml-base.en.bin" || {
        echo "Error: Failed to download base model"
        exit 1
    }
    echo -e "${GREEN}✓${NC} Downloaded ggml-base.en.bin"
fi

# Download small model (for accurate pass, recommended)
if [ -f "ggml-small.en.bin" ]; then
    echo -e "${GREEN}✓${NC} ggml-small.en.bin already exists (466MB)"
else
    echo "Downloading ggml-small.en.bin (466MB)..."
    wget -q --show-progress "${BASE_URL}/ggml-small.en.bin" || {
        echo "Error: Failed to download small model"
        exit 1
    }
    echo -e "${GREEN}✓${NC} Downloaded ggml-small.en.bin"
fi

# Optional: Download tiny model (fastest, least accurate)
read -p "Download ggml-tiny.en.bin (75MB) for testing? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "ggml-tiny.en.bin" ]; then
        echo -e "${GREEN}✓${NC} ggml-tiny.en.bin already exists"
    else
        echo "Downloading ggml-tiny.en.bin (75MB)..."
        wget -q --show-progress "${BASE_URL}/ggml-tiny.en.bin" || {
            echo "Warning: Failed to download tiny model (non-critical)"
        }
        echo -e "${GREEN}✓${NC} Downloaded ggml-tiny.en.bin"
    fi
fi

echo
echo -e "${GREEN}✓ Whisper models ready!${NC}"
echo
echo "Models installed to: $MODELS_DIR"
echo
echo "To use Whisper, add to ~/.config/voice-dictation/config.toml:"
echo
echo "  [daemon]"
echo "  transcription_engine = \"whisper\""
echo
echo "To switch back to Vosk:"
echo
echo "  [daemon]"
echo "  transcription_engine = \"vosk\""
echo
