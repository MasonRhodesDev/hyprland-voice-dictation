#!/bin/bash
# Download the Parakeet TDT 0.6b ONNX model from HuggingFace
# Used as a standalone alternative to: voice-dictation download-model

set -euo pipefail

MODEL_DIR="${HOME}/.config/voice-dictation/models/parakeet"
BASE_URL="https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main"
FILES=(
    "encoder-model.onnx"
    "encoder-model.onnx.data"
    "decoder_joint-model.onnx"
)

mkdir -p "$MODEL_DIR"
echo "Model directory: $MODEL_DIR"
echo "Source: $BASE_URL"
echo

for file in "${FILES[@]}"; do
    dest="$MODEL_DIR/$file"
    if [[ -f "$dest" && -s "$dest" ]]; then
        size=$(du -h "$dest" | cut -f1)
        echo "  $file — already exists ($size), skipping"
        continue
    fi
    echo "  Downloading $file..."
    curl --progress-bar --location --output "$dest" "$BASE_URL/$file"
done

echo
echo "Model download complete."
echo "You can now start the daemon: systemctl --user start voice-dictation"
