#!/bin/bash

echo "fast"
echo "accurate"

MODEL_DIR="$HOME/.config/voice-dictation/models"
if [ -d "$MODEL_DIR" ]; then
    for model in "$MODEL_DIR"/vosk-model-*; do
        if [ -d "$model" ]; then
            basename "$model"
        fi
    done
fi
