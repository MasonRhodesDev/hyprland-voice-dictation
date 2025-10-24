#!/bin/bash
# Lists available Vosk models filtered by language
# Usage: list-vosk-models.sh <language_code>
# Example: list-vosk-models.sh en

CACHE_FILE="/tmp/vosk-models-cache.txt"
CACHE_TTL=86400  # 24 hours
LANG_CODE="${1:-en}"

# Fetch and cache all models if cache is stale
if [ ! -f "$CACHE_FILE" ] || [ $(find "$CACHE_FILE" -mtime +1 2>/dev/null | wc -l) -gt 0 ]; then
    curl -s https://alphacephei.com/vosk/models 2>/dev/null | \
        grep -o 'vosk-model-[^"]*\.zip' | \
        sed 's/\.zip$//' | \
        sort -u > "$CACHE_FILE" 2>/dev/null
fi

# If cache exists, filter by language
if [ -f "$CACHE_FILE" ]; then
    grep "^vosk-model-${LANG_CODE}-" "$CACHE_FILE" 2>/dev/null | sort
fi

# Check locally installed models
LOCAL_MODELS_DIR="$HOME/.config/voice-dictation/models"
if [ -d "$LOCAL_MODELS_DIR" ]; then
    find "$LOCAL_MODELS_DIR" -maxdepth 1 -type d -name "vosk-model-${LANG_CODE}-*" 2>/dev/null | \
        xargs -n1 basename 2>/dev/null | sort -u
fi

# Always include custom option at the end
echo "custom"
