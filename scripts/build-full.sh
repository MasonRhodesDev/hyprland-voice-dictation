#!/bin/bash
# Build voice-dictation with all features
# Downloads libvosk if not present

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VOSK_VERSION="0.3.45"
VOSK_DIR="$PROJECT_DIR/.vosk"

# Download libvosk if needed
if [ ! -f "$VOSK_DIR/libvosk.so" ]; then
    echo "Downloading libvosk $VOSK_VERSION..."
    mkdir -p "$VOSK_DIR"
    cd "$VOSK_DIR"
    wget -q "https://github.com/alphacep/vosk-api/releases/download/v${VOSK_VERSION}/vosk-linux-x86_64-${VOSK_VERSION}.zip" -O vosk.zip
    unzip -q vosk.zip
    mv vosk-linux-x86_64-${VOSK_VERSION}/* .
    rmdir vosk-linux-x86_64-${VOSK_VERSION}
    rm vosk.zip
    echo "libvosk downloaded to $VOSK_DIR"
fi

# Set up environment for vosk
export LIBRARY_PATH="$VOSK_DIR:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$VOSK_DIR:$LD_LIBRARY_PATH"

# Build with all features
cd "$PROJECT_DIR"
echo "Building with all features..."
cargo build --release --features "vosk,parakeet"

echo ""
echo "Build complete!"
echo "Binary: $PROJECT_DIR/target/release/voice-dictation"
echo ""
echo "To run, set: export LD_LIBRARY_PATH=$VOSK_DIR:\$LD_LIBRARY_PATH"
