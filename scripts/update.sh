#!/bin/bash
set -e

REPO="voice-dictation-rust"
OWNER=$(echo "$GITHUB_REPOSITORY" | cut -d'/' -f1)
REPO_NAME=$(echo "$GITHUB_REPOSITORY" | cut -d'/' -f2)

if [ -z "$OWNER" ] || [ -z "$REPO_NAME" ]; then
    OWNER="${GITHUB_ACTOR:-mason}"
    REPO_NAME="voice-dictation-rust"
fi

echo "=== Voice Dictation Update Script ==="
echo ""

get_latest_release() {
    curl -s "https://api.github.com/repos/${OWNER}/${REPO_NAME}/releases/latest" | \
        grep '"tag_name":' | \
        sed -E 's/.*"v([^"]+)".*/\1/'
}

echo "Fetching latest release..."
LATEST_VERSION=$(get_latest_release)

if [ -z "$LATEST_VERSION" ]; then
    echo "Error: Could not fetch latest release version"
    exit 1
fi

echo "Latest version: $LATEST_VERSION"
echo ""

DOWNLOAD_URL="https://github.com/${OWNER}/${REPO_NAME}/releases/download/v${LATEST_VERSION}/voice-dictation-${LATEST_VERSION}-x86_64-linux.tar.gz"
TMP_DIR=$(mktemp -d)

echo "Downloading release..."
curl -fsSL "$DOWNLOAD_URL" -o "$TMP_DIR/voice-dictation.tar.gz"

echo "Extracting..."
tar -xzf "$TMP_DIR/voice-dictation.tar.gz" -C "$TMP_DIR"

echo "Stopping running instances..."
pkill -9 -f dictation-engine 2>/dev/null || true
pkill -9 -f dictation-gui 2>/dev/null || true

echo "Installing binaries..."
mkdir -p ~/.local/bin
cp "$TMP_DIR/voice-dictation-${LATEST_VERSION}/bin/dictation-engine" ~/.local/bin/
cp "$TMP_DIR/voice-dictation-${LATEST_VERSION}/bin/dictation-gui" ~/.local/bin/
chmod +x ~/.local/bin/dictation-engine
chmod +x ~/.local/bin/dictation-gui

echo "Installing scripts..."
mkdir -p ~/scripts
cp "$TMP_DIR/voice-dictation-${LATEST_VERSION}/scripts/dictation-control" ~/scripts/
cp "$TMP_DIR/voice-dictation-${LATEST_VERSION}/scripts/send_confirm.py" ~/scripts/
chmod +x ~/scripts/dictation-control
chmod +x ~/scripts/send_confirm.py

echo "Cleaning up..."
rm -rf "$TMP_DIR"

echo ""
echo "✓ Update complete! Version: $LATEST_VERSION"
echo ""
echo "The voice dictation system has been updated."
echo "Use '~/scripts/dictation-control toggle' to start using it."
