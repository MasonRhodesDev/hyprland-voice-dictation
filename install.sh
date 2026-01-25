#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOSK_VERSION="0.3.45"
VOSK_DIR="$SCRIPT_DIR/.vosk"
INSTALL_LIB_DIR="$HOME/.local/lib"

echo "=== Installing Voice Dictation System ==="
echo ""

# Check dependencies first
echo "Checking dependencies..."
if ! bash "$SCRIPT_DIR/scripts/check-deps.sh"; then
    echo ""
    echo "❌ Dependency check failed. Please install missing dependencies and try again."
    exit 1
fi
echo ""

# Download libvosk if needed
echo "=== Step 1: Checking libvosk ==="
if [ ! -f "$VOSK_DIR/libvosk.so" ]; then
    echo "  Downloading libvosk $VOSK_VERSION..."
    mkdir -p "$VOSK_DIR"
    cd "$VOSK_DIR"
    wget -q "https://github.com/alphacep/vosk-api/releases/download/v${VOSK_VERSION}/vosk-linux-x86_64-${VOSK_VERSION}.zip" -O vosk.zip
    unzip -q vosk.zip
    mv vosk-linux-x86_64-${VOSK_VERSION}/* .
    rmdir vosk-linux-x86_64-${VOSK_VERSION}
    rm vosk.zip
    cd "$SCRIPT_DIR"
    echo "  ✓ libvosk downloaded"
else
    echo "  ✓ libvosk already exists"
fi

# Set up build environment
export LIBRARY_PATH="$VOSK_DIR:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$VOSK_DIR:$LD_LIBRARY_PATH"

# Build release binary with all features
echo ""
echo "=== Step 2: Building binary (all features) ==="
cargo build --release

# Install libvosk to user lib directory
echo ""
echo "=== Step 3: Installing libvosk ==="
mkdir -p "$INSTALL_LIB_DIR"
cp "$VOSK_DIR/libvosk.so" "$INSTALL_LIB_DIR/"
echo "  ✓ libvosk installed to $INSTALL_LIB_DIR"

# Install binary
echo ""
echo "=== Step 4: Installing binary to ~/.local/bin ==="
cargo install --path . --root ~/.local --force

# Copy control scripts
echo ""
echo "=== Step 5: Installing scripts ==="
mkdir -p ~/scripts
cp scripts/dictation-control ~/scripts/
chmod +x ~/scripts/dictation-control
echo "  ✓ Scripts installed"

# Setup config directory
echo ""
echo "=== Step 6: Setting up configuration ==="
mkdir -p "$HOME/.config/voice-dictation/models"
echo "  ✓ Config directory ready"

# Install systemd service with LD_LIBRARY_PATH
echo ""
echo "=== Step 7: Installing systemd service ==="
mkdir -p "$HOME/.config/systemd/user"

# Create service file with library path and Wayland environment
cat > "$HOME/.config/systemd/user/voice-dictation.service" << EOF
[Unit]
Description=Voice Dictation Persistent Daemon with Integrated GUI
After=pipewire.service graphical-session.target
PartOf=graphical-session.target

[Service]
Type=simple
ExecStart=%h/.local/bin/voice-dictation daemon
Restart=on-failure
RestartSec=5
# Exit code 64 = UI reload requested, should trigger restart
RestartForceExitStatus=64
Environment="RUST_LOG=info"
Environment="GUI_LOG=info"
Environment="LD_LIBRARY_PATH=$INSTALL_LIB_DIR"
# Import Wayland/graphical environment
ImportEnvironment=WAYLAND_DISPLAY XDG_RUNTIME_DIR DISPLAY
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=graphical-session.target
EOF

# Import current graphical environment for systemd
systemctl --user import-environment WAYLAND_DISPLAY XDG_RUNTIME_DIR DISPLAY

systemctl --user daemon-reload
echo "  ✓ Service installed"

# Cleanup old state
echo ""
echo "=== Step 8: Cleaning up old state files ==="
pkill -9 -f "voice-dictation daemon" 2>/dev/null || true
rm -f /tmp/voice-dictation-active /tmp/voice-dictation-state
rm -f /tmp/voice-dictation*.sock

# Restart daemon if it was enabled
if systemctl --user is-enabled voice-dictation &>/dev/null; then
    echo ""
    echo "=== Step 9: Restarting daemon ==="
    systemctl --user restart voice-dictation
    echo "  ✓ Daemon restarted"
fi

echo ""
echo "✓ Installation complete!"
echo ""
echo "=== Library Path Setup ==="
echo ""
echo "Add to your shell profile (~/.bashrc or ~/.zshrc):"
echo "  export LD_LIBRARY_PATH=\"$INSTALL_LIB_DIR:\$LD_LIBRARY_PATH\""
echo ""
echo "=== Starting the Service ==="
echo ""
echo "Enable and start the daemon service:"
echo "  systemctl --user enable voice-dictation"
echo "  systemctl --user start voice-dictation"
echo ""
echo "Or run manually for testing:"
echo "  LD_LIBRARY_PATH=$INSTALL_LIB_DIR voice-dictation daemon"
echo ""
echo "=== Usage ==="
echo ""
echo "Commands:"
echo "  voice-dictation toggle   - Start recording or confirm transcription"
echo "  voice-dictation start    - Start recording session"
echo "  voice-dictation stop     - Stop recording session (cancel)"
echo "  voice-dictation confirm  - Confirm and finalize transcription"
echo "  voice-dictation status   - Show current status"
echo "  voice-dictation config   - Open configuration TUI"
echo ""
