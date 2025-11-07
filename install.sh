#!/bin/bash
set -e

echo "=== Installing Voice Dictation System ==="
echo ""

# Build release binaries
echo "1. Building binary..."
cargo build --release

# Install binary
echo ""
echo "2. Installing binary to ~/.local/bin..."
cargo install --path . --root ~/.local --force

# Copy control scripts
echo ""
echo "3. Installing control scripts to ~/scripts..."
mkdir -p ~/scripts
cp scripts/dictation-control ~/scripts/
chmod +x ~/scripts/dictation-control

# Setup config directory and download default models
echo ""
echo "4. Setting up configuration and models..."
mkdir -p "$HOME/.config/voice-dictation/models"

cd "$HOME/.config/voice-dictation/models"

# Download default preview model (fast)
if [ ! -d "vosk-model-en-us-daanzu-20200905-lgraph" ]; then
    echo "  Downloading preview model (fast, ~130MB)..."
    curl -L -O https://alphacephei.com/vosk/models/vosk-model-en-us-daanzu-20200905-lgraph.zip
    unzip -q vosk-model-en-us-daanzu-20200905-lgraph.zip
    rm vosk-model-en-us-daanzu-20200905-lgraph.zip
    echo "  ✓ Preview model installed"
else
    echo "  ✓ Preview model already exists"
fi

# Download default final model (accurate)
if [ ! -d "vosk-model-en-us-0.22" ]; then
    echo "  Downloading final model (accurate, ~1.8GB)..."
    curl -L -O https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
    unzip -q vosk-model-en-us-0.22.zip
    rm vosk-model-en-us-0.22.zip
    echo "  ✓ Final model installed"
else
    echo "  ✓ Final model already exists"
fi

cd - > /dev/null

# Install systemd service
echo ""
echo "5. Installing systemd service..."
mkdir -p "$HOME/.config/systemd/user"
cp packaging/systemd/voice-dictation.service "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
echo "  ✓ Service installed"

# Cleanup old state
echo ""
echo "6. Cleaning up old state files..."
pkill -9 -f voice-dictation 2>/dev/null || true
rm -f /tmp/voice-dictation-active /tmp/voice-dictation-state
rm -f /tmp/voice-dictation*.sock

echo ""
echo "✓ Installation complete!"
echo ""
echo "=== Starting the Service ==="
echo ""
echo "Enable and start the daemon service:"
echo "  systemctl --user enable voice-dictation"
echo "  systemctl --user start voice-dictation"
echo ""
echo "Or run manually for testing:"
echo "  voice-dictation daemon"
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
echo "Add a keybind in your compositor config:"
echo ""
echo "  Hyprland:  bind=\$Meh, V, exec, voice-dictation toggle"
echo "  Sway:      bindsym Mod4+Shift+Alt+v exec voice-dictation toggle"
echo "  KDE/GNOME: Use Settings → Keyboard → Custom Shortcuts"
echo ""
echo "Available commands:"
echo "  voice-dictation toggle   - Start recording or confirm transcription"
echo "  voice-dictation start    - Start recording session"
echo "  voice-dictation stop     - Stop recording session"
echo "  voice-dictation confirm  - Confirm and finalize transcription"
echo "  voice-dictation status   - Show current status"
echo "  voice-dictation config   - Open configuration TUI"
