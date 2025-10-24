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
cp scripts/send_confirm.py ~/scripts/
chmod +x ~/scripts/dictation-control
chmod +x ~/scripts/send_confirm.py

# Cleanup old state
echo ""
echo "4. Cleaning up old state files..."
pkill -9 -f voice-dictation 2>/dev/null || true
rm -f /tmp/voice-dictation-active /tmp/voice-dictation-state
rm -f /tmp/voice-dictation*.sock

echo ""
echo "✓ Installation complete!"
echo ""
echo "Usage:"
echo "  - Direct:           voice-dictation toggle"
echo "  - Via script:       ~/scripts/dictation-control toggle"
echo "  - Check status:     voice-dictation status"
echo ""
echo "Note: Make sure your Hyprland keybind points to:"
echo "  bind=\$Meh, V, exec, voice-dictation toggle"
echo ""
echo "Available commands:"
echo "  voice-dictation toggle   - Start recording or confirm transcription"
echo "  voice-dictation start    - Start recording session"
echo "  voice-dictation stop     - Stop recording session"
echo "  voice-dictation confirm  - Confirm and finalize transcription"
echo "  voice-dictation status   - Show current status"
