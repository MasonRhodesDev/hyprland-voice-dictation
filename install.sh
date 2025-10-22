#!/bin/bash
set -e

echo "=== Installing Voice Dictation System ==="
echo ""

# Build release binaries
echo "1. Building binaries..."
cargo build --release

# Install binaries
echo ""
echo "2. Installing binaries to ~/.local/bin..."
cargo install --path dictation-engine --root ~/.local --force
cargo install --path dictation-gui --root ~/.local --force

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
pkill -9 -f dictation-engine 2>/dev/null || true
pkill -9 -f dictation-gui 2>/dev/null || true
rm -f /tmp/voice-dictation-active /tmp/voice-dictation-state
rm -f /tmp/voice-dictation*.sock

echo ""
echo "✓ Installation complete!"
echo ""
echo "Usage:"
echo "  - Start recording:  ~/scripts/dictation-control toggle  (or MEH+v)"
echo "  - Stop & type:      ~/scripts/dictation-control toggle  (or MEH+v again)"
echo "  - Check status:     ~/scripts/dictation-control status"
echo ""
echo "Note: Make sure your Hyprland keybind points to:"
echo "  bind=\$Meh, V, exec, ~/scripts/dictation-control toggle"
