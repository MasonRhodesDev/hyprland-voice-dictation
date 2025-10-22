#!/bin/bash

echo "=== Uninstalling Voice Dictation System ==="
echo ""

# Kill ALL processes
echo "1. Killing all dictation processes..."
pkill -9 -f dictation-engine 2>/dev/null && echo "   - Killed dictation-engine processes" || echo "   - No dictation-engine processes"
pkill -9 -f dictation-gui 2>/dev/null && echo "   - Killed dictation-gui processes" || echo "   - No dictation-gui processes"
sleep 1

# Show remaining processes
REMAINING=$(ps aux | grep -E "dictation" | grep -v grep | wc -l)
if [ "$REMAINING" -gt 0 ]; then
    echo "   WARNING: $REMAINING processes still running:"
    ps aux | grep -E "dictation" | grep -v grep
    echo ""
    read -p "   Force kill these? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        killall -9 dictation-engine dictation-gui 2>/dev/null
    fi
fi

# Remove state and socket files
echo ""
echo "2. Removing state files..."
rm -f /tmp/voice-dictation-active /tmp/voice-dictation-state
rm -f /tmp/voice-dictation*.sock
rm -f /tmp/dictation*.log
echo "   - Cleaned /tmp/"

# Remove binaries
echo ""
echo "3. Removing binaries..."
rm -f ~/.local/bin/dictation-engine
rm -f ~/.local/bin/dictation-gui
echo "   - Removed from ~/.local/bin/"

# Remove scripts
echo ""
echo "4. Removing control scripts..."
rm -f ~/scripts/dictation-control
rm -f ~/scripts/send_confirm.py
echo "   - Removed from ~/scripts/"

echo ""
echo "✓ Uninstall complete!"
echo ""
echo "Note: Source code in ~/repos/voice-dictation-rust/ was NOT removed."
echo "To reinstall: cd ~/repos/voice-dictation-rust && ./install.sh"
