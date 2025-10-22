#!/bin/bash

echo "=== Manual Test Script ==="
echo ""
echo "This will:"
echo "1. Kill any existing processes"
echo "2. Start the engine and GUI"
echo "3. Monitor logs in real-time"
echo "4. Wait for you to speak and confirm"
echo ""
read -p "Press Enter to start..."

# Cleanup
pkill -9 -f dictation 2>/dev/null || true
rm -f /tmp/voice-dictation*.sock /tmp/voice-dictation-state
sleep 1

# Start in background
RUST_LOG=info ~/.local/bin/dictation-engine > /tmp/engine-manual.log 2>&1 &
ENGINE_PID=$!
sleep 2

RUST_LOG=info ~/.local/bin/dictation-gui > /tmp/gui-manual.log 2>&1 &
GUI_PID=$!
sleep 1

echo ""
echo "✓ Started (Engine PID: $ENGINE_PID, GUI PID: $GUI_PID)"
echo ""
echo "You should now see the Wayland overlay with spectrum bars."
echo ""
echo "Instructions:"
echo "1. Speak CLEARLY and LOUDLY into your Arctis 7 mic"
echo "2. Watch the transcription below (updates every second)"
echo "3. Press Ctrl+C when done, then run:"
echo "   python3 ~/repos/voice-dictation-rust/scripts/send_confirm.py"
echo ""
echo "Monitoring transcription (speak now):"
echo "---"

# Monitor transcription
tail -f /tmp/gui-manual.log | grep --line-buffered "Transcription:" | while read line; do
    # Extract just the text
    text=$(echo "$line" | grep -oP "Transcription: '\K[^']*")
    if [ -n "$text" ]; then
        echo "  $text"
    fi
done
