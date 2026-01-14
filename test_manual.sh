#!/bin/bash

echo "=== Manual Test Script ==="
echo ""
echo "This will:"
echo "1. Kill any existing processes"
echo "2. Start the daemon (with integrated GUI)"
echo "3. Monitor logs in real-time"
echo "4. Wait for you to speak and confirm"
echo ""
read -p "Press Enter to start..."

# Cleanup
pkill -9 -f "voice-dictation daemon" 2>/dev/null || true
rm -f /tmp/voice-dictation*.sock /tmp/voice-dictation-state
sleep 1

# Start daemon in background (GUI is integrated)
RUST_LOG=info GUI_LOG=info ~/.local/bin/voice-dictation daemon > /tmp/dictation-engine.log 2>&1 &
DAEMON_PID=$!
sleep 2

echo ""
echo "✓ Started (Daemon PID: $DAEMON_PID)"
echo ""
echo "You should now see the Wayland overlay with spectrum bars."
echo ""
echo "Instructions:"
echo "1. Speak CLEARLY and LOUDLY into your microphone"
echo "2. Watch the transcription below (updates every second)"
echo "3. Press Ctrl+C when done, then confirm with:"
echo "   voice-dictation confirm"
echo ""
echo "Monitoring transcription (speak now):"
echo "---"

# Monitor transcription
tail -f /tmp/dictation-engine.log | grep --line-buffered "Transcription:" | while read line; do
    # Extract just the text
    text=$(echo "$line" | grep -oP "Transcription: '\K[^']*")
    if [ -n "$text" ]; then
        echo "  $text"
    fi
done
