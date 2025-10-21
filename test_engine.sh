#!/bin/bash
set -e

echo "=== Testing dictation-engine with sample audio ==="

# Create a test audio file (sine wave tone)
echo "1. Generating test audio (3 second tone)..."
sox -n -r 16000 -c 1 /tmp/test_audio.wav synth 3 sine 440 vol 0.5

# Run engine with the test file
echo "2. Testing whisper transcription directly..."
whisper-cpp -m ~/repos/whisper.cpp/models/ggml-base.en.bin \
  -f ~/repos/whisper.cpp/samples/jfk.wav \
  -nt -np 2>&1 | grep "00:00" || true

echo ""
echo "3. Testing VAD with actual microphone..."
echo "   Checking audio levels..."

# Test if audio is being captured
parecord -d alsa_input.usb-SteelSeries_Arctis_Nova_7-00.mono-fallback \
  --rate=16000 --channels=1 --format=s16le /tmp/mic_test.wav &
RECORD_PID=$!
sleep 2
kill $RECORD_PID 2>/dev/null || true
wait $RECORD_PID 2>/dev/null || true

if [ -f /tmp/mic_test.wav ]; then
    SIZE=$(stat -c%s /tmp/mic_test.wav)
    echo "   Captured $SIZE bytes of audio"
    
    # Check RMS level
    sox /tmp/mic_test.wav -n stat 2>&1 | grep "RMS.*amplitude" || echo "   Audio captured"
else
    echo "   ERROR: Failed to capture audio"
fi

echo ""
echo "4. Issue: VAD threshold might be too strict or mic input is mono downsampled incorrectly"
echo "   Current threshold: -40dB"
echo "   Suggestion: Lower to -50dB or add debug output"
