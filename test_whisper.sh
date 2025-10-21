#!/bin/bash
# Test whisper.cpp with sample audio
echo "Testing whisper.cpp transcription..."
whisper-cpp -m ~/repos/whisper.cpp/models/ggml-base.en.bin \
  -f ~/repos/whisper.cpp/samples/jfk.wav \
  -nt -np 2>&1 | grep -A1 "00:00:00"
