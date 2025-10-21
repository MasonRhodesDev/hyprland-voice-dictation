# Testing dictation-engine

## Status

✅ **Phase 1 Complete: dictation-engine is WORKING!**

## What Works

- Audio capture from microphone (384kHz → 16kHz resampling)
- Voice Activity Detection (speech vs silence)
- Speech transcription via whisper.cpp
- Keyboard text injection via wtype
- Start/stop via control script

## How to Test

### Start dictation:
```bash
~/scripts/dictation-control start
```

### Check if running:
```bash
~/scripts/dictation-control status
```

### View logs:
```bash
tail -f /tmp/dictation-engine.log
```

### Stop dictation:
```bash
~/scripts/dictation-control stop
```

## How it Works

1. Engine listens continuously for audio input
2. VAD detects when you start speaking (threshold: -40dB)
3. Records audio until 800ms of silence
4. Transcribes speech with whisper.cpp
5. Types the transcription via wtype

## Current Configuration

- Sample rate: 16kHz
- VAD threshold: -40dB
- Silence duration: 800ms (24 frames @ 30fps)
- Typing delay: 10ms between characters
- Word delay: 50ms between words
- Whisper model: base.en (142MB)

## Known Issues

- No GUI yet (Phase 2)
- No IPC server yet (needed for GUI)
- Runs in background, no visual feedback

## Next Steps

- Phase 2: Implement Wayland GUI with spectrum visualization
- Phase 3: Add IPC between engine and GUI
- Phase 4: Integrate with Hyprland keybind (Meh+V)
