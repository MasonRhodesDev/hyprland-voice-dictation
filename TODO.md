# TODO

## Completed ✓

- [x] Two-model architecture (fast preview + accurate correction)
- [x] Live transcription preview via IPC
- [x] Wayland overlay with spectrum visualization
- [x] Text rendering in overlay (fontdue)
- [x] Confirm-to-type workflow
- [x] Audio IPC for spectrum
- [x] Control IPC for transcription updates
- [x] Toggle script with state machine
- [x] Install/uninstall scripts
- [x] RPM packaging
- [x] Automated testing with virtual audio
- [x] Vosk integration (replaced Whisper.cpp)
- [x] Audio capture with cpal
- [x] Keyboard injection with wtype
- [x] VAD implementation

## Current Issues

- [ ] Fix text duplications (Vosk chunking creates overlaps)
- [ ] Prevent multiple GUI processes when keybind spammed
- [ ] Improve text rendering (wrapping, better font)
- [ ] First few words sometimes cut off

## Future Improvements

### Accuracy
- [ ] Add custom dictionary support for technical terms
- [ ] Implement n-gram language model boosting
- [ ] Add punctuation model
- [ ] Support for multiple languages
- [ ] Test alternative models (Whisper.cpp, Coqui STT)

### UX
- [ ] Visual indicator when voice detected (pulsing animation)
- [ ] Show confidence scores in overlay
- [ ] Keyboard shortcut to cancel/undo last dictation
- [ ] Edit transcription in overlay before typing
- [ ] Audio feedback (beep on start/stop)
- [ ] Show "Recording..." vs "Processing..." states

### Performance
- [ ] Lazy-load accurate model (only when needed)
- [ ] Cache frequently used words
- [ ] Optimize audio buffer management
- [ ] Reduce GUI render updates when text unchanged
- [ ] Process correction pass in background thread

### Features
- [ ] Voice commands ("new line", "delete last word", "undo")
- [ ] Integration with clipboard
- [ ] Dictate into specific applications
- [ ] Recording/playback for debugging
- [ ] Configuration file support (~/.config/voice-dictation/config.toml)
- [ ] Multiple model selection via config

### Packaging
- [ ] Add to AUR (Arch)
- [ ] Create .deb package (Ubuntu/Debian)
- [ ] Add to Nix packages
- [ ] Include systemd user services in RPM
- [ ] Add desktop entry file

## Known Bugs

1. **Text duplications**: Vosk's internal chunking causes overlapping audio segments
2. **Cut-off start**: First words sometimes missed if speech starts immediately
3. **Multiple processes**: Keybind spam launches multiple GUIs (need debouncing)
4. **No error feedback**: User doesn't know if transcription failed

## Architecture Notes

- Engine uses two Vosk models: small (40MB, fast) and large (1.8GB, accurate)
- IPC uses two Unix sockets: audio samples + control messages
- Control socket protocol: JSON messages with length prefix
- Audio socket protocol: raw f32 samples in 512-sample chunks
- State machine: stopped → recording → (confirm) → typing → stopped
