# Implementation Checklist

## Phase 0: Setup & Dependencies

- [ ] Verify whisper.cpp installation
  - [ ] Clone repo to `~/repos/whisper.cpp`
  - [ ] Build binary: `make`
  - [ ] Download base.en model
  - [ ] Symlink to `~/.local/bin/whisper`
  - [ ] Test: `whisper --help`

- [ ] Verify system dependencies
  - [ ] Check wtype: `which wtype`
  - [ ] Check PulseAudio/PipeWire: `pactl list sources`
  - [ ] Check Wayland compositor: `echo $WAYLAND_DISPLAY`

- [ ] Setup Rust workspace
  - [x] Create workspace `Cargo.toml`
  - [ ] Create `dictation-engine/` subdirectory
  - [ ] Create `dictation-gui/` subdirectory
  - [ ] Create `scripts/` subdirectory

## Phase 1: dictation-engine Core

### 1.1 Audio Capture
- [ ] Add cpal dependency
- [ ] Implement `audio.rs`
  - [ ] Get default input device
  - [ ] Create audio stream (16kHz, mono, f32)
  - [ ] Setup ring buffer (5s capacity)
  - [ ] Handle stream errors

### 1.2 Voice Activity Detection
- [ ] Implement `vad.rs`
  - [ ] Energy-based RMS calculation
  - [ ] dB conversion
  - [ ] Frame-based state tracking
  - [ ] Hysteresis (3 frames speech, 24 frames silence)
  - [ ] Export audio segments on silence

### 1.3 Whisper Integration
- [ ] Implement `whisper.rs`
  - [ ] Write audio buffer to WAV file
  - [ ] Spawn whisper.cpp subprocess
  - [ ] Parse JSON output
  - [ ] Handle subprocess errors
  - [ ] Cleanup temporary files

### 1.4 Keyboard Injection
- [ ] Implement `keyboard.rs`
  - [ ] Spawn wtype subprocess
  - [ ] Type text word-at-a-time
  - [ ] Add inter-word delays
  - [ ] Handle special characters

### 1.5 IPC Server
- [ ] Implement `ipc.rs`
  - [ ] Create Unix domain socket
  - [ ] Accept client connections
  - [ ] Broadcast audio samples at 60Hz
  - [ ] Non-blocking sends
  - [ ] Graceful disconnect handling

### 1.6 Main Loop
- [ ] Implement `main.rs`
  - [ ] Parse CLI arguments
  - [ ] Load configuration
  - [ ] Initialize audio capture
  - [ ] Start IPC server
  - [ ] Run VAD loop
  - [ ] Handle signals (SIGTERM, SIGINT)
  - [ ] Cleanup on exit

## Phase 2: dictation-gui Core

### 2.1 Wayland Layer-Shell
- [ ] Add smithay-client-toolkit dependency
- [ ] Implement `wayland.rs`
  - [ ] Connect to Wayland compositor
  - [ ] Create layer-shell surface
  - [ ] Configure surface (overlay, bottom-center)
  - [ ] Setup EGL/wgpu context
  - [ ] Handle compositor events

### 2.2 FFT Analysis
- [ ] Add rustfft dependency
- [ ] Implement `fft.rs`
  - [ ] Apply Hanning window
  - [ ] Compute FFT (512 samples)
  - [ ] Extract magnitude spectrum
  - [ ] Map to 8 frequency bands
  - [ ] Smooth band values (60% prev + 40% new)
  - [ ] Normalize to 0.0-1.0

### 2.3 Rendering
- [ ] Choose rendering backend (wgpu vs tiny-skia)
- [ ] Implement `renderer.rs`
  - [ ] Initialize graphics context
  - [ ] Create pill-shaped background (200x50, rounded)
  - [ ] Draw 8 vertical bars
  - [ ] Map normalized values to heights (5-30px)
  - [ ] Apply Catppuccin colors
  - [ ] Render at 60fps

### 2.4 IPC Client
- [ ] Implement `ipc.rs`
  - [ ] Connect to Unix socket
  - [ ] Receive audio samples
  - [ ] Handle disconnects (auto-reconnect)
  - [ ] Exponential backoff on errors

### 2.5 Main Loop
- [ ] Implement `main.rs`
  - [ ] Parse CLI arguments
  - [ ] Initialize Wayland
  - [ ] Connect to IPC socket
  - [ ] Run event loop (Wayland + IPC)
  - [ ] Update FFT and render at 60fps
  - [ ] Handle signals (SIGTERM, SIGINT)

## Phase 3: Control Script

- [ ] Implement `scripts/dictation-control`
  - [ ] State file management (`/tmp/voice-dictation-active`)
  - [ ] Process launch/kill logic
  - [ ] Toggle function
  - [ ] Error handling (processes not found)

## Phase 4: Configuration

- [ ] Implement config loading
  - [ ] Create default config structure
  - [ ] Read from `~/.config/voice-dictation/config.toml`
  - [ ] Fallback to defaults if missing
  - [ ] Validate config values

- [ ] Generate default config file
  - [ ] Create example `config.toml`
  - [ ] Add inline documentation

## Phase 5: Integration & Testing

### 5.1 Unit Tests
- [ ] Test VAD algorithm
- [ ] Test FFT band mapping
- [ ] Test config parsing
- [ ] Test IPC protocol

### 5.2 Integration Tests
- [ ] Test audio capture → VAD → whisper
- [ ] Test whisper → keyboard injection
- [ ] Test IPC server ↔ client
- [ ] Test Wayland layer-shell rendering

### 5.3 End-to-End Testing
- [ ] Test full pipeline: speak → transcribe → type
- [ ] Test GUI spectrum visualization accuracy
- [ ] Test toggle script (start/stop)
- [ ] Test error recovery scenarios
- [ ] Test on actual Hyprland setup

## Phase 6: Polish & Deployment

- [ ] Performance optimization
  - [ ] Profile CPU usage
  - [ ] Profile memory usage
  - [ ] Optimize FFT computation
  - [ ] Reduce render overhead

- [ ] Error handling & logging
  - [ ] Add structured logging (tracing crate)
  - [ ] Log all errors to stderr
  - [ ] Add debug mode (verbose logging)

- [ ] Documentation
  - [ ] Add inline code comments
  - [ ] Update README with actual usage
  - [ ] Add troubleshooting guide
  - [ ] Document config options

- [ ] Installation
  - [ ] Build release binaries
  - [ ] Copy to `~/.local/bin/`
  - [ ] Copy control script to `~/scripts/`
  - [ ] Update Hyprland keybind config
  - [ ] Test clean install

## Phase 7: Hyprland Integration

- [ ] Update `~/.config/hypr/keybinds.conf`
  - [ ] Add Meh+V keybind
  - [ ] Reload Hyprland config

- [ ] Update layer-shell window rules (if needed)
  - [ ] Ensure overlay appears on all workspaces
  - [ ] Disable animations for overlay

## Phase 8: Migration from Python

- [ ] Stop old Python dictation system
  - [ ] Kill old processes
  - [ ] Remove old scripts from `~/scripts/`
  - [ ] Archive old code

- [ ] Update chezmoi dotfiles repo
  - [ ] Remove old Python scripts
  - [ ] Add new Rust binaries (if tracking)
  - [ ] Update control script
  - [ ] Commit changes

## Future Enhancements (Post-MVP)

- [ ] Multiple language support
- [ ] Custom vocabulary/phrases
- [ ] Voice commands (punctuation, formatting)
- [ ] Real-time streaming (no VAD pauses)
- [ ] GPU-accelerated whisper
- [ ] Hot-reload configuration
- [ ] Metrics dashboard
- [ ] Speaker diarization
- [ ] Punctuation auto-insertion

## Notes

- Use `tokio` async runtime for engine (IPC + subprocess management)
- Use `async-std` or `smol` for GUI (lighter weight)
- Consider `silero-vad` crate for better VAD (if energy-based is insufficient)
- Test with both PulseAudio and PipeWire
- Ensure all file paths use `shellexpand` for `~` expansion
