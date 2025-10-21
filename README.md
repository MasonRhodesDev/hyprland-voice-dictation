# Voice Dictation (Rust)

Fast local voice dictation with real-time spectrum visualization for Wayland/Hyprland.

## Architecture

```
┌─────────────────────┐
│  dictation-control  │  Toggle script (bash)
│   (Meh+V keybind)   │  Manages lifecycle
└──────────┬──────────┘
           │
    ┌──────┴───────┐
    │              │
┌───▼────────┐ ┌──▼─────────────┐
│    GUI     │ │     Engine     │
│  (Wayland  │ │   (Audio →     │
│   overlay) │ │  Whisper.cpp)  │
└────────────┘ └────────────────┘
     │                  │
     └──────┬───────────┘
            │
      Unix Socket
    (Audio samples)
```

## Components

### 1. dictation-engine (Rust binary)
**Purpose:** Audio capture, speech recognition, text injection

**Tech stack:**
- `cpal` - Cross-platform audio I/O
- `whisper.cpp` - Speech-to-text (subprocess)
- `wtype` - Wayland keyboard injection
- `tokio` - Async runtime

**Flow:**
1. Capture audio continuously
2. VAD (Voice Activity Detection) detects speech/silence
3. On silence → save WAV → call whisper.cpp → get text
4. Type transcription word-at-a-time (respects natural pauses)
5. Broadcast audio samples to GUI via Unix socket

### 2. dictation-gui (Rust binary)
**Purpose:** Non-interactive Wayland overlay with spectrum visualization

**Tech stack:**
- `smithay-client-toolkit` - Wayland layer-shell protocol
- `wgpu` or `tiny-skia` - Rendering engine
- `rustfft` - FFT spectrum analysis
- `tokio` - Async socket client

**Flow:**
1. Create Wayland layer-shell surface (top layer, no keyboard input)
2. Position at bottom-center (200x50px pill)
3. Read audio samples from Unix socket
4. Compute 8-band FFT spectrum (100Hz-7kHz)
5. Render animated bars at 60fps

### 3. dictation-control (Bash script)
**Purpose:** Lifecycle management for Hyprland keybind

**Flow:**
- Start: Launch engine + GUI, create state file
- Stop: Signal both processes, cleanup state
- Toggle: Check state file, start or stop

## Features

- **Whisper.cpp** - State-of-the-art local speech recognition
- **Word-at-a-time typing** - Natural pauses preserved, no word smashing
- **Real-time spectrum** - 8-band frequency visualization
- **Wayland layer-shell** - Non-intrusive overlay, doesn't steal focus
- **Fast startup** - Rust binaries, <100ms launch time
- **Modular** - Swap components independently

## Dependencies

### System Requirements
- Wayland compositor (Hyprland, Sway, etc.)
- PulseAudio or PipeWire
- `whisper.cpp` binary
- `wtype` (Wayland keyboard tool)

### Rust Dependencies
See individual `Cargo.toml` files in:
- `dictation-engine/`
- `dictation-gui/`

## Setup

### 1. Install whisper.cpp
```bash
# Clone and build whisper.cpp
cd ~/repos
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
make

# Download model (base.en for speed/accuracy balance)
bash ./models/download-ggml-model.sh base.en

# Symlink binary to PATH
ln -s ~/repos/whisper.cpp/main ~/.local/bin/whisper
```

### 2. Install wtype
```bash
# Fedora
sudo dnf install wtype

# Arch
sudo pacman -S wtype
```

### 3. Build voice-dictation
```bash
cd ~/repos/voice-dictation-rust
cargo build --release

# Install binaries
cp target/release/dictation-engine ~/.local/bin/
cp target/release/dictation-gui ~/.local/bin/

# Install control script
cp scripts/dictation-control ~/scripts/
chmod +x ~/scripts/dictation-control
```

### 4. Configure Hyprland keybind
Add to `~/.config/hypr/keybinds.conf`:
```
bind = CTRL_ALT_SHIFT, V, exec, ~/scripts/dictation-control toggle
```

## Usage

- Press `Meh+V` (Ctrl+Alt+Shift+V) to toggle dictation
- Speak naturally with pauses between phrases
- GUI shows spectrum visualization when active
- Press `Meh+V` again to stop

## Project Structure

```
voice-dictation-rust/
├── README.md              # This file
├── ARCHITECTURE.md        # Technical design document
├── TODO.md               # Implementation checklist
├── Cargo.toml            # Workspace configuration
├── dictation-engine/     # Audio → Whisper → Keyboard
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs       # Entry point
│       ├── audio.rs      # cpal audio capture
│       ├── vad.rs        # Voice activity detection
│       ├── whisper.rs    # whisper.cpp interface
│       ├── keyboard.rs   # wtype text injection
│       └── ipc.rs        # Unix socket server
├── dictation-gui/        # Wayland overlay + spectrum
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs       # Entry point
│       ├── wayland.rs    # Layer-shell setup
│       ├── renderer.rs   # Spectrum visualization
│       ├── fft.rs        # FFT audio analysis
│       └── ipc.rs        # Unix socket client
└── scripts/
    └── dictation-control # Bash toggle script
```

## Configuration

Configuration options will be read from `~/.config/voice-dictation/config.toml`:

```toml
[whisper]
model_path = "~/.local/share/whisper/models/ggml-base.en.bin"
language = "en"

[audio]
sample_rate = 16000
channels = 1

[vad]
# Energy threshold for speech detection
energy_threshold = 0.02
# Silence duration before ending transcription (ms)
silence_duration = 800

[gui]
width = 200
height = 50
position = "bottom-center"
offset_x = -100  # Center offset adjustment
offset_y = 50    # Pixels from bottom

[keyboard]
typing_delay_ms = 10  # Delay between keypresses
```

## Troubleshooting

### whisper.cpp not found
```bash
which whisper
# Should output: /home/mason/.local/bin/whisper
```

### GUI not showing
Check Wayland layer-shell support:
```bash
# Hyprland supports layer-shell natively
hyprctl layers
```

### No audio capture
Check PulseAudio/PipeWire:
```bash
pactl list sources short
# Find your microphone device
```

### Text not typing
Ensure wtype is installed:
```bash
which wtype
# Should output: /usr/bin/wtype
```

## License

MIT OR Apache-2.0

## Contributing

This is a personal dotfiles project. Feel free to fork and adapt to your needs.
