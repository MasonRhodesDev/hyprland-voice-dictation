# Voice Dictation System

Offline voice dictation for Linux with Wayland overlay using Whisper or Vosk speech recognition.

## Features

- **Whisper & Vosk engines**: Choose between Whisper (better accuracy) or Vosk (original behavior)
- **Post-processing pipeline**: Automatic acronym detection, capitalization, and grammar checking (Harper)
- **Two-model approach**: Fast Vosk model for live preview, accurate model (Whisper/Vosk) for final text
- **Configuration TUI**: Interactive terminal UI for managing settings and models
- **Preview-then-type**: See transcription in overlay before it's typed
- **Wayland overlay**: Smooth animated GUI with audio spectrum and live transcription
- **Compositor-agnostic**: Works with any Wayland compositor (Hyprland, Sway, KDE, GNOME, etc.)
- **Single binary**: Unified binary with subcommands for daemon, GUI, and control

## Prerequisites

- **Linux** (Fedora/Arch tested)
- **Wayland compositor** - X11 not supported
- **Rust 1.70+**
- **System packages**: wtype, pipewire, alsa-lib-devel, fontconfig-devel, freetype-devel
- **Disk space**:
  - Whisper models: ~150MB (small), ~500MB (medium)
  - Vosk models: ~40MB (small), ~2GB (large)

Check dependencies:
```bash
bash scripts/check-deps.sh
```

Install missing packages:
```bash
# Fedora
sudo dnf install rust cargo wtype pipewire alsa-lib-devel fontconfig-devel freetype-devel

# Arch
sudo pacman -S rust cargo wtype pipewire alsa-lib fontconfig freetype2
```

## Quick Start

```bash
# Build and install
cargo build --release
cargo install --path .

# Configure (interactive TUI)
voice-dictation config

# The TUI will:
# - Let you choose Whisper or Vosk engine
# - Select models for preview and final transcription
# - Configure post-processing (acronyms, grammar, etc.)
# - Offer to download missing models automatically

# Test with real microphone
bash test_manual.sh
```

## Usage

### Keybind Setup

Add to your compositor config:

**Hyprland:**
```
bind=$Meh, V, exec, voice-dictation toggle
```

**Sway:**
```
bindsym Mod4+Shift+Alt+v exec voice-dictation toggle
```

**KDE Plasma (Wayland):**
- System Settings → Shortcuts → Custom Shortcuts
- Add new command: `voice-dictation toggle`

**GNOME (Wayland):**
- Settings → Keyboard → Custom Shortcuts
- Add new command: `voice-dictation toggle`

Then:
1. **Press your keybind** to start recording
2. **Speak clearly** into microphone
3. **Press keybind again** to confirm and type (Whisper processes the audio with post-processing)

### Command Line

```bash
# Toggle recording (start/confirm)
voice-dictation toggle

# Or use separate commands:
voice-dictation start     # Start recording
voice-dictation status    # Check status
voice-dictation confirm   # Confirm and type
voice-dictation stop      # Stop without typing

# Configure settings
voice-dictation config

# Advanced: Run daemon and GUI manually
voice-dictation daemon    # Run engine daemon
voice-dictation gui       # Run GUI overlay
```

## Configuration

The configuration TUI (`voice-dictation config`) allows you to set:

- **Transcription Engine**: Choose Whisper (better accuracy) or Vosk (original)
- **Audio Device**: Select from available ALSA/PipeWire devices
- **Models**: Configure preview and final transcription models
- **Post-processing**:
  - Acronym detection (e.g., "a p i" → "API")
  - Capitalization (first word, "I", after sentence endings)
  - Grammar & spell checking via Harper (developer-friendly)
- **GUI Settings**: Window size, position, colors, fonts
- **Advanced**: Sample rate, VAD threshold, language support

Configuration is stored in `~/.config/voice-dictation/config.toml`.

The TUI is powered by [schema-tui](https://github.com/masonyoungblood/schema-tui) - a JSON Schema-based configuration interface.

## Development

```bash
# Quick rebuild
cargo build --release

# Format code
cargo fmt

# Run linter
cargo clippy

# Test with real microphone
bash test_manual.sh

# Monitor logs
tail -f /tmp/dictation-engine.log
```

## Models

Models are stored in `~/.config/voice-dictation/models/`.

**Whisper Models** (GGML format):
- **ggml-tiny.en.bin** (~75MB): Fastest, lower accuracy
- **ggml-base.en.bin** (~142MB): Good balance
- **ggml-small.en.bin** (~466MB): Recommended for CPU, high accuracy
- **ggml-medium.en.bin** (~1.5GB): Best accuracy, slower

**Vosk Models**:
- **vosk-model-en-us-daanzu-20200905-lgraph** (~40MB): Fast, for preview
- **vosk-model-en-us-0.22** (~1.8GB): Accurate, for final transcription

The configuration TUI will offer to download missing models automatically.

## Troubleshooting

**No overlay?**
```bash
pgrep -f "voice-dictation daemon" || echo "Daemon not running"
tail /tmp/dictation-engine.log
```

**No transcription?**
```bash
tail -f /tmp/dictation-engine.log | grep "Transcription:"
```

**Text not being typed?**
- Ensure text input field is focused
- Check: `tail /tmp/dictation-engine.log`
- Verify: `which wtype`

**Kill stuck processes:**
```bash
pkill -9 -f dictation
```

**Monitor logs:**
```bash
make logs
# or
tail -f /tmp/dictation-engine.log
```

## Project Structure

```
voice-dictation-rust/
├── README.md
├── Cargo.toml                  # Workspace configuration
├── src/
│   └── main.rs                 # Main binary with subcommands
├── dictation-engine/           # Audio → Speech Recognition → Keyboard
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs              # Engine entry point
│       ├── engine.rs           # Engine trait
│       ├── whisper_engine.rs   # Whisper implementation
│       ├── vosk_engine.rs      # Vosk implementation
│       ├── model_manager.rs    # Model loading and management
│       ├── post_processing/    # Post-processing pipeline
│       │   ├── mod.rs
│       │   ├── acronym.rs      # Acronym detection
│       │   ├── punctuation.rs  # Capitalization
│       │   └── grammar.rs      # Harper grammar checking
│       ├── control_ipc.rs      # Control socket server
│       ├── ipc.rs              # Audio socket server
│       └── keyboard.rs         # wtype injection
├── slint-gui/                  # Wayland overlay (Slint framework)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs              # GUI entry point
│       └── monitor.rs          # Monitor detection
├── dictation-types/            # Shared types for daemon-GUI communication
│   └── src/lib.rs
├── config-schema.json          # JSON Schema for configuration
├── scripts/                    # Utility scripts
│   ├── check-deps.sh
│   ├── download-whisper-models.sh
│   ├── list-audio-devices.sh
│   └── list-vosk-models.sh
└── test_manual.sh              # Manual test with mic
```

## Dependencies

**Runtime:**
- Rust/Cargo
- Whisper or Vosk models (downloaded via config TUI)
- `wtype` - Wayland keyboard injection
- PipeWire / ALSA - Audio input
- Wayland compositor (Hyprland, Sway, KDE, GNOME, etc.)

**Rust Libraries:**
- `whisper-rs` - Whisper.cpp bindings
- `vosk` - Vosk speech recognition
- `slint` - GUI framework
- `harper-core` - Grammar and spell checking
- `schema-tui` - Configuration interface
- `tokio` - Async runtime

## Known Issues

- Whisper can be slow on CPU (~3-10s for final processing depending on model/speech length)
- First few words may be cut off if speech starts immediately
- Post-processing adds ~20ms overhead (can be disabled in config)

## License

MIT OR Apache-2.0
