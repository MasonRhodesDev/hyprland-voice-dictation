# Voice Dictation System

Offline voice dictation for Linux with Wayland overlay using Vosk speech recognition.

## Features

- **Two-model approach**: Fast model for live preview, accurate model for final text
- **Preview-then-type**: See transcription in overlay before it's typed
- **Wayland overlay**: Shows audio spectrum and live transcription text
- **Compositor-agnostic**: Works with any Wayland compositor (Hyprland, Sway, KDE, GNOME, etc.)
- **Toggle interface**: Keybind to start, keybind again to confirm and type

## Prerequisites

- **Linux** (Fedora/Arch tested)
- **Wayland compositor** - X11 not supported
- **Rust 1.70+**
- **System packages**: wtype, pipewire, alsa-lib-devel, fontconfig-devel, freetype-devel
- **2GB disk space** for Vosk models

Check dependencies:
```bash
make check
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
# Download models (2GB, one-time)
make deps

# Build and install
make install

# Test with automated audio
make test
```

## Usage

### Keybind Setup

Add to your compositor config:

**Hyprland:**
```
bind=$Meh, V, exec, ~/scripts/dictation-control toggle
```

**Sway:**
```
bindsym Mod4+Shift+Alt+v exec ~/scripts/dictation-control toggle
```

**KDE Plasma (Wayland):**
- System Settings → Shortcuts → Custom Shortcuts
- Add new command: `~/scripts/dictation-control toggle`

**GNOME (Wayland):**
- Settings → Keyboard → Custom Shortcuts
- Add new command: `~/scripts/dictation-control toggle`

Then:
1. **Press your keybind** to start recording
2. **Speak clearly** into microphone
3. **Press keybind again** to confirm and type

### Command Line

```bash
# Start recording
~/scripts/dictation-control start

# Check status
~/scripts/dictation-control status

# Confirm and type
~/scripts/dictation-control confirm

# Or use toggle
~/scripts/dictation-control toggle  # starts
~/scripts/dictation-control toggle  # confirms
```

## Development

```bash
make dev           # Quick rebuild and install
make test-manual   # Test with real microphone
make fmt           # Format code
make lint          # Run clippy
make logs          # Monitor live logs
make help          # Show all commands
```

## Alternative Installation

### From RPM (Fedora)

```bash
make rpm
sudo dnf install ~/rpmbuild/RPMS/x86_64/voice-dictation-*.rpm
mkdir -p ~/scripts
cp /usr/share/voice-dictation/scripts/* ~/scripts/
```

### Uninstall

```bash
make uninstall
```

## Models

Located in `models/`:
- **vosk-model-small-en-us-0.15** (40MB): Fast, ~90% accuracy, live preview
- **vosk-model-en-us-0.22** (1.8GB): Accurate, ~94% accuracy, final correction

## Troubleshooting

**No overlay?**
```bash
pgrep dictation-gui || echo "GUI not running"
tail /tmp/dictation-gui.log
```

**No transcription?**
```bash
tail -f /tmp/dictation-gui.log | grep "Transcription:"
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
tail -f /tmp/dictation-engine.log /tmp/dictation-gui.log
```

## Project Structure

```
voice-dictation-rust/
├── README.md
├── TODO.md
├── install.sh              # One-command install
├── test_jfk_automated.sh   # Automated test with virtual audio
├── test_manual.sh          # Manual test with mic
├── Cargo.toml              # Workspace configuration
├── dictation-engine/       # Audio → Vosk → Keyboard
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs         # Main loop, model management
│       ├── control_ipc.rs  # Control socket server
│       ├── ipc.rs          # Audio socket server
│       └── keyboard.rs     # wtype injection
├── dictation-gui/          # Wayland overlay
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs         # Wayland window + event loop
│       ├── control_ipc.rs  # Control socket client
│       ├── ipc.rs          # Audio socket client
│       ├── wayland.rs      # Layer-shell setup
│       ├── renderer.rs     # Spectrum + text rendering
│       └── fft.rs          # FFT analysis
├── models/                 # Vosk models
│   ├── vosk-model-small-en-us-0.15/
│   └── vosk-model-en-us-0.22/
└── scripts/
    └── dictation-control   # Toggle script
```

## Dependencies

- Rust/Cargo
- Vosk models (included in `models/`)
- `wtype` - Wayland keyboard injection
- `pactl` / PipeWire - Audio
- Wayland compositor (Hyprland, Sway, KDE, GNOME, etc.)

## Known Issues

- Vosk has ~5-10% error rate (inherent to open-source models)
- First few words may be cut off if speech starts immediately
- Duplications can still occur due to Vosk's internal chunking

## License

MIT OR Apache-2.0
