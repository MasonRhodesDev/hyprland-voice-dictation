# Hyprland Voice Dictation

Offline voice dictation for Hyprland using NVIDIA Parakeet TDT speech recognition. Press a key to start recording, press again to transcribe and type the result into any focused window.

## Features

- **Offline, private** — all processing runs locally, no cloud API
- **NVIDIA Parakeet TDT 0.6b** — high-accuracy English speech recognition via ONNX Runtime
- **Silero VAD** — voice activity detection to trim silence automatically
- **Harper grammar checker** — optional light grammar correction on transcribed text
- **Slint overlay** — transparent HUD showing recording state and live transcription
- **System tray** — status icon with device selection and quick controls
- **D-Bus control** — clean interface for keybind integration
- **systemd daemon** — persistent background service with watchdog support
- **playerctl integration** — auto-pause/resume media during recording

## Requirements

- Wayland compositor (Hyprland, Sway, etc.)
- `wtype` — keyboard input injection
- PipeWire or ALSA audio
- ~1.6 GB disk space for the Parakeet model

Optional: `playerctl` for media pause/resume.

## Installation

### Arch Linux (from source)

```bash
git clone https://github.com/MasonRhodesDev/hyprland-voice-dictation
cd hyprland-voice-dictation

# Build
cargo build --release

# Install binary
install -Dm755 target/release/voice-dictation ~/.local/bin/voice-dictation

# Install systemd service
install -Dm644 packaging/systemd/voice-dictation.service \
    ~/.config/systemd/user/voice-dictation.service

systemctl --user daemon-reload
```

### From release tarball

Download the latest release tarball from the [releases page](https://github.com/MasonRhodesDev/hyprland-voice-dictation/releases), extract, and copy the binary:

```bash
tar -xzf hyprland-voice-dictation-*-x86_64-linux.tar.gz
cd hyprland-voice-dictation-*/
install -Dm755 voice-dictation ~/.local/bin/voice-dictation
install -Dm644 voice-dictation.service ~/.config/systemd/user/voice-dictation.service
systemctl --user daemon-reload
```

## Download the Model

The Parakeet model (~1.6 GB) is not included and must be downloaded separately:

```bash
voice-dictation download-model
```

This downloads the model from HuggingFace to `~/.config/voice-dictation/models/parakeet/`. Files already present are skipped.

Alternatively, use the standalone shell script (requires `curl`):

```bash
bash scripts/download-parakeet-model.sh
```

## Setup

### Start the daemon

```bash
# Enable on login
systemctl --user enable --now voice-dictation

# Check status
systemctl --user status voice-dictation
journalctl --user -u voice-dictation -f
```

### Hyprland keybind

Add to `~/.config/hypr/hyprland.conf`:

```
bind = SUPER, V, exec, voice-dictation toggle
```

Press `Super+V` to start recording. Press again to confirm and type the transcription.

### Other compositors

Any Wayland compositor supporting `wtype` works. Map `voice-dictation toggle` to a key using your compositor's keybind system.

## CLI Usage

```
voice-dictation <COMMAND>

Commands:
  daemon              Start the dictation engine daemon
  start               Start a recording session
  stop                Cancel recording
  confirm             Finalize and type the transcription
  toggle              Start if idle, confirm if recording
  status              Show daemon and subsystem status
  config              Open the configuration TUI
  download-model      Download Parakeet model from HuggingFace
  list-audio-devices  List available audio input devices
  diagnose            Show diagnostics (model paths, audio, config)
  debug list          List saved debug recordings
  debug play FILE     Play a debug recording
```

## Configuration

Run `voice-dictation config` to open the interactive configuration TUI.

Config file: `~/.config/voice-dictation/config.toml`

```toml
# Audio device (leave empty for system default)
audio_device = ""

# Audio backend: "pipewire" or "alsa"
audio_backend = "pipewire"

# Grammar checking
grammar_check = true
```

Run `voice-dictation diagnose` to inspect the current configuration and model status.

## Troubleshooting

**Daemon not starting:**
```bash
journalctl --user -u voice-dictation -n 50
voice-dictation diagnose
```

**Model missing:**
```bash
voice-dictation download-model
```

**No audio input / wrong device:**
```bash
voice-dictation list-audio-devices
# Then set audio_device in config
voice-dictation config
```

**wtype not found:**
```bash
# Arch
sudo pacman -S wtype
# Fedora
sudo dnf install wtype
```

## Project Structure

```
src/main.rs                   CLI frontend and D-Bus client
dictation-engine/             Core library
  src/lib.rs                  Daemon entry point and state machine
  src/engine/                 Parakeet ONNX inference
  src/audio/                  PipeWire/ALSA capture
  src/vad.rs                  Silero VAD
  src/post_processing/        Grammar and text cleanup
dictation-types/              Shared types
slint-gui/                    Overlay HUD (Slint UI)
packaging/
  arch/PKGBUILD               Arch Linux package
  systemd/voice-dictation.service
scripts/
  check-deps.sh               Dependency checker
  download-parakeet-model.sh  Standalone model downloader
  list-audio-devices.sh       List audio devices
config-schema.json            Config schema for the TUI
```

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE) at your option.
