# Audio Setup

## Audio Backend Selection

Voice dictation supports two audio backends, configured via `audio_backend` in `config.toml`:

| Backend | Value | When to use |
|---------|-------|-------------|
| Auto (default) | `"auto"` | Prefer PipeWire, fall back to cpal |
| Native PipeWire | `"pipewire"` | Multiple mics, low latency, mic sharing with other apps |
| cpal (ALSA) | `"cpal"` | Compatibility fallback, single device |

```toml
[daemon]
audio_backend = "auto"   # or "pipewire" / "cpal"
```

### PipeWire vs cpal

**PipeWire** (`pipewire`) uses the native PipeWire API to capture audio. It allows the same microphone to be shared with other applications simultaneously (e.g., video calls), and supports multi-device capture natively.

**cpal** (`cpal`) uses ALSA under the hood. It is a reliable cross-platform fallback but may hold exclusive device access on some configurations.

## Multi-Device Audio (StreamMuxer)

When `audio_device = "all"` is set, the StreamMuxer captures from multiple input devices simultaneously and routes the loudest active device to the transcription engine.

```toml
[daemon]
audio_device = "all"
```

### How StreamMuxer Works

1. All non-monitor audio input devices are captured in parallel.
2. Each device's audio is scored on a rolling window (RMS energy).
3. The highest-scoring device is selected as the active source.
4. Once a device becomes active, it stays active for `muxer_sticky_duration_ms` to avoid rapid switching mid-word.
5. After the sticky period, a `muxer_cooldown_ms` cooldown prevents immediate re-switching.

### Muxer Configuration

```toml
[daemon]
muxer_sticky_duration_ms = 500   # How long active device stays active (ms)
muxer_cooldown_ms = 200          # Cooldown after device switch (ms)
muxer_switch_threshold = 0.15    # Min energy difference to trigger switch (0.0–1.0)
muxer_scoring_window_ms = 100    # Rolling window for energy scoring (ms)
```

**Tuning tips:**
- Increase `muxer_sticky_duration_ms` if you hear mid-word device switching.
- Decrease `muxer_switch_threshold` if a second mic isn't being picked up fast enough.
- Increase `muxer_scoring_window_ms` for smoother switching with noisy environments.

## Device Selection

Use `voice-dictation list-audio-devices` to see available devices:

```
default        # System default input
all            # All devices via StreamMuxer
alsa_input.pci-0000_00_1f.3.analog-stereo   # Specific device
```

Set a specific device in config:
```toml
[daemon]
audio_device = "alsa_input.pci-0000_00_1f.3.analog-stereo"
```

## Diagnosing Audio Issues

Run `voice-dictation diagnose` for a full diagnostic report:
- Lists detected audio input devices
- Shows configured backend and muxer settings
- Reports engine availability
- Shows debug audio recording status

Enable debug recording to capture audio for inspection:
```bash
VOICE_DICTATION_DEBUG_AUDIO=1 voice-dictation daemon
```

Then use `voice-dictation debug list` and `voice-dictation debug play <file>` to review recordings.
