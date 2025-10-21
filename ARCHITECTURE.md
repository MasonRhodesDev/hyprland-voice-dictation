# Architecture Documentation

## System Overview

Voice dictation system with three decoupled components communicating via IPC and process signals.

## Component Details

### 1. dictation-engine

**Responsibilities:**
- Audio capture from default input device
- Voice Activity Detection (VAD)
- Speech-to-text transcription via whisper.cpp
- Keyboard text injection via wtype
- Audio sample broadcasting for GUI

**State Machine:**

```
┌─────────┐
│  Idle   │
└────┬────┘
     │ Speech detected
     ▼
┌─────────┐
│Recording│ ◄──┐
└────┬────┘    │
     │         │ Speech continues
     │ Silence │
     │ detected│
     ▼         │
┌─────────┐   │
│ Process │───┘
│ Whisper │
└────┬────┘
     │ Got transcription
     ▼
┌─────────┐
│  Type   │
│  Text   │
└────┬────┘
     │
     ▼
┌─────────┐
│  Idle   │
└─────────┘
```

**Audio Pipeline:**

```
Input Device (Mic)
       │
       ▼
    cpal::Stream
       │
       ▼
  Ring Buffer (5s)
       │
       ├─────────────────┐
       │                 │
       ▼                 ▼
   VAD Monitor    Unix Socket Broadcast
   (Energy)         (to GUI, 60Hz)
       │
       ▼
  Silence Detected
       │
       ▼
  Extract Audio Window
       │
       ▼
   Write WAV File
       │
       ▼
  whisper.cpp subprocess
       │
       ▼
  Parse JSON output
       │
       ▼
  wtype text injection
```

**VAD Algorithm:**

```rust
// Simple energy-based VAD
fn is_speech(samples: &[f32]) -> bool {
    let rms = sqrt(mean(samples.map(|s| s * s)));
    let db = 20.0 * log10(rms);
    
    // Threshold: -40dB
    db > -40.0
}

// State tracking
struct VadState {
    speech_frames: usize,
    silence_frames: usize,
    is_speaking: bool,
}

// Hysteresis: require 3 frames to trigger
const SPEECH_TRIGGER_FRAMES: usize = 3;
const SILENCE_TRIGGER_FRAMES: usize = 24; // ~800ms at 30fps
```

**IPC Server:**

```rust
// Unix domain socket: /tmp/voice-dictation.sock
// Message format: Binary audio samples (f32 little-endian)
// Protocol: 512 samples per message @ 60Hz

struct AudioMessage {
    samples: [f32; 512],  // 32ms at 16kHz
    timestamp_ms: u64,
}
```

### 2. dictation-gui

**Responsibilities:**
- Wayland layer-shell surface creation
- Audio sample consumption from Unix socket
- FFT spectrum computation (8 bands)
- Real-time rendering at 60fps

**Wayland Layer-Shell Configuration:**

```rust
// Layer: Overlay (above all windows)
// Keyboard interactivity: None (on_demand = false)
// Anchor: Bottom | HCenter
// Size: 200x50
// Margin: bottom = 50px
// Namespace: "voice-dictation"

LayerSurface {
    layer: Layer::Overlay,
    keyboard_interactivity: KeyboardInteractivity::None,
    anchor: Anchor::BOTTOM,
    exclusive_zone: 0,
    size: (200, 50),
    margin: EdgeInsets { bottom: 50, ..default }
}
```

**FFT Spectrum Bands:**

```
Sample rate: 16kHz
FFT size: 512 samples (32ms window)
Frequency resolution: 16000 / 512 = 31.25 Hz/bin

Band mapping (8 bars):
1. 100-250 Hz   (bins 3-8)    - Low bass
2. 250-500 Hz   (bins 8-16)   - Upper bass
3. 500-1000 Hz  (bins 16-32)  - Low mids
4. 1000-2000 Hz (bins 32-64)  - Mids
5. 2000-3000 Hz (bins 64-96)  - Upper mids
6. 3000-4000 Hz (bins 96-128) - Presence
7. 4000-5000 Hz (bins 128-160)- High presence
8. 5000-7000 Hz (bins 160-224)- Highs/air
```

**Rendering Pipeline:**

```
Unix Socket
     │
     ▼
Audio Samples (512 f32)
     │
     ▼
Hanning Window
     │
     ▼
FFT (rustfft)
     │
     ▼
Magnitude Spectrum
     │
     ▼
Band Power (8 bands)
     │
     ▼
Smoothing (60% prev + 40% new)
     │
     ▼
Normalize (0.0-1.0)
     │
     ▼
Map to Bar Heights (5-30px)
     │
     ▼
Render (wgpu/skia)
     │
     ▼
Present Frame
```

**Color Scheme:**

```rust
// Catppuccin Mocha
const BACKGROUND: Color = rgba(30, 30, 46, 0.95);  // Base
const BAR_COLOR: Color = rgba(137, 180, 250, 0.8); // Blue
const BORDER_RADIUS: f32 = 30.0; // Pill shape
```

### 3. dictation-control

**State File:** `/tmp/voice-dictation-active`

**Toggle Logic:**

```bash
if [ -f "$STATE_FILE" ]; then
    # Stop: kill both processes
    pkill -f dictation-engine
    pkill -f dictation-gui
    rm -f "$STATE_FILE"
else
    # Start: launch both processes
    dictation-engine &
    sleep 0.1  # Wait for socket creation
    dictation-gui &
    touch "$STATE_FILE"
fi
```

## Inter-Process Communication

### Unix Domain Socket

**Path:** `/tmp/voice-dictation.sock`

**Protocol:**
- Type: `SOCK_STREAM` (TCP-like)
- Messages: Fixed 2048 bytes (512 samples × 4 bytes/f32)
- Rate: 60 messages/second
- Encoding: Little-endian f32

**Flow:**

```
dictation-engine (server)          dictation-gui (client)
       │                                  │
       │ bind(/tmp/voice-dictation.sock)│
       │◄─────────────────────────────────┤
       │                                  │
       │                    connect()    │
       │◄─────────────────────────────────┤
       │                                  │
       │ send(AudioMessage)              │
       ├─────────────────────────────────►│
       │                                  │ FFT + Render
       │ send(AudioMessage)              │
       ├─────────────────────────────────►│
       │                                  │ FFT + Render
       │          ...                    │
```

**Error Handling:**
- Server: Non-blocking sends, drop messages if client disconnected
- Client: Auto-reconnect on disconnect (exponential backoff)
- Both: Graceful shutdown on socket errors

## Signal Handling

```rust
// dictation-engine
tokio::signal::ctrl_c() => cleanup()
SIGTERM => cleanup()

fn cleanup() {
    // 1. Stop audio capture
    // 2. Close Unix socket
    // 3. Kill any running whisper.cpp subprocess
    // 4. Exit gracefully
}

// dictation-gui
tokio::signal::ctrl_c() => exit()
SIGTERM => exit()
```

## Configuration

**Location:** `~/.config/voice-dictation/config.toml`

```toml
[whisper]
model_path = "~/.local/share/whisper/models/ggml-base.en.bin"
binary_path = "~/.local/bin/whisper"
language = "en"
threads = 4

[audio]
sample_rate = 16000
channels = 1
buffer_duration_ms = 5000  # 5s ring buffer

[vad]
energy_threshold_db = -40.0
speech_trigger_frames = 3
silence_trigger_frames = 24  # ~800ms

[keyboard]
typing_delay_ms = 10
word_delay_ms = 50  # Pause between words

[gui]
width = 200
height = 50
position = "bottom-center"
offset_x = -100
offset_y = 50
fps = 60

[ipc]
socket_path = "/tmp/voice-dictation.sock"
send_rate_hz = 60
```

## Performance Targets

- **Engine startup:** <100ms
- **GUI startup:** <50ms
- **Audio latency:** <50ms (capture to processing)
- **Transcription latency:** <2s (for 5s audio clip)
- **Typing latency:** <100ms (after transcription)
- **GUI render time:** <16ms (60fps)
- **Memory usage:**
  - Engine: <50MB
  - GUI: <30MB
  - Combined: <80MB

## Error Recovery

### whisper.cpp Subprocess Failure
- Log error to stderr
- Skip transcription for this segment
- Continue capturing audio
- User sees no typed output (expected for failed segment)

### Audio Device Disconnected
- Log error to stderr
- Attempt to reconnect every 1s
- If successful, resume capture
- If failed after 10s, exit with error code

### Unix Socket Disconnection
- Engine: Continue operation (GUI optional)
- GUI: Attempt reconnect with exponential backoff
- Max reconnect attempts: 10

### Wayland Compositor Crash
- GUI exits gracefully
- Engine continues running
- User can toggle again to restart GUI

## Security Considerations

1. **Unix socket permissions:** 0600 (owner read/write only)
2. **No network exposure:** All IPC is local
3. **Temporary files:** `/tmp/` with random suffixes, cleaned on exit
4. **No audio recording to disk:** Only temporary WAV for whisper.cpp
5. **Process isolation:** Engine and GUI run as separate processes

## Future Enhancements

- [ ] Multiple language support
- [ ] Custom vocabulary/phrases
- [ ] Voice commands (e.g., "new line", "period")
- [ ] Punctuation auto-insertion
- [ ] Speaker diarization
- [ ] Real-time streaming transcription (no VAD pauses)
- [ ] GPU-accelerated whisper.cpp
- [ ] Hot-reload configuration
- [ ] Metrics/telemetry dashboard
