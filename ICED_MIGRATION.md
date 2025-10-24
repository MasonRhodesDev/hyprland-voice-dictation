# GUI Implementation Status

## Current Status

The voice dictation GUI uses **manual tiny-skia + Wayland layer-shell** rendering (`src/main.rs`).

An **Iced GUI framework** experimental version exists (`src/main_iced.rs`) but is not the default due to limitations.

## What Changed

### Added
- **Iced 0.13** - Modern Rust GUI framework with built-in animation support
- `src/main_iced.rs` - New Iced-based main application
- `src/spectrum_widget.rs` - Spectrum bars as Iced canvas widget
- `src/spinner_widget.rs` - Spinner animation as Iced canvas widget
- `src/ipc_subscription.rs` - IPC integration via Iced subscriptions
- `scripts/run-gui-overlay.sh` - Launch script with compositor-specific overlay configuration

### Preserved
- All animations (spectrum bars, spinner, collapse)
- Audio processing (IPC, FFT, VAD)
- Control messaging system
- Visual appearance and behavior

### Removed/Deprecated
- `src/main.rs` renamed to `dictation-gui-legacy` binary
- Manual Wayland buffer management
- Custom text rendering with fontdue
- Manual animation interpolation

## Current Implementation (Default)

**Binary:** `dictation-gui` (from `src/main.rs`)

### Features
- ✅ Wayland **Overlay layer** - draws above fullscreen apps
- ✅ Never takes focus - uses `KeyboardInteractivity::None`
- ✅ Transparent borders
- ✅ Dynamic window sizing (listening vs spinner states)
- ✅ Centered text at bottom
- ✅ All animations working

### Building & Running
```bash
cargo build --release
make install
~/scripts/dictation-control toggle
```

## Experimental Iced Version

**Binary:** `dictation-gui-iced` (from `src/main_iced.rs`)

### Limitations
- ❌ Cannot use Wayland Overlay layer (Iced 0.13 lacks layer-shell support)
- ❌ Acts as a window, not an overlay
- ❌ Gets focus when clicked
- ❌ Does NOT draw over fullscreen apps

### When to use
- Testing Iced's animation capabilities
- Cross-platform development (non-Wayland)
- Learning Iced framework patterns

### Running
```bash
./target/release/dictation-gui-iced
```

## Benefits

1. **Smooth Animations** - Iced handles interpolation and rendering efficiently
2. **Better Text Handling** - Native text widget with proper layout
3. **Less Code** - ~40% reduction in GUI code
4. **Maintainability** - Standard GUI framework patterns
5. **Cross-platform** - Same rendering on all platforms

## Overlay Configuration

### Current Status

Iced 0.13 doesn't have native Wayland layer-shell support, so we use `window::Level::AlwaysOnTop` which works for most cases but **doesn't draw over fullscreen applications**.

### Workarounds

#### For Hyprland Users
The `run-gui-overlay.sh` script automatically configures Hyprland window rules to make the overlay float above fullscreen apps:

```bash
hyprctl keyword windowrulev2 "pin,title:^(Voice Dictation)$"
```

#### For Sway Users
Add to your Sway config:

```
for_window [title="Voice Dictation"] floating enable, sticky enable
```

#### For Other Compositors
You may need compositor-specific configuration to make windows stay on top of fullscreen applications.

### Future Improvements

Iced is working on native layer-shell support. Once available, we can:
1. Use `Layer::Overlay` to draw over fullscreen apps natively
2. Remove compositor-specific workarounds
3. Have consistent behavior across all Wayland compositors

## Architecture

### Message Flow
```
Audio Socket → IPC Subscription → SpectrumUpdate Message → Update State → Redraw
Control Socket → IPC Subscription → StateChange Message → Update State → Redraw
Timer → Tick Message → Update Animation → Redraw
```

### Rendering Pipeline
```
App State → view() → Iced Elements → Canvas Widgets → GPU/CPU Renderer → Screen
```

## Customization

### Text Configuration
Edit `main_iced.rs`:
```rust
const TEXT_MAX_LINES: usize = 2;  // Max lines to display
const TEXT_LINE_HEIGHT: f32 = 30.0;  // Height per line
```

### Animation Speed
```rust
// In subscription()
iced::time::every(Duration::from_millis(16))  // 60 FPS
```

### Colors
Iced uses theme system. To load from matugen colors, implement custom theme in `main_iced.rs`.

## Known Issues

1. **No fullscreen overlay** - Requires compositor-specific configuration (see above)
2. **Text wrapping** - Currently line-based, not word-wrap optimized
3. **Theme loading** - Matugen colors not yet integrated (uses default dark theme)

## Migration Notes

If you need to revert to the legacy renderer:

```bash
cargo build --release --bin dictation-gui-legacy
./target/release/dictation-gui-legacy
```

The legacy renderer (`src/main.rs`) is preserved for reference.
