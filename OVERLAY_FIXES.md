# Wayland Overlay Layer Fixes

## Issues Fixed

### 1. ✅ Wrong Wayland Layer
**Problem:** GUI was using `Layer::Top` which doesn't draw over fullscreen apps  
**Fix:** Changed to `Layer::Overlay` in `src/wayland.rs:67`
```rust
Layer::Overlay,  // OVERLAY layer draws above fullscreen apps
```

### 2. ✅ Taking Focus
**Problem:** GUI window was getting keyboard focus  
**Fix:** Already set to `KeyboardInteractivity::None` in `src/wayland.rs:73`
```rust
layer_surface.set_keyboard_interactivity(KeyboardInteractivity::None);
```

### 3. ✅ Visible Borders
**Problem:** Large padding/borders around content  
**Fix:** Reduced padding in `src/renderer.rs:260` from 40px to 20px margins
```rust
let available_width = self.width as f32 - 20.0;  // Reduced padding
let start_x = 10.0 + ...;
```

### 4. ✅ Not Centered at Bottom
**Problem:** Text not centered, wrong positioning  
**Fix:** Already anchored to bottom in `src/wayland.rs:72` with proper centering in renderer

### 5. ✅ Wrong Window Sizing
**Problem:** Window didn't resize between listening/spinner/closing states  
**Fix:** Added dynamic resizing in `src/main.rs:238-271`
```rust
let new_height = match current_state {
    GuiState::Listening => LISTENING_HEIGHT,  // 120px
    GuiState::Processing => SPINNER_HEIGHT,   // 100px
    GuiState::Closing => SPINNER_HEIGHT,      // 100px
};
```

## Configuration

### Window Sizes
```rust
const WIDTH: u32 = 400;
const LISTENING_HEIGHT: u32 = 120;  // Spectrum + text (1-2 lines)
const SPINNER_HEIGHT: u32 = 100;    // Just spinner box
```

### Layer Settings
```rust
Layer::Overlay              // Highest layer, above fullscreen
Anchor::BOTTOM              // Bottom of screen
KeyboardInteractivity::None // Never take focus
exclusive_zone: -1          // Don't reserve space
margin: (0, 0, 50, 0)       // 50px from bottom
```

## Testing

The GUI should now:
- ✅ Draw over fullscreen applications
- ✅ Never steal focus
- ✅ Be centered at bottom with minimal borders
- ✅ Resize smoothly between states
- ✅ Show 1-2 lines of text maximum
- ✅ Display all animations correctly

## Verification

Run the GUI:
```bash
make install
~/scripts/dictation-control toggle
```

Then test:
1. Open a fullscreen app (Firefox F11, mpv --fullscreen)
2. Start dictation - overlay should appear on top
3. Click on it - should NOT take focus
4. Speak - should see spectrum bars
5. Release - should see spinner, then collapse
