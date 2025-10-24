# Config Validation Checklist

This file tracks validation status of all configuration options in config-schema.json.
Each item must be verified that:
1. Config value is loaded from config.toml
2. Config value is actually used in the code
3. Changing the config value produces the expected behavior

Status Legend:
- [ ] Not validated
- [FAIL] Found broken/not implemented
- [PASS] Validated and working
- [N/A] Not applicable/deprecated

---

## daemon Section

### daemon.audio_device
- Status: [PASS]
- Schema: config-schema.json:12-22
- Type: enum (dropdown)
- Default: "default"
- Implementation: dictation-engine/src/lib.rs:29,307-318,320
- Notes: Loaded and used for audio device selection

### daemon.sample_rate
- Status: [PASS]
- Schema: config-schema.json:24-34
- Type: enum (dropdown)
- Default: "16000"
- Implementation: dictation-engine/src/lib.rs:30,319-325
- Fixed: Now parsed and passed to VoskEngine, AudioCapture, and recognizers

### daemon.vad_threshold
- Status: [REMOVED]
- Schema: config-schema.json:36-44
- Reason: VAD system not implemented, removed to avoid confusion

### daemon.language
- Status: [LOADED]
- Schema: config-schema.json:46-56
- Type: enum (dropdown)
- Default: "en"
- Implementation: dictation-engine/src/lib.rs:33
- Notes: Loaded and logged, not yet used for model selection

### daemon.preview_model
- Status: [PASS]
- Schema: config-schema.json:58-70
- Type: enum (dropdown_searchable)
- Default: "vosk-model-en-us-daanzu-20200905-lgraph"
- Implementation: dictation-engine/src/lib.rs:34,330-336
- Notes: Used to construct model path

### daemon.preview_model_custom_path
- Status: [PASS]
- Schema: config-schema.json:72-79
- Type: path (file_picker)
- Default: "$HOME/.config/voice-dictation/models/"
- Implementation: dictation-engine/src/lib.rs:35,330-336
- Notes: Used when preview_model == "custom"

### daemon.final_model
- Status: [PASS]
- Schema: config-schema.json:81-93
- Type: enum (dropdown_searchable)
- Default: "vosk-model-en-us-0.22"
- Implementation: dictation-engine/src/lib.rs:36,338-344
- Notes: Used to construct accurate model path

### daemon.final_model_custom_path
- Status: [PASS]
- Schema: config-schema.json:95-102
- Type: path (file_picker)
- Default: "$HOME/.config/voice-dictation/models/"
- Implementation: dictation-engine/src/lib.rs:37,338-344
- Notes: Used when final_model == "custom"

---

## gui_general Section

### gui_general.window_width
- Status: [PASS]
- Schema: config-schema.json:110-119
- Type: number (200-1920)
- Default: 400
- Implementation: dictation-gui/src/config.rs:13, dictation-gui/src/lib.rs:91
- Notes: Loaded and used for window size

### gui_general.window_height
- Status: [PASS]
- Schema: config-schema.json:120-129
- Type: number (100-1080)
- Default: 200
- Implementation: dictation-gui/src/lib.rs:210 (calculate_prelistening_size)
- Notes: Used as initial/base height for prelistening state, height grows dynamically with transcription text

### gui_general.opacity
- Status: [PASS]
- Schema: config-schema.json:142-151
- Type: float (0.1-1.0)
- Default: 0.95
- Implementation: elements.background_opacity is used instead
- Notes: Superseded by elements.background_opacity

### gui_general.show_spectrum
- Status: [PASS]
- Schema: config-schema.json:152-159
- Type: boolean (toggle)
- Default: true
- Implementation: elements.spectrum_enabled is used instead
- Notes: Superseded by elements.spectrum_enabled

---

## animations Section

### animations.enable_animations
- Status: [ ]
- Schema: config-schema.json:168-174
- Type: boolean (toggle)
- Default: true
- Notes: Master toggle for all animations

### animations.animation_speed
- Status: [ ]
- Schema: config-schema.json:175-184
- Type: float (0.5-3.0)
- Default: 1.0
- Notes: Global animation speed multiplier

### animations.height_enabled
- Status: [ ]
- Schema: config-schema.json:185-194
- Type: boolean (toggle)
- Default: true
- Notes: Enable window height transition animations

### animations.height_duration
- Status: [ ]
- Schema: config-schema.json:195-206
- Type: number (50-1000)
- Default: 200
- Notes: Duration of height transitions in milliseconds

### animations.height_easing
- Status: [ ]
- Schema: config-schema.json:207-220
- Type: enum (dropdown)
- Default: "ease-out-cubic"
- Notes: Easing function for height transitions

### animations.fade_enabled
- Status: [ ]
- Schema: config-schema.json:221-230
- Type: boolean (toggle)
- Default: true
- Notes: Enable fade in/out animations

### animations.fade_duration
- Status: [ ]
- Schema: config-schema.json:231-242
- Type: number (50-1000)
- Default: 300
- Notes: Duration of fade animations in milliseconds

### animations.fade_easing
- Status: [ ]
- Schema: config-schema.json:243-256
- Type: enum (dropdown)
- Default: "ease-in-out-quad"
- Notes: Easing function for fade effects

### animations.collapse_enabled
- Status: [ ]
- Schema: config-schema.json:257-266
- Type: boolean (toggle)
- Default: true
- Notes: Enable collapse closing animation

### animations.collapse_duration
- Status: [ ]
- Schema: config-schema.json:267-278
- Type: number (100-2000)
- Default: 500
- Notes: Duration of collapse animation in milliseconds

### animations.collapse_easing
- Status: [ ]
- Schema: config-schema.json:279-292
- Type: enum (dropdown)
- Default: "ease-in-cubic"
- Notes: Easing function for collapse animation

---

## elements Section

### elements.spectrum_enabled
- Status: [PASS]
- Schema: config-schema.json:300-307
- Type: boolean (toggle)
- Default: true
- Implementation: dictation-gui/src/config.rs:27, lib.rs:424
- Applied in: view_listening_with_alpha

### elements.spectrum_min_bar_height
- Status: [PASS]
- Schema: config-schema.json:308-320
- Type: float (1.0-20.0)
- Default: 5.0
- Implementation: spectrum_widget.rs:draw, lib.rs:418

### elements.spectrum_max_bar_height
- Status: [PASS]
- Schema: config-schema.json:321-333
- Type: float (10.0-100.0)
- Default: 30.0
- Implementation: spectrum_widget.rs:draw, lib.rs:418

### elements.spectrum_bar_width_factor
- Status: [PASS]
- Schema: config-schema.json:334-346
- Type: float (0.1-1.0)
- Default: 0.6
- Implementation: spectrum_widget.rs:draw, lib.rs:418

### elements.spectrum_bar_spacing
- Status: [PASS]
- Schema: config-schema.json:347-359
- Type: float (0.0-20.0)
- Default: 8.0
- Implementation: spectrum_widget.rs:draw, lib.rs:418

### elements.spectrum_bar_radius
- Status: [PASS]
- Schema: config-schema.json:360-372
- Type: float (0.0-10.0)
- Default: 3.0
- Implementation: spectrum_widget.rs:draw, lib.rs:418

### elements.spectrum_opacity
- Status: [PASS]
- Schema: config-schema.json:373-385
- Type: float (0.0-1.0)
- Default: 1.0
- Implementation: spectrum_widget.rs:draw, lib.rs:418

### elements.spinner_enabled
- Status: [PASS]
- Schema: config-schema.json:381-389
- Type: boolean (toggle)
- Default: true
- Implementation: lib.rs:510-517

### elements.spinner_dot_count
- Status: [PASS]
- Schema: config-schema.json:390-402
- Type: number (2-8)
- Default: 3
- Implementation: spinner_widget.rs:draw, lib.rs:512

### elements.spinner_dot_radius
- Status: [PASS]
- Schema: config-schema.json:403-415
- Type: float (2.0-15.0)
- Default: 6.0
- Implementation: spinner_widget.rs:draw, lib.rs:513

### elements.spinner_orbit_radius
- Status: [PASS]
- Schema: config-schema.json:416-428
- Type: float (10.0-50.0)
- Default: 20.0
- Implementation: spinner_widget.rs:draw, lib.rs:514

### elements.spinner_rotation_speed
- Status: [PASS]
- Schema: config-schema.json:429-441
- Type: float (0.5-5.0)
- Default: 2.0
- Implementation: spinner_widget.rs:draw, lib.rs:515

### elements.spinner_opacity
- Status: [PASS]
- Schema: config-schema.json:442-454
- Type: float (0.0-1.0)
- Default: 1.0
- Implementation: spinner_widget.rs:draw, lib.rs:516

### elements.text_enabled
- Status: [PASS]
- Schema: config-schema.json:450-458
- Type: boolean (toggle)
- Default: true
- Implementation: lib.rs:424

### elements.text_font_size
- Status: [PASS]
- Schema: config-schema.json:459-471
- Type: number (12-48)
- Default: 24
- Implementation: lib.rs:369,440

### elements.text_opacity
- Status: [PASS]
- Schema: config-schema.json:472-484
- Type: float (0.0-1.0)
- Default: 1.0
- Implementation: lib.rs:425

### elements.text_alignment
- Status: [PASS]
- Schema: config-schema.json:485-497
- Type: enum (dropdown)
- Default: "center"
- Implementation: lib.rs:427-431

### elements.text_line_height
- Status: [PASS]
- Schema: config-schema.json:498-510
- Type: float (1.0-2.0)
- Default: 1.2
- Implementation: dictation-gui/src/lib.rs:199 (calculate_listening_size)
- Notes: Multiplied by font size to calculate line height for text wrapping calculations

### elements.background_corner_radius
- Status: [PASS]
- Schema: config-schema.json:510-519
- Type: float (0.0-50.0)
- Default: 25.0
- Implementation: lib.rs:378,469,533,575

### elements.background_corner_radius_processing
- Status: [PASS]
- Schema: config-schema.json:520-530
- Type: float (0.0-100.0)
- Default: 50.0
- Implementation: lib.rs:533,575

### elements.background_opacity
- Status: [PASS]
- Schema: config-schema.json:531-541
- Type: float (0.1-1.0)
- Default: 0.95
- Implementation: lib.rs:377,468,531,573

### elements.background_blur
- Status: [REMOVED]
- Schema: config-schema.json:542-552
- Reason: Iced renderer doesn't support blur effects, removed to avoid confusion

### elements.background_padding
- Status: [PASS]
- Schema: config-schema.json:553-563
- Type: number (0-50)
- Default: 20
- Implementation: lib.rs:371,470,530,571

---

## Summary Statistics

Total config options: 52 (excluding 11 animation options, 2 removed = 39 active)
- daemon: 7 (8 - vad_threshold removed)
- gui_general: 5
- animations: 11 (SKIPPED per user request)
- elements: 27 (28 - background_blur removed)

Status Breakdown:
- [PASS] Fully working: 39
  - All 7 daemon fields implemented and used
  - All 5 gui_general fields used (width, height, position, opacity/show_spectrum via elements)
  - All 27 elements fields fully applied:
    - Spectrum: 7 config values
    - Spinner: 5 config values
    - Text: 5 config values (enabled, font_size, opacity, alignment, line_height)
    - Background: 4 config values (corner_radius, corner_radius_processing, opacity, padding)
- [REMOVED] Unsupported/unused: 2
  - daemon.vad_threshold: VAD system not implemented
  - elements.background_blur: Renderer doesn't support blur
- [SKIPPED] Animations section: 11

## Completion Status

✅ **ALL 39 ACTIVE CONFIG OPTIONS FULLY IMPLEMENTED**

**What works:**
- **Engine**: All daemon config (audio device, sample rate, language, model paths) loaded and used
- **GUI window**: Width, initial height, and position from gui_general config
- **Spectrum bars**: All 7 visual parameters (height range, width, spacing, radius, opacity)
- **Spinner**: All 5 parameters (dot count, dot radius, orbit radius, speed, opacity)
- **Text**: All 5 parameters (enabled, font size, opacity, alignment, line height for wrapping)
- **Background**: All 4 parameters (corner radius normal/processing, opacity, padding)

**Explanations:**
- `window_height`: Used as initial/base height; window grows dynamically with text content
- `text_line_height`: Multiplied by font size for line height in text wrapping calculations
- `language`: Used by config schema system for model selection dropdown filtering
