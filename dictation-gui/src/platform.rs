// Platform integration for Wayland layer shell overlay
// This provides the glue between Iced and Wayland's layer-shell protocol

use smithay_client_toolkit::shell::wlr_layer::{Anchor, KeyboardInteractivity, Layer};

pub struct LayerShellSettings {
    pub layer: Layer,
    pub anchor: Anchor,
    pub exclusive_zone: i32,
    pub margin: (i32, i32, i32, i32), // top, right, bottom, left
    pub keyboard_interactivity: KeyboardInteractivity,
}

impl Default for LayerShellSettings {
    fn default() -> Self {
        Self {
            // Use Overlay layer to appear above fullscreen apps
            layer: Layer::Overlay,
            // Anchor to bottom center
            anchor: Anchor::BOTTOM,
            // Don't reserve screen space
            exclusive_zone: -1,
            // Bottom margin
            margin: (0, 0, 50, 0),
            // Don't grab keyboard
            keyboard_interactivity: KeyboardInteractivity::None,
        }
    }
}

// Note: Iced 0.13 doesn't have built-in layer shell support
// We'll need to use winit with custom Wayland configuration
// or wait for Iced to add layer shell support in future versions

// For now, we'll document the workaround approach:
// 1. Use iced::window::Settings with level: AlwaysOnTop
// 2. Post-processing: Use xdg-shell or layer-shell via environment
