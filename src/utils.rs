//! Utility functions for model and device listing

use cpal::traits::{DeviceTrait, HostTrait};

/// List available models (Parakeet only)
pub fn list_models() -> Vec<String> {
    vec!["parakeet:default".to_string()]
}

/// List preview (fast) models — same as final since Parakeet is the only engine
pub fn list_preview_models(_language: &str) -> Vec<String> {
    list_models()
}

/// List final (accurate) models — same as preview since Parakeet is the only engine
pub fn list_final_models(_language: &str) -> Vec<String> {
    list_models()
}

/// List available audio input devices
pub fn list_audio_devices() -> Vec<String> {
    let mut devices = Vec::new();

    // Standard options
    devices.push("default".to_string());

    // Enumerate actual devices via cpal
    let host = cpal::default_host();
    if let Ok(input_devices) = host.input_devices() {
        for device in input_devices {
            if let Ok(name) = device.name() {
                // Skip monitors and HDMI
                let name_lower = name.to_lowercase();
                if name_lower.contains(".monitor") || name_lower.contains("hdmi") {
                    continue;
                }
                if !devices.contains(&name) {
                    devices.push(name);
                }
            }
        }
    }

    devices
}

/// Get a summary of available engines for display
pub fn get_engine_summary() -> String {
    "parakeet".to_string()
}
