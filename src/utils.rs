//! Utility functions for model and device listing

use dictation_engine::audio_backend::{BackendType, DeviceInfo};

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

/// List available audio input devices with human-readable descriptions
pub fn list_audio_devices() -> Vec<DeviceInfo> {
    dictation_engine::audio_backend::list_devices(BackendType::Auto).unwrap_or_default()
}

/// Get a summary of available engines for display
pub fn get_engine_summary() -> String {
    "parakeet".to_string()
}
