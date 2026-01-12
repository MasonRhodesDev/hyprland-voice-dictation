//! Utility functions for model and device listing with runtime detection

use cpal::traits::{DeviceTrait, HostTrait};
use std::path::Path;
use std::sync::OnceLock;

/// Known Vosk models available for download
const VOSK_MODELS: &[(&str, &str, bool)] = &[
    // (model_name, language_prefix, is_lgraph/fast)
    ("vosk-model-en-us-daanzu-20200905-lgraph", "en", true),
    ("vosk-model-en-us-0.22-lgraph", "en", true),
    ("vosk-model-small-en-us-0.15", "en", true),
    ("vosk-model-en-us-daanzu-20200905", "en", false),
    ("vosk-model-en-us-0.22", "en", false),
    ("vosk-model-en-us-0.21", "en", false),
    ("vosk-model-en-us-librispeech-0.2", "en", false),
    ("vosk-model-en-us-aspire-0.2", "en", false),
    ("vosk-model-en-us-0.42-gigaspeech", "en", false),
    ("vosk-model-en-in-0.5", "en", false),
    // Spanish
    ("vosk-model-small-es-0.42", "es", true),
    ("vosk-model-es-0.42", "es", false),
    // French
    ("vosk-model-small-fr-0.22", "fr", true),
    ("vosk-model-fr-0.22", "fr", false),
    // German
    ("vosk-model-small-de-0.15", "de", true),
    ("vosk-model-de-0.21", "de", false),
    // Russian
    ("vosk-model-small-ru-0.22", "ru", true),
    ("vosk-model-ru-0.42", "ru", false),
    // Chinese
    ("vosk-model-small-cn-0.22", "cn", true),
    ("vosk-model-cn-0.22", "cn", false),
    // Japanese
    ("vosk-model-small-ja-0.22", "ja", true),
    ("vosk-model-ja-0.22", "ja", false),
    // Italian
    ("vosk-model-small-it-0.22", "it", true),
    ("vosk-model-it-0.22", "it", false),
    // Portuguese
    ("vosk-model-small-pt-0.3", "pt", true),
    ("vosk-model-pt-fb-v0.1.1-20220516_2113", "pt", false),
    // Dutch
    ("vosk-model-small-nl-0.22", "nl", true),
    ("vosk-model-nl-spraakherkenning-0.6", "nl", false),
    // Arabic
    ("vosk-model-ar-mgb2-0.4", "ar", false),
    // Hindi
    ("vosk-model-small-hi-0.22", "hi", true),
    ("vosk-model-hi-0.22", "hi", false),
];

/// Cached engine availability (computed once at startup)
static ENGINE_AVAILABILITY: OnceLock<EngineAvailability> = OnceLock::new();

/// Runtime engine availability info
#[derive(Debug, Clone)]
pub struct EngineAvailability {
    pub vosk: bool,
    pub whisper: bool,
    pub parakeet: bool,
    pub gpu: bool,
}

impl EngineAvailability {
    /// Get cached engine availability (computed once)
    pub fn get() -> &'static Self {
        ENGINE_AVAILABILITY.get_or_init(Self::detect)
    }

    /// Detect available engines at runtime
    fn detect() -> Self {
        Self {
            vosk: Self::check_vosk(),
            whisper: Self::check_whisper(),
            parakeet: Self::check_parakeet(),
            gpu: Self::check_gpu(),
        }
    }

    /// Check if vosk is available (libvosk.so must be loadable)
    fn check_vosk() -> bool {
        // Check common library paths for libvosk.so
        let mut lib_paths = vec![
            "/usr/lib/libvosk.so".to_string(),
            "/usr/lib64/libvosk.so".to_string(),
            "/usr/local/lib/libvosk.so".to_string(),
            "/usr/local/lib64/libvosk.so".to_string(),
        ];

        // Add ~/.local/lib (where install.sh puts it)
        if let Ok(home) = std::env::var("HOME") {
            lib_paths.push(format!("{}/.local/lib/libvosk.so", home));
        }

        // Also check LD_LIBRARY_PATH
        if let Ok(ld_path) = std::env::var("LD_LIBRARY_PATH") {
            for dir in ld_path.split(':') {
                let lib_path = Path::new(dir).join("libvosk.so");
                if lib_path.exists() {
                    return true;
                }
            }
        }

        // Check standard paths
        for path in &lib_paths {
            if Path::new(path).exists() {
                return true;
            }
        }

        false
    }

    /// Check if whisper is available (always true - compiled from source)
    fn check_whisper() -> bool {
        true
    }

    /// Check if parakeet is available (always true - ONNX runtime bundled)
    fn check_parakeet() -> bool {
        true
    }

    /// Check if GPU acceleration is available (CUDA)
    fn check_gpu() -> bool {
        // Check for CUDA libraries
        let cuda_paths = [
            "/usr/lib/libcudart.so",
            "/usr/lib64/libcudart.so",
            "/usr/local/cuda/lib64/libcudart.so",
        ];

        for path in &cuda_paths {
            if Path::new(path).exists() {
                return true;
            }
        }

        // Check LD_LIBRARY_PATH for CUDA
        if let Ok(ld_path) = std::env::var("LD_LIBRARY_PATH") {
            for dir in ld_path.split(':') {
                let lib_path = Path::new(dir).join("libcudart.so");
                if lib_path.exists() {
                    return true;
                }
            }
        }

        false
    }
}

/// Check if a specific model exists on disk
pub fn model_exists(model_spec: &str, models_dir: &Path) -> bool {
    let parts: Vec<&str> = model_spec.splitn(2, ':').collect();
    if parts.len() != 2 {
        return false;
    }

    let (engine, model_name) = (parts[0], parts[1]);

    match engine {
        "vosk" => models_dir.join(model_name).exists(),
        "whisper" => models_dir.join(model_name).exists(),
        "parakeet" => {
            if model_name == "default" {
                models_dir.join("parakeet").exists()
            } else {
                models_dir.join(model_name).exists()
            }
        }
        _ => false,
    }
}

/// List preview (fast) models for a given language
/// Only lists models for engines available at runtime
pub fn list_preview_models(language: &str) -> Vec<String> {
    let avail = EngineAvailability::get();
    let mut models = Vec::new();

    // Parakeet is fast enough for preview (English only)
    if avail.parakeet && language == "en" {
        models.push("parakeet:default".to_string());
    }

    // Vosk lgraph/small models for the language (fast, best for preview)
    if avail.vosk {
        for (name, lang, is_fast) in VOSK_MODELS {
            if *lang == language && *is_fast {
                models.push(format!("vosk:{}", name));
            }
        }
    }

    models
}

/// List final (accurate) models for a given language
/// Only lists models for engines available at runtime
pub fn list_final_models(language: &str) -> Vec<String> {
    let avail = EngineAvailability::get();
    let mut models = Vec::new();

    // Parakeet is highly accurate (English only)
    if avail.parakeet && language == "en" {
        models.push("parakeet:default".to_string());
    }

    // Whisper models (best accuracy, ordered by size)
    if avail.whisper {
        models.push("whisper:ggml-tiny.en.bin".to_string());
        models.push("whisper:ggml-base.en.bin".to_string());
        models.push("whisper:ggml-small.en.bin".to_string());
        models.push("whisper:ggml-medium.en.bin".to_string());
    }

    // Vosk accurate models (non-lgraph) for the language
    if avail.vosk {
        for (name, lang, is_fast) in VOSK_MODELS {
            if *lang == language && !*is_fast {
                models.push(format!("vosk:{}", name));
            }
        }
    }

    models
}

/// List available audio input devices
pub fn list_audio_devices() -> Vec<String> {
    let mut devices = Vec::new();

    // Standard options
    devices.push("default".to_string());
    devices.push("all".to_string());

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
    let avail = EngineAvailability::get();
    let mut engines = Vec::new();

    if avail.parakeet {
        engines.push("parakeet");
    }
    if avail.whisper {
        engines.push(if avail.gpu { "whisper (GPU)" } else { "whisper" });
    }
    if avail.vosk {
        engines.push("vosk");
    }

    if engines.is_empty() {
        "No engines available".to_string()
    } else {
        engines.join(", ")
    }
}
