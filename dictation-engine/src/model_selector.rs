//! Model selection and engine factory
//!
//! Provides a unified interface for parsing model specifications and creating
//! transcription engines. Supports both Parakeet TDT and CTC model variants.
//!
//! Model specification format: `parakeet:<model_name>`
//! - Names starting with `ctc-` route to the CTC engine (e.g., `parakeet:ctc-1.1b`)
//! - All other names route to the TDT engine (e.g., `parakeet:default`)

use anyhow::{anyhow, Result};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

use crate::ctc_direct_engine::CtcDirectEngine;
use crate::ctc_engine::CtcEngine;
use crate::engine::TranscriptionEngine;
use crate::hotword_trie;
use crate::parakeet_engine::ParakeetEngine;

/// Parsed model specification from config
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub model_name: String,
}

impl ModelSpec {
    /// Returns true if this spec selects a CTC model variant
    pub fn is_ctc(&self) -> bool {
        self.model_name.starts_with("ctc-") || self.model_name == "ctc"
    }

    /// Returns true if this spec selects the CTC Direct (ONNX + beam search) engine
    pub fn is_ctc_direct(&self) -> bool {
        self.model_name == "ctc-direct"
    }
}

impl std::fmt::Display for ModelSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "parakeet:{}", self.model_name)
    }
}

impl ModelSpec {
    /// Parse a model specification string (format: "parakeet:model_name")
    ///
    /// # Examples
    /// - "parakeet:default" -> TDT engine
    /// - "parakeet:ctc-1.1b" -> CTC engine
    /// - "parakeet:ctc-0.6b" -> CTC engine
    pub fn parse(spec: &str) -> Result<Self> {
        let parts: Vec<&str> = spec.splitn(2, ':').collect();
        if parts.len() != 2 {
            return Err(anyhow!(
                "Invalid model spec '{}', expected format 'parakeet:model_name'",
                spec
            ));
        }

        if parts[0] != "parakeet" {
            return Err(anyhow!(
                "Unsupported engine '{}'. Only 'parakeet' is supported.",
                parts[0]
            ));
        }

        Ok(Self {
            model_name: parts[1].to_string(),
        })
    }

    /// Get the base models directory
    fn get_models_dir() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home)
            .join(".config")
            .join("voice-dictation")
            .join("models")
    }

    /// Get the full path to the model directory.
    ///
    /// CTC models are stored in `models/parakeet-ctc/`, TDT in `models/parakeet/`.
    /// `ctc-direct` also uses `models/parakeet-ctc/` (same model files, different engine).
    pub fn model_path(&self) -> PathBuf {
        if self.is_ctc() || self.is_ctc_direct() {
            Self::get_models_dir().join("parakeet-ctc")
        } else {
            Self::get_models_dir().join("parakeet")
        }
    }

    /// Check if the model is available on the filesystem
    pub fn is_available(&self) -> bool {
        let path = self.model_path();
        if self.is_ctc() || self.is_ctc_direct() {
            // CTC needs a single model ONNX file + tokenizer
            has_onnx_model(&path) && path.join("tokenizer.json").exists()
        } else {
            // TDT needs encoder and decoder ONNX files
            path.join("encoder-model.onnx").exists()
                && path.join("decoder_joint-model.onnx").exists()
        }
    }

    /// Create a transcription engine from this specification
    pub fn create_engine(&self, sample_rate: u32) -> Result<Arc<dyn TranscriptionEngine>> {
        if self.is_ctc_direct() {
            info!(
                "Creating CTC Direct engine (ONNX + beam search) for model '{}'",
                self.model_name
            );
            let model_path = self.model_path();
            let engine = CtcDirectEngine::new(model_path, sample_rate)?;
            Ok(Arc::new(engine))
        } else if self.is_ctc() {
            info!(
                "Creating parakeet CTC engine with model '{}'",
                self.model_name
            );
            let model_path = self.model_path();
            let hotwords_path = Some(hotword_trie::default_hotwords_path());
            // Default beam width of 10 when hotwords are present
            let beam_width = 10;
            let engine = CtcEngine::new(model_path, sample_rate, hotwords_path, beam_width)?;
            Ok(Arc::new(engine))
        } else {
            info!(
                "Creating parakeet TDT engine with model '{}'",
                self.model_name
            );
            let model_path = self.model_path();
            let engine = ParakeetEngine::new(model_path, sample_rate)?;
            Ok(Arc::new(engine))
        }
    }
}

/// Check if a directory contains any .onnx model file
fn has_onnx_model(dir: &std::path::Path) -> bool {
    let candidates = ["model.onnx", "model_fp16.onnx", "model_int8.onnx", "model_q4.onnx"];
    for candidate in &candidates {
        if dir.join(candidate).exists() {
            return true;
        }
    }
    // Search for any .onnx file
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if entry
                .path()
                .extension()
                .and_then(|s| s.to_str())
                == Some("onnx")
            {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_parakeet_spec() {
        let spec = ModelSpec::parse("parakeet:default").unwrap();
        assert_eq!(spec.model_name, "default");
        assert!(!spec.is_ctc());
    }

    #[test]
    fn test_parse_ctc_spec() {
        let spec = ModelSpec::parse("parakeet:ctc-1.1b").unwrap();
        assert_eq!(spec.model_name, "ctc-1.1b");
        assert!(spec.is_ctc());

        let spec = ModelSpec::parse("parakeet:ctc-0.6b").unwrap();
        assert!(spec.is_ctc());
    }

    #[test]
    fn test_parse_invalid_format() {
        assert!(ModelSpec::parse("invalid").is_err());
        assert!(ModelSpec::parse("vosk:model").is_err());
        assert!(ModelSpec::parse("whisper:model").is_err());
    }

    #[test]
    fn test_display() {
        let spec = ModelSpec::parse("parakeet:default").unwrap();
        assert_eq!(format!("{}", spec), "parakeet:default");

        let spec = ModelSpec::parse("parakeet:ctc-1.1b").unwrap();
        assert_eq!(format!("{}", spec), "parakeet:ctc-1.1b");
    }

    #[test]
    fn test_model_path_tdt() {
        let spec = ModelSpec::parse("parakeet:default").unwrap();
        let path = spec.model_path();
        assert!(path.to_str().unwrap().ends_with("models/parakeet"));
    }

    #[test]
    fn test_model_path_ctc() {
        let spec = ModelSpec::parse("parakeet:ctc-1.1b").unwrap();
        let path = spec.model_path();
        assert!(path.to_str().unwrap().ends_with("models/parakeet-ctc"));
    }

    #[test]
    fn test_parse_ctc_direct_spec() {
        let spec = ModelSpec::parse("parakeet:ctc-direct").unwrap();
        assert_eq!(spec.model_name, "ctc-direct");
        assert!(spec.is_ctc_direct());
    }

    #[test]
    fn test_model_path_ctc_direct() {
        let spec = ModelSpec::parse("parakeet:ctc-direct").unwrap();
        let path = spec.model_path();
        // ctc-direct uses the same parakeet-ctc model directory
        assert!(path.to_str().unwrap().ends_with("models/parakeet-ctc"));
    }
}
