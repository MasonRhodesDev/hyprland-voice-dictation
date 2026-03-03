//! Model selection and engine factory (Parakeet-only)
//!
//! Provides a unified interface for parsing model specifications and creating
//! transcription engines. Only the Parakeet engine is supported.

use anyhow::{anyhow, Result};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

use crate::engine::TranscriptionEngine;
use crate::parakeet_engine::ParakeetEngine;

/// Parsed model specification from config
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub model_name: String,
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
    /// - "parakeet:default"
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

    /// Get the full path to the model
    pub fn model_path(&self) -> PathBuf {
        Self::get_models_dir().join("parakeet")
    }

    /// Check if the model is available on the filesystem
    pub fn is_available(&self) -> bool {
        let path = self.model_path();
        // Parakeet TDT needs encoder and decoder ONNX files
        path.join("encoder-model.onnx").exists()
            && path.join("decoder_joint-model.onnx").exists()
    }

    /// Create a transcription engine from this specification
    pub fn create_engine(&self, sample_rate: u32) -> Result<Arc<dyn TranscriptionEngine>> {
        info!("Creating parakeet engine with model '{}'", self.model_name);
        let model_path = self.model_path();
        let engine = ParakeetEngine::new(model_path, sample_rate)?;
        Ok(Arc::new(engine))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_parakeet_spec() {
        let spec = ModelSpec::parse("parakeet:default").unwrap();
        assert_eq!(spec.model_name, "default");
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
    }
}
