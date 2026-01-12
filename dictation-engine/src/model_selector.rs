//! Model selection and engine factory
//!
//! Provides a unified interface for parsing model specifications and creating
//! transcription engines. This is the ONLY place where engine-specific code
//! should exist - the rest of the application uses trait objects.

use anyhow::{anyhow, Result};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

use crate::engine::TranscriptionEngine;
use crate::model_manager;
use crate::whisper_engine::WhisperEngine;

#[cfg(feature = "vosk")]
use crate::vosk_engine::VoskEngine;

#[cfg(feature = "parakeet")]
use crate::parakeet_engine::ParakeetEngine;

/// Supported transcription engine types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineType {
    Vosk,
    Whisper,
    Parakeet,
}

impl std::fmt::Display for EngineType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineType::Vosk => write!(f, "vosk"),
            EngineType::Whisper => write!(f, "whisper"),
            EngineType::Parakeet => write!(f, "parakeet"),
        }
    }
}

/// Parsed model specification from config
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub engine: EngineType,
    pub model_name: String,
}

impl ModelSpec {
    /// Parse a model specification string (format: "engine:model_name")
    ///
    /// # Examples
    /// - "whisper:ggml-small.en.bin"
    /// - "vosk:vosk-model-en-us-0.22"
    /// - "parakeet:default"
    pub fn parse(spec: &str) -> Result<Self> {
        let parts: Vec<&str> = spec.splitn(2, ':').collect();
        if parts.len() != 2 {
            return Err(anyhow!(
                "Invalid model spec '{}', expected format 'engine:model_name'",
                spec
            ));
        }

        let engine = match parts[0] {
            "vosk" => EngineType::Vosk,
            "whisper" => EngineType::Whisper,
            "parakeet" => EngineType::Parakeet,
            other => {
                return Err(anyhow!(
                    "Unknown engine '{}', valid options: vosk, whisper, parakeet",
                    other
                ))
            }
        };

        Ok(Self {
            engine,
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
        let base_dir = Self::get_models_dir();
        match self.engine {
            EngineType::Vosk => base_dir.join(&self.model_name),
            EngineType::Whisper => base_dir.join("whisper").join(&self.model_name),
            EngineType::Parakeet => base_dir.join("parakeet"),
        }
    }

    /// Check if the model is available on the filesystem
    pub fn is_available(&self) -> bool {
        let path = self.model_path();
        match self.engine {
            EngineType::Vosk => path.exists() && path.is_dir(),
            EngineType::Whisper => path.exists() && path.is_file(),
            EngineType::Parakeet => {
                // Parakeet TDT needs encoder and decoder ONNX files
                path.join("encoder-model.onnx").exists()
                    && path.join("decoder_joint-model.onnx").exists()
            }
        }
    }

    /// Create a transcription engine from this specification
    ///
    /// This is the factory method that creates the appropriate engine type
    /// based on the parsed specification. The rest of the application should
    /// only interact with engines through the TranscriptionEngine trait.
    pub fn create_engine(&self, sample_rate: u32) -> Result<Arc<dyn TranscriptionEngine>> {
        info!(
            "Creating {} engine with model '{}'",
            self.engine, self.model_name
        );

        match self.engine {
            EngineType::Whisper => {
                let models_dir = Self::get_models_dir().join("whisper");
                let models_dir_str = models_dir.to_str()
                    .ok_or_else(|| anyhow!("Models directory path contains invalid UTF-8"))?;
                let model_path =
                    model_manager::ensure_whisper_model(&self.model_name, models_dir_str)?;
                let engine = WhisperEngine::new(model_path.to_str().unwrap(), sample_rate)?;
                Ok(Arc::new(engine))
            }

            #[cfg(feature = "vosk")]
            EngineType::Vosk => {
                let model_path = self.model_path();
                if !model_path.exists() {
                    return Err(anyhow!(
                        "Vosk model not found at {:?}. Download it first.",
                        model_path
                    ));
                }
                let engine = VoskEngine::new(model_path.to_str().unwrap(), sample_rate)?;
                Ok(Arc::new(engine))
            }

            #[cfg(not(feature = "vosk"))]
            EngineType::Vosk => Err(anyhow!(
                "Vosk engine not available. Rebuild with --features vosk"
            )),

            #[cfg(feature = "parakeet")]
            EngineType::Parakeet => {
                let model_path = self.model_path();
                let engine = ParakeetEngine::new(model_path, sample_rate)?;
                Ok(Arc::new(engine))
            }

            #[cfg(not(feature = "parakeet"))]
            EngineType::Parakeet => Err(anyhow!(
                "Parakeet engine not available. Rebuild with --features parakeet"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_whisper_spec() {
        let spec = ModelSpec::parse("whisper:ggml-small.en.bin").unwrap();
        assert_eq!(spec.engine, EngineType::Whisper);
        assert_eq!(spec.model_name, "ggml-small.en.bin");
    }

    #[test]
    fn test_parse_vosk_spec() {
        let spec = ModelSpec::parse("vosk:vosk-model-en-us-0.22").unwrap();
        assert_eq!(spec.engine, EngineType::Vosk);
        assert_eq!(spec.model_name, "vosk-model-en-us-0.22");
    }

    #[test]
    fn test_parse_parakeet_spec() {
        let spec = ModelSpec::parse("parakeet:default").unwrap();
        assert_eq!(spec.engine, EngineType::Parakeet);
        assert_eq!(spec.model_name, "default");
    }

    #[test]
    fn test_parse_invalid_format() {
        assert!(ModelSpec::parse("invalid").is_err());
        assert!(ModelSpec::parse("unknown:model").is_err());
    }

    #[test]
    fn test_model_path_whisper() {
        let spec = ModelSpec::parse("whisper:ggml-small.en.bin").unwrap();
        let path = spec.model_path();
        assert!(path.to_string_lossy().contains("whisper"));
        assert!(path.to_string_lossy().contains("ggml-small.en.bin"));
    }

    #[test]
    fn test_model_path_vosk() {
        let spec = ModelSpec::parse("vosk:vosk-model-en-us-0.22").unwrap();
        let path = spec.model_path();
        assert!(path.to_string_lossy().contains("vosk-model-en-us-0.22"));
    }
}
