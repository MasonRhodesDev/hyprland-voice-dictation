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
use crate::openai_engine::OpenAiEngine;
use crate::parakeet_engine::ParakeetEngine;
use crate::stream_engine::{LocalEngineDriver, LocalModel, StreamingEngine};

/// Transcription provider selected by a model spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    /// Local Parakeet ONNX models (TDT / CTC / CTC-direct).
    Parakeet,
    /// Hosted OpenAI transcription (batch).
    OpenAi,
}

/// Parsed model specification from config
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub provider: Provider,
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
        let provider = match self.provider {
            Provider::Parakeet => "parakeet",
            Provider::OpenAi => "openai",
        };
        write!(f, "{}:{}", provider, self.model_name)
    }
}

impl ModelSpec {
    /// Parse a model specification string (format: "provider:model_name").
    ///
    /// # Examples
    /// - "parakeet:default"         -> local TDT engine
    /// - "parakeet:ctc-1.1b"        -> local CTC engine
    /// - "openai:gpt-4o-transcribe" -> hosted OpenAI engine (opt-in)
    pub fn parse(spec: &str) -> Result<Self> {
        let parts: Vec<&str> = spec.splitn(2, ':').collect();
        if parts.len() != 2 {
            return Err(anyhow!(
                "Invalid model spec '{}', expected format 'provider:model_name'",
                spec
            ));
        }

        let provider = match parts[0] {
            "parakeet" => Provider::Parakeet,
            "openai" => Provider::OpenAi,
            other => {
                return Err(anyhow!(
                    "Unsupported engine '{}'. Supported: 'parakeet', 'openai'.",
                    other
                ))
            }
        };

        Ok(Self { provider, model_name: parts[1].to_string() })
    }

    /// Get the base models directory
    fn get_models_dir() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home).join(".config").join("voice-dictation").join("models")
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

    /// Check whether the selected engine is usable.
    ///
    /// Parakeet checks for on-disk ONNX model files; OpenAI checks that an API
    /// key is present in the environment (no local model needed).
    pub fn is_available(&self) -> bool {
        if self.provider == Provider::OpenAi {
            return std::env::var("OPENAI_API_KEY").map(|k| !k.is_empty()).unwrap_or(false);
        }
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

    /// Create a pull-based transcription engine (retained for test-loop-ui).
    /// Only valid for `parakeet` specs; use `create_streaming_engine` otherwise.
    pub fn create_engine(&self, sample_rate: u32) -> Result<Arc<dyn TranscriptionEngine>> {
        if self.provider != Provider::Parakeet {
            return Err(anyhow!(
                "create_engine supports only 'parakeet'; use create_streaming_engine for '{}'",
                self
            ));
        }
        if self.is_ctc_direct() {
            info!(
                "Creating CTC Direct engine (ONNX + beam search) for model '{}'",
                self.model_name
            );
            let model_path = self.model_path();
            let engine = CtcDirectEngine::new(model_path, sample_rate)?;
            Ok(Arc::new(engine))
        } else if self.is_ctc() {
            info!("Creating parakeet CTC engine with model '{}'", self.model_name);
            let model_path = self.model_path();
            let hotwords_path = Some(hotword_trie::default_hotwords_path());
            // Default beam width of 10 when hotwords are present
            let beam_width = 10;
            let engine = CtcEngine::new(model_path, sample_rate, hotwords_path, beam_width)?;
            Ok(Arc::new(engine))
        } else {
            info!("Creating parakeet TDT engine with model '{}'", self.model_name);
            let model_path = self.model_path();
            let engine = ParakeetEngine::new(model_path, sample_rate)?;
            Ok(Arc::new(engine))
        }
    }

    /// Create an event-emitting [`StreamingEngine`] for this spec by wrapping the
    /// selected local model in a `LocalEngineDriver`. This is the daemon's engine
    /// factory; the older `create_engine` (pull-based trait) is retained for
    /// test-loop-ui.
    pub fn create_streaming_engine(&self, sample_rate: u32) -> Result<Arc<dyn StreamingEngine>> {
        match self.provider {
            Provider::OpenAi => {
                info!("Creating OpenAI streaming engine (model '{}')", self.model_name);
                Ok(Arc::new(OpenAiEngine::new(self.model_name.clone(), sample_rate)?))
            }
            Provider::Parakeet => {
                let model = self.build_local_model(sample_rate)?;
                Ok(Arc::new(LocalEngineDriver::new(model)))
            }
        }
    }

    /// Construct the selected local model as a `LocalModel` trait object.
    fn build_local_model(&self, sample_rate: u32) -> Result<Arc<dyn LocalModel>> {
        let model_path = self.model_path();
        if self.is_ctc_direct() {
            info!(
                "Creating CTC Direct engine (ONNX + beam search) for model '{}'",
                self.model_name
            );
            Ok(Arc::new(CtcDirectEngine::new(model_path, sample_rate)?))
        } else if self.is_ctc() {
            info!("Creating parakeet CTC engine with model '{}'", self.model_name);
            let hotwords_path = Some(hotword_trie::default_hotwords_path());
            Ok(Arc::new(CtcEngine::new(model_path, sample_rate, hotwords_path, 10)?))
        } else {
            info!("Creating parakeet TDT engine with model '{}'", self.model_name);
            Ok(Arc::new(ParakeetEngine::new(model_path, sample_rate)?))
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
            if entry.path().extension().and_then(|s| s.to_str()) == Some("onnx") {
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
    fn test_parse_openai_spec() {
        let spec = ModelSpec::parse("openai:gpt-4o-transcribe").unwrap();
        assert_eq!(spec.provider, Provider::OpenAi);
        assert_eq!(spec.model_name, "gpt-4o-transcribe");
        assert!(!spec.is_ctc());
        assert_eq!(format!("{}", spec), "openai:gpt-4o-transcribe");
    }

    #[test]
    fn test_parakeet_spec_has_parakeet_provider() {
        let spec = ModelSpec::parse("parakeet:default").unwrap();
        assert_eq!(spec.provider, Provider::Parakeet);
        // create_engine (pull trait) rejects non-parakeet specs.
        let openai = ModelSpec::parse("openai:gpt-4o-transcribe").unwrap();
        assert!(openai.create_engine(16000).is_err());
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
