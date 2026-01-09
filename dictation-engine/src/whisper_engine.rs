use super::chunking::{transcribe_chunked, ChunkConfig};
use super::engine::TranscriptionEngine;
use anyhow::Result;
use std::sync::{Arc, Mutex};
use tracing::{debug, error, info};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Whisper-based speech-to-text transcription engine.
///
/// Uses OpenAI's Whisper model via whisper.cpp Rust bindings for
/// high-accuracy offline transcription with native punctuation support.
///
/// Features:
/// - Better accuracy than Vosk on technical terms
/// - Native punctuation and capitalization
/// - Handles proper nouns and complex vocabulary
/// - CPU-efficient with quantized models
#[allow(dead_code)]
pub struct WhisperEngine {
    context: Arc<WhisperContext>,
    accumulated_text: Arc<Mutex<String>>,
    audio_buffer: Arc<Mutex<Vec<i16>>>,
    sample_rate: u32,
    /// Chunking configuration for long audio (30s chunks, 2s overlap)
    chunk_config: ChunkConfig,
}

#[allow(dead_code)]
impl WhisperEngine {
    /// Create a new Whisper transcription engine.
    ///
    /// # Arguments
    /// * `model_path` - Path to the Whisper GGML model file (e.g., "ggml-base.en.bin")
    /// * `sample_rate` - Audio sample rate in Hz (must be 16000 for Whisper)
    ///
    /// # Returns
    /// * `Ok(WhisperEngine)` if model loaded successfully
    /// * `Err` if model loading failed
    ///
    /// # Example
    /// ```ignore
    /// let engine = WhisperEngine::new("models/ggml-small.en.bin", 16000)?;
    /// ```
    pub fn new(model_path: &str, sample_rate: u32) -> Result<Self> {
        info!("Loading Whisper model from: {}", model_path);

        if sample_rate != 16000 {
            return Err(anyhow::anyhow!(
                "Whisper requires 16kHz sample rate, got {}Hz",
                sample_rate
            ));
        }

        let context = WhisperContext::new_with_params(
            model_path,
            WhisperContextParameters::default(),
        )
        .map_err(|e| anyhow::anyhow!("Failed to load Whisper model: {:?}", e))?;

        info!("✓ Whisper model loaded successfully");

        // Whisper has 30s context window; use 30s chunks with 2s overlap
        let chunk_config = ChunkConfig::new(30, 2, sample_rate);

        Ok(Self {
            context: Arc::new(context),
            accumulated_text: Arc::new(Mutex::new(String::new())),
            audio_buffer: Arc::new(Mutex::new(Vec::new())),
            sample_rate,
            chunk_config,
        })
    }

    /// Transcribe a single chunk of i16 audio samples
    fn transcribe_chunk(&self, context: &WhisperContext, samples: &[i16]) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        // Convert i16 PCM samples to f32 mono required by Whisper
        let mut float_samples = vec![0.0f32; samples.len()];
        whisper_rs::convert_integer_to_float_audio(samples, &mut float_samples)
            .map_err(|e| anyhow::anyhow!("Audio conversion i16→f32 failed: {:?}", e))?;

        // Create transcription state from context
        let mut state = context
            .create_state()
            .map_err(|e| {
                error!("Failed to create Whisper state: {:?}", e);
                anyhow::anyhow!("Failed to create Whisper state: {:?}", e)
            })?;

        // Configure transcription parameters
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        // Language and output settings
        params.set_language(Some("en"));
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        // Translation and formatting
        params.set_translate(false);
        params.set_no_context(true);
        params.set_single_segment(false);

        debug!("transcribe_chunk: processing {:.2}s of audio",
               float_samples.len() as f32 / self.sample_rate as f32);

        // Run the actual transcription
        state
            .full(params, &float_samples[..])
            .map_err(|e| {
                error!("Whisper transcription failed: {:?}", e);
                anyhow::anyhow!("Whisper transcription failed: {:?}", e)
            })?;

        // Extract text from all segments using iterator
        let segments: Vec<String> = state
            .as_iter()
            .filter_map(|segment| {
                segment
                    .to_str_lossy()
                    .ok()
                    .map(|text| text.trim().to_string())
            })
            .filter(|text| !text.is_empty())
            .collect();

        Ok(segments.join(" "))
    }

    /// Run a correction pass using an accurate Whisper model.
    ///
    /// This method transcribes the entire accumulated audio buffer using
    /// a larger/more accurate Whisper model for the final result.
    /// Long audio is automatically chunked to avoid context limits.
    ///
    /// # Arguments
    /// * `accurate_context` - Whisper context for the accurate model
    ///
    /// # Returns
    /// * Final transcription with punctuation and capitalization
    pub fn run_correction_pass(&self, accurate_context: &WhisperContext) -> Result<String> {
        info!("Running Whisper correction pass...");

        let audio_buffer = self.audio_buffer.lock()
            .map_err(|e| anyhow::anyhow!("Audio buffer lock poisoned: {}", e))?;

        if audio_buffer.is_empty() {
            info!("Audio buffer empty, returning empty string");
            return Ok(String::new());
        }

        let samples = audio_buffer.clone();
        drop(audio_buffer);

        let duration_secs = samples.len() as f32 / self.sample_rate as f32;
        info!("Running Whisper transcription on {:.2}s of audio...", duration_secs);

        // Use chunking for long audio
        let result = transcribe_chunked(&samples, &self.chunk_config, |chunk| {
            self.transcribe_chunk(accurate_context, chunk)
        })?;

        info!("✓ Whisper transcription complete: {} characters", result.len());

        Ok(result)
    }
}

impl TranscriptionEngine for WhisperEngine {
    fn process_audio(&self, samples: &[i16]) -> Result<()> {
        let mut audio_buffer = self.audio_buffer.lock()
            .map_err(|e| anyhow::anyhow!("Audio buffer lock poisoned: {}", e))?;
        audio_buffer.extend_from_slice(samples);
        Ok(())
    }

    fn get_current_text(&self) -> Result<String> {
        // Whisper doesn't support incremental transcription efficiently,
        // so show recording duration as feedback instead of empty string.
        let buffer = self.audio_buffer.lock()
            .map_err(|e| anyhow::anyhow!("Audio buffer lock poisoned: {}", e))?;

        if buffer.is_empty() {
            Ok(String::new())
        } else {
            let duration = buffer.len() as f32 / self.sample_rate as f32;
            Ok(format!("Recording... ({:.1}s)", duration))
        }
    }

    fn get_final_result(&self) -> Result<String> {
        let text = self.accumulated_text.lock()
            .map_err(|e| anyhow::anyhow!("Accumulated text lock poisoned: {}", e))?;
        Ok(text.clone())
    }

    fn get_audio_buffer(&self) -> Vec<i16> {
        self.audio_buffer.lock()
            .map(|guard| guard.clone())
            .unwrap_or_default()
    }

    fn reset(&self) {
        if let Ok(mut buffer) = self.audio_buffer.lock() {
            buffer.clear();
        }
        if let Ok(mut text) = self.accumulated_text.lock() {
            text.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compilation() {
        // Placeholder test to verify the module compiles
        // Actual functionality testing requires Whisper model files
        assert!(true);
    }
}
