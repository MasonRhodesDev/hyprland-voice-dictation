//! Parakeet TDT transcription engine
//!
//! Fast CPU-optimized speech recognition using NVIDIA's Parakeet TDT model via ONNX.
//! Requires `parakeet` feature to be enabled.
//!
//! Long audio is automatically chunked into segments to avoid context limits.

#[cfg(feature = "parakeet")]
use anyhow::Result;
#[cfg(feature = "parakeet")]
use parakeet_rs::{ParakeetTDT, Transcriber};
#[cfg(feature = "parakeet")]
use std::path::PathBuf;
#[cfg(feature = "parakeet")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "parakeet")]
use tracing::{debug, info};

#[cfg(feature = "parakeet")]
use crate::chunking::{transcribe_chunked, ChunkConfig};
#[cfg(feature = "parakeet")]
use crate::engine::TranscriptionEngine;

/// Parakeet TDT-based transcription engine
///
/// Uses NVIDIA's Parakeet TDT model for fast, accurate transcription.
/// TDT (Token-and-Duration Transducer) provides better accuracy than CTC.
/// Achieves ~5x realtime performance on CPU.
///
/// Preview uses incremental transcription: only new audio since last call
/// is transcribed and appended to cached results for rolling preview.
#[cfg(feature = "parakeet")]
pub struct ParakeetEngine {
    parakeet: Arc<Mutex<ParakeetTDT>>,
    audio_buffer: Arc<Mutex<Vec<i16>>>,
    sample_rate: u32,
    /// Cached transcription text (accumulated from incremental transcriptions)
    current_text: Arc<Mutex<String>>,
    /// Position in audio_buffer up to which we've transcribed (for incremental preview)
    last_transcribed_len: Arc<Mutex<usize>>,
    /// Chunking configuration for long audio
    chunk_config: ChunkConfig,
}

#[cfg(feature = "parakeet")]
impl ParakeetEngine {
    /// Create a new Parakeet engine
    ///
    /// # Arguments
    /// * `model_path` - Path to the Parakeet model directory
    /// * `sample_rate` - Audio sample rate (must be 16000 for Parakeet)
    pub fn new(model_path: PathBuf, sample_rate: u32) -> Result<Self> {
        info!("Loading Parakeet model from {:?}", model_path);

        // Parakeet requires 16kHz audio
        if sample_rate != 16000 {
            anyhow::bail!("Parakeet requires 16kHz audio, got {} Hz", sample_rate);
        }

        let parakeet = ParakeetTDT::from_pretrained(model_path.to_str().unwrap_or("."), None)?;

        // Configure chunking for Parakeet's attention limits (30s safe for CPU)
        let chunk_config = ChunkConfig::new(30, 2, sample_rate);

        Ok(Self {
            parakeet: Arc::new(Mutex::new(parakeet)),
            audio_buffer: Arc::new(Mutex::new(Vec::new())),
            sample_rate,
            current_text: Arc::new(Mutex::new(String::new())),
            last_transcribed_len: Arc::new(Mutex::new(0)),
            chunk_config,
        })
    }

    /// Ensure the Parakeet model is downloaded
    pub fn ensure_model(model_dir: &std::path::Path) -> Result<PathBuf> {
        let model_path = model_dir.join("parakeet");
        if model_path.exists() {
            return Ok(model_path);
        }

        std::fs::create_dir_all(&model_path)?;

        info!("Parakeet model not found at {:?}", model_path);
        info!("Please download the model using:");
        info!("  huggingface-cli download nvidia/parakeet-tdt-0.6b --local-dir {:?}", model_path);

        anyhow::bail!(
            "Parakeet model not found. Download with: huggingface-cli download nvidia/parakeet-tdt-0.6b --local-dir {}",
            model_path.display()
        )
    }

    /// Write audio buffer to a temporary WAV file for transcription
    fn write_temp_wav(&self, samples: &[i16]) -> Result<tempfile::NamedTempFile> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut temp_file = tempfile::Builder::new()
            .suffix(".wav")
            .tempfile()?;

        {
            let mut writer = hound::WavWriter::new(&mut temp_file, spec)?;
            for &sample in samples {
                writer.write_sample(sample)?;
            }
            writer.finalize()?;
        }

        temp_file.as_file_mut().sync_all()?;
        Ok(temp_file)
    }

    /// Run transcription on a single chunk of audio
    fn transcribe_chunk(&self, samples: &[i16]) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        let temp_file = self.write_temp_wav(samples)?;
        let mut parakeet = self.parakeet.lock().unwrap();
        let result = parakeet.transcribe_file(temp_file.path(), None)?;

        Ok(result.text)
    }

    /// Run transcription on accumulated audio, chunking if necessary
    fn transcribe_buffer(&self, samples: &[i16]) -> Result<String> {
        if samples.is_empty() {
            debug!("transcribe_buffer: empty samples");
            return Ok(String::new());
        }

        // Check audio statistics
        let max_sample = samples.iter().map(|s| s.abs()).max().unwrap_or(0);
        let rms = (samples.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / samples.len() as f64).sqrt();
        let duration_secs = samples.len() as f32 / self.sample_rate as f32;
        debug!(
            "transcribe_buffer: {} samples, max={}, rms={:.1}, duration={:.2}s",
            samples.len(),
            max_sample,
            rms,
            duration_secs
        );

        // Use the generalized chunking utility
        transcribe_chunked(samples, &self.chunk_config, |chunk| {
            self.transcribe_chunk(chunk)
        })
    }
}

#[cfg(feature = "parakeet")]
impl TranscriptionEngine for ParakeetEngine {
    fn process_audio(&self, samples: &[i16]) -> Result<()> {
        // ONLY buffer audio here - never run transcription
        // Transcription happens in the preview task (100ms polling) and final result
        // Running it here blocks audio capture and causes data loss
        let mut buffer = self.audio_buffer.lock().unwrap();
        buffer.extend_from_slice(samples);
        Ok(())
    }

    fn get_current_text(&self) -> Result<String> {
        // Incremental transcription for preview: only transcribe NEW audio
        // This is called from the preview task (100ms polling), not the audio thread
        let buffer = self.audio_buffer.lock().unwrap();
        if buffer.is_empty() {
            return Ok(String::new());
        }

        let current_len = buffer.len();
        let mut last_len = self.last_transcribed_len.lock().unwrap();

        // If no new audio, return cached result
        if current_len <= *last_len {
            return Ok(self.current_text.lock().unwrap().clone());
        }

        // Need minimum ~0.15s (2400 samples) to avoid ndarray shape overflow in model
        let new_samples = current_len - *last_len;
        if new_samples < 2400 {
            return Ok(self.current_text.lock().unwrap().clone());
        }

        // Transcribe only the new chunk
        let new_audio: Vec<i16> = buffer[*last_len..].to_vec();
        drop(buffer);

        let new_text = self.transcribe_chunk(&new_audio)?;

        // Append to cached text
        let mut cached = self.current_text.lock().unwrap();
        if !new_text.is_empty() {
            if !cached.is_empty() {
                cached.push(' ');
            }
            cached.push_str(&new_text);
            debug!("Incremental transcription: +{} samples -> '{}' (total: {} chars)",
                   new_samples, new_text, cached.len());
        }
        *last_len = current_len;

        Ok(cached.clone())
    }

    fn get_final_result(&self) -> Result<String> {
        let buffer = self.audio_buffer.lock().unwrap();
        let samples = buffer.clone();
        drop(buffer);
        self.transcribe_buffer(&samples)
    }

    fn get_audio_buffer(&self) -> Vec<i16> {
        let buffer = self.audio_buffer.lock().unwrap();
        buffer.clone()
    }

    fn reset(&self) {
        let mut buffer = self.audio_buffer.lock().unwrap();
        buffer.clear();
        let mut text = self.current_text.lock().unwrap();
        text.clear();
        let mut last_len = self.last_transcribed_len.lock().unwrap();
        *last_len = 0;
    }
}

// Stub when feature not enabled
#[cfg(not(feature = "parakeet"))]
pub struct ParakeetEngine;

#[cfg(not(feature = "parakeet"))]
impl ParakeetEngine {
    pub fn new(_model_path: std::path::PathBuf, _sample_rate: u32) -> anyhow::Result<Self> {
        anyhow::bail!("Parakeet feature not enabled. Rebuild with --features parakeet")
    }
}
