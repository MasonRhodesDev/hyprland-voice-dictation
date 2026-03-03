//! Parakeet TDT transcription engine
//!
//! Fast CPU-optimized speech recognition using NVIDIA's Parakeet TDT model via ONNX.
//!
//! Long audio is automatically chunked into segments to avoid context limits.

use anyhow::Result;
use parakeet_rs::{ParakeetTDT, TimestampMode, Transcriber};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tracing::{debug, info};

use crate::chunking::{transcribe_chunked_with_timestamps, ChunkConfig, TimestampedChunkResult};
use crate::engine::TranscriptionEngine;

// Audio thresholds (at 16kHz sample rate)
const MIN_AUDIO_SAMPLES: usize = 2400; // 0.15s minimum for transcription
const RETRANSCRIBE_THRESHOLD: usize = 4800; // 0.3s of new audio before re-transcribing

/// Parakeet TDT-based transcription engine
///
/// Uses NVIDIA's Parakeet TDT model for fast, accurate transcription.
/// TDT (Token-and-Duration Transducer) provides better accuracy than CTC.
/// Achieves ~5x realtime performance on CPU.
///
/// Preview uses incremental transcription: only new audio since last call
/// is transcribed and appended to cached results for rolling preview.
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
            audio_buffer: Arc::new(Mutex::new(Vec::with_capacity(480_000))), // ~30s at 16kHz
            sample_rate,
            current_text: Arc::new(Mutex::new(String::new())),
            last_transcribed_len: Arc::new(Mutex::new(0)),
            chunk_config,
        })
    }

    /// Convert i16 samples to f32 for parakeet-rs
    fn samples_to_f32(samples: &[i16]) -> Vec<f32> {
        samples.iter().map(|&s| s as f32 / 32768.0).collect()
    }

    /// Run transcription on a single chunk of audio
    fn transcribe_chunk(&self, samples: &[i16]) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        let f32_samples = Self::samples_to_f32(samples);
        let mut parakeet = self.parakeet.lock()
            .map_err(|e| anyhow::anyhow!("Parakeet model lock poisoned: {}", e))?;
        let result = parakeet.transcribe_samples(f32_samples, self.sample_rate, 1, None)?;

        Ok(result.text)
    }

    /// Run transcription on a single chunk with word-level timestamps
    fn transcribe_chunk_with_timestamps(&self, samples: &[i16]) -> Result<TimestampedChunkResult> {
        if samples.is_empty() {
            return Ok(TimestampedChunkResult { text: String::new(), words: Vec::new() });
        }

        let f32_samples = Self::samples_to_f32(samples);
        let mut parakeet = self.parakeet.lock()
            .map_err(|e| anyhow::anyhow!("Parakeet model lock poisoned: {}", e))?;
        let result = parakeet.transcribe_samples(f32_samples, self.sample_rate, 1, Some(TimestampMode::Words))?;

        Ok(TimestampedChunkResult {
            text: result.text,
            words: result.tokens,
        })
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

        // Normalize audio levels for consistent transcription quality
        let normalized = normalize_audio(samples, 3000.0, 20.0);
        let samples = &normalized;

        // Use timestamped chunking for better merge accuracy when chunking is needed
        if self.chunk_config.needs_chunking(samples) {
            return transcribe_chunked_with_timestamps(samples, &self.chunk_config, |chunk| {
                self.transcribe_chunk_with_timestamps(chunk)
            });
        }

        // Short audio: single-pass transcription
        self.transcribe_chunk(samples)
    }
}

/// Normalize audio to a target RMS level for consistent transcription quality.
///
/// Different microphones produce different volume levels. Normalizing ensures
/// consistent SNR in the spectrogram regardless of input device.
///
/// Skips normalization when audio is near-silent (RMS < 1.0) or already
/// near the target level (gain within 5%).
fn normalize_audio(samples: &[i16], target_rms: f32, max_gain: f32) -> Vec<i16> {
    if samples.is_empty() {
        return Vec::new();
    }

    let rms = (samples.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / samples.len() as f64).sqrt() as f32;

    // Skip if near-silent
    if rms < 1.0 {
        return samples.to_vec();
    }

    let gain = (target_rms / rms).min(max_gain);

    // Skip if already near target (gain within 5%)
    if (gain - 1.0).abs() < 0.05 {
        return samples.to_vec();
    }

    debug!("normalize_audio: rms={:.1}, gain={:.2}x", rms, gain);

    samples.iter().map(|&s| {
        let amplified = s as f32 * gain;
        amplified.clamp(i16::MIN as f32, i16::MAX as f32) as i16
    }).collect()
}

impl TranscriptionEngine for ParakeetEngine {
    fn process_audio(&self, samples: &[i16]) -> Result<()> {
        // ONLY buffer audio here - never run transcription
        // Transcription happens in the preview task (100ms polling) and final result
        // Running it here blocks audio capture and causes data loss
        let mut buffer = self.audio_buffer.lock()
            .map_err(|e| anyhow::anyhow!("Audio buffer lock poisoned: {}", e))?;
        buffer.extend_from_slice(samples);
        Ok(())
    }

    fn get_current_text(&self) -> Result<String> {
        // Full-buffer transcription for preview: same approach as get_final_result()
        // This produces coherent output without word-boundary duplicates
        //
        // Lock ordering: audio_buffer -> last_transcribed_len -> current_text
        // This must be consistent with reset() to avoid deadlocks

        let buffer = self.audio_buffer.lock()
            .map_err(|e| anyhow::anyhow!("Audio buffer lock poisoned: {}", e))?;

        if buffer.is_empty() {
            return Ok(String::new());
        }

        // Need minimum audio to transcribe
        if buffer.len() < MIN_AUDIO_SAMPLES {
            return Ok(String::new());
        }

        let current_len = buffer.len();
        let last_len_val = {
            let last_len = self.last_transcribed_len.lock()
                .map_err(|e| anyhow::anyhow!("Last transcribed len lock poisoned: {}", e))?;
            *last_len
        };

        // Only re-transcribe when enough new audio accumulated
        // This balances responsiveness vs CPU usage
        if current_len <= last_len_val + RETRANSCRIBE_THRESHOLD {
            let cached = self.current_text.lock()
                .map_err(|e| anyhow::anyhow!("Current text lock poisoned: {}", e))?;
            return Ok(cached.clone());
        }

        // Transcribe FULL buffer (same as get_final_result)
        let full_audio = buffer.clone();
        drop(buffer);

        debug!("Preview transcription: {} samples ({:.2}s)",
               full_audio.len(), full_audio.len() as f32 / 16000.0);

        let full_text = self.transcribe_buffer(&full_audio)?;

        // Replace cache with new result (not append)
        // Lock ordering: current_text -> last_transcribed_len
        {
            let mut cached = self.current_text.lock()
                .map_err(|e| anyhow::anyhow!("Current text lock poisoned: {}", e))?;
            *cached = full_text.clone();
        }
        {
            let mut last_len = self.last_transcribed_len.lock()
                .map_err(|e| anyhow::anyhow!("Last transcribed len lock poisoned: {}", e))?;
            *last_len = current_len;
        }

        Ok(full_text)
    }

    fn get_final_result(&self) -> Result<String> {
        let buffer = self.audio_buffer.lock()
            .map_err(|e| anyhow::anyhow!("Audio buffer lock poisoned: {}", e))?;
        let samples = buffer.clone();
        drop(buffer);
        self.transcribe_buffer(&samples)
    }

    fn get_cached_text(&self) -> String {
        // Return the cached preview text without re-transcribing
        // Useful in single-model mode where preview already has full transcription
        self.current_text.lock()
            .map(|guard| guard.clone())
            .unwrap_or_default()
    }

    fn get_audio_buffer(&self) -> Vec<i16> {
        self.audio_buffer.lock()
            .map(|guard| guard.clone())
            .unwrap_or_default()
    }

    fn reset(&self) {
        // Lock ordering: audio_buffer -> current_text -> last_transcribed_len
        // Using if-let to gracefully handle poisoned locks without panicking
        if let Ok(mut buffer) = self.audio_buffer.lock() {
            buffer.clear();
        }
        if let Ok(mut text) = self.current_text.lock() {
            text.clear();
        }
        if let Ok(mut last_len) = self.last_transcribed_len.lock() {
            *last_len = 0;
        }
    }
}
