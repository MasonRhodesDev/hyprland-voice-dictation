//! Voice Activity Detection module
//!
//! Provides VAD trait and implementations for detecting speech in audio.
//! Includes both simple dB-threshold and Silero neural network detection.

use anyhow::Result;
use tracing::debug;

/// Trait for voice activity detection implementations
pub trait VoiceActivityDetector: Send + Sync {
    /// Process audio samples and return true if speech is detected
    fn process(&mut self, samples: &[i16]) -> Result<bool>;

    /// Reset internal state (call between recordings)
    fn reset(&mut self);
}

/// Simple dB-based VAD (always available)
pub struct DbThresholdVad {
    threshold_db: f32,
}

impl DbThresholdVad {
    pub fn new(threshold_db: f32) -> Self {
        Self { threshold_db }
    }

    fn calculate_rms(samples: &[i16]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = samples.iter().map(|&s| (s as f64).powi(2)).sum();
        (sum / samples.len() as f64).sqrt() as f32
    }

    fn rms_to_db(rms: f32) -> f32 {
        if rms <= 0.0 {
            return -100.0;
        }
        20.0 * (rms / 32768.0).log10()
    }
}

impl VoiceActivityDetector for DbThresholdVad {
    fn process(&mut self, samples: &[i16]) -> Result<bool> {
        let rms = Self::calculate_rms(samples);
        let db = Self::rms_to_db(rms);
        Ok(db > self.threshold_db)
    }

    fn reset(&mut self) {
        // No state to reset
    }
}

/// Silero VAD implementation using ONNX model
pub mod silero {
    use super::*;
    use ort::session::{Session, builder::GraphOptimizationLevel};
    use ort::value::Value;
    use std::path::Path;
    use tracing::warn;

    /// Silero VAD detector using ONNX Runtime
    pub struct SileroVadDetector {
        session: Session,
        threshold: f32,
        sample_rate: i64,
        /// Internal state tensors for streaming
        state: Vec<f32>,
        sr_tensor: Vec<i64>,
        /// Accumulated samples for batch processing
        buffer: Vec<f32>,
        /// Minimum samples needed for VAD (512 for 16kHz, 256 for 8kHz)
        min_samples: usize,
    }

    impl SileroVadDetector {
        /// Create a new Silero VAD detector
        ///
        /// # Arguments
        /// * `model_path` - Path to the silero_vad.onnx model file
        /// * `threshold` - Speech probability threshold (0.0-1.0, default 0.5)
        /// * `sample_rate` - Audio sample rate (8000 or 16000)
        pub fn new(model_path: &Path, threshold: f32, sample_rate: u32) -> Result<Self> {
            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .commit_from_file(model_path)?;

            // Silero VAD requires specific chunk sizes
            let min_samples = if sample_rate == 8000 { 256 } else { 512 };

            // Initialize state: h and c tensors (2, 1, 64)
            let state = vec![0.0f32; 2 * 1 * 64];
            let sr_tensor = vec![sample_rate as i64];

            Ok(Self {
                session,
                threshold,
                sample_rate: sample_rate as i64,
                state,
                sr_tensor,
                buffer: Vec::with_capacity(min_samples * 2),
                min_samples,
            })
        }

        /// Download the Silero VAD model if not present
        pub fn ensure_model(model_dir: &Path) -> Result<std::path::PathBuf> {
            let model_path = model_dir.join("silero_vad.onnx");
            if model_path.exists() {
                return Ok(model_path);
            }

            std::fs::create_dir_all(model_dir)?;

            // Download from official Silero VAD releases
            let url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx";
            debug!("Downloading Silero VAD model from {}", url);

            let response = reqwest::blocking::get(url)?;
            let bytes = response.bytes()?;
            std::fs::write(&model_path, &bytes)?;

            debug!("Silero VAD model saved to {:?}", model_path);
            Ok(model_path)
        }

        /// Convert i16 samples to f32 normalized
        fn samples_to_f32(samples: &[i16]) -> Vec<f32> {
            samples.iter().map(|&s| s as f32 / 32768.0).collect()
        }
    }

    impl VoiceActivityDetector for SileroVadDetector {
        fn process(&mut self, samples: &[i16]) -> Result<bool> {
            // Convert and accumulate samples
            self.buffer.extend(Self::samples_to_f32(samples));

            // Process when we have enough samples
            if self.buffer.len() < self.min_samples {
                return Ok(false); // Not enough data yet
            }

            // Process in chunks of min_samples
            let mut speech_detected = false;
            while self.buffer.len() >= self.min_samples {
                let chunk: Vec<f32> = self.buffer.drain(..self.min_samples).collect();

                // Prepare input tensors as ort Values
                let input_array = ndarray::Array2::from_shape_vec(
                    (1, self.min_samples),
                    chunk
                )?;
                let input_value = Value::from_array(input_array)?;

                let state_array = ndarray::Array3::from_shape_vec(
                    (2, 1, 64),
                    self.state.clone()
                )?;
                let state_value = Value::from_array(state_array)?;

                let sr_array = ndarray::Array1::from_vec(self.sr_tensor.clone());
                let sr_value = Value::from_array(sr_array)?;

                // Run inference
                let inputs = ort::inputs![
                    "input" => input_value,
                    "state" => state_value,
                    "sr" => sr_value,
                ];

                match self.session.run(inputs) {
                    Ok(outputs) => {
                        // Extract probability from output - ort 2.0 returns (&Shape, &[T])
                        if let Ok((_shape, data)) = outputs["output"].try_extract_tensor::<f32>() {
                            let prob_val = data[0];
                            if prob_val > self.threshold {
                                speech_detected = true;
                                debug!("VAD: speech probability {:.3}", prob_val);
                            }
                        }

                        // Update state from output
                        if let Ok((_shape, data)) = outputs["stateN"].try_extract_tensor::<f32>() {
                            self.state = data.to_vec();
                        }
                    }
                    Err(e) => {
                        warn!("VAD inference error: {}", e);
                    }
                }
            }

            Ok(speech_detected)
        }

        fn reset(&mut self) {
            self.buffer.clear();
            self.state = vec![0.0f32; 2 * 1 * 64];
        }
    }
}

/// Create the appropriate VAD based on config
pub fn create_vad(
    vad_enabled: bool,
    vad_threshold: f32,
    silence_threshold_db: f32,
    sample_rate: u32,
) -> Box<dyn VoiceActivityDetector> {
    if vad_enabled {
        // Try to load Silero VAD
        let model_dir = dirs::data_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("voice-dictation")
            .join("models");

        match silero::SileroVadDetector::ensure_model(&model_dir) {
            Ok(model_path) => {
                match silero::SileroVadDetector::new(&model_path, vad_threshold, sample_rate) {
                    Ok(detector) => {
                        debug!("Using Silero VAD with threshold {}", vad_threshold);
                        return Box::new(detector);
                    }
                    Err(e) => {
                        tracing::warn!("Failed to create Silero VAD: {}, falling back to dB threshold", e);
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Failed to download Silero model: {}, falling back to dB threshold", e);
            }
        }
    }

    debug!("Using dB threshold VAD with threshold {} dB", silence_threshold_db);
    Box::new(DbThresholdVad::new(silence_threshold_db))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_db_threshold_vad_silence() {
        let mut vad = DbThresholdVad::new(-40.0);
        let silence = vec![0i16; 512];
        assert!(!vad.process(&silence).unwrap());
    }

    #[test]
    fn test_db_threshold_vad_loud() {
        let mut vad = DbThresholdVad::new(-40.0);
        let loud: Vec<i16> = (0..512).map(|i| ((i % 100) * 300) as i16).collect();
        assert!(vad.process(&loud).unwrap());
    }

    #[test]
    fn test_db_threshold_vad_reset() {
        let mut vad = DbThresholdVad::new(-40.0);
        vad.reset(); // Should not panic
    }

    #[test]
    fn test_create_vad_returns_db_threshold() {
        // Without silero-vad feature, should always return DbThresholdVad
        let mut vad = create_vad(true, 0.5, -40.0, 16000);

        // Test that it works like DbThresholdVad
        let silence = vec![0i16; 512];
        assert!(!vad.process(&silence).unwrap());
    }

    #[test]
    fn test_create_vad_disabled() {
        let mut vad = create_vad(false, 0.5, -40.0, 16000);

        // Should still work (uses dB threshold)
        let silence = vec![0i16; 512];
        assert!(!vad.process(&silence).unwrap());
    }

    #[test]
    fn test_rms_calculation() {
        // Test with known values
        let samples = vec![100i16; 100];
        let rms = DbThresholdVad::calculate_rms(&samples);
        assert!((rms - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_rms_to_db() {
        // Full scale should be 0 dB
        let db = DbThresholdVad::rms_to_db(32768.0);
        assert!((db - 0.0).abs() < 0.1);

        // Silence should be very negative
        let db = DbThresholdVad::rms_to_db(0.0);
        assert!(db < -90.0);
    }
}
