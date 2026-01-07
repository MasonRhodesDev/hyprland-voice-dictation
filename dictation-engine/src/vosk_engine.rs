use super::engine::TranscriptionEngine;
use anyhow::Result;
use std::sync::{Arc, Mutex};
use tracing::info;
use vosk::{Model, Recognizer};

/// Vosk-based speech-to-text transcription engine.
///
/// Uses Vosk models for fast, offline transcription. Suitable for
/// real-time preview and accurate correction passes.
pub struct VoskEngine {
    recognizer: Arc<Mutex<Recognizer>>,
    accumulated_text: Arc<Mutex<String>>,
    audio_buffer: Arc<Mutex<Vec<i16>>>,
}

/// Remove duplicate suffix from accumulated text when adding new chunk.
///
/// Vosk's internal buffering can cause the same words to appear at the
/// end of one chunk and the beginning of the next. This function detects
/// and removes such overlaps.
///
/// # Example
/// ```ignore
/// let result = remove_duplicate_suffix("one two three", "three four five");
/// assert_eq!(result, "four five");
/// ```
pub fn remove_duplicate_suffix(accumulated: &str, new_chunk: &str) -> String {
    let acc_words: Vec<&str> = accumulated.split_whitespace().collect();
    let new_words: Vec<&str> = new_chunk.split_whitespace().collect();

    if acc_words.is_empty() || new_words.is_empty() {
        return new_chunk.to_string();
    }

    for overlap_len in (1..=acc_words.len().min(new_words.len())).rev() {
        let acc_suffix = &acc_words[acc_words.len() - overlap_len..];
        let new_prefix = &new_words[..overlap_len];

        if acc_suffix == new_prefix {
            return new_words[overlap_len..].join(" ");
        }
    }

    new_chunk.to_string()
}

impl VoskEngine {
    /// Create a new Vosk transcription engine.
    ///
    /// # Arguments
    /// * `model_path` - Path to the Vosk model directory
    /// * `sample_rate` - Audio sample rate in Hz
    pub fn new(model_path: &str, sample_rate: u32) -> Result<Self> {
        info!("Loading Vosk model from {}", model_path);
        let model =
            Model::new(model_path).ok_or_else(|| anyhow::anyhow!("Failed to load model"))?;
        let mut recognizer = Recognizer::new(&model, sample_rate as f32)
            .ok_or_else(|| anyhow::anyhow!("Failed to create recognizer"))?;

        let silence = vec![0i16; sample_rate as usize / 10];
        let _ = recognizer.accept_waveform(&silence);

        Ok(Self {
            recognizer: Arc::new(Mutex::new(recognizer)),
            accumulated_text: Arc::new(Mutex::new(String::new())),
            audio_buffer: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Internal audio processing implementation.
    fn process_audio_internal(&self, samples: &[i16]) -> Result<()> {
        let mut audio_buffer = self.audio_buffer.lock()
            .map_err(|e| anyhow::anyhow!("Audio buffer lock poisoned: {}", e))?;
        audio_buffer.extend_from_slice(samples);
        drop(audio_buffer);

        let mut recognizer = self.recognizer.lock()
            .map_err(|e| anyhow::anyhow!("Recognizer lock poisoned: {}", e))?;
        let state = recognizer.accept_waveform(samples)?;

        if state == vosk::DecodingState::Finalized {
            let result = recognizer.result();
            if let Some(finalized) = result.single() {
                let text = finalized.text.to_string().trim().to_string();
                if !text.is_empty() {
                    let mut accumulated = self.accumulated_text.lock()
                        .map_err(|e| anyhow::anyhow!("Accumulated text lock poisoned: {}", e))?;

                    let deduplicated = remove_duplicate_suffix(&accumulated, &text);

                    if !deduplicated.is_empty() {
                        if !accumulated.is_empty() {
                            accumulated.push(' ');
                        }
                        accumulated.push_str(&deduplicated);
                        info!("Accumulated chunk: '{}'", deduplicated);
                    }
                }
            }
        }

        Ok(())
    }

    /// Run a correction pass using an accurate Vosk model.
    ///
    /// Processes the entire accumulated audio buffer with a larger/more
    /// accurate model to produce the final transcription.
    ///
    /// # Arguments
    /// * `accurate_model` - The accurate Vosk model to use
    /// * `sample_rate` - Audio sample rate in Hz
    pub fn run_correction_pass(&self, accurate_model: &Model, sample_rate: u32) -> Result<String> {
        info!("Running correction pass with accurate Vosk model...");

        let mut accurate_recognizer = Recognizer::new(accurate_model, sample_rate as f32)
            .ok_or_else(|| anyhow::anyhow!("Failed to create accurate recognizer"))?;

        let audio_buffer = self.audio_buffer.lock()
            .map_err(|e| anyhow::anyhow!("Audio buffer lock poisoned: {}", e))?;

        const CHUNK_SIZE: usize = 8000;
        for chunk in audio_buffer.chunks(CHUNK_SIZE) {
            accurate_recognizer.accept_waveform(chunk)?;
        }

        let result = accurate_recognizer.final_result();
        if let Some(text) = result.single().map(|r| r.text.to_string()) {
            Ok(text.trim().to_string())
        } else {
            Ok(String::new())
        }
    }

    /// Get the current full text including partial results.
    fn get_current_full_text(&self) -> Result<String> {
        let mut recognizer = self.recognizer.lock()
            .map_err(|e| anyhow::anyhow!("Recognizer lock poisoned: {}", e))?;
        let accumulated = self.accumulated_text.lock()
            .map_err(|e| anyhow::anyhow!("Accumulated text lock poisoned: {}", e))?;

        let partial_result = recognizer.partial_result();
        let partial = partial_result.partial.to_string().trim().to_string();

        if partial.is_empty() {
            Ok(accumulated.clone())
        } else if accumulated.is_empty() {
            Ok(partial)
        } else {
            Ok(format!("{} {}", accumulated, partial))
        }
    }

    /// Get the final result from the preview model.
    fn get_final_result_internal(&self) -> Result<String> {
        let mut recognizer = self.recognizer.lock()
            .map_err(|e| anyhow::anyhow!("Recognizer lock poisoned: {}", e))?;
        let mut accumulated = self.accumulated_text.lock()
            .map_err(|e| anyhow::anyhow!("Accumulated text lock poisoned: {}", e))?;

        let result = recognizer.final_result();
        if let Some(final_chunk) = result.single() {
            let text = final_chunk.text.to_string().trim().to_string();
            if !text.is_empty() {
                if !accumulated.is_empty() {
                    accumulated.push(' ');
                }
                accumulated.push_str(&text);
            }
        }

        Ok(accumulated.clone())
    }
}

impl TranscriptionEngine for VoskEngine {
    fn process_audio(&self, samples: &[i16]) -> Result<()> {
        self.process_audio_internal(samples)
    }

    fn get_current_text(&self) -> Result<String> {
        self.get_current_full_text()
    }

    fn get_final_result(&self) -> Result<String> {
        self.get_final_result_internal()
    }

    fn get_audio_buffer(&self) -> Vec<i16> {
        self.audio_buffer.lock()
            .map(|guard| guard.clone())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remove_duplicate_suffix_no_overlap() {
        let result = remove_duplicate_suffix("one two", "three four");
        assert_eq!(result, "three four");
    }

    #[test]
    fn test_remove_duplicate_suffix_partial_overlap() {
        let result = remove_duplicate_suffix("one two three", "three four five");
        assert_eq!(result, "four five");
    }

    #[test]
    fn test_remove_duplicate_suffix_full_overlap() {
        let result = remove_duplicate_suffix("one two", "one two");
        assert_eq!(result, "");
    }

    #[test]
    fn test_remove_duplicate_suffix_empty_accumulated() {
        let result = remove_duplicate_suffix("", "one two three");
        assert_eq!(result, "one two three");
    }

    #[test]
    fn test_remove_duplicate_suffix_empty_new() {
        let result = remove_duplicate_suffix("one two three", "");
        assert_eq!(result, "");
    }

    #[test]
    fn test_remove_duplicate_suffix_longer_overlap() {
        let result = remove_duplicate_suffix("one two three four", "two three four five six");
        assert_eq!(result, "five six");
    }
}
