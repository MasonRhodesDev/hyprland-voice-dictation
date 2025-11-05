use anyhow::Result;

/// Trait for speech-to-text transcription engines.
///
/// Provides a unified interface for different transcription backends
/// (Vosk, Whisper, etc.) to be used interchangeably in the dictation system.
///
/// Engines must be thread-safe (Send + Sync) as they are shared across
/// async tasks for preview updates and accurate transcription.
pub trait TranscriptionEngine: Send + Sync {
    /// Process incoming audio samples and add them to the internal buffer.
    ///
    /// Audio format: 16-bit signed integer PCM, mono, at the configured sample rate.
    ///
    /// # Arguments
    /// * `samples` - Audio samples to process
    ///
    /// # Returns
    /// * `Ok(())` if processing succeeded
    /// * `Err` if processing failed
    fn process_audio(&self, samples: &[i16]) -> Result<()>;

    /// Get the current transcription text (for live preview).
    ///
    /// This method is called frequently (every 200ms) to update the GUI
    /// with the current transcription state.
    ///
    /// # Returns
    /// * Current partial transcription text
    fn get_current_text(&self) -> Result<String>;

    /// Get the final transcription result from the preview model.
    ///
    /// Called when the user wants to finalize without running the
    /// accurate correction pass (fast finalization).
    ///
    /// # Returns
    /// * Final transcription from the preview/fast model
    fn get_final_result(&self) -> Result<String>;

    /// Get a copy of the accumulated audio buffer.
    ///
    /// Used by the accurate model to run a correction pass on the
    /// entire buffered audio after the user confirms.
    ///
    /// # Returns
    /// * Complete audio buffer accumulated during recording
    fn get_audio_buffer(&self) -> Vec<i16>;
}
