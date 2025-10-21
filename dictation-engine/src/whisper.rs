// Whisper.cpp subprocess interface

use anyhow::Result;
use std::process::Command;

pub struct WhisperTranscriber {
    binary_path: String,
    model_path: String,
    language: String,
    threads: usize,
}

impl WhisperTranscriber {
    pub fn new(binary_path: String, model_path: String, language: String) -> Self {
        Self {
            binary_path,
            model_path,
            language,
            threads: 4,
        }
    }
    
    pub async fn transcribe(&self, audio_samples: &[f32], sample_rate: u32) -> Result<String> {
        // TODO: Write audio to temporary WAV file
        // TODO: Spawn whisper.cpp subprocess
        // TODO: Parse JSON output
        // TODO: Cleanup temp file
        // TODO: Return transcribed text
        todo!()
    }
}

fn write_wav_file(path: &str, samples: &[f32], sample_rate: u32) -> Result<()> {
    // TODO: Use hound crate to write WAV
    todo!()
}
