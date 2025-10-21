use anyhow::{Context, Result};
use hound::{WavSpec, WavWriter};
use std::path::PathBuf;
use std::process::Command;
use tempfile::NamedTempFile;
use tracing::{debug, info};

pub struct WhisperTranscriber {
    binary_path: PathBuf,
    model_path: PathBuf,
    language: String,
    threads: usize,
}

impl WhisperTranscriber {
    pub fn new(binary_path: String, model_path: String, language: String) -> Result<Self> {
        let binary_path = PathBuf::from(shellexpand::tilde(&binary_path).to_string());
        let model_path = PathBuf::from(shellexpand::tilde(&model_path).to_string());

        if !binary_path.exists() {
            anyhow::bail!("Whisper binary not found: {:?}", binary_path);
        }

        if !model_path.exists() {
            anyhow::bail!("Whisper model not found: {:?}", model_path);
        }

        info!("Whisper configured: binary={:?}, model={:?}", binary_path, model_path);

        Ok(Self {
            binary_path,
            model_path,
            language,
            threads: 6,
        })
    }

    pub async fn transcribe(&self, audio_samples: &[f32], sample_rate: u32) -> Result<String> {
        let temp_file = NamedTempFile::new()?;
        let wav_path = temp_file.path();
        
        debug!("Writing {} samples to temporary WAV file", audio_samples.len());
        write_wav_file(wav_path, audio_samples, sample_rate)?;

        let output = tokio::task::spawn_blocking({
            let binary_path = self.binary_path.clone();
            let model_path = self.model_path.clone();
            let language = self.language.clone();
            let threads = self.threads;
            let wav_path = wav_path.to_path_buf();

            move || {
                Command::new(&binary_path)
                    .arg("-m")
                    .arg(&model_path)
                    .arg("-f")
                    .arg(&wav_path)
                    .arg("-t")
                    .arg(threads.to_string())
                    .arg("-l")
                    .arg(&language)
                    .arg("-nt") // no timestamps
                    .arg("-np") // no progress
                    .arg("-bs") // beam size
                    .arg("1")   // beam size 1 for speed
                    .arg("-bo") // best of
                    .arg("1")   // best of 1 for speed
                    .output()
            }
        })
        .await??;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Whisper transcription failed: {}", stderr);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let transcription = parse_whisper_output(&stdout)?;

        debug!("Transcribed: {}", transcription);
        Ok(transcription)
    }
}

fn write_wav_file(path: &std::path::Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec)?;

    for &sample in samples {
        let sample_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(sample_i16)?;
    }

    writer.finalize()?;
    Ok(())
}

fn parse_whisper_output(output: &str) -> Result<String> {
    // With -nt -np flags, whisper outputs plain text
    let text = output.trim();
    
    if text.is_empty() {
        anyhow::bail!("Empty transcription from whisper");
    }
    
    // Remove special tokens like [BLANK_AUDIO], [silence], etc.
    let cleaned = text
        .replace("[BLANK_AUDIO]", "")
        .replace("[silence]", "")
        .replace("[SILENCE]", "")
        .trim()
        .to_string();
    
    if cleaned.is_empty() {
        anyhow::bail!("Only special tokens in transcription");
    }
    
    Ok(cleaned)
}
