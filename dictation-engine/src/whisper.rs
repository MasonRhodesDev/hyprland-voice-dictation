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
            threads: 4,
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
                    .arg("-nt")
                    .arg("-np")
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
    for line in output.lines() {
        if line.contains("-->") {
            if let Some(text_start) = line.find(']') {
                let text = line[text_start + 1..].trim();
                if !text.is_empty() {
                    return Ok(text.to_string());
                }
            }
        }
    }

    anyhow::bail!("No transcription found in whisper output");
}
