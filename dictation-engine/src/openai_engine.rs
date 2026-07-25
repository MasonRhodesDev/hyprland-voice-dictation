//! Hosted OpenAI transcription engine (batch).
//!
//! Opt-in alternative to the local Parakeet engines, selected via the config
//! model spec `openai:<model>` (e.g. `openai:gpt-4o-transcribe`). Implements the
//! push-based [`StreamingEngine`] contract directly:
//!
//! - `process_audio` buffers samples (like the local path).
//! - No `Partial` events — batch transcription can't stream, so the preview
//!   overlay stays quiet while speaking.
//! - `finish()` uploads the full buffer as a WAV to the OpenAI transcription
//!   endpoint on the tokio runtime and emits a single `Final` (or `Error`).
//!
//! The API key is read from `OPENAI_API_KEY` at construction; if it is absent
//! the engine fails to construct and the selector stays on the default engine.
//! There is **no** cross-engine fallback: a request failure surfaces as
//! `TranscriptEvent::Error`.

use std::io::Cursor;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use tokio::sync::mpsc;
use tracing::{debug, error};

use crate::stream_engine::{EngineError, EventStream, StreamingEngine, TranscriptEvent};

const TRANSCRIPTIONS_URL: &str = "https://api.openai.com/v1/audio/transcriptions";

pub struct OpenAiEngine {
    api_key: String,
    model: String,
    sample_rate: u32,
    buffer: Arc<Mutex<Vec<i16>>>,
    event_tx: Arc<Mutex<Option<mpsc::UnboundedSender<TranscriptEvent>>>>,
    client: reqwest::Client,
}

impl OpenAiEngine {
    /// Construct the engine. Fails if `OPENAI_API_KEY` is unset/empty so the
    /// selector can fall back to the default engine and log the reason.
    pub fn new(model: String, sample_rate: u32) -> Result<Self> {
        let api_key =
            std::env::var("OPENAI_API_KEY").ok().filter(|k| !k.is_empty()).ok_or_else(|| {
                anyhow::anyhow!("OPENAI_API_KEY is not set; the openai engine cannot be used")
            })?;

        Ok(Self {
            api_key,
            model,
            sample_rate,
            buffer: Arc::new(Mutex::new(Vec::new())),
            event_tx: Arc::new(Mutex::new(None)),
            client: reqwest::Client::new(),
        })
    }
}

impl StreamingEngine for OpenAiEngine {
    fn process_audio(&self, samples: &[i16]) -> Result<()> {
        self.buffer
            .lock()
            .map_err(|e| anyhow::anyhow!("audio buffer lock poisoned: {e}"))?
            .extend_from_slice(samples);
        Ok(())
    }

    fn subscribe(&self) -> EventStream {
        let (tx, rx) = mpsc::unbounded_channel();
        if let Ok(mut slot) = self.event_tx.lock() {
            *slot = Some(tx);
        }
        rx
    }

    fn finish(&self) {
        let audio = self.buffer.lock().map(|b| b.clone()).unwrap_or_default();
        debug!("openai finish: {} samples @ {} Hz", audio.len(), self.sample_rate);
        let sample_rate = self.sample_rate;
        let model = self.model.clone();
        let api_key = self.api_key.clone();
        let client = self.client.clone();
        let event_tx = Arc::clone(&self.event_tx);

        // Runs on the tokio runtime (finish() is called from the async daemon loop).
        tokio::spawn(async move {
            let event = match transcribe(&client, &api_key, &model, &audio, sample_rate).await {
                Ok(text) => {
                    debug!("OpenAI transcription: '{}'", text);
                    TranscriptEvent::Final(text)
                }
                Err(e) => {
                    error!("OpenAI transcription failed: {}", e);
                    TranscriptEvent::Error(EngineError::Backend(e.to_string()))
                }
            };
            if let Ok(guard) = event_tx.lock() {
                if let Some(tx) = guard.as_ref() {
                    let _ = tx.send(event);
                }
            }
        });
    }

    fn reset(&self) {
        if let Ok(mut buf) = self.buffer.lock() {
            buf.clear();
        }
        if let Ok(mut slot) = self.event_tx.lock() {
            *slot = None;
        }
    }

    fn get_audio_buffer(&self) -> Vec<i16> {
        self.buffer.lock().map(|b| b.clone()).unwrap_or_default()
    }
}

/// POST the buffer as a WAV to the OpenAI transcription endpoint; return the text.
async fn transcribe(
    client: &reqwest::Client,
    api_key: &str,
    model: &str,
    samples: &[i16],
    sample_rate: u32,
) -> Result<String> {
    let wav = encode_wav(samples, sample_rate)?;
    debug!(
        "openai upload: {} wav bytes ({} samples @ {} Hz)",
        wav.len(),
        samples.len(),
        sample_rate
    );

    let part = reqwest::multipart::Part::bytes(wav).file_name("audio.wav").mime_str("audio/wav")?;
    let form = reqwest::multipart::Form::new().text("model", model.to_string()).part("file", part);

    let resp = client.post(TRANSCRIPTIONS_URL).bearer_auth(api_key).multipart(form).send().await?;

    let status = resp.status();
    let body = resp.text().await?;
    if !status.is_success() {
        anyhow::bail!("HTTP {}: {}", status, body);
    }

    // Default response_format is JSON: { "text": "..." }.
    let parsed: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| anyhow::anyhow!("invalid transcription response: {e}: {body}"))?;
    let text = parsed.get("text").and_then(|t| t.as_str()).unwrap_or_default();
    Ok(text.to_string())
}

/// Encode i16 mono PCM as a 16-bit WAV in memory.
fn encode_wav(samples: &[i16], sample_rate: u32) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::<u8>::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
        for &s in samples {
            writer.write_sample(s)?;
        }
        writer.finalize()?;
    }
    Ok(cursor.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_wav_has_riff_header_and_data() {
        let wav = encode_wav(&[0, 1, -1, 100, -100], 16000).unwrap();
        // RIFF/WAVE header + fmt + data chunk (44-byte header + 5*2 bytes).
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(wav.len(), 44 + 5 * 2);
    }

    #[test]
    fn new_fails_without_key() {
        // Only meaningful when the key is absent in the test environment.
        if std::env::var("OPENAI_API_KEY").map(|k| k.is_empty()).unwrap_or(true) {
            assert!(OpenAiEngine::new("gpt-4o-transcribe".into(), 16000).is_err());
        }
    }
}
