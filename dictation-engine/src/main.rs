use anyhow::Result;
use tokio::time::{sleep, Duration};
use tracing::{error, info, warn};

mod audio;
mod keyboard;
mod vad;
mod whisper;

use audio::AudioCapture;
use keyboard::KeyboardInjector;
use vad::{VadDetector, VadEvent};
use whisper::WhisperTranscriber;

const SAMPLE_RATE: u32 = 16000;
const VAD_FRAME_DURATION_MS: u64 = 30;
const VAD_THRESHOLD_DB: f32 = -40.0;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    info!("Starting dictation-engine");

    let mut audio = AudioCapture::new(SAMPLE_RATE, 1)?;
    audio.start()?;

    let mut vad = VadDetector::new(VAD_THRESHOLD_DB);

    let whisper = WhisperTranscriber::new(
        "~/.local/bin/whisper-cpp".to_string(),
        "~/repos/whisper.cpp/models/ggml-base.en.bin".to_string(),
        "en".to_string(),
    )?;

    let keyboard = KeyboardInjector::new(10, 50);

    let vad_frame_samples = (SAMPLE_RATE as u64 * VAD_FRAME_DURATION_MS / 1000) as usize;
    
    let mut speech_start_time = None;

    info!("Listening for speech...");

    loop {
        sleep(Duration::from_millis(VAD_FRAME_DURATION_MS)).await;

        let samples = audio.get_latest_samples(vad_frame_samples);
        if samples.len() < vad_frame_samples {
            continue;
        }

        // Debug: check for NaN values
        if samples.iter().any(|s| s.is_nan() || s.is_infinite()) {
            warn!("Invalid audio samples detected (NaN/Inf), skipping frame");
            continue;
        }

        let event = vad.process_frame(&samples);

        // Periodically log VAD status
        static mut FRAME_COUNT: usize = 0;
        unsafe {
            FRAME_COUNT += 1;
            if FRAME_COUNT % 100 == 0 {
                let rms: f32 = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
                let rms = rms.sqrt();
                let db = if rms > 0.0 { 20.0 * rms.log10() } else { -100.0 };
                info!("VAD status: speaking={}, rms={:.6}, db={:.1}dB", vad.is_speaking(), rms, db);
            }
        }

        match event {
            VadEvent::SpeechStart => {
                info!("Speech detected, recording...");
                speech_start_time = Some(std::time::Instant::now());
            }
            VadEvent::SpeechEnd => {
                if let Some(start_time) = speech_start_time.take() {
                    let duration = start_time.elapsed();
                    info!("Speech ended after {:.1}s, transcribing...", duration.as_secs_f32());

                    let duration_ms = duration.as_millis() as u64 + 500;
                    let speech_samples = audio.get_samples_for_duration(duration_ms);

                    if speech_samples.is_empty() {
                        warn!("No audio samples captured");
                        continue;
                    }

                    match whisper.transcribe(&speech_samples, SAMPLE_RATE).await {
                        Ok(text) => {
                            info!("Transcription: {}", text);
                            if let Err(e) = keyboard.type_text(&text).await {
                                error!("Failed to type text: {}", e);
                            }
                        }
                        Err(e) => {
                            error!("Transcription failed: {}", e);
                        }
                    }

                    info!("Listening for speech...");
                }
            }
            VadEvent::None => {}
        }
    }
}
