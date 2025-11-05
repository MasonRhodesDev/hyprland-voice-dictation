use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Stream, StreamConfig};
use serde::Deserialize;
use std::fs;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{error, info, warn};
use vosk::Model;

pub mod control_ipc;
pub mod ipc;
mod engine;
mod keyboard;
mod model_manager;
mod post_processing;
mod vosk_engine;
mod whisper_engine;

use control_ipc::{ControlMessage, ControlServer};
use engine::TranscriptionEngine;
use keyboard::KeyboardInjector;
use post_processing::Pipeline;
use vosk_engine::VoskEngine;
use whisper_engine::WhisperEngine;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const CONTROL_SOCKET_PATH: &str = "/tmp/voice-dictation-control.sock";
const AUDIO_SOCKET_PATH: &str = "/tmp/voice-dictation.sock";

#[derive(Debug, Deserialize)]
struct Config {
    daemon: DaemonConfig,
}

#[derive(Debug, Deserialize)]
struct DaemonConfig {
    audio_device: String,
    sample_rate: String,
    #[serde(default = "default_language")]
    language: String,

    // Engine selection
    #[serde(default = "default_transcription_engine")]
    transcription_engine: String,

    // Vosk models
    #[serde(default = "default_preview_model")]
    preview_model: String,
    #[serde(default = "default_preview_model_custom_path")]
    preview_model_custom_path: String,
    #[serde(default = "default_final_model")]
    final_model: String,
    #[serde(default = "default_final_model_custom_path")]
    final_model_custom_path: String,

    // Whisper models
    #[serde(default = "default_whisper_preview_model")]
    whisper_preview_model: String,
    #[serde(default = "default_whisper_final_model")]
    whisper_final_model: String,
    #[serde(default = "default_whisper_model_path")]
    whisper_model_path: String,

    // Post-processing
    #[serde(default = "default_enable_acronyms")]
    enable_acronyms: bool,
    #[serde(default = "default_enable_punctuation")]
    enable_punctuation: bool,
    #[serde(default = "default_enable_grammar")]
    enable_grammar: bool,
}

fn default_language() -> String { "en".to_string() }
fn default_preview_model() -> String { "vosk-model-en-us-daanzu-20200905-lgraph".to_string() }
fn default_preview_model_custom_path() -> String { 
    std::env::var("HOME")
        .map(|h| format!("{}/.config/voice-dictation/models/", h))
        .unwrap_or_else(|_| "./models/".to_string())
}
fn default_final_model() -> String { "vosk-model-en-us-0.22".to_string() }
fn default_final_model_custom_path() -> String {
    std::env::var("HOME")
        .map(|h| format!("{}/.config/voice-dictation/models/", h))
        .unwrap_or_else(|_| "./models/".to_string())
}
fn default_transcription_engine() -> String { "whisper".to_string() }
fn default_whisper_preview_model() -> String { "ggml-base.en.bin".to_string() }
fn default_whisper_final_model() -> String { "ggml-small.en.bin".to_string() }
fn default_whisper_model_path() -> String {
    std::env::var("HOME")
        .map(|h| format!("{}/.config/voice-dictation/models/whisper/", h))
        .unwrap_or_else(|_| "./models/whisper/".to_string())
}
fn default_enable_acronyms() -> bool { true }
fn default_enable_punctuation() -> bool { true }
fn default_enable_grammar() -> bool { true }

fn load_config() -> Result<Config> {
    let home = std::env::var("HOME")?;
    let config_path = format!("{}/.config/voice-dictation/config.toml", home);
    
    let config_str = fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("Failed to read config file {}: {}", config_path, e))?;
    
    let config: Config = toml::from_str(&config_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
    
    Ok(config)
}

/// Runtime engine selection wrapper.
enum Engine {
    Vosk(Arc<VoskEngine>),
    Whisper(Arc<WhisperEngine>),
}

impl Engine {
    /// Get a reference to the underlying engine as a TranscriptionEngine trait object.
    fn as_trait(&self) -> &dyn TranscriptionEngine {
        match self {
            Engine::Vosk(e) => e.as_ref(),
            Engine::Whisper(e) => e.as_ref(),
        }
    }
}

/// Accurate model wrapper for correction pass.
enum AccurateModel {
    Vosk(Model),
    Whisper(WhisperContext),
}

struct AudioCapture {
    stream: Option<Stream>,
}

impl AudioCapture {
    fn new(tx: mpsc::UnboundedSender<Vec<i16>>, device_name: Option<&str>, sample_rate: u32) -> Result<Self> {
        let host = cpal::default_host();
        
        info!("Available audio input devices from cpal:");
        if let Ok(devices) = host.input_devices() {
            for device in devices {
                if let Ok(name) = device.name() {
                    info!("  - '{}'", name);
                }
            }
        }
        
        let device = if let Some(name) = device_name {
            info!("Searching for configured device: '{}'", name);
            if name == "default" {
                info!("Using default audio input device");
                host.default_input_device().ok_or_else(|| anyhow::anyhow!("No default input device"))?
            } else {
                info!("Searching for audio device: {}", name);
                let mut found_device = None;
                
                for device in host.input_devices()? {
                    if let Ok(device_name) = device.name() {
                        if device_name == name {
                            found_device = Some(device);
                            break;
                        }
                    }
                }
                
                found_device.ok_or_else(|| {
                    warn!("Configured device '{}' not found, falling back to default", name);
                    anyhow::anyhow!("Audio device '{}' not found", name)
                }).or_else(|_| {
                    host.default_input_device().ok_or_else(|| anyhow::anyhow!("No input device available"))
                })?
            }
        } else {
            info!("No device configured, using default");
            host.default_input_device().ok_or_else(|| anyhow::anyhow!("No default input device"))?
        };

        info!("Using input device: {}", device.name()?);

        let config = StreamConfig {
            channels: 1,
            sample_rate: cpal::SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let stream = device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let samples: Vec<i16> =
                    data.iter().map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16).collect();
                let _ = tx.send(samples);
            },
            |err| error!("Audio stream error: {}", err),
            None,
        )?;

        Ok(Self { stream: Some(stream) })
    }

    fn start(&self) -> Result<()> {
        if let Some(stream) = &self.stream {
            stream.play()?;
            info!("Audio capture started");
        }
        Ok(())
    }
}

#[tokio::main]
pub async fn run() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    info!("Starting preview-based Vosk dictation engine");

    let config = load_config().unwrap_or_else(|e| {
        warn!("Failed to load config: {}, using defaults", e);
        Config {
            daemon: DaemonConfig {
                audio_device: "default".to_string(),
                sample_rate: "16000".to_string(),
                language: default_language(),
                transcription_engine: default_transcription_engine(),
                preview_model: default_preview_model(),
                preview_model_custom_path: default_preview_model_custom_path(),
                final_model: default_final_model(),
                final_model_custom_path: default_final_model_custom_path(),
                whisper_preview_model: default_whisper_preview_model(),
                whisper_final_model: default_whisper_final_model(),
                whisper_model_path: default_whisper_model_path(),
                enable_acronyms: default_enable_acronyms(),
                enable_punctuation: default_enable_punctuation(),
                enable_grammar: default_enable_grammar(),
            }
        }
    });

    let sample_rate: u32 = config.daemon.sample_rate.parse()
        .unwrap_or_else(|_| {
            warn!("Invalid sample_rate '{}', defaulting to 16000", config.daemon.sample_rate);
            16000
        });

    info!("Config loaded: audio_device={}, sample_rate={}, language={}", 
          config.daemon.audio_device, sample_rate, config.daemon.language);

    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    
    let preview_model_path = if config.daemon.preview_model == "custom" {
        shellexpand::full(&config.daemon.preview_model_custom_path)
            .map_err(|e| anyhow::anyhow!("Failed to expand preview model path: {}", e))?
            .to_string()
    } else {
        let base_path = format!("{}/.config/voice-dictation/models", home);
        format!("{}/{}", base_path, config.daemon.preview_model)
    };

    let final_model_path = if config.daemon.final_model == "custom" {
        shellexpand::full(&config.daemon.final_model_custom_path)
            .map_err(|e| anyhow::anyhow!("Failed to expand final model path: {}", e))?
            .to_string()
    } else {
        let base_path = format!("{}/.config/voice-dictation/models", home);
        format!("{}/{}", base_path, config.daemon.final_model)
    };

    let (audio_tx, mut audio_rx) = mpsc::unbounded_channel();

    let audio_device = if config.daemon.audio_device.is_empty() {
        None
    } else {
        let device_str = config.daemon.audio_device.as_str();
        let device_name = if let Some(idx) = device_str.find(" (") {
            &device_str[..idx]
        } else {
            device_str
        };
        Some(device_name)
    };

    let capture = AudioCapture::new(audio_tx, audio_device, sample_rate)?;
    capture.start()?;
    info!("Audio capture started - buffering...");

    info!("Loading fast model for live preview from: {}", preview_model_path);
    let engine = Arc::new(VoskEngine::new(&preview_model_path, sample_rate)?);
    let keyboard = Arc::new(KeyboardInjector::new(10, 50));

    // Load accurate model based on configuration
    let accurate_model_handle = {
        let engine_type = config.daemon.transcription_engine.clone();
        let vosk_final_path = final_model_path.clone();
        let whisper_model_name = config.daemon.whisper_final_model.clone();
        let whisper_model_dir = config.daemon.whisper_model_path.clone();

        tokio::spawn(async move {
            match engine_type.as_str() {
                "vosk" => {
                    info!("Preloading Vosk accurate model from: {}", vosk_final_path);
                    Model::new(&vosk_final_path).map(AccurateModel::Vosk)
                }
                "whisper" => {
                    info!("Ensuring Whisper model available: {}", whisper_model_name);

                    tokio::task::spawn_blocking(move || {
                        // Ensure model exists, download if necessary
                        let model_path = match model_manager::ensure_whisper_model(
                            &whisper_model_name,
                            &whisper_model_dir,
                        ) {
                            Ok(path) => path,
                            Err(e) => {
                                error!("Failed to obtain Whisper model: {}", e);
                                error!("Try running: ./scripts/download-whisper-models.sh");
                                return None;
                            }
                        };

                        info!("Loading Whisper model from: {}", model_path.display());

                        WhisperContext::new_with_params(
                            model_path.to_str().unwrap(),
                            WhisperContextParameters::default(),
                        )
                        .map(AccurateModel::Whisper)
                        .map_err(|e| {
                            error!("Whisper model load failed: {:?}", e);
                            e
                        })
                        .ok()
                    })
                    .await
                    .ok()
                    .flatten()
                }
                other => {
                    error!("Unknown transcription_engine '{}'. Valid: 'vosk' or 'whisper'", other);
                    None
                }
            }
        })
    };

    let audio_ipc = Arc::new(ipc::IpcServer::new(AUDIO_SOCKET_PATH.to_string()));
    audio_ipc.start_server();

    let mut control_server = ControlServer::new(CONTROL_SOCKET_PATH).await?;

    info!("Ready - waiting for GUI to connect");

    control_server.broadcast(&ControlMessage::Ready).await?;

    info!("Recording... (waiting for Confirm command)");

    let mut startup_buffer = Vec::new();
    while let Ok(samples) = audio_rx.try_recv() {
        startup_buffer.push(samples);
    }

    if !startup_buffer.is_empty() {
        info!("Processing {} buffered audio chunks from startup", startup_buffer.len());
        for samples in startup_buffer {
            if let Err(e) = engine.process_audio(&samples) {
                error!("Processing buffered audio error: {}", e);
            }
        }
    }

    let engine_clone = Arc::clone(&engine);
    let audio_ipc_clone = Arc::clone(&audio_ipc);
    let audio_task = tokio::spawn(async move {
        let mut buffer = Vec::new();

        while let Some(samples) = audio_rx.recv().await {
            let samples_f32: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();

            buffer.extend_from_slice(&samples_f32);

            while buffer.len() >= 512 {
                let chunk: Vec<f32> = buffer.drain(..512).collect();
                audio_ipc_clone.broadcast_samples(&chunk).await;
            }

            if let Err(e) = engine_clone.process_audio(&samples) {
                error!("Processing error: {}", e);
            }
        }
    });

    let engine_clone = Arc::clone(&engine);
    let control_server_shared = Arc::new(tokio::sync::Mutex::new(control_server));
    let control_clone_for_preview = Arc::clone(&control_server_shared);
    let enable_acronyms = config.daemon.enable_acronyms;
    let enable_punctuation = config.daemon.enable_punctuation;
    let enable_grammar = config.daemon.enable_grammar;
    let preview_task = tokio::spawn(async move {
        let mut check_interval = tokio::time::interval(std::time::Duration::from_millis(200));
        let pipeline = Pipeline::from_config(enable_acronyms, enable_punctuation, enable_grammar);

        loop {
            check_interval.tick().await;

            let mut server = control_clone_for_preview.lock().await;
            server.try_accept().await;
            drop(server);

            match engine_clone.get_current_text() {
                Ok(text_raw) => {
                    // Apply post-processing to preview text
                    let text_processed = match pipeline.process(&text_raw) {
                        Ok(processed) => processed,
                        Err(e) => {
                            error!("Preview post-processing error: {}", e);
                            text_raw.clone()
                        }
                    };

                    if !pipeline.is_empty() && text_raw != text_processed {
                        info!("[Preview] Raw: '{}'", text_raw);
                        info!("[Preview] Processed: '{}'", text_processed);
                    }

                    let mut server = control_clone_for_preview.lock().await;
                    let _ = server
                        .broadcast(&ControlMessage::TranscriptionUpdate {
                            text: text_processed,
                            is_final: false,
                        })
                        .await;
                }
                Err(e) => error!("Failed to get text: {}", e),
            }
        }
    });

    loop {
        let mut server = control_server_shared.lock().await;
        server.try_accept().await;

        if let Some(ControlMessage::Confirm) = server.receive_from_any().await {
            info!("Received Confirm command");
            drop(server);
            break;
        }
        drop(server);

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    }

    info!("Stopping recording...");

    audio_task.abort();
    preview_task.abort();

    let fast_result = engine.get_final_result()?;
    info!("Fast model result: '{}'", fast_result);

    let mut server = control_server_shared.lock().await;
    server
        .broadcast(&ControlMessage::TranscriptionUpdate {
            text: fast_result.clone(),
            is_final: true,
        })
        .await?;
    drop(server);

    if !fast_result.is_empty() {
        let mut server = control_server_shared.lock().await;
        server.broadcast(&ControlMessage::ProcessingStarted).await?;
        drop(server);

        info!("Waiting for accurate model to finish loading...");
        let accurate_model = accurate_model_handle
            .await?
            .ok_or_else(|| anyhow::anyhow!("Failed to load accurate model"))?;

        info!("Running correction pass...");
        let accurate_result = match &accurate_model {
            AccurateModel::Vosk(vosk_model) => {
                // Preview engine is always Vosk, call its correction method
                engine.run_correction_pass(vosk_model, sample_rate)?
            }
            AccurateModel::Whisper(whisper_context) => {
                // Get audio buffer from the preview engine (Vosk)
                let audio_buffer = engine.as_ref().get_audio_buffer();

                // Convert i16 → f32 for Whisper
                info!("Converting {} audio samples to float...", audio_buffer.len());
                let mut float_samples = vec![0.0f32; audio_buffer.len()];
                whisper_rs::convert_integer_to_float_audio(&audio_buffer, &mut float_samples)
                    .map_err(|e| anyhow::anyhow!("Audio conversion failed: {:?}", e))?;

                // Create transcription state
                let mut state = whisper_context
                    .create_state()
                    .map_err(|e| anyhow::anyhow!("Failed to create Whisper state: {:?}", e))?;

                // Configure parameters
                let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
                params.set_language(Some("en"));
                params.set_print_special(false);
                params.set_print_progress(false);
                params.set_print_realtime(false);
                params.set_print_timestamps(false);

                info!(
                    "Running Whisper transcription on {:.2}s of audio...",
                    float_samples.len() as f32 / sample_rate as f32
                );

                // Run transcription
                state
                    .full(params, &float_samples[..])
                    .map_err(|e| anyhow::anyhow!("Whisper transcription failed: {:?}", e))?;

                // Extract text from all segments using iterator
                let result: Vec<String> = state
                    .as_iter()
                    .filter_map(|segment| {
                        segment
                            .to_str_lossy()
                            .ok()
                            .map(|text| text.trim().to_string())
                    })
                    .filter(|text| !text.is_empty())
                    .collect();

                result.join(" ")
            }
        };
        info!("[Accurate] Raw: '{}'", accurate_result);

        // Apply post-processing pipeline
        let pipeline = Pipeline::from_config(
            config.daemon.enable_acronyms,
            config.daemon.enable_punctuation,
            config.daemon.enable_grammar,
        );
        let processed_result = pipeline.process(&accurate_result)?;

        if !pipeline.is_empty() && accurate_result != processed_result {
            info!("[Accurate] Processed: '{}'", processed_result);
        }

        info!("Typing final text...");
        keyboard.type_text(&processed_result).await?;
        info!("✓ Typed!");

        let mut server = control_server_shared.lock().await;
        server.broadcast(&ControlMessage::Complete).await?;
        drop(server);

        tokio::time::sleep(tokio::time::Duration::from_millis(350)).await;
    } else {
        info!("No text to type");

        let mut server = control_server_shared.lock().await;
        server.broadcast(&ControlMessage::Complete).await?;
        drop(server);

        tokio::time::sleep(tokio::time::Duration::from_millis(350)).await;
    }

    Ok(())
}
