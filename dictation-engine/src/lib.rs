use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Stream, StreamConfig};
use serde::Deserialize;
use std::fs;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc};
use tracing::{debug, error, info, warn};
use vosk::Model;

pub mod control_ipc;
pub mod dbus_control;
mod engine;
mod keyboard;
mod model_manager;
mod post_processing;
mod vosk_engine;
mod whisper_engine;

pub use dictation_types::{GuiControl, GuiState, GuiStatus};

use dbus_control::DaemonCommand;
use engine::TranscriptionEngine;
use keyboard::KeyboardInjector;
use post_processing::Pipeline;
use vosk_engine::VoskEngine;
use whisper_engine::WhisperEngine;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

// Daemon state machine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DaemonState {
    Idle,        // Waiting for StartRecording command, GUI hidden
    Recording,   // Actively recording audio and transcribing, GUI visible
    Processing,  // Running accurate model and typing, GUI visible with spinner
}

impl std::fmt::Display for DaemonState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DaemonState::Idle => write!(f, "idle"),
            DaemonState::Recording => write!(f, "recording"),
            DaemonState::Processing => write!(f, "processing"),
        }
    }
}

// Recording session context
struct RecordingSession {
    start_time: Instant,
    engine: Arc<VoskEngine>,
}

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

    fn stop(&self) -> Result<()> {
        if let Some(stream) = &self.stream {
            stream.pause()?;
            info!("Audio capture stopped");
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

    let (audio_tx, audio_rx) = mpsc::unbounded_channel();
    let audio_rx_shared = Arc::new(tokio::sync::Mutex::new(audio_rx));

    // Create GUI channels for integrated communication
    let (gui_control_tx, _) = broadcast::channel::<GuiControl>(100);
    let (spectrum_tx, _) = broadcast::channel::<Vec<f32>>(50);
    let (gui_status_tx, mut gui_status_rx) = mpsc::channel::<GuiStatus>(10);

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
    // Don't start audio capture yet - will be started when StartRecording received
    info!("Audio capture initialized (paused)");

    info!("Loading fast model for live preview from: {}", preview_model_path);
    let engine = Arc::new(VoskEngine::new(&preview_model_path, sample_rate)?);
    let keyboard = Arc::new(KeyboardInjector::new(10, 50));

    // Spawn integrated GUI
    info!("Spawning integrated GUI...");
    let gui_control_rx = gui_control_tx.subscribe();
    let spectrum_rx = spectrum_tx.subscribe();
    let _gui_handle = tokio::task::spawn_blocking(move || {
        dictation_gui::run_integrated(
            gui_control_rx,
            spectrum_rx,
            gui_status_tx,
        )
    });

    // Wait for GUI to initialize
    info!("Waiting for GUI to initialize...");
    match tokio::time::timeout(
        Duration::from_secs(5),
        gui_status_rx.recv()
    ).await {
        Ok(Some(GuiStatus::Ready)) => info!("GUI ready"),
        Ok(Some(GuiStatus::Error(e))) => {
            return Err(anyhow::anyhow!("GUI initialization failed: {}", e));
        }
        Ok(Some(GuiStatus::TransitionComplete { .. })) => {
            warn!("Unexpected TransitionComplete during init, continuing");
        }
        Ok(Some(GuiStatus::ShuttingDown)) => {
            return Err(anyhow::anyhow!("GUI is shutting down during init"));
        }
        Ok(None) => {
            return Err(anyhow::anyhow!("GUI status channel closed"));
        }
        Err(_) => {
            return Err(anyhow::anyhow!("GUI failed to start within 5 seconds"));
        }
    }

    // Load accurate model based on configuration (eagerly)
    info!("Loading accurate model in background...");
    let accurate_model_opt = {
        let engine_type = config.daemon.transcription_engine.clone();
        let vosk_final_path = final_model_path.clone();
        let whisper_model_name = config.daemon.whisper_final_model.clone();
        let whisper_model_dir = config.daemon.whisper_model_path.clone();

        tokio::task::spawn_blocking(move || {
            match engine_type.as_str() {
                "vosk" => {
                    info!("Loading Vosk accurate model from: {}", vosk_final_path);
                    Model::new(&vosk_final_path).map(AccurateModel::Vosk)
                }
                "whisper" => {
                    info!("Ensuring Whisper model available: {}", whisper_model_name);

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
                }
                other => {
                    error!("Unknown transcription_engine '{}'. Valid: 'vosk' or 'whisper'", other);
                    None
                }
            }
        }).await.ok().flatten()
    };

    let accurate_model = Arc::new(accurate_model_opt);

    // Create D-Bus service for control commands
    // IMPORTANT: Must keep connection alive for D-Bus service to remain registered
    let (dbus_conn, _command_sender, mut command_rx) = dbus_control::create_dbus_service().await?;
    let _dbus_conn = dbus_conn; // Keep alive but mark unused

    info!("Daemon initialized - entering idle state (GUI hidden)");

    // State machine variables
    let mut daemon_state = DaemonState::Idle;
    let mut session: Option<RecordingSession> = None;
    let mut audio_task: Option<tokio::task::JoinHandle<()>> = None;
    let mut preview_task: Option<tokio::task::JoinHandle<()>> = None;

    // ===== PERSISTENT STATE MACHINE LOOP =====
    loop {
        match daemon_state {
            DaemonState::Idle => {
                // Wait for D-Bus commands with timeout
                match tokio::time::timeout(Duration::from_millis(100), command_rx.recv()).await {
                    Ok(Some(cmd)) => match cmd {
                        DaemonCommand::StartRecording => {
                            info!("Received StartRecording command");

                            // Drain any stale audio samples from previous session
                            {
                                let mut rx = audio_rx_shared.lock().await;
                                while rx.try_recv().is_ok() {
                                    // Discard stale samples
                                }
                                info!("Drained audio channel before new session");
                            }

                            // Start new recording session
                            info!("Starting audio capture...");
                            capture.start()?;

                            // Create new engine for new session (Vosk doesn't have reset)
                            let session_engine = Arc::new(VoskEngine::new(&preview_model_path, sample_rate)?);

                            // Show GUI
                            gui_control_tx.send(GuiControl::SetListening)
                                .map_err(|e| anyhow::anyhow!("Failed to send SetListening: {}", e))?;

                            // Create session
                            session = Some(RecordingSession {
                                start_time: Instant::now(),
                                engine: Arc::clone(&session_engine),
                            });

                            // Start audio processing task
                            let engine_clone = Arc::clone(&session_engine);
                            let spectrum_tx_clone = spectrum_tx.clone();
                            let audio_rx_clone = Arc::clone(&audio_rx_shared);
                            audio_task = Some(tokio::spawn(async move {
                                let mut buffer = Vec::new();
                                loop {
                                    let samples = {
                                        let mut rx = audio_rx_clone.lock().await;
                                        rx.recv().await
                                    };

                                    match samples {
                                        Some(samples) => {
                                            let samples_f32: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
                                            buffer.extend_from_slice(&samples_f32);

                                            while buffer.len() >= 512 {
                                                let chunk: Vec<f32> = buffer.drain(..512).collect();
                                                let _ = spectrum_tx_clone.send(chunk);
                                            }

                                            if let Err(e) = engine_clone.process_audio(&samples) {
                                                error!("Processing error: {}", e);
                                            }
                                        }
                                        None => break,
                                    }
                                }
                            }));

                            // Start preview task
                            let engine_clone = Arc::clone(&session_engine);
                            let gui_control_tx_preview = gui_control_tx.clone();
                            let enable_acronyms = config.daemon.enable_acronyms;
                            let enable_punctuation = config.daemon.enable_punctuation;
                            let enable_grammar = config.daemon.enable_grammar;
                            preview_task = Some(tokio::spawn(async move {
                                let mut check_interval = tokio::time::interval(std::time::Duration::from_millis(200));
                                let pipeline = Pipeline::from_config(enable_acronyms, enable_punctuation, enable_grammar);

                                loop {
                                    check_interval.tick().await;

                                    match engine_clone.get_current_text() {
                                        Ok(text_raw) => {
                                            let text_processed = match pipeline.process(&text_raw) {
                                                Ok(processed) => processed,
                                                Err(e) => {
                                                    error!("Preview post-processing error: {}", e);
                                                    text_raw.clone()
                                                }
                                            };

                                            if !pipeline.is_empty() && text_raw != text_processed {
                                                debug!("[Preview] Raw: '{}' -> Processed: '{}'", text_raw, text_processed);
                                            }

                                            let _ = gui_control_tx_preview.send(GuiControl::UpdateTranscription {
                                                text: text_processed,
                                                is_final: false,
                                            });
                                        }
                                        Err(e) => error!("Failed to get text: {}", e),
                                    }
                                }
                            }));

                            daemon_state = DaemonState::Recording;
                            info!("Entered Recording state");
                        }
                        DaemonCommand::Shutdown => {
                            info!("Received Shutdown command");
                            // Send GUI exit
                            let _ = gui_control_tx.send(GuiControl::Exit);
                            break;
                        }
                        _ => {
                            warn!("Ignoring unexpected command in Idle state");
                        }
                    }
                    Ok(None) => {
                        // Channel closed
                        error!("D-Bus command channel closed");
                        break;
                    }
                    Err(_) => {
                        // Timeout - continue loop
                    }
                }
            }

            DaemonState::Recording => {
                // Check for D-Bus commands while recording (non-blocking)
                match tokio::time::timeout(Duration::from_millis(100), command_rx.recv()).await {
                    Ok(Some(cmd)) => match cmd {
                        DaemonCommand::Confirm => {
                            info!("Received Confirm command");
                            daemon_state = DaemonState::Processing;
                        }
                        DaemonCommand::StopRecording => {
                            info!("Received StopRecording (cancel)");

                            // Abort tasks
                            if let Some(task) = audio_task.take() {
                                task.abort();
                            }
                            if let Some(task) = preview_task.take() {
                                task.abort();
                            }

                            // Hide GUI
                            let _ = gui_control_tx.send(GuiControl::SetHidden);

                            session = None;
                            daemon_state = DaemonState::Idle;
                            info!("Returned to Idle state");
                        }
                        DaemonCommand::Shutdown => {
                            info!("Shutdown during recording");
                            // Abort tasks
                            if let Some(task) = audio_task.take() {
                                task.abort();
                            }
                            if let Some(task) = preview_task.take() {
                                task.abort();
                            }
                            let _ = gui_control_tx.send(GuiControl::Exit);
                            break;
                        }
                        _ => {
                            warn!("Ignoring unexpected command in Recording state");
                        }
                    }
                    Ok(None) => {
                        error!("D-Bus command channel closed");
                        break;
                    }
                    Err(_) => {
                        // Timeout - continue recording
                    }
                }
            }

            DaemonState::Processing => {
                info!("Entering Processing state");

                // Stop recording tasks
                if let Some(task) = audio_task.take() {
                    task.abort();
                }
                if let Some(task) = preview_task.take() {
                    task.abort();
                }

                // Get engine from session
                let session_engine = session.as_ref()
                    .ok_or_else(|| anyhow::anyhow!("No active session in Processing state"))?
                    .engine.clone();

                let fast_result = session_engine.get_final_result()?;
                info!("Fast model result: '{}'", fast_result);

                // Check if any audio was captured (use buffer length instead of text check)
                let audio_buffer_len = session_engine.as_ref().get_audio_buffer().len();
                info!("Audio buffer contains {} samples", audio_buffer_len);

                if audio_buffer_len > 0 {
                    // Send processing state to GUI
                    gui_control_tx.send(GuiControl::SetProcessing)
                        .map_err(|e| anyhow::anyhow!("Failed to send SetProcessing: {}", e))?;

                    // Check if accurate model is loaded
                    let model_ref = accurate_model.as_ref()
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("Accurate model not loaded"))?;

                    info!("Running correction pass...");
                    let accurate_result = match model_ref {
                        AccurateModel::Vosk(vosk_model) => {
                            session_engine.run_correction_pass(vosk_model, sample_rate)?
                        }
                        AccurateModel::Whisper(whisper_context) => {
                            let audio_buffer = session_engine.as_ref().get_audio_buffer();
                            info!("Converting {} audio samples to float...", audio_buffer.len());
                            let mut float_samples = vec![0.0f32; audio_buffer.len()];
                            whisper_rs::convert_integer_to_float_audio(&audio_buffer, &mut float_samples)
                                .map_err(|e| anyhow::anyhow!("Audio conversion failed: {:?}", e))?;

                            let mut state = whisper_context
                                .create_state()
                                .map_err(|e| anyhow::anyhow!("Failed to create Whisper state: {:?}", e))?;

                            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
                            params.set_language(Some("en"));
                            params.set_print_special(false);
                            params.set_print_progress(false);
                            params.set_print_realtime(false);
                            params.set_print_timestamps(false);

                            info!("Running Whisper transcription on {:.2}s of audio...",
                                float_samples.len() as f32 / sample_rate as f32);

                            state.full(params, &float_samples[..])
                                .map_err(|e| anyhow::anyhow!("Whisper transcription failed: {:?}", e))?;

                            let result: Vec<String> = state
                                .as_iter()
                                .filter_map(|segment| {
                                    segment.to_str_lossy().ok().map(|text| text.trim().to_string())
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

                    // Send to GUI via channel
                    gui_control_tx.send(GuiControl::SetClosing)
                        .map_err(|e| anyhow::anyhow!("Failed to send SetClosing: {}", e))?;

                    tokio::time::sleep(tokio::time::Duration::from_millis(350)).await;
                } else {
                    info!("No text to type");
                    gui_control_tx.send(GuiControl::SetClosing)
                        .map_err(|e| anyhow::anyhow!("Failed to send SetClosing: {}", e))?;
                    tokio::time::sleep(tokio::time::Duration::from_millis(350)).await;
                }

                // Hide GUI and return to Idle
                gui_control_tx.send(GuiControl::SetHidden)
                    .map_err(|e| anyhow::anyhow!("Failed to send SetHidden: {}", e))?;

                // Stop audio capture to prevent sample accumulation
                capture.stop()?;

                session = None;
                daemon_state = DaemonState::Idle;
                info!("Processing complete - returned to Idle state");
            }
        }
    }

    info!("Daemon shutting down");
    Ok(())
}
