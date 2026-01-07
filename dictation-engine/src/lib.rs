use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Stream, StreamConfig};
use notify::{Event, EventKind, RecursiveMode, Watcher};
use serde::Deserialize;
use std::fs;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc, RwLock};
use tracing::{debug, error, info, warn};
#[cfg(feature = "vosk")]
use vosk::Model;

pub mod control_ipc;
pub mod dbus_control;
mod engine;
mod keyboard;
mod model_manager;
mod post_processing;
pub mod user_dictionary;
#[cfg(feature = "vosk")]
mod vosk_engine;
mod whisper_engine;

pub use dictation_types::{GuiControl, GuiState, GuiStatus};

use dbus_control::DaemonCommand;
use engine::TranscriptionEngine;
use keyboard::KeyboardInjector;
use post_processing::Pipeline;
use user_dictionary::UserDictionary;
#[cfg(feature = "vosk")]
use vosk_engine::VoskEngine;
use whisper_engine::WhisperEngine;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

// Re-export DaemonState from dbus_control
use dbus_control::DaemonState;

// Recording session context
struct RecordingSession {
    #[allow(dead_code)]
    start_time: Instant,
    #[cfg(feature = "vosk")]
    engine: Arc<VoskEngine>,
    #[cfg(not(feature = "vosk"))]
    engine: Arc<WhisperEngine>,
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

    // Vosk models (only used when vosk feature is enabled)
    #[cfg_attr(not(feature = "vosk"), allow(dead_code))]
    #[serde(default = "default_preview_model")]
    preview_model: String,
    #[cfg_attr(not(feature = "vosk"), allow(dead_code))]
    #[serde(default = "default_preview_model_custom_path")]
    preview_model_custom_path: String,
    #[serde(default = "default_final_model")]
    final_model: String,
    #[serde(default = "default_final_model_custom_path")]
    final_model_custom_path: String,

    // Whisper models
    #[allow(dead_code)]
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

    // Audio capture
    #[serde(default = "default_silence_threshold_db")]
    silence_threshold_db: f32,
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
fn default_silence_threshold_db() -> f32 { -60.0 }

/// Convert decibels to linear amplitude (RMS threshold).
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

fn load_config() -> Result<Config> {
    let home = std::env::var("HOME")?;
    let config_path = format!("{}/.config/voice-dictation/config.toml", home);

    let config_str = fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("Failed to read config file {}: {}", config_path, e))?;

    let config: Config = toml::from_str(&config_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;

    Ok(config)
}

/// Watch dictionary files and reload on changes.
async fn watch_dictionary_files(user_dict: Arc<UserDictionary>) -> Result<()> {
    let paths = user_dict.watch_paths();

    if paths.is_empty() {
        info!("No dictionary files to watch");
        return Ok(());
    }

    info!("Watching dictionary files: {:?}", paths);

    let (tx, mut rx) = mpsc::channel(100);

    // Create watcher in a separate thread (notify requires blocking)
    let mut watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
        if let Ok(event) = res {
            let _ = tx.blocking_send(event);
        }
    })?;

    // Watch all dictionary paths
    for path in &paths {
        if path.exists() {
            watcher.watch(path, RecursiveMode::NonRecursive)?;
        } else {
            // Watch parent directory to detect file creation
            if let Some(parent) = path.parent() {
                if parent.exists() {
                    watcher.watch(parent, RecursiveMode::NonRecursive)?;
                }
            }
        }
    }

    // Keep watcher alive and process events
    while let Some(event) = rx.recv().await {
        match event.kind {
            EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_) => {
                // Check if event is for one of our dictionary files
                for path in &paths {
                    if event.paths.iter().any(|p| p == path) {
                        info!("Dictionary file changed: {:?}, reloading...", path);
                        if let Err(e) = user_dict.reload_all() {
                            warn!("Failed to reload dictionaries: {}", e);
                        } else {
                            info!("Dictionaries reloaded successfully");
                        }
                        break;
                    }
                }
            }
            _ => {}
        }
    }

    Ok(())
}

/// Runtime engine selection wrapper.
#[allow(dead_code)]
enum Engine {
    #[cfg(feature = "vosk")]
    Vosk(Arc<VoskEngine>),
    Whisper(Arc<WhisperEngine>),
}

#[allow(dead_code)]
impl Engine {
    /// Get a reference to the underlying engine as a TranscriptionEngine trait object.
    fn as_trait(&self) -> &dyn TranscriptionEngine {
        match self {
            #[cfg(feature = "vosk")]
            Engine::Vosk(e) => e.as_ref(),
            Engine::Whisper(e) => e.as_ref(),
        }
    }
}

/// Accurate model wrapper for correction pass.
enum AccurateModel {
    #[cfg(feature = "vosk")]
    Vosk(Model),
    Whisper(WhisperContext),
}

/// Configuration for lazy-loading the accurate model
#[derive(Clone)]
struct AccurateModelConfig {
    engine_type: String,
    #[cfg(feature = "vosk")]
    vosk_final_path: String,
    whisper_model_name: String,
    whisper_model_dir: String,
}

impl AccurateModelConfig {
    /// Load the accurate model (blocking operation, run in spawn_blocking)
    fn load(&self) -> Option<AccurateModel> {
        match self.engine_type.as_str() {
            #[cfg(feature = "vosk")]
            "vosk" => {
                info!("Loading Vosk accurate model from: {}", self.vosk_final_path);
                Model::new(&self.vosk_final_path).map(AccurateModel::Vosk)
            }
            #[cfg(not(feature = "vosk"))]
            "vosk" => {
                error!("Vosk support not compiled in. Set transcription_engine = \"whisper\" in config.");
                None
            }
            "whisper" => {
                info!("Ensuring Whisper model available: {}", self.whisper_model_name);

                // Ensure model exists, download if necessary
                let model_path = match model_manager::ensure_whisper_model(
                    &self.whisper_model_name,
                    &self.whisper_model_dir,
                ) {
                    Ok(path) => path,
                    Err(e) => {
                        error!("Failed to obtain Whisper model: {}", e);
                        error!("Try running: ./scripts/download-whisper-models.sh");
                        return None;
                    }
                };

                info!("Loading Whisper model from: {}", model_path.display());

                let model_path_str = match model_path.to_str() {
                    Some(s) => s,
                    None => {
                        error!("Whisper model path contains invalid UTF-8");
                        return None;
                    }
                };

                WhisperContext::new_with_params(
                    model_path_str,
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
    }
}

struct AudioCapture {
    streams: Vec<Stream>,
}

impl AudioCapture {
    /// Check if a device name indicates it's a real input device (not a monitor/loopback)
    fn is_real_input_device(name: &str) -> bool {
        let name_lower = name.to_lowercase();
        // Skip output monitors (loopback devices)
        if name_lower.contains(".monitor") || name_lower.contains("monitor") {
            return false;
        }
        // Skip HDMI outputs (usually no mic)
        if name_lower.contains("hdmi") {
            return false;
        }
        true
    }

    fn new(tx: mpsc::UnboundedSender<Vec<i16>>, device_name: Option<&str>, sample_rate: u32, silence_threshold: f32) -> Result<Self> {
        let host = cpal::default_host();

        info!("Available audio input devices from cpal:");
        let mut real_devices = Vec::new();
        if let Ok(devices) = host.input_devices() {
            for device in devices {
                if let Ok(name) = device.name() {
                    let is_real = Self::is_real_input_device(&name);
                    info!("  - '{}' {}", name, if is_real { "(will capture)" } else { "(skipped - monitor/hdmi)" });
                    if is_real {
                        real_devices.push(device);
                    }
                }
            }
        }

        // Determine which devices to use
        let devices_to_use: Vec<_> = match device_name {
            // "all" mode: capture from pipewire + all hardware devices
            // Silent chunks are filtered out, so only devices with real audio contribute
            Some("all") => {
                let mut devices = Vec::new();

                for device in real_devices {
                    if let Ok(name) = device.name() {
                        // Include pipewire (routes to default) and all hardware devices
                        if name == "pipewire" || name.starts_with("sysdefault:CARD=") {
                            devices.push(device);
                        }
                    }
                }

                if devices.is_empty() {
                    if let Some(default) = host.default_input_device() {
                        devices.push(default);
                    }
                }

                info!("Multi-device mode: capturing from {} device(s) (silent chunks filtered)", devices.len());
                devices
            }
            // "default" or None: use pipewire device for fastest route to PipeWire
            None | Some("default") => {
                let mut found = Vec::new();
                // Look for "pipewire" device first - it routes to PipeWire's default source
                for device in &real_devices {
                    if let Ok(name) = device.name() {
                        if name == "pipewire" {
                            info!("Using 'pipewire' device (routes to PipeWire default source)");
                            // Can't move from &real_devices, need to find it again
                            break;
                        }
                    }
                }
                // Re-search to actually take the device
                for device in real_devices {
                    if let Ok(name) = device.name() {
                        if name == "pipewire" {
                            found.push(device);
                            break;
                        }
                    }
                }
                if found.is_empty() {
                    info!("'pipewire' device not found, using system default");
                    if let Some(default) = host.default_input_device() {
                        found.push(default);
                    }
                }
                found
            }
            // Specific device requested
            Some(name) => {
                info!("Single-device mode: searching for '{}'", name);
                let mut found = Vec::new();
                for device in real_devices {
                    if let Ok(device_name) = device.name() {
                        if device_name == name {
                            found.push(device);
                            break;
                        }
                    }
                }
                if found.is_empty() {
                    warn!("Configured device '{}' not found, falling back to pipewire/default", name);
                    if let Some(default) = host.default_input_device() {
                        found.push(default);
                    }
                }
                found
            }
        };

        if devices_to_use.is_empty() {
            return Err(anyhow::anyhow!("No input devices available"));
        }

        let config = StreamConfig {
            channels: 1,
            sample_rate: cpal::SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let mut streams = Vec::new();
        for device in devices_to_use {
            let device_name = device.name().unwrap_or_else(|_| "unknown".to_string());
            let tx_clone = tx.clone();
            let threshold = silence_threshold;

            match device.build_input_stream(
                &config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    // Filter out silent chunks - only forward audio with meaningful amplitude
                    // This allows multi-device capture where silent devices don't corrupt the signal
                    let rms: f32 = (data.iter().map(|&s| s * s).sum::<f32>() / data.len() as f32).sqrt();
                    if rms < threshold {
                        return; // Skip silent chunks
                    }

                    let samples: Vec<i16> =
                        data.iter().map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16).collect();
                    let _ = tx_clone.send(samples);
                },
                |err| error!("Audio stream error: {}", err),
                None,
            ) {
                Ok(stream) => {
                    info!("Created audio stream for: {}", device_name);
                    streams.push(stream);
                }
                Err(e) => {
                    warn!("Failed to create stream for '{}': {}", device_name, e);
                }
            }
        }

        if streams.is_empty() {
            return Err(anyhow::anyhow!("Failed to create any audio streams"));
        }

        info!("Audio capture initialized with {} stream(s)", streams.len());
        Ok(Self { streams })
    }

    fn start(&self) -> Result<()> {
        for stream in &self.streams {
            stream.play()?;
        }
        info!("Audio capture started ({} streams)", self.streams.len());
        Ok(())
    }

    fn stop(&self) -> Result<()> {
        for stream in &self.streams {
            stream.pause()?;
        }
        info!("Audio capture stopped ({} streams)", self.streams.len());
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

    #[cfg(feature = "vosk")]
    info!("Starting preview-based Vosk dictation engine");
    #[cfg(not(feature = "vosk"))]
    info!("Starting Whisper-only dictation engine");

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
                silence_threshold_db: default_silence_threshold_db(),
            }
        }
    });

    let sample_rate: u32 = config.daemon.sample_rate.parse()
        .unwrap_or_else(|_| {
            warn!("Invalid sample_rate '{}', defaulting to 16000", config.daemon.sample_rate);
            16000
        });

    // Convert silence threshold from dB to linear RMS value
    let silence_threshold = db_to_linear(config.daemon.silence_threshold_db);
    info!("Silence threshold: {:.1}dB ({:.6} linear)", config.daemon.silence_threshold_db, silence_threshold);

    info!("Config loaded: audio_device={}, sample_rate={}, language={}",
          config.daemon.audio_device, sample_rate, config.daemon.language);

    // Initialize user dictionary
    let user_dict = Arc::new(UserDictionary::new().unwrap_or_else(|e| {
        warn!("Failed to initialize user dictionary: {}, spell checking will use defaults only", e);
        // Create empty dictionary that won't load any words
        UserDictionary::empty()
    }));
    info!("User dictionary initialized");

    // Spawn file watcher for dictionary hot-reload
    let user_dict_watcher = Arc::clone(&user_dict);
    tokio::spawn(async move {
        if let Err(e) = watch_dictionary_files(user_dict_watcher).await {
            error!("Dictionary file watcher error: {}", e);
        }
    });

    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    
    #[cfg(feature = "vosk")]
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

    // Create initial audio channel (will be replaced each recording session for hot-swap support)
    let (_audio_tx, audio_rx) = mpsc::unbounded_channel::<Vec<i16>>();
    let audio_rx_shared = Arc::new(tokio::sync::Mutex::new(audio_rx));

    // Create GUI channels for integrated communication
    let (gui_control_tx, _) = broadcast::channel::<GuiControl>(100);
    let (spectrum_tx, _) = broadcast::channel::<Vec<f32>>(50);
    let (gui_status_tx, mut gui_status_rx) = mpsc::channel::<GuiStatus>(10);

    // Store audio config for lazy capture creation (supports hot-swap and sleep/wake)
    let audio_device_config = config.daemon.audio_device.clone();
    info!("Audio capture will be created fresh each recording session (hot-swap support)");

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

    // Lazy model loading: store config, load on first confirm
    let accurate_model_config = AccurateModelConfig {
        engine_type: config.daemon.transcription_engine.clone(),
        #[cfg(feature = "vosk")]
        vosk_final_path: final_model_path.clone(),
        whisper_model_name: config.daemon.whisper_final_model.clone(),
        whisper_model_dir: config.daemon.whisper_model_path.clone(),
    };
    let accurate_model: Arc<RwLock<Option<AccurateModel>>> = Arc::new(RwLock::new(None));
    info!("Accurate model will be loaded on first use (lazy loading enabled)");

    // Create watch channel for state sharing with D-Bus
    let (state_tx, state_rx) = tokio::sync::watch::channel(DaemonState::Idle);

    // Create D-Bus service for control commands
    // IMPORTANT: Must keep connection alive for D-Bus service to remain registered
    let (dbus_conn, _command_sender, mut command_rx) = dbus_control::create_dbus_service(state_rx).await?;
    let _dbus_conn = dbus_conn; // Keep alive but mark unused

    info!("Daemon initialized - entering idle state (GUI hidden)");

    // State machine variables
    let mut daemon_state = DaemonState::Idle;
    let mut session: Option<RecordingSession> = None;
    let mut audio_task: Option<tokio::task::JoinHandle<()>> = None;
    let mut preview_task: Option<tokio::task::JoinHandle<()>> = None;
    let mut capture: Option<AudioCapture> = None;

    // ===== PERSISTENT STATE MACHINE LOOP =====
    loop {
        match daemon_state {
            DaemonState::Idle => {
                // Wait for D-Bus commands with timeout
                match tokio::time::timeout(Duration::from_millis(100), command_rx.recv()).await {
                    Ok(Some(cmd)) => match cmd {
                        DaemonCommand::StartRecording => {
                            info!("Received StartRecording command");

                            // Create fresh audio capture (supports hot-swap and sleep/wake recovery)
                            info!("Creating fresh audio capture...");
                            let (new_audio_tx, new_audio_rx) = mpsc::unbounded_channel();

                            // Parse audio device config
                            let audio_device = if audio_device_config.is_empty() {
                                None
                            } else {
                                let device_str = audio_device_config.as_str();
                                let device_name = if let Some(idx) = device_str.find(" (") {
                                    &device_str[..idx]
                                } else {
                                    device_str
                                };
                                Some(device_name)
                            };

                            // Create and start new capture
                            let new_capture = AudioCapture::new(new_audio_tx, audio_device, sample_rate, silence_threshold)?;
                            new_capture.start()?;
                            capture = Some(new_capture);

                            // Replace audio receiver
                            {
                                let mut rx = audio_rx_shared.lock().await;
                                *rx = new_audio_rx;
                                info!("Audio capture created and started");
                            }

                            // Create new engine for new session
                            #[cfg(feature = "vosk")]
                            let session_engine = Arc::new(VoskEngine::new(&preview_model_path, sample_rate)?);
                            #[cfg(not(feature = "vosk"))]
                            let session_engine = {
                                // Use whisper for preview when vosk is disabled
                                let whisper_preview_path = format!("{}/{}", config.daemon.whisper_model_path, config.daemon.whisper_preview_model);
                                let whisper_preview_path = model_manager::ensure_whisper_model(
                                    &config.daemon.whisper_preview_model,
                                    &config.daemon.whisper_model_path,
                                ).unwrap_or_else(|_| std::path::PathBuf::from(&whisper_preview_path));
                                let path_str = whisper_preview_path.to_str()
                                    .ok_or_else(|| anyhow::anyhow!("Whisper preview model path contains invalid UTF-8"))?;
                                Arc::new(WhisperEngine::new(path_str, sample_rate)?)
                            };

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
                            // Skip grammar checking in preview for speed (saves ~10-20ms per update)
                            let user_dict_preview = Arc::clone(&user_dict);
                            preview_task = Some(tokio::spawn(async move {
                                let mut check_interval = tokio::time::interval(std::time::Duration::from_millis(200));
                                let pipeline = Pipeline::from_config_with_dict(
                                    enable_acronyms,
                                    enable_punctuation,
                                    false,  // grammar disabled in preview for speed
                                    Some(user_dict_preview),
                                );

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
                            let _ = state_tx.send(daemon_state);
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
                            let _ = state_tx.send(daemon_state);
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
                            let _ = state_tx.send(daemon_state);
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

                    // Lazy load accurate model if not already loaded
                    {
                        let needs_load = accurate_model.read().await.is_none();
                        if needs_load {
                            info!("Loading accurate model (first use)...");
                            let config_clone = accurate_model_config.clone();
                            let loaded = tokio::task::spawn_blocking(move || {
                                config_clone.load()
                            }).await.ok().flatten();

                            if loaded.is_some() {
                                *accurate_model.write().await = loaded;
                                info!("Accurate model loaded successfully");
                            } else {
                                return Err(anyhow::anyhow!("Failed to load accurate model"));
                            }
                        }
                    }

                    // Get read lock on the model for transcription
                    let model_guard = accurate_model.read().await;
                    let model_ref = model_guard.as_ref()
                        .ok_or_else(|| anyhow::anyhow!("Accurate model not loaded"))?;

                    info!("Running correction pass...");
                    let accurate_result = match model_ref {
                        #[cfg(feature = "vosk")]
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
                    let pipeline = Pipeline::from_config_with_dict(
                        config.daemon.enable_acronyms,
                        config.daemon.enable_punctuation,
                        config.daemon.enable_grammar,
                        Some(Arc::clone(&user_dict)),
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

                // Stop and drop audio capture to release resources
                if let Some(ref cap) = capture {
                    cap.stop()?;
                }
                capture = None; // Drop capture to release device handles

                session = None;
                daemon_state = DaemonState::Idle;
                let _ = state_tx.send(daemon_state);
                info!("Processing complete - returned to Idle state");
            }
        }
    }

    info!("Daemon shutting down");
    Ok(())
}
