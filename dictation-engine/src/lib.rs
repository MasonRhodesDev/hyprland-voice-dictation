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

mod chunking;
pub mod control_ipc;
pub mod dbus_control;
mod debug_audio;
mod engine;
mod gpu_detect;
mod keyboard;
mod model_manager;
mod model_selector;
#[cfg(feature = "parakeet")]
pub mod parakeet_engine;
mod post_processing;
mod stream_muxer;
mod window_detect;
pub mod user_dictionary;
pub mod vad;
#[cfg(feature = "vosk")]
mod vosk_engine;
mod whisper_engine;

pub use dictation_types::{GuiControl, GuiState, GuiStatus};

use dbus_control::DaemonCommand;
use engine::TranscriptionEngine;
use keyboard::KeyboardInjector;
use model_selector::ModelSpec;
use post_processing::{Pipeline, SanitizationProcessor, TextProcessor};
use stream_muxer::{MuxerConfig, StreamMuxer};
use user_dictionary::UserDictionary;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

// Re-export DaemonState from dbus_control
use dbus_control::DaemonState;

// Recording session context
struct RecordingSession {
    #[allow(dead_code)]
    start_time: Instant,
    engine: Arc<dyn TranscriptionEngine>,
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

    // Activation mode: "toggle" (default) or "hold"
    #[serde(default = "default_activation_mode")]
    activation_mode: String,

    // Unified model selection (format: "engine:model_name")
    // e.g., "vosk:vosk-model-en-us-daanzu-20200905-lgraph" or "whisper:ggml-small.en.bin"
    #[serde(default = "default_preview_model")]
    preview_model: String,
    #[serde(default = "default_final_model")]
    final_model: String,

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
    #[serde(default = "default_debug_audio")]
    debug_audio: bool,

    // Voice Activity Detection
    #[serde(default = "default_vad_enabled")]
    vad_enabled: bool,
    #[serde(default = "default_vad_threshold")]
    vad_threshold: f32,

    // Stream muxer (for multi-device capture)
    #[serde(default = "default_muxer_sticky_duration_ms")]
    muxer_sticky_duration_ms: u64,
    #[serde(default = "default_muxer_cooldown_ms")]
    muxer_cooldown_ms: u64,
    #[serde(default = "default_muxer_switch_threshold")]
    muxer_switch_threshold: f32,
    #[serde(default = "default_muxer_scoring_window_ms")]
    muxer_scoring_window_ms: u64,
}

fn default_language() -> String { "en".to_string() }
fn default_activation_mode() -> String { "toggle".to_string() }
fn default_preview_model() -> String { "vosk:vosk-model-en-us-daanzu-20200905-lgraph".to_string() }
fn default_final_model() -> String { "whisper:ggml-small.en.bin".to_string() }
fn default_enable_acronyms() -> bool { true }
fn default_enable_punctuation() -> bool { true }
fn default_enable_grammar() -> bool { true }
fn default_silence_threshold_db() -> f32 { -60.0 }
fn default_debug_audio() -> bool { false }
fn default_vad_enabled() -> bool { true }
fn default_vad_threshold() -> f32 { 0.5 }
fn default_muxer_sticky_duration_ms() -> u64 { 500 }
fn default_muxer_cooldown_ms() -> u64 { 200 }
fn default_muxer_switch_threshold() -> f32 { 0.15 }
fn default_muxer_scoring_window_ms() -> u64 { 100 }

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


/// Accurate model wrapper for correction pass.
enum AccurateModel {
    #[cfg(feature = "vosk")]
    Vosk(Model),
    Whisper(WhisperContext),
    #[cfg(feature = "parakeet")]
    Parakeet(Arc<dyn TranscriptionEngine>),
}

/// Configuration for lazy-loading the accurate model using ModelSpec
#[derive(Clone)]
struct AccurateModelConfig {
    /// The model specification (e.g., "whisper:ggml-small.en.bin")
    spec: ModelSpec,
}

impl AccurateModelConfig {
    /// Load the accurate model (blocking operation, run in spawn_blocking)
    fn load(&self) -> Option<AccurateModel> {
        use model_selector::EngineType;

        match self.spec.engine {
            #[cfg(feature = "vosk")]
            EngineType::Vosk => {
                let model_path = self.spec.model_path();
                info!("Loading Vosk accurate model from: {:?}", model_path);
                if !model_path.exists() {
                    error!("Vosk model not found at {:?}", model_path);
                    return None;
                }
                Model::new(model_path.to_str()?).map(AccurateModel::Vosk)
            }
            #[cfg(not(feature = "vosk"))]
            EngineType::Vosk => {
                error!("Vosk support not compiled in. Select a whisper or parakeet model.");
                None
            }
            EngineType::Whisper => {
                info!("Ensuring Whisper model available: {}", self.spec.model_name);

                let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
                let whisper_model_dir = format!("{}/.config/voice-dictation/models/whisper", home);

                // Ensure model exists, download if necessary
                let model_path = match model_manager::ensure_whisper_model(
                    &self.spec.model_name,
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

                let model_path_str = match model_path.to_str() {
                    Some(s) => s,
                    None => {
                        error!("Whisper model path contains invalid UTF-8");
                        return None;
                    }
                };

                // Auto-detect GPU and use if available
                let params = if gpu_detect::cuda_available() {
                    info!("CUDA detected, enabling GPU acceleration for Whisper");
                    WhisperContextParameters::default()
                } else {
                    WhisperContextParameters::default()
                };

                WhisperContext::new_with_params(model_path_str, params)
                    .map(AccurateModel::Whisper)
                    .map_err(|e| {
                        error!("Whisper model load failed: {:?}", e);
                        e
                    })
                    .ok()
            }
            #[cfg(feature = "parakeet")]
            EngineType::Parakeet => {
                // Create Parakeet engine for accurate transcription
                info!("Loading Parakeet as accurate model...");
                match self.spec.create_engine(16000) {
                    Ok(engine) => Some(AccurateModel::Parakeet(engine)),
                    Err(e) => {
                        error!("Failed to load Parakeet accurate model: {:?}", e);
                        None
                    }
                }
            }
            #[cfg(not(feature = "parakeet"))]
            EngineType::Parakeet => {
                error!("Parakeet support not compiled in.");
                None
            }
        }
    }
}

struct AudioCapture {
    streams: Vec<Stream>,
    #[allow(dead_code)] // Kept alive for stream selection; may be used for debug finalization
    muxer: Arc<std::sync::Mutex<StreamMuxer>>,
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

    fn new(
        tx: mpsc::UnboundedSender<Vec<i16>>,
        device_name: Option<&str>,
        sample_rate: u32,
        silence_threshold: f32,
        muxer_config: MuxerConfig,
    ) -> Result<Self> {
        let host = cpal::default_host();

        // Determine which devices to use
        // Fast path: for "default" mode, skip slow device enumeration
        let devices_to_use: Vec<_> = match device_name {
            // "default" or None: use system default directly (fast path)
            None | Some("default") => {
                info!("Using system default audio device (fast path)");
                let mut found = Vec::new();
                if let Some(default) = host.default_input_device() {
                    if let Ok(name) = default.name() {
                        info!("Default device: '{}'", name);
                    }
                    found.push(default);
                }
                found
            }
            // "all" mode: capture from pipewire + all hardware devices (slow path)
            // StreamMuxer selects the best stream in real-time
            Some("all") => {
                info!("Multi-device mode: enumerating audio devices...");
                let mut real_devices = Vec::new();
                if let Ok(devices) = host.input_devices() {
                    for device in devices {
                        if let Ok(name) = device.name() {
                            let is_real = Self::is_real_input_device(&name);
                            info!("  - '{}' {}", name, if is_real { "(will capture)" } else { "(skipped)" });
                            if is_real {
                                real_devices.push(device);
                            }
                        }
                    }
                }

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

                info!("Multi-device mode: capturing from {} device(s) with StreamMuxer selection", devices.len());
                devices
            }
            // Specific device requested (need to enumerate to find it)
            Some(name) => {
                info!("Searching for device '{}'...", name);
                let mut found = Vec::new();
                if let Ok(devices) = host.input_devices() {
                    for device in devices {
                        if let Ok(device_name) = device.name() {
                            if device_name == name {
                                found.push(device);
                                break;
                            }
                        }
                    }
                }
                if found.is_empty() {
                    warn!("Device '{}' not found, using default", name);
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

        // Create crossbeam channel for muxer output (lock-free, for audio callback)
        let (muxer_tx, muxer_rx) = crossbeam_channel::bounded(100);

        // Create StreamMuxer
        let muxer = StreamMuxer::new(muxer_tx, muxer_config)?;
        let muxer = Arc::new(std::sync::Mutex::new(muxer));

        let config = StreamConfig {
            channels: 1,
            sample_rate: cpal::SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let mut streams = Vec::new();
        for device in devices_to_use {
            let stream_id = device.name().unwrap_or_else(|_| "unknown".to_string());
            let muxer_clone = Arc::clone(&muxer);
            let stream_id_clone = stream_id.clone();
            let threshold = silence_threshold;

            match device.build_input_stream(
                &config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    // Pre-filter obviously silent chunks to reduce muxer load
                    let rms: f32 = (data.iter().map(|&s| s * s).sum::<f32>() / data.len() as f32).sqrt();
                    if rms < threshold {
                        return; // Skip completely silent chunks
                    }

                    // Convert to i16
                    let samples: Vec<i16> =
                        data.iter().map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16).collect();

                    // Push to muxer for quality-based stream selection
                    if let Ok(mut muxer) = muxer_clone.lock() {
                        muxer.push_samples(&stream_id_clone, &samples);
                    }
                },
                |err| error!("Audio stream error: {}", err),
                None,
            ) {
                Ok(stream) => {
                    info!("Created audio stream for: {}", stream_id);
                    streams.push(stream);
                }
                Err(e) => {
                    warn!("Failed to create stream for '{}': {}", stream_id, e);
                }
            }
        }

        if streams.is_empty() {
            return Err(anyhow::anyhow!("Failed to create any audio streams"));
        }

        // Spawn thread to forward muxer output to async channel
        let tx_clone = tx;
        std::thread::spawn(move || {
            while let Ok(samples) = muxer_rx.recv() {
                if tx_clone.send(samples).is_err() {
                    break; // Channel closed
                }
            }
        });

        info!("Audio capture initialized with {} stream(s) and StreamMuxer", streams.len());
        Ok(Self { streams, muxer })
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

    /// Finalize the muxer (flush debug recordings).
    #[allow(dead_code)] // Available for future use
    fn finalize_muxer(&self) -> Result<()> {
        // Note: We can't take ownership of muxer here, so debug recordings
        // will be finalized when the muxer is dropped. For explicit finalization,
        // the caller should drop the AudioCapture.
        if let Ok(muxer) = self.muxer.lock() {
            if let Some(stream_id) = muxer.current_stream() {
                info!("Final selected stream: {}", stream_id);
            }
        }
        Ok(())
    }
}

/// Configuration for DeviceManager
#[derive(Clone)]
struct DeviceManagerConfig {
    device_name: Option<String>,
    sample_rate: u32,
    silence_threshold: f32,
    muxer_config: MuxerConfig,
}

/// Manages audio devices with eager loading and hotplug support.
/// Streams are pre-created at startup and only started/stopped on recording.
struct DeviceManager {
    config: DeviceManagerConfig,
    capture: Option<AudioCapture>,
    audio_tx: mpsc::UnboundedSender<Vec<i16>>,
    needs_recreate: Arc<std::sync::atomic::AtomicBool>,
}

impl DeviceManager {
    /// Create a new DeviceManager with pre-created audio streams.
    fn new(
        config: DeviceManagerConfig,
        audio_tx: mpsc::UnboundedSender<Vec<i16>>,
    ) -> Result<Self> {
        // Create initial capture (streams created but paused)
        info!("DeviceManager: Pre-creating audio streams...");
        let capture = Self::create_capture(&config, audio_tx.clone())?;

        Ok(Self {
            config,
            capture: Some(capture),
            audio_tx,
            needs_recreate: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }

    /// Create an AudioCapture with the given config
    fn create_capture(
        config: &DeviceManagerConfig,
        tx: mpsc::UnboundedSender<Vec<i16>>,
    ) -> Result<AudioCapture> {
        let device_name = config.device_name.as_deref();
        AudioCapture::new(
            tx,
            device_name,
            config.sample_rate,
            config.silence_threshold,
            config.muxer_config.clone(),
        )
    }

    /// Start recording (fast unless devices changed since last recording)
    fn start(&mut self) -> Result<()> {
        // Check if we need to recreate due to device changes
        if self.needs_recreate.swap(false, std::sync::atomic::Ordering::SeqCst) {
            info!("DeviceManager: Device change detected, recreating streams...");
            self.capture = None;
            self.capture = Some(Self::create_capture(&self.config, self.audio_tx.clone())?);
            info!("DeviceManager: Streams recreated");
        }

        if let Some(ref capture) = self.capture {
            capture.start()?;
        } else {
            return Err(anyhow::anyhow!("No audio capture available"));
        }
        Ok(())
    }

    /// Stop recording (streams paused but kept alive)
    fn stop(&self) -> Result<()> {
        if let Some(ref capture) = self.capture {
            capture.stop()?;
        }
        Ok(())
    }

    /// Spawn a background task to watch for device changes
    fn spawn_device_watcher(&self) {
        let needs_recreate = Arc::clone(&self.needs_recreate);

        std::thread::spawn(move || {
            // Watch /dev/snd for device changes (Linux-specific)
            let snd_path = std::path::Path::new("/dev/snd");
            if !snd_path.exists() {
                warn!("DeviceManager: /dev/snd not found, device hotplug detection disabled");
                return;
            }

            let flag = needs_recreate;
            let mut watcher = match notify::recommended_watcher(move |res: std::result::Result<Event, notify::Error>| {
                if let Ok(event) = res {
                    match event.kind {
                        EventKind::Create(_) | EventKind::Remove(_) => {
                            info!("DeviceManager: Audio device change detected, will recreate on next start");
                            flag.store(true, std::sync::atomic::Ordering::SeqCst);
                        }
                        _ => {}
                    }
                }
            }) {
                Ok(w) => w,
                Err(e) => {
                    error!("DeviceManager: Failed to create watcher: {}", e);
                    return;
                }
            };

            if let Err(e) = watcher.watch(snd_path, RecursiveMode::NonRecursive) {
                error!("DeviceManager: Failed to watch /dev/snd: {}", e);
                return;
            }

            info!("DeviceManager: Watching /dev/snd for device changes");

            // Keep thread alive
            loop {
                std::thread::sleep(Duration::from_secs(60));
            }
        });
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
                activation_mode: default_activation_mode(),
                preview_model: default_preview_model(),
                final_model: default_final_model(),
                enable_acronyms: default_enable_acronyms(),
                enable_punctuation: default_enable_punctuation(),
                enable_grammar: default_enable_grammar(),
                silence_threshold_db: default_silence_threshold_db(),
                debug_audio: default_debug_audio(),
                vad_enabled: default_vad_enabled(),
                vad_threshold: default_vad_threshold(),
                muxer_sticky_duration_ms: default_muxer_sticky_duration_ms(),
                muxer_cooldown_ms: default_muxer_cooldown_ms(),
                muxer_switch_threshold: default_muxer_switch_threshold(),
                muxer_scoring_window_ms: default_muxer_scoring_window_ms(),
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

    // Parse model specifications (format: "engine:model_name")
    let preview_spec = ModelSpec::parse(&config.daemon.preview_model)
        .map_err(|e| anyhow::anyhow!("Invalid preview_model '{}': {}", config.daemon.preview_model, e))?;
    let final_spec = ModelSpec::parse(&config.daemon.final_model)
        .map_err(|e| anyhow::anyhow!("Invalid final_model '{}': {}", config.daemon.final_model, e))?;

    info!("Preview model: {} ({})", preview_spec.model_name, preview_spec.engine);
    info!("Final model: {} ({})", final_spec.model_name, final_spec.engine);

    // Check if preview and final models are identical (skip accurate pass if same)
    let same_model_for_preview_and_final = preview_spec.engine == final_spec.engine
        && preview_spec.model_name == final_spec.model_name;
    if same_model_for_preview_and_final {
        info!("Preview and final models are identical - accurate pass will be skipped");
    }

    // Validate that configured models are available
    if !preview_spec.is_available() {
        return Err(anyhow::anyhow!(
            "Preview model '{}' not found at {:?}. Check that the model is installed.",
            config.daemon.preview_model,
            preview_spec.model_path()
        ));
    }
    if !final_spec.is_available() {
        return Err(anyhow::anyhow!(
            "Final model '{}' not found at {:?}. Check that the model is installed.",
            config.daemon.final_model,
            final_spec.model_path()
        ));
    }

    // Create audio channel (shared between DeviceManager and processing)
    let (audio_tx, audio_rx) = mpsc::unbounded_channel::<Vec<i16>>();
    let audio_rx_shared = Arc::new(tokio::sync::Mutex::new(audio_rx));

    // Create GUI channels for integrated communication
    let (gui_control_tx, _) = broadcast::channel::<GuiControl>(100);
    let (spectrum_tx, _) = broadcast::channel::<Vec<f32>>(50);
    let (gui_status_tx, mut gui_status_rx) = mpsc::channel::<GuiStatus>(10);

    // Parse audio device config
    let audio_device_name = if config.daemon.audio_device.is_empty() || config.daemon.audio_device == "default" {
        None
    } else {
        Some(config.daemon.audio_device.clone())
    };

    // Create muxer config
    let muxer_config = MuxerConfig {
        sticky_duration_ms: config.daemon.muxer_sticky_duration_ms,
        cooldown_ms: config.daemon.muxer_cooldown_ms,
        switch_threshold: config.daemon.muxer_switch_threshold,
        scoring_window_ms: config.daemon.muxer_scoring_window_ms,
        sample_rate,
        debug_audio: config.daemon.debug_audio,
    };

    // Create DeviceManager with eager-loaded audio streams
    info!("Creating DeviceManager with pre-loaded audio streams...");
    let device_manager_config = DeviceManagerConfig {
        device_name: audio_device_name,
        sample_rate,
        silence_threshold,
        muxer_config,
    };
    let mut device_manager = DeviceManager::new(device_manager_config, audio_tx)?;

    // Spawn device hotplug watcher
    device_manager.spawn_device_watcher();
    info!("Audio streams pre-loaded and ready (fast startup enabled)");

    let keyboard = Arc::new(KeyboardInjector::new(10, 50));

    // Spawn integrated GUI
    info!("Spawning integrated GUI...");
    // Pass sender clones to GUI - it will subscribe there and keep senders alive for channel lifetime
    let gui_control_tx_gui = gui_control_tx.clone();
    let spectrum_tx_gui = spectrum_tx.clone();
    // Get runtime handle to pass to GUI for spawning async tasks
    let runtime_handle = tokio::runtime::Handle::current();
    let _gui_handle = tokio::task::spawn_blocking(move || {
        dictation_gui::run_integrated(
            gui_control_tx_gui,
            spectrum_tx_gui,
            gui_status_tx,
            runtime_handle,
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

    // Pre-load preview engine at startup for instant recording start
    // This runs before D-Bus registration so blocking is acceptable
    info!("Pre-loading preview engine (blocking call before D-Bus)...");
    let preview_engine: Arc<dyn TranscriptionEngine> = preview_spec.create_engine(sample_rate)?;
    info!("Preview engine loaded and ready");

    // Lazy model loading: store config, load on first confirm
    let accurate_model_config = AccurateModelConfig {
        spec: final_spec.clone(),
    };
    let accurate_model: Arc<RwLock<Option<AccurateModel>>> = Arc::new(RwLock::new(None));

    // Create watch channel for state sharing with D-Bus
    let (state_tx, state_rx) = tokio::sync::watch::channel(DaemonState::Idle);

    // Create D-Bus service for control commands
    // IMPORTANT: Must keep connection alive for D-Bus service to remain registered
    let (dbus_conn, _command_sender, mut command_rx) = dbus_control::create_dbus_service(state_rx).await?;
    let _dbus_conn = dbus_conn; // Keep alive but mark unused

    // Pre-load accurate model in background (only if different from preview)
    if !same_model_for_preview_and_final {
        info!("Pre-loading accurate model in background...");
        let config_clone = accurate_model_config.clone();
        let accurate_model_clone = Arc::clone(&accurate_model);
        tokio::task::spawn(async move {
            let loaded = tokio::task::spawn_blocking(move || {
                config_clone.load()
            }).await.ok().flatten();

            if let Some(model) = loaded {
                *accurate_model_clone.write().await = Some(model);
                info!("Accurate model pre-loaded successfully (background)");
            } else {
                tracing::warn!("Failed to pre-load accurate model (will retry on first use)");
            }
        });
    } else {
        info!("Same model for preview/final - no accurate model pre-load needed");
    }

    info!("Daemon initialized - entering idle state (GUI hidden)");

    // State machine variables
    let mut daemon_state = DaemonState::Idle;
    let mut session: Option<RecordingSession> = None;
    let mut audio_task: Option<tokio::task::JoinHandle<()>> = None;
    let mut preview_task: Option<tokio::task::JoinHandle<()>> = None;
    // Cancellation channel for graceful task shutdown (keeps spectrum channel alive)
    let (cancel_tx, _cancel_rx) = tokio::sync::watch::channel(false);

    // ===== PERSISTENT STATE MACHINE LOOP =====
    loop {
        match daemon_state {
            DaemonState::Idle => {
                // Wait for D-Bus commands with timeout
                match tokio::time::timeout(Duration::from_millis(100), command_rx.recv()).await {
                    Ok(Some(cmd)) => match cmd {
                        DaemonCommand::StartRecording => {
                            info!("Received StartRecording command");

                            // Drain any stale audio data from the channel before starting
                            // This prevents audio captured before the user pressed record from being transcribed
                            {
                                let mut rx = audio_rx_shared.lock().await;
                                let mut drained = 0;
                                while rx.try_recv().is_ok() {
                                    drained += 1;
                                }
                                if drained > 0 {
                                    info!("Drained {} stale audio chunks from channel", drained);
                                }
                            }

                            // Start pre-loaded audio streams (fast - no device enumeration)
                            device_manager.start()?;
                            info!("Audio capture started (pre-loaded streams)");

                            // Reset the pre-loaded engine for new session
                            preview_engine.reset();
                            let session_engine = Arc::clone(&preview_engine);

                            // Signal UI to show - after audio is ready so user can start talking immediately
                            gui_control_tx.send(GuiControl::SetListening)
                                .map_err(|e| anyhow::anyhow!("Failed to send SetListening: {}", e))?;

                            // Create session
                            session = Some(RecordingSession {
                                start_time: Instant::now(),
                                engine: Arc::clone(&session_engine),
                            });

                            // Reset cancellation flag for new session
                            let _ = cancel_tx.send(false);

                            // Start audio processing task
                            let engine_clone = Arc::clone(&session_engine);
                            let spectrum_tx_clone = spectrum_tx.clone();
                            let audio_rx_clone = Arc::clone(&audio_rx_shared);
                            let mut cancel_rx = cancel_tx.subscribe();
                            audio_task = Some(tokio::spawn(async move {
                                let mut buffer = Vec::new();
                                loop {
                                    // Use select to allow graceful cancellation
                                    tokio::select! {
                                        biased;
                                        _ = cancel_rx.changed() => {
                                            if *cancel_rx.borrow() {
                                                debug!("Audio task: cancellation received");
                                                break;
                                            }
                                        }
                                        samples = async {
                                            let mut rx = audio_rx_clone.lock().await;
                                            rx.recv().await
                                        } => {
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
                                    }
                                }
                                debug!("Audio task: exiting gracefully");
                            }));

                            // Start preview task
                            let engine_clone = Arc::clone(&session_engine);
                            let gui_control_tx_preview = gui_control_tx.clone();
                            let enable_acronyms = config.daemon.enable_acronyms;
                            let enable_punctuation = config.daemon.enable_punctuation;
                            // Skip grammar checking in preview for speed (saves ~10-20ms per update)
                            let user_dict_preview = Arc::clone(&user_dict);
                            let mut cancel_rx_preview = cancel_tx.subscribe();
                            preview_task = Some(tokio::spawn(async move {
                                // 100ms polling for responsive text updates (was 200ms)
                                let mut check_interval = tokio::time::interval(std::time::Duration::from_millis(100));
                                let pipeline = Pipeline::from_config_with_dict(
                                    enable_acronyms,
                                    enable_punctuation,
                                    false,  // grammar disabled in preview for speed
                                    Some(user_dict_preview),
                                );

                                // Track text changes for VAD state sync
                                let mut last_text = String::new();
                                let mut last_text_change = Instant::now();
                                const TEXT_SETTLED_THRESHOLD_MS: u64 = 300;

                                loop {
                                    tokio::select! {
                                        biased;
                                        _ = cancel_rx_preview.changed() => {
                                            if *cancel_rx_preview.borrow() {
                                                debug!("Preview task: cancellation received");
                                                break;
                                            }
                                        }
                                        _ = check_interval.tick() => {
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

                                                    // Track text changes for VAD state
                                                    let text_changed = text_processed != last_text;
                                                    if text_changed {
                                                        last_text = text_processed.clone();
                                                        last_text_change = Instant::now();
                                                    }

                                                    // Determine VAD state
                                                    let text_settled = last_text_change.elapsed().as_millis() >= TEXT_SETTLED_THRESHOLD_MS as u128;
                                                    let is_speaking = !text_processed.is_empty() && !text_settled;

                                                    let _ = gui_control_tx_preview.send(GuiControl::UpdateTranscription {
                                                        text: text_processed,
                                                        is_final: false,
                                                    });

                                                    // Send VAD state for visual sync
                                                    let _ = gui_control_tx_preview.send(GuiControl::UpdateVadState {
                                                        is_speaking,
                                                        text_settled,
                                                    });
                                                }
                                                Err(e) => error!("Failed to get text: {}", e),
                                            }
                                        }
                                    }
                                }
                                debug!("Preview task: exiting gracefully");
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

                            // Stop audio capture (streams paused but kept alive)
                            let _ = device_manager.stop();

                            // Signal tasks to stop gracefully
                            let _ = cancel_tx.send(true);
                            // Wait for tasks to finish
                            if let Some(task) = audio_task.take() {
                                let _ = task.await;
                            }
                            if let Some(task) = preview_task.take() {
                                let _ = task.await;
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
                            // Stop audio capture
                            let _ = device_manager.stop();
                            // Signal tasks to stop gracefully
                            let _ = cancel_tx.send(true);
                            if let Some(task) = audio_task.take() {
                                let _ = task.await;
                            }
                            if let Some(task) = preview_task.take() {
                                let _ = task.await;
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

                // Signal tasks to stop gracefully
                let _ = cancel_tx.send(true);
                if let Some(task) = audio_task.take() {
                    let _ = task.await;
                }
                if let Some(task) = preview_task.take() {
                    let _ = task.await;
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
                    // Determine final text: skip accurate pass if models are identical
                    let accurate_result = if same_model_for_preview_and_final {
                        // Same model - use preview result directly (skip expensive re-transcription)
                        info!("Same model for preview/final - skipping accurate pass");
                        fast_result.clone()
                    } else {
                        // Different model - run accurate transcription
                        // Send processing state to GUI (only show spinner if doing work)
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
                        let result = match model_ref {
                            #[cfg(feature = "vosk")]
                            AccurateModel::Vosk(vosk_model) => {
                                // Get audio buffer from session engine
                                let audio_buffer = session_engine.get_audio_buffer();
                                info!("Running Vosk accurate transcription on {:.2}s of audio...",
                                    audio_buffer.len() as f32 / sample_rate as f32);

                                // Create a new recognizer for this transcription
                                let mut recognizer = vosk::Recognizer::new(vosk_model, sample_rate as f32)
                                    .ok_or_else(|| anyhow::anyhow!("Failed to create Vosk recognizer"))?;

                                // Process all audio in one go for accurate transcription
                                recognizer.accept_waveform(&audio_buffer);
                                let final_result = recognizer.final_result();

                                // Extract text from result
                                match final_result {
                                    vosk::CompleteResult::Single(single) => single.text.to_string(),
                                    vosk::CompleteResult::Multiple(multi) => {
                                        multi.alternatives.first()
                                            .map(|a| a.text.to_string())
                                            .unwrap_or_default()
                                    }
                                }
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

                                let segments: Vec<String> = state
                                    .as_iter()
                                    .filter_map(|segment| {
                                        segment.to_str_lossy().ok().map(|text| text.trim().to_string())
                                    })
                                    .filter(|text| !text.is_empty())
                                    .collect();

                                segments.join(" ")
                            }
                            #[cfg(feature = "parakeet")]
                            AccurateModel::Parakeet(parakeet_engine) => {
                                // Feed audio to Parakeet and get final result
                                let audio_buffer = session_engine.get_audio_buffer();
                                info!("Running Parakeet accurate transcription on {:.2}s of audio...",
                                    audio_buffer.len() as f32 / sample_rate as f32);

                                // Reset and process audio through Parakeet
                                parakeet_engine.reset();
                                parakeet_engine.process_audio(&audio_buffer)?;
                                parakeet_engine.get_final_result()?
                            }
                        };
                        info!("[Accurate] Raw: '{}'", result);
                        result
                    };

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

                    // Save debug audio if enabled
                    if debug_audio::is_debug_audio_enabled() {
                        let audio_buffer = session_engine.get_audio_buffer();
                        let metadata = debug_audio::AudioMetadata {
                            timestamp: chrono::Utc::now(),
                            duration_ms: (audio_buffer.len() as u64 * 1000) / sample_rate as u64,
                            sample_rate,
                            sample_count: audio_buffer.len(),
                            devices: vec![config.daemon.audio_device.clone()],
                            active_device: Some(config.daemon.audio_device.clone()),
                            preview_text: fast_result.clone(),
                            final_text: processed_result.clone(),
                            preview_engine: format!("{}", preview_spec.engine),
                            accurate_engine: format!("{}", final_spec.engine),
                            same_model_used: same_model_for_preview_and_final,
                        };
                        if let Err(e) = debug_audio::save_debug_audio(&audio_buffer, sample_rate, metadata) {
                            warn!("Failed to save debug audio: {}", e);
                        }
                    }

                    // Detect focused app and sanitize text accordingly
                    let app_category = window_detect::get_focused_app_category().await;
                    let sanitizer = SanitizationProcessor::for_category(app_category);
                    let sanitized_result = sanitizer.process(&processed_result)?;

                    // Copy to clipboard as backup (wl-copy for Wayland)
                    // Use spawn() not output() - wl-copy stays running to serve clipboard
                    match tokio::process::Command::new("wl-copy")
                        .arg(&sanitized_result)
                        .stdin(std::process::Stdio::null())
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .spawn()
                    {
                        Ok(_) => {
                            debug!("Copied to clipboard ({} chars)", sanitized_result.len());
                        }
                        Err(e) => {
                            warn!("Failed to run wl-copy: {}", e);
                        }
                    }

                    info!("Typing final text ({:?} mode)...", app_category);
                    keyboard.type_text(&sanitized_result).await?;
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

                // Stop audio capture (streams paused but kept alive for next session)
                let _ = device_manager.stop();

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
