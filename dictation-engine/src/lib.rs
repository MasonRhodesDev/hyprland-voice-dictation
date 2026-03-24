use anyhow::Result;
use notify::{Event, EventKind, RecursiveMode, Watcher};
use serde::Deserialize;
use std::fs;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use systemd::daemon::{notify, STATE_READY, STATE_WATCHDOG};
use tokio::sync::{broadcast, mpsc, Mutex, RwLock};
use tracing::{debug, error, info, warn};

pub mod audio_backend;
mod chunking;
pub mod control_ipc;
pub mod ctc_engine;
pub mod dbus_control;
mod debug_audio;
mod engine;
mod app_profile;
pub mod hotword_trie;
mod keyboard;
pub mod model_selector;
pub mod parakeet_engine;
pub mod post_processing;
mod window_detect;
mod window_target;
pub mod user_dictionary;
pub mod vad;
#[cfg(feature = "tray")]
mod tray;

pub use dictation_types::{GuiControl, GuiState, GuiStatus};

/// Check if media is playing and pause it. Returns true if media was paused.
fn pause_media_if_playing() -> bool {
    let Ok(output) = std::process::Command::new("playerctl")
        .arg("status")
        .output() else { return false };
    let playing = String::from_utf8_lossy(&output.stdout).contains("Playing");
    if playing {
        let _ = std::process::Command::new("playerctl").arg("pause").output();
        info!("Paused media playback");
    }
    playing
}

/// Resume media playback.
fn resume_media() {
    let _ = std::process::Command::new("playerctl").arg("play").output();
    info!("Resumed media playback");
}

use audio_backend::{AudioBackend, AudioBackendConfig, BackendType};
use dbus_control::DaemonCommand;
use engine::TranscriptionEngine;
use keyboard::KeyboardInjector;
use model_selector::ModelSpec;
use post_processing::{Pipeline, SanitizationProcessor, TextProcessor, WordSubstitutionProcessor};
use user_dictionary::UserDictionary;

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
#[allow(dead_code)]
struct DaemonConfig {
    audio_device: String,
    sample_rate: String,

    // Model selection (format: "parakeet:model_name")
    #[serde(default = "default_model", alias = "preview_model")]
    model: String,

    // Post-processing
    #[serde(default = "default_enable_acronyms")]
    enable_acronyms: bool,
    #[serde(default = "default_enable_punctuation")]
    enable_punctuation: bool,
    #[serde(default = "default_enable_grammar")]
    enable_grammar: bool,
    #[serde(default = "default_enable_word_substitution")]
    enable_word_substitution: bool,

    // Audio capture
    #[serde(default = "default_silence_threshold_db")]
    silence_threshold_db: f32,
    #[serde(default = "default_debug_audio")]
    debug_audio: bool,

    // Trailing audio buffer after stop command (captures final words)
    #[serde(default = "default_trailing_buffer_ms")]
    trailing_buffer_ms: u64,

    // Audio backend selection: "auto" (default), "cpal", or "pipewire"
    #[serde(default = "default_audio_backend")]
    audio_backend: String,

    // Idle release timeout: how long to keep mic open after stop before releasing (seconds)
    #[serde(default = "default_idle_release_timeout_secs")]
    idle_release_timeout_secs: u64,

    // Delay before resuming media playback after recording stops (milliseconds)
    #[serde(default = "default_media_resume_delay_ms")]
    media_resume_delay_ms: u64,

    // Engine idle timeout: drop ORT sessions after N seconds idle to reclaim BFCArena memory (seconds)
    #[serde(default = "default_engine_idle_timeout_secs")]
    engine_idle_timeout_secs: u64,
}

fn default_model() -> String { "parakeet:default".to_string() }
fn default_enable_acronyms() -> bool { true }
fn default_enable_punctuation() -> bool { true }
fn default_enable_grammar() -> bool { true }
fn default_enable_word_substitution() -> bool { true }
fn default_silence_threshold_db() -> f32 { -60.0 }
fn default_debug_audio() -> bool { false }
fn default_trailing_buffer_ms() -> u64 { 750 }
fn default_audio_backend() -> String { "auto".to_string() }
fn default_idle_release_timeout_secs() -> u64 { 30 }
fn default_media_resume_delay_ms() -> u64 { 25 }
fn default_engine_idle_timeout_secs() -> u64 { 300 }  // 5 minutes

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

/// Watch substitution file and reload on changes.
async fn watch_substitution_file(word_sub: WordSubstitutionProcessor) -> Result<()> {
    let path = WordSubstitutionProcessor::watch_path();

    if path.as_os_str().is_empty() {
        info!("No substitution file path to watch");
        return Ok(());
    }

    info!("Watching substitution file: {:?}", path);

    let (tx, mut rx) = mpsc::channel(100);

    let mut watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
        if let Ok(event) = res {
            let _ = tx.blocking_send(event);
        }
    })?;

    if path.exists() {
        watcher.watch(&path, RecursiveMode::NonRecursive)?;
    } else if let Some(parent) = path.parent() {
        if parent.exists() {
            watcher.watch(parent, RecursiveMode::NonRecursive)?;
        }
    }

    while let Some(event) = rx.recv().await {
        match event.kind {
            EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_) => {
                if event.paths.iter().any(|p| p == &path) {
                    info!("Substitution file changed: {:?}, reloading...", path);
                    if let Err(e) = word_sub.reload() {
                        warn!("Failed to reload substitutions: {}", e);
                    } else {
                        info!("Substitutions reloaded successfully");
                    }
                }
            }
            _ => {}
        }
    }

    Ok(())
}

/// Health state shared between subsystems and D-Bus service.
pub struct HealthState {
    /// Whether audio is flowing (updated by audio forwarding thread)
    pub audio_healthy: AtomicBool,
    /// Whether the engine has produced a successful transcription
    pub engine_healthy: AtomicBool,
    /// Whether the GUI is available
    pub gui_healthy: AtomicBool,
    /// Timestamp (ms since epoch) of last audio received
    pub last_audio_timestamp_ms: AtomicU64,
    /// Last error message (if any)
    pub last_error: RwLock<Option<String>>,
}

impl HealthState {
    fn new() -> Self {
        Self {
            audio_healthy: AtomicBool::new(false),
            engine_healthy: AtomicBool::new(false),
            gui_healthy: AtomicBool::new(false),
            last_audio_timestamp_ms: AtomicU64::new(0),
            last_error: RwLock::new(None),
        }
    }

    /// Check if all subsystems are healthy enough to send watchdog keepalive
    pub fn is_healthy(&self) -> bool {
        // Engine health is the critical check - if it loaded, we're functional
        // Audio health is only relevant during recording
        self.engine_healthy.load(Ordering::Relaxed)
    }
}

/// Configuration for DeviceManager
#[derive(Clone)]
struct DeviceManagerConfig {
    backend_type: BackendType,
    backend_config: AudioBackendConfig,
    /// Idle timeout before releasing microphone (seconds). 0 = release immediately.
    idle_release_timeout_secs: u64,
}

/// Manages audio devices with idle timeout and hotplug support.
struct DeviceManager {
    config: DeviceManagerConfig,
    backend: Option<Box<dyn AudioBackend>>,
    audio_tx: mpsc::UnboundedSender<Vec<i16>>,
    needs_recreate: Arc<std::sync::atomic::AtomicBool>,
    /// When the audio was last stopped (for idle timeout tracking)
    stopped_at: Option<Instant>,
}

impl DeviceManager {
    /// Create a new DeviceManager with pre-created audio backend.
    fn new(
        config: DeviceManagerConfig,
        audio_tx: mpsc::UnboundedSender<Vec<i16>>,
    ) -> Result<Self> {
        // Create initial backend (streams created but paused)
        info!("DeviceManager: Pre-creating audio backend ({:?})...", config.backend_type);
        let backend = Self::create_backend(&config, audio_tx.clone())?;

        Ok(Self {
            config,
            backend: Some(backend),
            audio_tx,
            needs_recreate: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            stopped_at: None,
        })
    }

    /// Create an audio backend with the given config
    fn create_backend(
        config: &DeviceManagerConfig,
        tx: mpsc::UnboundedSender<Vec<i16>>,
    ) -> Result<Box<dyn AudioBackend>> {
        audio_backend::create_backend(config.backend_type, tx, &config.backend_config)
    }

    /// Start recording - recreates audio backend if needed.
    /// Includes retry logic for transient device failures.
    fn start(&mut self) -> Result<()> {
        const MAX_RETRIES: u32 = 3;
        const RETRY_DELAY: std::time::Duration = std::time::Duration::from_millis(500);

        // Clear stopped_at - we're starting again, so no longer idle
        if self.stopped_at.take().is_some() {
            debug!("DeviceManager: Cleared idle timer (restarting before timeout)");
        }

        // Clear the needs_recreate flag (we'll recreate anyway if backend is None)
        self.needs_recreate.swap(false, std::sync::atomic::Ordering::SeqCst);

        // Recreate backend if it was released (dropped after idle) or device changed
        if self.backend.is_none() {
            info!("DeviceManager: Creating audio backend...");

            // Retry backend creation with backoff
            let mut last_error = None;
            for attempt in 1..=MAX_RETRIES {
                match Self::create_backend(&self.config, self.audio_tx.clone()) {
                    Ok(backend) => {
                        self.backend = Some(backend);
                        info!("DeviceManager: Audio backend created");
                        break;
                    }
                    Err(e) if attempt < MAX_RETRIES => {
                        warn!("DeviceManager: Backend creation failed (attempt {}): {}, retrying...", attempt, e);
                        last_error = Some(e);
                        std::thread::sleep(RETRY_DELAY);
                    }
                    Err(e) => {
                        last_error = Some(e);
                    }
                }
            }

            if self.backend.is_none() {
                return Err(last_error.unwrap_or_else(|| anyhow::anyhow!("Failed to create audio backend")));
            }
        }

        if let Some(ref backend) = self.backend {
            backend.start()?;
        } else {
            return Err(anyhow::anyhow!("No audio backend available"));
        }
        Ok(())
    }

    /// Stop recording.
    fn stop(&mut self) -> Result<()> {
        if let Some(ref backend) = self.backend {
            backend.stop()?;

            if backend.releases_on_stop() {
                let timeout_secs = self.config.idle_release_timeout_secs;
                if timeout_secs == 0 {
                    self.backend = None;
                    self.stopped_at = None;
                    info!("DeviceManager: Audio backend released immediately");
                } else {
                    self.stopped_at = Some(Instant::now());
                    info!("DeviceManager: Audio stopped, will release after {}s idle", timeout_secs);
                }
            } else {
                self.stopped_at = None;
                info!("DeviceManager: Audio stopped (backend kept open for sharing)");
            }
        }
        Ok(())
    }

    /// Flush any buffered audio data from the backend.
    fn flush(&self) -> Result<()> {
        if let Some(ref backend) = self.backend {
            backend.flush()?;
        }
        Ok(())
    }

    /// Check if idle timeout has expired and release backend if so.
    fn check_idle_timeout(&mut self) -> bool {
        if let Some(stopped_at) = self.stopped_at {
            let idle_duration = stopped_at.elapsed();
            let timeout = Duration::from_secs(self.config.idle_release_timeout_secs);
            if idle_duration >= timeout {
                self.release();
                self.stopped_at = None;
                return true;
            }
        }
        false
    }

    /// Release the audio backend (drop streams, release microphone).
    fn release(&mut self) {
        if self.backend.take().is_some() {
            info!("DeviceManager: Audio backend released after idle timeout");
        }
    }

    /// Switch to a different audio input device. Takes effect on next recording start.
    fn set_device(&mut self, device_name: Option<String>) {
        info!("DeviceManager: Switching device to {:?}", device_name.as_deref().unwrap_or("Default"));
        self.config.backend_config.device_name = device_name;
        // Drop existing backend so next start() recreates with the new device
        self.backend.take();
        self.stopped_at = None;
    }

    /// Spawn a background task to watch for device changes
    fn spawn_device_watcher(&self) {
        let needs_recreate = Arc::clone(&self.needs_recreate);

        std::thread::spawn(move || {
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

/// Drain remaining samples from audio channel with timeout.
#[allow(dead_code)]
async fn drain_audio_channel(
    audio_rx: &Arc<Mutex<mpsc::UnboundedReceiver<Vec<i16>>>>,
    engine: &Arc<dyn TranscriptionEngine>,
    timeout_ms: u64,
) -> usize {
    let deadline = tokio::time::Instant::now() + Duration::from_millis(timeout_ms);
    let mut drained = 0;

    loop {
        if tokio::time::Instant::now() >= deadline {
            break;
        }

        match tokio::time::timeout(Duration::from_millis(10), async {
            let mut rx = audio_rx.lock().await;
            rx.recv().await
        })
        .await
        {
            Ok(Some(samples)) => {
                if let Err(e) = engine.process_audio(&samples) {
                    error!("Processing error during drain: {}", e);
                }
                drained += 1;
            }
            Ok(None) => break,  // Channel closed
            Err(_) => break,    // Timeout - no more data
        }
    }

    debug!("Drained {} audio chunks from channel", drained);
    drained
}

#[tokio::main]
pub async fn run() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    info!("Starting Parakeet dictation engine");

    let config = load_config().unwrap_or_else(|e| {
        warn!("Failed to load config: {}, using defaults", e);
        Config {
            daemon: DaemonConfig {
                audio_device: "default".to_string(),
                sample_rate: "16000".to_string(),
                model: default_model(),
                enable_acronyms: default_enable_acronyms(),
                enable_punctuation: default_enable_punctuation(),
                enable_grammar: default_enable_grammar(),
                enable_word_substitution: default_enable_word_substitution(),
                silence_threshold_db: default_silence_threshold_db(),
                debug_audio: default_debug_audio(),
                trailing_buffer_ms: default_trailing_buffer_ms(),
                audio_backend: default_audio_backend(),
                idle_release_timeout_secs: default_idle_release_timeout_secs(),
                media_resume_delay_ms: default_media_resume_delay_ms(),
                engine_idle_timeout_secs: default_engine_idle_timeout_secs(),
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

    info!("Config loaded: audio_device={}, sample_rate={}",
          config.daemon.audio_device, sample_rate);

    // Initialize user dictionary
    let user_dict = Arc::new(UserDictionary::new().unwrap_or_else(|e| {
        warn!("Failed to initialize user dictionary: {}, spell checking will use defaults only", e);
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

    // Initialize word substitution processor
    let word_sub = if config.daemon.enable_word_substitution {
        match WordSubstitutionProcessor::new(Some(Arc::clone(&user_dict))) {
            Ok(ws) => {
                info!("Word substitution processor initialized");
                Some(ws)
            }
            Err(e) => {
                warn!("Failed to initialize word substitution processor: {}, disabled", e);
                None
            }
        }
    } else {
        None
    };

    // Spawn file watcher for substitutions hot-reload
    if let Some(ref ws) = word_sub {
        let ws_watcher = ws.clone();
        tokio::spawn(async move {
            if let Err(e) = watch_substitution_file(ws_watcher).await {
                error!("Substitution file watcher error: {}", e);
            }
        });
    }

    // Parse model specification (Parakeet only)
    let model_spec = ModelSpec::parse(&config.daemon.model)
        .map_err(|e| anyhow::anyhow!("Invalid model '{}': {}", config.daemon.model, e))?;

    info!("Model: {}", model_spec);

    // Validate that configured model is available
    if !model_spec.is_available() {
        return Err(anyhow::anyhow!(
            "Model '{}' not found at {:?}. Check that the model is installed.",
            config.daemon.model,
            model_spec.model_path()
        ));
    }

    // Create shared health state
    let health_state = Arc::new(HealthState::new());

    // Spawn dedicated watchdog task — decoupled from the event loop so long typing/processing
    // operations don't starve the watchdog and cause systemd to kill us.
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(15));
        loop {
            interval.tick().await;
            if let Err(e) = notify(false, [(STATE_WATCHDOG, "1")].iter()) {
                debug!("Failed to send watchdog keepalive: {}", e);
            }
        }
    });

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

    // Parse audio backend type
    let backend_type = BackendType::from_str(&config.daemon.audio_backend)
        .unwrap_or_else(|| {
            warn!("Unknown audio backend '{}', using auto", config.daemon.audio_backend);
            BackendType::Auto
        });

    // Create DeviceManager with eager-loaded audio backend
    info!("Creating DeviceManager with pre-loaded audio backend...");
    let device_manager_config = DeviceManagerConfig {
        backend_type,
        backend_config: AudioBackendConfig {
            device_name: audio_device_name.clone(),
            sample_rate,
            silence_threshold,
        },
        idle_release_timeout_secs: config.daemon.idle_release_timeout_secs,
    };
    let mut device_manager = DeviceManager::new(device_manager_config, audio_tx)?;

    // Spawn device hotplug watcher
    device_manager.spawn_device_watcher();
    info!("Audio streams pre-loaded and ready (fast startup enabled)");

    let keyboard = Arc::new(KeyboardInjector::new());

    // Spawn integrated GUI
    info!("Spawning integrated GUI...");
    let gui_control_tx_gui = gui_control_tx.clone();
    let spectrum_tx_gui = spectrum_tx.clone();
    let runtime_handle = tokio::runtime::Handle::current();

    let _gui_handle = tokio::task::spawn_blocking(move || {
        slint_gui::run_integrated(
            gui_control_tx_gui,
            spectrum_tx_gui,
            gui_status_tx,
            runtime_handle,
        )
    });

    // Wait for GUI to initialize (with timeout)
    info!("Waiting for GUI to initialize...");
    let gui_available = match tokio::time::timeout(
        Duration::from_secs(5),
        gui_status_rx.recv()
    ).await {
        Ok(Some(GuiStatus::Ready)) => {
            info!("GUI ready");
            true
        }
        Ok(Some(GuiStatus::Error(e))) => {
            warn!("GUI initialization failed: {}", e);
            warn!("Continuing without GUI overlay - daemon will operate in headless mode");
            false
        }
        Ok(Some(GuiStatus::TransitionComplete { .. })) => {
            warn!("Unexpected TransitionComplete during init, assuming GUI unavailable");
            false
        }
        Ok(Some(GuiStatus::ShuttingDown)) => {
            warn!("GUI is shutting down during init, continuing without GUI");
            false
        }
        Ok(None) => {
            warn!("GUI status channel closed, continuing without GUI");
            false
        }
        Err(_) => {
            warn!("GUI failed to start within 5 seconds (possible compositor compatibility issue)");
            warn!("Continuing without GUI overlay - daemon will operate in headless mode");
            info!("You can still use voice-dictation start/stop/confirm commands normally");
            false
        }
    };

    health_state.gui_healthy.store(gui_available, Ordering::Relaxed);

    if !gui_available {
        info!("Running in headless mode (no visual overlay)");
    }

    // Pre-load engine at startup for instant recording start
    info!("Pre-loading Parakeet engine (blocking call before D-Bus)...");
    let mut preview_engine: Option<Arc<dyn TranscriptionEngine>> = Some(model_spec.create_engine(sample_rate)?);
    let mut engine_stopped_at: Option<Instant> = None;
    info!("Parakeet engine loaded and ready");

    // Mark engine as healthy after successful load
    health_state.engine_healthy.store(true, Ordering::Relaxed);

    // Create watch channel for state sharing with D-Bus
    let (state_tx, state_rx) = tokio::sync::watch::channel(DaemonState::Idle);

    // Create D-Bus service for control commands with health state
    let (dbus_conn, command_sender, mut command_rx) =
        dbus_control::create_dbus_service(state_rx, Arc::clone(&health_state)).await?;
    let _dbus_conn = dbus_conn; // Keep alive

    #[cfg(feature = "tray")]
    let _tray_handle = {
        let tray_tx = command_sender.lock().await.clone();
        let tray_rx = state_tx.subscribe();
        tray::spawn_tray(tray_rx, tray_tx, backend_type, audio_device_name.clone()).await
    };

    // Keep command_sender alive (used by D-Bus service)
    let _command_sender = command_sender;

    info!("Daemon initialized - entering idle state (GUI hidden)");

    // Notify systemd that we're ready
    if let Err(e) = notify(false, [(STATE_READY, "1")].iter()) {
        warn!("Failed to notify systemd (Ready): {}", e);
    }

    // State machine variables
    let mut daemon_state = DaemonState::Idle;
    let mut session: Option<RecordingSession> = None;
    let mut audio_task: Option<tokio::task::JoinHandle<()>> = None;
    let mut preview_task: Option<tokio::task::JoinHandle<()>> = None;
    let mut media_was_playing = false;
    let mut window_target: Option<window_target::WindowTarget> = None;
    // Cancellation channel for graceful task shutdown
    let (cancel_tx, _cancel_rx) = tokio::sync::watch::channel(false);

    // Audio health monitoring constants
    let _audio_health_timeout = Duration::from_secs(3);

    // ===== PERSISTENT STATE MACHINE LOOP =====
    loop {

        match daemon_state {
            DaemonState::Idle => {
                // Check for idle timeout (release mic if idle too long)
                if device_manager.check_idle_timeout() {
                    debug!("Idle timeout expired, mic released");
                }

                // Check engine idle timeout (release ORT sessions to reclaim BFCArena memory)
                if let Some(stopped_at) = engine_stopped_at {
                    let timeout = Duration::from_secs(config.daemon.engine_idle_timeout_secs);
                    if stopped_at.elapsed() >= timeout && preview_engine.is_some() {
                        info!("Engine idle timeout expired, releasing ORT sessions to free memory");
                        preview_engine = None;
                        engine_stopped_at = None;
                        health_state.engine_healthy.store(false, Ordering::Relaxed);
                    }
                }

                // Wait for D-Bus commands with timeout
                match tokio::time::timeout(Duration::from_millis(100), command_rx.recv()).await {
                    Ok(Some(cmd)) => match cmd {
                        DaemonCommand::StartRecording => {
                            info!("Received StartRecording command");
                            // Capture focused window before pausing media (to lock typing target)
                            window_target = window_target::WindowTarget::capture().await;
                            if let Some(ref wt) = window_target {
                                info!("Captured window target: class={}", wt.class());
                            }
                            media_was_playing = pause_media_if_playing();

                            // Drain any stale audio data from the channel before starting
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

                            // Mark audio as healthy at start
                            health_state.audio_healthy.store(true, Ordering::Relaxed);

                            // Recreate engine if it was released due to idle timeout
                            if preview_engine.is_none() {
                                info!("Recreating transcription engine (was released for idle memory savings)...");
                                preview_engine = Some(model_spec.create_engine(sample_rate)?);
                                health_state.engine_healthy.store(true, Ordering::Relaxed);
                                info!("Engine recreated and ready");
                            }
                            engine_stopped_at = None;

                            // Reset the pre-loaded engine for new session
                            let engine = preview_engine.as_ref().unwrap();
                            engine.reset();
                            let session_engine = Arc::clone(engine);

                            // Signal UI to show
                            gui_control_tx.send(GuiControl::SetListening)
                                .map_err(|e| anyhow::anyhow!("Failed to send SetListening: {}", e))?;

                            // Create session
                            session = Some(RecordingSession {
                                start_time: Instant::now(),
                                engine: Arc::clone(&session_engine),
                            });

                            // Reset cancellation flag for new session
                            let _ = cancel_tx.send(false);

                            // Notify for waking preview task when new audio arrives
                            let audio_notify = Arc::new(tokio::sync::Notify::new());

                            // Start audio processing task
                            let engine_clone = Arc::clone(&session_engine);
                            let spectrum_tx_clone = spectrum_tx.clone();
                            let audio_rx_clone = Arc::clone(&audio_rx_shared);
                            let mut cancel_rx = cancel_tx.subscribe();
                            let trailing_buffer_ms = config.daemon.trailing_buffer_ms;
                            let health_clone = Arc::clone(&health_state);
                            let audio_notify_tx = Arc::clone(&audio_notify);
                            audio_task = Some(tokio::spawn(async move {
                                let mut buffer = Vec::new();
                                let trailing_duration = Duration::from_millis(trailing_buffer_ms);
                                let mut trailing_deadline: Option<tokio::time::Instant> = None;

                                loop {
                                    // Check if trailing period has elapsed FIRST
                                    if let Some(deadline) = trailing_deadline {
                                        if tokio::time::Instant::now() >= deadline {
                                            debug!("Audio task: trailing capture complete");
                                            break;
                                        }
                                    }

                                    // Use select to allow graceful cancellation
                                    tokio::select! {
                                        biased;
                                        _ = cancel_rx.changed() => {
                                            if *cancel_rx.borrow() && trailing_deadline.is_none() {
                                                debug!("Audio task: cancellation received, starting trailing capture");
                                                trailing_deadline = Some(tokio::time::Instant::now() + trailing_duration);
                                            }
                                        }
                                        samples = async {
                                            let mut rx = audio_rx_clone.lock().await;
                                            rx.recv().await
                                        } => {
                                            match samples {
                                                Some(samples) => {
                                                    // Update health timestamp
                                                    let now_ms = std::time::SystemTime::now()
                                                        .duration_since(std::time::UNIX_EPOCH)
                                                        .unwrap_or_default()
                                                        .as_millis() as u64;
                                                    health_clone.last_audio_timestamp_ms.store(now_ms, Ordering::Relaxed);
                                                    health_clone.audio_healthy.store(true, Ordering::Relaxed);

                                                    let samples_f32: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
                                                    buffer.extend_from_slice(&samples_f32);

                                                    while buffer.len() >= 512 {
                                                        let chunk: Vec<f32> = buffer.drain(..512).collect();
                                                        let _ = spectrum_tx_clone.send(chunk);
                                                    }

                                                    if let Err(e) = engine_clone.process_audio(&samples) {
                                                        error!("Processing error: {}", e);
                                                    }
                                                    audio_notify_tx.notify_one();
                                                }
                                                None => break,
                                            }
                                        }
                                        _ = tokio::time::sleep(Duration::from_millis(10)), if trailing_deadline.is_some() => {
                                            // Periodic wake-up during trailing period to check deadline
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
                            let enable_word_substitution = config.daemon.enable_word_substitution;
                            let word_sub_preview = word_sub.clone();
                            let user_dict_preview = Arc::clone(&user_dict);
                            let mut cancel_rx_preview = cancel_tx.subscribe();
                            let audio_notify_rx = Arc::clone(&audio_notify);
                            preview_task = Some(tokio::spawn(async move {
                                let pipeline = Pipeline::from_config_with_dict(
                                    enable_acronyms,
                                    enable_punctuation,
                                    false,  // grammar disabled in preview for speed
                                    Some(user_dict_preview),
                                    enable_word_substitution,
                                    word_sub_preview,
                                );

                                let mut last_text = String::new();
                                let mut last_text_change = Instant::now();
                                const TEXT_SETTLED_THRESHOLD_MS: u64 = 300;
                                const MAX_PREVIEW_WAIT_MS: u64 = 200;

                                loop {
                                    tokio::select! {
                                        biased;
                                        _ = cancel_rx_preview.changed() => {
                                            if *cancel_rx_preview.borrow() {
                                                debug!("Preview task: cancellation received");
                                                break;
                                            }
                                        }
                                        _ = async {
                                            // Wake on new audio or after max wait (for settled detection)
                                            tokio::select! {
                                                _ = audio_notify_rx.notified() => {}
                                                _ = tokio::time::sleep(Duration::from_millis(MAX_PREVIEW_WAIT_MS)) => {}
                                            }
                                        } => {
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

                                                    let text_changed = text_processed != last_text;
                                                    if text_changed {
                                                        last_text = text_processed.clone();
                                                        last_text_change = Instant::now();
                                                    }

                                                    let text_settled = last_text_change.elapsed().as_millis() >= TEXT_SETTLED_THRESHOLD_MS as u128;
                                                    let is_speaking = !text_processed.is_empty() && !text_settled;

                                                    let _ = gui_control_tx_preview.send(GuiControl::UpdateTranscription {
                                                        text: text_processed,
                                                        is_final: false,
                                                    });

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
                        DaemonCommand::SwitchDevice(name) => {
                            info!("Switching audio device to {:?}", name.as_deref().unwrap_or("Default"));
                            device_manager.set_device(name);
                        }
                        DaemonCommand::Shutdown => {
                            info!("Received Shutdown command");
                            let _ = gui_control_tx.send(GuiControl::Exit);
                            break;
                        }
                        _ => {
                            warn!("Ignoring unexpected command in Idle state");
                        }
                    }
                    Ok(None) => {
                        error!("D-Bus command channel closed");
                        break;
                    }
                    Err(_) => {
                        // Timeout - continue loop
                    }
                }
            }

            DaemonState::Recording => {
                // Audio health monitoring: check if audio task has crashed
                if let Some(ref task) = audio_task {
                    if task.is_finished() {
                        error!("Audio task died unexpectedly - recovering to Idle");
                        health_state.audio_healthy.store(false, Ordering::Relaxed);
                        *health_state.last_error.write().await = Some("Audio task crashed during recording".to_string());

                        // Clean up
                        audio_task = None;
                        if let Some(task) = preview_task.take() {
                            let _ = cancel_tx.send(true);
                            let _ = task.await;
                        }
                        let _ = device_manager.stop();
                        let _ = gui_control_tx.send(GuiControl::SetHidden);
                        session = None;
                        daemon_state = DaemonState::Idle;
                        let _ = state_tx.send(daemon_state);
                        info!("Recovered to Idle state after audio task crash");
                        continue;
                    }
                }

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

                            // 1. Stop audio backends (pause streams)
                            let _ = device_manager.stop();

                            // 2. Flush backend buffers
                            let _ = device_manager.flush();

                            // 3. Signal audio task to start trailing period
                            let _ = cancel_tx.send(true);

                            // 4. Wait for tasks to finish (includes trailing buffer period)
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

                            let _ = device_manager.stop();
                            let _ = device_manager.flush();
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
                        DaemonCommand::SwitchDevice(name) => {
                            warn!("Device switch to {:?} requested during recording, will apply on next session",
                                  name.as_deref().unwrap_or("Default"));
                            device_manager.set_device(name);
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

                if media_was_playing {
                    media_was_playing = false;
                    let delay = config.daemon.media_resume_delay_ms;
                    tokio::spawn(async move {
                        tokio::time::sleep(Duration::from_millis(delay)).await;
                        resume_media();
                    });
                }

                // Send SetProcessing IMMEDIATELY before any blocking work (shows spinner)
                gui_control_tx.send(GuiControl::SetProcessing)
                    .map_err(|e| anyhow::anyhow!("Failed to send SetProcessing: {}", e))?;

                // 1. Stop audio backends (pause streams)
                let _ = device_manager.stop();

                // 2. Flush backend buffers
                let _ = device_manager.flush();

                // 3. Signal audio task to start trailing period
                let _ = cancel_tx.send(true);

                // 4. Wait for audio task to finish (includes trailing buffer period)
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

                // Check if any audio was captured
                let audio_buffer_len = session_engine.as_ref().get_audio_buffer().len();
                info!("Audio buffer contains {} samples", audio_buffer_len);

                if audio_buffer_len > 0 {
                    // Run final transcription on full buffer (including trailing audio)
                    let preview_text = session_engine.as_ref().get_final_result()
                        .unwrap_or_else(|e| {
                            warn!("Final transcription failed: {}, falling back to cached text", e);
                            session_engine.as_ref().get_cached_text()
                        });
                    info!("Transcription: '{}'", preview_text);

                    // Apply post-processing pipeline
                    let pipeline = Pipeline::from_config_with_dict(
                        config.daemon.enable_acronyms,
                        config.daemon.enable_punctuation,
                        config.daemon.enable_grammar,
                        Some(Arc::clone(&user_dict)),
                        config.daemon.enable_word_substitution,
                        word_sub.clone(),
                    );
                    let processed_result = pipeline.process(&preview_text)?;

                    if !pipeline.is_empty() && preview_text != processed_result {
                        info!("[Final] Processed: '{}'", processed_result);
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
                            preview_text: preview_text.clone(),
                            final_text: processed_result.clone(),
                            preview_engine: "parakeet".to_string(),
                            accurate_engine: "parakeet".to_string(),
                            same_model_used: true,
                        };
                        if let Err(e) = debug_audio::save_debug_audio(&audio_buffer, sample_rate, metadata) {
                            warn!("Failed to save debug audio: {}", e);
                        }
                    }

                    // Build per-app profile from captured window class
                    let profile = match &window_target {
                        Some(wt) => app_profile::AppProfile::from_window_class(wt.class()),
                        None => app_profile::AppProfile::for_category(window_detect::AppCategory::General),
                    };

                    let sanitizer = SanitizationProcessor::new(profile.sanitization.clone(), profile.category);
                    let sanitized_result = sanitizer.process(&processed_result)?;

                    // Copy to clipboard as backup (wl-copy for Wayland)
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

                    // Refocus original window before typing (handles window switches during recording)
                    if let Some(ref wt) = window_target {
                        wt.refocus().await.ok();
                    }

                    let expected_typing_secs = (sanitized_result.len() as u64 * profile.word_delay_ms) / 1000;
                    if expected_typing_secs > 15 {
                        warn!("Typing will take ~{}s ({} chars at {}ms/char) — text is already in clipboard if interrupted", expected_typing_secs, sanitized_result.len(), profile.word_delay_ms);
                    }
                    info!("Typing final text ({:?} mode, delay={}ms)...", profile.category, profile.word_delay_ms);
                    keyboard.type_text(&sanitized_result, profile.word_delay_ms).await?;
                    info!("Typed!");

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
                engine_stopped_at = Some(Instant::now());
                daemon_state = DaemonState::Idle;
                let _ = state_tx.send(daemon_state);
                info!("Processing complete - returned to Idle state");
            }
        }
    }

    info!("Daemon shutting down");
    Ok(())
}
