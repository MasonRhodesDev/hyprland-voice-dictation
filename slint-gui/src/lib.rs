//! Slint-based GUI overlay for voice dictation
//!
//! Uses layer-shika for Wayland layer-shell integration with Slint.
//! Single persistent shell with dynamic property updates for mode switching.

use dictation_types::{GuiControl, GuiState, GuiStatus};
use layer_shika::calloop::TimeoutAction;
use layer_shika::prelude::*;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use slint_interpreter::Value;
use std::env;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc};
use tracing::{debug, error, info, warn};

mod monitor;

pub use monitor::get_active_monitor_sync;

/// Shared state between channel listener and GUI
pub struct SharedState {
    pub gui_state: GuiState,
    pub transcription: String,
    pub spectrum_values: Vec<f32>,
    pub closing_progress: f32,
    pub fade: f32,
    pub pre_listening: bool,
}

impl Default for SharedState {
    fn default() -> Self {
        Self {
            gui_state: GuiState::Hidden,
            transcription: String::new(),
            spectrum_values: vec![0.0; 8],
            closing_progress: 0.0,
            fade: 1.0,
            pre_listening: false,
        }
    }
}

/// Get the UI config directory path: ~/.config/voice-dictation/ui/
fn get_ui_config_dir() -> Option<PathBuf> {
    env::var_os("HOME").map(|home| {
        let mut path = PathBuf::from(home);
        path.push(".config/voice-dictation/ui");
        path
    })
}

/// Resolve UI file path: ~/.config/voice-dictation/ui/{name}.slint or bundled default
fn resolve_ui_path(name: &str) -> String {
    if let Some(config_dir) = get_ui_config_dir() {
        let config_path = config_dir.join(format!("{}.slint", name));
        if config_path.exists() {
            return config_path.to_string_lossy().to_string();
        }
    }

    // Fall back to bundled UI files
    format!("ui/{}.slint", name)
}

/// Spawn file watcher for UI hot-reload
fn spawn_ui_file_watcher(reload_flag: Arc<AtomicBool>) {
    let Some(ui_dir) = get_ui_config_dir() else {
        info!("No UI config directory found, hot-reload disabled");
        return;
    };

    if !ui_dir.exists() {
        info!("UI config directory doesn't exist: {:?}, hot-reload disabled", ui_dir);
        return;
    }

    std::thread::spawn(move || {
        let reload_flag_clone = reload_flag.clone();
        let mut watcher: RecommendedWatcher = match notify::recommended_watcher(
            move |res: std::result::Result<notify::Event, notify::Error>| {
                if let Ok(event) = res {
                    // Only reload on modify/create events for .slint files
                    if event.kind.is_modify() || event.kind.is_create() {
                        let is_slint = event.paths.iter().any(|p| {
                            p.extension().map_or(false, |ext| ext == "slint")
                        });
                        if is_slint {
                            info!("UI file changed, triggering reload...");
                            reload_flag_clone.store(true, Ordering::SeqCst);
                        }
                    }
                }
            },
        ) {
            Ok(w) => w,
            Err(e) => {
                error!("Failed to create file watcher: {}", e);
                return;
            }
        };

        if let Err(e) = watcher.watch(&ui_dir, RecursiveMode::NonRecursive) {
            error!("Failed to watch UI directory {:?}: {}", ui_dir, e);
            return;
        }

        info!("Watching UI directory for changes: {:?}", ui_dir);

        // Keep thread alive to maintain watcher
        loop {
            std::thread::sleep(Duration::from_secs(60));
        }
    });
}

/// Type alias for our Result to avoid conflict with layer-shika's Result
pub type GuiResult<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Run GUI integrated with daemon (channel-based communication)
pub fn run_integrated(
    gui_control_tx: broadcast::Sender<GuiControl>,
    spectrum_tx: broadcast::Sender<Vec<f32>>,
    gui_status_tx: mpsc::Sender<GuiStatus>,
    runtime_handle: tokio::runtime::Handle,
) -> GuiResult<()> {
    info!("Starting slint-gui (integrated mode)");

    env::set_var("SLINT_BACKEND", "winit-femtovg");

    // Create shared state
    let shared_state = Arc::new(RwLock::new(SharedState::default()));

    // Create reload flag for hot-reload
    let reload_flag = Arc::new(AtomicBool::new(false));

    // Subscribe to channels
    let gui_control_rx = gui_control_tx.subscribe();
    let spectrum_rx = spectrum_tx.subscribe();

    // Spawn channel listener (runs in tokio runtime)
    spawn_channel_listener(
        gui_control_rx,
        spectrum_rx,
        shared_state.clone(),
        gui_status_tx.clone(),
        runtime_handle.clone(),
    );

    // Spawn active monitor listener (updates global state on monitor change)
    monitor::spawn_active_monitor_listener(None);

    // Spawn UI file watcher for hot-reload
    spawn_ui_file_watcher(reload_flag.clone());

    // Send ready signal
    if let Err(e) = gui_status_tx.blocking_send(GuiStatus::Ready) {
        error!("Failed to send ready status: {}", e);
    } else {
        info!("Sent Ready status to daemon");
    }

    // Run the single persistent shell with reload support
    run_shell(shared_state, reload_flag)?;

    Ok(())
}

/// Spawn channel listener that updates shared state
fn spawn_channel_listener(
    mut gui_control_rx: broadcast::Receiver<GuiControl>,
    mut spectrum_rx: broadcast::Receiver<Vec<f32>>,
    shared_state: Arc<RwLock<SharedState>>,
    gui_status_tx: mpsc::Sender<GuiStatus>,
    runtime_handle: tokio::runtime::Handle,
) {
    // Control message listener
    let state_clone = shared_state.clone();
    let status_tx = gui_status_tx.clone();
    runtime_handle.spawn(async move {
        loop {
            match gui_control_rx.recv().await {
                Ok(control) => {
                    if let Ok(mut state) = state_clone.write() {
                        let old_state = state.gui_state;
                        match control {
                            GuiControl::Initialize => {
                                state.gui_state = GuiState::Hidden;
                            }
                            GuiControl::SetHidden => {
                                state.gui_state = GuiState::Hidden;
                            }
                            GuiControl::SetListening => {
                                state.gui_state = GuiState::Listening;
                                state.fade = 1.0;
                                state.pre_listening = false;
                            }
                            GuiControl::UpdateTranscription { text, .. } => {
                                state.transcription = text;
                            }
                            GuiControl::UpdateSpectrum(values) => {
                                state.spectrum_values = values;
                            }
                            GuiControl::UpdateVadState { .. } => {
                                // VAD state handled elsewhere
                            }
                            GuiControl::SetProcessing => {
                                state.gui_state = GuiState::Processing;
                                state.fade = 1.0;
                            }
                            GuiControl::SetClosing => {
                                state.gui_state = GuiState::Closing;
                                state.closing_progress = 0.0;
                            }
                            GuiControl::Exit => {
                                info!("Received Exit command");
                                std::process::exit(0);
                            }
                        }

                        let new_state = state.gui_state;
                        if old_state != new_state {
                            debug!("State transition: {:?} -> {:?}", old_state, new_state);
                            let _ = status_tx.try_send(GuiStatus::TransitionComplete {
                                from: old_state,
                                to: new_state,
                            });
                        }
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    warn!("Control channel lagged by {} messages", n);
                }
                Err(broadcast::error::RecvError::Closed) => {
                    info!("Control channel closed");
                    break;
                }
            }
        }
    });

    // Spectrum listener
    let state_clone = shared_state.clone();
    runtime_handle.spawn(async move {
        loop {
            match spectrum_rx.recv().await {
                Ok(raw_samples) => {
                    let bands = compute_spectrum_bands(&raw_samples);
                    if let Ok(mut state) = state_clone.write() {
                        state.spectrum_values = bands;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => {}
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    });
}

/// Simple spectrum computation - 8 frequency bands from audio samples
fn compute_spectrum_bands(samples: &[f32]) -> Vec<f32> {
    let len = samples.len();
    if len == 0 {
        return vec![0.0; 8];
    }

    let chunk_size = len / 8;
    if chunk_size == 0 {
        return vec![0.0; 8];
    }

    let mut bands = Vec::with_capacity(8);

    for i in 0..8 {
        let start = i * chunk_size;
        let end = if i == 7 { len } else { (i + 1) * chunk_size };
        let chunk = &samples[start..end];

        // RMS energy
        let sum: f32 = chunk.iter().map(|&x| x * x).sum();
        let rms = (sum / chunk.len() as f32).sqrt();

        // Normalize to 0-1 range (15x multiplier for visible movement)
        let normalized = (rms * 15.0).min(1.0);
        bands.push(normalized);
    }

    bands
}

/// Convert GuiState to mode integer for Slint
fn state_to_mode(state: GuiState) -> i32 {
    match state {
        GuiState::Hidden => 0,
        GuiState::PreListening => 1,
        GuiState::Listening => 1,
        GuiState::Processing => 2,
        GuiState::Closing => 3,
    }
}

/// Exit code indicating UI reload requested (triggers systemd restart)
const EXIT_CODE_RELOAD: i32 = 64;

/// Run the single persistent shell with dynamic property updates
fn run_shell(shared_state: Arc<RwLock<SharedState>>, reload_flag: Arc<AtomicBool>) -> GuiResult<()> {
    let ui_file = resolve_ui_path("dictation");
    info!("Loading UI from: {}", ui_file);

    // Build the shell with the unified component
    // Use max dimensions to accommodate all modes
    // Create surfaces on all monitors, control visibility in timer callback
    let mut runtime = Shell::from_file(&ui_file)
        .surface("Dictation")
        .width(380)  // Listening mode is widest
        .height(90)  // Listening mode is tallest
        .anchor(AnchorEdges::empty().with_bottom())
        .margin((0, 0, 50, 0))
        .layer(Layer::Overlay)
        .keyboard_interactivity(KeyboardInteractivity::None)
        .output_policy(OutputPolicy::AllOutputs)  // Surfaces on all monitors
        .build()
        .map_err(|e| format!("Failed to create shell: {}", e))?;

    // Get event loop handle for scheduling updates
    let event_loop = runtime.event_loop_handle();

    // Set up periodic timer to sync shared state to component properties
    // This runs inside the event loop and can safely access the component
    let update_interval = Duration::from_millis(16); // ~60fps

    event_loop
        .add_timer(update_interval, move |_deadline: Instant, app_state| {
            // Check for UI file reload request (dev workflow)
            if reload_flag.load(Ordering::SeqCst) {
                info!("UI file changed, reloading shell...");
                reload_flag.store(false, Ordering::SeqCst);
                std::process::exit(EXIT_CODE_RELOAD);
            }

            // Get active monitor from Hyprland
            let active_monitor = monitor::get_active_monitor();

            if let Ok(state) = shared_state.read() {
                // Iterate all surfaces with their output handles
                for (key, surface_state) in app_state.surfaces_with_keys() {
                    let component = surface_state.component_instance();

                    // Determine if this surface is on the active monitor
                    let is_active = if let Some(ref active_name) = active_monitor {
                        if let Some(output_info) = app_state.get_output_info(key.output_handle) {
                            output_info.name()
                                .map(|name| name == active_name)
                                .unwrap_or(false)
                        } else {
                            false
                        }
                    } else {
                        // Fallback: show on primary if can't determine active
                        app_state.get_output_info(key.output_handle)
                            .map(|info| info.is_primary())
                            .unwrap_or(false)
                    };

                    // If not on active monitor, hide by setting mode=0
                    let mode = if is_active {
                        state_to_mode(state.gui_state)
                    } else {
                        0  // Hidden
                    };

                    if let Err(e) = component.set_property("mode", Value::Number(mode as f64)) {
                        debug!("Failed to set mode: {}", e);
                    }

                    // Only update other properties for active surface
                    if is_active {
                        // Update spectrum for listening mode
                        if state.gui_state == GuiState::Listening || state.gui_state == GuiState::PreListening {
                            // Convert spectrum values to a model
                            let spectrum_values: [Value; 8] = [
                                Value::Number(state.spectrum_values.get(0).copied().unwrap_or(0.0) as f64),
                                Value::Number(state.spectrum_values.get(1).copied().unwrap_or(0.0) as f64),
                                Value::Number(state.spectrum_values.get(2).copied().unwrap_or(0.0) as f64),
                                Value::Number(state.spectrum_values.get(3).copied().unwrap_or(0.0) as f64),
                                Value::Number(state.spectrum_values.get(4).copied().unwrap_or(0.0) as f64),
                                Value::Number(state.spectrum_values.get(5).copied().unwrap_or(0.0) as f64),
                                Value::Number(state.spectrum_values.get(6).copied().unwrap_or(0.0) as f64),
                                Value::Number(state.spectrum_values.get(7).copied().unwrap_or(0.0) as f64),
                            ];
                            if let Err(e) = component.set_property("spectrum", Value::Model(spectrum_values.into())) {
                                debug!("Failed to set spectrum: {}", e);
                            }

                            // Update transcription text
                            if let Err(e) = component.set_property("text", Value::String(state.transcription.clone().into())) {
                                debug!("Failed to set text: {}", e);
                            }

                            // Update pre-listening flag
                            if let Err(e) = component.set_property("pre-listening", Value::Bool(state.pre_listening)) {
                                debug!("Failed to set pre-listening: {}", e);
                            }
                        }

                        // Update fade
                        if let Err(e) = component.set_property("fade", Value::Number(state.fade as f64)) {
                            debug!("Failed to set fade: {}", e);
                        }

                        // Update closing progress
                        if state.gui_state == GuiState::Closing {
                            if let Err(e) = component.set_property("closing-progress", Value::Number(state.closing_progress as f64)) {
                                debug!("Failed to set closing-progress: {}", e);
                            }
                        }
                    }
                }
            }

            // Return ToDuration to reschedule the timer
            TimeoutAction::ToDuration(update_interval)
        })
        .map_err(|e| format!("Failed to add timer: {}", e))?;

    info!("Starting shell event loop");
    runtime.run().map_err(|e| format!("Shell run error: {}", e))?;

    Ok(())
}
