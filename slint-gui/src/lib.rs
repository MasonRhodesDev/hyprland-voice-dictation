//! Slint-based GUI overlay for voice dictation
//!
//! Uses layer-shika for Wayland layer-shell integration with Slint.
//! Single persistent shell with dynamic property updates for mode switching.

use dictation_types::{GuiControl, GuiState, GuiStatus};
use hypr_slint_runtime::{IdleScheduler, WaitDecision, WakeHandle};
use layer_shika::prelude::*;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use slint_interpreter::Value;
use std::env;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
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
    /// Typing progress (0.0–1.0) while gui_state == Typing.
    pub typing_progress: f32,
}

/// Condvar paired with the shared runtime wake. Producers call `signal()` after
/// changing state. The UI loop wakes immediately, while the clock thread is only
/// active for visible animation or a real maintenance deadline.
struct UiClock {
    wake: WakeHandle,
    generation: Mutex<u64>,
    changed: Condvar,
    empty_since: Mutex<Option<Instant>>,
}

impl UiClock {
    fn new(wake: WakeHandle) -> Arc<Self> {
        Arc::new(Self {
            wake,
            generation: Mutex::new(0),
            changed: Condvar::new(),
            empty_since: Mutex::new(None),
        })
    }

    fn signal(&self) {
        self.wake.wake();
        let mut generation = self.generation.lock().expect("UI clock poisoned");
        *generation = generation.wrapping_add(1);
        self.changed.notify_one();
    }

    fn note_surface_count(&self, count: usize) {
        let mut empty_since = self.empty_since.lock().expect("surface deadline poisoned");
        match (count, empty_since.is_some()) {
            (0, false) => *empty_since = Some(Instant::now()),
            (1.., true) => *empty_since = None,
            _ => {}
        }
        drop(empty_since);
        self.changed.notify_one();
    }

    fn next_maintenance(&self, now: Instant) -> Option<Duration> {
        let topology = monitor::pending_compositor_change()
            .map(|changed| changed + COMPOSITOR_SETTLE)
            .map(|deadline| deadline.saturating_duration_since(now));
        let surfaces = *self.empty_since.lock().expect("surface deadline poisoned");
        let surfaces = surfaces
            .map(|empty| empty + SURFACE_LOSS_GRACE)
            .map(|deadline| deadline.saturating_duration_since(now));
        [topology, surfaces].into_iter().flatten().min()
    }
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
            typing_progress: 0.0,
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
fn spawn_ui_file_watcher(reload_flag: Arc<AtomicBool>, clock: Arc<UiClock>) {
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
        let clock_clone = clock.clone();
        let mut watcher: RecommendedWatcher = match notify::recommended_watcher(
            move |res: std::result::Result<notify::Event, notify::Error>| {
                if let Ok(event) = res {
                    // Only reload on modify/create events for .slint files
                    if event.kind.is_modify() || event.kind.is_create() {
                        let is_slint = event
                            .paths
                            .iter()
                            .any(|p| p.extension().is_some_and(|ext| ext == "slint"));
                        if is_slint {
                            info!("UI file changed, triggering reload...");
                            reload_flag_clone.store(true, Ordering::SeqCst);
                            clock_clone.signal();
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

    // Don't set SLINT_BACKEND - layer-shika uses slint-interpreter which doesn't need it
    // env::set_var("SLINT_BACKEND", "winit-femtovg");

    // Create shared state
    let shared_state = Arc::new(RwLock::new(SharedState::default()));

    let (wake_sender, wake_receiver) = layer_shika::calloop::channel::channel();
    let wake = WakeHandle::new(move || {
        let _ = wake_sender.send(());
    });
    let clock = UiClock::new(wake);
    let mut wake_receiver = Some(wake_receiver);

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
        clock.clone(),
    );

    // Spawn active monitor listener (updates global state on monitor change)
    let monitor_clock = clock.clone();
    monitor::spawn_active_monitor_listener(Arc::new(move || monitor_clock.signal()));

    // Spawn UI file watcher for hot-reload
    spawn_ui_file_watcher(reload_flag.clone(), clock.clone());

    spawn_ui_clock(shared_state.clone(), clock.clone());

    // Run the single persistent shell with reload support.
    // Shell creation can fail transiently — most commonly right after resume or
    // a monitor hotplug, when the compositor briefly reports zero outputs. Retry
    // with backoff instead of leaving the daemon permanently headless.
    info!("Creating Wayland layer shell (this may take a few seconds)...");
    let mut backoff = Duration::from_secs(1);
    loop {
        match run_shell(
            shared_state.clone(),
            reload_flag.clone(),
            gui_status_tx.clone(),
            clock.clone(),
            &mut wake_receiver,
        ) {
            Ok(()) => return Ok(()),
            Err(e) => {
                let msg = e.to_string();
                // Slint's platform can only be installed once per process. If
                // shell creation failed after installing it, retrying in-process
                // can never succeed — escalate to a process restart via systemd.
                if msg.to_lowercase().contains("platform") {
                    error!(
                        "Shell creation failed after Slint platform init ({msg}); \
                         exiting for systemd restart"
                    );
                    std::process::exit(EXIT_CODE_SURFACES_LOST);
                }

                error!("Failed to create shell: {msg}; retrying in {}s", backoff.as_secs());
                let _ = gui_status_tx.blocking_send(GuiStatus::Error(format!(
                    "Wayland layer shell initialization failed: {msg}. Retrying in {}s.",
                    backoff.as_secs()
                )));
                std::thread::sleep(backoff);
                backoff = (backoff * 2).min(Duration::from_secs(30));
            }
        }
    }
}

/// Spawn channel listener that updates shared state
fn spawn_channel_listener(
    mut gui_control_rx: broadcast::Receiver<GuiControl>,
    mut spectrum_rx: broadcast::Receiver<Vec<f32>>,
    shared_state: Arc<RwLock<SharedState>>,
    gui_status_tx: mpsc::Sender<GuiStatus>,
    runtime_handle: tokio::runtime::Handle,
    clock: Arc<UiClock>,
) {
    // Control message listener
    let state_clone = shared_state.clone();
    let status_tx = gui_status_tx.clone();
    let control_clock = clock.clone();
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
                            GuiControl::SetTranscribing => {
                                state.gui_state = GuiState::Transcribing;
                                state.fade = 1.0;
                                state.typing_progress = 0.0;
                            }
                            GuiControl::SetTyping { done, total } => {
                                state.gui_state = GuiState::Typing;
                                state.fade = 1.0;
                                state.typing_progress = if total > 0 {
                                    (done as f32 / total as f32).clamp(0.0, 1.0)
                                } else {
                                    0.0
                                };
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
                    control_clock.signal();
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
    let spectrum_clock = clock;
    runtime_handle.spawn(async move {
        loop {
            match spectrum_rx.recv().await {
                Ok(raw_samples) => {
                    let bands = compute_spectrum_bands(&raw_samples);
                    if let Ok(mut state) = state_clone.write() {
                        state.spectrum_values = bands;
                    }
                    spectrum_clock.signal();
                }
                Err(broadcast::error::RecvError::Lagged(_)) => {}
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    });
}

fn state_has_visible_animation(state: GuiState) -> bool {
    matches!(state, GuiState::Transcribing | GuiState::Typing | GuiState::Closing)
}

/// Run the presentation clock off the UI thread. It sleeps indefinitely while
/// hidden/static, wakes on producer events, and emits frame wakes only while a
/// visible Rust-driven animation is active.
fn spawn_ui_clock(shared_state: Arc<RwLock<SharedState>>, clock: Arc<UiClock>) {
    std::thread::Builder::new()
        .name("voice-ui-clock".into())
        .spawn(move || {
            let scheduler = IdleScheduler::default();
            let mut observed_generation = 0;
            loop {
                let animated = shared_state
                    .read()
                    .map(|state| state_has_visible_animation(state.gui_state))
                    .unwrap_or(false);
                let decision = scheduler.decide(animated, clock.next_maintenance(Instant::now()));

                let mut generation = clock.generation.lock().expect("UI clock poisoned");
                if *generation != observed_generation {
                    observed_generation = *generation;
                    continue;
                }

                match decision {
                    WaitDecision::Indefinite => {
                        generation = clock.changed.wait(generation).expect("UI clock poisoned");
                        observed_generation = *generation;
                    }
                    WaitDecision::Frame(delay) | WaitDecision::Timer(delay) => {
                        let (next, timeout) = clock
                            .changed
                            .wait_timeout(generation, delay)
                            .expect("UI clock poisoned");
                        generation = next;
                        observed_generation = *generation;
                        if timeout.timed_out() {
                            drop(generation);
                            clock.signal();
                        }
                    }
                }
            }
        })
        .expect("failed to start UI clock");
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
        // Both spinner states map to mode 2; the Typing view also reads `progress`.
        GuiState::Transcribing => 2,
        GuiState::Typing => 2,
        GuiState::Closing => 3,
    }
}

/// Exit code indicating UI reload requested (triggers systemd restart)
const EXIT_CODE_RELOAD: i32 = 64;

/// Exit code when all layer surfaces are lost (triggers systemd restart via Restart=on-failure)
const EXIT_CODE_SURFACES_LOST: i32 = 1;

/// Time to wait before exiting after all surfaces are lost. Generous on purpose:
/// during resume or DP re-enumeration all outputs
/// can disappear for several seconds and layer-shika recreates the surfaces when
/// they come back — exiting early just restarts the daemon into the same churn.
const SURFACE_LOSS_GRACE: Duration = Duration::from_secs(10);

/// How long compositor topology events must be quiet before we verify that every
/// monitor has an overlay surface (and restart only if one is missing).
const COMPOSITOR_SETTLE: Duration = Duration::from_secs(3);

/// Properties the timer callback drives on the active surface. The UI file (which
/// may be a user-customized copy under ~/.config) must declare all of these as
/// inputs, or those updates silently no-op. We check this once at startup so a
/// drifted UI fails loudly instead of mysteriously (frozen spinner, dead progress).
const REQUIRED_UI_PROPERTIES: &[&str] = &[
    "mode",
    "spinner-angle",
    "progress",
    "fade",
    "spectrum",
    "text",
    "pre-listening",
    "closing-progress",
];

/// Compile the resolved UI file and verify it declares every property the Rust
/// timer drives. Returns a human-readable error describing the drift, if any.
fn validate_ui_contract(ui_file: &str) -> std::result::Result<(), String> {
    let compiler = slint_interpreter::Compiler::default();
    let result = tokio::runtime::Builder::new_current_thread()
        .build()
        .map_err(|e| format!("could not start UI validation runtime: {e}"))?
        .block_on(compiler.build_from_path(ui_file));

    let errors: Vec<String> = result
        .diagnostics()
        .filter(|d| d.level() == slint_interpreter::DiagnosticLevel::Error)
        .map(|d| d.to_string())
        .collect();
    if !errors.is_empty() {
        return Err(format!("UI '{ui_file}' failed to compile: {}", errors.join("; ")));
    }

    let component =
        result.components().next().ok_or_else(|| format!("UI '{ui_file}' defines no component"))?;

    // Normalize '-'/'_' since Slint treats them interchangeably in identifiers.
    let present: std::collections::HashSet<String> =
        component.properties().map(|(name, _)| name.replace('-', "_")).collect();
    let missing: Vec<&str> = REQUIRED_UI_PROPERTIES
        .iter()
        .copied()
        .filter(|p| !present.contains(&p.replace('-', "_")))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "UI '{ui_file}' is missing required input properties {missing:?} — it has drifted \
             from the Rust property contract. Add them (see slint-gui/ui/dictation.slint)."
        ));
    }
    Ok(())
}

/// Run the single persistent shell with dynamic property updates
fn run_shell(
    shared_state: Arc<RwLock<SharedState>>,
    reload_flag: Arc<AtomicBool>,
    gui_status_tx: mpsc::Sender<GuiStatus>,
    clock: Arc<UiClock>,
    wake_receiver: &mut Option<layer_shika::calloop::channel::Channel<()>>,
) -> GuiResult<()> {
    let ui_file = resolve_ui_path("dictation");
    info!("Loading UI from: {}", ui_file);

    // Fail loud on UI contract drift (e.g. a customized config copy missing newer
    // properties) instead of silently no-op'ing property updates. We still proceed
    // with degraded rendering rather than crash-looping the daemon. Log only — a
    // GuiStatus::Error here would make the daemon mark a GUI that goes on to run
    // fine as permanently unavailable.
    if let Err(e) = validate_ui_contract(&ui_file) {
        error!("UI contract validation failed: {e}");
    } else {
        info!("UI contract OK ({} required properties present)", REQUIRED_UI_PROPERTIES.len());
    }

    // Build the shell with the unified component
    // Use max dimensions to accommodate all modes
    // Create surfaces on all monitors, control visibility in timer callback
    info!("Creating Shell from UI file...");
    let mut runtime = Shell::from_file(&ui_file)
        .surface("Dictation")
        .width(380) // Listening mode is widest
        .height(90) // Listening mode is tallest
        .anchor(AnchorEdges::empty().with_bottom())
        .margin((0, 0, 50, 0))
        .layer(Layer::Overlay)
        .keyboard_interactivity(KeyboardInteractivity::None)
        .output_policy(OutputPolicy::AllOutputs) // Surfaces on all monitors
        .build()
        .map_err(|e| format!("Failed to create shell: {}", e))?;

    info!("Shell created successfully");

    // Send Ready signal BEFORE starting the event loop
    if let Err(e) = gui_status_tx.blocking_send(GuiStatus::Ready) {
        error!("Failed to send ready status: {}", e);
    } else {
        info!("Sent Ready status to daemon - GUI is operational");
    }

    // Get event loop handle for scheduling updates
    info!("Getting event loop handle...");
    let event_loop = runtime.event_loop_handle();
    info!("Got event loop handle");

    let mut gui_initialized = false;
    // Last gui_state we emitted a log line for, so we log transitions instead of every frame.
    let mut last_logged_state: Option<GuiState> = None;
    // Spinner rotation, advanced here each tick (capped at the timer rate) instead of via a
    // free-running Slint animation, so a long/stuck spinner state can't drive uncapped repaints.
    let mut spinner_angle: f32 = 0.0;
    let mut last_frame = Instant::now();

    let wake_for_callback = clock.wake.clone();
    let clock_for_callback = clock.clone();
    let wake_receiver = wake_receiver.take().ok_or("UI wake source was already consumed")?;
    event_loop
        .insert_source(wake_receiver, move |event, (), app_state| {
            if !matches!(event, layer_shika::calloop::channel::Event::Msg(())) {
                return;
            }
            // Acknowledge the consumed edge before draining state so a concurrent
            // producer cannot be coalesced into an edge already removed by calloop.
            wake_for_callback.acknowledge();

            // Check for UI file reload request (dev workflow)
            if reload_flag.load(Ordering::SeqCst) {
                info!("UI file changed, reloading shell...");
                reload_flag.store(false, Ordering::SeqCst);
                std::process::exit(EXIT_CODE_RELOAD);
            }

            // After compositor topology changes (monitor add/remove, config
            // reload) settle, verify that every monitor has an overlay surface.
            // layer-shika reconciles surfaces on Wayland output hotplug itself;
            // restarting the process is the fallback, not the first response.
            if let Some(changed_at) = monitor::pending_compositor_change() {
                if changed_at.elapsed() >= COMPOSITOR_SETTLE {
                    monitor::clear_compositor_change();
                    if let Some(expected) = monitor::get_monitor_names_sync() {
                        let present: std::collections::HashSet<String> = app_state
                            .surfaces_with_keys()
                            .filter_map(|(key, _)| {
                                app_state
                                    .get_output_info(key.output_handle)
                                    .and_then(|info| info.name().map(|n| n.to_string()))
                            })
                            .collect();
                        let missing: Vec<&String> =
                            expected.iter().filter(|n| !present.contains(*n)).collect();
                        if missing.is_empty() {
                            info!(
                                "Compositor change reconciled in place ({} surfaces, monitors {:?})",
                                present.len(),
                                expected
                            );
                        } else {
                            error!(
                                "Monitors {:?} have no overlay surface after compositor change, \
                                 exiting for systemd restart",
                                missing
                            );
                            std::process::exit(EXIT_CODE_SURFACES_LOST);
                        }
                    } else {
                        debug!("Hyprland IPC unavailable, skipping surface verification");
                    }
                }
            }

            // Detect lost layer surfaces and exit for systemd restart
            let surface_count = app_state.surfaces_with_keys().count();
            if surface_count > 0 {
                gui_initialized = true;
                clock_for_callback.note_surface_count(surface_count);
            } else if gui_initialized {
                clock_for_callback.note_surface_count(0);
                let expired = clock_for_callback
                    .empty_since
                    .lock()
                    .expect("surface deadline poisoned")
                    .is_some_and(|since| since.elapsed() >= SURFACE_LOSS_GRACE);
                if expired {
                    error!(
                        "All layer surfaces lost for ~{}s after init, exiting for systemd restart",
                        SURFACE_LOSS_GRACE.as_secs()
                    );
                    std::process::exit(EXIT_CODE_SURFACES_LOST);
                }
            }

            // Get active monitor from Hyprland
            let active_monitor = monitor::get_active_monitor();

            if let Ok(state) = shared_state.read() {
                if state_has_visible_animation(state.gui_state) {
                    let now = Instant::now();
                    spinner_angle =
                        (spinner_angle + 360.0 * now.duration_since(last_frame).as_secs_f32())
                            % 360.0;
                    last_frame = now;
                } else {
                    last_frame = Instant::now();
                }
                // Log state transitions only (not every frame) to avoid flooding the journal.
                if last_logged_state != Some(state.gui_state) {
                    if state.gui_state != GuiState::Hidden {
                        info!("GUI state -> {:?} (active_monitor={:?})", state.gui_state, active_monitor);
                    } else {
                        debug!("GUI state -> Hidden");
                    }
                    last_logged_state = Some(state.gui_state);
                }

                // Graceful degradation: show on all monitors when detection unavailable
                let use_all_monitors = active_monitor.is_none()
                    || active_monitor.as_ref().is_some_and(|s| s.is_empty());
                if use_all_monitors && state.gui_state != GuiState::Hidden {
                    debug!("Monitor detection unavailable, showing GUI on all monitors");
                }

                // Iterate all surfaces with their output handles
                for (key, surface_state) in app_state.surfaces_with_keys() {
                    let component = surface_state.component_instance();

                    // Determine if this surface is on the active monitor
                    let output_name = app_state.get_output_info(key.output_handle)
                        .and_then(|info| info.name().map(|n| n.to_string()));

                    let is_active = if use_all_monitors {
                        // Show on all monitors when detection unavailable
                        state.gui_state != GuiState::Hidden
                    } else if let Some(ref active_name) = active_monitor {
                        // Normal behavior: only show on active monitor
                        output_name.as_ref()
                            .map(|name| name == active_name)
                            .unwrap_or(false)
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
                        // Drive the spinner rotation and typing progress (Slint only repaints
                        // when the visible mode actually references these properties).
                        if let Err(e) = component.set_property("spinner-angle", Value::Number(spinner_angle as f64)) {
                            debug!("Failed to set spinner-angle: {}", e);
                        }
                        if let Err(e) = component.set_property("progress", Value::Number(state.typing_progress as f64)) {
                            debug!("Failed to set progress: {}", e);
                        }

                        // Update spectrum for listening mode
                        if state.gui_state == GuiState::Listening || state.gui_state == GuiState::PreListening {
                            // Convert spectrum values to a model
                            let spectrum_values: [Value; 8] = [
                                Value::Number(state.spectrum_values.first().copied().unwrap_or(0.0) as f64),
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

        })
        .map_err(|e| format!("Failed to add UI wake source: {}", e))?;
    clock.signal();

    info!("Event-driven UI wake source ready; starting shell event loop");
    info!("Starting shell event loop");
    runtime.run().map_err(|e| format!("Shell run error: {}", e))?;

    // If we get here, the event loop exited (Wayland connection broken or signal received).
    // Exit the process so systemd can restart us with fresh surfaces.
    error!("Shell event loop exited unexpectedly, exiting for systemd restart");
    std::process::exit(EXIT_CODE_SURFACES_LOST);
}

#[cfg(test)]
mod idle_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn hidden_and_event_driven_listening_do_not_run_a_frame_clock() {
        let scheduler = IdleScheduler::default();
        for state in [GuiState::Hidden, GuiState::PreListening, GuiState::Listening] {
            assert!(!state_has_visible_animation(state));
            assert_eq!(scheduler.decide(false, None), WaitDecision::Indefinite);
        }
    }

    #[test]
    fn spinner_states_request_frames_and_hidden_defrosts_on_one_edge() {
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = calls.clone();
        let wake = WakeHandle::new(move || {
            seen.fetch_add(1, Ordering::SeqCst);
        });
        assert!(wake.wake());
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // Consuming/acknowledging the edge permits the next state change to wake.
        assert!(wake.acknowledge());
        assert!(wake.wake());
        assert_eq!(calls.load(Ordering::SeqCst), 2);

        let scheduler = IdleScheduler::new(Duration::from_millis(16));
        for state in [GuiState::Transcribing, GuiState::Typing, GuiState::Closing] {
            assert!(state_has_visible_animation(state));
            assert_eq!(
                scheduler.decide(true, None),
                WaitDecision::Frame(Duration::from_millis(16))
            );
        }
    }

    #[test]
    fn maintenance_deadlines_do_not_become_hidden_polling() {
        let clock = UiClock::new(WakeHandle::new(|| {}));
        assert_eq!(clock.next_maintenance(Instant::now()), None);

        clock.note_surface_count(0);
        let remaining = clock.next_maintenance(Instant::now()).expect("surface deadline");
        assert!(remaining <= SURFACE_LOSS_GRACE);
        assert!(remaining > Duration::from_secs(9));

        clock.note_surface_count(1);
        assert_eq!(clock.next_maintenance(Instant::now()), None);
    }
}
