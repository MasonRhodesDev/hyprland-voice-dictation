use tracing::{error, info};

pub mod animation;
pub mod animations;
pub mod channel_listener;
pub mod collapse_widget;
pub mod config;
pub mod fft;
pub mod layout;
pub mod monitor_detection;
pub mod per_monitor_window;
pub mod renderer;
pub mod renderer_v2;
pub mod shared_state;
pub mod spectrum_widget;
pub mod spinner_widget;
pub mod text_renderer;
pub mod wayland;

pub const SAMPLE_RATE: u32 = 16000;

pub fn run() -> Result<(), iced_layershell::Error> {
    let log_level = std::env::var("GUI_LOG").unwrap_or_else(|_| "error".to_string()).to_lowercase();

    let filter = match log_level.as_str() {
        "silent" => tracing::Level::ERROR,
        "error" => tracing::Level::ERROR,
        "warning" | "warn" => tracing::Level::WARN,
        "info" => tracing::Level::INFO,
        "debug" => tracing::Level::DEBUG,
        "verbose" | "trace" => tracing::Level::TRACE,
        _ => tracing::Level::ERROR,
    };

    tracing_subscriber::fmt().with_max_level(filter).init();

    info!("Starting dictation-gui with multi-monitor support");

    // Create shared state
    let shared_state = shared_state::SharedState::new();

    // Spawn Hyprland event listener
    info!("Spawning Hyprland event listener");
    monitor_detection::spawn_active_monitor_listener(shared_state.clone());

    // Enumerate monitors
    info!("Enumerating monitors...");
    let monitors = match monitor_detection::enumerate_monitors() {
        Ok(monitors) => {
            if monitors.is_empty() {
                tracing::error!("No monitors detected! Exiting.");
                std::process::exit(1);
            }
            monitors
        }
        Err(e) => {
            tracing::error!("Failed to enumerate monitors: {}. Exiting.", e);
            std::process::exit(1);
        }
    };

    info!("Detected {} monitor(s): {:?}", monitors.len(), monitors);

    // Spawn a window thread for each monitor
    let monitor_count = monitors.len();
    let mut handles = Vec::new();

    for (idx, monitor_name) in monitors.into_iter().enumerate() {
        let state_clone = shared_state.clone();
        let monitor_clone = monitor_name.clone();

        info!("Spawning window thread for monitor: {}", monitor_name);

        let handle = std::thread::spawn(move || {
            if let Err(e) = per_monitor_window::run_monitor_window(monitor_clone.clone(), state_clone) {
                tracing::error!("Window thread for monitor {} failed: {}", monitor_clone, e);
            }
        });

        handles.push(handle);

        // Brief delay between spawns to avoid race conditions
        if idx < monitor_count - 1 {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }

    info!("All monitor windows spawned, waiting for threads...");

    // Wait for all threads (they run indefinitely until exit)
    for handle in handles {
        let _ = handle.join();
    }

    Ok(())
}

/// Run GUI integrated with daemon (channel-based communication)
pub fn run_integrated(
    gui_control_rx: tokio::sync::broadcast::Receiver<dictation_types::GuiControl>,
    spectrum_rx: tokio::sync::broadcast::Receiver<Vec<f32>>,
    gui_status_tx: tokio::sync::mpsc::Sender<dictation_types::GuiStatus>,
) -> Result<(), iced_layershell::Error> {
    // Note: tracing subscriber is already initialized by the daemon in integrated mode
    // No need to initialize it again here

    info!("Starting dictation-gui (integrated mode) with multi-monitor support");

    // Create shared state
    let shared_state = shared_state::SharedState::new();

    // Spawn channel listeners (replaces background_tasks)
    info!("Spawning channel listeners");
    channel_listener::spawn_channel_listener(
        gui_control_rx,
        spectrum_rx,
        shared_state.clone(),
        gui_status_tx.clone(),
    );

    // Spawn Hyprland event listener
    info!("Spawning Hyprland event listener");
    monitor_detection::spawn_active_monitor_listener(shared_state.clone());

    // Enumerate monitors
    info!("Enumerating monitors...");
    let monitors = match monitor_detection::enumerate_monitors() {
        Ok(monitors) => {
            if monitors.is_empty() {
                tracing::error!("No monitors detected! Exiting.");
                let _ = gui_status_tx.blocking_send(dictation_types::GuiStatus::Error(
                    "No monitors detected".to_string(),
                ));
                std::process::exit(1);
            }
            monitors
        }
        Err(e) => {
            tracing::error!("Failed to enumerate monitors: {}. Exiting.", e);
            let _ = gui_status_tx.blocking_send(dictation_types::GuiStatus::Error(
                format!("Failed to enumerate monitors: {}", e),
            ));
            std::process::exit(1);
        }
    };

    info!("Detected {} monitor(s): {:?}", monitors.len(), monitors);

    // Spawn a window thread for each monitor
    let monitor_count = monitors.len();
    let mut handles = Vec::new();

    for (idx, monitor_name) in monitors.into_iter().enumerate() {
        let state_clone = shared_state.clone();
        let monitor_clone = monitor_name.clone();

        info!("Spawning window thread for monitor: {}", monitor_name);

        let handle = std::thread::spawn(move || {
            if let Err(e) = per_monitor_window::run_monitor_window(monitor_clone.clone(), state_clone) {
                tracing::error!("Window thread for monitor {} failed: {}", monitor_clone, e);
            }
        });

        handles.push(handle);

        // Brief delay between spawns to avoid race conditions
        if idx < monitor_count - 1 {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }

    info!("All monitor windows spawned");

    // Send ready signal to daemon
    if let Err(e) = gui_status_tx.blocking_send(dictation_types::GuiStatus::Ready) {
        error!("Failed to send ready status: {}", e);
    } else {
        info!("Sent Ready status to daemon");
    }

    info!("Waiting for threads...");

    // Wait for all threads (they run indefinitely until exit)
    for handle in handles {
        let _ = handle.join();
    }

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuiState {
    Hidden,
    PreListening,
    Listening,
    Processing,
    Closing,
}
