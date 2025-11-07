use smithay_client_toolkit::{
    delegate_output, delegate_registry,
    output::{OutputHandler, OutputState},
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
};
use std::sync::{Arc, RwLock};
use wayland_client::{
    globals::registry_queue_init, protocol::wl_output, Connection, QueueHandle,
};
use hyprland::event_listener::{EventListener, MonitorEventData};
use tracing::{debug, info, warn, error};

use crate::shared_state::SharedState;

/// Manages monitor detection via Wayland and active monitor tracking via Hyprland
pub struct MonitorDetector {
    registry_state: RegistryState,
    output_state: OutputState,
    pub detected_monitors: Vec<String>,
}

impl OutputHandler for MonitorDetector {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }

    fn new_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        if let Some(info) = self.output_state.info(&output) {
            let name = info.name.clone().unwrap_or_else(|| "Unknown".to_string());
            info!("Monitor detected: {} ({})", name, info.model);
            self.detected_monitors.push(name);
        }
    }

    fn update_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }

    fn output_destroyed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        if let Some(info) = self.output_state.info(&output) {
            if let Some(name) = &info.name {
                info!("Monitor removed: {}", name);
                self.detected_monitors.retain(|m| m != name);
            }
        }
    }
}

impl ProvidesRegistryState for MonitorDetector {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState];
}

delegate_output!(MonitorDetector);
delegate_registry!(MonitorDetector);

/// Enumerate all connected monitors via Wayland
pub fn enumerate_monitors() -> anyhow::Result<Vec<String>> {
    info!("Enumerating monitors via Wayland...");
    let conn = Connection::connect_to_env()?;
    let (globals, mut event_queue) = registry_queue_init(&conn)?;
    let qh = event_queue.handle();

    let mut detector = MonitorDetector {
        registry_state: RegistryState::new(&globals),
        output_state: OutputState::new(&globals, &qh),
        detected_monitors: Vec::new(),
    };

    // Process initial events to discover monitors
    event_queue.roundtrip(&mut detector)?;

    info!("Detected {} monitor(s): {:?}", detector.detected_monitors.len(), detector.detected_monitors);

    Ok(detector.detected_monitors)
}

/// Spawn a background task to listen for Hyprland active monitor events
/// Updates the shared state when the active monitor changes
pub fn spawn_active_monitor_listener(shared_state: Arc<RwLock<SharedState>>) {
    std::thread::spawn(move || {
        info!("Starting Hyprland active monitor event listener");

        // Get initial active monitor
        if let Some(initial_monitor) = get_active_monitor_sync() {
            info!("Initial active monitor: {}", initial_monitor);
            if let Ok(mut state) = shared_state.write() {
                state.set_active_monitor(initial_monitor);
            }
        }

        loop {
            match setup_event_listener(&shared_state) {
                Ok(_) => {
                    warn!("Event listener exited normally, restarting...");
                }
                Err(e) => {
                    error!("Event listener error: {}, restarting in 2s...", e);
                    std::thread::sleep(std::time::Duration::from_secs(2));
                }
            }
        }
    });
}

/// Set up Hyprland EventListener with active monitor change handler
fn setup_event_listener(shared_state: &Arc<RwLock<SharedState>>) -> anyhow::Result<()> {
    let mut listener = EventListener::new();

    // Clone for closure
    let state_clone = shared_state.clone();

    listener.add_active_monitor_changed_handler(move |data: MonitorEventData| {
        debug!("Active monitor changed: {}", data.monitor_name);
        if let Ok(mut state) = state_clone.write() {
            state.set_active_monitor(data.monitor_name.clone());
        } else {
            warn!("Failed to acquire write lock for active monitor update");
        }
    });

    info!("Hyprland EventListener registered, starting listener...");
    listener.start_listener()?;

    Ok(())
}

/// Get the currently active monitor from Hyprland (synchronous)
fn get_active_monitor_sync() -> Option<String> {
    use hyprland::data::Monitors;
    use hyprland::prelude::*;

    Monitors::get().ok().and_then(|monitors| {
        monitors
            .iter()
            .find(|m| m.focused)
            .map(|m| m.name.clone())
    })
}
