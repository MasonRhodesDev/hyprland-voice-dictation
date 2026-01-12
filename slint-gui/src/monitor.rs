//! Monitor detection and active monitor tracking for Hyprland

use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;
use tracing::{debug, warn};

/// Global active monitor name
static ACTIVE_MONITOR: std::sync::OnceLock<Arc<RwLock<String>>> = std::sync::OnceLock::new();

/// Get the currently active monitor name
pub fn get_active_monitor() -> Option<String> {
    ACTIVE_MONITOR
        .get()
        .and_then(|m| m.read().ok().map(|s| s.clone()))
}

/// Get the active monitor synchronously via Hyprland IPC
pub fn get_active_monitor_sync() -> Option<String> {
    use hyprland::data::Monitors;
    use hyprland::prelude::*;

    Monitors::get().ok().and_then(|monitors| {
        monitors
            .iter()
            .find(|m| m.focused)
            .map(|m| m.name.clone())
    })
}

/// Spawn a background thread to track active monitor changes
pub fn spawn_active_monitor_listener() {
    use hyprland::event_listener::{EventListener, MonitorEventData};

    // Initialize global state
    let monitor = Arc::new(RwLock::new(
        get_active_monitor_sync().unwrap_or_default(),
    ));
    let _ = ACTIVE_MONITOR.set(monitor.clone());

    thread::spawn(move || {
        loop {
            let monitor_clone = monitor.clone();
            let mut listener = EventListener::new();

            listener.add_active_monitor_changed_handler(move |data: MonitorEventData| {
                if let Ok(mut m) = monitor_clone.write() {
                    debug!("Active monitor changed to: {}", data.monitor_name);
                    *m = data.monitor_name.clone();
                }
            });

            if let Err(e) = listener.start_listener() {
                warn!("Hyprland event listener error: {}, restarting...", e);
                thread::sleep(Duration::from_secs(2));
            }
        }
    });
}
