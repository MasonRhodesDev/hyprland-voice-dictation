//! Monitor detection and active monitor tracking for Hyprland

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// If the event listener ran at least this long before returning, it was
/// genuinely connected (as opposed to failing the connection attempt itself).
/// Distinguishes "Hyprland restarted under us" from "IPC socket not up yet".
const CONNECTED_THRESHOLD: Duration = Duration::from_secs(5);

/// Delay between reconnect attempts
const RETRY_INTERVAL: Duration = Duration::from_secs(2);

/// Circuit breaker: max consecutive failed connection attempts before backing off
const MAX_CONSECUTIVE_FAILURES: u32 = 10;

/// Circuit breaker: cool-down period after repeated failures
const CIRCUIT_BREAKER_TIMEOUT: Duration = Duration::from_secs(60);

/// Global active monitor name. Empty string means "unknown" and makes the GUI
/// fall back to showing on all monitors.
static ACTIVE_MONITOR: OnceLock<Arc<RwLock<String>>> = OnceLock::new();

/// Timestamp of the most recent compositor topology change (monitor add/remove,
/// config reload, IPC loss). The GUI timer waits for this to settle, then
/// verifies its surfaces against the live monitor list instead of blindly
/// restarting the process on every event.
static COMPOSITOR_CHANGED_AT: OnceLock<Arc<RwLock<Option<Instant>>>> = OnceLock::new();

fn changed_at_cell() -> &'static Arc<RwLock<Option<Instant>>> {
    COMPOSITOR_CHANGED_AT.get_or_init(|| Arc::new(RwLock::new(None)))
}

/// Record a compositor topology change and refresh the active monitor, since
/// focus may have moved without an `activemon` event (e.g. the focused monitor
/// was unplugged).
pub fn note_compositor_change(reason: &str) {
    info!("Compositor change ({reason}); will verify surfaces after events settle");
    if let Ok(mut guard) = changed_at_cell().write() {
        *guard = Some(Instant::now());
    }
    if let Some(name) = get_active_monitor_sync() {
        set_active_monitor(name);
    }
}

/// When the last compositor change happened, if one is pending verification
pub fn pending_compositor_change() -> Option<Instant> {
    changed_at_cell().read().ok().and_then(|guard| *guard)
}

pub fn clear_compositor_change() {
    if let Ok(mut guard) = changed_at_cell().write() {
        *guard = None;
    }
}

fn set_active_monitor(name: String) {
    if let Some(cell) = ACTIVE_MONITOR.get() {
        if let Ok(mut guard) = cell.write() {
            *guard = name;
        }
    }
}

/// Get the currently active monitor name
pub fn get_active_monitor() -> Option<String> {
    ACTIVE_MONITOR.get().and_then(|m| m.read().ok().map(|s| s.clone()))
}

/// Get the active monitor synchronously via Hyprland IPC
pub fn get_active_monitor_sync() -> Option<String> {
    use hyprland::data::Monitors;
    use hyprland::prelude::*;

    Monitors::get()
        .ok()
        .and_then(|monitors| monitors.iter().find(|m| m.focused).map(|m| m.name.clone()))
}

/// Get the current monitor names via Hyprland IPC. Returns None when IPC is
/// unavailable (non-Hyprland compositor, Hyprland restarting).
pub fn get_monitor_names_sync() -> Option<Vec<String>> {
    use hyprland::data::Monitors;
    use hyprland::prelude::*;

    Monitors::get().ok().map(|monitors| monitors.iter().map(|m| m.name.clone()).collect())
}

/// Refresh Hyprland environment variables and verify socket accessibility
/// This helps handle Hyprland restarts or session switches gracefully
fn refresh_hyprland_environment() -> bool {
    use std::env;

    // Try to get fresh environment variables
    let instance_sig = match env::var("HYPRLAND_INSTANCE_SIGNATURE") {
        Ok(sig) => sig,
        Err(_) => {
            debug!("HYPRLAND_INSTANCE_SIGNATURE not set");
            return false;
        }
    };

    let runtime_dir = match hypr_paths::BaseDirs::from_env() {
        Ok(dirs) => dirs.runtime_dir().to_path_buf(),
        Err(_) => {
            debug!("XDG_RUNTIME_DIR not set");
            return false;
        }
    };

    // Construct expected socket path
    let socket_path = runtime_dir.join("hypr").join(&instance_sig).join(".socket.sock");

    // Verify socket exists
    if socket_path.exists() {
        debug!("Hyprland socket verified: {}", socket_path.display());
        true
    } else {
        debug!("Hyprland socket not found at: {}", socket_path.display());
        false
    }
}

/// Spawn a background thread to track active monitor changes
pub fn spawn_active_monitor_listener(reload_flag: Option<Arc<AtomicBool>>) {
    use hyprland::event_listener::{EventListener, MonitorEventData};

    // Initialize global state
    let initial = get_active_monitor_sync();
    info!("Initial active monitor from Hyprland IPC: {:?}", initial);
    let monitor = Arc::new(RwLock::new(initial.unwrap_or_default()));
    let _ = ACTIVE_MONITOR.set(monitor.clone());
    let _ = changed_at_cell();

    thread::spawn(move || {
        let mut consecutive_failures: u32 = 0;

        loop {
            refresh_hyprland_environment();

            let monitor_clone = monitor.clone();
            let reload_flag_clone = reload_flag.clone();
            let mut listener = EventListener::new();

            listener.add_active_monitor_changed_handler(move |data: MonitorEventData| {
                if let Ok(mut m) = monitor_clone.write() {
                    let old_monitor = m.clone();
                    debug!(
                        "Active monitor changed from '{}' to '{}'",
                        old_monitor, data.monitor_name
                    );
                    *m = data.monitor_name.clone();

                    // Trigger GUI reload if flag provided and monitor actually changed
                    if let Some(ref flag) = reload_flag_clone {
                        if old_monitor != data.monitor_name {
                            debug!("Setting reload flag for monitor switch");
                            flag.store(true, Ordering::SeqCst);
                        }
                    }
                }
            });

            // Topology changes: don't restart the GUI here. layer-shika creates
            // and destroys surfaces on Wayland output hotplug itself; we only
            // note the change so the GUI can verify the result once events settle.
            listener.add_monitor_added_handler(move |data| {
                note_compositor_change(&format!("monitor added: '{}'", data.name));
            });
            listener.add_monitor_removed_handler(move |name| {
                note_compositor_change(&format!("monitor removed: '{name}'"));
            });
            listener.add_config_reloaded_handler(move || {
                note_compositor_change("hyprland config reloaded");
            });

            let started = Instant::now();
            let result = listener.start_listener();
            let was_connected = started.elapsed() >= CONNECTED_THRESHOLD;

            match result {
                Ok(()) => {
                    info!("Hyprland event listener ended cleanly, reconnecting");
                }
                Err(ref e) if was_connected => {
                    warn!(
                        "Hyprland IPC connection lost after {}s (compositor restart?): {e}",
                        started.elapsed().as_secs()
                    );
                }
                Err(ref e) => {
                    warn!(
                        "Hyprland event listener connection failed (attempt {}/{}): {e}",
                        consecutive_failures + 1,
                        MAX_CONSECUTIVE_FAILURES
                    );
                }
            }

            if was_connected {
                consecutive_failures = 0;
                // Active-monitor state is now stale. Clear it so the GUI falls
                // back to showing on all monitors until we reconnect, instead of
                // pinning to a possibly-gone monitor name.
                set_active_monitor(String::new());
                note_compositor_change("hyprland IPC connection lost");
            } else {
                consecutive_failures += 1;
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                    warn!(
                        "Hyprland monitor listener failed {} times, backing off for {}s",
                        consecutive_failures,
                        CIRCUIT_BREAKER_TIMEOUT.as_secs()
                    );
                    thread::sleep(CIRCUIT_BREAKER_TIMEOUT);
                    consecutive_failures = 0;
                }
            }

            thread::sleep(RETRY_INTERVAL);

            // Re-sync active monitor before the next connection attempt
            if let Some(name) = get_active_monitor_sync() {
                set_active_monitor(name);
            }
        }
    });
}
