//! Monitor detection and active monitor tracking for Hyprland

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{debug, warn};

/// Circuit breaker: max consecutive failures before opening circuit
const MAX_CONSECUTIVE_FAILURES: u32 = 10; // 20 seconds of failures (10 * 2s retry interval)

/// Circuit breaker: cool-down period after circuit opens
const CIRCUIT_BREAKER_TIMEOUT: Duration = Duration::from_secs(60);

/// Health tracking for the monitor listener to implement circuit breaker pattern
struct MonitorListenerHealth {
    consecutive_failures: AtomicU32,
    circuit_open_until: Arc<RwLock<Option<Instant>>>,
}

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

/// Refresh Hyprland environment variables and verify socket accessibility
/// This helps handle Hyprland restarts or session switches gracefully
fn refresh_hyprland_environment() -> bool {
    use std::env;
    use std::path::Path;

    // Try to get fresh environment variables
    let instance_sig = match env::var("HYPRLAND_INSTANCE_SIGNATURE") {
        Ok(sig) => sig,
        Err(_) => {
            debug!("HYPRLAND_INSTANCE_SIGNATURE not set");
            return false;
        }
    };

    let runtime_dir = match env::var("XDG_RUNTIME_DIR") {
        Ok(dir) => dir,
        Err(_) => {
            debug!("XDG_RUNTIME_DIR not set");
            return false;
        }
    };

    // Construct expected socket path
    let socket_dir = format!("{}/hypr/{}", runtime_dir, instance_sig);
    let socket_path = format!("{}/.socket.sock", socket_dir);

    // Verify socket exists
    if Path::new(&socket_path).exists() {
        debug!("Hyprland socket verified: {}", socket_path);
        true
    } else {
        debug!("Hyprland socket not found at: {}", socket_path);
        false
    }
}

/// Spawn a background thread to track active monitor changes
pub fn spawn_active_monitor_listener(reload_flag: Option<Arc<std::sync::atomic::AtomicBool>>) {
    use hyprland::event_listener::{EventListener, MonitorEventData};

    // Initialize global state
    let monitor = Arc::new(RwLock::new(
        get_active_monitor_sync().unwrap_or_default(),
    ));
    let _ = ACTIVE_MONITOR.set(monitor.clone());

    // Create health tracker for circuit breaker
    let health = Arc::new(MonitorListenerHealth {
        consecutive_failures: AtomicU32::new(0),
        circuit_open_until: Arc::new(RwLock::new(None)),
    });

    thread::spawn(move || {
        loop {
            // Check circuit breaker state
            if let Ok(circuit) = health.circuit_open_until.read() {
                if let Some(open_until) = *circuit {
                    if Instant::now() < open_until {
                        // Circuit is open, wait before retrying
                        debug!("Circuit breaker open, waiting before retry");
                        thread::sleep(Duration::from_secs(10));
                        continue;
                    }
                }
            }

            // Refresh environment before reconnect attempt
            refresh_hyprland_environment();

            let monitor_clone = monitor.clone();
            let reload_flag_clone = reload_flag.clone();
            let mut listener = EventListener::new();

            listener.add_active_monitor_changed_handler(move |data: MonitorEventData| {
                if let Ok(mut m) = monitor_clone.write() {
                    let old_monitor = m.clone();
                    debug!("Active monitor changed from '{}' to '{}'", old_monitor, data.monitor_name);
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

            match listener.start_listener() {
                Ok(_) => {
                    // Success - reset failure counter
                    health.consecutive_failures.store(0, Ordering::SeqCst);
                    debug!("Hyprland monitor listener connected successfully");
                }
                Err(e) => {
                    // Failure - increment counter and check circuit breaker
                    let failures = health.consecutive_failures.fetch_add(1, Ordering::SeqCst) + 1;

                    if failures >= MAX_CONSECUTIVE_FAILURES {
                        // Open circuit breaker
                        warn!(
                            "Hyprland monitor listener failed {} times, opening circuit breaker for {}s: {}",
                            failures,
                            CIRCUIT_BREAKER_TIMEOUT.as_secs(),
                            e
                        );

                        if let Ok(mut circuit) = health.circuit_open_until.write() {
                            *circuit = Some(Instant::now() + CIRCUIT_BREAKER_TIMEOUT);
                        }

                        // Reset failure counter for next circuit attempt
                        health.consecutive_failures.store(0, Ordering::SeqCst);
                    } else {
                        warn!(
                            "Hyprland event listener error (attempt {}/{}): {}",
                            failures, MAX_CONSECUTIVE_FAILURES, e
                        );
                        thread::sleep(Duration::from_secs(2));
                    }
                }
            }
        }
    });
}
