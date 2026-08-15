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
    let value = hypr_ipc::hyprctl_json_std(&["-j", "monitors"]).ok()?;
    value.as_array()?.iter().find_map(|monitor| {
        if monitor.get("focused").and_then(|v| v.as_bool()).unwrap_or(false) {
            monitor.get("name").and_then(|n| n.as_str()).map(str::to_string)
        } else {
            None
        }
    })
}

/// Get the current monitor names via Hyprland IPC. Returns None when IPC is
/// unavailable (non-Hyprland compositor, Hyprland restarting).
pub fn get_monitor_names_sync() -> Option<Vec<String>> {
    let value = hypr_ipc::hyprctl_json_std(&["-j", "monitors"]).ok()?;
    Some(
        value
            .as_array()?
            .iter()
            .filter_map(|monitor| monitor.get("name").and_then(|n| n.as_str()).map(str::to_string))
            .collect(),
    )
}

/// Re-resolve the live Hyprland event socket (HIS, then lockfile rescan).
fn refresh_hyprland_environment() -> bool {
    match hypr_ipc::socket2_path() {
        Ok(path) => {
            debug!("Hyprland socket verified: {}", path.display());
            true
        }
        Err(_) => {
            debug!("Hyprland event socket not found");
            false
        }
    }
}

/// Spawn a background thread to track active monitor changes
pub fn spawn_active_monitor_listener(reload_flag: Option<Arc<AtomicBool>>) {
    let initial = get_active_monitor_sync();
    info!("Initial active monitor from Hyprland IPC: {:?}", initial);
    let monitor = Arc::new(RwLock::new(initial.unwrap_or_default()));
    let _ = ACTIVE_MONITOR.set(monitor.clone());
    let _ = changed_at_cell();

    thread::spawn(move || {
        let Ok(rt) = tokio::runtime::Builder::new_current_thread().enable_all().build() else {
            warn!("failed to start Hyprland monitor listener runtime");
            return;
        };
        rt.block_on(listen_socket2(monitor, reload_flag));
    });
}

async fn listen_socket2(monitor: Arc<RwLock<String>>, reload_flag: Option<Arc<AtomicBool>>) {
    use tokio::io::AsyncBufReadExt;

    let mut consecutive_failures: u32 = 0;
    loop {
        refresh_hyprland_environment();
        let started = Instant::now();
        match hypr_ipc::connect_socket2().await {
            Ok(stream) => {
                info!("connected to Hyprland event socket");
                consecutive_failures = 0;
                let mut lines = tokio::io::BufReader::new(stream).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let Some(frame) = hypr_ipc::parse_line(&line) else {
                        continue;
                    };
                    match frame.event.as_str() {
                        "focusedmon" => {
                            let name = frame.payload.split(',').next().unwrap_or("").to_string();
                            if let Ok(mut current) = monitor.write() {
                                let old_monitor = current.clone();
                                debug!(
                                    "Active monitor changed from '{}' to '{}'",
                                    old_monitor, name
                                );
                                *current = name.clone();
                                if let Some(ref flag) = reload_flag {
                                    if old_monitor != name {
                                        debug!("Setting reload flag for monitor switch");
                                        flag.store(true, Ordering::SeqCst);
                                    }
                                }
                            }
                        }
                        "monitoraddedv2" => {
                            let name = frame.payload.split(',').nth(1).unwrap_or("");
                            note_compositor_change(&format!("monitor added: '{name}'"));
                        }
                        "monitorremoved" => {
                            note_compositor_change(&format!(
                                "monitor removed: '{}'",
                                frame.payload
                            ));
                        }
                        "configreloaded" => {
                            note_compositor_change("hyprland config reloaded");
                        }
                        _ => {}
                    }
                }
                let was_connected = started.elapsed() >= CONNECTED_THRESHOLD;
                if was_connected {
                    warn!("Hyprland IPC connection lost after {}s", started.elapsed().as_secs());
                    set_active_monitor(String::new());
                    note_compositor_change("hyprland IPC connection lost");
                } else {
                    warn!("Hyprland event socket closed before staying connected");
                }
            }
            Err(e) => {
                consecutive_failures += 1;
                warn!(
                    "Hyprland event listener connection failed (attempt {}/{}): {e}",
                    consecutive_failures, MAX_CONSECUTIVE_FAILURES
                );
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                    warn!(
                        "Hyprland monitor listener failed {} times, backing off for {}s",
                        consecutive_failures,
                        CIRCUIT_BREAKER_TIMEOUT.as_secs()
                    );
                    tokio::time::sleep(CIRCUIT_BREAKER_TIMEOUT).await;
                    consecutive_failures = 0;
                }
            }
        }
        tokio::time::sleep(RETRY_INTERVAL).await;
        if let Some(name) = get_active_monitor_sync() {
            set_active_monitor(name);
        }
    }
}
