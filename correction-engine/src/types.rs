use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Instant;

/// What we injected into the target app — used to correlate AT-SPI2 events
/// back to our dictation output.
#[derive(Debug, Clone)]
pub struct InjectionContext {
    /// The full text that was injected via wtype
    pub text: String,
    /// When the injection happened (wall clock for persistence)
    pub timestamp: DateTime<Utc>,
    /// The monotonic instant of injection (for event correlation)
    pub instant: Instant,
    /// hyprctl window class of the target (e.g., "firefox", "org.gnome.TextEditor")
    pub window_class: String,
    /// hyprctl window title (for disambiguation)
    pub window_title: String,
}

/// A detected correction: what we injected vs what the user changed it to.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CorrectionPair {
    /// The substring we injected that was wrong
    pub original: String,
    /// What the user corrected it to
    pub corrected: String,
    /// ~20 chars before the correction site for disambiguation
    pub context_before: String,
    /// ~20 chars after the correction site for disambiguation
    pub context_after: String,
    /// Which application the correction happened in
    pub window_class: String,
    /// When the correction was detected
    pub timestamp: DateTime<Utc>,
}

/// Persistent record with frequency tracking — stored in corrections.json.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CorrectionRecord {
    /// The spoken/transcribed form that was incorrect
    pub original: String,
    /// The user's corrected form
    pub corrected: String,
    /// How many times this exact correction has been observed
    pub count: u32,
    /// When this correction was first seen
    pub first_seen: DateTime<Utc>,
    /// When this correction was most recently seen
    pub last_seen: DateTime<Utc>,
    /// Whether this has been auto-promoted to substitutions.txt
    pub promoted: bool,
}

/// A rejected correction pair — persisted so it's never re-recorded or promoted again.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlockedPair {
    /// The original (transcribed) form, lowercase
    pub original: String,
    /// The rejected corrected form, lowercase
    pub corrected: String,
    /// When the pair was removed/blocked
    pub blocked_at: DateTime<Utc>,
}

/// Aggregate statistics about the correction store.
#[derive(Debug, Clone, Default)]
pub struct CorrectionStats {
    /// Total number of unique correction pairs
    pub total_corrections: usize,
    /// Total number of correction observations (sum of all counts)
    pub total_observations: u32,
    /// Number of corrections that have been auto-promoted
    pub promoted_count: usize,
    /// Number of corrections pending promotion (count > 0 but < threshold)
    pub pending_count: usize,
}

/// Configuration for the correction monitor.
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Master toggle for correction learning
    pub enabled: bool,
    /// How many seconds to monitor for corrections after text injection
    pub monitor_duration_secs: u64,
    /// Number of times a correction must be seen before auto-promotion
    pub auto_promote_threshold: u32,
    /// Unpromoted pairs older than this (by last_seen) are pruned on load
    pub max_age_days: u32,
    /// Path to the corrections JSON store
    pub store_path: PathBuf,
    /// Path to the substitutions.txt file (for auto-promotion output)
    pub substitutions_path: PathBuf,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        let data_dir = dirs::data_dir().unwrap_or_else(|| PathBuf::from("~/.local/share"));
        let vd_dir = data_dir.join("voice-dictation");

        Self {
            enabled: true,
            monitor_duration_secs: 60,
            auto_promote_threshold: 3,
            max_age_days: 30,
            store_path: vd_dir.join("corrections.json"),
            substitutions_path: vd_dir.join("substitutions.txt"),
        }
    }
}

/// Our normalized AT-SPI2 text change event — decoupled from the atspi crate
/// so that diff.rs and event_filter.rs remain pure (no atspi dependency).
#[derive(Debug, Clone)]
pub struct TextChangeEvent {
    /// Whether text was inserted or deleted
    pub operation: TextChangeOp,
    /// Unicode character index where the change started
    pub start_pos: i32,
    /// Unicode character count of the changed text
    pub length: i32,
    /// The actual text that was inserted or deleted
    pub text: String,
    /// Monotonic timestamp of the event
    pub timestamp: Instant,
    /// Application name from the accessible object (AT-SPI2 source)
    pub source_app: String,
}

/// Direction of the text change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextChangeOp {
    Insert,
    Delete,
}

/// A delete+insert pair grouped into a single replacement operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplacementEvent {
    /// Text that was deleted (the original)
    pub deleted_text: String,
    /// Text that was inserted in its place (the replacement)
    pub inserted_text: String,
    /// Unicode character position where the replacement occurred
    pub position: i32,
    /// When the replacement was detected (timestamp of the insert)
    pub timestamp: Instant,
}
