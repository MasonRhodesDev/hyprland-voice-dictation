//! AT-SPI2-based correction detection engine.
//!
//! Monitors text changes in target applications after voice dictation injection,
//! detects user corrections, and feeds them back into the word substitution pipeline.
//!
//! # Architecture
//!
//! - **`types`**: Shared data types (no external deps)
//! - **`diff`**: Pure functions for extracting correction pairs from events
//! - **`event_filter`**: Pure functions for filtering events by time/position/app
//! - **`store`**: JSON persistence with frequency counting and auto-promotion
//! - **`connection`**: AT-SPI2 bus abstraction (trait-based for testability)
//!
//! # Usage
//!
//! ```rust,no_run
//! use correction_engine::{CorrectionMonitor, MonitorConfig, InjectionContext};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let monitor = CorrectionMonitor::new(MonitorConfig::default()).await?;
//!
//! // After injecting text via wtype...
//! let context = InjectionContext {
//!     text: "hello world".to_string(),
//!     timestamp: chrono::Utc::now(),
//!     instant: std::time::Instant::now(),
//!     window_class: "firefox".to_string(),
//!     window_title: "Example".to_string(),
//! };
//!
//! // Spawns background task, returns immediately
//! let _handle = monitor.start_monitoring(context);
//! # Ok(())
//! # }
//! ```

pub mod connection;
pub mod diff;
pub mod event_filter;
pub mod store;
pub mod types;

// Re-export primary API types
pub use connection::{AtspiConnection, MockTextChangeSource, TextChangeSource};
pub use store::CorrectionStore;
pub use types::{
    CorrectionPair, CorrectionStats, InjectionContext, MonitorConfig, TextChangeEvent,
};

use anyhow::Result;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

/// The main correction monitor that orchestrates AT-SPI2 event collection,
/// filtering, correction extraction, and persistence.
pub struct CorrectionMonitor {
    source: Arc<dyn TextChangeSource>,
    store: Arc<Mutex<CorrectionStore>>,
    config: MonitorConfig,
}

impl CorrectionMonitor {
    /// Create a new correction monitor with the production AT-SPI2 connection.
    pub async fn new(config: MonitorConfig) -> Result<Self> {
        let connection = AtspiConnection::connect().await;
        let store = CorrectionStore::load(&config)?;

        Ok(Self { source: Arc::new(connection), store: Arc::new(Mutex::new(store)), config })
    }

    /// Create a correction monitor with a custom event source (for testing).
    pub fn with_source(
        source: Arc<dyn TextChangeSource>,
        store: CorrectionStore,
        config: MonitorConfig,
    ) -> Self {
        Self { source, store: Arc::new(Mutex::new(store)), config }
    }

    /// Start monitoring for corrections after text injection.
    ///
    /// Spawns a detached background task that:
    /// 1. Subscribes to AT-SPI2 text-changed events
    /// 2. Collects events for `monitor_duration_secs`
    /// 3. Filters events by time/position/app
    /// 4. Extracts correction pairs
    /// 5. Records corrections in the store (with auto-promotion)
    ///
    /// Returns a JoinHandle that resolves with detected corrections.
    pub fn start_monitoring(
        &self,
        context: InjectionContext,
    ) -> tokio::task::JoinHandle<Vec<CorrectionPair>> {
        let source = Arc::clone(&self.source);
        let store = Arc::clone(&self.store);
        let config = self.config.clone();

        tokio::spawn(async move {
            match Self::run_monitoring(source, store, config, context).await {
                Ok(corrections) => corrections,
                Err(e) => {
                    error!("Correction monitoring failed: {}", e);
                    Vec::new()
                }
            }
        })
    }

    /// Internal monitoring loop.
    async fn run_monitoring(
        source: Arc<dyn TextChangeSource>,
        store: Arc<Mutex<CorrectionStore>>,
        config: MonitorConfig,
        context: InjectionContext,
    ) -> Result<Vec<CorrectionPair>> {
        if !source.is_available() {
            debug!("AT-SPI2 not available, skipping correction monitoring");
            return Ok(Vec::new());
        }

        let mut rx = source.subscribe().await?;
        let monitor_duration = Duration::from_secs(config.monitor_duration_secs);

        info!(
            "Correction monitoring started for {}s (text: '{}...')",
            config.monitor_duration_secs,
            &context.text[..context.text.len().min(30)]
        );

        // Collect events for the monitoring window
        let mut events = Vec::new();
        let deadline = tokio::time::Instant::now() + monitor_duration;

        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                break;
            }

            match tokio::time::timeout(remaining, rx.recv()).await {
                Ok(Some(event)) => {
                    debug!(
                        "AT-SPI2 event: {:?} at pos {} text='{}' from '{}'",
                        event.operation, event.start_pos, event.text, event.source_app
                    );
                    events.push(event);
                }
                Ok(None) => {
                    // Channel closed (source disconnected)
                    debug!("Event channel closed, ending monitoring early");
                    break;
                }
                Err(_) => {
                    // Timeout — monitoring window expired
                    break;
                }
            }
        }

        info!("Correction monitoring ended. {} events collected.", events.len());

        if events.is_empty() {
            return Ok(Vec::new());
        }

        info!("Monitoring window summary: {}", summarize_events(&events));

        // Filter events
        let filtered = event_filter::filter_events(&events, &context, &config);
        debug!("After filtering: {} of {} events are relevant", filtered.len(), events.len());

        if filtered.is_empty() {
            return Ok(Vec::new());
        }

        // Extract corrections
        let corrections = diff::extract_corrections(&context, &filtered);

        if corrections.is_empty() {
            debug!("No corrections extracted from events");
            return Ok(Vec::new());
        }

        info!("Detected {} correction(s)", corrections.len());

        // Record corrections in store
        let mut store = store.lock().await;
        for correction in &corrections {
            match store.record_correction(correction.clone()) {
                Ok(promoted) => {
                    if promoted {
                        info!(
                            "Correction auto-promoted to substitution: '{}' → '{}'",
                            correction.original, correction.corrected
                        );
                    }
                }
                Err(e) => {
                    warn!("Failed to record correction: {}", e);
                }
            }
        }

        Ok(corrections)
    }

    /// Check if AT-SPI2 is available on this system.
    pub fn is_available(&self) -> bool {
        self.source.is_available()
    }

    /// Get correction statistics.
    pub async fn stats(&self) -> CorrectionStats {
        let store = self.store.lock().await;
        store.stats()
    }
}

/// Summarize a monitoring window's events as counts grouped by source
/// application and event type. Purely diagnostic — helps explain surprise
/// event floods (e.g. 122 events from a single injection).
fn summarize_events(events: &[TextChangeEvent]) -> String {
    let mut per_app: std::collections::BTreeMap<&str, (usize, usize)> =
        std::collections::BTreeMap::new();
    for event in events {
        let entry = per_app.entry(event.source_app.as_str()).or_insert((0, 0));
        match event.operation {
            types::TextChangeOp::Insert => entry.0 += 1,
            types::TextChangeOp::Delete => entry.1 += 1,
        }
    }
    per_app
        .iter()
        .map(|(app, (inserts, deletes))| {
            format!("'{}': {} insert(s), {} delete(s)", app, inserts, deletes)
        })
        .collect::<Vec<_>>()
        .join("; ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TextChangeOp;
    use std::time::Instant;

    fn make_event(source_app: &str, operation: TextChangeOp) -> TextChangeEvent {
        TextChangeEvent {
            operation,
            start_pos: 0,
            length: 1,
            text: "x".to_string(),
            timestamp: Instant::now(),
            source_app: source_app.to_string(),
        }
    }

    #[test]
    fn test_summarize_events_groups_by_app_and_type() {
        let events = vec![
            make_event("firefox", TextChangeOp::Insert),
            make_event("firefox", TextChangeOp::Insert),
            make_event("firefox", TextChangeOp::Delete),
            make_event("Code", TextChangeOp::Insert),
        ];

        assert_eq!(
            summarize_events(&events),
            "'Code': 1 insert(s), 0 delete(s); 'firefox': 2 insert(s), 1 delete(s)"
        );
    }
}
