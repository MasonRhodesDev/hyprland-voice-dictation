//! Pure filtering functions for AT-SPI2 text change events.
//!
//! Filters events by time window, position range, and source application
//! to isolate only the events relevant to a specific text injection.

use crate::types::{InjectionContext, MonitorConfig, TextChangeEvent};
use std::time::Duration;

/// Filter a stream of text change events to only those relevant to our injection.
///
/// Applies three filters in sequence:
/// 1. **Time window**: Only events within `monitor_duration_secs` of the injection instant
/// 2. **Position range**: Only events whose position falls within (or adjacent to) the
///    injected text range
/// 3. **App filter**: Only events from the same application as the target window
pub fn filter_events(
    events: &[TextChangeEvent],
    context: &InjectionContext,
    config: &MonitorConfig,
) -> Vec<TextChangeEvent> {
    let injection_len = context.text.chars().count() as i32;
    let monitor_duration = Duration::from_secs(config.monitor_duration_secs);

    events
        .iter()
        .filter(|event| {
            // Time window filter: event must be within monitor duration of injection
            event.timestamp.duration_since(context.instant) <= monitor_duration
        })
        .filter(|event| {
            // Position filter: event must touch the injection range.
            // We allow slight overlap beyond the injection (corrections can
            // extend the text), but not events entirely outside.
            let event_end = event.start_pos + event.length;
            // Event overlaps with [0, injection_len] range
            event.start_pos <= injection_len && event_end >= 0
        })
        .filter(|event| {
            // App filter: event must be from the same application.
            // Use case-insensitive contains matching since AT-SPI2 app names
            // may not exactly match hyprctl window classes.
            if context.window_class.is_empty() {
                // If we don't know the target app, accept all events
                true
            } else {
                let source_lower = event.source_app.to_lowercase();
                let target_lower = context.window_class.to_lowercase();
                source_lower.contains(&target_lower) || target_lower.contains(&source_lower)
            }
        })
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TextChangeOp;
    use chrono::Utc;
    use std::path::PathBuf;
    use std::time::Instant;

    fn make_config(duration_secs: u64) -> MonitorConfig {
        MonitorConfig {
            enabled: true,
            monitor_duration_secs: duration_secs,
            auto_promote_threshold: 3,
            store_path: PathBuf::from("/tmp/test-corrections.json"),
            substitutions_path: PathBuf::from("/tmp/test-substitutions.txt"),
        }
    }

    fn make_context(text: &str, window_class: &str) -> (InjectionContext, Instant) {
        let base = Instant::now();
        let ctx = InjectionContext {
            text: text.to_string(),
            timestamp: Utc::now(),
            instant: base,
            window_class: window_class.to_string(),
            window_title: "Test".to_string(),
        };
        (ctx, base)
    }

    fn make_event(
        base: Instant,
        offset: Duration,
        pos: i32,
        len: i32,
        app: &str,
    ) -> TextChangeEvent {
        TextChangeEvent {
            operation: TextChangeOp::Insert,
            start_pos: pos,
            length: len,
            text: "x".repeat(len as usize),
            timestamp: base + offset,
            source_app: app.to_string(),
        }
    }

    #[test]
    fn test_time_window_filter() {
        let (context, base) = make_context("hello world", "firefox");
        let config = make_config(60); // 60s window

        let events = vec![
            make_event(base, Duration::from_secs(5), 3, 1, "firefox"),  // 5s — in window
            make_event(base, Duration::from_secs(30), 3, 1, "firefox"), // 30s — in window
            make_event(base, Duration::from_secs(90), 3, 1, "firefox"), // 90s — out of window
        ];

        let filtered = filter_events(&events, &context, &config);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_position_filter() {
        let (context, base) = make_context("hello world", "firefox"); // 11 chars, range 0-11
        let config = make_config(300);

        let events = vec![
            make_event(base, Duration::from_secs(1), 5, 1, "firefox"),  // pos 5 — in range
            make_event(base, Duration::from_secs(2), 15, 1, "firefox"), // pos 15 — out of range
            make_event(base, Duration::from_secs(3), 25, 1, "firefox"), // pos 25 — out of range
        ];

        let filtered = filter_events(&events, &context, &config);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].start_pos, 5);
    }

    #[test]
    fn test_app_filter() {
        let (context, base) = make_context("hello world", "firefox");
        let config = make_config(300);

        let events = vec![
            make_event(base, Duration::from_secs(1), 3, 1, "firefox"),       // same app
            make_event(base, Duration::from_secs(2), 3, 1, "code"),          // different app
            make_event(base, Duration::from_secs(3), 3, 1, "Firefox-esr"),   // contains match
        ];

        let filtered = filter_events(&events, &context, &config);
        assert_eq!(filtered.len(), 2); // firefox and Firefox-esr
    }

    #[test]
    fn test_empty_window_class_accepts_all() {
        let (context, base) = make_context("hello world", "");
        let config = make_config(300);

        let events = vec![
            make_event(base, Duration::from_secs(1), 3, 1, "firefox"),
            make_event(base, Duration::from_secs(2), 3, 1, "code"),
        ];

        let filtered = filter_events(&events, &context, &config);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_combined_filters() {
        let (context, base) = make_context("hello world", "firefox"); // 11 chars
        let config = make_config(60);

        let events = vec![
            // Valid: in time, in position, correct app
            make_event(base, Duration::from_secs(5), 3, 1, "firefox"),
            // Invalid: out of time
            make_event(base, Duration::from_secs(90), 3, 1, "firefox"),
            // Invalid: out of position
            make_event(base, Duration::from_secs(5), 50, 1, "firefox"),
            // Invalid: wrong app
            make_event(base, Duration::from_secs(5), 3, 1, "code"),
            // Valid: in time, in position, correct app (case-insensitive)
            make_event(base, Duration::from_secs(10), 6, 2, "Firefox"),
        ];

        let filtered = filter_events(&events, &context, &config);
        assert_eq!(filtered.len(), 2);
    }
}
