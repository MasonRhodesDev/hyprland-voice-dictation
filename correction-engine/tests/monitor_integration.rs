//! Integration tests for the CorrectionMonitor using mock AT-SPI2 sources.
//!
//! These tests exercise the full monitoring pipeline without a real AT-SPI2 bus:
//! mock events → filter → diff → store → persistence.

use chrono::Utc;
use correction_engine::connection::MockTextChangeSource;
use correction_engine::store::CorrectionStore;
use correction_engine::types::{InjectionContext, MonitorConfig, TextChangeEvent, TextChangeOp};
use correction_engine::CorrectionMonitor;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::TempDir;

fn make_config(dir: &std::path::Path, duration_secs: u64, threshold: u32) -> MonitorConfig {
    MonitorConfig {
        enabled: true,
        monitor_duration_secs: duration_secs,
        auto_promote_threshold: threshold,
        max_age_days: 30,
        store_path: dir.join("corrections.json"),
        substitutions_path: dir.join("substitutions.txt"),
    }
}

fn make_context(text: &str) -> InjectionContext {
    InjectionContext {
        text: text.to_string(),
        timestamp: Utc::now(),
        instant: Instant::now(),
        window_class: "test-app".to_string(),
        window_title: "Test Window".to_string(),
    }
}

fn make_event(op: TextChangeOp, pos: i32, text: &str) -> TextChangeEvent {
    TextChangeEvent {
        operation: op,
        start_pos: pos,
        length: text.chars().count() as i32,
        text: text.to_string(),
        timestamp: Instant::now(),
        source_app: "test-app".to_string(),
    }
}

#[tokio::test]
async fn test_full_correction_flow() {
    let dir = TempDir::new().unwrap();
    let config = make_config(dir.path(), 5, 3); // 5s monitoring, threshold 3
    let store = CorrectionStore::empty(config.clone());
    let (mock_source, tx) = MockTextChangeSource::new(true);

    let monitor = CorrectionMonitor::with_source(Arc::new(mock_source), store, config);

    let context = make_context("hello world");
    let handle = monitor.start_monitoring(context);

    // Give the monitor a moment to subscribe
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Simulate: user deletes "world" and types "earth"
    tx.send(make_event(TextChangeOp::Delete, 6, "world")).await.unwrap();
    tokio::time::sleep(Duration::from_millis(10)).await;
    tx.send(make_event(TextChangeOp::Insert, 6, "earth")).await.unwrap();

    // Close the channel to end monitoring early
    drop(tx);

    let corrections = handle.await.unwrap();
    assert_eq!(corrections.len(), 1);
    assert_eq!(corrections[0].original, "world");
    assert_eq!(corrections[0].corrected, "earth");
}

#[tokio::test]
async fn test_monitoring_timeout() {
    let dir = TempDir::new().unwrap();
    let config = make_config(dir.path(), 1, 3); // 1s monitoring window
    let store = CorrectionStore::empty(config.clone());
    let (mock_source, _tx) = MockTextChangeSource::new(true);

    let monitor = CorrectionMonitor::with_source(Arc::new(mock_source), store, config);

    let context = make_context("hello world");
    let handle = monitor.start_monitoring(context);

    // Don't send any events — wait for timeout
    let corrections = handle.await.unwrap();
    assert!(corrections.is_empty());
}

#[tokio::test]
async fn test_graceful_degradation() {
    let dir = TempDir::new().unwrap();
    let config = make_config(dir.path(), 2, 3);
    let store = CorrectionStore::empty(config.clone());
    let (mock_source, _tx) = MockTextChangeSource::new(false); // AT-SPI2 unavailable

    let monitor = CorrectionMonitor::with_source(Arc::new(mock_source), store, config);

    assert!(!monitor.is_available());

    let context = make_context("hello world");
    let handle = monitor.start_monitoring(context);

    let corrections = handle.await.unwrap();
    assert!(corrections.is_empty());
}

#[tokio::test]
async fn test_auto_promotion_e2e() {
    let dir = TempDir::new().unwrap();
    let config = make_config(dir.path(), 2, 3); // threshold 3

    // Run 3 monitoring sessions with the same correction
    for i in 0..3 {
        let store = CorrectionStore::load(&config).unwrap();
        let (mock_source, tx) = MockTextChangeSource::new(true);
        let monitor = CorrectionMonitor::with_source(Arc::new(mock_source), store, config.clone());

        let context = make_context("the cash is here");
        let handle = monitor.start_monitoring(context);

        tokio::time::sleep(Duration::from_millis(50)).await;

        // Same correction each time: cash → cache
        tx.send(make_event(TextChangeOp::Delete, 4, "cash")).await.unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        tx.send(make_event(TextChangeOp::Insert, 4, "cache")).await.unwrap();

        drop(tx);
        let corrections = handle.await.unwrap();
        assert_eq!(corrections.len(), 1, "Session {} should detect 1 correction", i);
    }

    // After 3 sessions, check that the correction was auto-promoted
    let subs_content = std::fs::read_to_string(config.substitutions_path).unwrap();
    assert!(
        subs_content.contains("cash -> cache"),
        "substitutions.txt should contain 'cash -> cache', got: {}",
        subs_content
    );
}

#[tokio::test]
async fn test_stats() {
    let dir = TempDir::new().unwrap();
    let config = make_config(dir.path(), 2, 5);
    let store = CorrectionStore::empty(config.clone());
    let (mock_source, tx) = MockTextChangeSource::new(true);

    let monitor = CorrectionMonitor::with_source(Arc::new(mock_source), store, config);

    let context = make_context("hello world");
    let handle = monitor.start_monitoring(context);

    tokio::time::sleep(Duration::from_millis(50)).await;

    tx.send(make_event(TextChangeOp::Delete, 6, "world")).await.unwrap();
    tokio::time::sleep(Duration::from_millis(10)).await;
    tx.send(make_event(TextChangeOp::Insert, 6, "earth")).await.unwrap();

    drop(tx);
    handle.await.unwrap();

    let stats = monitor.stats().await;
    assert_eq!(stats.total_corrections, 1);
    assert_eq!(stats.total_observations, 1);
    assert_eq!(stats.promoted_count, 0);
    assert_eq!(stats.pending_count, 1);
}
