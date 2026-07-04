//! End-to-end test for the real AT-SPI2 wiring.
//!
//! Unlike `monitor_integration.rs` (which drives the filter→diff→store pipeline
//! with a `MockTextChangeSource`), this test proves the *real* accessibility
//! plumbing works: that `AtspiConnection::connect()` + `.subscribe()` receives
//! genuine `TextChangeEvent`s emitted by a live GTK4 app on a real AT-SPI2 bus.
//!
//! It only runs its assertions inside the e2e container (where the entrypoint
//! sets `CORRECTION_E2E=1`, starts D-Bus + at-spi2-registryd + weston, and
//! launches `test_app.py`). Outside that environment — e.g. a developer machine
//! or the plain `cargo test` CI job — it skips cleanly so it never fails where
//! there is no accessibility bus to observe.
//!
//! Two facts about GTK4/AT-SPI2 shape the choreography, both verified live in
//! headless weston while writing this test:
//!
//!   * Ordering is load-bearing: GTK4's AT-SPI bridge only emits accessibility
//!     events for a toplevel realized while an AT client is already listening.
//!     So the test connects and subscribes *before* the app maps its window.
//!   * Accessibles are lazy: GTK4 does not instantiate a widget's accessible
//!     object (and therefore emits no text-changed events for it) until an AT
//!     client walks the accessibility tree. So the test traverses the tree from
//!     the desktop root before triggering the edit.
//!
//! Handshake with the GTK4 app (see `e2e/test_app.py` / `e2e/entrypoint.sh`),
//! via marker files under /tmp:
//!   1. This test connects to AT-SPI2, subscribes, then writes
//!      `/tmp/e2e-subscribed`.
//!   2. The app (already launched but window-less) sees that marker, maps its
//!      window with text "the cash is here", and writes `/tmp/e2e-ready`.
//!   3. This test walks the accessibility tree, then writes `/tmp/e2e-do-edit`.
//!   4. The app turns "cash" into "cache" in place and writes
//!      `/tmp/e2e-edit-done`.
//!   5. This test collects events and asserts the correction was observed.

use correction_engine::connection::{AtspiConnection, TextChangeSource};
use correction_engine::types::{TextChangeEvent, TextChangeOp};
use std::path::Path;
use std::time::Duration;
use tokio::time::{sleep, timeout};

const SUBSCRIBED_MARKER: &str = "/tmp/e2e-subscribed";
const READY_MARKER: &str = "/tmp/e2e-ready";
const TRIGGER_MARKER: &str = "/tmp/e2e-do-edit";

/// Poll for a marker file to appear, up to `deadline`.
async fn wait_for_marker(path: &str, deadline: Duration) -> bool {
    let start = std::time::Instant::now();
    while start.elapsed() < deadline {
        if Path::new(path).exists() {
            return true;
        }
        sleep(Duration::from_millis(100)).await;
    }
    Path::new(path).exists()
}

/// Traverse the AT-SPI accessible tree from the desktop root. Merely reading the
/// children forces each app (GTK4 in particular) to instantiate its accessible
/// objects on the bus, which is a precondition for it to emit text-changed
/// events. Best-effort: errors on individual nodes are ignored.
async fn walk_atspi_tree() {
    use atspi::object_ref::ObjectRefOwned;
    use atspi::proxy::accessible::{AccessibleProxy, ObjectRefExt};

    let conn = match atspi::AccessibilityConnection::new().await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("walk_atspi_tree: connect failed: {e}");
            return;
        }
    };
    let zconn = conn.connection().clone();

    // The desktop root lives on the registry at the well-known accessible path.
    let root_res = async {
        AccessibleProxy::builder(&zconn)
            .destination("org.a11y.atspi.Registry")?
            .path("/org/a11y/atspi/accessible/root")?
            .cache_properties(atspi::zbus::proxy::CacheProperties::No)
            .build()
            .await
    }
    .await;
    let root = match root_res {
        Ok(p) => p,
        Err(e) => {
            eprintln!("walk_atspi_tree: root proxy failed: {e}");
            return;
        }
    };

    // Iterative traversal with a bounded node budget so we never hang.
    let mut queue: Vec<ObjectRefOwned> = match root.get_children().await {
        Ok(kids) => kids,
        Err(e) => {
            eprintln!("walk_atspi_tree: root has no children: {e}");
            return;
        }
    };
    let mut visited = 0usize;
    while let Some(obj) = queue.pop() {
        visited += 1;
        if visited > 500 {
            break;
        }
        if let Ok(proxy) = obj.as_accessible_proxy(&zconn).await {
            // Reading role + children is what instantiates the accessible.
            let _ = proxy.get_role_name().await;
            if let Ok(kids) = proxy.get_children().await {
                queue.extend(kids);
            }
        }
    }
    eprintln!("walk_atspi_tree: visited {visited} accessible node(s)");
}

#[tokio::test]
async fn atspi_receives_real_text_change_events() {
    // Only assert inside the e2e container; skip everywhere else.
    if std::env::var("CORRECTION_E2E").as_deref() != Ok("1") {
        eprintln!("CORRECTION_E2E != 1 — skipping live AT-SPI2 e2e test (not in container)");
        return;
    }

    // Clean up any stale markers from a previous run (the entrypoint also does
    // this, but keep the test self-contained and deterministic).
    let _ = std::fs::remove_file(SUBSCRIBED_MARKER);
    let _ = std::fs::remove_file(TRIGGER_MARKER);

    // 1. Connect to the real AT-SPI2 bus and subscribe BEFORE the app maps its
    //    window (see module docs: ordering is load-bearing).
    let conn = AtspiConnection::connect().await;
    assert!(
        conn.is_available(),
        "AT-SPI2 bus reported unavailable inside the e2e container; \
         dbus/at-spi2-registryd wiring is broken"
    );
    let mut rx = conn.subscribe().await.expect("subscribe() failed");

    // Give the listener task a moment to register on the bus, then release the
    // app: it is waiting for this marker before creating its window.
    sleep(Duration::from_millis(500)).await;
    std::fs::write(SUBSCRIBED_MARKER, b"subscribed\n").expect("failed to write subscribed marker");

    // 2. Wait for the GTK4 app to map its window and signal readiness.
    assert!(
        wait_for_marker(READY_MARKER, Duration::from_secs(15)).await,
        "GTK4 test app never wrote {READY_MARKER}; app failed to start under weston"
    );

    // 3. Walk the AT-SPI accessible tree to force GTK4 to instantiate the
    //    entry's accessible object (see module docs: accessibles are lazy).
    sleep(Duration::from_millis(500)).await;
    walk_atspi_tree().await;

    // 4. Trigger the in-place cash->cache edit.
    std::fs::write(TRIGGER_MARKER, b"go\n").expect("failed to write trigger marker");

    // 5. Collect events with an overall deadline. Stop early once we have proven
    //    the correction.
    let mut events: Vec<TextChangeEvent> = Vec::new();
    let mut saw_insert_cache = false;
    let mut saw_delete_cash = false;

    let overall = timeout(Duration::from_secs(15), async {
        loop {
            match timeout(Duration::from_secs(2), rx.recv()).await {
                Ok(Some(ev)) => {
                    if ev.operation == TextChangeOp::Insert && ev.text.contains("cache") {
                        saw_insert_cache = true;
                    }
                    if ev.operation == TextChangeOp::Delete && ev.text.contains("cash") {
                        saw_delete_cash = true;
                    }
                    events.push(ev);

                    // Success as soon as we can prove the correction: any insert
                    // of "cache" (GTK may emit whole-buffer or per-word changes;
                    // the delete("cash") is a nice-to-have we also record).
                    if saw_insert_cache {
                        break;
                    }
                }
                // Channel closed unexpectedly — nothing more will arrive.
                Ok(None) => break,
                // Inner recv timeout — keep waiting until the overall deadline.
                Err(_elapsed) => continue,
            }
        }
    })
    .await;

    let dump = || -> String {
        events
            .iter()
            .map(|e| {
                format!(
                    "  {:?} start={} len={} text={:?} app={:?}",
                    e.operation, e.start_pos, e.length, e.text, e.source_app
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    assert!(
        !events.is_empty(),
        "No AT-SPI2 text-change events received at all within 15s. \
         The real accessibility wiring is not emitting — this is the thing under test.\n\
         (ready marker seen, subscription live, tree walked, trigger written, \
         app should have edited)"
    );

    assert!(
        saw_insert_cache,
        "Received {} AT-SPI2 event(s) but none was an Insert containing \"cache\" \
         (the correction). overall_deadline_hit={}\nEvents:\n{}",
        events.len(),
        overall.is_err(),
        dump()
    );

    eprintln!(
        "e2e OK: {} event(s); insert_cache={} delete_cash={}\n{}",
        events.len(),
        saw_insert_cache,
        saw_delete_cash,
        dump()
    );
}
