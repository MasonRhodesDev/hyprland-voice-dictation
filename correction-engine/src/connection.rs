//! AT-SPI2 accessibility bus connection management.
//!
//! Provides a trait-based abstraction over the AT-SPI2 event stream so that
//! the correction monitor can be tested with mock event sources.

use crate::types::{TextChangeEvent, TextChangeOp};
use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// Abstraction over text change event sources.
///
/// Production: `AtspiConnection` connects to the real AT-SPI2 bus.
/// Testing: `MockTextChangeSource` injects events from a channel.
#[async_trait]
pub trait TextChangeSource: Send + Sync {
    /// Subscribe to text change events. Returns a receiver that yields
    /// events as they occur. The sender side is managed internally.
    async fn subscribe(&self) -> Result<mpsc::Receiver<TextChangeEvent>>;

    /// Check if the event source is available and connected.
    fn is_available(&self) -> bool;
}

/// Production AT-SPI2 connection that listens for text-changed events
/// on the accessibility bus.
pub struct AtspiConnection {
    available: bool,
}

impl AtspiConnection {
    /// Attempt to connect to the AT-SPI2 accessibility bus.
    /// Returns a connection that may or may not be available.
    pub async fn connect() -> Self {
        match Self::try_connect().await {
            Ok(()) => {
                info!("AT-SPI2 accessibility bus connected");
                Self { available: true }
            }
            Err(e) => {
                warn!(
                    "AT-SPI2 accessibility bus not available: {} — correction detection will be disabled",
                    e
                );
                Self { available: false }
            }
        }
    }

    async fn try_connect() -> Result<()> {
        // Attempt to create an AccessibilityConnection to verify the bus is reachable
        let _conn = atspi::AccessibilityConnection::new().await?;
        Ok(())
    }
}

#[async_trait]
impl TextChangeSource for AtspiConnection {
    async fn subscribe(&self) -> Result<mpsc::Receiver<TextChangeEvent>> {
        let (tx, rx) = mpsc::channel(256);

        if !self.available {
            // Return an empty channel that will never yield events
            return Ok(rx);
        }

        // Spawn a task that connects to AT-SPI2 and forwards events
        tokio::spawn(async move {
            if let Err(e) = run_event_listener(tx).await {
                error!("AT-SPI2 event listener error: {}", e);
            }
        });

        Ok(rx)
    }

    fn is_available(&self) -> bool {
        self.available
    }
}

/// Internal event listener that connects to AT-SPI2, registers for
/// text-changed events, and forwards them through the channel.
async fn run_event_listener(tx: mpsc::Sender<TextChangeEvent>) -> Result<()> {
    use atspi::events::object::TextChangedEvent;
    use futures_util::StreamExt;

    let atspi = atspi::AccessibilityConnection::new().await?;

    // Register for text-changed events specifically
    atspi.register_event::<TextChangedEvent>().await?;
    debug!("Registered for AT-SPI2 TextChangedEvent");

    let events = atspi.event_stream();
    tokio::pin!(events);

    while let Some(item) = events.next().await {
        // The raw accessibility-bus stream also surfaces non-signal traffic
        // (e.g. method replies to our own event registration) as errors such as
        // `MissingInterface`. Those are not fatal — skip them and keep listening
        // rather than tearing down the whole subscription on the first one.
        let ev = match item {
            Ok(ev) => ev,
            Err(e) => {
                debug!("Skipping non-event AT-SPI2 stream item: {}", e);
                continue;
            }
        };

        // Try to extract a TextChangedEvent from the generic Event
        let Ok(text_event) = TextChangedEvent::try_from(ev) else {
            continue;
        };

        let operation = match text_event.operation {
            atspi::Operation::Insert => TextChangeOp::Insert,
            atspi::Operation::Delete => TextChangeOp::Delete,
        };

        // Extract the application name from the source object
        let source_app = text_event.item.name_as_str().unwrap_or("").to_string();

        let event = TextChangeEvent {
            operation,
            start_pos: text_event.start_pos,
            length: text_event.length,
            text: text_event.text,
            timestamp: std::time::Instant::now(),
            source_app,
        };

        if tx.send(event).await.is_err() {
            debug!("Event channel closed, stopping AT-SPI2 listener");
            break;
        }
    }

    Ok(())
}

/// Mock event source for testing. Events are pushed through a sender channel.
pub struct MockTextChangeSource {
    tx: mpsc::Sender<TextChangeEvent>,
    rx: tokio::sync::Mutex<Option<mpsc::Receiver<TextChangeEvent>>>,
    available: bool,
}

impl MockTextChangeSource {
    /// Create a new mock source. Returns the source and a sender for injecting events.
    pub fn new(available: bool) -> (Self, mpsc::Sender<TextChangeEvent>) {
        let (tx, rx) = mpsc::channel(256);
        let source = Self { tx: tx.clone(), rx: tokio::sync::Mutex::new(Some(rx)), available };
        (source, tx)
    }

    /// Get a reference to the sender for injecting events in tests.
    pub fn sender(&self) -> mpsc::Sender<TextChangeEvent> {
        self.tx.clone()
    }
}

#[async_trait]
impl TextChangeSource for MockTextChangeSource {
    async fn subscribe(&self) -> Result<mpsc::Receiver<TextChangeEvent>> {
        let mut rx_guard = self.rx.lock().await;
        rx_guard.take().ok_or_else(|| anyhow::anyhow!("MockTextChangeSource already subscribed"))
    }

    fn is_available(&self) -> bool {
        self.available
    }
}
