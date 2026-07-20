//! Event-emitting transcription contract.
//!
//! Transition target: this [`StreamingEngine`] trait is intended to replace the
//! pull-based [`crate::engine::TranscriptionEngine`]. Instead of the main loop
//! polling `get_current_text()` on its own 200ms cadence, an engine *pushes*
//! [`TranscriptEvent`]s on its own cadence and the loop simply consumes them.
//!
//! Local ONNX models are adapted *up* to this push contract via
//! [`LocalEngineDriver`], which owns the re-transcribe cadence on a dedicated
//! **blocking thread** — moving CPU-bound inference off the tokio runtime
//! (the old path ran inference inside a `tokio::spawn`ed task). Hosted engines
//! (e.g. OpenAI) implement [`StreamingEngine`] directly.
//!
//! Building Step 1: this module is additive and unit-tested but not yet wired
//! into the daemon loop (that is Step 2).

use anyhow::Result;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam_channel::{unbounded, Receiver, RecvTimeoutError, Sender};
use tokio::sync::mpsc as tokio_mpsc;
use tracing::{debug, error};

/// ~0.15s @ 16kHz: minimum buffered audio before the first partial transcription.
const MIN_AUDIO_SAMPLES: usize = 2400;
/// ~0.3s @ 16kHz: new audio required since the last partial before re-transcribing.
const RETRANSCRIBE_THRESHOLD: usize = 4800;
/// Wake cadence for the driver worker (matches the old preview-task poll interval).
const PARTIAL_TICK: Duration = Duration::from_millis(200);

/// A transcription update pushed by an engine on its own cadence.
#[derive(Debug, Clone)]
pub enum TranscriptEvent {
    /// Rolling preview. Replaces any prior partial (not additive).
    Partial(String),
    /// Committed transcript for the utterance.
    Final(String),
    /// Engine-level failure. Surfaced to the consumer; **never** an implicit
    /// signal to switch engines — one engine per session, no fallback.
    Error(EngineError),
}

/// An engine-level failure.
#[derive(Debug, Clone)]
pub enum EngineError {
    /// The model failed to produce a transcription.
    Transcription(String),
    /// A backend/transport failure (network, IO, etc.).
    Backend(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::Transcription(m) => write!(f, "transcription error: {m}"),
            EngineError::Backend(m) => write!(f, "backend error: {m}"),
        }
    }
}
impl std::error::Error for EngineError {}

/// The event stream handed to the consumer (daemon loop / GUI bridge).
pub type EventStream = tokio_mpsc::UnboundedReceiver<TranscriptEvent>;

/// Push-based transcription engine: feed audio, consume [`TranscriptEvent`]s.
///
/// Both local (via [`LocalEngineDriver`]) and hosted engines satisfy this, so
/// the daemon loop can consume any engine uniformly without branching on kind.
pub trait StreamingEngine: Send + Sync {
    /// Feed captured audio (16-bit PCM, mono, at the configured sample rate).
    fn process_audio(&self, samples: &[i16]) -> Result<()>;

    /// Subscribe to this engine's event stream for the current session.
    /// Calling again replaces the previous subscription.
    fn subscribe(&self) -> EventStream;

    /// Signal end-of-utterance. The engine finalizes and emits [`TranscriptEvent::Final`].
    fn finish(&self);

    /// Clear all state for a new session.
    fn reset(&self);

    /// A copy of the full captured audio buffer (for debug_audio, etc.).
    fn get_audio_buffer(&self) -> Vec<i16>;
}

/// A pure, synchronous local transcription model: samples in, text out.
///
/// The existing ONNX engines implement this by delegating to their internal
/// `transcribe_buffer`. Buffering and cadence are owned by [`LocalEngineDriver`],
/// so implementors stay stateless with respect to the audio buffer.
pub trait LocalModel: Send + Sync {
    fn transcribe(&self, samples: &[i16]) -> Result<String>;
}

/// Control messages from the [`StreamingEngine`] surface to the worker thread.
enum Ctrl {
    /// New audio was buffered — consider emitting a partial.
    Audio,
    /// End of utterance — emit a final over the full buffer.
    Finish,
    /// New session — reset cadence state.
    Reset,
    /// Tear down the worker thread.
    Shutdown,
}

/// Adapts a synchronous [`LocalModel`] to the push-based [`StreamingEngine`]
/// contract. Owns the audio buffer and runs the throttled re-transcribe loop on
/// a dedicated blocking thread (never on the tokio runtime).
pub struct LocalEngineDriver {
    buffer: Arc<Mutex<Vec<i16>>>,
    event_tx: Arc<Mutex<Option<tokio_mpsc::UnboundedSender<TranscriptEvent>>>>,
    ctrl_tx: Sender<Ctrl>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl LocalEngineDriver {
    pub fn new(model: Arc<dyn LocalModel>) -> Self {
        let buffer: Arc<Mutex<Vec<i16>>> = Arc::new(Mutex::new(Vec::new()));
        let event_tx: Arc<Mutex<Option<tokio_mpsc::UnboundedSender<TranscriptEvent>>>> =
            Arc::new(Mutex::new(None));
        let (ctrl_tx, ctrl_rx) = unbounded();

        let worker = {
            let buffer = Arc::clone(&buffer);
            let event_tx = Arc::clone(&event_tx);
            std::thread::Builder::new()
                .name("local-engine-driver".into())
                .spawn(move || run_worker(model, buffer, event_tx, ctrl_rx))
                .expect("failed to spawn LocalEngineDriver worker thread")
        };

        Self { buffer, event_tx, ctrl_tx, worker: Mutex::new(Some(worker)) }
    }
}

impl StreamingEngine for LocalEngineDriver {
    fn process_audio(&self, samples: &[i16]) -> Result<()> {
        self.buffer
            .lock()
            .map_err(|e| anyhow::anyhow!("audio buffer lock poisoned: {e}"))?
            .extend_from_slice(samples);
        // Worker may have already exited (shutdown); ignore send failure.
        let _ = self.ctrl_tx.send(Ctrl::Audio);
        Ok(())
    }

    fn subscribe(&self) -> EventStream {
        let (tx, rx) = tokio_mpsc::unbounded_channel();
        if let Ok(mut slot) = self.event_tx.lock() {
            *slot = Some(tx);
        }
        rx
    }

    fn finish(&self) {
        let _ = self.ctrl_tx.send(Ctrl::Finish);
    }

    fn reset(&self) {
        if let Ok(mut buf) = self.buffer.lock() {
            buf.clear();
        }
        if let Ok(mut slot) = self.event_tx.lock() {
            *slot = None;
        }
        let _ = self.ctrl_tx.send(Ctrl::Reset);
    }

    fn get_audio_buffer(&self) -> Vec<i16> {
        self.buffer.lock().map(|b| b.clone()).unwrap_or_default()
    }
}

impl Drop for LocalEngineDriver {
    fn drop(&mut self) {
        let _ = self.ctrl_tx.send(Ctrl::Shutdown);
        if let Ok(mut w) = self.worker.lock() {
            if let Some(handle) = w.take() {
                let _ = handle.join();
            }
        }
    }
}

/// Emit an event to the current subscriber, if any.
fn emit(
    slot: &Mutex<Option<tokio_mpsc::UnboundedSender<TranscriptEvent>>>,
    event: TranscriptEvent,
) {
    if let Ok(guard) = slot.lock() {
        if let Some(tx) = guard.as_ref() {
            let _ = tx.send(event);
        }
    }
}

/// The driver's blocking worker loop. Wakes on new audio or every
/// [`PARTIAL_TICK`]; emits [`TranscriptEvent::Partial`] once enough new audio
/// has accrued, and [`TranscriptEvent::Final`] on [`Ctrl::Finish`].
fn run_worker(
    model: Arc<dyn LocalModel>,
    buffer: Arc<Mutex<Vec<i16>>>,
    event_tx: Arc<Mutex<Option<tokio_mpsc::UnboundedSender<TranscriptEvent>>>>,
    ctrl_rx: Receiver<Ctrl>,
) {
    let mut last_len: usize = 0;

    loop {
        match ctrl_rx.recv_timeout(PARTIAL_TICK) {
            Ok(Ctrl::Shutdown) | Err(RecvTimeoutError::Disconnected) => break,

            Ok(Ctrl::Reset) => last_len = 0,

            Ok(Ctrl::Finish) => {
                let audio = match buffer.lock() {
                    Ok(b) => b.clone(),
                    Err(_) => {
                        error!("driver: audio buffer lock poisoned; stopping worker");
                        break;
                    }
                };
                match model.transcribe(&audio) {
                    Ok(text) => emit(&event_tx, TranscriptEvent::Final(text)),
                    Err(e) => emit(
                        &event_tx,
                        TranscriptEvent::Error(EngineError::Transcription(e.to_string())),
                    ),
                }
                last_len = 0;
            }

            Ok(Ctrl::Audio) | Err(RecvTimeoutError::Timeout) => {
                // Decide whether to transcribe without holding the lock over inference.
                let pending = {
                    let Ok(b) = buffer.lock() else {
                        error!("driver: audio buffer lock poisoned; stopping worker");
                        break;
                    };
                    let cur = b.len();
                    if cur < MIN_AUDIO_SAMPLES || cur <= last_len + RETRANSCRIBE_THRESHOLD {
                        None
                    } else {
                        Some((b.clone(), cur))
                    }
                };

                if let Some((audio, cur)) = pending {
                    match model.transcribe(&audio) {
                        Ok(text) => {
                            last_len = cur;
                            emit(&event_tx, TranscriptEvent::Partial(text));
                        }
                        Err(e) => emit(
                            &event_tx,
                            TranscriptEvent::Error(EngineError::Transcription(e.to_string())),
                        ),
                    }
                }
            }
        }
    }

    debug!("LocalEngineDriver worker exiting");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::time::{timeout, Duration as TDuration};

    /// Deterministic model: transcript encodes the sample count it saw.
    struct CountModel {
        calls: Arc<AtomicUsize>,
    }
    impl LocalModel for CountModel {
        fn transcribe(&self, samples: &[i16]) -> Result<String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(format!("len={}", samples.len()))
        }
    }

    struct FailModel;
    impl LocalModel for FailModel {
        fn transcribe(&self, _samples: &[i16]) -> Result<String> {
            anyhow::bail!("boom")
        }
    }

    const RECV_TIMEOUT: TDuration = TDuration::from_secs(3);

    #[tokio::test]
    async fn emits_partial_then_final() {
        let calls = Arc::new(AtomicUsize::new(0));
        let driver = LocalEngineDriver::new(Arc::new(CountModel { calls: calls.clone() }));
        let mut rx = driver.subscribe();

        // Enough new audio to clear MIN + RETRANSCRIBE_THRESHOLD → one partial.
        driver
            .process_audio(&vec![0i16; MIN_AUDIO_SAMPLES + RETRANSCRIBE_THRESHOLD + 1])
            .unwrap();

        let ev = timeout(RECV_TIMEOUT, rx.recv()).await.expect("partial timed out").expect("closed");
        assert!(matches!(ev, TranscriptEvent::Partial(ref t) if t.starts_with("len=")), "got {ev:?}");

        driver.finish();

        // The Final may arrive after another partial tick; drain a few events.
        let mut saw_final = false;
        for _ in 0..5 {
            match timeout(RECV_TIMEOUT, rx.recv()).await {
                Ok(Some(TranscriptEvent::Final(_))) => {
                    saw_final = true;
                    break;
                }
                Ok(Some(_)) => continue,
                _ => break,
            }
        }
        assert!(saw_final, "expected a Final event");
    }

    #[tokio::test]
    async fn finish_emits_final_below_partial_threshold() {
        let calls = Arc::new(AtomicUsize::new(0));
        let driver = LocalEngineDriver::new(Arc::new(CountModel { calls }));
        let mut rx = driver.subscribe();

        // Below MIN_AUDIO_SAMPLES → no partial, but finish must still finalize.
        driver.process_audio(&vec![0i16; 1000]).unwrap();
        driver.finish();

        let ev = timeout(RECV_TIMEOUT, rx.recv()).await.expect("final timed out").expect("closed");
        assert!(matches!(ev, TranscriptEvent::Final(ref t) if t == "len=1000"), "got {ev:?}");
    }

    #[tokio::test]
    async fn transcription_error_surfaces() {
        let driver = LocalEngineDriver::new(Arc::new(FailModel));
        let mut rx = driver.subscribe();

        driver.finish(); // triggers a final transcribe → error
        let ev = timeout(RECV_TIMEOUT, rx.recv()).await.expect("error timed out").expect("closed");
        assert!(matches!(ev, TranscriptEvent::Error(_)), "got {ev:?}");
    }

    #[test]
    fn reset_clears_buffer() {
        let calls = Arc::new(AtomicUsize::new(0));
        let driver = LocalEngineDriver::new(Arc::new(CountModel { calls }));
        driver.process_audio(&vec![1i16; 100]).unwrap();
        assert_eq!(driver.get_audio_buffer().len(), 100);
        driver.reset();
        assert!(driver.get_audio_buffer().is_empty());
    }
}
