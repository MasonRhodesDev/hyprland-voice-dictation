/// GUI control and status types for daemon ↔ GUI communication

/// Commands sent from daemon to GUI
#[derive(Debug, Clone)]
pub enum GuiControl {
    /// Initialize GUI (hidden, ready to show on demand)
    Initialize,

    /// Set GUI to hidden state (windows exist but invisible)
    SetHidden,

    /// Set GUI to listening mode (spectrum + transcription)
    SetListening,

    /// Update transcription text during listening
    UpdateTranscription {
        text: String,
        is_final: bool,
    },

    /// Update spectrum visualization data
    /// Frequency band values (typically 8-10 bands, 0.0-1.0 range)
    UpdateSpectrum(Vec<f32>),

    /// Update VAD (voice activity detection) state
    /// Used to sync visual feedback with actual speech detection
    UpdateVadState {
        /// Whether voice activity is currently detected
        is_speaking: bool,
        /// Whether transcription text has settled (no changes for 300ms+)
        text_settled: bool,
    },

    /// Transition to transcribing state (ASR running; spinner)
    SetTranscribing,

    /// Transition to / update typing state as the result is injected.
    /// `done`/`total` are word counts so the GUI can show real progress and
    /// observers can see the work is alive and advancing (not wedged).
    SetTyping {
        done: usize,
        total: usize,
    },

    /// Transition to closing state and begin shutdown animation
    SetClosing,

    /// Force immediate exit (for errors/cleanup)
    Exit,
}

/// Status messages sent from GUI to daemon
#[derive(Debug, Clone)]
pub enum GuiStatus {
    /// GUI has initialized and is ready
    Ready,

    /// State transition animation completed
    TransitionComplete {
        from: GuiState,
        to: GuiState,
    },

    /// GUI encountered an error
    Error(String),

    /// GUI is shutting down
    ShuttingDown,
}

/// GUI state (shared type for status messages)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuiState {
    Hidden,
    PreListening,
    Listening,
    /// ASR running — spinner, no progress known yet.
    Transcribing,
    /// Injecting the result — spinner plus typing progress (see SharedState).
    Typing,
    Closing,
}
