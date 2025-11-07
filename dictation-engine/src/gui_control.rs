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

    /// Transition to processing state (spinner animation)
    SetProcessing,

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
    Processing,
    Closing,
}
