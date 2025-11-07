use crate::GuiState;
use std::sync::{Arc, RwLock};

/// Shared state synchronized across all per-monitor windows and background tasks
#[derive(Debug, Clone)]
pub struct SharedState {
    /// Current GUI state controlled by the daemon
    pub gui_state: GuiState,

    /// Current transcription text from the engine
    pub transcription: String,

    /// Spectrum frequency band values for visualization
    pub spectrum_values: Vec<f32>,

    /// Name of the currently active monitor (e.g., "DP-1", "HDMI-A-1")
    pub active_monitor: String,

    /// Animation timer for intro/transition effects
    pub animation_time: f32,

    /// Animation timer for closing effect
    pub closing_animation_time: f32,
}

impl Default for SharedState {
    fn default() -> Self {
        Self {
            gui_state: GuiState::Hidden,
            transcription: String::new(),
            spectrum_values: vec![0.0; 10], // Default 10 bands
            active_monitor: String::new(),
            animation_time: 0.0,
            closing_animation_time: 0.0,
        }
    }
}

impl SharedState {
    /// Create a new shared state wrapped in Arc<RwLock<>>
    pub fn new() -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(Self::default()))
    }

    /// Update GUI state
    pub fn set_gui_state(&mut self, state: GuiState) {
        self.gui_state = state;
    }

    /// Update transcription text
    pub fn set_transcription(&mut self, text: String) {
        self.transcription = text;
    }

    /// Update spectrum values
    pub fn set_spectrum_values(&mut self, values: Vec<f32>) {
        self.spectrum_values = values;
    }

    /// Update active monitor name
    pub fn set_active_monitor(&mut self, monitor: String) {
        self.active_monitor = monitor;
    }

    /// Increment animation timers
    pub fn tick(&mut self, delta: f32) {
        self.animation_time += delta;
        if self.gui_state == GuiState::Closing {
            self.closing_animation_time += delta;
        }
    }

    /// Reset animation timers on state change
    pub fn reset_animations(&mut self) {
        self.animation_time = 0.0;
        self.closing_animation_time = 0.0;
    }
}
