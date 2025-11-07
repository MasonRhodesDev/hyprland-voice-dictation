/// Channel-based communication with daemon (replaces socket polling)

use dictation_types::{GuiControl, GuiStatus};
use std::sync::{Arc, RwLock};
use tokio::sync::{broadcast, mpsc};
use tracing::{debug, error, info};

use crate::{shared_state::SharedState, GuiState};

/// Spawn background tasks to listen for channel messages and update SharedState
pub fn spawn_channel_listener(
    mut gui_control_rx: broadcast::Receiver<GuiControl>,
    mut spectrum_rx: broadcast::Receiver<Vec<f32>>,
    shared_state: Arc<RwLock<SharedState>>,
    gui_status_tx: mpsc::Sender<GuiStatus>,
) {
    // Spawn control message listener
    let state_clone = shared_state.clone();
    let status_tx_clone = gui_status_tx.clone();
    tokio::spawn(async move {
        info!("Channel listener: Control task started");

        loop {
            match gui_control_rx.recv().await {
                Ok(GuiControl::Initialize) => {
                    info!("Channel listener: Initialize received");
                    if let Ok(mut state) = state_clone.write() {
                        state.set_gui_state(GuiState::Hidden);
                    }
                }
                Ok(GuiControl::SetHidden) => {
                    info!("Channel listener: SetHidden received");
                    if let Ok(mut state) = state_clone.write() {
                        state.set_gui_state(GuiState::Hidden);
                    }
                }
                Ok(GuiControl::SetListening) => {
                    info!("Channel listener: SetListening received");
                    if let Ok(mut state) = state_clone.write() {
                        state.set_gui_state(GuiState::Listening);
                    }
                }
                Ok(GuiControl::UpdateTranscription { text, is_final }) => {
                    debug!(
                        "Channel listener: UpdateTranscription '{}' (final: {})",
                        text, is_final
                    );
                    if let Ok(mut state) = state_clone.write() {
                        state.set_transcription(text);
                    }
                }
                Ok(GuiControl::UpdateSpectrum(values)) => {
                    // Spectrum updates are high-frequency, don't log at debug level
                    if let Ok(mut state) = state_clone.write() {
                        state.set_spectrum_values(values);
                    }
                }
                Ok(GuiControl::SetProcessing) => {
                    info!("Channel listener: SetProcessing received");
                    if let Ok(mut state) = state_clone.write() {
                        state.set_gui_state(GuiState::Processing);
                        state.reset_animations();
                    }
                }
                Ok(GuiControl::SetClosing) => {
                    info!("Channel listener: SetClosing received");
                    if let Ok(mut state) = state_clone.write() {
                        state.set_gui_state(GuiState::Closing);
                        state.reset_animations();
                    }
                }
                Ok(GuiControl::Exit) => {
                    info!("Channel listener: Exit received");
                    let _ = status_tx_clone.send(GuiStatus::ShuttingDown).await;
                    std::process::exit(0);
                }
                Err(broadcast::error::RecvError::Lagged(skipped)) => {
                    debug!("Channel listener: Lagged by {} messages", skipped);
                    continue;
                }
                Err(broadcast::error::RecvError::Closed) => {
                    error!("Channel listener: Control channel closed");
                    break;
                }
            }
        }

        error!("Channel listener: Control task exiting");
    });

    // Spawn spectrum listener
    tokio::spawn(async move {
        debug!("Channel listener: Spectrum task started");

        loop {
            match spectrum_rx.recv().await {
                Ok(values) => {
                    if let Ok(mut state) = shared_state.write() {
                        state.set_spectrum_values(values);
                    }
                }
                Err(broadcast::error::RecvError::Lagged(skipped)) => {
                    debug!("Channel listener: Spectrum lagged by {} messages", skipped);
                    continue;
                }
                Err(broadcast::error::RecvError::Closed) => {
                    error!("Channel listener: Spectrum channel closed");
                    break;
                }
            }
        }

        error!("Channel listener: Spectrum task exiting");
    });

    info!("Channel listeners spawned");
}
