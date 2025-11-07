use std::sync::{Arc, RwLock};
use tracing::{debug, error, info, trace};
use tokio::runtime::Runtime;

use crate::{control_ipc, fft, ipc, shared_state::SharedState, GuiState};

/// Spawn background task for audio IPC (spectrum data)
pub fn spawn_audio_task(shared_state: Arc<RwLock<SharedState>>) {
    std::thread::spawn(move || {
        info!("Starting audio IPC background task");

        let rt = Runtime::new().expect("Failed to create tokio runtime for audio task");
        rt.block_on(async move {
            let config = crate::config::load_config();
            let mut ipc_client = ipc::IpcClient::new(crate::SOCKET_PATH.to_string());
            let mut spectrum_analyzer = fft::SpectrumAnalyzer::new(
                crate::FFT_SIZE,
                crate::SAMPLE_RATE,
                config.elements.spectrum_smoothing_factor,
            );

            loop {
                debug!("Attempting to connect to audio socket...");
                if ipc_client.connect().await.is_err() {
                    trace!("Audio connect failed, retrying in 100ms");
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    continue;
                }

                info!("Connected to audio socket");

                loop {
                    trace!("Waiting for audio samples...");
                    match ipc_client.receive_samples().await {
                        Ok(samples) => {
                            trace!("Received {} audio samples", samples.len());
                            let spectrum_values = spectrum_analyzer.process(&samples);
                            debug!("Processed spectrum, updating shared state");

                            if let Ok(mut state) = shared_state.write() {
                                state.set_spectrum_values(spectrum_values);
                            } else {
                                error!("Failed to acquire write lock for spectrum update");
                            }
                        }
                        Err(e) => {
                            error!("Audio IPC error: {}. Reconnecting...", e);
                            break;
                        }
                    }
                }
            }
        });
    });
}

/// Spawn background task for control IPC (state and transcription)
pub fn spawn_control_task(shared_state: Arc<RwLock<SharedState>>) {
    std::thread::spawn(move || {
        info!("Starting control IPC background task");

        let rt = Runtime::new().expect("Failed to create tokio runtime for control task");
        rt.block_on(async move {
            let mut control_client =
                control_ipc::ControlClient::new(crate::CONTROL_SOCKET_PATH.to_string());

            loop {
                debug!("Attempting to connect to control socket...");
                if control_client.connect().await.is_err() {
                    error!("Failed to connect to control socket, retrying in 1s");
                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                    continue;
                }

                info!("Connected to control socket");

                loop {
                    trace!("Waiting for control message...");

                    let receive_result = tokio::time::timeout(
                        tokio::time::Duration::from_secs(5),
                        control_client.receive(),
                    )
                    .await;

                    match receive_result {
                        Ok(Ok(control_ipc::ControlMessage::TranscriptionUpdate {
                            text,
                            is_final,
                        })) => {
                            info!("Control task: Transcription '{}' (final: {})", text, is_final);

                            if let Ok(mut state) = shared_state.write() {
                                state.set_transcription(text);
                            } else {
                                error!("Failed to acquire write lock for transcription update");
                            }
                        }
                        Ok(Ok(control_ipc::ControlMessage::Ready)) => {
                            debug!("Control task: Engine ready");
                        }
                        Ok(Ok(control_ipc::ControlMessage::ProcessingStarted)) => {
                            info!("Control task: ProcessingStarted received");

                            if let Ok(mut state) = shared_state.write() {
                                state.set_gui_state(GuiState::Processing);
                                state.reset_animations();
                            } else {
                                error!("Failed to acquire write lock for state update");
                            }
                        }
                        Ok(Ok(control_ipc::ControlMessage::Complete)) => {
                            info!("Control task: Complete received");

                            if let Ok(mut state) = shared_state.write() {
                                state.set_gui_state(GuiState::Closing);
                                state.reset_animations();
                            } else {
                                error!("Failed to acquire write lock for state update");
                            }
                        }
                        Ok(Ok(control_ipc::ControlMessage::Confirm)) => {
                            debug!("Control task: Confirm received (ignored)");
                        }
                        Ok(Err(e)) => {
                            error!("Control task: receive error: {}", e);
                            break;
                        }
                        Err(_) => {
                            trace!("Control task: Timeout waiting for message (normal)");
                        }
                    }
                }

                debug!("Control task inner loop exited, reconnecting...");
            }
        });
    });
}
