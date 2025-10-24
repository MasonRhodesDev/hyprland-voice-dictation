use iced::futures::SinkExt;
use iced::{stream, Subscription};
use std::hash::Hash;
use tracing::{debug, error, info, trace};

use crate::{control_ipc, fft, ipc, GuiState, Message};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IpcSubscriptionKind {
    Audio,
    Control,
}

pub fn audio_subscription() -> Subscription<Message> {
    #[derive(Hash)]
    struct AudioIpc;
    
    trace!("audio_subscription() called");
    
    Subscription::run_with_id(
        std::any::TypeId::of::<AudioIpc>(),
        stream::channel(100, move |mut output| async move {
            info!("Audio subscription stream starting");
            let mut ipc_client = ipc::IpcClient::new(crate::SOCKET_PATH.to_string());
            let mut spectrum_analyzer = fft::SpectrumAnalyzer::new(crate::FFT_SIZE, crate::SAMPLE_RATE);
            
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
                            debug!("Processed spectrum, sending update");
                            let _ = output.send(Message::SpectrumUpdate(spectrum_values)).await;
                        }
                        Err(e) => {
                            error!("Audio IPC error: {}. Reconnecting...", e);
                            let _ = output.send(Message::IpcError(format!("Audio: {}", e))).await;
                            break;
                        }
                    }
                }
            }
        }),
    )
}

pub fn control_subscription() -> Subscription<Message> {
    #[derive(Hash)]
    struct ControlIpc;
    
    trace!("control_subscription() called");
    
    Subscription::run_with_id(
        std::any::TypeId::of::<ControlIpc>(),
        stream::channel(100, move |mut output| async move {
            info!("Control subscription stream starting");
            let mut control_client = control_ipc::ControlClient::new(crate::CONTROL_SOCKET_PATH.to_string());
            
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
                    eprintln!("Waiting for control message...");
                    
                    let receive_result = tokio::time::timeout(
                        tokio::time::Duration::from_secs(5),
                        control_client.receive()
                    ).await;
                    
                    match receive_result {
                        Ok(Ok(control_ipc::ControlMessage::TranscriptionUpdate { text, is_final })) => {
                            info!("SUBSCRIPTION: Transcription '{}' (final: {})", text, is_final);
                            debug!("SUBSCRIPTION: Sending TranscriptionUpdate to app");
                            let send_result = output.send(Message::TranscriptionUpdate(text)).await;
                            trace!("SUBSCRIPTION: Send result: {:?}", send_result);
                        }
                        Ok(Ok(control_ipc::ControlMessage::Ready)) => {
                            debug!("SUBSCRIPTION: Engine ready");
                        }
                        Ok(Ok(control_ipc::ControlMessage::ProcessingStarted)) => {
                            info!("SUBSCRIPTION: ProcessingStarted received");
                            eprintln!("SUBSCRIPTION: ProcessingStarted received");
                            debug!("SUBSCRIPTION: Sending StateChange(Processing) to app");
                            let send_result = output.send(Message::StateChange(GuiState::Processing)).await;
                            eprintln!("SUBSCRIPTION: Send result: {:?}", send_result);
                            trace!("SUBSCRIPTION: Send result: {:?}", send_result);
                        }
                        Ok(Ok(control_ipc::ControlMessage::Complete)) => {
                            info!("SUBSCRIPTION: Complete received");
                            eprintln!("SUBSCRIPTION: Complete received");
                            debug!("SUBSCRIPTION: Sending StateChange(Closing) to app");
                            let send_result = output.send(Message::StateChange(GuiState::Closing)).await;
                            eprintln!("SUBSCRIPTION: Send result: {:?}", send_result);
                            trace!("SUBSCRIPTION: Send result: {:?}", send_result);
                        }
                        Ok(Ok(control_ipc::ControlMessage::Confirm)) => {
                            debug!("SUBSCRIPTION: Confirm received (ignored)");
                        }
                        Ok(Err(e)) => {
                            error!("SUBSCRIPTION: Control receive error: {}", e);
                            eprintln!("SUBSCRIPTION: Control receive error: {}", e);
                            let _ = output.send(Message::IpcError(format!("Control: {}", e))).await;
                            break;
                        }
                        Err(_) => {
                            error!("SUBSCRIPTION: Timeout waiting for control message");
                            eprintln!("SUBSCRIPTION: Timeout waiting for control message");
                        }
                    }
                }
                
                debug!("Control subscription inner loop exited, reconnecting...");
            }
        }),
    )
}
