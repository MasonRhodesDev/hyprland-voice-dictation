use anyhow::Result;
use zbus::{interface, ConnectionBuilder};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tokio::sync::{Mutex, watch};
use tracing::info;

use crate::HealthState;

/// Daemon state enum shared between lib.rs and dbus_control.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DaemonState {
    Idle,        // Waiting for StartRecording command, GUI hidden
    Recording,   // Actively recording audio and transcribing, GUI visible
    Processing,  // Running transcription and typing, GUI visible with spinner
}

impl std::fmt::Display for DaemonState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DaemonState::Idle => write!(f, "idle"),
            DaemonState::Recording => write!(f, "recording"),
            DaemonState::Processing => write!(f, "processing"),
        }
    }
}

/// D-Bus service interface for voice dictation control
pub struct VoiceDictationService {
    command_sender: Arc<Mutex<tokio::sync::mpsc::Sender<DaemonCommand>>>,
    state_receiver: watch::Receiver<DaemonState>,
    health_state: Arc<HealthState>,
}

/// Commands that can be sent from D-Bus to the daemon
#[derive(Debug, Clone)]
pub enum DaemonCommand {
    StartRecording,
    StopRecording,
    Confirm,
    Shutdown,
}

/// Response from status query
#[derive(Debug, Clone)]
pub struct StatusInfo {
    pub state: String,
    pub session_active: bool,
}

#[interface(name = "com.voicedictation.Control")]
impl VoiceDictationService {
    /// Start a new recording session
    async fn start_recording(&self) -> zbus::fdo::Result<()> {
        info!("D-Bus: StartRecording called");
        let sender = self.command_sender.lock().await;
        sender.send(DaemonCommand::StartRecording).await
            .map_err(|e| zbus::fdo::Error::Failed(format!("Failed to send command: {}", e)))?;
        Ok(())
    }

    /// Stop the current recording session (cancel)
    async fn stop_recording(&self) -> zbus::fdo::Result<()> {
        info!("D-Bus: StopRecording called");
        let sender = self.command_sender.lock().await;
        sender.send(DaemonCommand::StopRecording).await
            .map_err(|e| zbus::fdo::Error::Failed(format!("Failed to send command: {}", e)))?;
        Ok(())
    }

    /// Confirm and finalize the current transcription
    async fn confirm(&self) -> zbus::fdo::Result<()> {
        info!("D-Bus: Confirm called");
        let sender = self.command_sender.lock().await;
        sender.send(DaemonCommand::Confirm).await
            .map_err(|e| zbus::fdo::Error::Failed(format!("Failed to send command: {}", e)))?;
        Ok(())
    }

    /// Get current daemon status
    async fn status(&self) -> zbus::fdo::Result<(String, bool)> {
        info!("D-Bus: Status called");
        let state = *self.state_receiver.borrow();
        let session_active = state != DaemonState::Idle;
        Ok((state.to_string(), session_active))
    }

    /// Get health status of all subsystems
    async fn health_check(&self) -> zbus::fdo::Result<(String, String, String)> {
        info!("D-Bus: HealthCheck called");

        let gui_status = if self.health_state.gui_healthy.load(Ordering::Relaxed) {
            "healthy"
        } else {
            "unavailable"
        };

        let engine_status = if self.health_state.engine_healthy.load(Ordering::Relaxed) {
            "healthy"
        } else {
            "not loaded"
        };

        let audio_status = if self.health_state.audio_healthy.load(Ordering::Relaxed) {
            "healthy"
        } else {
            let state = *self.state_receiver.borrow();
            if state == DaemonState::Recording {
                "unhealthy"
            } else {
                "idle"
            }
        };

        Ok((gui_status.to_string(), engine_status.to_string(), audio_status.to_string()))
    }

    /// Shutdown the daemon gracefully
    async fn shutdown(&self) -> zbus::fdo::Result<()> {
        info!("D-Bus: Shutdown called");
        let sender = self.command_sender.lock().await;
        sender.send(DaemonCommand::Shutdown).await
            .map_err(|e| zbus::fdo::Error::Failed(format!("Failed to send command: {}", e)))?;
        Ok(())
    }
}

/// Create and register D-Bus service
pub async fn create_dbus_service(
    state_receiver: watch::Receiver<DaemonState>,
    health_state: Arc<HealthState>,
) -> Result<(
    zbus::Connection,
    Arc<Mutex<tokio::sync::mpsc::Sender<DaemonCommand>>>,
    tokio::sync::mpsc::Receiver<DaemonCommand>,
)> {
    let (command_tx, command_rx) = tokio::sync::mpsc::channel(10);
    let command_sender = Arc::new(Mutex::new(command_tx));

    let service = VoiceDictationService {
        command_sender: Arc::clone(&command_sender),
        state_receiver,
        health_state,
    };

    let connection = ConnectionBuilder::session()?
        .name("com.voicedictation.Daemon")?
        .serve_at("/com/voicedictation/Control", service)?
        .build()
        .await?;

    info!("D-Bus service registered at com.voicedictation.Daemon");

    Ok((connection, command_sender, command_rx))
}
