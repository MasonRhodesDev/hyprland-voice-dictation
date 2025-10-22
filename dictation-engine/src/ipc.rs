use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Mutex;
use tracing::{debug, error, info};

const SAMPLES_PER_MESSAGE: usize = 512;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum EngineMessage {
    TranscriptionUpdate { text: String, is_final: bool },
    Ready,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum GuiCommand {
    Confirm, // User pressed MEH+v to confirm and type
}

pub struct IpcServer {
    socket_path: String,
    clients: Arc<Mutex<Vec<UnixStream>>>,
}

impl IpcServer {
    pub fn new(socket_path: String) -> Self {
        Self {
            socket_path,
            clients: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn start_server(self: &Arc<Self>) {
        let server = self.clone();
        tokio::spawn(async move {
            if let Err(e) = server.run_server().await {
                error!("IPC server error: {}", e);
            }
        });
    }

    async fn run_server(&self) -> Result<()> {
        let _ = std::fs::remove_file(&self.socket_path);

        let listener = UnixListener::bind(&self.socket_path)
            .context("Failed to bind Unix socket")?;
        info!("IPC server listening on {}", self.socket_path);

        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    info!("New IPC client connected");
                    let mut clients = self.clients.lock().await;
                    clients.push(stream);
                }
                Err(e) => {
                    error!("Failed to accept connection: {}", e);
                }
            }
        }
    }

    pub async fn broadcast_samples(&self, samples: &[f32]) {
        if samples.len() != SAMPLES_PER_MESSAGE {
            debug!("Wrong sample count: {} (expected {})", samples.len(), SAMPLES_PER_MESSAGE);
            return;
        }
        
        let client_count = self.clients.lock().await.len();
        if client_count > 0 {
            debug!("Broadcasting {} samples to {} clients", samples.len(), client_count);
            self.send_to_clients(samples).await;
        }
    }

    async fn send_to_clients(&self, samples: &[f32]) {
        let mut clients = self.clients.lock().await;
        let mut to_remove = Vec::new();

        let bytes: Vec<u8> = samples
            .iter()
            .flat_map(|&s| s.to_le_bytes())
            .collect();

        for (i, client) in clients.iter_mut().enumerate() {
            if let Err(e) = client.write_all(&bytes).await {
                debug!("Failed to send to client {}: {}", i, e);
                to_remove.push(i);
            }
        }

        for &i in to_remove.iter().rev() {
            clients.remove(i);
            info!("Client {} disconnected", i);
        }
    }
}

impl Drop for IpcServer {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.socket_path);
    }
}
