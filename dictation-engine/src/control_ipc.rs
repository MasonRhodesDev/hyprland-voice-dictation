use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlMessage {
    Ready,
    TranscriptionUpdate { text: String, is_final: bool },
    Confirm,
}

pub struct ControlServer {
    listener: UnixListener,
    clients: Vec<UnixStream>,
}

impl ControlServer {
    pub async fn new(socket_path: &str) -> Result<Self> {
        if Path::new(socket_path).exists() {
            std::fs::remove_file(socket_path)?;
        }

        let listener = UnixListener::bind(socket_path)?;
        info!("Control IPC server listening on {}", socket_path);

        Ok(Self {
            listener,
            clients: Vec::new(),
        })
    }

    pub async fn _accept_client(&mut self) -> Result<()> {
        match self.listener.accept().await {
            Ok((stream, _)) => {
                info!("Control client connected");
                self.clients.push(stream);
                Ok(())
            }
            Err(e) => Err(e.into()),
        }
    }

    pub async fn broadcast(&mut self, msg: &ControlMessage) -> Result<()> {
        let data = serde_json::to_vec(msg)?;
        let len = data.len() as u32;

        let mut disconnected = Vec::new();

        for (idx, client) in self.clients.iter_mut().enumerate() {
            if client.write_u32(len).await.is_err()
                || client.write_all(&data).await.is_err()
            {
                disconnected.push(idx);
            }
        }

        for idx in disconnected.iter().rev() {
            self.clients.remove(*idx);
        }

        Ok(())
    }

    pub async fn try_accept(&mut self) {
        tokio::select! {
            result = self.listener.accept() => {
                if let Ok((stream, _)) = result {
                    info!("Control client connected");
                    self.clients.push(stream);
                }
            }
            _ = tokio::time::sleep(tokio::time::Duration::from_millis(10)) => {}
        }
    }

    pub async fn receive_from_any(&mut self) -> Option<ControlMessage> {
        if self.clients.is_empty() {
            return None;
        }

        let mut buffer = vec![0u8; 4];
        let mut disconnected = Vec::new();

        for (idx, client) in self.clients.iter_mut().enumerate() {
            match client.try_read(&mut buffer) {
                Ok(0) => {
                    disconnected.push(idx);
                }
                Ok(n) if n >= 4 => {
                    let len = u32::from_be_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
                    let mut msg_buf = vec![0u8; len as usize];
                    
                    match client.read_exact(&mut msg_buf).await {
                        Ok(_) => {
                            if let Ok(msg) = serde_json::from_slice(&msg_buf) {
                                return Some(msg);
                            }
                        }
                        Err(_) => disconnected.push(idx),
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(_) => disconnected.push(idx),
                _ => {}
            }
        }

        for idx in disconnected.iter().rev() {
            info!("Control client disconnected");
            self.clients.remove(*idx);
        }

        None
    }
}


