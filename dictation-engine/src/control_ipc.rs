use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlMessage {
    // Legacy messages (keep for compatibility)
    Ready,
    TranscriptionUpdate { text: String, is_final: bool },
    Confirm,
    ProcessingStarted,
    Complete,

    // Session control messages (CLI → Daemon)
    StartRecording,
    StopRecording,
    StatusQuery,
    StatusResponse {
        state: String,
        session_active: bool
    },
    Shutdown,
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

        Ok(Self { listener, clients: Vec::new() })
    }

    pub async fn broadcast(&mut self, msg: &ControlMessage) -> Result<()> {
        let data = serde_json::to_vec(msg)?;
        let len = data.len() as u32;

        let mut disconnected = Vec::new();

        for (idx, client) in self.clients.iter_mut().enumerate() {
            if client.write_u32(len).await.is_err() || client.write_all(&data).await.is_err() {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_message_ready_serialize() {
        let msg = ControlMessage::Ready;
        let json = serde_json::to_string(&msg);
        assert!(json.is_ok());
    }

    #[test]
    fn test_control_message_confirm_serialize() {
        let msg = ControlMessage::Confirm;
        let json = serde_json::to_string(&msg);
        assert!(json.is_ok());
    }

    #[test]
    fn test_control_message_transcription_serialize() {
        let msg =
            ControlMessage::TranscriptionUpdate { text: "test text".to_string(), is_final: true };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("test text"));
    }

    #[test]
    fn test_control_message_roundtrip() {
        let original = ControlMessage::TranscriptionUpdate {
            text: "hello world".to_string(),
            is_final: false,
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: ControlMessage = serde_json::from_str(&json).unwrap();

        match parsed {
            ControlMessage::TranscriptionUpdate { text, is_final } => {
                assert_eq!(text, "hello world");
                assert!(!is_final);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[tokio::test]
    async fn test_control_server_new() {
        let socket_path = "/tmp/test_control_server_12345.sock";
        let _ = std::fs::remove_file(socket_path);

        let result = ControlServer::new(socket_path).await;
        assert!(result.is_ok());

        if let Ok(_server) = result {
            let _ = std::fs::remove_file(socket_path);
        }
    }

    #[tokio::test]
    async fn test_control_server_broadcast_no_clients() {
        let socket_path = "/tmp/test_control_broadcast_12345.sock";
        let _ = std::fs::remove_file(socket_path);

        let mut server = ControlServer::new(socket_path).await.unwrap();
        let result = server.broadcast(&ControlMessage::Ready).await;
        assert!(result.is_ok());

        let _ = std::fs::remove_file(socket_path);
    }
}
