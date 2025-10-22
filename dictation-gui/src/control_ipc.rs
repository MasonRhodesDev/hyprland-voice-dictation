use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlMessage {
    Ready,
    TranscriptionUpdate { text: String, is_final: bool },
    Confirm,
}

pub struct ControlClient {
    pub stream: Option<UnixStream>,
    pub socket_path: String,
}

impl ControlClient {
    pub fn new(socket_path: String) -> Self {
        Self {
            stream: None,
            socket_path,
        }
    }

    pub async fn connect(&mut self) -> Result<()> {
        let stream = UnixStream::connect(&self.socket_path).await?;
        self.stream = Some(stream);
        info!("Connected to control server");
        Ok(())
    }

    #[allow(dead_code)]
    pub async fn send(&mut self, msg: &ControlMessage) -> Result<()> {
        if let Some(stream) = &mut self.stream {
            let data = serde_json::to_vec(msg)?;
            let len = data.len() as u32;
            stream.write_u32(len).await?;
            stream.write_all(&data).await?;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Not connected"))
        }
    }

    pub async fn receive(&mut self) -> Result<ControlMessage> {
        if let Some(stream) = &mut self.stream {
            let len = stream.read_u32().await?;
            let mut buffer = vec![0u8; len as usize];
            stream.read_exact(&mut buffer).await?;
            let msg = serde_json::from_slice(&buffer)?;
            Ok(msg)
        } else {
            Err(anyhow::anyhow!("Not connected"))
        }
    }

    #[allow(dead_code)]
    pub async fn _reconnect(&mut self) -> Result<()> {
        self.stream = None;
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        self.connect().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_message_serialize() {
        let msg = ControlMessage::Ready;
        let serialized = serde_json::to_string(&msg);
        assert!(serialized.is_ok());
    }

    #[test]
    fn test_control_message_deserialize() {
        let json = r#"{"Ready":null}"#;
        let msg: Result<ControlMessage, _> = serde_json::from_str(json);
        assert!(msg.is_ok());
    }

    #[test]
    fn test_control_message_transcription_update() {
        let msg = ControlMessage::TranscriptionUpdate {
            text: "hello".to_string(),
            is_final: false,
        };
        let serialized = serde_json::to_string(&msg).unwrap();
        let deserialized: ControlMessage = serde_json::from_str(&serialized).unwrap();
        
        match deserialized {
            ControlMessage::TranscriptionUpdate { text, is_final } => {
                assert_eq!(text, "hello");
                assert!(!is_final);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_control_client_new() {
        let client = ControlClient::new("/tmp/test_control.sock".to_string());
        assert!(client.stream.is_none());
        assert_eq!(client.socket_path, "/tmp/test_control.sock");
    }

    #[tokio::test]
    async fn test_control_client_connect_nonexistent() {
        let mut client = ControlClient::new("/tmp/nonexistent_control_12345.sock".to_string());
        let result = client.connect().await;
        assert!(result.is_err());
    }
}
