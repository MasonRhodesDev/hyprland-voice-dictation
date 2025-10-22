use anyhow::{Context, Result};
use tokio::io::AsyncReadExt;
use tokio::net::UnixStream;
use tracing::{debug, warn};

const SAMPLES_PER_MESSAGE: usize = 512;
const BYTES_PER_MESSAGE: usize = SAMPLES_PER_MESSAGE * 4;

pub struct IpcClient {
    socket_path: String,
    stream: Option<UnixStream>,
}

impl IpcClient {
    pub fn new(socket_path: String) -> Self {
        Self {
            socket_path,
            stream: None,
        }
    }

    pub async fn connect(&mut self) -> Result<()> {
        debug!("Connecting to IPC socket: {}", self.socket_path);
        let stream = UnixStream::connect(&self.socket_path)
            .await
            .context("Failed to connect to IPC socket")?;
        self.stream = Some(stream);
        debug!("Connected to IPC socket");
        Ok(())
    }

    pub async fn receive_samples(&mut self) -> Result<Vec<f32>> {
        let stream = self
            .stream
            .as_mut()
            .context("Not connected to IPC socket")?;

        let mut buffer = [0u8; BYTES_PER_MESSAGE];
        stream
            .read_exact(&mut buffer)
            .await
            .context("Failed to read from IPC socket")?;

        let samples: Vec<f32> = buffer
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(samples)
    }

    pub async fn reconnect(&mut self) -> Result<()> {
        warn!("Attempting to reconnect to IPC socket");
        self.stream = None;
        self.connect().await
    }
}
