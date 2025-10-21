// Unix domain socket IPC client

use anyhow::Result;
use tokio::net::UnixStream;

pub struct IpcClient {
    socket_path: String,
}

impl IpcClient {
    pub fn new(socket_path: String) -> Self {
        Self { socket_path }
    }
    
    pub async fn connect(&self) -> Result<UnixStream> {
        // TODO: Connect to Unix socket
        // TODO: Handle connection errors
        todo!()
    }
    
    pub async fn receive_samples(&mut self) -> Result<Vec<f32>> {
        // TODO: Read 2048 bytes (512 f32 samples)
        // TODO: Parse little-endian f32
        // TODO: Handle disconnects (auto-reconnect)
        todo!()
    }
}
