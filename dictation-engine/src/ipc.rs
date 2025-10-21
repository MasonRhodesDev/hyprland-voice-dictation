// Unix domain socket IPC server

use anyhow::Result;
use tokio::net::UnixListener;

pub struct IpcServer {
    socket_path: String,
}

impl IpcServer {
    pub fn new(socket_path: String) -> Self {
        Self { socket_path }
    }
    
    pub async fn start(&self) -> Result<()> {
        // TODO: Remove existing socket file
        // TODO: Create UnixListener
        // TODO: Accept connections
        // TODO: Spawn task per connection
        // TODO: Broadcast audio samples
        todo!()
    }
    
    pub async fn broadcast_samples(&self, samples: &[f32]) -> Result<()> {
        // TODO: Send to all connected clients
        // TODO: Non-blocking send
        // TODO: Remove disconnected clients
        todo!()
    }
}
