use anyhow::Result;
use tracing::info;

mod audio;
mod vad;
mod whisper;
mod keyboard;
mod ipc;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    
    info!("Starting dictation-engine");
    
    // TODO: Load configuration
    // TODO: Initialize audio capture
    // TODO: Start IPC server
    // TODO: Run VAD loop
    // TODO: Handle signals
    
    Ok(())
}
