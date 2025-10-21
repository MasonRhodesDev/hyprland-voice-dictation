use anyhow::Result;
use tracing::info;

mod wayland;
mod renderer;
mod fft;
mod ipc;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    
    info!("Starting dictation-gui");
    
    // TODO: Initialize Wayland
    // TODO: Connect to IPC socket
    // TODO: Run event loop
    
    Ok(())
}
