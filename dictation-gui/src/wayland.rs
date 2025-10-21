// Wayland layer-shell surface setup

use anyhow::Result;

pub struct WaylandSurface {
    // TODO: smithay-client-toolkit types
}

impl WaylandSurface {
    pub fn new() -> Result<Self> {
        // TODO: Connect to Wayland display
        // TODO: Get globals (compositor, layer-shell)
        // TODO: Create surface
        // TODO: Configure layer-shell
        todo!()
    }
    
    pub fn configure(&mut self, width: u32, height: u32) -> Result<()> {
        // TODO: Set size
        // TODO: Set anchor (bottom-center)
        // TODO: Set layer (overlay)
        // TODO: Set keyboard interactivity (none)
        todo!()
    }
    
    pub fn run(&mut self) -> Result<()> {
        // TODO: Event loop
        // TODO: Handle frame callbacks
        // TODO: Render on each frame
        todo!()
    }
}
