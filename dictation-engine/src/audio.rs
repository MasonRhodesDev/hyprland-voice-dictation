// Audio capture using cpal

use anyhow::Result;

pub struct AudioCapture {
    // TODO: cpal stream
    // TODO: ring buffer
}

impl AudioCapture {
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        // TODO: Get default input device
        // TODO: Build stream config
        // TODO: Create stream with callback
        todo!()
    }
    
    pub fn start(&mut self) -> Result<()> {
        // TODO: Start audio stream
        todo!()
    }
    
    pub fn stop(&mut self) -> Result<()> {
        // TODO: Stop audio stream
        todo!()
    }
    
    pub fn get_samples(&self, duration_ms: u64) -> Vec<f32> {
        // TODO: Extract samples from ring buffer
        todo!()
    }
}
