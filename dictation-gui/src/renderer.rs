// Spectrum visualization renderer

use anyhow::Result;

pub struct SpectrumRenderer {
    width: u32,
    height: u32,
    bar_count: usize,
    // TODO: wgpu or tiny-skia rendering context
}

impl SpectrumRenderer {
    pub fn new(width: u32, height: u32) -> Result<Self> {
        Ok(Self {
            width,
            height,
            bar_count: 8,
        })
    }
    
    pub fn render(&mut self, band_values: &[f32]) -> Result<()> {
        // TODO: Clear background (pill shape)
        // TODO: Draw 8 vertical bars
        // TODO: Map values (0.0-1.0) to heights (5-30px)
        // TODO: Apply colors
        // TODO: Present frame
        todo!()
    }
}
