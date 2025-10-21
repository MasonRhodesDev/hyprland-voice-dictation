// FFT-based spectrum analysis

use rustfft::{FftPlanner, num_complex::Complex};

pub struct SpectrumAnalyzer {
    fft_size: usize,
    sample_rate: u32,
    smoothed_bands: Vec<f32>,
    smoothing_factor: f32,
}

impl SpectrumAnalyzer {
    pub fn new(fft_size: usize, sample_rate: u32) -> Self {
        Self {
            fft_size,
            sample_rate,
            smoothed_bands: vec![0.0; 8],
            smoothing_factor: 0.6,
        }
    }
    
    pub fn process(&mut self, samples: &[f32]) -> Vec<f32> {
        // Apply Hanning window
        let windowed = apply_hanning_window(samples);
        
        // Compute FFT
        let spectrum = compute_fft(&windowed);
        
        // Extract 8 frequency bands
        let bands = extract_frequency_bands(&spectrum, self.sample_rate, self.fft_size);
        
        // Smooth band values
        for (i, &band_value) in bands.iter().enumerate() {
            self.smoothed_bands[i] = self.smoothing_factor * self.smoothed_bands[i]
                + (1.0 - self.smoothing_factor) * band_value;
        }
        
        // Normalize to 0.0-1.0
        normalize(&self.smoothed_bands)
    }
}

fn apply_hanning_window(samples: &[f32]) -> Vec<f32> {
    let n = samples.len();
    samples
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let window = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos());
            s * window
        })
        .collect()
}

fn compute_fft(samples: &[f32]) -> Vec<f32> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(samples.len());
    
    let mut buffer: Vec<Complex<f32>> = samples
        .iter()
        .map(|&s| Complex::new(s, 0.0))
        .collect();
    
    fft.process(&mut buffer);
    
    // Return magnitudes
    buffer.iter().map(|c| c.norm()).collect()
}

fn extract_frequency_bands(spectrum: &[f32], sample_rate: u32, fft_size: usize) -> Vec<f32> {
    let freq_resolution = sample_rate as f32 / fft_size as f32;
    
    let bands = [
        (100.0, 250.0),
        (250.0, 500.0),
        (500.0, 1000.0),
        (1000.0, 2000.0),
        (2000.0, 3000.0),
        (3000.0, 4000.0),
        (4000.0, 5000.0),
        (5000.0, 7000.0),
    ];
    
    bands
        .iter()
        .map(|(low, high)| {
            let low_bin = (low / freq_resolution) as usize;
            let high_bin = (high / freq_resolution) as usize;
            
            let sum: f32 = spectrum[low_bin..high_bin].iter().sum();
            sum / (high_bin - low_bin) as f32
        })
        .collect()
}

fn normalize(values: &[f32]) -> Vec<f32> {
    let max = values.iter().cloned().fold(0.0f32, f32::max);
    if max > 0.0 {
        values.iter().map(|&v| v / max).collect()
    } else {
        vec![0.0; values.len()]
    }
}
