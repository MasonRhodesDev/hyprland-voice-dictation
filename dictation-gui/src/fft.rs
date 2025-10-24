// FFT-based spectrum analysis

use rustfft::{num_complex::Complex, FftPlanner};

pub struct SpectrumAnalyzer {
    fft_size: usize,
    sample_rate: u32,
    smoothed_bands: Vec<f32>,
    smoothing_factor: f32,
}

impl SpectrumAnalyzer {
    pub fn new(fft_size: usize, sample_rate: u32, smoothing_factor: f32) -> Self {
        Self { fft_size, sample_rate, smoothed_bands: vec![0.0; 8], smoothing_factor }
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

    let mut buffer: Vec<Complex<f32>> = samples.iter().map(|&s| Complex::new(s, 0.0)).collect();

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hanning_window() {
        let samples = vec![1.0f32; 8];
        let windowed = apply_hanning_window(&samples);

        assert_eq!(windowed.len(), 8);
        assert!(windowed[0] < 1.0);
        assert!(windowed[windowed.len() - 1] < 1.0);
        assert!(windowed[windowed.len() / 2] > 0.5);
    }

    #[test]
    fn test_normalize_basic() {
        let values = vec![0.5, 1.0, 0.25, 0.75];
        let normalized = normalize(&values);

        assert_eq!(normalized.len(), 4);
        assert!((normalized[1] - 1.0).abs() < 0.001);
        assert!((normalized[0] - 0.5).abs() < 0.001);
        assert!((normalized[2] - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_normalize_all_zeros() {
        let values = vec![0.0f32; 8];
        let normalized = normalize(&values);

        assert_eq!(normalized.len(), 8);
        assert!(normalized.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_normalize_negative_values() {
        let values = vec![-1.0, 2.0, -0.5];
        let normalized = normalize(&values);

        assert!((normalized[1] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_fft_dc_component() {
        let samples = vec![1.0f32; 16];
        let spectrum = compute_fft(&samples);

        assert_eq!(spectrum.len(), 16);
        assert!(spectrum[0] > 0.0);
    }

    #[test]
    fn test_compute_fft_zero_input() {
        let samples = vec![0.0f32; 16];
        let spectrum = compute_fft(&samples);

        assert_eq!(spectrum.len(), 16);
        assert!(spectrum.iter().all(|&v| v.abs() < 0.001));
    }

    #[test]
    fn test_extract_frequency_bands() {
        let spectrum = vec![1.0f32; 512];
        let bands = extract_frequency_bands(&spectrum, 16000, 512);

        assert_eq!(bands.len(), 8);
        assert!(bands.iter().all(|&b| b > 0.0));
    }

    #[test]
    fn test_spectrum_analyzer_new() {
        let analyzer = SpectrumAnalyzer::new(512, 16000);

        assert_eq!(analyzer.fft_size, 512);
        assert_eq!(analyzer.sample_rate, 16000);
        assert_eq!(analyzer.smoothed_bands.len(), 8);
    }

    #[test]
    fn test_spectrum_analyzer_process() {
        let mut analyzer = SpectrumAnalyzer::new(512, 16000);
        let samples = vec![0.1f32; 512];

        let bands = analyzer.process(&samples);

        assert_eq!(bands.len(), 8);
        assert!(bands.iter().all(|&b| (0.0..=1.0).contains(&b)));
    }

    #[test]
    fn test_spectrum_analyzer_smoothing() {
        let mut analyzer = SpectrumAnalyzer::new(512, 16000);

        let loud = vec![0.5f32; 512];
        let quiet = vec![0.01f32; 512];

        analyzer.process(&loud);
        let bands_after_loud = analyzer.smoothed_bands.clone();

        analyzer.process(&quiet);
        let bands_after_quiet = analyzer.smoothed_bands.clone();

        for i in 0..8 {
            assert!(bands_after_quiet[i] < bands_after_loud[i]);
        }
    }

    #[test]
    fn test_spectrum_analyzer_zero_input() {
        let mut analyzer = SpectrumAnalyzer::new(512, 16000);
        let silence = vec![0.0f32; 512];

        let bands = analyzer.process(&silence);

        assert_eq!(bands.len(), 8);
        assert!(bands.iter().all(|&b| b == 0.0));
    }
}
