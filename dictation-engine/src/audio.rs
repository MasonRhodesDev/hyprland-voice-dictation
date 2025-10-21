use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Stream, StreamConfig};
use ringbuf::traits::{Consumer, Observer, RingBuffer};
use ringbuf::HeapRb;
use std::sync::{Arc, Mutex};
use tracing::{debug, info};

const BUFFER_DURATION_SECS: usize = 5;

pub struct AudioCapture {
    sample_rate: u32,
    stream: Option<Stream>,
    buffer: Arc<Mutex<HeapRb<f32>>>,
}

impl AudioCapture {
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        if channels != 1 {
            anyhow::bail!("Only mono audio (1 channel) is supported");
        }

        let buffer_size = sample_rate as usize * BUFFER_DURATION_SECS;
        let buffer = Arc::new(Mutex::new(HeapRb::<f32>::new(buffer_size)));

        info!("Initializing audio capture: {}Hz, {} channel(s)", sample_rate, channels);

        Ok(Self {
            sample_rate,
            stream: None,
            buffer,
        })
    }

    pub fn start(&mut self) -> Result<()> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .context("No input device available")?;

        info!("Using input device: {}", device.name()?);

        // Get the device's supported config
        let mut supported_configs = device.supported_input_configs()?;
        let supported_config = supported_configs
            .next()
            .context("No supported input config")?
            .with_max_sample_rate();

        info!("Device sample rate: {}Hz", supported_config.sample_rate().0);

        let config: StreamConfig = supported_config.into();
        let device_sample_rate = config.sample_rate.0;

        let buffer = Arc::clone(&self.buffer);
        let target_sample_rate = self.sample_rate;

        let stream = device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mut buf = buffer.lock().unwrap();
                
                // Simple downsampling if needed
                if device_sample_rate == target_sample_rate {
                    // No resampling needed
                    for &sample in data {
                        let _ = buf.push_overwrite(sample);
                    }
                } else {
                    // Downsample by skipping samples
                    let ratio = device_sample_rate as f32 / target_sample_rate as f32;
                    let mut sample_index = 0.0;
                    while (sample_index as usize) < data.len() {
                        let _ = buf.push_overwrite(data[sample_index as usize]);
                        sample_index += ratio;
                    }
                }
            },
            |err| {
                tracing::error!("Audio stream error: {}", err);
            },
            None,
        )?;

        stream.play()?;
        self.stream = Some(stream);

        debug!("Audio stream started");
        Ok(())
    }

    pub fn stop(&mut self) -> Result<()> {
        if let Some(stream) = self.stream.take() {
            drop(stream);
            debug!("Audio stream stopped");
        }
        Ok(())
    }

    pub fn get_latest_samples(&self, count: usize) -> Vec<f32> {
        let buf = self.buffer.lock().unwrap();
        let available = buf.occupied_len();
        let to_read = count.min(available);
        
        let mut samples = Vec::with_capacity(to_read);
        samples.extend(buf.iter().skip(available.saturating_sub(to_read)).take(to_read));
        
        samples
    }

    pub fn get_samples_for_duration(&self, duration_ms: u64) -> Vec<f32> {
        let sample_count = (self.sample_rate as u64 * duration_ms / 1000) as usize;
        self.get_latest_samples(sample_count)
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

impl Drop for AudioCapture {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}
