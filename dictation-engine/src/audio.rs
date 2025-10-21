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

        let config = StreamConfig {
            channels: 1,
            sample_rate: cpal::SampleRate(self.sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let buffer = Arc::clone(&self.buffer);

        let stream = device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mut buf = buffer.lock().unwrap();
                for &sample in data {
                    let _ = buf.push_overwrite(sample);
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
