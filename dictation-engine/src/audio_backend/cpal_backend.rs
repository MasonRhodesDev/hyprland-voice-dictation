//! cpal-based audio backend implementation.
//!
//! Uses the cpal crate for cross-platform audio capture. On Linux, this typically
//! uses ALSA under the hood (with PipeWire providing ALSA compatibility).

use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Stream, StreamConfig};
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tracing::{error, info, warn};

use super::{AudioBackend, AudioBackendConfig, AudioBackendFactory, DeviceInfo};

/// cpal-based audio capture backend.
pub struct CpalBackend {
    streams: Vec<Stream>,
    /// Tracks stream IDs that have errored (for log-once behavior)
    errored_streams: Arc<Mutex<HashSet<String>>>,
    /// Timestamp (ms since epoch) of last successful audio send, for health monitoring
    last_audio_timestamp: Arc<AtomicU64>,
    /// Count of dropped samples due to channel backpressure
    samples_dropped: Arc<AtomicU64>,
}

impl CpalBackend {
    /// Check if a device name indicates it's a real input device (not a monitor/loopback)
    fn is_real_input_device(name: &str) -> bool {
        let name_lower = name.to_lowercase();
        // Skip output monitors (loopback devices)
        if name_lower.contains(".monitor") || name_lower.contains("monitor") {
            return false;
        }
        // Skip HDMI outputs (usually no mic)
        if name_lower.contains("hdmi") {
            return false;
        }
        true
    }

    fn new(
        tx: mpsc::UnboundedSender<Vec<i16>>,
        config: &AudioBackendConfig,
    ) -> Result<Self> {
        let host = cpal::default_host();

        // Determine which device to use (single device only)
        let device_name = config.device_name.as_deref();
        let device = match device_name {
            // "default" or None: use system default directly (fast path)
            None | Some("default") => {
                info!("Using system default audio device (fast path)");
                let dev = host.default_input_device()
                    .ok_or_else(|| anyhow::anyhow!("No default input device available"))?;
                if let Ok(name) = dev.name() {
                    info!("Default device: '{}'", name);
                }
                dev
            }
            // "all" is no longer supported, fall back to default
            Some("all") => {
                warn!("Multi-device 'all' mode is no longer supported, using default device");
                host.default_input_device()
                    .ok_or_else(|| anyhow::anyhow!("No default input device available"))?
            }
            // Specific device requested (need to enumerate to find it)
            Some(name) => {
                info!("Searching for device '{}'...", name);
                let mut found = None;
                if let Ok(devices) = host.input_devices() {
                    for device in devices {
                        if let Ok(device_name) = device.name() {
                            if device_name == name {
                                found = Some(device);
                                break;
                            }
                        }
                    }
                }
                match found {
                    Some(dev) => dev,
                    None => {
                        warn!("Device '{}' not found, using default", name);
                        host.default_input_device()
                            .ok_or_else(|| anyhow::anyhow!("No default input device available"))?
                    }
                }
            }
        };

        // Create crossbeam channel for bridging audio callback to async channel
        let (cb_tx, cb_rx) = crossbeam_channel::bounded::<Vec<i16>>(100);

        let stream_config = StreamConfig {
            channels: 1,
            sample_rate: cpal::SampleRate(config.sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let errored_streams: Arc<Mutex<HashSet<String>>> =
            Arc::new(Mutex::new(HashSet::new()));
        let last_audio_timestamp = Arc::new(AtomicU64::new(0));
        let samples_dropped = Arc::new(AtomicU64::new(0));

        let stream_id = device.name().unwrap_or_else(|_| "unknown".to_string());
        let threshold = config.silence_threshold;

        // Clone for error callback
        let error_stream_id = stream_id.clone();
        let errored_streams_clone = Arc::clone(&errored_streams);
        let samples_dropped_clone = Arc::clone(&samples_dropped);

        let stream = device.build_input_stream(
            &stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // Pre-filter obviously silent chunks
                let rms: f32 =
                    (data.iter().map(|&s| s * s).sum::<f32>() / data.len() as f32).sqrt();
                if rms < threshold {
                    return; // Skip completely silent chunks
                }

                // Convert to i16
                let samples: Vec<i16> = data
                    .iter()
                    .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
                    .collect();

                // Send directly via crossbeam channel (no muxer)
                if cb_tx.try_send(samples).is_err() {
                    samples_dropped_clone.fetch_add(1, Ordering::Relaxed);
                }
            },
            move |err| {
                // Log once per stream
                if let Ok(mut errored) = errored_streams_clone.lock() {
                    if errored.insert(error_stream_id.clone()) {
                        error!(
                            "Audio stream '{}' error: {} (will retry on device reconnection)",
                            error_stream_id, err
                        );
                    }
                }
            },
            None,
        ).map_err(|e| anyhow::anyhow!("Failed to create audio stream for '{}': {}", stream_id, e))?;

        info!("Created audio stream for: {}", stream_id);

        // Spawn thread to forward crossbeam channel to async mpsc channel
        let tx_clone = tx;
        let last_ts_clone = Arc::clone(&last_audio_timestamp);
        let drops_clone = Arc::clone(&samples_dropped);
        std::thread::spawn(move || {
            let mut last_drop_log = 0u64;
            while let Ok(samples) = cb_rx.recv() {
                // Update last audio timestamp for health monitoring
                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;
                last_ts_clone.store(now_ms, Ordering::Relaxed);

                // Check and log drops periodically
                let current_drops = drops_clone.load(Ordering::Relaxed);
                if current_drops > last_drop_log && current_drops % 100 == 0 {
                    warn!("Audio samples dropped: {} total", current_drops);
                    last_drop_log = current_drops;
                }

                if tx_clone.send(samples).is_err() {
                    break; // Channel closed
                }
            }
        });

        info!("CpalBackend initialized with single stream (direct channel)");
        Ok(Self {
            streams: vec![stream],
            errored_streams,
            last_audio_timestamp,
            samples_dropped,
        })
    }

    /// Get the timestamp (ms since epoch) of the last audio received
    pub fn last_audio_timestamp_ms(&self) -> u64 {
        self.last_audio_timestamp.load(Ordering::Relaxed)
    }

    /// Get the total number of dropped sample chunks
    pub fn samples_dropped_count(&self) -> u64 {
        self.samples_dropped.load(Ordering::Relaxed)
    }

    /// Returns true if at least one stream is healthy (not errored)
    #[allow(dead_code)]
    pub fn has_healthy_streams(&self) -> bool {
        if let Ok(errored) = self.errored_streams.lock() {
            errored.len() < self.streams.len()
        } else {
            false
        }
    }
}

impl AudioBackend for CpalBackend {
    fn start(&self) -> Result<()> {
        for stream in &self.streams {
            stream.play()?;
        }
        info!("CpalBackend started ({} streams)", self.streams.len());
        Ok(())
    }

    fn stop(&self) -> Result<()> {
        for stream in &self.streams {
            stream.pause()?;
        }
        info!("CpalBackend stopped ({} streams)", self.streams.len());
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        // Wait for cpal callbacks to complete (10-20ms buffers, 2x safety)
        std::thread::sleep(std::time::Duration::from_millis(50));
        info!("CpalBackend flushed");
        Ok(())
    }

    fn releases_on_stop(&self) -> bool {
        // cpal/ALSA backend should release mic after idle to allow browsers to use it
        true
    }
}

impl AudioBackendFactory for CpalBackend {
    fn create(
        tx: mpsc::UnboundedSender<Vec<i16>>,
        config: &AudioBackendConfig,
    ) -> Result<Box<dyn AudioBackend>> {
        Ok(Box::new(Self::new(tx, config)?))
    }

    fn list_devices() -> Result<Vec<DeviceInfo>> {
        let host = cpal::default_host();
        let default_name = host
            .default_input_device()
            .and_then(|d| d.name().ok());

        let mut devices = Vec::new();
        if let Ok(input_devices) = host.input_devices() {
            for device in input_devices {
                if let Ok(name) = device.name() {
                    if Self::is_real_input_device(&name) {
                        let is_default = default_name.as_ref() == Some(&name);
                        devices.push(DeviceInfo { name, is_default });
                    }
                }
            }
        }

        Ok(devices)
    }
}
