//! cpal-based audio backend implementation.
//!
//! Uses the cpal crate for cross-platform audio capture. On Linux, this typically
//! uses ALSA under the hood (with PipeWire providing ALSA compatibility).

use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Stream, StreamConfig};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tracing::{error, info, warn};

use crate::stream_muxer::StreamMuxer;

use super::{AudioBackend, AudioBackendConfig, AudioBackendFactory, DeviceInfo};

/// cpal-based audio capture backend.
pub struct CpalBackend {
    streams: Vec<Stream>,
    #[allow(dead_code)] // Kept alive for stream selection; may be used for debug finalization
    muxer: Arc<Mutex<StreamMuxer>>,
    /// Tracks stream IDs that have errored (for log-once behavior)
    errored_streams: Arc<Mutex<HashSet<String>>>,
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

        // Determine which devices to use
        // Fast path: for "default" mode, skip slow device enumeration
        let device_name = config.device_name.as_deref();
        let devices_to_use: Vec<_> = match device_name {
            // "default" or None: use system default directly (fast path)
            None | Some("default") => {
                info!("Using system default audio device (fast path)");
                let mut found = Vec::new();
                if let Some(default) = host.default_input_device() {
                    if let Ok(name) = default.name() {
                        info!("Default device: '{}'", name);
                    }
                    found.push(default);
                }
                found
            }
            // "all" mode: capture from pipewire + all hardware devices (slow path)
            // StreamMuxer selects the best stream in real-time
            Some("all") => {
                info!("Multi-device mode: enumerating audio devices...");
                let mut real_devices = Vec::new();
                if let Ok(devices) = host.input_devices() {
                    for device in devices {
                        if let Ok(name) = device.name() {
                            let is_real = Self::is_real_input_device(&name);
                            info!(
                                "  - '{}' {}",
                                name,
                                if is_real { "(will capture)" } else { "(skipped)" }
                            );
                            if is_real {
                                real_devices.push(device);
                            }
                        }
                    }
                }

                let mut devices = Vec::new();
                for device in real_devices {
                    if let Ok(name) = device.name() {
                        // Include pipewire (routes to default) and all hardware devices
                        if name == "pipewire" || name.starts_with("sysdefault:CARD=") {
                            devices.push(device);
                        }
                    }
                }

                if devices.is_empty() {
                    if let Some(default) = host.default_input_device() {
                        devices.push(default);
                    }
                }

                info!(
                    "Multi-device mode: capturing from {} device(s) with StreamMuxer selection",
                    devices.len()
                );
                devices
            }
            // Specific device requested (need to enumerate to find it)
            Some(name) => {
                info!("Searching for device '{}'...", name);
                let mut found = Vec::new();
                if let Ok(devices) = host.input_devices() {
                    for device in devices {
                        if let Ok(device_name) = device.name() {
                            if device_name == name {
                                found.push(device);
                                break;
                            }
                        }
                    }
                }
                if found.is_empty() {
                    warn!("Device '{}' not found, using default", name);
                    if let Some(default) = host.default_input_device() {
                        found.push(default);
                    }
                }
                found
            }
        };

        if devices_to_use.is_empty() {
            return Err(anyhow::anyhow!("No input devices available"));
        }

        // Create crossbeam channel for muxer output (lock-free, for audio callback)
        let (muxer_tx, muxer_rx) = crossbeam_channel::bounded(100);

        // Create StreamMuxer
        let muxer = StreamMuxer::new(muxer_tx, config.muxer_config.clone())?;
        let muxer = Arc::new(Mutex::new(muxer));

        let stream_config = StreamConfig {
            channels: 1,
            sample_rate: cpal::SampleRate(config.sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let errored_streams: Arc<Mutex<HashSet<String>>> =
            Arc::new(Mutex::new(HashSet::new()));

        let mut streams = Vec::new();
        for device in devices_to_use {
            let stream_id = device.name().unwrap_or_else(|_| "unknown".to_string());
            let muxer_clone = Arc::clone(&muxer);
            let stream_id_clone = stream_id.clone();
            let threshold = config.silence_threshold;

            // Clone for error callback
            let error_stream_id = stream_id.clone();
            let errored_streams_clone = Arc::clone(&errored_streams);

            match device.build_input_stream(
                &stream_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    // Pre-filter obviously silent chunks to reduce muxer load
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

                    // Push to muxer for quality-based stream selection
                    if let Ok(mut muxer) = muxer_clone.lock() {
                        muxer.push_samples(&stream_id_clone, &samples);
                    }
                },
                move |err| {
                    // Log once per stream - insert returns true if value was not present
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
            ) {
                Ok(stream) => {
                    info!("Created audio stream for: {}", stream_id);
                    streams.push(stream);
                }
                Err(e) => {
                    warn!("Failed to create stream for '{}': {}", stream_id, e);
                }
            }
        }

        if streams.is_empty() {
            return Err(anyhow::anyhow!("Failed to create any audio streams"));
        }

        // Spawn thread to forward muxer output to async channel
        let tx_clone = tx;
        std::thread::spawn(move || {
            while let Ok(samples) = muxer_rx.recv() {
                if tx_clone.send(samples).is_err() {
                    break; // Channel closed
                }
            }
        });

        info!(
            "CpalBackend initialized with {} stream(s) and StreamMuxer",
            streams.len()
        );
        Ok(Self {
            streams,
            muxer,
            errored_streams,
        })
    }

    /// Returns true if at least one stream is healthy (not errored)
    #[allow(dead_code)] // Available for future use
    pub fn has_healthy_streams(&self) -> bool {
        if let Ok(errored) = self.errored_streams.lock() {
            errored.len() < self.streams.len()
        } else {
            false // Assume unhealthy if lock fails
        }
    }

    /// Returns list of errored stream IDs
    #[allow(dead_code)] // Available for future use
    pub fn errored_stream_ids(&self) -> Vec<String> {
        if let Ok(errored) = self.errored_streams.lock() {
            errored.iter().cloned().collect()
        } else {
            Vec::new()
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

        // Flush the muxer to forward any buffered samples
        if let Ok(mut muxer) = self.muxer.lock() {
            muxer.flush();
        }

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
