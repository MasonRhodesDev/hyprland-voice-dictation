//! Audio backend abstraction layer.
//!
//! This module provides a trait-based abstraction for audio capture backends,
//! allowing different implementations (cpal, pipewire-rs) to be used interchangeably.

pub mod cpal_backend;

#[cfg(feature = "pipewire")]
pub mod pipewire_backend;

use anyhow::Result;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::stream_muxer::MuxerConfig;

/// Configuration for creating an audio backend.
#[derive(Clone)]
pub struct AudioBackendConfig {
    /// Device name to capture from. None or "default" for system default, "all" for multi-device.
    pub device_name: Option<String>,
    /// Sample rate in Hz (typically 16000 for speech recognition).
    pub sample_rate: u32,
    /// RMS threshold below which audio is considered silence.
    pub silence_threshold: f32,
    /// Configuration for the stream muxer (used in multi-device mode).
    pub muxer_config: MuxerConfig,
}

/// Information about an available audio input device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device name/identifier.
    pub name: String,
    /// Whether this is the system default input device.
    pub is_default: bool,
}

/// Trait for audio capture backends.
///
/// Implementations handle the low-level audio capture from microphones,
/// converting to i16 samples and sending through the provided channel.
///
/// Note: Not required to be Send because backends are managed on the main thread.
/// cpal streams in particular are !Send on some platforms.
pub trait AudioBackend {
    /// Start capturing audio. Samples will be sent through the channel provided at creation.
    fn start(&self) -> Result<()>;

    /// Stop capturing audio (pause streams).
    fn stop(&self) -> Result<()>;

    /// Flush any buffered audio data.
    ///
    /// Called after stop() to ensure all in-flight samples are delivered.
    /// Default implementation waits for typical buffer to drain.
    fn flush(&self) -> Result<()> {
        // Default: wait for typical buffer to drain
        std::thread::sleep(std::time::Duration::from_millis(50));
        Ok(())
    }

    /// Whether this backend should release the microphone after an idle timeout.
    ///
    /// - `true`: Backend uses exclusive-ish access (cpal/ALSA), should release after idle
    ///   to allow other apps (browsers) to use the mic.
    /// - `false`: Backend supports native sharing (pipewire-rs), can keep mic open indefinitely.
    fn releases_on_stop(&self) -> bool;
}

/// Factory trait for creating audio backends.
///
/// Each backend implementation provides a factory that can enumerate devices
/// and create backend instances.
pub trait AudioBackendFactory {
    /// Create a new audio backend instance.
    ///
    /// # Arguments
    /// * `tx` - Channel sender for audio samples (Vec<i16> chunks)
    /// * `config` - Backend configuration
    fn create(
        tx: mpsc::UnboundedSender<Vec<i16>>,
        config: &AudioBackendConfig,
    ) -> Result<Box<dyn AudioBackend>>
    where
        Self: Sized;

    /// List available input devices.
    fn list_devices() -> Result<Vec<DeviceInfo>>
    where
        Self: Sized;
}

/// Supported audio backend types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendType {
    /// Auto-detect: prefer PipeWire, fall back to cpal.
    #[default]
    Auto,
    /// cpal backend (cross-platform, uses ALSA on Linux).
    Cpal,
    /// PipeWire backend (native Linux PipeWire, supports mic sharing).
    #[cfg(feature = "pipewire")]
    Pipewire,
}

impl BackendType {
    /// Parse backend type from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "cpal" | "alsa" => Some(Self::Cpal),
            #[cfg(feature = "pipewire")]
            "pipewire" | "pw" => Some(Self::Pipewire),
            _ => None,
        }
    }
}

/// Create an audio backend of the specified type.
///
/// For `BackendType::Auto`, tries PipeWire first and falls back to cpal.
pub fn create_backend(
    backend_type: BackendType,
    tx: mpsc::UnboundedSender<Vec<i16>>,
    config: &AudioBackendConfig,
) -> Result<Box<dyn AudioBackend>> {
    match backend_type {
        BackendType::Auto => create_backend_auto(tx, config),
        BackendType::Cpal => {
            info!("Using cpal audio backend");
            cpal_backend::CpalBackend::create(tx, config)
        }
        #[cfg(feature = "pipewire")]
        BackendType::Pipewire => {
            info!("Using PipeWire audio backend");
            pipewire_backend::PipewireBackend::create(tx, config)
        }
    }
}

/// Create a backend with auto-detection: prefer PipeWire, fall back to cpal.
fn create_backend_auto(
    tx: mpsc::UnboundedSender<Vec<i16>>,
    config: &AudioBackendConfig,
) -> Result<Box<dyn AudioBackend>> {
    #[cfg(feature = "pipewire")]
    {
        // Try PipeWire first
        if pipewire_backend::PipewireBackend::is_available() {
            info!("PipeWire availability check passed, attempting to create backend...");
            match pipewire_backend::PipewireBackend::create(tx.clone(), config) {
                Ok(backend) => {
                    info!("✓ Using PipeWire audio backend (auto-detected)");
                    info!("  → Supports concurrent mic access (no browser conflicts)");
                    return Ok(backend);
                }
                Err(e) => {
                    warn!("✗ PipeWire backend creation failed: {e}");
                    warn!("  → Falling back to cpal/ALSA (will hold exclusive mic access)");
                    warn!("  → Check: systemctl --user status pipewire");
                }
            }
        } else {
            warn!("✗ PipeWire not available on system");
            warn!("  → Check: systemctl --user status pipewire");
            warn!("  → Using cpal/ALSA backend (will hold exclusive mic access)");
        }
    }

    #[cfg(not(feature = "pipewire"))]
    {
        warn!("✗ PipeWire feature not enabled at compile time");
        warn!("  → Rebuild with: cargo build --features pipewire");
        warn!("  → Using cpal/ALSA backend (will hold exclusive mic access)");
    }

    // Fall back to cpal
    info!("Using cpal/ALSA audio backend");
    info!("  → Will release mic after idle timeout (default: 30s)");
    cpal_backend::CpalBackend::create(tx, config)
}

/// List devices for the specified backend type.
pub fn list_devices(backend_type: BackendType) -> Result<Vec<DeviceInfo>> {
    match backend_type {
        BackendType::Auto | BackendType::Cpal => cpal_backend::CpalBackend::list_devices(),
        #[cfg(feature = "pipewire")]
        BackendType::Pipewire => pipewire_backend::PipewireBackend::list_devices(),
    }
}
