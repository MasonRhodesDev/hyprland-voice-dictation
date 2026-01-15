//! Audio backend abstraction layer.
//!
//! This module provides a trait-based abstraction for audio capture backends,
//! allowing different implementations (cpal, pipewire-rs) to be used interchangeably.

pub mod cpal_backend;

use anyhow::Result;
use tokio::sync::mpsc;

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
    /// cpal backend (cross-platform, uses ALSA on Linux).
    #[default]
    Cpal,
    // PipeWire backend (native Linux PipeWire, future implementation).
    // Pipewire,
}

impl BackendType {
    /// Parse backend type from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "cpal" | "alsa" => Some(Self::Cpal),
            // "pipewire" | "pw" => Some(Self::Pipewire),
            _ => None,
        }
    }
}

/// Create an audio backend of the specified type.
pub fn create_backend(
    backend_type: BackendType,
    tx: mpsc::UnboundedSender<Vec<i16>>,
    config: &AudioBackendConfig,
) -> Result<Box<dyn AudioBackend>> {
    match backend_type {
        BackendType::Cpal => cpal_backend::CpalBackend::create(tx, config),
        // BackendType::Pipewire => pipewire_backend::PipewireBackend::create(tx, config),
    }
}

/// List devices for the specified backend type.
pub fn list_devices(backend_type: BackendType) -> Result<Vec<DeviceInfo>> {
    match backend_type {
        BackendType::Cpal => cpal_backend::CpalBackend::list_devices(),
        // BackendType::Pipewire => pipewire_backend::PipewireBackend::list_devices(),
    }
}
