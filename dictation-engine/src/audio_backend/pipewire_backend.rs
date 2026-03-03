//! PipeWire native audio backend.
//!
//! This backend uses pipewire-rs for native PipeWire audio capture,
//! enabling proper mic sharing with browsers without requiring idle timeouts.

use anyhow::{anyhow, Context, Result};
use pipewire as pw;
use pw::spa::param::audio::{AudioFormat, AudioInfoRaw};
use pw::spa::pod::Pod;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use super::{AudioBackend, AudioBackendConfig, AudioBackendFactory, DeviceInfo};

/// Commands sent to the PipeWire thread.
enum PwCommand {
    Start,
    Stop,
    Flush,
    Quit,
}

/// Information about a discovered audio source node.
#[derive(Clone, Debug)]
struct AudioSourceInfo {
    /// PipeWire node ID (for logging only)
    id: u32,
    /// Node name (e.g., "alsa_input.usb-...")
    name: String,
    /// Object serial number (for stream targeting)
    object_serial: u32,
    /// Description (e.g., "USB Microphone")
    description: String,
    /// Media class (should be "Audio/Source")
    #[allow(dead_code)]
    media_class: String,
}

/// PipeWire native audio backend.
///
/// Uses a dedicated thread for the PipeWire MainLoop since it cannot
/// run on tokio's async runtime.
pub struct PipewireBackend {
    /// Channel to send commands to the PipeWire thread.
    control_tx: std::sync::mpsc::Sender<PwCommand>,
    /// Handle to the PipeWire thread.
    _thread: thread::JoinHandle<()>,
    /// Whether the stream is currently capturing.
    is_running: Arc<AtomicBool>,
}

impl PipewireBackend {
    /// Check if PipeWire is available on the system.
    pub fn is_available() -> bool {
        pw::init();

        match pw::main_loop::MainLoop::new(None) {
            Ok(_) => {
                debug!("PipeWire is available");
                true
            }
            Err(e) => {
                debug!("PipeWire not available: {e}");
                false
            }
        }
    }
}

impl AudioBackendFactory for PipewireBackend {
    fn create(
        tx: mpsc::UnboundedSender<Vec<i16>>,
        config: &AudioBackendConfig,
    ) -> Result<Box<dyn AudioBackend>> {
        info!("Creating PipeWire audio backend...");

        // Initialize PipeWire
        pw::init();

        let sample_rate = config.sample_rate;
        let silence_threshold = config.silence_threshold;
        let is_running = Arc::new(AtomicBool::new(false));
        let is_running_clone = is_running.clone();

        // Create control channel
        let (control_tx, control_rx) = std::sync::mpsc::channel::<PwCommand>();

        // Create crossbeam channel for bridging PW callback to async channel
        let (cb_tx, cb_rx) = crossbeam_channel::bounded::<Vec<i16>>(100);

        // Spawn forwarder thread: crossbeam -> async mpsc
        let tx_clone = tx.clone();
        thread::spawn(move || {
            while let Ok(samples) = cb_rx.recv() {
                if tx_clone.send(samples).is_err() {
                    break;
                }
            }
        });

        // Resolve device name to PipeWire target serial
        let device_name = config.device_name.clone();
        let target_serial = match &device_name {
            Some(name) if name != "default" => {
                match enumerate_audio_sources() {
                    Ok(sources) => {
                        let found = sources.iter().find(|s| s.name == *name);
                        if let Some(source) = found {
                            info!("Resolved device '{}' to PipeWire serial {}", name, source.object_serial);
                            Some(source.object_serial)
                        } else {
                            warn!("Device '{}' not found in PipeWire sources, using default", name);
                            None
                        }
                    }
                    Err(e) => {
                        warn!("Failed to enumerate PipeWire sources: {e}, using default");
                        None
                    }
                }
            }
            _ => None,
        };

        // Spawn PipeWire thread
        let thread = thread::Builder::new()
            .name("pipewire-audio".into())
            .spawn(move || {
                if let Err(e) = run_pipewire_thread(
                    control_rx,
                    cb_tx,
                    sample_rate,
                    silence_threshold,
                    is_running_clone,
                    target_serial,
                ) {
                    error!("PipeWire thread error: {e}");
                }
            })
            .context("Failed to spawn PipeWire thread")?;

        Ok(Box::new(PipewireBackend {
            control_tx,
            _thread: thread,
            is_running,
        }))
    }

    fn list_devices() -> Result<Vec<DeviceInfo>> {
        pw::init();

        let sources = enumerate_audio_sources()?;
        let devices: Vec<DeviceInfo> = sources
            .into_iter()
            .map(|s| DeviceInfo {
                description: s.description,
                name: s.name,
                is_default: false,
            })
            .collect();

        if devices.is_empty() {
            Ok(vec![DeviceInfo {
                name: "pipewire".to_string(),
                description: "PipeWire Default".to_string(),
                is_default: true,
            }])
        } else {
            Ok(devices)
        }
    }
}

impl AudioBackend for PipewireBackend {
    fn start(&self) -> Result<()> {
        self.control_tx
            .send(PwCommand::Start)
            .map_err(|_| anyhow!("PipeWire thread not responding"))?;
        info!("PipewireBackend: started");
        Ok(())
    }

    fn stop(&self) -> Result<()> {
        self.control_tx
            .send(PwCommand::Stop)
            .map_err(|_| anyhow!("PipeWire thread not responding"))?;
        info!("PipewireBackend: stopped");
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        self.control_tx
            .send(PwCommand::Flush)
            .map_err(|_| anyhow!("PipeWire thread not responding"))?;
        std::thread::sleep(std::time::Duration::from_millis(20));
        info!("PipewireBackend: flushed");
        Ok(())
    }

    fn releases_on_stop(&self) -> bool {
        // PipeWire supports native mic sharing - no need to release
        false
    }
}

impl Drop for PipewireBackend {
    fn drop(&mut self) {
        let _ = self.control_tx.send(PwCommand::Quit);
    }
}

/// Enumerate audio source nodes from PipeWire.
fn enumerate_audio_sources() -> Result<Vec<AudioSourceInfo>> {
    use std::cell::Cell;

    let mainloop = pw::main_loop::MainLoop::new(None)
        .context("Failed to create PipeWire MainLoop")?;

    let context = pw::context::Context::new(&mainloop)
        .context("Failed to create PipeWire Context")?;

    let core = context
        .connect(None)
        .context("Failed to connect to PipeWire daemon")?;

    let registry = core
        .get_registry()
        .context("Failed to get PipeWire Registry")?;

    let sources: Rc<RefCell<Vec<AudioSourceInfo>>> = Rc::new(RefCell::new(Vec::new()));
    let done = Rc::new(Cell::new(false));

    let sources_clone = sources.clone();
    let done_clone = done.clone();
    let mainloop_weak = mainloop.downgrade();

    let pending = core.sync(0).context("Failed to sync with PipeWire core")?;

    let _core_listener = core
        .add_listener_local()
        .done(move |id, seq| {
            if id == pw::core::PW_ID_CORE && seq == pending {
                done_clone.set(true);
                if let Some(ml) = mainloop_weak.upgrade() {
                    ml.quit();
                }
            }
        })
        .register();

    let _registry_listener = registry
        .add_listener_local()
        .global(move |global| {
            if global.type_ == pw::types::ObjectType::Node {
                if let Some(props) = &global.props {
                    let media_class = props.get("media.class").unwrap_or("");
                    if media_class == "Audio/Source" {
                        let name = props.get("node.name").unwrap_or("unknown").to_string();
                        let description = props
                            .get("node.description")
                            .or_else(|| props.get("node.nick"))
                            .unwrap_or(&name)
                            .to_string();

                        let object_serial = props
                            .get("object.serial")
                            .and_then(|s| s.parse::<u32>().ok())
                            .unwrap_or(global.id);

                        if !name.contains(".monitor") && !description.to_lowercase().contains("monitor") {
                            debug!(
                                "Found audio source: id={}, serial={}, name='{}', desc='{}'",
                                global.id, object_serial, name, description
                            );

                            sources_clone.borrow_mut().push(AudioSourceInfo {
                                id: global.id,
                                name,
                                object_serial,
                                description,
                                media_class: media_class.to_string(),
                            });
                        }
                    }
                }
            }
        })
        .register();

    while !done.get() {
        mainloop.run();
    }

    let result = sources.borrow().clone();
    info!("Enumerated {} PipeWire audio sources", result.len());
    Ok(result)
}

/// Build the audio format pod for stream negotiation.
fn build_audio_format_pod(sample_rate: u32) -> Result<Vec<u8>> {
    let mut audio_info = AudioInfoRaw::new();
    audio_info.set_format(AudioFormat::F32LE);
    audio_info.set_rate(sample_rate);
    audio_info.set_channels(1); // Mono for speech recognition

    let mut buffer = vec![0u8; 1024];
    pw::spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(&mut buffer),
        &pw::spa::pod::Value::Object(pw::spa::pod::Object {
            type_: pw::spa::sys::SPA_TYPE_OBJECT_Format,
            id: pw::spa::sys::SPA_PARAM_EnumFormat,
            properties: audio_info.into(),
        }),
    )
    .context("Failed to serialize audio format")?;

    Ok(buffer)
}

/// Run the PipeWire MainLoop with single-stream capture.
fn run_pipewire_thread(
    control_rx: std::sync::mpsc::Receiver<PwCommand>,
    audio_tx: crossbeam_channel::Sender<Vec<i16>>,
    sample_rate: u32,
    silence_threshold: f32,
    is_running: Arc<AtomicBool>,
    target_serial: Option<u32>,
) -> Result<()> {
    let mainloop = pw::main_loop::MainLoop::new(None)
        .context("Failed to create PipeWire MainLoop")?;

    let context = pw::context::Context::new(&mainloop)
        .context("Failed to create PipeWire Context")?;

    let core = context
        .connect(None)
        .context("Failed to connect to PipeWire daemon")?;

    // Build audio format pod
    let format_buffer = build_audio_format_pod(sample_rate)?;

    let samples_dropped = Arc::new(AtomicU64::new(0));

    let stream_name = if target_serial.is_some() { "targeted" } else { "default" };
    let (stream, _listener) = create_capture_stream(
        &core,
        target_serial,
        stream_name,
        &format_buffer,
        silence_threshold,
        audio_tx,
        samples_dropped.clone(),
        is_running.clone(),
    )?;

    info!("Created PipeWire capture stream (target_serial: {:?})", target_serial);

    // Run mainloop with command polling
    let loop_clone = mainloop.loop_();
    let control_rx = std::sync::Arc::new(std::sync::Mutex::new(control_rx));
    let is_running_for_timer = is_running.clone();
    let mainloop_weak = mainloop.downgrade();

    let _timer = loop_clone.add_timer(move |_| {
        let rx = match control_rx.lock() {
            Ok(rx) => rx,
            Err(e) => {
                error!("PipeWire: Failed to lock control_rx: {}", e);
                return;
            }
        };
        while let Ok(cmd) = rx.try_recv() {
            match cmd {
                PwCommand::Start => {
                    is_running_for_timer.store(true, Ordering::Relaxed);
                    debug!("PipeWire: recording started");
                }
                PwCommand::Stop => {
                    is_running_for_timer.store(false, Ordering::Relaxed);
                    debug!("PipeWire: recording stopped");
                }
                PwCommand::Flush => {
                    debug!("PipeWire: flush (no-op, direct channel)");
                }
                PwCommand::Quit => {
                    if let Some(ml) = mainloop_weak.upgrade() {
                        ml.quit();
                    }
                }
            }
        }
    });

    let _ = _timer.update_timer(
        Some(std::time::Duration::from_millis(10)),
        Some(std::time::Duration::from_millis(10)),
    );

    // Keep stream alive
    let _stream = stream;

    mainloop.run();

    info!("PipeWire thread exiting");
    Ok(())
}

/// Create a capture stream for a specific audio source.
fn create_capture_stream(
    core: &pw::core::Core,
    target_serial: Option<u32>,
    stream_name: &str,
    format_buffer: &[u8],
    silence_threshold: f32,
    audio_tx: crossbeam_channel::Sender<Vec<i16>>,
    samples_dropped: Arc<AtomicU64>,
    is_running: Arc<AtomicBool>,
) -> Result<(pw::stream::Stream, pw::stream::StreamListener<()>)> {
    let mut props = pw::properties::properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Communication",
        *pw::keys::NODE_NAME => "voice-dictation",
        *pw::keys::APP_NAME => "Voice Dictation",
    };

    if let Some(serial) = target_serial {
        props.insert("target.object", serial.to_string());
    }

    let stream = pw::stream::Stream::new(core, stream_name, props)
        .context("Failed to create PipeWire stream")?;

    let is_running_clone = is_running.clone();

    let listener = stream
        .add_local_listener_with_user_data(())
        .param_changed(|_, _, id, param| {
            if id == pw::spa::param::ParamType::Format.as_raw() {
                if let Some(_param) = param {
                    debug!("PipeWire stream format negotiated");
                }
            }
        })
        .process(move |stream, _| {
            if !is_running_clone.load(Ordering::Relaxed) {
                return;
            }

            if let Some(mut buffer) = stream.dequeue_buffer() {
                let datas = buffer.datas_mut();
                if datas.is_empty() {
                    return;
                }

                let data = &mut datas[0];
                let chunk = data.chunk();
                let size = chunk.size() as usize;
                let offset = chunk.offset() as usize;

                if size > 0 {
                    if let Some(slice) = data.data() {
                        if offset + size <= slice.len() {
                            let f32_samples: &[f32] = unsafe {
                                std::slice::from_raw_parts(
                                    slice[offset..].as_ptr() as *const f32,
                                    size / std::mem::size_of::<f32>(),
                                )
                            };

                            // Pre-filter silence
                            let rms: f32 = (f32_samples.iter().map(|&s| s * s).sum::<f32>()
                                / f32_samples.len() as f32)
                                .sqrt();
                            if rms < silence_threshold {
                                return;
                            }

                            let i16_samples: Vec<i16> = f32_samples
                                .iter()
                                .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
                                .collect();

                            if !i16_samples.is_empty() {
                                if audio_tx.try_send(i16_samples).is_err() {
                                    samples_dropped.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                    }
                }
            }
        })
        .register()?;

    let pod_ref = unsafe { Pod::from_raw(format_buffer.as_ptr() as *const pw::spa::sys::spa_pod) };

    stream.connect(
        pw::spa::utils::Direction::Input,
        None,
        pw::stream::StreamFlags::AUTOCONNECT
            | pw::stream::StreamFlags::MAP_BUFFERS
            | pw::stream::StreamFlags::RT_PROCESS,
        &mut [pod_ref],
    )?;

    Ok((stream, listener))
}
