//! PipeWire native audio backend.
//!
//! This backend uses pipewire-rs for native PipeWire audio capture,
//! enabling proper mic sharing with browsers without requiring idle timeouts.
//!
//! Supports multi-device capture: enumerates all Audio/Source nodes and
//! routes them through StreamMuxer for quality-based selection.

use anyhow::{anyhow, Context, Result};
use pipewire as pw;
use pw::spa::param::audio::{AudioFormat, AudioInfoRaw};
use pw::spa::pod::Pod;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::stream_muxer::{MuxerConfig, StreamMuxer};

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
        // Try to initialize PipeWire - this will fail if libpipewire isn't available
        // or if the daemon isn't running
        pw::init();

        // Try to create a MainLoop - this is a lightweight check
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
        info!("Creating PipeWire audio backend (multi-device)...");

        // Initialize PipeWire
        pw::init();

        let sample_rate = config.sample_rate;
        let muxer_config = config.muxer_config.clone();
        let is_running = Arc::new(AtomicBool::new(false));
        let is_running_clone = is_running.clone();

        // Create control channel
        let (control_tx, control_rx) = std::sync::mpsc::channel::<PwCommand>();

        // Create muxer output channel (lock-free for real-time thread)
        let (muxer_tx, muxer_rx) = crossbeam_channel::bounded::<Vec<i16>>(100);

        // Spawn forwarder thread: crossbeam -> async mpsc
        let tx_clone = tx.clone();
        thread::spawn(move || {
            while let Ok(samples) = muxer_rx.recv() {
                if tx_clone.send(samples).is_err() {
                    break;
                }
            }
        });

        // Spawn PipeWire thread
        let thread = thread::Builder::new()
            .name("pipewire-audio".into())
            .spawn(move || {
                if let Err(e) = run_pipewire_thread_multidevice(
                    control_rx,
                    muxer_tx,
                    sample_rate,
                    muxer_config,
                    is_running_clone,
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
        // Enumerate actual PipeWire audio sources
        pw::init();

        let sources = enumerate_audio_sources()?;
        let devices: Vec<DeviceInfo> = sources
            .into_iter()
            .map(|s| DeviceInfo {
                name: s.description,
                is_default: false, // PipeWire doesn't expose default in this enumeration
            })
            .collect();

        if devices.is_empty() {
            // Fallback to showing pipewire as single device
            Ok(vec![DeviceInfo {
                name: "pipewire".to_string(),
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
        // Send flush command to PipeWire thread to flush muxer buffers
        self.control_tx
            .send(PwCommand::Flush)
            .map_err(|_| anyhow!("PipeWire thread not responding"))?;

        // Wait a bit for the flush to complete
        // Timer checks every 10ms, so 20ms should be enough
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
        // Signal the PipeWire thread to quit
        let _ = self.control_tx.send(PwCommand::Quit);
        // Note: We don't join the thread here to avoid blocking
        // The thread will exit when it processes the Quit command
    }
}

/// Enumerate audio source nodes from PipeWire.
///
/// Creates a temporary mainloop to query the registry for all Audio/Source nodes.
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

    // Collect discovered sources
    let sources: Rc<RefCell<Vec<AudioSourceInfo>>> = Rc::new(RefCell::new(Vec::new()));
    let done = Rc::new(Cell::new(false));

    let sources_clone = sources.clone();
    let done_clone = done.clone();
    let mainloop_weak = mainloop.downgrade();

    // Trigger sync to know when enumeration is complete
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
            // Check if this is an Audio/Source node
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

                        // Get object.serial for reliable stream targeting
                        let object_serial = props
                            .get("object.serial")
                            .and_then(|s| s.parse::<u32>().ok())
                            .unwrap_or(global.id); // Fallback to id

                        // Skip monitor/loopback sources
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

    // Run mainloop until sync completes
    while !done.get() {
        mainloop.run();
    }

    let result = sources.borrow().clone();
    info!("Enumerated {} PipeWire audio sources", result.len());
    Ok(result)
}

/// Check if a source name indicates a real input (not monitor/loopback).
fn is_real_audio_source(name: &str, description: &str) -> bool {
    let name_lower = name.to_lowercase();
    let desc_lower = description.to_lowercase();

    // Skip monitor/loopback sources
    if name_lower.contains(".monitor") || desc_lower.contains("monitor") {
        return false;
    }
    // Skip HDMI (usually outputs, not inputs)
    if name_lower.contains("hdmi") || desc_lower.contains("hdmi") {
        return false;
    }
    true
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

/// Run the PipeWire MainLoop with multi-device capture.
///
/// Enumerates all Audio/Source nodes and creates a stream for each,
/// routing audio through StreamMuxer for quality-based selection.
fn run_pipewire_thread_multidevice(
    control_rx: std::sync::mpsc::Receiver<PwCommand>,
    muxer_tx: crossbeam_channel::Sender<Vec<i16>>,
    sample_rate: u32,
    muxer_config: MuxerConfig,
    is_running: Arc<AtomicBool>,
) -> Result<()> {
    // Create StreamMuxer for quality-based stream selection
    let muxer = Rc::new(RefCell::new(StreamMuxer::new(muxer_tx, muxer_config)?));

    // Create MainLoop
    let mainloop = pw::main_loop::MainLoop::new(None)
        .context("Failed to create PipeWire MainLoop")?;

    let context = pw::context::Context::new(&mainloop)
        .context("Failed to create PipeWire Context")?;

    let core = context
        .connect(None)
        .context("Failed to connect to PipeWire daemon")?;

    // Enumerate audio sources
    let sources = enumerate_audio_sources()?;

    if sources.is_empty() {
        warn!("No PipeWire audio sources found, creating default stream");
    }

    // Build audio format pod (shared by all streams)
    let format_buffer = build_audio_format_pod(sample_rate)?;

    // Keep track of streams and their listeners (must stay alive)
    let mut streams: Vec<pw::stream::Stream> = Vec::new();
    let mut _listeners: Vec<pw::stream::StreamListener<()>> = Vec::new();

    if sources.is_empty() {
        // Fallback: create a single stream connected to default source
        let (stream, listener) = create_capture_stream(
            &core,
            None, // Default source
            "default",
            &format_buffer,
            sample_rate,
            muxer.clone(),
            is_running.clone(),
        )?;
        streams.push(stream);
        _listeners.push(listener);
        info!("Created default PipeWire capture stream");
    } else {
        // Create a stream for each audio source
        for source in &sources {
            if !is_real_audio_source(&source.name, &source.description) {
                debug!("Skipping non-input source: {}", source.name);
                continue;
            }

            match create_capture_stream(
                &core,
                Some(source.object_serial),
                &source.name,
                &format_buffer,
                sample_rate,
                muxer.clone(),
                is_running.clone(),
            ) {
                Ok((stream, listener)) => {
                    info!(
                        "Created PipeWire stream for: {} (id={}, serial={})",
                        source.description, source.id, source.object_serial
                    );
                    streams.push(stream);
                    _listeners.push(listener);
                }
                Err(e) => {
                    warn!(
                        "Failed to create stream for '{}': {}",
                        source.description, e
                    );
                }
            }
        }
    }

    if streams.is_empty() {
        return Err(anyhow!("Failed to create any PipeWire capture streams"));
    }

    info!(
        "PipeWire multi-device capture ready with {} stream(s)",
        streams.len()
    );

    // Run mainloop with command polling
    let loop_clone = mainloop.loop_();

    // Add a timer to poll for commands
    let control_rx = std::sync::Arc::new(std::sync::Mutex::new(control_rx));
    let is_running_for_timer = is_running.clone();
    let muxer_for_flush = muxer.clone();
    let mainloop_weak = mainloop.downgrade();

    let _timer = loop_clone.add_timer(move |_| {
        let rx = control_rx.lock().unwrap();
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
                    // Flush muxer buffers to forward any remaining samples
                    if let Ok(mut muxer) = muxer_for_flush.try_borrow_mut() {
                        muxer.flush();
                    }
                    debug!("PipeWire: buffers flushed");
                }
                PwCommand::Quit => {
                    if let Some(ml) = mainloop_weak.upgrade() {
                        ml.quit();
                    }
                }
            }
        }
    });

    // Update timer every 10ms
    let _ = _timer.update_timer(
        Some(std::time::Duration::from_millis(10)),
        Some(std::time::Duration::from_millis(10)),
    );

    // Run the mainloop (blocks until quit)
    mainloop.run();

    info!("PipeWire thread exiting");
    Ok(())
}

/// Create a capture stream for a specific audio source.
///
/// If `target_serial` is None, connects to the default source.
/// Uses object.serial for reliable PipeWire stream targeting.
fn create_capture_stream(
    core: &pw::core::Core,
    target_serial: Option<u32>,
    stream_name: &str,
    format_buffer: &[u8],
    _sample_rate: u32,
    muxer: Rc<RefCell<StreamMuxer>>,
    is_running: Arc<AtomicBool>,
) -> Result<(pw::stream::Stream, pw::stream::StreamListener<()>)> {
    // Create stream properties
    let mut props = pw::properties::properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Communication",
        *pw::keys::NODE_NAME => "voice-dictation",
        *pw::keys::APP_NAME => "Voice Dictation",
    };

    // Target specific node by object.serial (not node.id)
    if let Some(serial) = target_serial {
        props.insert("target.object", serial.to_string());
    }

    let stream = pw::stream::Stream::new(core, stream_name, props)
        .context("Failed to create PipeWire stream")?;

    // Set up stream listener with StreamMuxer integration
    let stream_id = stream_name.to_string();
    let muxer_clone = muxer.clone();
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
                            // Convert f32 samples to i16
                            let f32_samples: &[f32] = unsafe {
                                std::slice::from_raw_parts(
                                    slice[offset..].as_ptr() as *const f32,
                                    size / std::mem::size_of::<f32>(),
                                )
                            };

                            let i16_samples: Vec<i16> = f32_samples
                                .iter()
                                .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
                                .collect();

                            if !i16_samples.is_empty() {
                                // Push to StreamMuxer for quality-based selection
                                if let Ok(mut muxer) = muxer_clone.try_borrow_mut() {
                                    muxer.push_samples(&stream_id, &i16_samples);
                                }
                            }
                        }
                    }
                }
            }
        })
        .register()?;

    // Build format pod reference
    let pod_ref = unsafe { Pod::from_raw(format_buffer.as_ptr() as *const pw::spa::sys::spa_pod) };

    // Connect stream
    // Note: We pass None for target_id since we set "target.object" property instead
    // PipeWire's connect() target_id is for a different purpose (SPA port ID)
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
