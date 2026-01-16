//! PipeWire native audio backend.
//!
//! This backend uses pipewire-rs for native PipeWire audio capture,
//! enabling proper mic sharing with browsers without requiring idle timeouts.

use anyhow::{anyhow, Context, Result};
use pipewire as pw;
use pw::spa::param::audio::{AudioFormat, AudioInfoRaw};
use pw::spa::pod::Pod;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use tokio::sync::mpsc;
use tracing::{debug, error, info};

use super::{AudioBackend, AudioBackendConfig, AudioBackendFactory, DeviceInfo};

/// Commands sent to the PipeWire thread.
enum PwCommand {
    Start,
    Stop,
    Quit,
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
        info!("Creating PipeWire audio backend...");

        // Initialize PipeWire
        pw::init();

        let sample_rate = config.sample_rate;
        let is_running = Arc::new(AtomicBool::new(false));
        let is_running_clone = is_running.clone();

        // Create control channel
        let (control_tx, control_rx) = std::sync::mpsc::channel::<PwCommand>();

        // Create audio sample channel (lock-free for real-time thread)
        let (audio_tx, audio_rx) = crossbeam_channel::unbounded::<Vec<i16>>();

        // Spawn forwarder thread: crossbeam -> async mpsc
        let tx_clone = tx.clone();
        thread::spawn(move || {
            while let Ok(samples) = audio_rx.recv() {
                if tx_clone.send(samples).is_err() {
                    break;
                }
            }
        });

        // Spawn PipeWire thread
        let thread = thread::Builder::new()
            .name("pipewire-audio".into())
            .spawn(move || {
                if let Err(e) = run_pipewire_thread(control_rx, audio_tx, sample_rate, is_running_clone) {
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
        // PipeWire handles device routing automatically via the session manager
        // We just expose "default" which routes through PipeWire's configured default
        Ok(vec![DeviceInfo {
            name: "pipewire".to_string(),
            is_default: true,
        }])
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

/// Run the PipeWire MainLoop on a dedicated thread.
fn run_pipewire_thread(
    control_rx: std::sync::mpsc::Receiver<PwCommand>,
    audio_tx: crossbeam_channel::Sender<Vec<i16>>,
    sample_rate: u32,
    is_running: Arc<AtomicBool>,
) -> Result<()> {
    // Create MainLoop
    let mainloop = pw::main_loop::MainLoop::new(None)
        .context("Failed to create PipeWire MainLoop")?;

    let context = pw::context::Context::new(&mainloop)
        .context("Failed to create PipeWire Context")?;

    let core = context
        .connect(None)
        .context("Failed to connect to PipeWire daemon")?;

    // Set up audio format
    let mut audio_info = AudioInfoRaw::new();
    audio_info.set_format(AudioFormat::F32LE);
    audio_info.set_rate(sample_rate);
    audio_info.set_channels(1); // Mono for speech recognition

    // Build format pod
    let mut buffer = vec![0u8; 1024];
    let _pod = pw::spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(&mut buffer),
        &pw::spa::pod::Value::Object(pw::spa::pod::Object {
            type_: pw::spa::sys::SPA_TYPE_OBJECT_Format,
            id: pw::spa::sys::SPA_PARAM_EnumFormat,
            properties: audio_info.into(),
        }),
    )
    .context("Failed to serialize audio format")?;

    let pod_ref = unsafe {
        Pod::from_raw(buffer.as_ptr() as *const pw::spa::sys::spa_pod)
    };

    // Create stream with properties
    let props = pw::properties::properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Communication",
        *pw::keys::NODE_NAME => "voice-dictation",
        *pw::keys::APP_NAME => "Voice Dictation",
    };

    let stream = pw::stream::Stream::new(&core, "audio-capture", props)
        .context("Failed to create PipeWire stream")?;

    // Set up stream listener
    let audio_tx_clone = audio_tx.clone();
    let is_running_clone = is_running.clone();

    let _listener = stream
        .add_local_listener_with_user_data(())
        .param_changed(|_, _, id, param| {
            if id == pw::spa::param::ParamType::Format.as_raw() {
                if let Some(_param) = param {
                    debug!("PipeWire stream format changed");
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
                                let _ = audio_tx_clone.try_send(i16_samples);
                            }
                        }
                    }
                }
            }
        })
        .register()?;

    // Connect stream
    stream.connect(
        pw::spa::utils::Direction::Input,
        None, // Connect to default source
        pw::stream::StreamFlags::AUTOCONNECT
            | pw::stream::StreamFlags::MAP_BUFFERS
            | pw::stream::StreamFlags::RT_PROCESS,
        &mut [pod_ref],
    )?;

    info!("PipeWire stream connected");

    // Run mainloop with command polling
    let loop_clone = mainloop.loop_();

    // Add a timer to poll for commands
    let control_rx = std::sync::Arc::new(std::sync::Mutex::new(control_rx));
    let is_running_for_timer = is_running.clone();
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
