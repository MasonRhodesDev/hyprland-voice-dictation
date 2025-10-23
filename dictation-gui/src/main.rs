use anyhow::{Context, Result};
use std::sync::{Arc, Mutex};
use std::thread;
use tracing::{error, info};

mod control_ipc;
mod fft;
mod ipc;
mod renderer;
mod wayland;

use control_ipc::ControlMessage;
use dictation_gui::GuiState;
use fft::SpectrumAnalyzer;
use renderer::SpectrumRenderer;

const SOCKET_PATH: &str = "/tmp/voice-dictation.sock";
const CONTROL_SOCKET_PATH: &str = "/tmp/voice-dictation-control.sock";
const WIDTH: u32 = 400;
const MIN_HEIGHT: u32 = 55;
const MAX_HEIGHT: u32 = 200;
const SAMPLE_RATE: u32 = 16000;
const FFT_SIZE: usize = 512;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    info!("Starting dictation-gui");

    let band_values = Arc::new(Mutex::new(vec![0.0f32; 8]));
    let band_values_clone = band_values.clone();
    
    let transcription_text = Arc::new(Mutex::new(String::new()));
    let transcription_text_clone = transcription_text.clone();
    
    let gui_state = Arc::new(Mutex::new(GuiState::Listening));
    let gui_state_clone = gui_state.clone();

    thread::spawn(move || {
        info!("Wayland thread starting...");
        match run_wayland_window(band_values_clone, transcription_text_clone, gui_state_clone) {
            Ok(_) => info!("Wayland thread exited normally"),
            Err(e) => error!("Wayland thread error: {}", e),
        }
    });

    let mut ipc_client = ipc::IpcClient::new(SOCKET_PATH.to_string());
    let mut spectrum_analyzer = SpectrumAnalyzer::new(FFT_SIZE, SAMPLE_RATE);
    let mut audio_connected = false;

    info!("GUI initialized");

    let transcription_clone = transcription_text.clone();
    let gui_state_clone2 = gui_state.clone();
    tokio::spawn(async move {
        let mut control = control_ipc::ControlClient::new(CONTROL_SOCKET_PATH.to_string());
        loop {
            if control.connect().await.is_ok() {
                info!("Connected to control socket");
                loop {
                    match control.receive().await {
                        Ok(ControlMessage::TranscriptionUpdate { text, is_final }) => {
                            info!("Transcription: '{}' (final: {})", text, is_final);
                            if let Ok(mut locked) = transcription_clone.lock() {
                                *locked = text.clone();
                            }
                        }
                        Ok(ControlMessage::Ready) => {
                            info!("Engine ready");
                        }
                        Ok(ControlMessage::ProcessingStarted) => {
                            info!("Entering processing state");
                            if let Ok(mut locked) = gui_state_clone2.lock() {
                                *locked = GuiState::Processing;
                            }
                        }
                        Ok(ControlMessage::Complete) => {
                            info!("Entering closing state");
                            if let Ok(mut locked) = gui_state_clone2.lock() {
                                if *locked == GuiState::Processing {
                                    *locked = GuiState::Closing;
                                }
                            }
                        }
                        Ok(_) => {}
                        Err(e) => {
                            error!("Control receive error: {}", e);
                            break;
                        }
                    }
                }
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    });

    let mut frame_count = 0;
    loop {
        if !audio_connected {
            if ipc_client.connect().await.is_ok() {
                info!("Connected to audio socket");
                audio_connected = true;
            } else {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                continue;
            }
        }

        match ipc_client.receive_samples().await {
            Ok(samples) => {
                let new_values = spectrum_analyzer.process(&samples);
                
                if let Ok(mut locked) = band_values.lock() {
                    *locked = new_values;
                }

                frame_count += 1;
                if frame_count == 1 {
                    info!("Receiving and processing audio");
                }
            }
            Err(e) => {
                error!("Audio IPC error: {}. Reconnecting...", e);
                audio_connected = false;
                let _ = ipc_client.reconnect().await;
            }
        }
    }
}

fn run_wayland_window(band_values: Arc<Mutex<Vec<f32>>>, transcription_text: Arc<Mutex<String>>, gui_state: Arc<Mutex<GuiState>>) -> Result<()> {
    use memmap2::MmapMut;
    use std::os::fd::AsFd;
    use wayland_client::protocol::{wl_shm};

    let current_width = WIDTH;
    let mut current_height = MIN_HEIGHT;

    info!("Creating Wayland connection...");
    let (mut app_state, conn, _qh) = wayland::AppState::new()?;
    info!("Wayland connection established");
    
    let mut event_queue = conn.new_event_queue::<wayland::AppState>();
    let qh2 = event_queue.handle();
    
    info!("Creating layer surface...");
    app_state.create_layer_surface(&qh2, current_width, current_height);
    
    info!("Processing Wayland events and waiting for configure...");
    
    // Do a blocking roundtrip to ensure the compositor processes our surface
    conn.roundtrip()?;
    info!("Roundtrip complete");
    
    // Now wait for the configure event with blocking dispatch
    let start = std::time::Instant::now();
    while !app_state.configured && start.elapsed() < std::time::Duration::from_secs(5) {
        match event_queue.blocking_dispatch(&mut app_state) {
            Ok(_) => {
                conn.flush()?;
                if app_state.configured {
                    info!("Wayland surface configured by compositor!");
                    break;
                }
            }
            Err(e) => {
                error!("Dispatch error: {:?}", e);
                break;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    
    if !app_state.configured {
        error!("Wayland configure timeout after 5 seconds - compositor not responding");
        return Err(anyhow::anyhow!("Configure timeout - compositor not responding"));
    }
    
    let mut shm: Option<wl_shm::WlShm> = None;
    for global in app_state.registry_state.globals() {
        if global.interface == "wl_shm" {
            let version = global.version.min(1);
            shm = Some(
                app_state
                    .registry_state
                    .bind_specific(&qh2, global.name, version..=version, ())?,
            );
            break;
        }
    }
    
    let shm = shm.context("wl_shm not found")?;

    let stride = current_width * 4;
    
    let tmp_file = tempfile::tempfile()?;
    tmp_file.set_len((stride * 400) as u64)?;
    
    let pool = shm.create_pool(tmp_file.as_fd(), (stride * 400) as i32, &qh2, ());
    let mut buffer = pool.create_buffer(
        0,
        current_width as i32,
        current_height as i32,
        stride as i32,
        wl_shm::Format::Argb8888,
        &qh2,
        (),
    );

    let mut mmap = unsafe { MmapMut::map_mut(&tmp_file)? };
    let mut renderer = SpectrumRenderer::new(current_width, current_height)?;

    info!("Wayland layer surface ready");

    let mut frame = 0;
    let mut previous_state = GuiState::Listening;
    let mut state_start_time = std::time::Instant::now();
    let mut last_text = String::new();
    let mut display_text = String::new();
    
    loop {
        // Non-blocking dispatch to process Wayland events
        let _ = event_queue.dispatch_pending(&mut app_state);
        conn.flush()?;

        if let Some(context) = &app_state.context {
            let current_state = *gui_state.lock().unwrap();
            
            // Track state changes
            if current_state != previous_state {
                info!("State transition: {:?} -> {:?}", previous_state, current_state);
                state_start_time = std::time::Instant::now();
                
                // When entering Processing, capture the current text
                if current_state == GuiState::Processing {
                    display_text = last_text.clone();
                }
                
                previous_state = current_state;
            }
            
            let state_elapsed = state_start_time.elapsed().as_secs_f32();
            
            // Exit after closing animation completes
            if current_state == GuiState::Closing && state_elapsed > 0.3 {
                info!("Closing animation complete, exiting");
                break;
            }
            
            let values = band_values.lock().unwrap().clone();
            let text = transcription_text.lock().unwrap().clone();
            
            // Update display text only during Listening state
            if current_state == GuiState::Listening {
                display_text = text.clone();
            }
            
            // Resize window if text changed during Listening
            if text != last_text && current_state == GuiState::Listening {
                let new_height = renderer::calculate_text_height(&text, current_width).min(MAX_HEIGHT);
                if new_height != current_height {
                    current_height = new_height;
                    renderer = SpectrumRenderer::new(current_width, current_height)?;
                    
                    buffer = pool.create_buffer(
                        0,
                        current_width as i32,
                        current_height as i32,
                        stride as i32,
                        wl_shm::Format::Argb8888,
                        &qh2,
                        (),
                    );
                    
                    if let Some(layer_surface) = &context.layer_surface {
                        layer_surface.set_size(current_width, current_height);
                        context.wl_surface.commit();
                    }
                }
                last_text = text.clone();
            }
            
            let pixels = renderer.render(&values, &display_text, current_state, state_elapsed);

            // Convert RGBA to BGRA (tiny-skia outputs RGBA, Wayland ARGB8888 is actually BGRA in memory)
            let pixel_count = (current_width * current_height * 4) as usize;
            for i in (0..pixel_count).step_by(4) {
                mmap[i] = pixels[i + 2];     // B
                mmap[i + 1] = pixels[i + 1]; // G  
                mmap[i + 2] = pixels[i];     // R
                mmap[i + 3] = pixels[i + 3]; // A
            }
            mmap.flush()?;

            context.wl_surface.attach(Some(&buffer), 0, 0);
            context
                .wl_surface
                .damage_buffer(0, 0, current_width as i32, current_height as i32);
            context.wl_surface.commit();

            frame += 1;
            if frame == 1 {
                info!("GUI overlay visible");
            }
        }
        
        std::thread::sleep(std::time::Duration::from_millis(16)); // ~60 FPS
    }
    
    Ok(())
}
