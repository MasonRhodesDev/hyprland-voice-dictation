use anyhow::Result;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use dictation_gui::{renderer::SpectrumRenderer, wayland, GuiState};

use memmap2::MmapMut;
use std::os::fd::AsFd;
use wayland_client::protocol::wl_shm;

const WIDTH: u32 = 400;
const HEIGHT: u32 = 150;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║          VOICE DICTATION GUI STATE TEST               ║");
    println!("╚════════════════════════════════════════════════════════╝\n");
    println!("⚡ Testing state-driven GUI with animations...");
    println!("📺 Window size: {}x{}", WIDTH, HEIGHT);
    println!("📍 Position: Bottom center of screen");
    println!("\n🎨 Test sequence:");
    println!("   0-8s:  LISTENING state (narrow bars, live text)");
    println!("   8-11s: PROCESSING state (spinning dots, rounded)");
    println!("   11-11.5s: CLOSING state (fade out animation)");
    println!("   11.5s: GUI exits");
    println!();

    let band_values = Arc::new(Mutex::new(vec![0.0f32; 8]));
    let transcription_text = Arc::new(Mutex::new(String::from("Testing GUI - Can you see this?")));
    let gui_state = Arc::new(Mutex::new(GuiState::Listening));

    let band_values_clone = band_values.clone();
    let transcription_text_clone = transcription_text.clone();
    let gui_state_clone = gui_state.clone();

    let _wayland_thread = thread::spawn(move || {
        if let Err(e) =
            run_test_window(band_values_clone, transcription_text_clone, gui_state_clone)
        {
            eprintln!("❌ Wayland thread error: {}", e);
        }
    });

    println!("✓ Wayland thread started");
    thread::sleep(Duration::from_millis(2500));

    println!("\n🔍 Checking Hyprland layers...");
    let output = std::process::Command::new("hyprctl")
        .arg("layers")
        .output()
        .expect("Failed to run hyprctl");

    let layers_output = String::from_utf8_lossy(&output.stdout);

    if layers_output.contains("voice-dictation") {
        println!("✅ SUCCESS! Hyprland sees 'voice-dictation' layer");
    } else {
        println!("❌ PROBLEM! Hyprland does NOT see 'voice-dictation' layer");
    }
    println!();

    let start = Instant::now();
    let mut frame_count = 0;
    let mut last_second = 0;

    loop {
        let elapsed = start.elapsed().as_secs_f32();

        // Animate spectrum bars
        {
            let mut bands = band_values.lock().unwrap();
            for (i, band) in bands.iter_mut().enumerate() {
                let freq = 0.5 + i as f32 * 0.3;
                *band = (0.3 + 0.7 * (elapsed * freq + i as f32).sin()).abs();
            }
        }

        // Update text with frame counter
        {
            let mut text = transcription_text.lock().unwrap();
            let current_state = *gui_state.lock().unwrap();
            *text = format!(
                "Frame {} - Time: {:.1}s - State: {:?}",
                frame_count, elapsed, current_state
            );
        }

        // State transitions
        if (8.0..8.1).contains(&elapsed) {
            let mut state = gui_state.lock().unwrap();
            if *state == GuiState::Listening {
                println!("⏱️  8s: Transitioning to PROCESSING state");
                *state = GuiState::Processing;
                *transcription_text.lock().unwrap() = "Processing your speech...".to_string();
            }
        }

        if (11.0..11.1).contains(&elapsed) {
            let mut state = gui_state.lock().unwrap();
            if *state == GuiState::Processing {
                println!("⏱️  11s: Transitioning to CLOSING state");
                *state = GuiState::Closing;
            }
        }

        if elapsed >= 11.5 {
            println!("\n✓ Test complete! All states tested.");
            println!("   Total frames rendered: {}", frame_count);
            std::process::exit(0);
        }

        let current_second = elapsed as u64;
        if current_second > last_second {
            println!("⏱️  {} seconds elapsed - {} frames rendered", current_second, frame_count);
            last_second = current_second;
        }

        frame_count += 1;
        thread::sleep(Duration::from_millis(50));
    }
}

fn run_test_window(
    band_values: Arc<Mutex<Vec<f32>>>,
    transcription_text: Arc<Mutex<String>>,
    gui_state: Arc<Mutex<GuiState>>,
) -> Result<()> {
    println!("🔧 Creating Wayland connection...");
    let (mut app_state, conn, _qh) = wayland::AppState::new()?;
    println!("✓ Wayland connection established");

    let mut event_queue = conn.new_event_queue::<wayland::AppState>();
    let qh2 = event_queue.handle();

    println!("🔧 Creating layer surface...");
    app_state.create_layer_surface(&qh2, WIDTH, HEIGHT);

    println!("🔧 Processing Wayland events and waiting for configure...");

    conn.roundtrip()?;
    println!("✓ Roundtrip complete");

    let start = Instant::now();
    while !app_state.configured && start.elapsed() < Duration::from_secs(5) {
        match event_queue.blocking_dispatch(&mut app_state) {
            Ok(_) => {
                conn.flush()?;
                if app_state.configured {
                    println!("✓ Wayland surface configured by compositor!");
                    break;
                }
            }
            Err(e) => {
                println!("⚠️  Dispatch error: {:?}", e);
                break;
            }
        }
        thread::sleep(Duration::from_millis(10));
    }

    if !app_state.configured {
        println!("⚠️  Warning: Wayland configure timeout (this may be normal)");
    }

    let mut shm: Option<wl_shm::WlShm> = None;
    for global in app_state.registry_state.globals() {
        if global.interface == "wl_shm" {
            let version = global.version.min(1);
            shm = Some(app_state.registry_state.bind_specific(
                &qh2,
                global.name,
                version..=version,
                (),
            )?);
            break;
        }
    }

    let shm = shm.ok_or_else(|| anyhow::anyhow!("wl_shm not found"))?;
    println!("✓ Shared memory (wl_shm) ready");

    let stride = WIDTH * 4;
    let size = stride * HEIGHT;

    let tmp_file = tempfile::tempfile()?;
    tmp_file.set_len(size as u64)?;

    let pool = shm.create_pool(tmp_file.as_fd(), size as i32, &qh2, ());
    let buffer = pool.create_buffer(
        0,
        WIDTH as i32,
        HEIGHT as i32,
        stride as i32,
        wl_shm::Format::Argb8888,
        &qh2,
        (),
    );

    let mut mmap = unsafe { MmapMut::map_mut(&tmp_file)? };
    let mut renderer = SpectrumRenderer::new(WIDTH, HEIGHT)?;

    println!("✓ Renderer initialized");
    println!("✓ Buffer created and mapped");
    println!("🎨 Starting render loop...\n");

    let mut frame = 0;
    let start_time = Instant::now();
    let mut previous_state = GuiState::Listening;
    let mut state_start_time = Instant::now();

    loop {
        let _ = event_queue.dispatch_pending(&mut app_state);
        conn.flush()?;

        if let Some(context) = &app_state.context {
            let current_state = *gui_state.lock().unwrap();

            // Track state transitions
            if current_state != previous_state {
                state_start_time = Instant::now();
                previous_state = current_state;
            }

            let state_elapsed = state_start_time.elapsed().as_secs_f32();

            // Exit after closing animation
            if current_state == GuiState::Closing && state_elapsed > 0.5 {
                break;
            }

            let values = band_values.lock().unwrap().clone();
            let text = transcription_text.lock().unwrap().clone();
            let total_elapsed = start_time.elapsed().as_secs_f32();
            let pixels =
                renderer.render(&values, &text, current_state, state_elapsed, total_elapsed);

            for i in (0..pixels.len()).step_by(4) {
                mmap[i] = pixels[i + 2];
                mmap[i + 1] = pixels[i + 1];
                mmap[i + 2] = pixels[i];
                mmap[i + 3] = pixels[i + 3];
            }
            mmap.flush()?;

            context.wl_surface.attach(Some(&buffer), 0, 0);
            context.wl_surface.damage_buffer(0, 0, WIDTH as i32, HEIGHT as i32);
            context.wl_surface.commit();

            frame += 1;
            if frame == 1 {
                println!("🎉 FIRST FRAME RENDERED - GUI IS NOW VISIBLE!");
            }
        }

        thread::sleep(Duration::from_millis(16));
    }

    Ok(())
}
