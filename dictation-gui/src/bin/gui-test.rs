use anyhow::Result;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use dictation_gui::renderer::SpectrumRenderer;
use dictation_gui::wayland;

use memmap2::MmapMut;
use std::os::fd::AsFd;
use wayland_client::protocol::wl_shm;

const WIDTH: u32 = 400;
const HEIGHT: u32 = 150;
const TEST_DURATION_SECS: u64 = 30;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    
    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║          VOICE DICTATION GUI TEST                     ║");
    println!("╚════════════════════════════════════════════════════════╝\n");
    println!("⚡ Starting Wayland overlay test...");
    println!("📺 Window size: {}x{}", WIDTH, HEIGHT);
    println!("📍 Position: Bottom center of screen");
    println!("⏱️  Duration: {} seconds", TEST_DURATION_SECS);
    println!("\n🎨 You should see:");
    println!("   • Animated spectrum bars (orange/pink)");
    println!("   • Text: 'Testing GUI - Can you see this?'");
    println!("   • Bars moving like audio visualization");
    println!("\n🔍 If you see NOTHING:");
    println!("   • Check hyprctl layers | grep voice");
    println!("   • The Wayland layer-shell may not be working\n");
    
    let band_values = Arc::new(Mutex::new(vec![0.0f32; 8]));
    let transcription_text = Arc::new(Mutex::new(String::from("Testing GUI - Can you see this?")));
    let is_finalizing = Arc::new(Mutex::new(false));
    
    let band_values_clone = band_values.clone();
    let transcription_text_clone = transcription_text.clone();
    
    let wayland_thread = thread::spawn(move || {
        if let Err(e) = run_test_window(band_values_clone, transcription_text_clone, is_finalizing) {
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
        if let Some(line) = layers_output.lines().find(|l| l.contains("voice-dictation")) {
            println!("   Layer info: {}", line.trim());
        }
    } else {
        println!("❌ PROBLEM! Hyprland does NOT see 'voice-dictation' layer");
        println!("   Checking overlay level:");
        for line in layers_output.lines() {
            if line.contains("Layer level 3 (overlay)") {
                println!("   {}", line);
                let mut in_overlay = false;
                for next_line in layers_output.lines().skip_while(|l| l != &line) {
                    if in_overlay && next_line.contains("Layer level") {
                        break;
                    }
                    if in_overlay && next_line.trim().is_empty() {
                        println!("   (overlay level is EMPTY)");
                        break;
                    }
                    if in_overlay {
                        println!("   {}", next_line);
                    }
                    if next_line.contains("Layer level 3 (overlay)") {
                        in_overlay = true;
                    }
                }
            }
        }
    }
    println!();
    
    let start = Instant::now();
    let mut frame_count = 0;
    let mut last_second = 0;
    
    while start.elapsed() < Duration::from_secs(TEST_DURATION_SECS) {
        let elapsed = start.elapsed().as_secs_f32();
        
        {
            let mut bands = band_values.lock().unwrap();
            for (i, band) in bands.iter_mut().enumerate() {
                let freq = 0.5 + i as f32 * 0.3;
                *band = (0.3 + 0.7 * (elapsed * freq + i as f32).sin()).abs();
            }
        }
        
        {
            let mut text = transcription_text.lock().unwrap();
            *text = format!(
                "Testing GUI - Frame {} - Time: {:.1}s",
                frame_count,
                elapsed
            );
        }
        
        let current_second = elapsed as u64;
        if current_second > last_second {
            println!("⏱️  {} seconds elapsed - {} frames rendered", current_second, frame_count);
            last_second = current_second;
        }
        
        frame_count += 1;
        thread::sleep(Duration::from_millis(50));
    }
    
    println!("\n✓ Test complete! Shutting down...");
    println!("   Total frames rendered: {}", frame_count);
    
    std::process::exit(0);
}

fn run_test_window(
    band_values: Arc<Mutex<Vec<f32>>>,
    transcription_text: Arc<Mutex<String>>,
    is_finalizing: Arc<Mutex<bool>>,
) -> Result<()> {
    println!("🔧 Creating Wayland connection...");
    let (mut app_state, conn, _qh) = wayland::AppState::new()?;
    println!("✓ Wayland connection established");
    
    let mut event_queue = conn.new_event_queue::<wayland::AppState>();
    let qh2 = event_queue.handle();
    
    println!("🔧 Creating layer surface...");
    app_state.create_layer_surface(&qh2);
    
    println!("🔧 Processing Wayland events and waiting for configure...");
    
    // Do a blocking roundtrip to ensure the compositor processes our surface
    conn.roundtrip()?;
    println!("✓ Roundtrip complete");
    
    // Now wait for the configure event with blocking dispatch
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
        println!("⚠️  Warning: Wayland configure timeout after 5 seconds");
        println!("   This means the compositor never acknowledged our surface");
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
    loop {
        let _ = event_queue.dispatch_pending(&mut app_state);
        conn.flush()?;
        
        if let Some(context) = &app_state.context {
            let finalizing = *is_finalizing.lock().unwrap();
            let values = if finalizing {
                vec![0.0; 8]
            } else {
                band_values.lock().unwrap().clone()
            };
            let text = transcription_text.lock().unwrap().clone();
            let pixels = renderer.render(&values, &text);
            
            for i in (0..pixels.len()).step_by(4) {
                mmap[i] = pixels[i + 2];
                mmap[i + 1] = pixels[i + 1];
                mmap[i + 2] = pixels[i];
                mmap[i + 3] = pixels[i + 3];
            }
            mmap.flush()?;
            
            context.wl_surface.attach(Some(&buffer), 0, 0);
            context
                .wl_surface
                .damage_buffer(0, 0, WIDTH as i32, HEIGHT as i32);
            context.wl_surface.commit();
            
            frame += 1;
            if frame == 1 {
                println!("🎉 FIRST FRAME RENDERED - GUI IS NOW VISIBLE!");
            }
        }
        
        thread::sleep(Duration::from_millis(16));
    }
}
