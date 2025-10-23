use anyhow::Result;
use dictation_gui::animations::{self, Colors};
use memmap2::MmapMut;
use std::os::fd::AsFd;
use std::time::Instant;
use tiny_skia::{Color, Pixmap};
use wayland_client::protocol::wl_shm;

const WIDTH: u32 = 400;
const HEIGHT: u32 = 100;

fn main() -> Result<()> {
    println!("Animation Test - Press Ctrl+C to exit");
    println!("Testing: Collapse animation");
    println!("Duration: 0.7 seconds, looping");

    let (mut app_state, conn, _qh) = dictation_gui::wayland::AppState::new()?;

    let mut event_queue = conn.new_event_queue::<dictation_gui::wayland::AppState>();
    let qh2 = event_queue.handle();

    app_state.create_layer_surface(&qh2, WIDTH, HEIGHT);

    conn.roundtrip()?;

    let start = std::time::Instant::now();
    while !app_state.configured && start.elapsed() < std::time::Duration::from_secs(5) {
        match event_queue.blocking_dispatch(&mut app_state) {
            Ok(_) => {
                conn.flush()?;
                if app_state.configured {
                    break;
                }
            }
            Err(e) => {
                eprintln!("Dispatch error: {:?}", e);
                break;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    if !app_state.configured {
        return Err(anyhow::anyhow!("Configure timeout"));
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

    let stride = WIDTH * 4;
    let tmp_file = tempfile::tempfile()?;
    tmp_file.set_len((stride * HEIGHT) as u64)?;

    let pool = shm.create_pool(tmp_file.as_fd(), (stride * HEIGHT) as i32, &qh2, ());
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

    let colors = Colors {
        background: Color::from_rgba8(0, 0, 0, 230),
        bar: Color::from_rgba8(255, 255, 255, 255),
    };

    println!("\nAnimation running...\n");

    let animation_start = Instant::now();
    let mut loop_count = 0;

    loop {
        let _ = event_queue.dispatch_pending(&mut app_state);
        conn.flush()?;

        if let Some(context) = &app_state.context {
            let elapsed = animation_start.elapsed().as_secs_f32();
            let animation_time = elapsed % 0.9;

            if animation_time < 0.05 && elapsed > 0.5 {
                loop_count += 1;
                println!("Loop #{} complete - repeating animation", loop_count);
            }

            let mut pixmap = Pixmap::new(WIDTH, HEIGHT).unwrap();
            pixmap.fill(Color::TRANSPARENT);

            animations::render_collapse(
                &mut pixmap,
                colors,
                animation_time,
                elapsed,
                WIDTH,
                HEIGHT,
            );

            let pixels = pixmap.data();
            let pixel_count = (WIDTH * HEIGHT * 4) as usize;
            for i in (0..pixel_count).step_by(4) {
                mmap[i] = pixels[i + 2];
                mmap[i + 1] = pixels[i + 1];
                mmap[i + 2] = pixels[i];
                mmap[i + 3] = pixels[i + 3];
            }
            mmap.flush()?;

            context.wl_surface.attach(Some(&buffer), 0, 0);
            context.wl_surface.damage_buffer(0, 0, WIDTH as i32, HEIGHT as i32);
            context.wl_surface.commit();
        }

        std::thread::sleep(std::time::Duration::from_millis(16));
    }
}
