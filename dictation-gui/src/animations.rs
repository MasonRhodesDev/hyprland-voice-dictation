use tiny_skia::*;

#[derive(Clone, Copy)]
pub struct Colors {
    pub background: Color,
    pub bar: Color,
}

pub enum ClosingAnimation {
    Collapse,
}

pub fn render_collapse(
    pixmap: &mut Pixmap,
    colors: Colors,
    state_elapsed: f32,
    total_time: f32,
    width: u32,
    height: u32,
) {
    let duration = 0.5;
    let progress = (state_elapsed / duration).min(1.0);

    let center_x = width as f32 / 2.0;
    let center_y = height as f32 / 2.0;

    let dot_count = 3;
    let dot_radius = 6.0;
    let orbit_radius = 20.0;
    let rotation_speed = 2.0;

    let mut paint = Paint { anti_alias: true, ..Default::default() };

    let ease = ease_in_cubic(progress);

    let padding = 12.0;
    let spinner_diameter = (orbit_radius + dot_radius) * 2.0;
    let box_size = (spinner_diameter + padding * 2.0) * (1.0 - ease);

    if box_size > 1.0 {
        let box_x = (width as f32 - box_size) / 2.0;
        let box_y = (height as f32 - box_size) / 2.0;

        let alpha = 1.0 - ease;
        let mut bg_color = colors.background;
        bg_color.apply_opacity(alpha);
        paint.set_color(bg_color);

        let corner_radius = 50.0 * (1.0 - ease);
        let path = create_rounded_rect(box_x, box_y, box_size, box_size, corner_radius);

        pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
    }

    let current_orbit_radius = orbit_radius * (1.0 - ease);
    let current_dot_radius = dot_radius * (1.0 - ease);

    let alpha = 1.0 - ease;
    let mut dot_color = colors.bar;
    dot_color.apply_opacity(alpha);
    paint.set_color(dot_color);

    if current_dot_radius > 0.1 {
        for i in 0..dot_count {
            let angle = (total_time * rotation_speed)
                + (i as f32 * std::f32::consts::TAU / dot_count as f32);
            let x = center_x + current_orbit_radius * angle.cos();
            let y = center_y + current_orbit_radius * angle.sin();

            if let Some(path) = create_circle(x, y, current_dot_radius) {
                pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
            }
        }
    }
}

fn ease_in_cubic(t: f32) -> f32 {
    t * t * t
}

fn create_circle(cx: f32, cy: f32, radius: f32) -> Option<Path> {
    let mut pb = PathBuilder::new();
    pb.move_to(cx + radius, cy);

    let kappa = 0.5522848;
    let kr = radius * kappa;

    pb.cubic_to(cx + radius, cy - kr, cx + kr, cy - radius, cx, cy - radius);
    pb.cubic_to(cx - kr, cy - radius, cx - radius, cy - kr, cx - radius, cy);
    pb.cubic_to(cx - radius, cy + kr, cx - kr, cy + radius, cx, cy + radius);
    pb.cubic_to(cx + kr, cy + radius, cx + radius, cy + kr, cx + radius, cy);
    pb.close();

    pb.finish()
}

fn create_rounded_rect(x: f32, y: f32, width: f32, height: f32, radius: f32) -> Path {
    let mut pb = PathBuilder::new();
    let radius = radius.min(width / 2.0).min(height / 2.0);

    pb.move_to(x + radius, y);
    pb.line_to(x + width - radius, y);
    pb.quad_to(x + width, y, x + width, y + radius);
    pb.line_to(x + width, y + height - radius);
    pb.quad_to(x + width, y + height, x + width - radius, y + height);
    pb.line_to(x + radius, y + height);
    pb.quad_to(x, y + height, x, y + height - radius);
    pb.line_to(x, y + radius);
    pb.quad_to(x, y, x + radius, y);
    pb.close();

    pb.finish().unwrap()
}
