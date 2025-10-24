use crate::animation::{ease_spinner_rotation, CollapseAnimation, HeightAnimation};
use crate::text_renderer::TextRenderer;
use crate::{animations, GuiState};
use anyhow::Result;
use tiny_skia::*;

const BAR_COUNT: usize = 8;
const MIN_BAR_HEIGHT: f32 = 5.0;
const MAX_BAR_HEIGHT: f32 = 30.0;
const CORNER_RADIUS: f32 = 25.0;
const CORNER_RADIUS_PROCESSING: f32 = 50.0;
const BAR_WIDTH_FACTOR: f32 = 0.6;
const BAR_SPACING: f32 = 8.0;
const SPECTRUM_HEIGHT: f32 = 50.0;

#[derive(Clone, Copy)]
struct Colors {
    background: Color,
    bar: Color,
}

impl Default for Colors {
    fn default() -> Self {
        Self {
            background: Color::from_rgba8(0, 0, 0, 230),
            bar: Color::from_rgba8(255, 255, 255, 255),
        }
    }
}

pub struct ModernRenderer {
    width: u32,
    height: u32,
    pixmap: Pixmap,
    colors: Colors,
    text_renderer: TextRenderer,
    height_animation: Option<HeightAnimation>,
    collapse_animation: Option<CollapseAnimation>,
}

impl ModernRenderer {
    pub fn new(width: u32, height: u32) -> Result<Self> {
        let pixmap = Pixmap::new(width, height).expect("Failed to create pixmap");
        let colors = Self::load_colors();
        let text_renderer = TextRenderer::new();

        Ok(Self {
            width,
            height,
            pixmap,
            colors,
            text_renderer,
            height_animation: None,
            collapse_animation: None,
        })
    }

    fn load_colors() -> Colors {
        let config_path =
            std::env::var("HOME").map(|h| format!("{}/.config/matugen/colors.css", h)).ok();

        if let Some(path) = config_path {
            if std::path::Path::new(&path).exists() {
                if let Ok(contents) = std::fs::read_to_string(&path) {
                    return Self::parse_colors(&contents);
                }
            }
        }

        Colors::default()
    }

    fn parse_colors(css: &str) -> Colors {
        let mut bg = None;
        let mut primary = None;

        for line in css.lines() {
            let line = line.trim();
            if line.starts_with("@define-color surface ") {
                bg = Self::parse_color_value(line);
            } else if line.starts_with("@define-color primary ") {
                primary = Self::parse_color_value(line);
            }
        }

        Colors {
            background: bg.unwrap_or_else(|| Color::from_rgba8(0, 0, 0, 230)),
            bar: primary.unwrap_or_else(|| Color::from_rgba8(255, 255, 255, 255)),
        }
    }

    fn parse_color_value(line: &str) -> Option<Color> {
        let hex = line.split('#').nth(1)?.split(';').next()?;
        let hex = hex.trim();

        if hex.len() == 6 {
            let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
            Some(Color::from_rgba8(r, g, b, 230))
        } else {
            None
        }
    }

    pub fn start_height_transition(&mut self, target: f32) {
        self.height_animation = Some(HeightAnimation::new(self.height as f32, target));
    }

    pub fn get_current_height(&self) -> u32 {
        if let Some(anim) = &self.height_animation {
            if !anim.is_complete() {
                return anim.current_value() as u32;
            }
        }
        self.height
    }

    pub fn render(
        &mut self,
        band_values: &[f32],
        text: &str,
        state: GuiState,
        state_time: f32,
        total_time: f32,
    ) -> &[u8] {
        self.pixmap.fill(Color::TRANSPARENT);

        match state {
            GuiState::PreListening => self.render_listening(band_values, text),
            GuiState::Listening => self.render_listening(band_values, text),
            GuiState::Processing => self.render_processing(total_time),
            GuiState::Closing => self.render_closing(state_time, total_time),
        }

        self.pixmap.data()
    }

    fn render_listening(&mut self, band_values: &[f32], text: &str) {
        let mut paint = Paint::default();
        paint.anti_alias = true;

        // Draw background
        paint.set_color(self.colors.background);
        let content_path = Self::create_rounded_rect(
            0.0,
            0.0,
            self.width as f32,
            self.height as f32,
            CORNER_RADIUS,
        );
        self.pixmap.fill_path(
            &content_path,
            &paint,
            FillRule::Winding,
            Transform::identity(),
            None,
        );

        // Spectrum bars
        let total_spacing = BAR_SPACING * (BAR_COUNT - 1) as f32;
        let available_width = self.width as f32 - 20.0;
        let bar_width = ((available_width - total_spacing) / BAR_COUNT as f32) * BAR_WIDTH_FACTOR;
        let start_x =
            10.0 + (available_width - (bar_width * BAR_COUNT as f32 + total_spacing)) / 2.0;
        let center_y = SPECTRUM_HEIGHT / 2.0;
        let bar_radius = 3.0;

        paint.set_color(self.colors.bar);

        for (i, &value) in band_values.iter().take(BAR_COUNT).enumerate() {
            let bar_height = MIN_BAR_HEIGHT + value * (MAX_BAR_HEIGHT - MIN_BAR_HEIGHT);
            let x = start_x + i as f32 * (bar_width + BAR_SPACING);
            let y = center_y - bar_height / 2.0;

            let bar_path = Self::create_rounded_rect(x, y, bar_width, bar_height, bar_radius);

            self.pixmap.fill_path(
                &bar_path,
                &paint,
                FillRule::Winding,
                Transform::identity(),
                None,
            );
        }

        // Render text below spectrum
        let text_y = SPECTRUM_HEIGHT + 10.0;
        let text_height = self.height as f32 - text_y - 10.0;

        self.text_renderer.render_text(
            &mut self.pixmap,
            text,
            10.0,
            text_y,
            self.width as f32 - 20.0,
            text_height,
            self.colors.bar,
        );
    }

    fn render_processing(&mut self, animation_time: f32) {
        let mut paint = Paint::default();
        paint.set_color(self.colors.background);
        paint.anti_alias = true;

        // Small rounded box for spinner
        let dot_radius = 6.0;
        let orbit_radius = 20.0;
        let rotation_speed = ease_spinner_rotation(animation_time);

        let padding = 12.0;
        let spinner_diameter = (orbit_radius + dot_radius) * 2.0;
        let box_size = spinner_diameter + padding * 2.0;

        let box_x = (self.width as f32 - box_size) / 2.0;
        let box_y = (self.height as f32 - box_size) / 2.0;

        let path =
            Self::create_rounded_rect(box_x, box_y, box_size, box_size, CORNER_RADIUS_PROCESSING);

        self.pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);

        // Spinning dots
        paint.set_color(self.colors.bar);

        let center_x = self.width as f32 / 2.0;
        let center_y = self.height as f32 / 2.0;
        let dot_count = 3;

        for i in 0..dot_count {
            let angle = rotation_speed + (i as f32 * std::f32::consts::TAU / dot_count as f32);
            let x = center_x + orbit_radius * angle.cos();
            let y = center_y + orbit_radius * angle.sin();

            let mut pb = PathBuilder::new();
            pb.move_to(x + dot_radius, y);

            let kappa = 0.5522848;
            let kr = dot_radius * kappa;

            pb.cubic_to(x + dot_radius, y - kr, x + kr, y - dot_radius, x, y - dot_radius);
            pb.cubic_to(x - kr, y - dot_radius, x - dot_radius, y - kr, x - dot_radius, y);
            pb.cubic_to(x - dot_radius, y + kr, x - kr, y + dot_radius, x, y + dot_radius);
            pb.cubic_to(x + kr, y + dot_radius, x + dot_radius, y + kr, x + dot_radius, y);
            pb.close();

            if let Some(path) = pb.finish() {
                self.pixmap.fill_path(
                    &path,
                    &paint,
                    FillRule::Winding,
                    Transform::identity(),
                    None,
                );
            }
        }
    }

    fn render_closing(&mut self, state_elapsed: f32, total_time: f32) {
        if self.collapse_animation.is_none() {
            self.collapse_animation = Some(CollapseAnimation::new());
        }

        let anim_colors =
            animations::Colors { background: self.colors.background, bar: self.colors.bar };

        animations::render_collapse(
            &mut self.pixmap,
            anim_colors,
            state_elapsed,
            total_time,
            self.width,
            self.height,
        );
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

    pub fn calculate_text_height(&mut self, text: &str, width: f32) -> f32 {
        SPECTRUM_HEIGHT + self.text_renderer.calculate_text_height(text, width - 20.0) + 20.0
    }
}
