use crate::animations::{self, ClosingAnimation};
use crate::GuiState;
use anyhow::Result;
use tiny_skia::*;

const BAR_COUNT: usize = 8;
const MIN_BAR_HEIGHT: f32 = 5.0;
const MAX_BAR_HEIGHT: f32 = 30.0;
const CORNER_RADIUS: f32 = 25.0;
const CORNER_RADIUS_PROCESSING: f32 = 50.0;
const BAR_WIDTH_FACTOR: f32 = 0.6;
const BAR_SPACING: f32 = 8.0;

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

pub struct SpectrumRenderer {
    width: u32,
    height: u32,
    pixmap: Pixmap,
    colors: Colors,
    closing_animation: ClosingAnimation,
}

pub fn calculate_text_height(text: &str, width: u32) -> u32 {
    if text.is_empty() {
        return 55;
    }

    use fontdue::layout::{CoordinateSystem, HorizontalAlign, Layout, LayoutSettings, TextStyle};
    use fontdue::Font;

    let font_data = include_bytes!("/usr/share/fonts/google-carlito-fonts/Carlito-Regular.ttf");
    if let Ok(font) = Font::from_bytes(font_data as &[u8], fontdue::FontSettings::default()) {
        let mut layout = Layout::new(CoordinateSystem::PositiveYDown);
        layout.reset(&LayoutSettings {
            x: 10.0,
            y: 0.0,
            max_width: Some(width as f32 - 20.0),
            max_height: None,
            wrap_style: fontdue::layout::WrapStyle::Word,
            wrap_hard_breaks: true,
            horizontal_align: HorizontalAlign::Center,
            ..Default::default()
        });
        layout.append(&[&font], &TextStyle::new(text, 18.0, 0));

        let glyphs = layout.glyphs();
        if let Some(last_glyph) = glyphs.last() {
            let text_height = last_glyph.y as u32 + 25;
            return 50 + text_height + 20;
        }
    }

    70
}

impl SpectrumRenderer {
    pub fn new(width: u32, height: u32) -> Result<Self> {
        let pixmap = Pixmap::new(width, height).expect("Failed to create pixmap");
        let colors = Self::load_colors();

        Ok(Self { width, height, pixmap, colors, closing_animation: ClosingAnimation::Collapse })
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
            GuiState::Listening => self.render_listening(band_values, text),
            GuiState::Processing => self.render_processing(text, total_time),
            GuiState::Closing => self.render_closing(text, state_time, total_time),
        }

        self.pixmap.data()
    }

    fn render_listening(&mut self, band_values: &[f32], text: &str) {
        let mut paint = Paint::default();
        paint.set_color(self.colors.background);
        paint.anti_alias = true;

        // Background
        let path = Self::create_rounded_rect(
            0.0,
            0.0,
            self.width as f32,
            self.height as f32,
            CORNER_RADIUS,
        );

        self.pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);

        // Spectrum bars (top section) - narrower with more spacing
        let spectrum_height = 50.0;
        let total_spacing = BAR_SPACING * (BAR_COUNT - 1) as f32;
        let available_width = self.width as f32 - 40.0;
        let bar_width = ((available_width - total_spacing) / BAR_COUNT as f32) * BAR_WIDTH_FACTOR;
        let start_x =
            20.0 + (available_width - (bar_width * BAR_COUNT as f32 + total_spacing)) / 2.0;
        let center_y = spectrum_height / 2.0;
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
        self.render_text(text, 55.0);
    }

    fn render_processing(&mut self, _text: &str, animation_time: f32) {
        let mut paint = Paint::default();
        paint.set_color(self.colors.background);
        paint.anti_alias = true;

        // Small rounded box just for spinner
        let dot_count = 3;
        let dot_radius = 6.0;
        let orbit_radius = 20.0;
        let rotation_speed = 2.0;

        let padding = 12.0;
        let spinner_diameter = (orbit_radius + dot_radius) * 2.0;
        let box_size = spinner_diameter + padding * 2.0;

        let box_x = (self.width as f32 - box_size) / 2.0;
        let box_y = (self.height as f32 - box_size) / 2.0;

        let path =
            Self::create_rounded_rect(box_x, box_y, box_size, box_size, CORNER_RADIUS_PROCESSING);

        self.pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);

        // Spinning dots centered
        paint.set_color(self.colors.bar);

        let center_x = self.width as f32 / 2.0;
        let center_y = self.height as f32 / 2.0;

        for i in 0..dot_count {
            let angle = (animation_time * rotation_speed)
                + (i as f32 * std::f32::consts::TAU / dot_count as f32);
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

    fn render_closing(&mut self, _text: &str, state_elapsed: f32, total_time: f32) {
        let anim_colors =
            animations::Colors { background: self.colors.background, bar: self.colors.bar };

        match self.closing_animation {
            ClosingAnimation::Collapse => {
                animations::render_collapse(
                    &mut self.pixmap,
                    anim_colors,
                    state_elapsed,
                    total_time,
                    self.width,
                    self.height,
                );
            }
        }
    }

    fn render_text(&mut self, text: &str, y_start: f32) {
        if text.is_empty() {
            return;
        }

        use fontdue::layout::{
            CoordinateSystem, HorizontalAlign, Layout, LayoutSettings, TextStyle,
        };
        use fontdue::Font;

        let font_data = include_bytes!("/usr/share/fonts/google-carlito-fonts/Carlito-Regular.ttf");
        if let Ok(font) = Font::from_bytes(font_data as &[u8], fontdue::FontSettings::default()) {
            let available_height = self.height as f32 - y_start - 10.0;

            // Layout with CENTER alignment built-in
            let mut layout = Layout::new(CoordinateSystem::PositiveYDown);
            layout.reset(&LayoutSettings {
                x: 0.0,
                y: 0.0,
                max_width: Some(self.width as f32 - 40.0),
                max_height: None,
                wrap_style: fontdue::layout::WrapStyle::Word,
                wrap_hard_breaks: true,
                horizontal_align: HorizontalAlign::Center,
                ..Default::default()
            });
            layout.append(&[&font], &TextStyle::new(text, 18.0, 0));

            let glyphs = layout.glyphs();
            if glyphs.is_empty() {
                return;
            }

            // Calculate scroll offset
            let max_y =
                glyphs.iter().map(|g| g.y).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0);
            let total_text_height = max_y + 25.0;
            let scroll_offset = if total_text_height > available_height {
                total_text_height - available_height
            } else {
                0.0
            };

            // Render glyphs (already centered by fontdue)
            for glyph in glyphs {
                let (metrics, bitmap) = font.rasterize_config(glyph.key);

                // Add left margin to center in window
                let final_x = glyph.x + 20.0;
                let final_y = glyph.y + y_start - scroll_offset;

                let glyph_x = final_x as i32;
                let glyph_y = final_y as i32;

                for y in 0..metrics.height {
                    for x in 0..metrics.width {
                        let px = glyph_x + x as i32;
                        let py = glyph_y + y as i32;

                        if px >= 0 && px < self.width as i32 && py >= 0 && py < self.height as i32 {
                            let alpha = bitmap[y * metrics.width + x] as f32 / 255.0;
                            if alpha > 0.0 {
                                let offset = (py as u32 * self.width + px as u32) * 4;
                                if offset + 3 < self.pixmap.data().len() as u32 {
                                    let data = self.pixmap.data_mut();
                                    let bg_r = data[offset as usize] as f32 / 255.0;
                                    let bg_g = data[offset as usize + 1] as f32 / 255.0;
                                    let bg_b = data[offset as usize + 2] as f32 / 255.0;
                                    let bg_a = data[offset as usize + 3] as f32 / 255.0;

                                    let out_a = alpha + bg_a * (1.0 - alpha);
                                    let out_r = (1.0 * alpha + bg_r * bg_a * (1.0 - alpha))
                                        / out_a.max(0.001);
                                    let out_g = (1.0 * alpha + bg_g * bg_a * (1.0 - alpha))
                                        / out_a.max(0.001);
                                    let out_b = (1.0 * alpha + bg_b * bg_a * (1.0 - alpha))
                                        / out_a.max(0.001);

                                    data[offset as usize] = (out_r * 255.0) as u8;
                                    data[offset as usize + 1] = (out_g * 255.0) as u8;
                                    data[offset as usize + 2] = (out_b * 255.0) as u8;
                                    data[offset as usize + 3] = (out_a * 255.0) as u8;
                                }
                            }
                        }
                    }
                }
            }
        }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_colors_default() {
        let colors = Colors::default();
        assert_eq!(colors.background.red(), 0.0);
        assert_eq!(colors.background.green(), 0.0);
        assert_eq!(colors.background.blue(), 0.0);
        assert!((colors.background.alpha() * 255.0 - 230.0).abs() < 1.0);
    }

    #[test]
    fn test_parse_color_value_valid() {
        let line = "@define-color primary #FF5733;";
        let color = SpectrumRenderer::parse_color_value(line);
        assert!(color.is_some());
        if let Some(c) = color {
            assert!((c.red() - 1.0).abs() < 0.01);
            assert!((c.green() - 87.0 / 255.0).abs() < 0.01);
            assert!((c.blue() - 51.0 / 255.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_parse_color_value_invalid() {
        let line = "@define-color primary #ZZ5733;";
        let color = SpectrumRenderer::parse_color_value(line);
        assert!(color.is_none());
    }

    #[test]
    fn test_parse_color_value_short() {
        let line = "@define-color primary #FFF;";
        let color = SpectrumRenderer::parse_color_value(line);
        assert!(color.is_none());
    }

    #[test]
    fn test_renderer_new() {
        let result = SpectrumRenderer::new(400, 150);
        assert!(result.is_ok());

        if let Ok(renderer) = result {
            assert_eq!(renderer.width, 400);
            assert_eq!(renderer.height, 150);
        }
    }

    #[test]
    fn test_renderer_render_empty() {
        let mut renderer = SpectrumRenderer::new(400, 150).unwrap();
        let bands = vec![0.0f32; 8];
        let pixels = renderer.render(&bands, "", GuiState::Listening, 0.0, 0.0);

        assert_eq!(pixels.len(), (400 * 150 * 4) as usize);
    }

    #[test]
    fn test_renderer_render_with_bands() {
        let mut renderer = SpectrumRenderer::new(400, 150).unwrap();
        let bands = vec![0.5f32; 8];
        let pixels = renderer.render(&bands, "", GuiState::Listening, 0.0, 0.0);

        assert_eq!(pixels.len(), (400 * 150 * 4) as usize);
    }

    #[test]
    fn test_renderer_render_with_text() {
        let mut renderer = SpectrumRenderer::new(400, 150).unwrap();
        let bands = vec![0.0f32; 8];
        let pixels = renderer.render(&bands, "Hello World", GuiState::Listening, 0.0, 0.0);

        assert_eq!(pixels.len(), (400 * 150 * 4) as usize);
    }

    #[test]
    fn test_create_rounded_rect() {
        let path = SpectrumRenderer::create_rounded_rect(0.0, 0.0, 100.0, 50.0, 10.0);
        let bounds = path.bounds();
        assert!(bounds.width() > 0.0);
    }

    #[test]
    fn test_create_rounded_rect_zero_radius() {
        let path = SpectrumRenderer::create_rounded_rect(0.0, 0.0, 100.0, 50.0, 0.0);
        let bounds = path.bounds();
        assert!(bounds.width() > 0.0);
    }

    #[test]
    fn test_create_rounded_rect_large_radius() {
        let path = SpectrumRenderer::create_rounded_rect(0.0, 0.0, 100.0, 50.0, 100.0);
        let bounds = path.bounds();
        assert!(bounds.width() > 0.0);
    }

    #[test]
    fn test_parse_colors_empty() {
        let css = "";
        let colors = SpectrumRenderer::parse_colors(css);
        assert!((colors.background.alpha() * 255.0 - 230.0).abs() < 1.0);
    }

    #[test]
    fn test_parse_colors_valid() {
        let css = r#"
            @define-color surface #1e1e1e;
            @define-color primary #ff6b35;
        "#;
        let colors = SpectrumRenderer::parse_colors(css);
        assert!((colors.background.red() - 30.0 / 255.0).abs() < 0.01);
        assert!((colors.bar.red() - 1.0).abs() < 0.01);
    }
}
