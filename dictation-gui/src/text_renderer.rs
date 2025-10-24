// Simplified text renderer using fontdue (keeping it simple for now)
// TODO: Upgrade to cosmic-text 0.13+ when available for better text layout

use tiny_skia::*;

const TEXT_MAX_LINES: usize = 2;
const TEXT_FONT_SIZE: f32 = 18.0;
const TEXT_LINE_HEIGHT: f32 = 30.0;

pub struct TextRenderer {
    font: fontdue::Font,
}

impl TextRenderer {
    pub fn new() -> Self {
        let font_data = include_bytes!("/usr/share/fonts/google-carlito-fonts/Carlito-Regular.ttf");
        let font = fontdue::Font::from_bytes(font_data as &[u8], fontdue::FontSettings::default())
            .expect("Failed to load font");
        
        Self { font }
    }
    
    pub fn render_text(
        &mut self,
        pixmap: &mut Pixmap,
        text: &str,
        x: f32,
        y: f32,
        width: f32,
        _height: f32,
        color: Color,
    ) {
        if text.is_empty() {
            return;
        }
        
        use fontdue::layout::{CoordinateSystem, HorizontalAlign, Layout, LayoutSettings, TextStyle};
        
        let mut layout = Layout::new(CoordinateSystem::PositiveYDown);
        layout.reset(&LayoutSettings {
            x: 0.0,
            y: 0.0,
            max_width: Some(width),
            max_height: None,
            wrap_style: fontdue::layout::WrapStyle::Word,
            wrap_hard_breaks: true,
            horizontal_align: HorizontalAlign::Center,
            ..Default::default()
        });
        layout.append(&[&self.font], &TextStyle::new(text, TEXT_FONT_SIZE, 0));
        
        let glyphs = layout.glyphs();
        if glyphs.is_empty() {
            return;
        }
        
        // Get lines
        let lines = self.get_lines(&glyphs);
        
        // Show only last N lines
        let visible_lines = if lines.len() > TEXT_MAX_LINES {
            &lines[lines.len() - TEXT_MAX_LINES..]
        } else {
            &lines
        };
        
        let scroll_offset = if !visible_lines.is_empty() {
            visible_lines[0].min_y
        } else {
            0.0
        };
        
        // Render visible glyphs
        for glyph in glyphs {
            let in_visible_range = visible_lines.iter().any(|line| {
                (glyph.y - line.min_y).abs() <= 5.0
            });
            
            if !in_visible_range {
                continue;
            }
            
            let (metrics, bitmap) = self.font.rasterize_config(glyph.key);
            
            let final_x = glyph.x + x;
            let final_y = glyph.y + y - scroll_offset;
            
            let glyph_x = final_x as i32;
            let glyph_y = final_y as i32;
            
            for dy in 0..metrics.height {
                for dx in 0..metrics.width {
                    let px = glyph_x + dx as i32;
                    let py = glyph_y + dy as i32;
                    
                    if px >= 0 && px < pixmap.width() as i32 && py >= 0 && py < pixmap.height() as i32 {
                        let alpha = bitmap[dy * metrics.width + dx] as f32 / 255.0;
                        if alpha > 0.0 {
                            self.blend_pixel(pixmap, px as u32, py as u32, color, alpha);
                        }
                    }
                }
            }
        }
    }
    
    fn blend_pixel(&self, pixmap: &mut Pixmap, x: u32, y: u32, color: Color, alpha: f32) {
        let offset = (y * pixmap.width() + x) * 4;
        if offset + 3 < pixmap.data().len() as u32 {
            let data = pixmap.data_mut();
            let bg_r = data[offset as usize] as f32 / 255.0;
            let bg_g = data[offset as usize + 1] as f32 / 255.0;
            let bg_b = data[offset as usize + 2] as f32 / 255.0;
            let bg_a = data[offset as usize + 3] as f32 / 255.0;
            
            let fg_r = color.red();
            let fg_g = color.green();
            let fg_b = color.blue();
            
            let out_a = alpha + bg_a * (1.0 - alpha);
            let out_r = (fg_r * alpha + bg_r * bg_a * (1.0 - alpha)) / out_a.max(0.001);
            let out_g = (fg_g * alpha + bg_g * bg_a * (1.0 - alpha)) / out_a.max(0.001);
            let out_b = (fg_b * alpha + bg_b * bg_a * (1.0 - alpha)) / out_a.max(0.001);
            
            data[offset as usize] = (out_r * 255.0) as u8;
            data[offset as usize + 1] = (out_g * 255.0) as u8;
            data[offset as usize + 2] = (out_b * 255.0) as u8;
            data[offset as usize + 3] = (out_a * 255.0) as u8;
        }
    }
    
    fn get_lines(&self, glyphs: &[fontdue::layout::GlyphPosition]) -> Vec<LineInfo> {
        if glyphs.is_empty() {
            return vec![];
        }
        
        let mut lines = vec![];
        let mut current_line = LineInfo { min_y: glyphs[0].y, max_y: glyphs[0].y };
        
        for glyph in glyphs.iter().skip(1) {
            if (glyph.y - current_line.min_y).abs() > 5.0 {
                lines.push(current_line);
                current_line = LineInfo { min_y: glyph.y, max_y: glyph.y };
            } else {
                current_line.min_y = current_line.min_y.min(glyph.y);
                current_line.max_y = current_line.max_y.max(glyph.y);
            }
        }
        lines.push(current_line);
        
        lines
    }
    
    pub fn calculate_text_height(&mut self, text: &str, width: f32) -> f32 {
        if text.is_empty() {
            return TEXT_LINE_HEIGHT;
        }
        
        use fontdue::layout::{CoordinateSystem, HorizontalAlign, Layout, LayoutSettings, TextStyle};
        
        let mut layout = Layout::new(CoordinateSystem::PositiveYDown);
        layout.reset(&LayoutSettings {
            x: 0.0,
            y: 0.0,
            max_width: Some(width),
            max_height: None,
            wrap_style: fontdue::layout::WrapStyle::Word,
            wrap_hard_breaks: true,
            horizontal_align: HorizontalAlign::Center,
            ..Default::default()
        });
        layout.append(&[&self.font], &TextStyle::new(text, TEXT_FONT_SIZE, 0));
        
        let glyphs = layout.glyphs();
        if glyphs.is_empty() {
            return TEXT_LINE_HEIGHT;
        }
        
        let line_count = self.count_lines(&glyphs).min(TEXT_MAX_LINES).max(1);
        line_count as f32 * TEXT_LINE_HEIGHT
    }
    
    fn count_lines(&self, glyphs: &[fontdue::layout::GlyphPosition]) -> usize {
        if glyphs.is_empty() {
            return 0;
        }
        
        let mut lines = 1;
        let mut last_y = glyphs[0].y;
        
        for glyph in glyphs.iter().skip(1) {
            if (glyph.y - last_y).abs() > 5.0 {
                lines += 1;
                last_y = glyph.y;
            }
        }
        
        lines
    }
}

struct LineInfo {
    min_y: f32,
    max_y: f32,
}
