use iced::widget::canvas::{self, Geometry, Path};
use iced::{Color, Point, Rectangle, Renderer, Theme};
use std::f32::consts::TAU;

pub struct Spinner {
    time: f32,
    dot_count: u32,
    dot_radius: f32,
    orbit_radius: f32,
    rotation_speed: f32,
    opacity: f32,
}

impl Spinner {
    pub fn new(time: f32, dot_count: u32, dot_radius: f32, orbit_radius: f32, rotation_speed: f32, opacity: f32) -> Self {
        Self { time, dot_count, dot_radius, orbit_radius, rotation_speed, opacity }
    }
}

impl<Message> canvas::Program<Message> for Spinner {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: iced::mouse::Cursor,
    ) -> Vec<Geometry> {
        let mut frame = canvas::Frame::new(renderer, bounds.size());

        let center_x = bounds.width / 2.0;
        let center_y = bounds.height / 2.0;

        let dot_color = Color { r: 1.0, g: 1.0, b: 1.0, a: self.opacity };

        for i in 0..self.dot_count {
            let angle = (self.time * self.rotation_speed) + (i as f32 * TAU / self.dot_count as f32);
            let x = center_x + self.orbit_radius * angle.cos();
            let y = center_y + self.orbit_radius * angle.sin();

            let circle = Path::circle(Point::new(x, y), self.dot_radius);
            frame.fill(&circle, dot_color);
        }

        vec![frame.into_geometry()]
    }
}
