use iced::widget::canvas::{self, Geometry, Path};
use iced::{Color, Point, Rectangle, Renderer, Theme};
use std::f32::consts::TAU;

const DOT_COUNT: usize = 3;
const DOT_RADIUS: f32 = 6.0;
const INITIAL_ORBIT_RADIUS: f32 = 20.0;

pub struct CollapsingDots {
    progress: f32,
    time: f32,
}

impl CollapsingDots {
    pub fn new(progress: f32, time: f32) -> Self {
        Self { progress, time }
    }
}

impl<Message> canvas::Program<Message> for CollapsingDots {
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

        let orbit_radius = INITIAL_ORBIT_RADIUS * (1.0 - self.progress);
        let alpha = 1.0 - self.progress;
        let dot_color = Color::from_rgba(1.0, 1.0, 1.0, alpha);

        const ROTATION_SPEED: f32 = 2.0;

        for i in 0..DOT_COUNT {
            let angle = (self.time * ROTATION_SPEED) + (i as f32 * TAU / DOT_COUNT as f32);
            let x = center_x + orbit_radius * angle.cos();
            let y = center_y + orbit_radius * angle.sin();

            let circle = Path::circle(Point::new(x, y), DOT_RADIUS);
            frame.fill(&circle, dot_color);
        }

        vec![frame.into_geometry()]
    }
}
