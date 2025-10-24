use iced::widget::canvas::{self, Geometry, Path};
use iced::{Color, Point, Rectangle, Renderer, Theme};
use std::f32::consts::TAU;

const DOT_COUNT: usize = 3;
const DOT_RADIUS: f32 = 6.0;
const ORBIT_RADIUS: f32 = 20.0;
const ROTATION_SPEED: f32 = 2.0;

pub struct Spinner {
    time: f32,
}

impl Spinner {
    pub fn new(time: f32) -> Self {
        Self { time }
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

        let dot_color = Color::from_rgb(1.0, 1.0, 1.0);

        for i in 0..DOT_COUNT {
            let angle = (self.time * ROTATION_SPEED) + (i as f32 * TAU / DOT_COUNT as f32);
            let x = center_x + ORBIT_RADIUS * angle.cos();
            let y = center_y + ORBIT_RADIUS * angle.sin();

            let circle = Path::circle(Point::new(x, y), DOT_RADIUS);
            frame.fill(&circle, dot_color);
        }

        vec![frame.into_geometry()]
    }
}
