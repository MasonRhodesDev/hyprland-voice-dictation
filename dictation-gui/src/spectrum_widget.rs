use iced::advanced::layout::{self, Layout};
use iced::advanced::renderer;
use iced::advanced::widget::{self, Widget};
use iced::advanced::{Clipboard, Shell};
use iced::mouse;
use iced::{Color, Element, Length, Rectangle, Size, Vector, event, Border};

const MIN_BAR_HEIGHT: f32 = 5.0;
const MAX_BAR_HEIGHT: f32 = 30.0;
const BAR_WIDTH_FACTOR: f32 = 0.6;
const BAR_SPACING: f32 = 8.0;
const BAR_RADIUS: f32 = 3.0;

pub struct SpectrumBars {
    values: Vec<f32>,
    height: f32,
    width: f32,
}

impl SpectrumBars {
    pub fn new(values: Vec<f32>) -> Self {
        Self {
            values,
            height: 50.0,
            width: 400.0,
        }
    }

    pub fn height(mut self, height: f32) -> Self {
        self.height = height;
        self
    }

    pub fn width(mut self, width: f32) -> Self {
        self.width = width;
        self
    }
}

impl<Message, Theme, Renderer> Widget<Message, Theme, Renderer> for SpectrumBars
where
    Renderer: renderer::Renderer,
{
    fn size(&self) -> Size<Length> {
        Size {
            width: Length::Fixed(self.width),
            height: Length::Fixed(self.height),
        }
    }

    fn layout(
        &self,
        _tree: &mut widget::Tree,
        _renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        let size = limits.resolve(
            Length::Fixed(self.width),
            Length::Fixed(self.height),
            Size::new(self.width, self.height),
        );
        layout::Node::new(size)
    }

    fn draw(
        &self,
        _tree: &widget::Tree,
        renderer: &mut Renderer,
        _theme: &Theme,
        _style: &renderer::Style,
        layout: Layout<'_>,
        _cursor: mouse::Cursor,
        _viewport: &Rectangle,
    ) {
        let bounds = layout.bounds();
        let bar_count = self.values.len();

        if bar_count == 0 {
            return;
        }

        let total_spacing = BAR_SPACING * (bar_count - 1) as f32;
        let available_width = bounds.width - 20.0;
        let bar_width = ((available_width - total_spacing) / bar_count as f32) * BAR_WIDTH_FACTOR;
        let start_x = bounds.x + 10.0 + (available_width - (bar_width * bar_count as f32 + total_spacing)) / 2.0;
        let center_y = bounds.y + bounds.height / 2.0;

        for (i, &value) in self.values.iter().enumerate() {
            let bar_height = MIN_BAR_HEIGHT + value * (MAX_BAR_HEIGHT - MIN_BAR_HEIGHT);
            let x = start_x + i as f32 * (bar_width + BAR_SPACING);
            let y = center_y - bar_height / 2.0;

            let bar_rect = Rectangle {
                x,
                y,
                width: bar_width,
                height: bar_height,
            };

            renderer.fill_quad(
                renderer::Quad {
                    bounds: bar_rect,
                    border: Border {
                        radius: BAR_RADIUS.into(),
                        width: 0.0,
                        color: Color::TRANSPARENT,
                    },
                    shadow: Default::default(),
                },
                Color::WHITE,
            );
        }
    }

    fn on_event(
        &mut self,
        _state: &mut widget::Tree,
        _event: iced::Event,
        _layout: Layout<'_>,
        _cursor: mouse::Cursor,
        _renderer: &Renderer,
        _clipboard: &mut dyn Clipboard,
        _shell: &mut Shell<'_, Message>,
        _viewport: &Rectangle,
    ) -> event::Status {
        event::Status::Ignored
    }
}

impl<'a, Message, Theme, Renderer> From<SpectrumBars> for Element<'a, Message, Theme, Renderer>
where
    Message: 'a,
    Theme: 'a,
    Renderer: renderer::Renderer + 'a,
{
    fn from(widget: SpectrumBars) -> Self {
        Self::new(widget)
    }
}
