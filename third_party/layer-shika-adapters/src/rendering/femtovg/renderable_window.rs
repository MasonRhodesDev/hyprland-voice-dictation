use crate::errors::Result;
use slint::{
    PhysicalSize, WindowSize,
    platform::{WindowAdapter, WindowEvent},
};
use std::cell::Cell;

pub enum RenderState {
    Clean,
    Dirty,
}

pub trait RenderableWindow: WindowAdapter {
    fn render_frame_if_dirty(&self) -> Result<()>;
    fn set_scale_factor(&self, scale_factor: f32);
    fn scale_factor(&self) -> f32;
    fn render_state(&self) -> &Cell<RenderState>;
    fn size_cell(&self) -> &Cell<PhysicalSize>;

    fn request_redraw(&self) {
        self.render_state().set(RenderState::Dirty);
    }

    fn size_impl(&self) -> PhysicalSize {
        self.size_cell().get()
    }

    fn set_size_impl(&self, size: WindowSize) {
        self.size_cell().set(size.to_physical(self.scale_factor()));
        self.window().dispatch_event(WindowEvent::Resized {
            size: size.to_logical(self.scale_factor()),
        });
    }
}
