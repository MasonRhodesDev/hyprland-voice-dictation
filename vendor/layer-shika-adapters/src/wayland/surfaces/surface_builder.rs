use std::rc::Rc;
use std::result::Result as StdResult;
use slint::{
    platform::{set_platform, Platform, WindowAdapter},
    PhysicalSize, PlatformError,
};
use slint_interpreter::{ComponentDefinition, CompilationResult};
use smithay_client_toolkit::reexports::protocols_wlr::layer_shell::v1::client::zwlr_layer_surface_v1::ZwlrLayerSurfaceV1;
use wayland_client::{protocol::{wl_pointer::WlPointer, wl_surface::WlSurface}, Connection};
use wayland_protocols::wp::fractional_scale::v1::client::wp_fractional_scale_v1::WpFractionalScaleV1;
use wayland_protocols::wp::viewporter::client::wp_viewport::WpViewport;
use crate::errors::{LayerShikaError, Result};
use crate::rendering::femtovg::main_window::FemtoVGWindow;
use crate::rendering::slint_integration::platform::CustomSlintPlatform;

use super::surface_state::SurfaceState;

pub struct PlatformWrapper(pub Rc<CustomSlintPlatform>);

impl Platform for PlatformWrapper {
    fn create_window_adapter(&self) -> StdResult<Rc<dyn WindowAdapter>, PlatformError> {
        self.0.create_window_adapter()
    }
}

pub struct SurfaceStateBuilder {
    pub component_definition: Option<ComponentDefinition>,
    pub compilation_result: Option<Rc<CompilationResult>>,
    pub surface: Option<Rc<WlSurface>>,
    pub layer_surface: Option<Rc<ZwlrLayerSurfaceV1>>,
    pub fractional_scale: Option<Rc<WpFractionalScaleV1>>,
    pub viewport: Option<Rc<WpViewport>>,
    pub size: Option<PhysicalSize>,
    pub output_size: Option<PhysicalSize>,
    pub pointer: Option<Rc<WlPointer>>,
    pub window: Option<Rc<FemtoVGWindow>>,
    pub connection: Option<Rc<Connection>>,
    pub scale_factor: f32,
    pub height: u32,
    pub width: u32,
    pub exclusive_zone: i32,
}

impl SurfaceStateBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_surface(mut self, surface: Rc<WlSurface>) -> Self {
        self.surface = Some(surface);
        self
    }

    #[must_use]
    pub fn with_layer_surface(mut self, layer_surface: Rc<ZwlrLayerSurfaceV1>) -> Self {
        self.layer_surface = Some(layer_surface);
        self
    }

    #[must_use]
    pub const fn with_size(mut self, size: PhysicalSize) -> Self {
        self.size = Some(size);
        self
    }

    #[must_use]
    pub const fn with_output_size(mut self, output_size: PhysicalSize) -> Self {
        self.output_size = Some(output_size);
        self
    }

    #[must_use]
    pub fn with_pointer(mut self, pointer: Rc<WlPointer>) -> Self {
        self.pointer = Some(pointer);
        self
    }

    #[must_use]
    pub fn with_window(mut self, window: Rc<FemtoVGWindow>) -> Self {
        self.window = Some(window);
        self
    }

    #[must_use]
    pub const fn with_scale_factor(mut self, scale_factor: f32) -> Self {
        self.scale_factor = scale_factor;
        self
    }

    #[must_use]
    pub const fn with_height(mut self, height: u32) -> Self {
        self.height = height;
        self
    }

    #[must_use]
    pub const fn with_exclusive_zone(mut self, exclusive_zone: i32) -> Self {
        self.exclusive_zone = exclusive_zone;
        self
    }

    #[must_use]
    pub const fn with_width(mut self, width: u32) -> Self {
        self.width = width;
        self
    }

    #[must_use]
    pub fn with_component_definition(mut self, component_definition: ComponentDefinition) -> Self {
        self.component_definition = Some(component_definition);
        self
    }

    #[must_use]
    pub fn with_compilation_result(
        mut self,
        compilation_result: Option<Rc<CompilationResult>>,
    ) -> Self {
        self.compilation_result = compilation_result;
        self
    }

    #[must_use]
    pub fn with_fractional_scale(mut self, fractional_scale: Rc<WpFractionalScaleV1>) -> Self {
        self.fractional_scale = Some(fractional_scale);
        self
    }

    #[must_use]
    pub fn with_viewport(mut self, viewport: Rc<WpViewport>) -> Self {
        self.viewport = Some(viewport);
        self
    }

    #[must_use]
    pub fn with_connection(mut self, connection: Rc<Connection>) -> Self {
        self.connection = Some(connection);
        self
    }

    pub fn build(self) -> Result<(SurfaceState, Rc<CustomSlintPlatform>)> {
        let platform = CustomSlintPlatform::new(self.window.as_ref().ok_or_else(|| {
            LayerShikaError::InvalidInput {
                message: "Window is required".into(),
            }
        })?);
        set_platform(Box::new(PlatformWrapper(Rc::clone(&platform))))
            .map_err(|e| LayerShikaError::PlatformSetup { source: e })?;

        let state = SurfaceState::new(self)?;
        Ok((state, platform))
    }
}

impl Default for SurfaceStateBuilder {
    fn default() -> Self {
        Self {
            component_definition: None,
            compilation_result: None,
            surface: None,
            layer_surface: None,
            fractional_scale: None,
            viewport: None,
            size: None,
            output_size: None,
            pointer: None,
            window: None,
            connection: None,
            scale_factor: 1.0,
            height: 30,
            width: 0,
            exclusive_zone: -1,
        }
    }
}
