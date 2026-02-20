use layer_shika_domain::config::SurfaceConfig as DomainSurfaceConfig;
use layer_shika_domain::value_objects::anchor::AnchorEdges;
use layer_shika_domain::value_objects::handle::SurfaceHandle;
use layer_shika_domain::value_objects::keyboard_interactivity::KeyboardInteractivity as DomainKeyboardInteractivity;
use layer_shika_domain::value_objects::layer::Layer;
use layer_shika_domain::value_objects::margins::Margins;
use layer_shika_domain::value_objects::output_policy::OutputPolicy;
use slint_interpreter::{CompilationResult, ComponentDefinition};
use smithay_client_toolkit::reexports::protocols_wlr::layer_shell::v1::client::{
    zwlr_layer_shell_v1::{self},
    zwlr_layer_surface_v1::{Anchor, KeyboardInteractivity as WaylandKeyboardInteractivity},
};
use std::rc::Rc;

#[derive(Debug, Clone, Copy)]
pub(crate) struct LayerSurfaceConfig {
    pub anchor: Anchor,
    pub margin: Margins,
    pub exclusive_zone: i32,
    pub keyboard_interactivity: WaylandKeyboardInteractivity,
    pub height: u32,
    pub width: u32,
}

#[derive(Clone)]
pub struct WaylandSurfaceConfig {
    pub surface_handle: SurfaceHandle,
    pub surface_name: String,
    pub height: u32,
    pub width: u32,
    pub layer: zwlr_layer_shell_v1::Layer,
    pub margin: Margins,
    pub anchor: Anchor,
    pub keyboard_interactivity: WaylandKeyboardInteractivity,
    pub exclusive_zone: i32,
    pub scale_factor: f32,
    pub namespace: String,
    pub component_definition: ComponentDefinition,
    pub compilation_result: Option<Rc<CompilationResult>>,
    pub output_policy: OutputPolicy,
}

impl WaylandSurfaceConfig {
    #[must_use]
    pub fn from_domain_config(
        surface_handle: SurfaceHandle,
        surface_name: impl Into<String>,
        component_definition: ComponentDefinition,
        compilation_result: Option<Rc<CompilationResult>>,
        domain_config: DomainSurfaceConfig,
    ) -> Self {
        Self {
            surface_handle,
            surface_name: surface_name.into(),
            height: domain_config.dimensions.height(),
            width: domain_config.dimensions.width(),
            layer: convert_layer(domain_config.layer),
            margin: domain_config.margin,
            anchor: convert_anchor(domain_config.anchor),
            keyboard_interactivity: convert_keyboard_interactivity(
                domain_config.keyboard_interactivity,
            ),
            exclusive_zone: domain_config.exclusive_zone,
            scale_factor: domain_config.scale_factor.value(),
            namespace: domain_config.namespace,
            component_definition,
            compilation_result,
            output_policy: domain_config.output_policy,
        }
    }
}

const fn convert_layer(layer: Layer) -> zwlr_layer_shell_v1::Layer {
    match layer {
        Layer::Background => zwlr_layer_shell_v1::Layer::Background,
        Layer::Bottom => zwlr_layer_shell_v1::Layer::Bottom,
        Layer::Top => zwlr_layer_shell_v1::Layer::Top,
        Layer::Overlay => zwlr_layer_shell_v1::Layer::Overlay,
    }
}

const fn convert_anchor(anchor: AnchorEdges) -> Anchor {
    let mut result = Anchor::empty();

    if anchor.has_top() {
        result = result.union(Anchor::Top);
    }
    if anchor.has_bottom() {
        result = result.union(Anchor::Bottom);
    }
    if anchor.has_left() {
        result = result.union(Anchor::Left);
    }
    if anchor.has_right() {
        result = result.union(Anchor::Right);
    }

    result
}

const fn convert_keyboard_interactivity(
    mode: DomainKeyboardInteractivity,
) -> WaylandKeyboardInteractivity {
    match mode {
        DomainKeyboardInteractivity::None => WaylandKeyboardInteractivity::None,
        DomainKeyboardInteractivity::Exclusive => WaylandKeyboardInteractivity::Exclusive,
        DomainKeyboardInteractivity::OnDemand => WaylandKeyboardInteractivity::OnDemand,
    }
}

#[derive(Clone)]
pub struct ShellSurfaceConfig {
    pub name: String,
    pub config: WaylandSurfaceConfig,
}

#[derive(Clone)]
pub struct MultiSurfaceConfig {
    pub surfaces: Vec<ShellSurfaceConfig>,
    pub compilation_result: Rc<CompilationResult>,
}

impl MultiSurfaceConfig {
    pub fn new(compilation_result: Rc<CompilationResult>) -> Self {
        Self {
            surfaces: Vec::new(),
            compilation_result,
        }
    }

    #[must_use]
    pub fn add_surface(mut self, name: impl Into<String>, config: WaylandSurfaceConfig) -> Self {
        self.surfaces.push(ShellSurfaceConfig {
            name: name.into(),
            config,
        });
        self
    }

    pub fn primary_config(&self) -> Option<&WaylandSurfaceConfig> {
        self.surfaces.first().map(|s| &s.config)
    }
}
