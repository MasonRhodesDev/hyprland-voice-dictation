use layer_shika_domain::dimensions::{LogicalSize as DomainLogicalSize, ScaleFactor};
use layer_shika_domain::surface_dimensions::SurfaceDimensions;
use layer_shika_domain::value_objects::popup_config::PopupConfig;
use log::info;
use slint::PhysicalSize;
use smithay_client_toolkit::reexports::protocols_wlr::layer_shell::v1::client::zwlr_layer_surface_v1::ZwlrLayerSurfaceV1;
use std::cell::Cell;
use std::rc::Rc;
use wayland_client::{
    protocol::{wl_compositor::WlCompositor, wl_seat::WlSeat, wl_surface::WlSurface},
    QueueHandle,
};
use wayland_protocols::wp::fractional_scale::v1::client::{
    wp_fractional_scale_manager_v1::WpFractionalScaleManagerV1,
    wp_fractional_scale_v1::WpFractionalScaleV1,
};
use wayland_protocols::wp::viewporter::client::{
    wp_viewport::WpViewport, wp_viewporter::WpViewporter,
};
use wayland_protocols::xdg::shell::client::{
    xdg_popup::XdgPopup,
    xdg_positioner::{Anchor, ConstraintAdjustment, Gravity, XdgPositioner},
    xdg_surface::XdgSurface,
    xdg_wm_base::XdgWmBase,
};

use super::app_state::AppState;

pub struct PopupSurfaceParams<'a> {
    pub compositor: &'a WlCompositor,
    pub xdg_wm_base: &'a XdgWmBase,
    pub parent_layer_surface: &'a ZwlrLayerSurfaceV1,
    pub fractional_scale_manager: Option<&'a WpFractionalScaleManagerV1>,
    pub viewporter: Option<&'a WpViewporter>,
    pub queue_handle: &'a QueueHandle<AppState>,
    pub popup_config: PopupConfig,
    pub physical_size: PhysicalSize,
    pub scale_factor: f32,
}

pub struct PopupSurface {
    pub surface: Rc<WlSurface>,
    pub xdg_surface: Rc<XdgSurface>,
    pub xdg_popup: Rc<XdgPopup>,
    pub fractional_scale: Option<Rc<WpFractionalScaleV1>>,
    pub viewport: Option<Rc<WpViewport>>,
    popup_config: PopupConfig,
    xdg_wm_base: Rc<XdgWmBase>,
    queue_handle: QueueHandle<AppState>,
    scale_factor: Cell<f32>,
}

impl PopupSurface {
    pub fn create(params: &PopupSurfaceParams<'_>) -> Self {
        let surface = Rc::new(params.compositor.create_surface(params.queue_handle, ()));

        let xdg_surface = Rc::new(params.xdg_wm_base.get_xdg_surface(
            &surface,
            params.queue_handle,
            (),
        ));

        let positioner = Self::create_positioner(params);

        let xdg_popup = Rc::new(xdg_surface.get_popup(None, &positioner, params.queue_handle, ()));

        info!("Attaching popup to layer surface via get_popup");
        params.parent_layer_surface.get_popup(&xdg_popup);

        let fractional_scale = params.fractional_scale_manager.map(|manager| {
            info!("Creating fractional scale object for popup surface");
            Rc::new(manager.get_fractional_scale(&surface, params.queue_handle, ()))
        });

        let viewport = params.viewporter.map(|vp| {
            info!("Creating viewport for popup surface");
            Rc::new(vp.get_viewport(&surface, params.queue_handle, ()))
        });

        #[allow(clippy::cast_possible_wrap)]
        #[allow(clippy::cast_precision_loss)]
        #[allow(clippy::cast_possible_truncation)]
        if let Some(ref vp) = viewport {
            let logical_width = (params.physical_size.width as f32 / params.scale_factor) as i32;
            let logical_height = (params.physical_size.height as f32 / params.scale_factor) as i32;
            info!(
                "Setting viewport destination to logical size: {}x{} (physical: {}x{}, scale: {})",
                logical_width,
                logical_height,
                params.physical_size.width,
                params.physical_size.height,
                params.scale_factor
            );
            vp.set_destination(logical_width, logical_height);
        }

        surface.set_buffer_scale(1);

        Self {
            surface,
            xdg_surface,
            xdg_popup,
            fractional_scale,
            viewport,
            popup_config: params.popup_config,
            xdg_wm_base: Rc::new(params.xdg_wm_base.clone()),
            queue_handle: params.queue_handle.clone(),
            scale_factor: Cell::new(params.scale_factor),
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_possible_wrap)]
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_precision_loss)]
    fn create_positioner(params: &PopupSurfaceParams<'_>) -> XdgPositioner {
        let positioner = params
            .xdg_wm_base
            .create_positioner(params.queue_handle, ());

        let calculated_x = params.popup_config.calculated_top_left_x() as i32;
        let calculated_y = params.popup_config.calculated_top_left_y() as i32;

        info!(
            "Popup positioning: reference=({}, {}), mode={:?}, calculated_top_left=({}, {})",
            params.popup_config.reference_x(),
            params.popup_config.reference_y(),
            params.popup_config.positioning_mode(),
            calculated_x,
            calculated_y
        );

        let logical_width = (params.physical_size.width as f32 / params.scale_factor) as i32;
        let logical_height = (params.physical_size.height as f32 / params.scale_factor) as i32;

        positioner.set_anchor_rect(calculated_x, calculated_y, 1, 1);
        positioner.set_size(logical_width, logical_height);
        positioner.set_anchor(Anchor::TopLeft);
        positioner.set_gravity(Gravity::BottomRight);
        positioner.set_constraint_adjustment(ConstraintAdjustment::None);

        positioner
    }

    pub fn grab(&self, seat: &WlSeat, serial: u32) {
        info!("Grabbing popup with serial {serial}");
        self.xdg_popup.grab(seat, serial);

        info!("Committing popup surface to trigger configure event");
        self.surface.commit();
    }

    pub fn update_viewport_size(&self, logical_width: i32, logical_height: i32) {
        if let Some(ref vp) = self.viewport {
            log::debug!(
                "Updating popup viewport destination to logical size: {}x{}",
                logical_width,
                logical_height
            );
            vp.set_destination(logical_width, logical_height);

            self.reposition_popup(logical_width, logical_height);
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_possible_wrap)]
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_precision_loss)]
    fn reposition_popup(&self, logical_width: i32, logical_height: i32) {
        let scale_factor = self.scale_factor.get();

        let updated_config = PopupConfig::new(
            self.popup_config.reference_x(),
            self.popup_config.reference_y(),
            SurfaceDimensions::from_logical(
                DomainLogicalSize::from_raw(logical_width as f32, logical_height as f32),
                ScaleFactor::from_raw(scale_factor),
            ),
            self.popup_config.positioning_mode(),
            self.popup_config.output_bounds(),
        );

        let calculated_x = updated_config.calculated_top_left_x() as i32;
        let calculated_y = updated_config.calculated_top_left_y() as i32;

        info!(
            "Repositioning popup: reference=({}, {}), new_size=({}x{}), new_top_left=({}, {})",
            self.popup_config.reference_x(),
            self.popup_config.reference_y(),
            logical_width,
            logical_height,
            calculated_x,
            calculated_y
        );

        let positioner = self.xdg_wm_base.create_positioner(&self.queue_handle, ());
        positioner.set_anchor_rect(calculated_x, calculated_y, 1, 1);
        positioner.set_size(logical_width, logical_height);
        positioner.set_anchor(Anchor::TopLeft);
        positioner.set_gravity(Gravity::BottomRight);
        positioner.set_constraint_adjustment(ConstraintAdjustment::None);

        self.xdg_popup.reposition(&positioner, 0);
    }

    pub fn destroy(&self) {
        info!("Destroying popup surface");
        self.xdg_popup.destroy();
        self.xdg_surface.destroy();
        self.surface.destroy();
    }
}
