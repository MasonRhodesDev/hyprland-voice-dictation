use crate::wayland::surfaces::app_state::AppState;
use crate::wayland::surfaces::display_metrics::DisplayMetrics;
use crate::wayland::surfaces::surface_state::SurfaceState;
use layer_shika_domain::value_objects::output_info::OutputGeometry;
use log::{debug, info};
use smithay_client_toolkit::reexports::protocols_wlr::layer_shell::v1::client::{
    zwlr_layer_shell_v1::ZwlrLayerShellV1,
    zwlr_layer_surface_v1::{self, ZwlrLayerSurfaceV1},
};
use wayland_client::{
    Connection, Dispatch, Proxy, QueueHandle, WEnum,
    globals::GlobalListContents,
    protocol::{
        wl_compositor::WlCompositor,
        wl_output::{self, WlOutput},
        wl_pointer::{self, WlPointer},
        wl_registry::Event,
        wl_registry::WlRegistry,
        wl_region::WlRegion,
        wl_seat::WlSeat,
        wl_surface::WlSurface,
    },
};
use wayland_protocols::wp::fractional_scale::v1::client::{
    wp_fractional_scale_manager_v1::WpFractionalScaleManagerV1,
    wp_fractional_scale_v1::{self, WpFractionalScaleV1},
};
use wayland_protocols::wp::viewporter::client::{
    wp_viewport::WpViewport, wp_viewporter::WpViewporter,
};
use wayland_protocols::xdg::shell::client::{
    xdg_popup::{self, XdgPopup},
    xdg_positioner::XdgPositioner,
    xdg_surface::{self, XdgSurface},
    xdg_wm_base::{self, XdgWmBase},
};

impl Dispatch<ZwlrLayerSurfaceV1, ()> for AppState {
    fn event(
        state: &mut Self,
        layer_surface: &ZwlrLayerSurfaceV1,
        event: zwlr_layer_surface_v1::Event,
        _data: &(),
        _conn: &Connection,
        _queue_handle: &QueueHandle<Self>,
    ) {
        match event {
            zwlr_layer_surface_v1::Event::Configure {
                serial,
                width,
                height,
            } => {
                info!(
                    "Layer surface configured with compositor size: {}x{}",
                    width, height
                );
                let layer_surface_id = layer_surface.id();
                let Some(surface) = state.get_output_by_layer_surface_mut(&layer_surface_id) else {
                    info!(
                        "Could not find window for layer surface {:?}",
                        layer_surface_id
                    );
                    return;
                };

                surface.handle_layer_surface_configure(layer_surface, serial, width, height);
            }
            zwlr_layer_surface_v1::Event::Closed => {
                let layer_surface_id = layer_surface.id();
                if let Some(surface) = state.get_output_by_layer_surface_mut(&layer_surface_id) {
                    surface.handle_layer_surface_closed();
                }
            }
            _ => {}
        }
    }
}

impl Dispatch<WlOutput, ()> for AppState {
    #[allow(clippy::cognitive_complexity)]
    fn event(
        state: &mut Self,
        proxy: &WlOutput,
        event: <WlOutput as Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        qhandle: &QueueHandle<Self>,
    ) {
        let output_id = proxy.id();
        let handle = state.get_handle_by_output_id(&output_id);

        match event {
            wl_output::Event::Mode {
                flags: WEnum::Value(mode_flags),
                width,
                height,
                ..
            } => {
                let is_current = mode_flags.contains(wl_output::Mode::Current);
                let is_preferred = mode_flags.contains(wl_output::Mode::Preferred);
                info!(
                    "WlOutput mode: {}x{} (current: {}, preferred: {})",
                    width, height, is_current, is_preferred
                );
                if is_current {
                    for surface in state.all_surfaces_for_output_mut(&output_id) {
                        surface.handle_output_mode(width, height);
                    }
                }
            }
            wl_output::Event::Mode { .. } => {
                debug!("WlOutput mode event with unknown flags value");
            }
            wl_output::Event::Description { ref description } => {
                info!("WlOutput description: {description:?}");
                if let Some(handle) = handle {
                    if let Some(info) = state.get_output_info_mut(handle) {
                        info.set_description(description.clone());
                    }
                }
            }
            wl_output::Event::Scale { ref factor } => {
                info!("WlOutput factor scale: {factor:?}");
                if let Some(handle) = handle {
                    if let Some(info) = state.get_output_info_mut(handle) {
                        info.set_scale(*factor);
                    }
                }
            }
            wl_output::Event::Name { ref name } => {
                info!("WlOutput name: {name:?}");
                if let Some(handle) = handle {
                    if let Some(info) = state.get_output_info_mut(handle) {
                        info.set_name(name.clone());
                    }
                }
            }
            wl_output::Event::Geometry {
                x,
                y,
                physical_width,
                physical_height,
                subpixel,
                make,
                model,
                transform,
            } => {
                info!(
                    "WlOutput geometry: x={x}, y={y}, physical_width={physical_width}, physical_height={physical_height}, subpixel={subpixel:?}, make={make:?}, model={model:?}, transform={transform:?}"
                );
                if let Some(handle) = handle {
                    if let Some(info) = state.get_output_info_mut(handle) {
                        let mut geometry =
                            OutputGeometry::new(x, y, physical_width, physical_height);
                        if !make.is_empty() {
                            geometry = geometry.with_make(make);
                        }
                        if !model.is_empty() {
                            geometry = geometry.with_model(model);
                        }
                        info.set_geometry(geometry);
                    }
                }
            }
            wl_output::Event::Done => {
                info!("WlOutput done for output {:?}", output_id);

                if let Some(manager) = state.output_manager() {
                    let manager_ref = manager.borrow();
                    if manager_ref.has_pending_output(&output_id) {
                        drop(manager_ref);

                        info!(
                            "Output {:?} configuration complete, finalizing...",
                            output_id
                        );

                        let manager_ref = manager.borrow();
                        if let Err(e) = manager_ref.finalize_output(&output_id, state, qhandle) {
                            info!("Failed to finalize output {:?}: {e}", output_id);
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

impl Dispatch<WlPointer, ()> for AppState {
    fn event(
        state: &mut Self,
        _proxy: &WlPointer,
        event: <WlPointer as Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        match event {
            wl_pointer::Event::Enter {
                serial,
                surface,
                surface_x,
                surface_y,
            } => {
                let surface_id = surface.id();

                if let Some(key) = state.get_key_by_surface(&surface_id).cloned() {
                    if let Some(layer_surface) = state.get_surface_by_key_mut(&key) {
                        layer_surface.handle_pointer_enter(serial, &surface, surface_x, surface_y);
                    }
                    state.set_active_surface_key(Some(key));
                } else if let Some(key) = state.get_key_by_popup(&surface_id).cloned() {
                    if let Some(layer_surface) = state.get_surface_by_key_mut(&key) {
                        layer_surface.handle_pointer_enter(serial, &surface, surface_x, surface_y);
                    }
                    state.set_active_surface_key(Some(key));
                }
            }

            wl_pointer::Event::Motion {
                surface_x,
                surface_y,
                ..
            } => {
                if let Some(surface) = state.active_surface_mut() {
                    surface.handle_pointer_motion(surface_x, surface_y);
                }
            }

            wl_pointer::Event::Leave { .. } => {
                if let Some(surface) = state.active_surface_mut() {
                    surface.handle_pointer_leave();
                }
                state.set_active_surface_key(None);
            }

            wl_pointer::Event::Button {
                serial,
                state: button_state,
                ..
            } => {
                if let Some(surface) = state.active_surface_mut() {
                    surface.handle_pointer_button(serial, button_state);
                }
            }
            _ => {}
        }
    }
}

impl Dispatch<WpFractionalScaleV1, ()> for AppState {
    fn event(
        state: &mut Self,
        proxy: &WpFractionalScaleV1,
        event: wp_fractional_scale_v1::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        if let wp_fractional_scale_v1::Event::PreferredScale { scale } = event {
            let scale_float = DisplayMetrics::scale_factor_from_120ths(scale);
            info!("Fractional scale received: {scale_float} ({scale}x)");

            for surface in state.all_outputs_mut() {
                surface.handle_fractional_scale(proxy, scale);
            }
        }
    }
}

impl Dispatch<XdgWmBase, ()> for AppState {
    fn event(
        _state: &mut Self,
        xdg_wm_base: &XdgWmBase,
        event: xdg_wm_base::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        if let xdg_wm_base::Event::Ping { serial } = event {
            SurfaceState::handle_xdg_wm_base_ping(xdg_wm_base, serial);
        }
    }
}

impl Dispatch<XdgPopup, ()> for AppState {
    fn event(
        state: &mut Self,
        xdg_popup: &XdgPopup,
        event: xdg_popup::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        match event {
            xdg_popup::Event::Configure {
                x,
                y,
                width,
                height,
            } => {
                let popup_id = xdg_popup.id();
                for surface in state.all_outputs_mut() {
                    if let Some(popup_manager) = surface.popup_manager() {
                        if popup_manager.find_by_xdg_popup(&popup_id).is_some() {
                            surface.handle_xdg_popup_configure(xdg_popup, x, y, width, height);
                            break;
                        }
                    }
                }
            }
            xdg_popup::Event::PopupDone => {
                info!("XdgPopup dismissed by compositor");
                let popup_id = xdg_popup.id();

                for surface in state.all_outputs_mut() {
                    let popup_handle = surface
                        .popup_manager()
                        .as_ref()
                        .and_then(|pm| pm.find_by_xdg_popup(&popup_id));

                    if popup_handle.is_some() {
                        surface.handle_xdg_popup_done(xdg_popup);
                        break;
                    }
                }
            }
            xdg_popup::Event::Repositioned { token } => {
                info!("XdgPopup repositioned with token {token}");

                let popup_id = xdg_popup.id();
                for surface in state.all_outputs_mut() {
                    if let Some(popup_manager) = surface.popup_manager() {
                        if let Some(handle) = popup_manager.find_by_xdg_popup(&popup_id) {
                            info!("Committing popup surface after reposition");
                            popup_manager.commit_popup_surface(handle.key());
                            break;
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

impl Dispatch<XdgSurface, ()> for AppState {
    fn event(
        state: &mut Self,
        xdg_surface: &XdgSurface,
        event: xdg_surface::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        if let xdg_surface::Event::Configure { serial } = event {
            let xdg_surface_id = xdg_surface.id();
            for surface in state.all_outputs_mut() {
                if let Some(popup_manager) = surface.popup_manager() {
                    if popup_manager.find_by_xdg_surface(&xdg_surface_id).is_some() {
                        surface.handle_xdg_surface_configure(xdg_surface, serial);
                        break;
                    }
                }
            }
        }
    }
}

impl Dispatch<WlRegistry, GlobalListContents> for AppState {
    fn event(
        state: &mut Self,
        registry: &WlRegistry,
        event: <WlRegistry as Proxy>::Event,
        _data: &GlobalListContents,
        _conn: &Connection,
        qhandle: &QueueHandle<Self>,
    ) {
        match event {
            Event::Global {
                name,
                interface,
                version,
            } => {
                if interface == "wl_output" {
                    info!(
                        "Hot-plugged output detected! Binding wl_output with name {name}, version {version}"
                    );

                    let output = registry.bind::<WlOutput, _, _>(name, 4.min(version), qhandle, ());
                    let output_id = output.id();

                    if let Some(manager) = state.output_manager() {
                        let mut manager_ref = manager.borrow_mut();
                        let handle = manager_ref.register_output(output, qhandle);
                        info!("Registered hot-plugged output with handle {handle:?}");

                        state.register_registry_name(name, output_id);
                    } else {
                        info!("No output manager available yet (startup initialization)");
                    }
                }
            }
            Event::GlobalRemove { name } => {
                info!("Registry global removed: name {name}");

                if let Some(output_id) = state.unregister_registry_name(name) {
                    info!("Output with registry name {name} removed, cleaning up...");

                    if let Some(manager) = state.output_manager() {
                        let mut manager_ref = manager.borrow_mut();
                        manager_ref.remove_output(&output_id, state);
                    }
                }
            }
            _ => {}
        }
    }
}

macro_rules! impl_empty_dispatch_app {
    ($(($t:ty, $u:ty)),+) => {
        $(
            impl Dispatch<$t, $u> for AppState {
                fn event(
                    _state: &mut Self,
                    _proxy: &$t,
                    _event: <$t as wayland_client::Proxy>::Event,
                    _data: &$u,
                    _conn: &Connection,
                    _qhandle: &QueueHandle<Self>,
                ) {
                  debug!("Implement empty dispatch event for {:?}", stringify!($t));
                }
            }
        )+
    };
}

impl_empty_dispatch_app!(
    (WlCompositor, ()),
    (WlSurface, ()),
    (WlRegion, ()),
    (ZwlrLayerShellV1, ()),
    (WlSeat, ()),
    (WpFractionalScaleManagerV1, ()),
    (WpViewporter, ()),
    (WpViewport, ()),
    (XdgPositioner, ())
);
