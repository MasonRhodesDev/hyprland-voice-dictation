use crate::{
    bind_globals, errors::LayerShikaError,
    rendering::egl::render_context_manager::RenderContextManager,
};
use log::info;
use smithay_client_toolkit::reexports::protocols_wlr::layer_shell::v1::client::zwlr_layer_shell_v1::ZwlrLayerShellV1;
use std::rc::Rc;
use wayland_client::{
    globals::registry_queue_init,
    protocol::{wl_compositor::WlCompositor, wl_output::WlOutput, wl_seat::WlSeat},
    Connection, Proxy, QueueHandle,
};
use wayland_protocols::wp::fractional_scale::v1::client::wp_fractional_scale_manager_v1::WpFractionalScaleManagerV1;
use wayland_protocols::wp::viewporter::client::wp_viewporter::WpViewporter;
use wayland_protocols::xdg::shell::client::xdg_wm_base::XdgWmBase;

use crate::wayland::surfaces::app_state::AppState;

pub struct GlobalContext {
    pub compositor: WlCompositor,
    pub outputs: Vec<WlOutput>,
    pub layer_shell: ZwlrLayerShellV1,
    pub seat: WlSeat,
    pub xdg_wm_base: Option<XdgWmBase>,
    pub fractional_scale_manager: Option<WpFractionalScaleManagerV1>,
    pub viewporter: Option<WpViewporter>,
    pub render_context_manager: Rc<RenderContextManager>,
}

impl GlobalContext {
    pub fn initialize(
        connection: &Connection,
        queue_handle: &QueueHandle<AppState>,
    ) -> Result<Self, LayerShikaError> {
        let global_list = registry_queue_init::<AppState>(connection)
            .map(|(global_list, _)| global_list)
            .map_err(|e| LayerShikaError::GlobalInitialization { source: e })?;

        let (compositor, layer_shell, seat) = bind_globals!(
            &global_list,
            queue_handle,
            (WlCompositor, compositor, 3..=6),
            (ZwlrLayerShellV1, layer_shell, 1..=5),
            (WlSeat, seat, 1..=9)
        )?;

        let output_names: Vec<u32> = global_list
            .contents()
            .clone_list()
            .into_iter()
            .filter(|global| global.interface == "wl_output")
            .map(|global| {
                info!(
                    "Found wl_output global with name: {} at version {}",
                    global.name, global.version
                );
                global.name
            })
            .collect();

        info!(
            "Total unique wl_output globals found: {}",
            output_names.len()
        );

        let outputs: Vec<WlOutput> = output_names
            .iter()
            .map(|&name| {
                info!("Binding wl_output with name: {}", name);
                global_list
                    .registry()
                    .bind::<WlOutput, _, _>(name, 4, queue_handle, ())
            })
            .collect();

        if outputs.is_empty() {
            return Err(LayerShikaError::InvalidInput {
                message: "No outputs found".into(),
            });
        }

        info!("Discovered {} output(s)", outputs.len());

        let xdg_wm_base = global_list
            .bind::<XdgWmBase, _, _>(queue_handle, 1..=6, ())
            .ok();

        let fractional_scale_manager = global_list
            .bind::<WpFractionalScaleManagerV1, _, _>(queue_handle, 1..=1, ())
            .ok();

        let viewporter = global_list
            .bind::<WpViewporter, _, _>(queue_handle, 1..=1, ())
            .ok();

        if xdg_wm_base.is_none() {
            info!("xdg-shell protocol not available, popup support disabled");
        }

        if fractional_scale_manager.is_none() {
            info!("Fractional scale protocol not available, using integer scaling");
        }

        if viewporter.is_none() {
            info!("Viewporter protocol not available");
        }

        let render_context_manager = RenderContextManager::new(&connection.display().id())?;

        Ok(Self {
            compositor,
            outputs,
            layer_shell,
            seat,
            xdg_wm_base,
            fractional_scale_manager,
            viewporter,
            render_context_manager,
        })
    }
}
