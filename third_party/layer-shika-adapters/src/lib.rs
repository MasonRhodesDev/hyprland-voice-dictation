#![allow(clippy::pub_use)]

pub mod errors;
pub(crate) mod rendering;
pub(crate) mod wayland;

pub use rendering::femtovg::popup_window::PopupWindow;

pub use wayland::config::{MultiSurfaceConfig, ShellSurfaceConfig, WaylandSurfaceConfig};
pub use wayland::ops::WaylandSystemOps;
pub use wayland::shell_adapter::WaylandShellSystem;
pub use wayland::surfaces::app_state::AppState;
pub use wayland::surfaces::popup_manager::PopupManager;
pub use wayland::surfaces::surface_state::SurfaceState;

pub mod platform {
    pub use slint;
    pub use slint_interpreter;

    pub mod calloop {
        pub use smithay_client_toolkit::reexports::calloop::channel;
        pub use smithay_client_toolkit::reexports::calloop::generic::Generic;
        pub use smithay_client_toolkit::reexports::calloop::timer::{TimeoutAction, Timer};
        pub use smithay_client_toolkit::reexports::calloop::{
            EventSource, InsertError, Interest, Mode, PostAction, RegistrationToken,
        };
    }

    pub mod wayland {
        pub use smithay_client_toolkit::reexports::protocols_wlr::layer_shell::v1::client::{
            zwlr_layer_shell_v1::Layer as WaylandLayer,
            zwlr_layer_surface_v1::{
                Anchor, KeyboardInteractivity as WaylandKeyboardInteractivity,
            },
        };
    }
}
