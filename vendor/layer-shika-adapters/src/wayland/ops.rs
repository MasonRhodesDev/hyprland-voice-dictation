use crate::errors::Result;
use crate::wayland::config::ShellSurfaceConfig;
use crate::wayland::surfaces::app_state::AppState;
use layer_shika_domain::value_objects::output_handle::OutputHandle;
use slint_interpreter::ComponentInstance;
use smithay_client_toolkit::reexports::calloop::LoopHandle;

pub trait WaylandSystemOps {
    fn run(&mut self) -> Result<()>;

    fn spawn_surface(&mut self, config: &ShellSurfaceConfig) -> Result<Vec<OutputHandle>>;

    fn despawn_surface(&mut self, name: &str) -> Result<()>;

    fn app_state(&self) -> &AppState;

    fn app_state_mut(&mut self) -> &mut AppState;

    fn event_loop_handle(&self) -> LoopHandle<'static, AppState>;

    fn component_instance(&self) -> Result<&ComponentInstance>;
}
