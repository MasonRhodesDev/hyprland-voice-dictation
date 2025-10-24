pub mod animation;
pub mod animations;
pub mod control_ipc;
pub mod fft;
pub mod ipc;
pub mod layout;
pub mod renderer;
pub mod renderer_v2;
pub mod text_renderer;
pub mod wayland;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuiState {
    Listening,
    Processing,
    Closing,
}
