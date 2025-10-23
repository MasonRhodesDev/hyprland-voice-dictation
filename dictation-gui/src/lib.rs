pub mod control_ipc;
pub mod fft;
pub mod ipc;
pub mod renderer;
pub mod wayland;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuiState {
    Listening,
    Processing,
    Closing,
}
