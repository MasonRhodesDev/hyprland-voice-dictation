use iced::widget::{canvas, column, container, scrollable, text, Space};
use iced::{Alignment, Color, Element, Length, Task, time};
use iced_layershell::build_pattern::application;
use iced_layershell::reexport::{Anchor, KeyboardInteractivity, Layer};
use iced_layershell::settings::LayerShellSettings;
use iced_layershell::to_layer_message;
use std::time::Duration;
use tracing::{debug, info, trace};

mod collapse_widget;
mod control_ipc;
mod fft;
mod ipc;
mod ipc_subscription;
mod spectrum_widget;
mod spinner_widget;

use collapse_widget::CollapsingDots;
use fft::SpectrumAnalyzer;
use spectrum_widget::SpectrumBars;
use spinner_widget::Spinner;

const WIDTH: u32 = 400;
const SAMPLE_RATE: u32 = 16000;
const FFT_SIZE: usize = 512;
const SOCKET_PATH: &str = "/tmp/voice-dictation.sock";
const CONTROL_SOCKET_PATH: &str = "/tmp/voice-dictation-control.sock";

pub fn main() -> Result<(), iced_layershell::Error> {
    let log_level = std::env::var("GUI_LOG")
        .unwrap_or_else(|_| "error".to_string())
        .to_lowercase();
    
    let filter = match log_level.as_str() {
        "silent" => tracing::Level::ERROR,
        "error" => tracing::Level::ERROR,
        "warning" | "warn" => tracing::Level::WARN,
        "info" => tracing::Level::INFO,
        "debug" => tracing::Level::DEBUG,
        "verbose" | "trace" => tracing::Level::TRACE,
        _ => tracing::Level::ERROR,
    };
    
    tracing_subscriber::fmt()
        .with_max_level(filter)
        .init();
    
    info!("Starting dictation-gui with iced_layershell");

    application(namespace, update, view)
        .layer_settings(LayerShellSettings {
            size: Some((WIDTH, 160)),
            anchor: Anchor::Bottom | Anchor::Left | Anchor::Right,
            layer: Layer::Overlay,
            keyboard_interactivity: KeyboardInteractivity::None,
            margin: (0, 0, 10, 0),
            ..Default::default()
        })
        .subscription(subscription)
        .style(style)
        .run()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GuiState {
    Listening,
    Processing,
    Closing,
}

#[derive(Default)]
struct DictationOverlay {
    state: GuiState,
    transcription: String,
    band_values: Vec<f32>,
    animation_time: f32,
    analyzer: Option<SpectrumAnalyzer>,
    closing_animation_time: f32,
}

impl Default for GuiState {
    fn default() -> Self {
        GuiState::Listening
    }
}

#[to_layer_message]
#[derive(Debug, Clone)]
enum Message {
    Tick,
    SpectrumUpdate(Vec<f32>),
    TranscriptionUpdate(String),
    StateChange(GuiState),
    IpcError(String),
    Exit,
}

fn namespace(_overlay: &DictationOverlay) -> String {
    String::from("Dictation Overlay")
}

fn subscription(_overlay: &DictationOverlay) -> iced::Subscription<Message> {
    iced::Subscription::batch([
        time::every(Duration::from_millis(16)).map(|_| Message::Tick),
        ipc_subscription::audio_subscription(),
        ipc_subscription::control_subscription(),
    ])
}

fn update(overlay: &mut DictationOverlay, message: Message) -> Task<Message> {
    match message {
        Message::Tick => {
            trace!("UPDATE: Tick (animation_time: {:.3})", overlay.animation_time);
            overlay.animation_time += 0.016;
            
            if overlay.state == GuiState::Closing {
                overlay.closing_animation_time += 0.016;
                if overlay.closing_animation_time >= 0.5 {
                    info!("Closing animation complete, exiting");
                    return Task::perform(async {}, |_| Message::Exit);
                }
            }
            
            Task::none()
        }

        Message::SpectrumUpdate(values) => {
            let max_val = values.iter().cloned().fold(0.0f32, f32::max);
            debug!("UPDATE: SpectrumUpdate (values: {}, max: {:.3})", values.len(), max_val);
            trace!("UPDATE: Spectrum values: {:?}", values);
            overlay.band_values = values;
            Task::none()
        }

        Message::TranscriptionUpdate(text) => {
            if !text.is_empty() {
                info!("UPDATE: Transcription = '{}'", text);
                debug!("UPDATE: Transcription length: {} chars", text.len());
            } else {
                trace!("UPDATE: Transcription (empty)");
            }
            overlay.transcription = text;
            Task::none()
        }

        Message::StateChange(state) => {
            info!("UPDATE: State change {:?} -> {:?}", overlay.state, state);
            eprintln!("STATE CHANGE: {:?} -> {:?}", overlay.state, state);
            overlay.state = state;
            
            if state == GuiState::Closing {
                overlay.closing_animation_time = 0.0;
            }
            
            Task::none()
        }

        Message::IpcError(err) => {
            tracing::warn!("UPDATE: IPC error: {}", err);
            Task::none()
        }

        Message::Exit => {
            info!("EXIT: Exiting application");
            std::process::exit(0);
        }

        _ => {
            debug!("UPDATE: Unhandled message");
            Task::none()
        }
    }
}

fn view<'a>(overlay: &'a DictationOverlay) -> Element<'a, Message> {
    match overlay.state {
        GuiState::Listening => view_listening(overlay),
        GuiState::Processing => view_processing(overlay),
        GuiState::Closing => view_closing(overlay),
    }
}

fn view_listening<'a>(overlay: &'a DictationOverlay) -> Element<'a, Message> {
    let band_values = if overlay.band_values.is_empty() {
        vec![0.0; 8]
    } else {
        overlay.band_values.clone()
    };

    let spectrum = SpectrumBars::new(band_values)
        .height(50.0)
        .width(WIDTH as f32);

    let text_content = if overlay.transcription.is_empty() {
        text("Listening...").size(18).color(Color::WHITE)
    } else {
        text(&overlay.transcription).size(18).color(Color::WHITE)
    };

    let scrollable_text = scrollable(
        container(text_content)
            .width(Length::Fill)
            .padding(10)
    )
    .height(Length::Fixed(90.0));

    let content = column![
        spectrum,
        scrollable_text,
    ]
    .spacing(5)
    .padding(10)
    .width(Length::Fill);

    container(content)
        .width(Length::Fill)
        .padding(5)
        .style(|_theme: &iced::Theme| {
            container::Style {
                background: Some(iced::Background::Color(Color::from_rgba8(0, 0, 0, 0.9))),
                border: iced::Border {
                    radius: 15.0.into(),
                    ..Default::default()
                },
                ..Default::default()
            }
        })
        .into()
}

fn view_processing<'a>(overlay: &'a DictationOverlay) -> Element<'a, Message> {
    let spinner = canvas(Spinner::new(overlay.animation_time))
        .width(Length::Fixed(100.0))
        .height(Length::Fixed(100.0));

    let content = column![
        Space::with_height(Length::Fixed(10.0)),
        spinner,
        Space::with_height(Length::Fixed(10.0)),
    ]
    .align_x(Alignment::Center)
    .width(Length::Fill);

    container(content)
        .width(Length::Fill)
        .padding(5)
        .style(|_theme: &iced::Theme| {
            container::Style {
                background: Some(iced::Background::Color(Color::from_rgba8(0, 0, 0, 0.9))),
                border: iced::Border {
                    radius: 50.0.into(),
                    ..Default::default()
                },
                ..Default::default()
            }
        })
        .into()
}

fn view_closing<'a>(overlay: &'a DictationOverlay) -> Element<'a, Message> {
    let progress = (overlay.closing_animation_time / 0.5).min(1.0);
    let alpha = 0.9 * (1.0 - progress);
    
    let collapse = canvas(CollapsingDots::new(progress))
        .width(Length::Fixed(100.0))
        .height(Length::Fixed(100.0));
    
    let content = column![
        Space::with_height(Length::Fixed(10.0)),
        collapse,
        Space::with_height(Length::Fixed(10.0)),
    ]
    .align_x(Alignment::Center)
    .width(Length::Fill);
    
    container(content)
        .width(Length::Fill)
        .padding(5)
        .style(move |_theme: &iced::Theme| {
            container::Style {
                background: Some(iced::Background::Color(Color::from_rgba8(0, 0, 0, alpha))),
                border: iced::Border {
                    radius: 50.0.into(),
                    ..Default::default()
                },
                ..Default::default()
            }
        })
        .into()
}

fn style(_overlay: &DictationOverlay, theme: &iced::Theme) -> iced_layershell::Appearance {
    iced_layershell::Appearance {
        background_color: Color::TRANSPARENT,
        text_color: theme.palette().text,
    }
}
