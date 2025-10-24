use iced::widget::{canvas, column, container, scrollable, text};
use iced::{alignment, time, Alignment, Color, Element, Length, Task};
use iced_layershell::build_pattern::application;
use iced_layershell::reexport::{Anchor, KeyboardInteractivity, Layer};
use iced_layershell::settings::LayerShellSettings;
use iced_layershell::to_layer_message;
use std::time::Duration;
use tracing::{debug, info, trace};

pub mod animation;
pub mod animations;
pub mod collapse_widget;
pub mod control_ipc;
pub mod fft;
pub mod ipc;
pub mod ipc_subscription;
pub mod layout;
pub mod renderer;
pub mod renderer_v2;
pub mod spectrum_widget;
pub mod spinner_widget;
pub mod text_renderer;
pub mod wayland;

use collapse_widget::CollapsingDots;
use fft::SpectrumAnalyzer;
use spectrum_widget::SpectrumBars;
use spinner_widget::Spinner;

const WIDTH: u32 = 400;
const SAMPLE_RATE: u32 = 16000;
const FFT_SIZE: usize = 512;
const SOCKET_PATH: &str = "/tmp/voice-dictation.sock";
const CONTROL_SOCKET_PATH: &str = "/tmp/voice-dictation-control.sock";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextAlign {
    Left,
    Center,
    Right,
}

impl TextAlign {
    fn to_horizontal(&self) -> alignment::Horizontal {
        match self {
            TextAlign::Left => alignment::Horizontal::Left,
            TextAlign::Center => alignment::Horizontal::Center,
            TextAlign::Right => alignment::Horizontal::Right,
        }
    }
}

pub fn run() -> Result<(), iced_layershell::Error> {
    let log_level = std::env::var("GUI_LOG").unwrap_or_else(|_| "error".to_string()).to_lowercase();

    let filter = match log_level.as_str() {
        "silent" => tracing::Level::ERROR,
        "error" => tracing::Level::ERROR,
        "warning" | "warn" => tracing::Level::WARN,
        "info" => tracing::Level::INFO,
        "debug" => tracing::Level::DEBUG,
        "verbose" | "trace" => tracing::Level::TRACE,
        _ => tracing::Level::ERROR,
    };

    tracing_subscriber::fmt().with_max_level(filter).init();

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
pub enum GuiState {
    PreListening,
    Listening,
    Processing,
    Closing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransitionPhase {
    Idle,
    Transitioning,
}

const SPECTRUM_HEIGHT: f32 = 50.0;
const SPECTRUM_WIDTH: f32 = 400.0;
const SPINNER_SIZE: f32 = 100.0;
const CONTAINER_PADDING: f32 = 10.0;
const CONTENT_SPACING: f32 = 5.0;
const TEXT_LINE_HEIGHT: f32 = 22.0;
const TEXT_SIZE: f32 = 18.0;
const MAX_TEXT_LINES: usize = 2;

const CHAR_WIDTH: f32 = TEXT_SIZE * 0.6;
const CHARS_PER_LINE: usize = ((SPECTRUM_WIDTH - CONTAINER_PADDING * 2.0) / CHAR_WIDTH) as usize;
const TEXT_ALIGNMENT: TextAlign = TextAlign::Center;

const LISTENING_WIDTH: f32 = SPECTRUM_WIDTH;
const PROCESSING_SIZE: (f32, f32) =
    (SPINNER_SIZE + CONTAINER_PADDING * 2.0, SPINNER_SIZE + CONTAINER_PADDING * 2.0);
const TRANSITION_DURATION: f32 = 0.5;

struct DictationOverlay {
    state: GuiState,
    transition_phase: TransitionPhase,
    transition_progress: f32,
    previous_state: Option<GuiState>,
    current_size: (f32, f32),
    target_size: (f32, f32),
    transcription: String,
    band_values: Vec<f32>,
    animation_time: f32,
    analyzer: Option<SpectrumAnalyzer>,
    closing_animation_time: f32,
}

impl Default for DictationOverlay {
    fn default() -> Self {
        Self {
            state: GuiState::PreListening,
            transition_phase: TransitionPhase::Transitioning,
            transition_progress: 0.0,
            previous_state: None,
            current_size: (0.0, 0.0),
            target_size: calculate_prelistening_size(),
            transcription: String::new(),
            band_values: Vec::new(),
            animation_time: 0.0,
            analyzer: None,
            closing_animation_time: 0.0,
        }
    }
}

impl Default for GuiState {
    fn default() -> Self {
        GuiState::PreListening
    }
}

impl Default for TransitionPhase {
    fn default() -> Self {
        TransitionPhase::Idle
    }
}

fn ease_out_cubic(t: f32) -> f32 {
    1.0 - (1.0 - t).powi(3)
}

fn ease_in_out_cubic(t: f32) -> f32 {
    if t < 0.5 {
        4.0 * t.powi(3)
    } else {
        1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
    }
}

fn interpolate_size(from: (f32, f32), to: (f32, f32), progress: f32) -> (f32, f32) {
    let eased = ease_in_out_cubic(progress);
    (from.0 + (to.0 - from.0) * eased, from.1 + (to.1 - from.1) * eased)
}

fn calculate_listening_size(transcription: &str) -> (f32, f32) {
    let base_height = SPECTRUM_HEIGHT + CONTAINER_PADDING * 2.0;

    if transcription.is_empty() {
        return (LISTENING_WIDTH, base_height);
    }

    let char_count = transcription.len();
    let line_count =
        ((char_count as f32 / CHARS_PER_LINE as f32).ceil() as usize).max(1).min(MAX_TEXT_LINES);
    let text_height = line_count as f32 * TEXT_LINE_HEIGHT;

    let total_height = base_height + CONTENT_SPACING + text_height;
    (LISTENING_WIDTH, total_height)
}

fn calculate_prelistening_size() -> (f32, f32) {
    (SPECTRUM_HEIGHT * 2.4, SPECTRUM_HEIGHT + CONTAINER_PADDING * 2.0)
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

            if overlay.transition_phase == TransitionPhase::Transitioning {
                overlay.transition_progress += 0.016 / TRANSITION_DURATION;

                if overlay.transition_progress >= 1.0 {
                    overlay.transition_progress = 1.0;
                    overlay.current_size = overlay.target_size;
                    overlay.transition_phase = TransitionPhase::Idle;
                    let completed_state = overlay.state;
                    overlay.previous_state = None;
                    debug!("Transition complete to {:?}", completed_state);

                    if completed_state == GuiState::PreListening {
                        return Task::perform(async {}, |_| {
                            Message::StateChange(GuiState::Listening)
                        });
                    }
                } else {
                    overlay.current_size = interpolate_size(
                        overlay.current_size,
                        overlay.target_size,
                        overlay.transition_progress,
                    );
                }
            }

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

            if overlay.state == GuiState::Listening {
                let new_size = calculate_listening_size(&overlay.transcription);
                if new_size != overlay.target_size {
                    overlay.target_size = new_size;
                    overlay.transition_phase = TransitionPhase::Transitioning;
                    overlay.transition_progress = 0.0;
                }
            }

            Task::none()
        }

        Message::StateChange(state) => {
            info!("UPDATE: State change {:?} -> {:?}", overlay.state, state);

            overlay.previous_state = Some(overlay.state);
            overlay.state = state;
            overlay.transition_phase = TransitionPhase::Transitioning;
            overlay.transition_progress = 0.0;

            overlay.target_size = match state {
                GuiState::PreListening => calculate_prelistening_size(),
                GuiState::Listening => calculate_listening_size(&overlay.transcription),
                GuiState::Processing => PROCESSING_SIZE,
                GuiState::Closing => {
                    overlay.closing_animation_time = 0.0;
                    (0.0, 0.0)
                }
            };

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
    match (overlay.transition_phase, overlay.state, overlay.previous_state) {
        (TransitionPhase::Transitioning, GuiState::Listening, Some(GuiState::PreListening)) => {
            view_transition_prelistening_to_listening(overlay)
        }
        (TransitionPhase::Transitioning, GuiState::Processing, Some(GuiState::Listening)) => {
            view_transition_listening_to_processing(overlay)
        }
        (_, GuiState::PreListening, _) => view_prelistening(overlay),
        (_, GuiState::Listening, _) => view_listening(overlay),
        (_, GuiState::Processing, _) => view_processing(overlay),
        (_, GuiState::Closing, _) => view_closing(overlay),
    }
}

fn view_prelistening<'a>(overlay: &'a DictationOverlay) -> Element<'a, Message> {
    let (width, height) = overlay.current_size;
    let alpha = overlay.transition_progress;

    let text_content = text("Starting...").size(16).color(Color::from_rgba(1.0, 1.0, 1.0, alpha));

    let content = column![text_content].align_x(Alignment::Center).padding(10);

    let inner = container(content)
        .width(Length::Fixed(width))
        .height(Length::Fixed(height))
        .padding(10)
        .style(move |_theme: &iced::Theme| container::Style {
            background: Some(iced::Background::Color(Color::from_rgba(0.0, 0.0, 0.0, 0.9 * alpha))),
            border: iced::Border { radius: 15.0.into(), ..Default::default() },
            ..Default::default()
        });

    container(inner).center_x(Length::Fill).center_y(Length::Fill).into()
}

fn view_transition_prelistening_to_listening<'a>(
    overlay: &'a DictationOverlay,
) -> Element<'a, Message> {
    view_listening(overlay)
}

fn view_transition_listening_to_processing<'a>(
    overlay: &'a DictationOverlay,
) -> Element<'a, Message> {
    let progress = overlay.transition_progress;

    if progress < 0.5 {
        let listening_alpha = 1.0 - (progress * 2.0);
        view_listening_with_alpha(overlay, listening_alpha)
    } else {
        let processing_alpha = (progress - 0.5) * 2.0;
        view_processing_with_alpha(overlay, processing_alpha)
    }
}

fn view_listening<'a>(overlay: &'a DictationOverlay) -> Element<'a, Message> {
    view_listening_with_alpha(overlay, 1.0)
}

fn view_listening_with_alpha<'a>(
    overlay: &'a DictationOverlay,
    alpha: f32,
) -> Element<'a, Message> {
    let (width, height) = overlay.current_size;

    let band_values =
        if overlay.band_values.is_empty() { vec![0.0; 8] } else { overlay.band_values.clone() };

    let spectrum = SpectrumBars::new(band_values).height(SPECTRUM_HEIGHT).width(SPECTRUM_WIDTH);

    let spectrum_container = container(spectrum).width(Length::Fill).center_x(Length::Fill);

    let mut content_items = vec![spectrum_container.into()];

    if !overlay.transcription.is_empty() {
        let text_color = Color::from_rgba(1.0, 1.0, 1.0, alpha);
        let text_widget = text(&overlay.transcription).size(TEXT_SIZE).color(text_color);

        let text_content =
            container(text_widget).width(Length::Fill).align_x(match TEXT_ALIGNMENT {
                TextAlign::Left => Alignment::Start,
                TextAlign::Center => Alignment::Center,
                TextAlign::Right => Alignment::End,
            });

        let text_height = height - SPECTRUM_HEIGHT - (CONTAINER_PADDING * 2.0) - CONTENT_SPACING;

        let scrollable_text = scrollable(text_content)
            .width(Length::Fill)
            .height(Length::Fixed(text_height))
            .direction(scrollable::Direction::Vertical(
                scrollable::Scrollbar::default().anchor(scrollable::Anchor::End),
            ))
            .style(|_theme: &iced::Theme, _status| scrollable::Style {
                container: container::Style::default(),
                vertical_rail: scrollable::Rail {
                    background: None,
                    border: iced::Border::default(),
                    scroller: scrollable::Scroller {
                        color: Color::TRANSPARENT,
                        border: iced::Border::default(),
                    },
                },
                horizontal_rail: scrollable::Rail {
                    background: None,
                    border: iced::Border::default(),
                    scroller: scrollable::Scroller {
                        color: Color::TRANSPARENT,
                        border: iced::Border::default(),
                    },
                },
                gap: None,
            });

        content_items.push(scrollable_text.into());
    }

    let content = column(content_items).spacing(CONTENT_SPACING);

    let inner = container(content)
        .width(Length::Fixed(width))
        .height(Length::Fixed(height))
        .padding(CONTAINER_PADDING)
        .style(move |_theme: &iced::Theme| container::Style {
            background: Some(iced::Background::Color(Color::from_rgba(0.0, 0.0, 0.0, 0.9 * alpha))),
            border: iced::Border { radius: 15.0.into(), ..Default::default() },
            ..Default::default()
        });

    container(inner).center_x(Length::Fill).center_y(Length::Fill).into()
}

fn view_processing<'a>(overlay: &'a DictationOverlay) -> Element<'a, Message> {
    view_processing_with_alpha(overlay, 1.0)
}

fn view_processing_with_alpha<'a>(
    overlay: &'a DictationOverlay,
    alpha: f32,
) -> Element<'a, Message> {
    let (width, height) = overlay.current_size;

    let spinner = canvas(Spinner::new(overlay.animation_time))
        .width(Length::Fixed(SPINNER_SIZE))
        .height(Length::Fixed(SPINNER_SIZE));

    let content = container(spinner)
        .width(Length::Fill)
        .height(Length::Fill)
        .center_x(Length::Fill)
        .center_y(Length::Fill);

    let inner = container(content)
        .width(Length::Fixed(width))
        .height(Length::Fixed(height))
        .padding(CONTAINER_PADDING)
        .style(move |_theme: &iced::Theme| container::Style {
            background: Some(iced::Background::Color(Color::from_rgba(0.0, 0.0, 0.0, 0.9 * alpha))),
            border: iced::Border { radius: 50.0.into(), ..Default::default() },
            ..Default::default()
        });

    container(inner).center_x(Length::Fill).center_y(Length::Fill).into()
}

fn view_closing<'a>(overlay: &'a DictationOverlay) -> Element<'a, Message> {
    let progress = (overlay.closing_animation_time / 0.5).min(1.0);
    let alpha = 0.9 * (1.0 - progress);

    let collapse = CollapsingDots::new(progress, overlay.animation_time);

    let collapse_canvas =
        canvas(collapse).width(Length::Fixed(SPINNER_SIZE)).height(Length::Fixed(SPINNER_SIZE));

    let (width, height) = overlay.current_size;
    let shrink_width = width * (1.0 - progress);
    let shrink_height = height * (1.0 - progress);

    let content = container(collapse_canvas)
        .width(Length::Fill)
        .height(Length::Fill)
        .center_x(Length::Fill)
        .center_y(Length::Fill);

    let inner = container(content)
        .width(Length::Fixed(shrink_width.max(1.0)))
        .height(Length::Fixed(shrink_height.max(1.0)))
        .padding(CONTAINER_PADDING)
        .style(move |_theme: &iced::Theme| container::Style {
            background: Some(iced::Background::Color(Color::from_rgba(0.0, 0.0, 0.0, alpha))),
            border: iced::Border { radius: 50.0.into(), ..Default::default() },
            ..Default::default()
        });

    container(inner).center_x(Length::Fill).center_y(Length::Fill).into()
}

fn style(_overlay: &DictationOverlay, theme: &iced::Theme) -> iced_layershell::Appearance {
    iced_layershell::Appearance {
        background_color: Color::TRANSPARENT,
        text_color: theme.palette().text,
    }
}
