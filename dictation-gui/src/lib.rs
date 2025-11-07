use iced::widget::{canvas, column, container, scrollable, text};
use iced::{time, Alignment, Color, Element, Length, Task};
use iced_layershell::to_layer_message;
use std::time::Duration;
use tracing::{debug, error, info, trace};

pub mod animation;
pub mod animations;
pub mod channel_listener;
pub mod collapse_widget;
pub mod config;
pub mod fft;
pub mod layout;
pub mod monitor_detection;
pub mod per_monitor_window;
pub mod renderer;
pub mod renderer_v2;
pub mod shared_state;
pub mod spectrum_widget;
pub mod spinner_widget;
pub mod text_renderer;
pub mod wayland;

use collapse_widget::CollapsingDots;
use spectrum_widget::SpectrumBars;
use spinner_widget::Spinner;

const FFT_SIZE: usize = 512;
pub const SAMPLE_RATE: u32 = 16000;

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

    info!("Starting dictation-gui with multi-monitor support");

    // Create shared state
    let shared_state = shared_state::SharedState::new();

    // Spawn Hyprland event listener
    info!("Spawning Hyprland event listener");
    monitor_detection::spawn_active_monitor_listener(shared_state.clone());

    // Enumerate monitors
    info!("Enumerating monitors...");
    let monitors = match monitor_detection::enumerate_monitors() {
        Ok(monitors) => {
            if monitors.is_empty() {
                tracing::error!("No monitors detected! Exiting.");
                std::process::exit(1);
            }
            monitors
        }
        Err(e) => {
            tracing::error!("Failed to enumerate monitors: {}. Exiting.", e);
            std::process::exit(1);
        }
    };

    info!("Detected {} monitor(s): {:?}", monitors.len(), monitors);

    // Spawn a window thread for each monitor
    let monitor_count = monitors.len();
    let mut handles = Vec::new();

    for (idx, monitor_name) in monitors.into_iter().enumerate() {
        let state_clone = shared_state.clone();
        let monitor_clone = monitor_name.clone();

        info!("Spawning window thread for monitor: {}", monitor_name);

        let handle = std::thread::spawn(move || {
            if let Err(e) = per_monitor_window::run_monitor_window(monitor_clone.clone(), state_clone) {
                tracing::error!("Window thread for monitor {} failed: {}", monitor_clone, e);
            }
        });

        handles.push(handle);

        // Brief delay between spawns to avoid race conditions
        if idx < monitor_count - 1 {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }

    info!("All monitor windows spawned, waiting for threads...");

    // Wait for all threads (they run indefinitely until exit)
    for handle in handles {
        let _ = handle.join();
    }

    Ok(())
}

/// Run GUI integrated with daemon (channel-based communication)
pub fn run_integrated(
    gui_control_rx: tokio::sync::broadcast::Receiver<dictation_types::GuiControl>,
    spectrum_rx: tokio::sync::broadcast::Receiver<Vec<f32>>,
    gui_status_tx: tokio::sync::mpsc::Sender<dictation_types::GuiStatus>,
) -> Result<(), iced_layershell::Error> {
    // Note: tracing subscriber is already initialized by the daemon in integrated mode
    // No need to initialize it again here

    info!("Starting dictation-gui (integrated mode) with multi-monitor support");

    // Create shared state
    let shared_state = shared_state::SharedState::new();

    // Spawn channel listeners (replaces background_tasks)
    info!("Spawning channel listeners");
    channel_listener::spawn_channel_listener(
        gui_control_rx,
        spectrum_rx,
        shared_state.clone(),
        gui_status_tx.clone(),
    );

    // Spawn Hyprland event listener
    info!("Spawning Hyprland event listener");
    monitor_detection::spawn_active_monitor_listener(shared_state.clone());

    // Enumerate monitors
    info!("Enumerating monitors...");
    let monitors = match monitor_detection::enumerate_monitors() {
        Ok(monitors) => {
            if monitors.is_empty() {
                tracing::error!("No monitors detected! Exiting.");
                let _ = gui_status_tx.blocking_send(dictation_types::GuiStatus::Error(
                    "No monitors detected".to_string(),
                ));
                std::process::exit(1);
            }
            monitors
        }
        Err(e) => {
            tracing::error!("Failed to enumerate monitors: {}. Exiting.", e);
            let _ = gui_status_tx.blocking_send(dictation_types::GuiStatus::Error(
                format!("Failed to enumerate monitors: {}", e),
            ));
            std::process::exit(1);
        }
    };

    info!("Detected {} monitor(s): {:?}", monitors.len(), monitors);

    // Spawn a window thread for each monitor
    let monitor_count = monitors.len();
    let mut handles = Vec::new();

    for (idx, monitor_name) in monitors.into_iter().enumerate() {
        let state_clone = shared_state.clone();
        let monitor_clone = monitor_name.clone();

        info!("Spawning window thread for monitor: {}", monitor_name);

        let handle = std::thread::spawn(move || {
            if let Err(e) = per_monitor_window::run_monitor_window(monitor_clone.clone(), state_clone) {
                tracing::error!("Window thread for monitor {} failed: {}", monitor_clone, e);
            }
        });

        handles.push(handle);

        // Brief delay between spawns to avoid race conditions
        if idx < monitor_count - 1 {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }

    info!("All monitor windows spawned");

    // Send ready signal to daemon
    if let Err(e) = gui_status_tx.blocking_send(dictation_types::GuiStatus::Ready) {
        error!("Failed to send ready status: {}", e);
    } else {
        info!("Sent Ready status to daemon");
    }

    info!("Waiting for threads...");

    // Wait for all threads (they run indefinitely until exit)
    for handle in handles {
        let _ = handle.join();
    }

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuiState {
    Hidden,
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
const CONTENT_SPACING: f32 = 5.0;
const MAX_TEXT_LINES: usize = 2;

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
    closing_animation_time: f32,
    config: config::Config,
}

impl Default for DictationOverlay {
    fn default() -> Self {
        let config = config::load_config();
        let target_size = calculate_prelistening_size(&config);
        Self {
            state: GuiState::PreListening,
            transition_phase: TransitionPhase::Transitioning,
            transition_progress: 0.0,
            previous_state: None,
            current_size: (0.0, 0.0),
            target_size,
            transcription: String::new(),
            band_values: Vec::new(),
            animation_time: 0.0,
            closing_animation_time: 0.0,
            config,
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

fn ease_in_out_cubic(t: f32) -> f32 {
    if t < 0.5 {
        4.0 * t.powi(3)
    } else {
        1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
    }
}

fn get_transition_duration(overlay: &DictationOverlay) -> f32 {
    let anims = &overlay.config.animations;
    match (overlay.previous_state, overlay.state) {
        (Some(GuiState::PreListening), GuiState::Listening) => anims.transition_to_listening_duration as f32 / 1000.0,
        (Some(GuiState::Listening), GuiState::Processing) => {
            anims.listening_content_out_fade_duration.max(anims.processing_content_in_fade_duration) as f32 / 1000.0
        },
        (Some(GuiState::Processing), GuiState::Closing) | (_, GuiState::Closing) => anims.closing_background_duration as f32 / 1000.0,
        _ => 0.5, // Default fallback
    }
}

fn interpolate_size(from: (f32, f32), to: (f32, f32), progress: f32) -> (f32, f32) {
    let eased = ease_in_out_cubic(progress);
    (from.0 + (to.0 - from.0) * eased, from.1 + (to.1 - from.1) * eased)
}

fn calculate_listening_size(transcription: &str, config: &config::Config) -> (f32, f32) {
    let padding = config.elements.background_padding as f32;
    let base_height = SPECTRUM_HEIGHT + padding * 2.0;
    let width = config.gui_general.window_width as f32;

    if transcription.is_empty() {
        return (width, base_height);
    }

    let text_font_size = config.elements.text_font_size as f32;
    let char_width = text_font_size * 0.6;
    let chars_per_line = ((width - padding * 2.0) / char_width) as usize;
    
    let char_count = transcription.len();
    let line_count = ((char_count as f32 / chars_per_line as f32).ceil() as usize).max(1).min(MAX_TEXT_LINES);
    let text_line_height = text_font_size * config.elements.text_line_height;
    let text_height = line_count as f32 * text_line_height;

    let total_height = base_height + CONTENT_SPACING + text_height;
    (width, total_height)
}

fn calculate_prelistening_size(config: &config::Config) -> (f32, f32) {
    let padding = config.elements.background_padding as f32;
    let initial_height = config.gui_general.window_height as f32;
    (SPECTRUM_HEIGHT * 2.4, initial_height.max(SPECTRUM_HEIGHT + padding * 2.0))
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

fn subscription(overlay: &DictationOverlay) -> iced::Subscription<Message> {
    let update_interval_ms = 1000 / overlay.config.elements.spectrum_update_rate as u64;
    time::every(Duration::from_millis(update_interval_ms)).map(|_| Message::Tick)
}

fn update(overlay: &mut DictationOverlay, message: Message) -> Task<Message> {
    match message {
        Message::Tick => {
            trace!("UPDATE: Tick (animation_time: {:.3})", overlay.animation_time);
            let delta_time = 1.0 / overlay.config.elements.spectrum_update_rate as f32;
            overlay.animation_time += delta_time;

            if overlay.transition_phase == TransitionPhase::Transitioning {
                let transition_duration = get_transition_duration(overlay);
                overlay.transition_progress += delta_time / transition_duration;

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
                overlay.closing_animation_time += delta_time;
                let closing_duration = overlay.config.animations.closing_background_duration as f32 / 1000.0;
                if overlay.closing_animation_time >= closing_duration {
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
                let new_size = calculate_listening_size(&overlay.transcription, &overlay.config);
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
                GuiState::Hidden => (0.0, 0.0),
                GuiState::PreListening => calculate_prelistening_size(&overlay.config),
                GuiState::Listening => calculate_listening_size(&overlay.transcription, &overlay.config),
                GuiState::Processing => {
                    let cfg = &overlay.config.elements;
                    let padding = cfg.background_padding as f32;
                    let spinner_size = (cfg.spinner_orbit_radius * 2.0 + cfg.spinner_dot_radius * 2.0) * 1.5;
                    let size = spinner_size + padding * 2.0;
                    (size, size)
                },
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
        (_, GuiState::Hidden, _) => {
            // Hidden state: return empty/invisible element
            container(text(""))
                .width(Length::Fixed(0.0))
                .height(Length::Fixed(0.0))
                .into()
        }
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
    let cfg = &overlay.config.elements;
    let alpha = overlay.transition_progress;

    let text_content = text("Starting...").size(cfg.text_font_size as f32).color(Color::from_rgba(1.0, 1.0, 1.0, cfg.text_opacity * alpha));

    let padding = cfg.background_padding as f32;
    let content = column![text_content].align_x(Alignment::Center).padding(padding);

    let bg_opacity = cfg.background_opacity * alpha;
    let corner_radius = cfg.background_corner_radius;

    let inner = container(content)
        .width(Length::Fixed(width))
        .height(Length::Fixed(height))
        .padding(padding)
        .style(move |_theme: &iced::Theme| container::Style {
            background: Some(iced::Background::Color(Color::from_rgba(0.0, 0.0, 0.0, bg_opacity))),
            border: iced::Border { radius: corner_radius.into(), ..Default::default() },
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
    let cfg = &overlay.config.elements;

    let band_values =
        if overlay.band_values.is_empty() { vec![0.0; 8] } else { overlay.band_values.clone() };

    let spectrum = SpectrumBars::new(
        band_values,
        cfg.spectrum_min_bar_height,
        cfg.spectrum_max_bar_height,
        cfg.spectrum_bar_width_factor,
        cfg.spectrum_bar_spacing,
        cfg.spectrum_bar_radius,
        cfg.spectrum_opacity * alpha,
    ).height(SPECTRUM_HEIGHT).width(SPECTRUM_WIDTH);

    let spectrum_container = container(spectrum).width(Length::Fill).center_x(Length::Fill);

    let mut content_items = vec![spectrum_container.into()];

    if !overlay.transcription.is_empty() && cfg.text_enabled {
        let text_color = Color::from_rgba(1.0, 1.0, 1.0, cfg.text_opacity * alpha);
        let text_widget = text(&overlay.transcription).size(cfg.text_font_size as f32).color(text_color);

        let text_alignment = match cfg.text_alignment.as_str() {
            "left" => Alignment::Start,
            "right" => Alignment::End,
            _ => Alignment::Center,
        };

        let text_content = container(text_widget).width(Length::Fill).align_x(text_alignment);

        let padding = cfg.background_padding as f32;
        let text_height = height - SPECTRUM_HEIGHT - (padding * 2.0) - CONTENT_SPACING;

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

    let bg_opacity = cfg.background_opacity * alpha;
    let corner_radius = cfg.background_corner_radius;
    let padding = cfg.background_padding as f32;

    let inner = container(content)
        .width(Length::Fixed(width))
        .height(Length::Fixed(height))
        .padding(padding)
        .style(move |_theme: &iced::Theme| container::Style {
            background: Some(iced::Background::Color(Color::from_rgba(0.0, 0.0, 0.0, bg_opacity))),
            border: iced::Border { radius: corner_radius.into(), ..Default::default() },
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
    let cfg = &overlay.config.elements;

    let spinner_size = (cfg.spinner_orbit_radius * 2.0 + cfg.spinner_dot_radius * 2.0) * 1.5;

    let spinner = canvas(Spinner::new(
        overlay.animation_time,
        cfg.spinner_dot_count,
        cfg.spinner_dot_radius,
        cfg.spinner_orbit_radius,
        cfg.spinner_rotation_speed,
        cfg.spinner_opacity * alpha,
    ))
        .width(Length::Fixed(spinner_size))
        .height(Length::Fixed(spinner_size));

    let content = container(spinner)
        .width(Length::Fill)
        .height(Length::Fill)
        .center_x(Length::Fill)
        .center_y(Length::Fill);

    let bg_opacity = cfg.background_opacity * alpha;
    let corner_radius = cfg.background_corner_radius_processing;
    let padding = cfg.background_padding as f32;

    let inner = container(content)
        .width(Length::Fixed(width))
        .height(Length::Fixed(height))
        .padding(padding)
        .style(move |_theme: &iced::Theme| container::Style {
            background: Some(iced::Background::Color(Color::from_rgba(0.0, 0.0, 0.0, bg_opacity))),
            border: iced::Border { radius: corner_radius.into(), ..Default::default() },
            ..Default::default()
        });

    container(inner).center_x(Length::Fill).center_y(Length::Fill).into()
}

fn view_closing<'a>(overlay: &'a DictationOverlay) -> Element<'a, Message> {
    let cfg = &overlay.config.elements;
    let closing_duration = overlay.config.animations.closing_background_duration as f32 / 1000.0;
    let progress = (overlay.closing_animation_time / closing_duration).min(1.0);
    let alpha = cfg.background_opacity * (1.0 - progress);

    let collapse = CollapsingDots::new(progress, overlay.animation_time);

    let spinner_size = (cfg.spinner_orbit_radius * 2.0 + cfg.spinner_dot_radius * 2.0) * 1.5;
    let collapse_canvas =
        canvas(collapse).width(Length::Fixed(spinner_size)).height(Length::Fixed(spinner_size));

    let (width, height) = overlay.current_size;
    let shrink_width = width * (1.0 - progress);
    let shrink_height = height * (1.0 - progress);

    let content = container(collapse_canvas)
        .width(Length::Fill)
        .height(Length::Fill)
        .center_x(Length::Fill)
        .center_y(Length::Fill);

    let padding = cfg.background_padding as f32;
    let corner_radius = cfg.background_corner_radius_processing;

    let inner = container(content)
        .width(Length::Fixed(shrink_width.max(1.0)))
        .height(Length::Fixed(shrink_height.max(1.0)))
        .padding(padding)
        .style(move |_theme: &iced::Theme| container::Style {
            background: Some(iced::Background::Color(Color::from_rgba(0.0, 0.0, 0.0, alpha))),
            border: iced::Border { radius: corner_radius.into(), ..Default::default() },
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
