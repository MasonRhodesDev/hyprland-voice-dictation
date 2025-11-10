use iced::widget::{canvas, column, container, scrollable, text};
use iced::{time, Alignment, Color, Element, Length, Task};
use iced_layershell::build_pattern::application;
use iced_layershell::reexport::{Anchor, KeyboardInteractivity, Layer};
use iced_layershell::settings::{LayerShellSettings, StartMode};
use iced_layershell::to_layer_message;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tracing::debug;

use crate::collapse_widget::CollapsingDots;
use crate::spectrum_widget::SpectrumBars;
use crate::spinner_widget::Spinner;
use crate::{config, shared_state::SharedState, GuiState};

const SPECTRUM_HEIGHT: f32 = 50.0;
const SPECTRUM_WIDTH: f32 = 400.0;
const CONTENT_SPACING: f32 = 5.0;
const MAX_TEXT_LINES: usize = 2;

/// Per-monitor window that reads from shared state
pub struct MonitorWindow {
    monitor_name: String,
    shared_state: Arc<RwLock<SharedState>>,
    config: config::Config,

    // Local cached state for rendering
    cached_state: GuiState,
    cached_transcription: String,
    cached_spectrum: Vec<f32>,
    cached_animation_time: f32,
    cached_closing_time: f32,

    // Transition tracking
    transition_phase: TransitionPhase,
    transition_progress: f32,
    previous_state: Option<GuiState>,
    current_size: (f32, f32),
    target_size: (f32, f32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransitionPhase {
    Idle,
    Transitioning,
}

#[to_layer_message]
#[derive(Debug, Clone)]
enum Message {
    Tick,
}

impl MonitorWindow {
    pub fn new(monitor_name: String, shared_state: Arc<RwLock<SharedState>>) -> Self {
        let config = config::load_config();
        let initial_size = calculate_prelistening_size(&config);

        Self {
            monitor_name,
            shared_state,
            config,
            cached_state: GuiState::Hidden,
            cached_transcription: String::new(),
            cached_spectrum: Vec::new(),
            cached_animation_time: 0.0,
            cached_closing_time: 0.0,
            transition_phase: TransitionPhase::Idle,
            transition_progress: 0.0,
            previous_state: None,
            current_size: initial_size,
            target_size: initial_size,
        }
    }
}

fn namespace(window: &MonitorWindow) -> String {
    format!("dictation-overlay-{}", window.monitor_name)
}

fn subscription(_window: &MonitorWindow) -> iced::Subscription<Message> {
    // Only tick for animations, state updates via SharedState from channel listener
    time::every(Duration::from_millis(16)).map(|_| Message::Tick)
}

fn update(window: &mut MonitorWindow, message: Message) -> Task<Message> {
    match message {
        Message::Tick => {
            let delta_time = 0.016; // ~60fps

            // Read from shared state and update local cache
            if let Ok(mut state) = window.shared_state.write() {
                state.tick(delta_time);
                window.cached_animation_time = state.animation_time;
                window.cached_closing_time = state.closing_animation_time;

                let new_state = state.gui_state;
                let new_transcription = state.transcription.clone();
                let new_spectrum = state.spectrum_values.clone();

                // Detect state change
                if new_state != window.cached_state {
                    debug!("[{}] State change: {:?} -> {:?}", window.monitor_name, window.cached_state, new_state);
                    window.previous_state = Some(window.cached_state);
                    window.cached_state = new_state;
                    window.transition_phase = TransitionPhase::Transitioning;
                    window.transition_progress = 0.0;

                    window.target_size = match new_state {
                        GuiState::Hidden => (0.0, 0.0),
                        GuiState::PreListening => calculate_prelistening_size(&window.config),
                        GuiState::Listening => calculate_listening_size(&new_transcription, &window.config),
                        GuiState::Processing => {
                            let cfg = &window.config.elements;
                            let padding = cfg.background_padding as f32;
                            let spinner_size = (cfg.spinner_orbit_radius * 2.0 + cfg.spinner_dot_radius * 2.0) * 1.5;
                            let size = spinner_size + padding * 2.0;
                            (size, size)
                        },
                        GuiState::Closing => (0.0, 0.0),
                    };
                }

                // Update transcription
                if new_transcription != window.cached_transcription {
                    window.cached_transcription = new_transcription.clone();

                    // Recalculate size if listening
                    if window.cached_state == GuiState::Listening {
                        let new_size = calculate_listening_size(&window.cached_transcription, &window.config);
                        if new_size != window.target_size {
                            window.target_size = new_size;
                            window.transition_phase = TransitionPhase::Transitioning;
                            window.transition_progress = 0.0;
                        }
                    }
                }

                // Update spectrum
                window.cached_spectrum = new_spectrum;
            }

            // Handle transitions
            if window.transition_phase == TransitionPhase::Transitioning {
                let transition_duration = get_transition_duration(window);
                window.transition_progress += delta_time / transition_duration;

                if window.transition_progress >= 1.0 {
                    window.transition_progress = 1.0;
                    window.current_size = window.target_size;
                    window.transition_phase = TransitionPhase::Idle;
                    let completed_state = window.cached_state;
                    window.previous_state = None;
                    debug!("[{}] Transition complete to {:?}", window.monitor_name, completed_state);

                    if completed_state == GuiState::PreListening {
                        if let Ok(mut state) = window.shared_state.write() {
                            state.set_gui_state(GuiState::Listening);
                        }
                    }
                } else {
                    window.current_size = interpolate_size(
                        window.current_size,
                        window.target_size,
                        window.transition_progress,
                    );
                }
            }

            // Check if closing animation complete
            if window.cached_state == GuiState::Closing {
                let closing_duration = window.config.animations.closing_background_duration as f32 / 1000.0;
                if window.cached_closing_time >= closing_duration {
                    debug!("[{}] Closing animation complete, transitioning to Hidden", window.monitor_name);
                    window.cached_state = GuiState::Hidden;
                }
            }

            Task::none()
        }
        _ => Task::none(), // Handle layer-shell messages
    }
}

fn view(window: &MonitorWindow) -> Element<'_, Message> {
    // Visibility check: only render if this monitor is active
    let is_active = if let Ok(state) = window.shared_state.read() {
        state.active_monitor == window.monitor_name
    } else {
        false
    };

    // Daemon state check: only show in certain states (Hidden is always invisible)
    let should_show = matches!(
        window.cached_state,
        GuiState::Listening | GuiState::Processing | GuiState::Closing
    );

    // Calculate final alpha (the "final filter")
    let visibility_alpha = if is_active && should_show {
        1.0
    } else {
        0.0
    };

    // Render based on state
    let content = match (window.transition_phase, window.cached_state, window.previous_state) {
        (_, GuiState::Hidden, _) => view_hidden(window),
        (TransitionPhase::Transitioning, GuiState::Listening, Some(GuiState::PreListening)) => {
            view_listening(window, visibility_alpha)
        }
        (TransitionPhase::Transitioning, GuiState::Processing, Some(GuiState::Listening)) => {
            view_transition_listening_to_processing(window, visibility_alpha)
        }
        (_, GuiState::PreListening, _) => view_prelistening(window, visibility_alpha),
        (_, GuiState::Listening, _) => view_listening(window, visibility_alpha),
        (_, GuiState::Processing, _) => view_processing(window, visibility_alpha),
        (_, GuiState::Closing, _) => view_closing(window, visibility_alpha),
    };

    content
}

fn view_hidden(_window: &MonitorWindow) -> Element<'_, Message> {
    // Completely invisible/empty element when hidden
    container(text(""))
        .width(Length::Fixed(0.0))
        .height(Length::Fixed(0.0))
        .into()
}

fn view_prelistening(window: &MonitorWindow, visibility_alpha: f32) -> Element<'_, Message> {
    let (width, height) = window.current_size;
    let cfg = &window.config.elements;
    let alpha = window.transition_progress * visibility_alpha;

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

fn view_listening(window: &MonitorWindow, visibility_alpha: f32) -> Element<'_, Message> {
    view_listening_with_alpha(window, visibility_alpha)
}

fn view_transition_listening_to_processing(window: &MonitorWindow, visibility_alpha: f32) -> Element<'_, Message> {
    let progress = window.transition_progress;

    if progress < 0.5 {
        let listening_alpha = (1.0 - (progress * 2.0)) * visibility_alpha;
        view_listening_with_alpha(window, listening_alpha)
    } else {
        let processing_alpha = ((progress - 0.5) * 2.0) * visibility_alpha;
        view_processing_with_alpha(window, processing_alpha)
    }
}

fn view_listening_with_alpha(window: &MonitorWindow, alpha: f32) -> Element<'_, Message> {
    let (width, height) = window.current_size;
    let cfg = &window.config.elements;

    let band_values = if window.cached_spectrum.is_empty() {
        vec![0.0; 8]
    } else {
        window.cached_spectrum.clone()
    };

    let spectrum = SpectrumBars::new(
        band_values,
        cfg.spectrum_min_bar_height,
        cfg.spectrum_max_bar_height,
        cfg.spectrum_bar_width_factor,
        cfg.spectrum_bar_spacing,
        cfg.spectrum_bar_radius,
        cfg.spectrum_opacity * alpha,
    )
    .height(SPECTRUM_HEIGHT)
    .width(SPECTRUM_WIDTH);

    let spectrum_container = container(spectrum).width(Length::Fill).center_x(Length::Fill);

    let mut content_items = vec![spectrum_container.into()];

    if !window.cached_transcription.is_empty() && cfg.text_enabled {
        let text_color = Color::from_rgba(1.0, 1.0, 1.0, cfg.text_opacity * alpha);
        let text_widget = text(&window.cached_transcription).size(cfg.text_font_size as f32).color(text_color);

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

fn view_processing(window: &MonitorWindow, visibility_alpha: f32) -> Element<'_, Message> {
    view_processing_with_alpha(window, visibility_alpha)
}

fn view_processing_with_alpha(window: &MonitorWindow, alpha: f32) -> Element<'_, Message> {
    let (width, height) = window.current_size;
    let cfg = &window.config.elements;

    let spinner_size = (cfg.spinner_orbit_radius * 2.0 + cfg.spinner_dot_radius * 2.0) * 1.5;

    let spinner = canvas(Spinner::new(
        window.cached_animation_time,
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

fn view_closing(window: &MonitorWindow, visibility_alpha: f32) -> Element<'_, Message> {
    let cfg = &window.config.elements;
    let closing_duration = window.config.animations.closing_background_duration as f32 / 1000.0;
    let progress = (window.cached_closing_time / closing_duration).min(1.0);
    let alpha = cfg.background_opacity * (1.0 - progress) * visibility_alpha;

    let collapse = CollapsingDots::new(progress, window.cached_animation_time);

    let spinner_size = (cfg.spinner_orbit_radius * 2.0 + cfg.spinner_dot_radius * 2.0) * 1.5;
    let collapse_canvas = canvas(collapse).width(Length::Fixed(spinner_size)).height(Length::Fixed(spinner_size));

    let (width, height) = window.current_size;
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

fn style(_window: &MonitorWindow, theme: &iced::Theme) -> iced_layershell::Appearance {
    iced_layershell::Appearance {
        background_color: Color::TRANSPARENT,
        text_color: theme.palette().text,
    }
}

// Helper functions

fn ease_in_out_cubic(t: f32) -> f32 {
    if t < 0.5 {
        4.0 * t.powi(3)
    } else {
        1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
    }
}

fn get_transition_duration(window: &MonitorWindow) -> f32 {
    let anims = &window.config.animations;
    match (window.previous_state, window.cached_state) {
        (Some(GuiState::PreListening), GuiState::Listening) => anims.transition_to_listening_duration as f32 / 1000.0,
        (Some(GuiState::Listening), GuiState::Processing) => {
            anims.listening_content_out_fade_duration.max(anims.processing_content_in_fade_duration) as f32 / 1000.0
        },
        (Some(GuiState::Processing), GuiState::Closing) | (_, GuiState::Closing) => anims.closing_background_duration as f32 / 1000.0,
        _ => 0.5,
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

/// Run a monitor window for a specific monitor
pub fn run_monitor_window(
    monitor_name: String,
    shared_state: Arc<RwLock<SharedState>>,
) -> Result<(), iced_layershell::Error> {
    let config = config::load_config();

    let anchor = match config.gui_general.position.as_str() {
        "top" => Anchor::Top | Anchor::Left | Anchor::Right,
        "center" => Anchor::Left | Anchor::Right,
        "bottom" => Anchor::Bottom | Anchor::Left | Anchor::Right,
        _ => Anchor::Bottom | Anchor::Left | Anchor::Right,
    };

    let margin = match config.gui_general.position.as_str() {
        "top" => (10, 0, 0, 0),
        "center" => (0, 0, 0, 0),
        "bottom" => (0, 0, 10, 0),
        _ => (0, 0, 10, 0),
    };

    let monitor_name_clone = monitor_name.clone();

    application(namespace, update, view)
        .layer_settings(LayerShellSettings {
            size: Some((config.gui_general.window_width, 160)),
            anchor,
            layer: Layer::Overlay,
            keyboard_interactivity: KeyboardInteractivity::None,
            margin,
            start_mode: StartMode::TargetScreen(monitor_name_clone),
            ..Default::default()
        })
        .subscription(subscription)
        .style(style)
        .run_with(move || {
            debug!("Initializing window for monitor: {}", monitor_name);
            (MonitorWindow::new(monitor_name, shared_state), Task::none())
        })
}
