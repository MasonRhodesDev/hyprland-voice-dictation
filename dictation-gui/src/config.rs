use serde::Deserialize;
use std::fs;
use tracing::warn;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub gui_general: GuiGeneralConfig,
    #[serde(default)]
    pub animations: AnimationsConfig,
    #[serde(default)]
    pub elements: ElementsConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GuiGeneralConfig {
    #[serde(default = "default_window_width")]
    pub window_width: u32,
    #[serde(default = "default_window_height")]
    pub window_height: u32,
    #[serde(default = "default_position")]
    pub position: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnimationsConfig {
    #[serde(default = "default_true")]
    pub enable_animations: bool,
    #[serde(default = "default_animation_speed")]
    pub animation_speed: f32,
    
    #[serde(default = "default_startup_fade_duration")]
    pub startup_fade_duration: u32,
    #[serde(default = "default_startup_fade_easing")]
    pub startup_fade_easing: String,
    
    #[serde(default = "default_transition_to_listening_duration")]
    pub transition_to_listening_duration: u32,
    #[serde(default = "default_transition_to_listening_easing")]
    pub transition_to_listening_easing: String,
    
    #[serde(default = "default_listening_content_out_fade_duration")]
    pub listening_content_out_fade_duration: u32,
    #[serde(default = "default_listening_content_out_fade_easing")]
    pub listening_content_out_fade_easing: String,
    
    #[serde(default = "default_processing_content_in_fade_duration")]
    pub processing_content_in_fade_duration: u32,
    #[serde(default = "default_processing_content_in_fade_easing")]
    pub processing_content_in_fade_easing: String,
    
    #[serde(default = "default_closing_background_duration")]
    pub closing_background_duration: u32,
    #[serde(default = "default_closing_background_easing")]
    pub closing_background_easing: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ElementsConfig {
    #[serde(default = "default_true")]
    pub spectrum_enabled: bool,
    #[serde(default = "default_spectrum_min_bar_height")]
    pub spectrum_min_bar_height: f32,
    #[serde(default = "default_spectrum_max_bar_height")]
    pub spectrum_max_bar_height: f32,
    #[serde(default = "default_spectrum_bar_width_factor")]
    pub spectrum_bar_width_factor: f32,
    #[serde(default = "default_spectrum_bar_spacing")]
    pub spectrum_bar_spacing: f32,
    #[serde(default = "default_spectrum_bar_radius")]
    pub spectrum_bar_radius: f32,
    #[serde(default = "default_opacity_one")]
    pub spectrum_opacity: f32,
    #[serde(default = "default_spectrum_smoothing_factor")]
    pub spectrum_smoothing_factor: f32,
    #[serde(default = "default_spectrum_update_rate")]
    pub spectrum_update_rate: u32,
    
    #[serde(default = "default_true")]
    pub spinner_enabled: bool,
    #[serde(default = "default_spinner_dot_count")]
    pub spinner_dot_count: u32,
    #[serde(default = "default_spinner_dot_radius")]
    pub spinner_dot_radius: f32,
    #[serde(default = "default_spinner_orbit_radius")]
    pub spinner_orbit_radius: f32,
    #[serde(default = "default_spinner_rotation_speed")]
    pub spinner_rotation_speed: f32,
    #[serde(default = "default_opacity_one")]
    pub spinner_opacity: f32,
    
    #[serde(default = "default_true")]
    pub text_enabled: bool,
    #[serde(default = "default_text_font_size")]
    pub text_font_size: u32,
    #[serde(default = "default_opacity_one")]
    pub text_opacity: f32,
    #[serde(default = "default_text_alignment")]
    pub text_alignment: String,
    #[serde(default = "default_text_line_height")]
    pub text_line_height: f32,
    #[serde(default = "default_text_appear_duration")]
    pub text_appear_duration: u32,
    #[serde(default = "default_text_scroll_speed")]
    pub text_scroll_speed: f32,
    
    #[serde(default = "default_background_corner_radius")]
    pub background_corner_radius: f32,
    #[serde(default = "default_background_corner_radius_processing")]
    pub background_corner_radius_processing: f32,
    #[serde(default = "default_background_opacity")]
    pub background_opacity: f32,
    #[serde(default = "default_background_padding")]
    pub background_padding: u32,
}

fn default_window_width() -> u32 { 400 }
fn default_window_height() -> u32 { 200 }
fn default_position() -> String { "bottom".to_string() }

fn default_true() -> bool { true }
fn default_opacity_one() -> f32 { 1.0 }
fn default_animation_speed() -> f32 { 1.0 }

fn default_startup_fade_duration() -> u32 { 300 }
fn default_startup_fade_easing() -> String { "ease-in-out-quad".to_string() }
fn default_transition_to_listening_duration() -> u32 { 500 }
fn default_transition_to_listening_easing() -> String { "ease-in-out-cubic".to_string() }
fn default_listening_content_out_fade_duration() -> u32 { 200 }
fn default_listening_content_out_fade_easing() -> String { "ease-out".to_string() }
fn default_processing_content_in_fade_duration() -> u32 { 200 }
fn default_processing_content_in_fade_easing() -> String { "ease-in".to_string() }
fn default_closing_background_duration() -> u32 { 500 }
fn default_closing_background_easing() -> String { "ease-in-cubic".to_string() }

fn default_spectrum_min_bar_height() -> f32 { 5.0 }
fn default_spectrum_max_bar_height() -> f32 { 30.0 }
fn default_spectrum_bar_width_factor() -> f32 { 0.6 }
fn default_spectrum_bar_spacing() -> f32 { 8.0 }
fn default_spectrum_bar_radius() -> f32 { 3.0 }
fn default_spectrum_smoothing_factor() -> f32 { 0.6 }
fn default_spectrum_update_rate() -> u32 { 60 }

fn default_spinner_dot_count() -> u32 { 3 }
fn default_spinner_dot_radius() -> f32 { 6.0 }
fn default_spinner_orbit_radius() -> f32 { 20.0 }
fn default_spinner_rotation_speed() -> f32 { 2.0 }

fn default_text_font_size() -> u32 { 24 }
fn default_text_alignment() -> String { "center".to_string() }
fn default_text_line_height() -> f32 { 1.2 }
fn default_text_appear_duration() -> u32 { 150 }
fn default_text_scroll_speed() -> f32 { 1.0 }

fn default_background_corner_radius() -> f32 { 25.0 }
fn default_background_corner_radius_processing() -> f32 { 50.0 }
fn default_background_opacity() -> f32 { 0.95 }
fn default_background_padding() -> u32 { 20 }

impl Default for GuiGeneralConfig {
    fn default() -> Self {
        Self {
            window_width: default_window_width(),
            window_height: default_window_height(),
            position: default_position(),
        }
    }
}

impl Default for AnimationsConfig {
    fn default() -> Self {
        Self {
            enable_animations: default_true(),
            animation_speed: default_animation_speed(),
            startup_fade_duration: default_startup_fade_duration(),
            startup_fade_easing: default_startup_fade_easing(),
            transition_to_listening_duration: default_transition_to_listening_duration(),
            transition_to_listening_easing: default_transition_to_listening_easing(),
            listening_content_out_fade_duration: default_listening_content_out_fade_duration(),
            listening_content_out_fade_easing: default_listening_content_out_fade_easing(),
            processing_content_in_fade_duration: default_processing_content_in_fade_duration(),
            processing_content_in_fade_easing: default_processing_content_in_fade_easing(),
            closing_background_duration: default_closing_background_duration(),
            closing_background_easing: default_closing_background_easing(),
        }
    }
}

impl Default for ElementsConfig {
    fn default() -> Self {
        Self {
            spectrum_enabled: default_true(),
            spectrum_min_bar_height: default_spectrum_min_bar_height(),
            spectrum_max_bar_height: default_spectrum_max_bar_height(),
            spectrum_bar_width_factor: default_spectrum_bar_width_factor(),
            spectrum_bar_spacing: default_spectrum_bar_spacing(),
            spectrum_bar_radius: default_spectrum_bar_radius(),
            spectrum_opacity: default_opacity_one(),
            spectrum_smoothing_factor: default_spectrum_smoothing_factor(),
            spectrum_update_rate: default_spectrum_update_rate(),
            
            spinner_enabled: default_true(),
            spinner_dot_count: default_spinner_dot_count(),
            spinner_dot_radius: default_spinner_dot_radius(),
            spinner_orbit_radius: default_spinner_orbit_radius(),
            spinner_rotation_speed: default_spinner_rotation_speed(),
            spinner_opacity: default_opacity_one(),
            
            text_enabled: default_true(),
            text_font_size: default_text_font_size(),
            text_opacity: default_opacity_one(),
            text_alignment: default_text_alignment(),
            text_line_height: default_text_line_height(),
            text_appear_duration: default_text_appear_duration(),
            text_scroll_speed: default_text_scroll_speed(),
            
            background_corner_radius: default_background_corner_radius(),
            background_corner_radius_processing: default_background_corner_radius_processing(),
            background_opacity: default_background_opacity(),
            background_padding: default_background_padding(),
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            gui_general: GuiGeneralConfig::default(),
            animations: AnimationsConfig::default(),
            elements: ElementsConfig::default(),
        }
    }
}

pub fn load_config() -> Config {
    let home = match std::env::var("HOME") {
        Ok(h) => h,
        Err(_) => {
            warn!("HOME env var not set, using default config");
            return Config::default();
        }
    };
    
    let config_path = format!("{}/.config/voice-dictation/config.toml", home);
    
    let config_str = match fs::read_to_string(&config_path) {
        Ok(s) => s,
        Err(_) => {
            warn!("Could not read config file at {}, using defaults", config_path);
            return Config::default();
        }
    };
    
    match toml::from_str::<Config>(&config_str) {
        Ok(config) => {
            tracing::info!("Loaded GUI config from {}", config_path);
            config
        }
        Err(e) => {
            warn!("Failed to parse config: {}, using defaults", e);
            Config::default()
        }
    }
}
