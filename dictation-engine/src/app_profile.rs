//! Per-app behavioral settings derived from window class name

use crate::post_processing::SanitizationRules;
use crate::window_detect::AppCategory;

pub struct AppProfile {
    pub category: AppCategory,
    pub word_delay_ms: u64,
    pub sanitization: SanitizationRules,
}

impl AppProfile {
    pub fn for_category(category: AppCategory) -> Self {
        let (word_delay_ms, sanitization) = match category {
            AppCategory::Terminal => (50, SanitizationRules::for_category(category)),
            _ => (0, SanitizationRules::for_category(category)),
        };
        Self {
            category,
            word_delay_ms,
            sanitization,
        }
    }

    pub fn from_window_class(class: &str) -> Self {
        let category = match class {
            "kitty"
            | "Alacritty"
            | "foot"
            | "org.wezfurlong.wezterm"
            | "com.mitchellh.ghostty"
            | "ghostty"
            | "tmux" => AppCategory::Terminal,
            _ => AppCategory::General,
        };
        Self::for_category(category)
    }

    pub fn detect() -> Self {
        // Synchronous fallback using window_detect
        Self::for_category(AppCategory::General)
    }
}
