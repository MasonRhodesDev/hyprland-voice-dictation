//! Application category detection for context-aware text sanitization
//!
//! Currently provides manual configuration only.
//! Future: Plugin system for compositor-specific detection.

/// Categories of applications for sanitization rules
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AppCategory {
    /// Terminal emulators - need shell character escaping
    Terminal,
    /// Web browsers - generally safe
    Browser,
    /// Code editors - may need some escaping
    Editor,
    /// Chat/messaging apps - generally safe
    Chat,
    /// Everything else - minimal sanitization
    #[default]
    General,
}

impl AppCategory {
    /// Parse from config string
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "terminal" | "term" => AppCategory::Terminal,
            "browser" | "web" => AppCategory::Browser,
            "editor" | "code" => AppCategory::Editor,
            "chat" | "messaging" => AppCategory::Chat,
            _ => AppCategory::General,
        }
    }
}

/// Get the application category for sanitization
///
/// Currently returns Terminal mode by default (escapes shell chars).
/// Future: Plugin system for automatic detection.
pub async fn get_focused_app_category() -> AppCategory {
    // Default to Terminal mode - escapes $, `, \, !
    // Most dictation happens in terminals
    AppCategory::Terminal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_str() {
        assert_eq!(AppCategory::from_str("terminal"), AppCategory::Terminal);
        assert_eq!(AppCategory::from_str("TERMINAL"), AppCategory::Terminal);
        assert_eq!(AppCategory::from_str("term"), AppCategory::Terminal);
        assert_eq!(AppCategory::from_str("browser"), AppCategory::Browser);
        assert_eq!(AppCategory::from_str("editor"), AppCategory::Editor);
        assert_eq!(AppCategory::from_str("chat"), AppCategory::Chat);
        assert_eq!(AppCategory::from_str("anything"), AppCategory::General);
    }
}
