//! Text sanitization processor for context-aware character handling
//!
//! Sanitizes transcribed text based on the target application type.
//! Uses smart hybrid approach: escape common shell chars, remove dangerous ones.

use super::TextProcessor;
use crate::window_detect::AppCategory;
use anyhow::Result;
use tracing::debug;

/// Rules for text sanitization based on app category
#[derive(Debug, Clone)]
pub struct SanitizationRules {
    /// Escape shell special characters ($, `, \, !)
    pub escape_shell_chars: bool,
    /// Strip control characters (0x00-0x1F except whitespace)
    pub strip_control_chars: bool,
    /// Strip ANSI escape sequences
    pub strip_ansi_escapes: bool,
}

impl SanitizationRules {
    /// Create rules for a specific app category
    pub fn for_category(category: AppCategory) -> Self {
        match category {
            AppCategory::Terminal => Self {
                escape_shell_chars: true,
                strip_control_chars: true,
                strip_ansi_escapes: true,
            },
            AppCategory::Editor => Self {
                escape_shell_chars: false,
                strip_control_chars: true,
                strip_ansi_escapes: true,
            },
            AppCategory::Browser | AppCategory::Chat | AppCategory::General => Self {
                escape_shell_chars: false,
                strip_control_chars: true,
                strip_ansi_escapes: true,
            },
        }
    }
}

/// Processor that sanitizes text for safe input into various applications
pub struct SanitizationProcessor {
    rules: SanitizationRules,
    category: AppCategory,
}

impl SanitizationProcessor {
    /// Create a new sanitization processor with the given rules
    pub fn new(rules: SanitizationRules, category: AppCategory) -> Self {
        Self { rules, category }
    }

    /// Create a processor for a specific app category
    pub fn for_category(category: AppCategory) -> Self {
        Self {
            rules: SanitizationRules::for_category(category),
            category,
        }
    }
}

impl TextProcessor for SanitizationProcessor {
    fn process(&self, text: &str) -> Result<String> {
        let mut result = text.to_string();
        let original_len = result.len();

        // Strip ANSI escape sequences first
        if self.rules.strip_ansi_escapes {
            result = strip_ansi_escapes(&result);
        }

        // Strip control characters
        if self.rules.strip_control_chars {
            result = strip_control_chars(&result);
        }

        // Escape shell special characters (must be last to not interfere)
        if self.rules.escape_shell_chars {
            result = escape_shell_chars(&result);
        }

        if result.len() != original_len {
            debug!(
                "Sanitized text for {:?}: {} -> {} chars",
                self.category,
                original_len,
                result.len()
            );
        }

        Ok(result)
    }
}

/// Strip ANSI escape sequences (CSI sequences like \x1b[...m)
fn strip_ansi_escapes(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            // Check for CSI sequence: ESC [
            if chars.peek() == Some(&'[') {
                chars.next(); // consume '['
                // Skip until we hit a letter (the terminator)
                while let Some(&c) = chars.peek() {
                    chars.next();
                    if c.is_ascii_alphabetic() {
                        break;
                    }
                }
                continue;
            }
            // Check for OSC sequence: ESC ]
            if chars.peek() == Some(&']') {
                chars.next(); // consume ']'
                // Skip until BEL (\x07) or ST (\x1b\)
                while let Some(c) = chars.next() {
                    if c == '\x07' {
                        break;
                    }
                    if c == '\x1b' && chars.peek() == Some(&'\\') {
                        chars.next();
                        break;
                    }
                }
                continue;
            }
            // Skip lone ESC
            continue;
        }
        result.push(ch);
    }

    result
}

/// Strip control characters and problematic Unicode that can break terminals/React
fn strip_control_chars(text: &str) -> String {
    text.chars()
        .filter(|&ch| {
            // Keep standard whitespace
            if ch == '\n' || ch == '\t' || ch == '\r' || ch == ' ' {
                return true;
            }
            // Remove control characters (0x00-0x1F and 0x7F DEL)
            if ch.is_control() {
                return false;
            }
            // Remove zero-width characters (break React text nodes)
            if matches!(ch, '\u{200B}'..='\u{200D}' | '\u{FEFF}' | '\u{00AD}') {
                return false;
            }
            // Remove bidirectional formatting (cause rendering issues)
            if matches!(ch, '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}' | '\u{061C}') {
                return false;
            }
            // Remove variation selectors
            if matches!(ch, '\u{FE00}'..='\u{FE0F}') {
                return false;
            }
            // Remove other format characters
            if matches!(ch, '\u{180E}' | '\u{200E}' | '\u{200F}') {
                return false;
            }
            true
        })
        .collect()
}

/// Escape shell special characters for safe terminal input
fn escape_shell_chars(text: &str) -> String {
    let mut result = String::with_capacity(text.len() * 2);

    for ch in text.chars() {
        match ch {
            // Variable expansion
            '$' => result.push_str("\\$"),
            // Command substitution (backtick)
            '`' => result.push_str("\\`"),
            // Escape character itself
            '\\' => result.push_str("\\\\"),
            // History expansion (bash)
            '!' => result.push_str("\\!"),
            // Everything else passes through
            _ => result.push(ch),
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_control_chars() {
        // Basic control chars
        assert_eq!(strip_control_chars("hello\x00world"), "helloworld");
        assert_eq!(strip_control_chars("hello\x07bell"), "hellobell");
        assert_eq!(strip_control_chars("hello\nworld"), "hello\nworld");
        assert_eq!(strip_control_chars("hello\tworld"), "hello\tworld");

        // Zero-width characters (break React)
        assert_eq!(strip_control_chars("hello\u{200B}world"), "helloworld"); // zero-width space
        assert_eq!(strip_control_chars("hello\u{FEFF}world"), "helloworld"); // BOM
        assert_eq!(strip_control_chars("hello\u{00AD}world"), "helloworld"); // soft hyphen

        // Bidirectional marks
        assert_eq!(strip_control_chars("hello\u{202A}world"), "helloworld"); // LRE
        assert_eq!(strip_control_chars("hello\u{202E}world"), "helloworld"); // RLO
        assert_eq!(strip_control_chars("hello\u{2066}world"), "helloworld"); // LRI
    }

    #[test]
    fn test_strip_ansi_escapes() {
        // CSI sequence (colors)
        assert_eq!(strip_ansi_escapes("\x1b[31mred\x1b[0m"), "red");
        // Multiple sequences
        assert_eq!(strip_ansi_escapes("\x1b[1;32mbold green\x1b[0m"), "bold green");
        // OSC sequence (title)
        assert_eq!(strip_ansi_escapes("\x1b]0;title\x07text"), "text");
    }

    #[test]
    fn test_escape_shell_chars() {
        assert_eq!(escape_shell_chars("echo $HOME"), "echo \\$HOME");
        assert_eq!(escape_shell_chars("echo `date`"), "echo \\`date\\`");
        assert_eq!(escape_shell_chars("path\\to\\file"), "path\\\\to\\\\file");
        assert_eq!(escape_shell_chars("wow!"), "wow\\!");
    }

    #[test]
    fn test_terminal_sanitization() {
        let processor = SanitizationProcessor::for_category(AppCategory::Terminal);

        // Shell chars should be escaped
        let result = processor.process("echo $HOME").unwrap();
        assert_eq!(result, "echo \\$HOME");

        // Control chars should be stripped
        let result = processor.process("hello\x00world").unwrap();
        assert_eq!(result, "helloworld");

        // ANSI should be stripped
        let result = processor.process("\x1b[31mred\x1b[0m text").unwrap();
        assert_eq!(result, "red text");
    }

    #[test]
    fn test_general_sanitization() {
        let processor = SanitizationProcessor::for_category(AppCategory::General);

        // Shell chars should NOT be escaped
        let result = processor.process("echo $HOME").unwrap();
        assert_eq!(result, "echo $HOME");

        // But control chars should still be stripped
        let result = processor.process("hello\x00world").unwrap();
        assert_eq!(result, "helloworld");
    }

    // === EDGE CASE TESTS ===

    #[test]
    fn test_empty_input() {
        let processor = SanitizationProcessor::for_category(AppCategory::Terminal);
        let result = processor.process("").unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_whitespace_only() {
        let processor = SanitizationProcessor::for_category(AppCategory::Terminal);

        // Spaces should pass through
        assert_eq!(processor.process("   ").unwrap(), "   ");

        // Tabs should pass through
        assert_eq!(processor.process("\t\t").unwrap(), "\t\t");

        // Newlines should pass through
        assert_eq!(processor.process("\n\n").unwrap(), "\n\n");
    }

    #[test]
    fn test_unicode_emoji_passthrough() {
        let processor = SanitizationProcessor::for_category(AppCategory::Terminal);

        // Normal emojis should pass through
        let result = processor.process("hello 👋 world").unwrap();
        assert!(result.contains("👋"));

        // Complex emoji sequences
        let result = processor.process("test 🎉🎊🎁 done").unwrap();
        assert!(result.contains("🎉"));
        assert!(result.contains("🎊"));
    }

    #[test]
    fn test_variation_selector_stripped() {
        let processor = SanitizationProcessor::for_category(AppCategory::Terminal);

        // Variation selectors (U+FE00-U+FE0F) should be stripped
        let with_selector = "test\u{FE0F}text";
        let result = processor.process(with_selector).unwrap();
        assert_eq!(result, "testtext");
    }

    #[test]
    fn test_all_bidi_marks_stripped() {
        let processor = SanitizationProcessor::for_category(AppCategory::Terminal);

        // All bidi marks should be stripped
        let marks = [
            '\u{202A}', // LRE
            '\u{202B}', // RLE
            '\u{202C}', // PDF
            '\u{202D}', // LRO
            '\u{202E}', // RLO
            '\u{2066}', // LRI
            '\u{2067}', // RLI
            '\u{2068}', // FSI
            '\u{2069}', // PDI
            '\u{061C}', // ALM
        ];

        for mark in marks {
            let input = format!("hello{}world", mark);
            let result = processor.process(&input).unwrap();
            assert_eq!(result, "helloworld", "Failed to strip {:?}", mark);
        }
    }

    #[test]
    fn test_complex_ansi_sequences() {
        let processor = SanitizationProcessor::for_category(AppCategory::Terminal);

        // Multiple SGR parameters
        let result = processor.process("\x1b[38;5;196mred\x1b[0m").unwrap();
        assert_eq!(result, "red");

        // 24-bit color
        let result = processor.process("\x1b[38;2;255;0;0mred\x1b[0m").unwrap();
        assert_eq!(result, "red");

        // Nested sequences
        let result = processor.process("\x1b[1m\x1b[31mbold red\x1b[0m\x1b[0m").unwrap();
        assert_eq!(result, "bold red");
    }

    #[test]
    fn test_osc_sequence_with_st() {
        // OSC terminated by ST (ESC \) instead of BEL
        let result = strip_ansi_escapes("\x1b]0;title\x1b\\text");
        assert_eq!(result, "text");
    }

    #[test]
    fn test_mixed_problematic_chars() {
        let processor = SanitizationProcessor::for_category(AppCategory::Terminal);

        // Combine multiple issues
        let input = "\x1b[31m$HOME\u{200B}\x00test\u{202E}!\x1b[0m";
        let result = processor.process(input).unwrap();

        // Should strip ANSI, control chars, zero-width, bidi
        // Should escape $ and !
        assert!(result.contains("\\$HOME"));
        assert!(result.contains("test"));
        assert!(result.contains("\\!"));
        assert!(!result.contains("\x1b"));
        assert!(!result.contains("\x00"));
    }

    #[test]
    fn test_all_shell_chars_escaped() {
        let processor = SanitizationProcessor::for_category(AppCategory::Terminal);

        let input = "echo $VAR `cmd` path\\file wow!";
        let result = processor.process(input).unwrap();

        assert_eq!(result, "echo \\$VAR \\`cmd\\` path\\\\file wow\\!");
    }

    #[test]
    fn test_long_text_sanitization() {
        let processor = SanitizationProcessor::for_category(AppCategory::Terminal);

        // Generate a long text with various issues
        let mut input = String::new();
        for i in 0..1000 {
            if i % 100 == 0 {
                input.push_str("$HOME ");
            } else if i % 50 == 0 {
                input.push('\u{200B}'); // zero-width
            } else {
                input.push_str("word ");
            }
        }

        let result = processor.process(&input).unwrap();

        // Should have escaped the shell vars
        assert!(result.contains("\\$HOME"));
        // Should have stripped zero-width chars
        assert!(!result.contains('\u{200B}'));
    }

    #[test]
    fn test_category_rules() {
        // Terminal escapes shell chars
        let terminal = SanitizationRules::for_category(AppCategory::Terminal);
        assert!(terminal.escape_shell_chars);
        assert!(terminal.strip_control_chars);
        assert!(terminal.strip_ansi_escapes);

        // Editor does NOT escape shell chars
        let editor = SanitizationRules::for_category(AppCategory::Editor);
        assert!(!editor.escape_shell_chars);
        assert!(editor.strip_control_chars);

        // Browser does NOT escape shell chars
        let browser = SanitizationRules::for_category(AppCategory::Browser);
        assert!(!browser.escape_shell_chars);

        // Chat does NOT escape shell chars
        let chat = SanitizationRules::for_category(AppCategory::Chat);
        assert!(!chat.escape_shell_chars);

        // General does NOT escape shell chars
        let general = SanitizationRules::for_category(AppCategory::General);
        assert!(!general.escape_shell_chars);
    }

    #[test]
    fn test_format_chars_stripped() {
        let processor = SanitizationProcessor::for_category(AppCategory::Terminal);

        // Mongolian vowel separator (U+180E)
        let result = processor.process("hello\u{180E}world").unwrap();
        assert_eq!(result, "helloworld");

        // LRM (U+200E) and RLM (U+200F)
        let result = processor.process("hello\u{200E}world").unwrap();
        assert_eq!(result, "helloworld");

        let result = processor.process("hello\u{200F}world").unwrap();
        assert_eq!(result, "helloworld");
    }

    #[test]
    fn test_transcription_realistic_outputs() {
        let processor = SanitizationProcessor::for_category(AppCategory::Terminal);

        // Realistic transcription output that might cause issues
        let inputs = [
            "Hello, my name is John.",
            "I'd like to run echo $HOME in the terminal.",
            "The command is `ls -la`.",
            "Wow! That's amazing!",
            "Let me check the file path\\config\\settings.",
        ];

        let expected = [
            "Hello, my name is John.",
            "I'd like to run echo \\$HOME in the terminal.",
            "The command is \\`ls -la\\`.",
            "Wow\\! That's amazing\\!",
            "Let me check the file path\\\\config\\\\settings.",
        ];

        for (input, expect) in inputs.iter().zip(expected.iter()) {
            let result = processor.process(input).unwrap();
            assert_eq!(&result, *expect, "Failed for input: {}", input);
        }
    }
}
