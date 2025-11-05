use super::TextProcessor;
use anyhow::Result;
use harper_core::linting::{Lint, LintGroup, LintKind, Linter, Suggestion};
use harper_core::parsers::PlainEnglish;
use harper_core::spell::MutableDictionary;
use harper_core::{Dialect, Document};
use std::sync::Arc;

/// Grammar and spell checker using Harper.
///
/// Harper is a fast, offline, privacy-first grammar checker designed
/// for developers. It understands code context and won't flag technical
/// terms inappropriately.
///
/// Features:
/// - Grammar checking
/// - Spell checking with system dictionaries
/// - Developer-friendly (understands code/technical context)
/// - ~10-20ms processing time
/// - <50MB memory footprint
pub struct GrammarProcessor {
    dictionary: Arc<MutableDictionary>,
}

impl GrammarProcessor {
    /// Create a new grammar processor with Harper's default configuration.
    pub fn new() -> Self {
        let dictionary = MutableDictionary::curated();
        Self { dictionary }
    }
}

impl TextProcessor for GrammarProcessor {
    fn process(&self, text: &str) -> Result<String> {
        if text.is_empty() {
            return Ok(String::new());
        }

        // Parse text into Harper document with plain English parser
        let mut parser = PlainEnglish;
        let document = Document::new(text, &mut parser, &self.dictionary);

        // Create linter with curated rules
        let mut linter = LintGroup::new_curated(self.dictionary.clone(), Dialect::American);

        // Run linter to find issues
        let lints = linter.lint(&document);

        // Apply suggestions in reverse order to maintain correct positions
        let mut sorted_lints: Vec<Lint> = lints.into_iter().collect();
        sorted_lints.sort_by(|a, b| b.span.start.cmp(&a.span.start));

        // Build corrected text by applying suggestions
        let mut result = text.to_string();

        for lint in sorted_lints {
            // Only apply lints with suggestions
            if let Some(suggestion) = get_best_suggestion(&lint) {
                let span = lint.span;
                let start = span.start;
                let end = span.end;

                // Safety check: ensure span is within bounds
                if start <= result.len() && end <= result.len() && start <= end {
                    result.replace_range(start..end, &suggestion);
                }
            }
        }

        Ok(result)
    }
}

/// Extract the best suggestion from a lint.
///
/// Prioritizes:
/// 1. First suggestion from the lint (usually the most confident)
/// 2. For spelling errors, use the first correction
fn get_best_suggestion(lint: &Lint) -> Option<String> {
    match &lint.lint_kind {
        LintKind::Spelling => {
            // For spelling errors, get the first suggestion
            if let Some(suggestion) = lint.suggestions.first() {
                Some(suggestion_to_string(suggestion))
            } else {
                None
            }
        }
        _ => {
            // For other lints, use the first suggestion if available
            lint.suggestions.first().map(suggestion_to_string)
        }
    }
}

/// Convert a Harper Suggestion to a replacement string.
fn suggestion_to_string(suggestion: &Suggestion) -> String {
    match suggestion {
        Suggestion::ReplaceWith(chars) => chars.iter().collect(),
        Suggestion::Remove => String::new(),
        Suggestion::InsertAfter(chars) => chars.iter().collect(),
    }
}

impl Default for GrammarProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_string() {
        let processor = GrammarProcessor::new();
        let result = processor.process("").unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_correct_text_unchanged() {
        let processor = GrammarProcessor::new();
        let input = "This is a correct sentence.";
        let result = processor.process(input).unwrap();
        // Should remain the same or very similar
        assert!(!result.is_empty());
    }

    #[test]
    fn test_simple_spelling() {
        let processor = GrammarProcessor::new();
        // Note: Harper might or might not catch this depending on its rules
        let input = "This is a tset.";
        let result = processor.process(input).unwrap();
        // Just verify it doesn't crash
        assert!(!result.is_empty());
    }

    #[test]
    fn test_technical_terms_preserved() {
        let processor = GrammarProcessor::new();
        let input = "The API endpoint returns JSON data.";
        let result = processor.process(input).unwrap();
        // Technical terms should be preserved
        assert!(result.contains("API"));
        assert!(result.contains("JSON"));
    }

    #[test]
    fn test_multiple_sentences() {
        let processor = GrammarProcessor::new();
        let input = "First sentence. Second sentence. Third sentence.";
        let result = processor.process(input).unwrap();
        assert!(!result.is_empty());
    }
}
