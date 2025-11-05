use super::TextProcessor;
use anyhow::Result;

/// Simple rule-based punctuation and capitalization processor.
///
/// Applies the following transformations:
/// - Capitalizes the first word
/// - Capitalizes the pronoun "I" (including in contractions)
/// - Capitalizes words following sentence endings (. ? !)
///
/// This processor is designed to be fast (<5ms) and requires
/// no external dependencies or model files.
pub struct PunctuationProcessor;

impl PunctuationProcessor {
    pub fn new() -> Self {
        Self
    }
}

impl TextProcessor for PunctuationProcessor {
    fn process(&self, text: &str) -> Result<String> {
        if text.is_empty() {
            return Ok(String::new());
        }

        let mut result = String::with_capacity(text.len());
        let mut capitalize_next = true;

        for word in text.split_whitespace() {
            let processed = if capitalize_next {
                capitalize_first(word)
            } else {
                capitalize_pronoun_i(word)
            };

            result.push_str(&processed);
            result.push(' ');

            // Check if this word ends with a sentence terminator
            capitalize_next = ends_with_sentence_terminator(&processed);
        }

        // Remove trailing space
        Ok(result.trim_end().to_string())
    }
}

/// Capitalize the first character of a word.
fn capitalize_first(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => {
            let mut result = String::with_capacity(word.len());
            result.push(first.to_uppercase().next().unwrap());
            result.push_str(chars.as_str());
            result
        }
    }
}

/// Capitalize pronoun "I" if it appears standalone or in contractions.
///
/// Examples:
/// - "i" → "I"
/// - "i'm" → "I'm"
/// - "i'll" → "I'll"
/// - "i've" → "I've"
fn capitalize_pronoun_i(word: &str) -> String {
    if word.len() == 1 && word == "i" {
        return "I".to_string();
    }

    // Handle contractions like "i'm", "i'll", "i've"
    if word.len() > 1 && word.starts_with('i') && !word.chars().nth(1).unwrap().is_alphanumeric() {
        let mut result = String::with_capacity(word.len());
        result.push('I');
        result.push_str(&word[1..]);
        return result;
    }

    word.to_string()
}

/// Check if a word ends with a sentence terminator.
fn ends_with_sentence_terminator(word: &str) -> bool {
    word.ends_with('.') || word.ends_with('?') || word.ends_with('!')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_string() {
        let processor = PunctuationProcessor::new();
        let result = processor.process("").unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_capitalize_first_word() {
        let processor = PunctuationProcessor::new();
        let result = processor.process("hello world").unwrap();
        assert_eq!(result, "Hello world");
    }

    #[test]
    fn test_capitalize_pronoun_i() {
        let processor = PunctuationProcessor::new();
        let result = processor.process("i think i should go").unwrap();
        assert_eq!(result, "I think I should go");
    }

    #[test]
    fn test_capitalize_pronoun_i_contractions() {
        let processor = PunctuationProcessor::new();

        let result = processor.process("i'm happy").unwrap();
        assert_eq!(result, "I'm happy");

        let result = processor.process("i'll be there").unwrap();
        assert_eq!(result, "I'll be there");

        let result = processor.process("i've seen it").unwrap();
        assert_eq!(result, "I've seen it");
    }

    #[test]
    fn test_capitalize_after_period() {
        let processor = PunctuationProcessor::new();
        let result = processor.process("hello world. this is a test").unwrap();
        assert_eq!(result, "Hello world. This is a test");
    }

    #[test]
    fn test_capitalize_after_question() {
        let processor = PunctuationProcessor::new();
        let result = processor.process("are you sure? yes i am").unwrap();
        assert_eq!(result, "Are you sure? Yes I am");
    }

    #[test]
    fn test_capitalize_after_exclamation() {
        let processor = PunctuationProcessor::new();
        let result = processor.process("wow! that's amazing").unwrap();
        assert_eq!(result, "Wow! That's amazing");
    }

    #[test]
    fn test_multiple_sentences() {
        let processor = PunctuationProcessor::new();
        let result = processor
            .process("hello there. how are you? i am fine!")
            .unwrap();
        assert_eq!(result, "Hello there. How are you? I am fine!");
    }

    #[test]
    fn test_preserve_existing_punctuation() {
        let processor = PunctuationProcessor::new();
        let result = processor.process("hello, world! it's nice.").unwrap();
        assert_eq!(result, "Hello, world! It's nice.");
    }

    #[test]
    fn test_single_word() {
        let processor = PunctuationProcessor::new();
        let result = processor.process("hello").unwrap();
        assert_eq!(result, "Hello");
    }

    #[test]
    fn test_word_with_period() {
        let processor = PunctuationProcessor::new();
        let result = processor.process("hello.").unwrap();
        assert_eq!(result, "Hello.");
    }
}
